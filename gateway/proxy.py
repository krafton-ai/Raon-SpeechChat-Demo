"""Bidirectional frame relay between WebSocket client and Ray worker actor."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from proto.close_reasons import (
    CLIENT_DISCONNECT,
    INTERNAL_ERROR,
    INVALID_CLIENT_FRAME,
    OVERLOADED_BACKLOG,
    WORKER_UNHEALTHY,
    normalize_close_reason,
)
from proto.messages import Frame, MessageKind

logger = logging.getLogger(__name__)

_AUDIO_QUEUE_MAX_FRAMES = 90
_AUDIO_BATCH_FRAMES = 3


class SessionProxy:
    """Manages one client↔worker WebSocket session.

    Uses the combined feed_and_decode actor method to minimize Ray IPC
    round-trips: each audio frame from the client triggers exactly ONE
    Ray call that feeds audio, runs decode, and returns output frames.
    """

    def __init__(
        self,
        session_id: str,
        ping_interval: float = 60.0,
        ping_timeout: float = 30.0,
    ) -> None:
        self.session_id = session_id
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self._closed = False
        self._last_pong: float = time.monotonic()
        self.close_reason: str = CLIENT_DISCONNECT
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=_AUDIO_QUEUE_MAX_FRAMES,
        )  # ~7.2s buffer at 80ms/frame
        self._gateway_queue_drops: int = 0
        self._consecutive_drops: int = 0
        self._rpc_latency_samples: deque[float] = deque(maxlen=256)

    def metrics(self) -> dict:
        """Return current queue metrics for this session."""
        samples = list(self._rpc_latency_samples)
        if samples:
            rpc_avg = sum(samples) / len(samples)
            sorted_samples = sorted(samples)
            rank = int(round(0.95 * (len(sorted_samples) - 1)))
            rpc_p95 = sorted_samples[max(0, min(rank, len(sorted_samples) - 1))]
        else:
            rpc_avg = 0.0
            rpc_p95 = 0.0
        return {
            "queue_depth": self._audio_queue.qsize(),
            "queue_drops": self._gateway_queue_drops,
            "consecutive_drops": self._consecutive_drops,
            "rpc_latency_ms_avg": rpc_avg,
            "rpc_latency_ms_p95": rpc_p95,
        }

    async def _send_error_and_close(
        self,
        websocket: WebSocket,
        *,
        message: str,
        reason: str,
    ) -> None:
        """Best-effort protocol error signaling to the client."""
        try:
            await websocket.send_bytes(Frame.error(message).encode())
        except Exception:
            pass
        try:
            await websocket.send_bytes(Frame.close(reason=reason).encode())
        except Exception:
            pass

    async def _recv_loop(self, websocket: WebSocket) -> None:
        """Read WebSocket frames from client and enqueue audio payloads."""
        try:
            while not self._closed:
                try:
                    data = await websocket.receive_bytes()
                except WebSocketDisconnect:
                    logger.info("client disconnected (session=%s)", self.session_id)
                    self.close_reason = CLIENT_DISCONNECT
                    break
                try:
                    frame = Frame.decode(data)
                except ValueError as exc:
                    logger.info("invalid frame from client (session=%s): %s", self.session_id, exc)
                    self.close_reason = INVALID_CLIENT_FRAME
                    await self._send_error_and_close(
                        websocket,
                        message=f"invalid frame: {exc}",
                        reason=INVALID_CLIENT_FRAME,
                    )
                    self._closed = True
                    break
                if frame.kind == MessageKind.PONG:
                    self._last_pong = time.monotonic()
                    continue
                if frame.kind == MessageKind.PING:
                    await websocket.send_bytes(Frame(kind=MessageKind.PONG, payload=b"").encode())
                    continue
                if frame.kind == MessageKind.CLOSE:
                    self.close_reason = CLIENT_DISCONNECT
                    self._closed = True
                    break
                if frame.kind == MessageKind.AUDIO:
                    if self._audio_queue.full():
                        # Drop oldest frame to make room
                        try:
                            self._audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        self._gateway_queue_drops += 1
                        self._consecutive_drops += 1
                        if self._consecutive_drops >= 30:
                            logger.warning(
                                "overloaded backlog: %d consecutive drops (session=%s)",
                                self._consecutive_drops,
                                self.session_id,
                            )
                            self.close_reason = OVERLOADED_BACKLOG
                            await self._send_error_and_close(
                                websocket,
                                message="session overloaded: audio backlog exceeded",
                                reason=OVERLOADED_BACKLOG,
                            )
                            self._closed = True
                            break
                    self._audio_queue.put_nowait(frame.payload)
                    continue

                logger.info(
                    "unsupported client frame kind=%s (session=%s)",
                    frame.kind,
                    self.session_id,
                )
                self.close_reason = INVALID_CLIENT_FRAME
                await self._send_error_and_close(
                    websocket,
                    message=f"unsupported frame kind: {int(frame.kind)}",
                    reason=INVALID_CLIENT_FRAME,
                )
                self._closed = True
                break
        except Exception as exc:
            logger.warning("_recv_loop error (session=%s): %s", self.session_id, exc)
            self.close_reason = INTERNAL_ERROR
        finally:
            self._closed = True

    def _collect_batch(self) -> bytes | None:
        """Drain a small audio batch for lower relay latency."""
        payloads: list[bytes] = []
        try:
            payloads.append(self._audio_queue.get_nowait())
        except asyncio.QueueEmpty:
            return None
        self._consecutive_drops = 0
        for _ in range(_AUDIO_BATCH_FRAMES - 1):
            try:
                payloads.append(self._audio_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return b"".join(payloads)

    async def _send_loop(
        self,
        websocket: WebSocket,
        worker_actor: "ray.actor.ActorHandle",
    ) -> None:
        """Pipeline RPCs: feed audio and await results concurrently.

        Split into a feeder (collects audio, fires RPCs) and a drainer
        (awaits results in order, sends to client). The feeder is never
        blocked waiting for inference — it keeps shipping audio to the
        worker while previous RPCs are still in flight.
        """
        # Bounded queue of in-flight RPC futures, processed in order.
        _MAX_INFLIGHT = 3
        rpc_queue: asyncio.Queue[tuple[float, object]] = asyncio.Queue(
            maxsize=_MAX_INFLIGHT,
        )

        async def _feeder() -> None:
            """Collect audio batches and fire RPCs without waiting for results."""
            try:
                while not self._closed:
                    try:
                        first_payload = await asyncio.wait_for(
                            self._audio_queue.get(), timeout=0.5,
                        )
                    except asyncio.TimeoutError:
                        continue
                    self._consecutive_drops = 0
                    payloads = [first_payload]
                    for _ in range(_AUDIO_BATCH_FRAMES - 1):
                        try:
                            payloads.append(self._audio_queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                    combined = b"".join(payloads)
                    rpc_ref = worker_actor.feed_and_decode.remote(
                        self.session_id, combined,
                    )
                    # Blocks if _MAX_INFLIGHT RPCs already in flight (backpressure)
                    await rpc_queue.put((time.monotonic(), rpc_ref))
            except Exception as exc:
                logger.warning("_feeder error (session=%s): %s", self.session_id, exc)
                self.close_reason = INTERNAL_ERROR
                self._closed = True

        async def _drainer() -> None:
            """Await RPC results in order and relay output frames to client."""
            try:
                while not self._closed:
                    try:
                        rpc_start, rpc_ref = await asyncio.wait_for(
                            rpc_queue.get(), timeout=0.5,
                        )
                    except asyncio.TimeoutError:
                        continue
                    output_frames: list[Frame] = await rpc_ref
                    rpc_elapsed_ms = (time.monotonic() - rpc_start) * 1000
                    self._rpc_latency_samples.append(rpc_elapsed_ms)
                    for out_frame in output_frames:
                        await websocket.send_bytes(out_frame.encode())
                        if out_frame.kind == MessageKind.CLOSE:
                            reason = out_frame.payload.decode(
                                "utf-8", errors="ignore",
                            ).strip()
                            self.close_reason = normalize_close_reason(reason)
                            self._closed = True
                            break
            except WebSocketDisconnect:
                self.close_reason = CLIENT_DISCONNECT
                self._closed = True
            except Exception as exc:
                logger.warning("_drainer error (session=%s): %s", self.session_id, exc)
                self.close_reason = INTERNAL_ERROR
                self._closed = True

        feed_task = asyncio.create_task(_feeder())
        drain_task = asyncio.create_task(_drainer())
        try:
            _done, pending = await asyncio.wait(
                [feed_task, drain_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            self._closed = True
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception as exc:
            logger.warning("_send_loop error (session=%s): %s", self.session_id, exc)
            self.close_reason = INTERNAL_ERROR
        finally:
            self._closed = True

    async def _keepalive(self, websocket: WebSocket) -> None:
        """Send PING at configured interval; close if PONG not received in time."""
        ping_frame = Frame(kind=MessageKind.PING, payload=b"")
        try:
            while not self._closed:
                await asyncio.sleep(self.ping_interval)
                if self._closed:
                    break
                ping_sent_at = time.monotonic()
                await websocket.send_bytes(ping_frame.encode())
                await asyncio.sleep(self.ping_timeout)
                if self._closed:
                    break
                if self._last_pong < ping_sent_at:
                    logger.warning("PING timeout (session=%s)", self.session_id)
                    self.close_reason = CLIENT_DISCONNECT
                    self._closed = True
                    break
        except WebSocketDisconnect:
            self.close_reason = CLIENT_DISCONNECT
            pass
        except Exception as exc:
            logger.debug("keepalive error (session=%s): %s", self.session_id, exc)
            self.close_reason = INTERNAL_ERROR
        finally:
            self._closed = True

    async def _worker_watchdog(self, worker_actor: "ray.actor.ActorHandle") -> None:
        """Periodically check worker liveness; close session if unreachable."""
        health_timeout = max(10.0, self.ping_timeout)
        while not self._closed:
            await asyncio.sleep(10.0)
            if self._closed:
                break
            try:
                await asyncio.wait_for(worker_actor.health.remote(), timeout=health_timeout)
            except Exception as exc:
                logger.warning("worker unreachable (session=%s): %s", self.session_id, exc)
                self.close_reason = WORKER_UNHEALTHY
                self._closed = True
                break

    async def run(
        self,
        websocket: WebSocket,
        worker_actor: "ray.actor.ActorHandle",
    ) -> None:
        """Run recv, send, keepalive, and worker watchdog concurrently."""
        self._closed = False
        self._last_pong = time.monotonic()
        self.close_reason = CLIENT_DISCONNECT
        self._audio_queue = asyncio.Queue(maxsize=_AUDIO_QUEUE_MAX_FRAMES)
        self._gateway_queue_drops = 0
        self._consecutive_drops = 0
        self._rpc_latency_samples = deque(maxlen=256)

        tasks = [
            asyncio.create_task(self._recv_loop(websocket)),
            asyncio.create_task(self._send_loop(websocket, worker_actor)),
            asyncio.create_task(self._keepalive(websocket)),
            asyncio.create_task(self._worker_watchdog(worker_actor)),
        ]
        try:
            _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            self._closed = True
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception as exc:
            logger.error("proxy run error (session=%s): %s", self.session_id, exc)
            self._closed = True
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
