"""Ray actor wrapper around RaonEngine.

Provides remote-callable methods for session management and inference.
Each actor claims one GPU via @ray.remote(num_gpus=1).

Architecture: The actor runs an internal decode loop that processes ALL
active sessions in round-robin. External callers only feed audio (buffer)
and drain output queues — they never trigger inference directly.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import TYPE_CHECKING

from proto.close_reasons import CLIENT_DISCONNECT, INTERNAL_ERROR
from proto.messages import Frame, MessageKind
from worker.session_leak_logging import actor_queue_fields, log_session_leak_detail, log_session_leak_summary

if TYPE_CHECKING:
    import ray

    from proto.config import SessionConfig, WorkerConfig


logger = logging.getLogger(__name__)


def _queue_detail_fields(output_queues: dict[str, deque]) -> dict[str, object]:
    return {"tracked_sessions": list(output_queues)}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    rank = int(round((p / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(rank, len(ordered) - 1))]


def _make_actor_cls() -> type:
    """Build the Ray actor class inside a function to defer the ray import."""
    import ray as _ray

    @_ray.remote(num_gpus=1)
    class RaonWorkerActor:
        """Ray actor that wraps a RaonEngine on a single GPU.

        Internal decode loop processes all sessions round-robin.
        feed_audio/get_output are non-blocking buffer operations.
        """

        def __init__(self) -> None:
            self._engine: "RaonEngine | None" = None
            self._started_at: float = 0.0
            self._output_queues: dict[str, deque] = {}
            self._loop_running = False
            self._last_cleanup_time: float = 0.0
            raw_max_decode = os.environ.get("FD_MAX_DECODE_FRAMES_PER_CALL", "1")
            try:
                parsed = int(raw_max_decode)
            except ValueError:
                parsed = 1
            self._max_decode_frames_per_call = max(1, min(parsed, 3))
            self._feed_and_decode_samples: deque[float] = deque(maxlen=2048)
            self._slow_feed_and_decode = 0

        def initialize(self, worker_config: "WorkerConfig") -> dict:
            """Load the model onto the GPU assigned by Ray."""
            import torch

            from worker.engine import RaonEngine

            # Ray already isolates GPUs via CUDA_VISIBLE_DEVICES for each actor.
            # Do NOT override it — just use device 0 (the only visible GPU).

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            self._engine = RaonEngine(config=worker_config)
            self._engine.load_model()
            self._started_at = time.time()
            self.start_cleanup_loop()

            return {
                "status": "initialized",
                "gpu_id": worker_config.gpu_id,
                "max_sessions": worker_config.max_sessions,
            }

        def start_cleanup_loop(self):
            import threading

            def _loop():
                import time as _time
                while True:
                    _time.sleep(5.0)
                    if self._engine is not None:
                        closed = self._engine.cleanup_idle_sessions()
                        for sid in closed:
                            removed = self._output_queues.pop(sid, None)
                            fields = actor_queue_fields(
                                session_id=sid,
                                queue_count=len(self._output_queues),
                                queue_depth=len(removed) if removed is not None else None,
                                engine_has_session=sid in self._engine.sessions,
                            )
                            log_session_leak_summary(logger, "actor_queue_removed_idle", **fields)
                            log_session_leak_detail(
                                logger,
                                "actor_queue_removed_idle",
                                **fields,
                                **_queue_detail_fields(self._output_queues),
                            )

            t = threading.Thread(target=_loop, daemon=True)
            t.start()

        def create_session(
            self, session_id: str, session_config: "SessionConfig"
        ) -> bool:
            """Create a new Raon session. Returns True on success."""
            assert self._engine is not None, "Actor not initialized"
            ok = self._engine.create_session(session_id, session_config)
            if ok:
                self._output_queues[session_id] = deque()
                fields = actor_queue_fields(
                    session_id=session_id,
                    queue_count=len(self._output_queues),
                    queue_depth=len(self._output_queues[session_id]),
                    engine_has_session=session_id in self._engine.sessions,
                )
                log_session_leak_summary(logger, "actor_queue_created", **fields)
                log_session_leak_detail(
                    logger,
                    "actor_queue_created",
                    **fields,
                    **_queue_detail_fields(self._output_queues),
                )
            return ok

        def close_session(self, session_id: str, reason: str = CLIENT_DISCONNECT) -> None:
            """Close and free resources for a session."""
            assert self._engine is not None, "Actor not initialized"
            self._engine.close_session(session_id, reason=reason)
            removed = self._output_queues.pop(session_id, None)
            fields = actor_queue_fields(
                session_id=session_id,
                queue_count=len(self._output_queues),
                queue_depth=len(removed) if removed is not None else None,
                engine_has_session=session_id in self._engine.sessions,
            )
            log_session_leak_summary(logger, "actor_queue_removed", **fields, reason=reason)
            log_session_leak_detail(
                logger,
                "actor_queue_removed",
                **fields,
                reason=reason,
                **_queue_detail_fields(self._output_queues),
            )

        def feed_audio(self, session_id: str, pcm_bytes: bytes) -> None:
            """Buffer incoming PCM audio for a session (non-blocking)."""
            assert self._engine is not None, "Actor not initialized"
            self._engine.feed_audio(session_id, pcm_bytes)

        def get_output(self, session_id: str) -> "list[Frame]":
            """Drain pending output frames from the queue (non-blocking).

            Does NOT trigger inference — frames are produced by run_decode_loop.
            Returns a CLOSE frame if the session no longer exists in the engine.
            """
            if self._engine is not None and session_id not in self._engine.sessions:
                removed = self._output_queues.pop(session_id, None)
                fields = actor_queue_fields(
                    session_id=session_id,
                    queue_count=len(self._output_queues),
                    queue_depth=len(removed) if removed is not None else None,
                    engine_has_session=False,
                )
                log_session_leak_summary(logger, "actor_queue_orphaned", **fields, reason="engine_missing_session")
                log_session_leak_detail(
                    logger,
                    "actor_queue_orphaned",
                    **fields,
                    reason="engine_missing_session",
                    **_queue_detail_fields(self._output_queues),
                )
                return [Frame(kind=MessageKind.CLOSE, payload=b"")]
            q = self._output_queues.get(session_id)
            if not q:
                return []
            frames = list(q)
            q.clear()
            return frames

        def feed_and_decode(
            self, session_id: str, pcm_bytes: bytes
        ) -> "list[Frame]":
            """Combined feed + decode + drain in ONE Ray call.

            Feeds audio, runs decode ONLY for the requesting session
            (no round-robin overhead), and returns output frames inline.
            """
            assert self._engine is not None, "Actor not initialized"
            t_start = time.perf_counter()

            # 1. Feed audio into this session's buffer
            self._engine.feed_audio(session_id, pcm_bytes)

            # 2. Decode ONLY this session (not all sessions)
            result_frames: list = []
            state = self._engine.sessions.get(session_id)
            if state is not None:
                frame_bytes = state.config.audio.frame_size * 4
                # Fairness guard: decode at most N frames per RPC (default N=1).
                for _ in range(self._max_decode_frames_per_call):
                    state = self._engine.sessions.get(session_id)
                    if state is None:
                        break
                    should_step = bool(state.close_requested_reason) or len(state.raw_input_bytes) >= frame_bytes
                    if not should_step:
                        break
                    output = self._engine.run_step(session_id)
                    if output:
                        result_frames.extend(output)
                    if any(frame.kind == MessageKind.CLOSE for frame in output):
                        break

            elapsed_ms = (time.perf_counter() - t_start) * 1000
            self._feed_and_decode_samples.append(elapsed_ms)
            if elapsed_ms > 200:
                self._slow_feed_and_decode += 1
                logger.warning(
                    "slow_feed_and_decode session=%s elapsed=%.1fms frames=%d",
                    session_id, elapsed_ms, len(result_frames),
                )

            # 4. Check if session was cleaned up
            if session_id not in self._engine.sessions:
                if not any(frame.kind == MessageKind.CLOSE for frame in result_frames):
                    result_frames.append(Frame.close(reason=INTERNAL_ERROR))
                return result_frames

            return result_frames

        def run_decode_loop(self, max_steps: int = 500) -> dict:
            """Process all active sessions in round-robin for up to max_steps.

            Called periodically by the gateway/launch orchestrator.
            Returns stats about what was processed.
            """
            assert self._engine is not None, "Actor not initialized"
            steps = 0
            frames_produced = 0

            for _ in range(max_steps):
                ready_sessions = []
                for sid, state in self._engine.sessions.items():
                    frame_bytes = state.config.audio.frame_size * 4
                    if len(state.raw_input_bytes) >= frame_bytes:
                        ready_sessions.append(sid)

                if not ready_sessions:
                    break

                for sid in ready_sessions:
                    output = self._engine.run_step(sid)
                    if output:
                        q = self._output_queues.get(sid)
                        if q is not None:
                            q.extend(output)
                            frames_produced += len(output)
                    steps += 1

            return {"steps": steps, "frames_produced": frames_produced}

        def health(self) -> dict:
            """Report actor health: session count, GPU id, uptime."""
            if self._engine is None:
                return {
                    "status": "not_initialized",
                    "uptime": 0.0,
                }
            info = self._engine.health()
            info["uptime"] = time.time() - self._started_at
            samples = list(self._feed_and_decode_samples)
            feed_and_decode_metrics = {
                "count": len(samples),
                "avg": (sum(samples) / len(samples)) if samples else 0.0,
                "p50": _percentile(samples, 50),
                "p95": _percentile(samples, 95),
                "p99": _percentile(samples, 99),
                "max": max(samples) if samples else 0.0,
            }
            metrics = info.setdefault("metrics", {})
            sampled = metrics.setdefault("sampled", {})
            sampled["feed_and_decode_ms"] = feed_and_decode_metrics
            counters = metrics.setdefault("counters", {})
            counters["slow_feed_and_decode"] = self._slow_feed_and_decode
            info["max_decode_frames_per_call"] = self._max_decode_frames_per_call
            return info

        def list_sessions(self) -> list[str]:
            """Return IDs of all active sessions."""
            if self._engine is None:
                return []
            return list(self._engine.sessions.keys())

        def cleanup_idle(self) -> list[str]:
            """Close idle sessions and return their IDs."""
            if self._engine is None:
                return []
            closed = self._engine.cleanup_idle_sessions()
            for sid in closed:
                removed = self._output_queues.pop(sid, None)
                fields = actor_queue_fields(
                    session_id=sid,
                    queue_count=len(self._output_queues),
                    queue_depth=len(removed) if removed is not None else None,
                    engine_has_session=sid in self._engine.sessions,
                )
                log_session_leak_summary(logger, "actor_queue_removed_idle", **fields)
                log_session_leak_detail(
                    logger,
                    "actor_queue_removed_idle",
                    **fields,
                    **_queue_detail_fields(self._output_queues),
                )
            return closed

    return RaonWorkerActor


# Lazy singleton so `import actor` doesn't require ray at import time.
_actor_cls: type | None = None


def get_raon_actor_cls() -> type:
    """Return the Ray remote actor class, creating it on first call."""
    global _actor_cls
    if _actor_cls is None:
        _actor_cls = _make_actor_cls()
    return _actor_cls



def __getattr__(name: str) -> object:
    if name == "RaonWorkerActor":
        return get_raon_actor_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
