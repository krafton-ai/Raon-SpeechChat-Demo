"""FastAPI WebSocket gateway for full-duplex streaming."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import ray
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from starlette.websockets import WebSocketDisconnect

from gateway.proxy import SessionProxy
from proto.close_reasons import CLIENT_DISCONNECT, INTERNAL_ERROR, normalize_close_reason
from proto.config import GatewayConfig, SamplingConfig, SessionConfig
from proto.messages import Frame
from proto.prompt_map import resolve_prompt_language, resolve_speak_first
from worker.session_leak_logging import log_session_leak_summary

logger = logging.getLogger(__name__)

_start_time = time.time()
_config = GatewayConfig()
_active_sessions: dict[str, SessionProxy] = {}  # session_id → SessionProxy
_metrics_lock = threading.Lock()
_metric_samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=2048))
_metric_counters: dict[str, int] = defaultdict(int)

# Serve only from the Next.js static export.
_FRONTEND_DIR = Path(__file__).parent.parent / "frontend-next" / "out"
if not _FRONTEND_DIR.exists():
    raise RuntimeError(
        f"Required frontend bundle is missing: {_FRONTEND_DIR}. "
        "Build/export frontend-next before starting the gateway."
    )


def _as_int(raw: str | None, default: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _as_float(raw: str | None, default: float) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _observe_metric(name: str, value: float) -> None:
    with _metrics_lock:
        _metric_samples[name].append(float(value))


def _inc_counter(name: str, amount: int = 1) -> None:
    with _metrics_lock:
        _metric_counters[name] += int(amount)


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


def _sample_summary(name: str) -> dict[str, float]:
    with _metrics_lock:
        values = list(_metric_samples.get(name, ()))
    if not values:
        return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values),
    }


def _metrics_snapshot() -> dict:
    with _metrics_lock:
        names = list(_metric_samples.keys())
        counters = dict(_metric_counters)
    return {
        "counters": counters,
        "sampled": {name: _sample_summary(name) for name in names},
    }


def create_app(config: Optional[GatewayConfig] = None) -> FastAPI:
    global _config
    if config is not None:
        _config = config

    app = FastAPI(title="fd-demo gateway")

    @app.get("/health")
    async def health() -> JSONResponse:
        router_status: dict = {}
        status = "ok"
        try:
            router = ray.get_actor("fd_router", namespace="default")
            router_status = await router.status.remote()
            if router_status.get("healthy_worker_count", 0) == 0 and router_status.get("workers"):
                status = "degraded"
        except Exception as exc:
            status = "degraded"
            router_status = {"available": False, "error": str(exc)}

        gateway_total_queue_drops = sum(
            p.metrics()["queue_drops"] for p in _active_sessions.values()
        )
        gateway_active_queue_depth = sum(
            p.metrics()["queue_depth"] for p in _active_sessions.values()
        )

        return JSONResponse(
            {
                "status": status,
                "mode": "ray",
                "active_sessions": len(_active_sessions),
                "uptime_seconds": round(time.time() - _start_time, 2),
                "gateway_total_queue_drops": gateway_total_queue_drops,
                "gateway_active_queue_depth": gateway_active_queue_depth,
                "router": router_status,
                "metrics": _metrics_snapshot(),
            }
        )

    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/fd-demo/", status_code=307)

    @app.get("/fd-demo")
    async def fd_demo_no_slash() -> RedirectResponse:
        return RedirectResponse(url="/fd-demo/", status_code=307)

    @app.get("/fd-demo/")
    async def index() -> FileResponse:
        return FileResponse(_FRONTEND_DIR / "index.html")

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(status_code=204)

    from starlette.staticfiles import StaticFiles

    _next_dir = _FRONTEND_DIR / "_next"
    if _next_dir.exists():
        app.mount("/fd-demo/_next", StaticFiles(directory=str(_next_dir)), name="nextjs-assets-prefixed")
        app.mount("/_next", StaticFiles(directory=str(_next_dir)), name="nextjs-assets")

    async def _ws_chat_impl(websocket: WebSocket) -> None:
        t_accept = time.perf_counter()
        query = websocket.query_params
        prompt = query.get("prompt") or "eng:full_duplex:listen-first"
        speak_first = resolve_speak_first(prompt, "system")
        prompt_language = resolve_prompt_language(
            prompt,
            "system",
            default="kor" if query.get("prompt_language") == "kor" else "eng",
        )
        system_prompt_style = query.get("system_prompt_style") or "generic"
        system_prompt_persona = (query.get("system_prompt_persona") or "").strip() or None
        system_prompt_context = (query.get("system_prompt_context") or "").strip() or None
        custom_system_prompt = (query.get("custom_system_prompt") or "").strip() or None
        temperature = _as_float(query.get("temperature"), 0.7)
        top_k = _as_int(query.get("top_k"), 50)
        top_p = _as_float(query.get("top_p"), 0.8)
        eos_penalty = _as_float(query.get("eos_penalty"), 0.0)
        bc_penalty = _as_float(query.get("bc_penalty"), 0.0)
        repetition_penalty = _as_float(query.get("repetition_penalty"), 1.0)
        speaker_mode = query.get("speaker_mode") or "default"
        speaker_key = query.get("speaker_key") or ""

        logger.info(
            "ws handshake path=%s origin=%r host=%r xfp=%r",
            websocket.url.path,
            websocket.headers.get("origin"),
            websocket.headers.get("host"),
            websocket.headers.get("x-forwarded-proto"),
        )

        await websocket.accept()
        session_id = str(uuid.uuid4())
        close_reason = CLIENT_DISCONNECT
        log_session_leak_summary(
            logger,
            "gateway_accept",
            session_id=session_id,
            active_count=len(_active_sessions),
        )
        logger.info(
            "new session %s prompt=%r language=%s speak_first=%s style=%s temperature=%.2f speaker_mode=%s",
            session_id,
            prompt,
            prompt_language,
            speak_first,
            system_prompt_style,
            temperature,
            speaker_mode,
        )

        session_cfg = SessionConfig(
            session_id=session_id,
            prompt=prompt,
            prompt_language=prompt_language,
            speak_first=speak_first,
            system_prompt_style=system_prompt_style,
            system_prompt_persona=system_prompt_persona,
            system_prompt_context=system_prompt_context,
            custom_system_prompt=custom_system_prompt,
            sampling=SamplingConfig(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_penalty=eos_penalty,
                bc_penalty=bc_penalty,
                repetition_penalty=repetition_penalty,
            ),
            speaker_mode=speaker_mode,
            speaker_key=speaker_key or None,
        )

        router = None
        worker_actor = None
        worker_id = "unknown"
        assigned_gpu = "unknown"
        t_assigned = None
        t_ready = None

        try:
            router = ray.get_actor("fd_router", namespace="default")

            t_assign = time.perf_counter()
            reservation = await router.reserve.remote(session_id)
            t_assigned = time.perf_counter()
            _observe_metric("accept_to_assigned_ms", (t_assigned - t_accept) * 1000.0)

            if reservation is None:
                _inc_counter("reject_no_workers")
                logger.warning("no workers available for session %s", session_id)
                await websocket.send_bytes(Frame.error("no workers available").encode())
                close_reason = INTERNAL_ERROR
                await websocket.close()
                return

            worker_actor = reservation["actor_handle"]
            worker_id = reservation["worker_id"]
            assigned_gpu = reservation["gpu_id"]
            logger.info("session %s reserved on worker=%s gpu=%s", session_id, worker_id, assigned_gpu)
            log_session_leak_summary(
                logger,
                "gateway_reserved",
                session_id=session_id,
                active_count=len(_active_sessions),
                worker_id=worker_id,
                gpu_id=assigned_gpu,
            )

            t_create = time.perf_counter()
            created = await worker_actor.create_session.remote(session_id, session_cfg)
            if not created:
                raise RuntimeError("worker rejected session creation")

            committed = await router.commit.remote(session_id, worker_id)
            if not committed:
                raise RuntimeError("router commit failed")

            await websocket.send_bytes(Frame.ready().encode())
            t_ready = time.perf_counter()
            _observe_metric("assigned_to_ready_ms", (t_ready - t_create) * 1000.0)
            _observe_metric("accept_to_ready_ms", (t_ready - t_accept) * 1000.0)
            logger.info(
                "READY frame sent (session=%s worker=%s gpu=%s)",
                session_id,
                worker_id,
                assigned_gpu,
            )
        except Exception as exc:
            _inc_counter("session_init_failures")
            close_reason = INTERNAL_ERROR
            logger.error(
                "session init failed (session=%s worker=%s gpu=%s): %s",
                session_id,
                worker_id,
                assigned_gpu,
                exc,
            )
            worker_cleanup_completed = False
            router_release_completed = False
            router_release_removed = False
            try:
                await websocket.send_bytes(Frame.error(f"session init failed: {exc}").encode())
            except Exception:
                pass
            if worker_actor is not None:
                try:
                    await worker_actor.close_session.remote(session_id, close_reason)
                    worker_cleanup_completed = True
                except Exception:
                    logger.warning(
                        "close_session failed (session=%s reason=%s), worker cleanup may be incomplete",
                        session_id, close_reason, exc_info=True,
                    )
            if router is not None:
                try:
                    release_result = await router.release.remote(session_id, close_reason)
                    router_release_completed = True
                    router_release_removed = bool((release_result or {}).get("released"))
                except Exception:
                    logger.warning(
                        "router.release failed (session=%s reason=%s), router state may be stale",
                        session_id, close_reason, exc_info=True,
                    )
            log_session_leak_summary(
                logger,
                "gateway_cleanup_init_failure",
                session_id=session_id,
                active_count=len(_active_sessions),
                worker_id=worker_id,
                gpu_id=assigned_gpu,
                reason=close_reason,
                worker_cleanup_completed=worker_cleanup_completed,
                router_release_completed=router_release_completed,
                router_release_removed=router_release_removed,
            )
            await websocket.close()
            return

        proxy = SessionProxy(
            session_id=session_id,
            ping_interval=_config.ping_interval,
            ping_timeout=_config.ping_timeout,
        )
        active_count_before = len(_active_sessions)
        _active_sessions[session_id] = proxy
        log_session_leak_summary(
            logger,
            "gateway_active_insert",
            session_id=session_id,
            active_count_before=active_count_before,
            active_count_after=len(_active_sessions),
            worker_id=worker_id,
            gpu_id=assigned_gpu,
        )
        try:
            await proxy.run(websocket, worker_actor)
            close_reason = normalize_close_reason(proxy.close_reason)
        except WebSocketDisconnect:
            close_reason = CLIENT_DISCONNECT
            logger.info("client disconnected (session=%s)", session_id)
        except Exception as exc:
            close_reason = INTERNAL_ERROR
            logger.error("proxy error (session=%s): %s", session_id, exc)
        finally:
            active_count_before = len(_active_sessions)
            _active_sessions.pop(session_id, None)
            close_reason = normalize_close_reason(close_reason)
            _inc_counter(f"close_reason:{close_reason}")
            worker_cleanup_completed = False
            if worker_actor is not None:
                try:
                    await worker_actor.close_session.remote(session_id, close_reason)
                    worker_cleanup_completed = True
                except Exception:
                    logger.warning(
                        "close_session failed (session=%s reason=%s), worker cleanup may be incomplete",
                        session_id, close_reason, exc_info=True,
                    )
            router_release_completed = False
            router_release_removed = False
            if router is not None:
                try:
                    release_result = await router.release.remote(session_id, close_reason)
                    router_release_completed = True
                    router_release_removed = bool((release_result or {}).get("released"))
                except Exception:
                    logger.warning(
                        "router.release failed (session=%s reason=%s), router state may be stale",
                        session_id, close_reason, exc_info=True,
                    )
            log_session_leak_summary(
                logger,
                "gateway_cleanup_done",
                session_id=session_id,
                active_count_before=active_count_before,
                active_count_after=len(_active_sessions),
                worker_id=worker_id,
                gpu_id=assigned_gpu,
                reason=close_reason,
                worker_cleanup_completed=worker_cleanup_completed,
                router_release_completed=router_release_completed,
                router_release_removed=router_release_removed,
            )
            logger.info(
                "session %s cleaned up worker=%s gpu=%s reason=%s assigned_ms=%.1f ready_ms=%.1f",
                session_id,
                worker_id,
                assigned_gpu,
                close_reason,
                ((t_assigned - t_accept) * 1000.0) if t_assigned else -1.0,
                ((t_ready - t_assigned) * 1000.0) if (t_ready and t_assigned) else -1.0,
            )

    @app.websocket("/ws/chat")
    async def ws_chat(websocket: WebSocket) -> None:
        await _ws_chat_impl(websocket)

    @app.websocket("/ws/chat/")
    async def ws_chat_slash(websocket: WebSocket) -> None:
        await _ws_chat_impl(websocket)

    @app.websocket("/fd-demo/ws/chat")
    async def ws_chat_prefixed(websocket: WebSocket) -> None:
        await _ws_chat_impl(websocket)

    @app.websocket("/fd-demo/ws/chat/")
    async def ws_chat_prefixed_slash(websocket: WebSocket) -> None:
        await _ws_chat_impl(websocket)

    return app


def main() -> None:
    cfg = GatewayConfig()
    app = create_app(cfg)
    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        ws="wsproto",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )


if __name__ == "__main__":
    main()
