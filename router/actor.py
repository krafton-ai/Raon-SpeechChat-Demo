"""RouterActor Ray remote class for session scheduling and worker health."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict


def make_router_actor_cls() -> type:
    """Build the RouterActor Ray class (defers ray import until after ray.init)."""
    import ray

    from proto.close_reasons import INTERNAL_ERROR, normalize_close_reason
    from router.registry import WorkerRegistry

    @ray.remote
    class RouterActor:
        def __init__(
            self,
            health_interval: float = 5.0,
            reconcile_interval: float = 7.5,
            health_timeout: float = 2.5,
            reserved_ttl_sec: float = 60.0,
        ) -> None:
            import logging as _logging

            self._registry = WorkerRegistry()
            self._log = _logging.getLogger("fd-demo.router")
            self._started_at = time.time()
            self._health_interval = max(1.0, float(health_interval))
            self._reconcile_interval = max(2.0, float(reconcile_interval))
            self._health_timeout = max(0.5, float(health_timeout))
            self._reserved_ttl_sec = max(5.0, float(reserved_ttl_sec))
            self._metrics = defaultdict(int)
            self._loop_tasks_started = False

        def _ensure_background_tasks(self) -> None:
            if self._loop_tasks_started:
                return
            self._loop_tasks_started = True
            asyncio.create_task(self._health_loop())
            asyncio.create_task(self._reconcile_loop())

        async def _health_loop(self) -> None:
            while True:
                await asyncio.sleep(self._health_interval)
                workers = await self._registry.list_workers()
                for worker in workers:
                    try:
                        health = await asyncio.wait_for(
                            worker.actor_handle.health.remote(),
                            timeout=self._health_timeout,
                        )
                        reported_healthy = bool(health.get("healthy", True))
                        fatal_error = health.get("fatal_error")
                        if fatal_error:
                            reported_healthy = False
                            self._metrics["worker_fatal_reports"] += 1
                        await self._registry.set_worker_health(
                            worker.worker_id,
                            healthy=reported_healthy,
                            actor_session_count=int(health.get("session_count", 0)),
                            actor_max_sessions=int(health.get("max_sessions", worker.max_sessions)),
                            error=(str(fatal_error) if fatal_error else None),
                        )
                    except Exception as exc:
                        self._metrics["worker_health_failures"] += 1
                        await self._registry.set_worker_health(
                            worker.worker_id,
                            healthy=False,
                            error=str(exc),
                        )

        async def _reconcile_loop(self) -> None:
            while True:
                await asyncio.sleep(self._reconcile_interval)
                workers = await self._registry.list_workers()
                for worker in workers:
                    try:
                        live_sessions = await asyncio.wait_for(
                            worker.actor_handle.list_sessions.remote(),
                            timeout=self._health_timeout,
                        )
                        report = await self._registry.reconcile_worker_sessions(
                            worker.worker_id,
                            live_sessions,
                            reserved_ttl_sec=self._reserved_ttl_sec,
                        )
                        self._metrics["reconcile_runs"] += 1
                        self._metrics["reconcile_stale_active_removed"] += int(report["stale_active_removed"])
                        self._metrics["reconcile_stale_reserved_removed"] += int(report["stale_reserved_removed"])
                        self._metrics["reconcile_reserved_promoted"] += int(report["reserved_promoted_to_active"])
                        orphan_sessions = list(report["orphan_sessions"])
                        if orphan_sessions:
                            self._metrics["reconcile_orphan_sessions"] += len(orphan_sessions)
                        for sid in orphan_sessions:
                            try:
                                await worker.actor_handle.close_session.remote(sid, INTERNAL_ERROR)
                                self._metrics["reconcile_orphan_closed"] += 1
                            except Exception:
                                self._metrics["reconcile_orphan_close_failures"] += 1
                    except Exception as exc:
                        self._metrics["reconcile_failures"] += 1
                        await self._registry.set_worker_health(
                            worker.worker_id,
                            healthy=False,
                            error=f"reconcile: {exc}",
                        )

        async def register_worker(self, worker_id, actor_handle, gpu_id, max_sessions):
            await self._registry.register(worker_id, actor_handle, gpu_id, max_sessions)
            self._ensure_background_tasks()

        async def reserve(self, session_id):
            self._ensure_background_tasks()
            chosen = await self._registry.reserve_session(session_id)
            if chosen is None:
                return None
            self._metrics["session_reserved"] += 1
            return {
                "worker_id": chosen.worker_id,
                "gpu_id": chosen.gpu_id,
                "actor_handle": chosen.actor_handle,
            }

        async def commit(self, session_id, worker_id=None):
            ok = await self._registry.commit_session(session_id, worker_id=worker_id)
            if ok:
                self._metrics["session_committed"] += 1
            return ok

        async def assign(self, session_id):
            reserved = await self.reserve(session_id)
            if reserved is None:
                return None
            return reserved["actor_handle"]

        async def release(self, session_id, reason=INTERNAL_ERROR):
            assignment = await self._registry.release_session(session_id)
            norm_reason = normalize_close_reason(reason)
            if assignment is not None:
                self._metrics["session_released"] += 1
                self._metrics[f"close_reason:{norm_reason}"] += 1
                return {
                    "released": True,
                    "reason": norm_reason,
                    "worker_id": assignment.worker_id,
                }
            self._metrics["release_missing"] += 1
            self._metrics[f"release_missing_reason:{norm_reason}"] += 1
            return {
                "released": False,
                "reason": norm_reason,
                "worker_id": None,
            }

        async def worker_count(self):
            return await self._registry.healthy_worker_count()

        async def has_placeable_worker(self):
            return (await self._registry.placeable_worker_count()) > 0

        async def status(self):
            workers = await self._registry.list_workers()
            assignments = await self._registry.list_assignments()
            placeable_count = await self._registry.placeable_worker_count()
            healthy_count = await self._registry.healthy_worker_count()
            return {
                "router_uptime_seconds": max(0.0, time.time() - self._started_at),
                "workers": [
                    {
                        "worker_id": worker.worker_id,
                        "gpu_id": worker.gpu_id,
                        "healthy": worker.healthy,
                        "current_sessions": worker.current_sessions,
                        "max_sessions": worker.max_sessions,
                        "last_heartbeat": (
                            worker.last_heartbeat.isoformat() if worker.last_heartbeat else None
                        ),
                        "last_health_check": (
                            worker.last_health_check.isoformat() if worker.last_health_check else None
                        ),
                        "last_error": worker.last_error,
                        "actor_session_count": worker.actor_session_count,
                        "actor_max_sessions": worker.actor_max_sessions,
                    }
                    for worker in workers
                ],
                "session_count": len(assignments),
                "assignments": assignments,
                "placeable_worker_count": placeable_count,
                "healthy_worker_count": healthy_count,
                "metrics": dict(self._metrics),
            }

    return RouterActor
