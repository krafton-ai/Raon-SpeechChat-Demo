"""Session scheduler: placement strategy and health monitoring for GPU workers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from proto.config import SessionConfig
from router.registry import WorkerRegistry

logger = logging.getLogger(__name__)


class SessionScheduler:
    """Places sessions on GPU workers using a least-loaded strategy."""

    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry = registry

    async def place_session(self, session_config: SessionConfig) -> tuple[str, Any]:
        """Select the least-loaded healthy worker and return (worker_id, actor_handle).

        Raises RuntimeError if no healthy worker has available capacity.
        """
        workers = await self._registry.list_workers()
        candidates = [w for w in workers if w.has_capacity]

        if not candidates:
            raise RuntimeError("No healthy GPU worker with available capacity")

        # Least-loaded with deterministic tie-breakers.
        chosen = min(candidates, key=lambda w: (w.load_ratio, w.current_sessions, w.worker_id))
        return chosen.worker_id, chosen.actor_handle

    async def health_check_loop(self, interval: float = 10.0) -> None:
        """Periodically ping all registered workers and update their health status."""
        while True:
            await asyncio.sleep(interval)
            workers = await self._registry.list_workers()
            for worker in workers:
                try:
                    health = await worker.actor_handle.health.remote()
                    actor_session_count = None
                    actor_max_sessions = None
                    if isinstance(health, dict):
                        actor_session_count = health.get("session_count")
                        actor_max_sessions = health.get("max_sessions")
                    await self._registry.set_worker_health(
                        worker.worker_id,
                        healthy=True,
                        actor_session_count=actor_session_count,
                        actor_max_sessions=actor_max_sessions,
                        error=None,
                    )
                except Exception as exc:
                    logger.warning(
                        "Health check failed for worker %s: %s", worker.worker_id, exc
                    )
                    await self._registry.set_worker_health(
                        worker.worker_id,
                        healthy=False,
                        error=str(exc),
                    )

    async def rebalance(self) -> None:
        """Log a warning if load skew between workers exceeds 2x."""
        workers = await self._registry.list_workers()
        healthy = [w for w in workers if w.healthy and w.max_sessions > 0]
        if len(healthy) < 2:
            return

        ratios = [w.load_ratio for w in healthy]
        max_ratio = max(ratios)
        min_ratio = min(ratios)

        if min_ratio > 0 and max_ratio / min_ratio > 2.0:
            logger.warning(
                "Load imbalance detected: max=%.2f min=%.2f (skew >2x). "
                "Consider rebalancing sessions across workers.",
                max_ratio,
                min_ratio,
            )
