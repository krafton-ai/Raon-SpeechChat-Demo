"""Worker registry for GPU actor management and session affinity tracking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class WorkerInfo:
    """Metadata and state for a registered GPU worker actor."""
    worker_id: str
    actor_handle: Any
    gpu_id: int
    max_sessions: int
    current_sessions: int = 0
    last_health_check: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    healthy: bool = True
    last_error: Optional[str] = None
    actor_session_count: Optional[int] = None
    actor_max_sessions: Optional[int] = None

    @property
    def load_ratio(self) -> float:
        if self.max_sessions == 0:
            return 1.0
        return self.current_sessions / self.max_sessions

    @property
    def has_capacity(self) -> bool:
        return self.healthy and self.current_sessions < self.max_sessions


@dataclass
class SessionAssignment:
    worker_id: str
    state: str  # reserved | active
    updated_at: datetime = field(default_factory=datetime.utcnow)


class WorkerRegistry:
    """In-memory registry of GPU worker Ray actors with session affinity."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerInfo] = {}
        self._session_map: dict[str, SessionAssignment] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        worker_id: str,
        actor_handle: Any,
        gpu_id: int,
        max_sessions: int,
    ) -> None:
        """Register a Ray actor as a GPU worker."""
        async with self._lock:
            self._workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                actor_handle=actor_handle,
                gpu_id=gpu_id,
                max_sessions=max_sessions,
            )

    async def deregister(self, worker_id: str) -> None:
        """Remove a worker and release any sessions assigned to it."""
        async with self._lock:
            self._workers.pop(worker_id, None)
            # Clean up any sessions that were assigned to this worker
            stale = [sid for sid, assignment in self._session_map.items() if assignment.worker_id == worker_id]
            for sid in stale:
                del self._session_map[sid]

    async def get_worker(self, worker_id: str) -> WorkerInfo:
        """Look up a worker by ID. Raises KeyError if not found."""
        async with self._lock:
            if worker_id not in self._workers:
                raise KeyError(f"Worker not found: {worker_id}")
            return self._workers[worker_id]

    async def list_workers(self) -> list[WorkerInfo]:
        """Return all registered workers."""
        async with self._lock:
            return list(self._workers.values())

    async def get_worker_for_session(self, session_id: str) -> Optional[WorkerInfo]:
        """Look up the worker assigned to a session, or None if not assigned."""
        async with self._lock:
            assignment = self._session_map.get(session_id)
            if assignment is None:
                return None
            return self._workers.get(assignment.worker_id)

    async def assign_session(self, session_id: str, worker_id: str) -> None:
        """Compatibility helper: record an active session mapping."""
        async with self._lock:
            if worker_id not in self._workers:
                raise KeyError(f"Worker not found: {worker_id}")
            existing = self._session_map.get(session_id)
            if existing is not None and existing.worker_id == worker_id and existing.state == "active":
                existing.updated_at = datetime.utcnow()
                return
            if existing is not None and existing.worker_id != worker_id:
                self._decrement_worker_session(existing.worker_id)
            self._session_map[session_id] = SessionAssignment(
                worker_id=worker_id,
                state="active",
            )
            self._workers[worker_id].current_sessions += 1

    async def reserve_session(self, session_id: str) -> Optional[WorkerInfo]:
        """Atomically choose a worker and reserve one capacity slot for session_id."""
        async with self._lock:
            existing = self._session_map.get(session_id)
            if existing is not None:
                worker = self._workers.get(existing.worker_id)
                if worker is not None:
                    return worker

            candidates = [w for w in self._workers.values() if w.has_capacity]
            if not candidates:
                return None

            # Least-loaded, deterministic tiebreaker by worker_id.
            chosen = min(candidates, key=lambda w: (w.load_ratio, w.current_sessions, w.worker_id))
            self._session_map[session_id] = SessionAssignment(
                worker_id=chosen.worker_id,
                state="reserved",
            )
            chosen.current_sessions += 1
            return chosen

    async def commit_session(self, session_id: str, worker_id: Optional[str] = None) -> bool:
        """Mark a reserved session as active.

        If a reservation mapping was reaped before commit but the worker finished
        creating the session, callers can provide worker_id to recover mapping.
        """
        async with self._lock:
            assignment = self._session_map.get(session_id)
            if assignment is None:
                if worker_id is None:
                    return False
                worker = self._workers.get(worker_id)
                if worker is None:
                    return False
                self._session_map[session_id] = SessionAssignment(
                    worker_id=worker_id,
                    state="active",
                )
                worker.current_sessions += 1
                return True
            if worker_id is not None and assignment.worker_id != worker_id:
                return False
            if assignment.state == "active":
                assignment.updated_at = datetime.utcnow()
                return True
            if assignment.state != "reserved":
                return False
            assignment.state = "active"
            assignment.updated_at = datetime.utcnow()
            return True

    async def release_session(self, session_id: str) -> Optional[SessionAssignment]:
        """Remove a session mapping and decrement worker session count."""
        async with self._lock:
            assignment = self._session_map.pop(session_id, None)
            if assignment is None:
                return None
            self._decrement_worker_session(assignment.worker_id)
            return assignment

    async def set_worker_health(
        self,
        worker_id: str,
        *,
        healthy: bool,
        actor_session_count: Optional[int] = None,
        actor_max_sessions: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow()
        async with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                return
            worker.healthy = healthy
            worker.last_health_check = now
            worker.last_heartbeat = now
            worker.last_error = error
            if actor_session_count is not None:
                worker.actor_session_count = int(actor_session_count)
            if actor_max_sessions is not None:
                worker.actor_max_sessions = int(actor_max_sessions)

    async def list_assignments(self) -> dict[str, dict[str, str]]:
        async with self._lock:
            return {
                sid: {
                    "worker_id": assignment.worker_id,
                    "state": assignment.state,
                    "updated_at": assignment.updated_at.isoformat(),
                }
                for sid, assignment in self._session_map.items()
            }

    async def session_count(self) -> int:
        async with self._lock:
            return len(self._session_map)

    async def healthy_worker_count(self) -> int:
        async with self._lock:
            return len([w for w in self._workers.values() if w.healthy])

    async def placeable_worker_count(self) -> int:
        async with self._lock:
            return len([w for w in self._workers.values() if w.has_capacity])

    async def reconcile_worker_sessions(
        self,
        worker_id: str,
        live_session_ids: list[str],
        reserved_ttl_sec: float = 60.0,
    ) -> dict[str, object]:
        """Reconcile registry mappings against worker-local live sessions."""
        now = datetime.utcnow()
        live_set = set(live_session_ids)
        stale_active: list[str] = []
        stale_reserved: list[str] = []
        promoted_reserved: list[str] = []

        async with self._lock:
            expected_active = {
                sid
                for sid, assignment in self._session_map.items()
                if assignment.worker_id == worker_id and assignment.state == "active"
            }
            expected_reserved = {
                sid
                for sid, assignment in self._session_map.items()
                if assignment.worker_id == worker_id and assignment.state == "reserved"
            }

            stale_active = sorted(expected_active - live_set)
            for sid in stale_active:
                removed = self._session_map.pop(sid, None)
                if removed is not None:
                    self._decrement_worker_session(removed.worker_id)

            for sid in sorted(expected_reserved):
                assignment = self._session_map.get(sid)
                if assignment is None:
                    continue
                if sid in live_set:
                    assignment.state = "active"
                    assignment.updated_at = now
                    promoted_reserved.append(sid)
                    continue
                age_sec = (now - assignment.updated_at).total_seconds()
                if age_sec > reserved_ttl_sec:
                    self._session_map.pop(sid, None)
                    self._decrement_worker_session(assignment.worker_id)
                    stale_reserved.append(sid)

            orphan_sessions = sorted(live_set - expected_active - expected_reserved)

        return {
            "worker_id": worker_id,
            "expected_active": len(expected_active),
            "expected_reserved": len(expected_reserved),
            "live_sessions": len(live_set),
            "stale_active_removed": len(stale_active),
            "stale_reserved_removed": len(stale_reserved),
            "reserved_promoted_to_active": len(promoted_reserved),
            "stale_active_session_ids": stale_active,
            "stale_reserved_session_ids": stale_reserved,
            "reserved_promoted_session_ids": promoted_reserved,
            "orphan_sessions": orphan_sessions,
        }

    def _decrement_worker_session(self, worker_id: str) -> None:
        worker = self._workers.get(worker_id)
        if worker is None:
            return
        worker.current_sessions = max(0, worker.current_sessions - 1)
