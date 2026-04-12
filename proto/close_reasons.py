"""Standardized session close/degrade reason codes."""

from __future__ import annotations

OVERLOADED_BACKLOG = "overloaded_backlog"
WORKER_UNHEALTHY = "worker_unhealthy"
IDLE_TIMEOUT = "idle_timeout"
CLIENT_DISCONNECT = "client_disconnect"
INTERNAL_ERROR = "internal_error"
INVALID_CLIENT_FRAME = "invalid_client_frame"

VALID_CLOSE_REASONS = {
    OVERLOADED_BACKLOG,
    WORKER_UNHEALTHY,
    IDLE_TIMEOUT,
    CLIENT_DISCONNECT,
    INTERNAL_ERROR,
    INVALID_CLIENT_FRAME,
}


def normalize_close_reason(reason: str | None) -> str:
    """Normalize unknown/empty reason to INTERNAL_ERROR."""
    if not reason:
        return INTERNAL_ERROR
    if reason in VALID_CLOSE_REASONS:
        return reason
    return INTERNAL_ERROR
