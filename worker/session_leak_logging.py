"""Shared logging helpers for session leak investigation."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("session_leak_logging")


def session_leak_logging_enabled() -> bool:
    raw = os.environ.get("FD_ENABLE_SESSION_LEAK_LOGGING")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def format_log_kv(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(fields):
        value = fields[key]
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def log_session_leak_summary(target_logger: logging.Logger, stage: str, **fields: Any) -> None:
    rendered = format_log_kv(fields)
    if rendered:
        target_logger.info("session_leak_summary stage=%s %s", stage, rendered)
    else:
        target_logger.info("session_leak_summary stage=%s", stage)


def log_session_leak_detail(target_logger: logging.Logger, stage: str, **fields: Any) -> None:
    if not session_leak_logging_enabled():
        return
    rendered = format_log_kv(fields)
    if rendered:
        target_logger.info("session_leak_detail stage=%s %s", stage, rendered)
    else:
        target_logger.info("session_leak_detail stage=%s", stage)


def engine_registry_fields(
    *,
    active_sessions: list[str],
    pending_sessions: list[str],
    session_id: str | None = None,
    has_state: bool | None = None,
    has_lock: bool | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "session_id": session_id,
        "active_count": len(active_sessions),
        "pending_count": len(pending_sessions),
        "has_state": has_state,
        "has_lock": has_lock,
    }
    if session_leak_logging_enabled():
        fields["active_sessions"] = active_sessions
        fields["pending_sessions"] = pending_sessions
    return fields


def actor_queue_fields(
    *,
    session_id: str,
    queue_count: int,
    queue_depth: int | None = None,
    engine_has_session: bool | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "queue_count": queue_count,
        "queue_depth": queue_depth,
        "engine_has_session": engine_has_session,
    }


def cleanup_resource_fields(
    *,
    session_id: str | None = None,
    stream_id: int | None = None,
    has_past_key_values: bool | None = None,
    has_audio_cache: bool | None = None,
    has_talker_cache: bool | None = None,
    has_talker_mask: bool | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "stream_id": stream_id,
        "has_past_key_values": has_past_key_values,
        "has_audio_cache": has_audio_cache,
        "has_talker_cache": has_talker_cache,
        "has_talker_mask": has_talker_mask,
    }
