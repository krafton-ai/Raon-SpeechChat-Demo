"""Helpers for exception-safe duplex cleanup paths."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from worker.session_leak_logging import (
    cleanup_resource_fields,
    format_log_kv,
    log_session_leak_detail,
    log_session_leak_summary,
)

logger = logging.getLogger(__name__)


def _run_cleanup_step(step_name: str, fn: Callable[[], object], **fields: Any) -> None:
    try:
        fn()
    except Exception:
        rendered_fields = {"step": step_name, **fields}
        log_session_leak_summary(logger, "cleanup_step_failed", **rendered_fields)
        logger.warning(
            "session_leak_summary stage=cleanup_step_failed %s",
            format_log_kv(rendered_fields),
            exc_info=True,
        )


def _state_cleanup_fields(state: Any) -> dict[str, Any]:
    return cleanup_resource_fields(
        session_id=getattr(state, "session_id", None),
        stream_id=getattr(state, "audio_decoder_stream_id", None),
        has_past_key_values=getattr(state, "past_key_values", None) is not None,
        has_audio_cache=getattr(state, "audio_input_encoder_cache", None) is not None,
        has_talker_cache=getattr(state, "talker_past_key_values", None) is not None,
        has_talker_mask=getattr(state, "talker_attention_mask", None) is not None,
    )


def cleanup_failed_duplex_init(
    *,
    stream_id: int | None,
    past_key_values: object | None,
    audio_input_encoder_cache: object | None,
    drain_audio_decoding_queue: Callable[[int], object],
    destroy_audio_decoder_stream: Callable[[int], object],
    free_past_key_values: Callable[[object], object],
) -> None:
    """Release any resources allocated before init_duplex_decoding_state failed."""
    fields = cleanup_resource_fields(
        stream_id=stream_id,
        has_past_key_values=past_key_values is not None,
        has_audio_cache=audio_input_encoder_cache is not None,
    )
    log_session_leak_summary(logger, "cleanup_failed_init_start", **fields)
    log_session_leak_detail(logger, "cleanup_failed_init_resources", **fields)
    if stream_id is not None:
        _run_cleanup_step(
            "drain_audio_decoder_stream",
            lambda stream_id=stream_id: drain_audio_decoding_queue(stream_id),
            **fields,
        )
        _run_cleanup_step(
            "destroy_audio_decoder_stream",
            lambda stream_id=stream_id: destroy_audio_decoder_stream(stream_id),
            **fields,
        )
    if past_key_values is not None:
        _run_cleanup_step(
            "free_past_key_values",
            lambda past_key_values=past_key_values: free_past_key_values(past_key_values),
            **fields,
        )
    if audio_input_encoder_cache is not None and hasattr(audio_input_encoder_cache, "reset"):
        _run_cleanup_step("reset_audio_input_encoder_cache", audio_input_encoder_cache.reset, **fields)
    log_session_leak_summary(logger, "cleanup_failed_init_done", **fields)


def free_duplex_state_best_effort(
    *,
    state: Any,
    drain_audio_decoding_queue: Callable[[int], object],
    destroy_audio_decoder_stream: Callable[[int], object],
    free_past_key_values: Callable[[object], object],
) -> None:
    """Run duplex cleanup steps even if one backend cleanup action fails."""
    fields = _state_cleanup_fields(state)
    log_session_leak_summary(logger, "free_duplex_state_start", **fields)
    log_session_leak_detail(logger, "free_duplex_state_resources", **fields)
    _run_cleanup_step(
        "drain_audio_decoder_stream",
        lambda: drain_audio_decoding_queue(state.audio_decoder_stream_id),
        **fields,
    )
    _run_cleanup_step(
        "destroy_audio_decoder_stream",
        lambda: destroy_audio_decoder_stream(state.audio_decoder_stream_id),
        **fields,
    )
    _run_cleanup_step(
        "free_past_key_values",
        lambda: free_past_key_values(state.past_key_values),
        **fields,
    )
    _run_cleanup_step("reset_decoding_state", state._reset, **fields)
    log_session_leak_summary(logger, "free_duplex_state_done", **fields)


def release_transient_streaming_state(cache: object | None, owner_cache: object | None) -> bool:
    """Reset a transient encoder cache when it is distinct from the session-owned cache."""
    if cache is None or cache is owner_cache:
        return False
    reset = getattr(cache, "reset", None)
    if not callable(reset):
        return False
    pool_owner = getattr(cache, "_pool_owner", None)
    pool_slot = getattr(cache, "_pool_idx", None)
    reset()
    log_session_leak_detail(
        logger,
        "warmup_transient_cache_released",
        has_audio_cache=True,
        shared_with_owner=False,
        pool_slot=pool_slot,
        pool_available_after=len(pool_owner._cache_available) if pool_owner is not None else None,
    )
    return True
