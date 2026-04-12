"""Helpers for classifying model-level talker state warnings."""

from __future__ import annotations


def get_talker_state_warning(
    *,
    has_talker_cache: bool,
    has_talker_mask: bool,
    is_reuse_init: bool,
) -> str | None:
    """Return a warning only for dirty fresh-session initialization."""
    if is_reuse_init:
        return None

    dirty_parts: list[str] = []
    if has_talker_cache:
        dirty_parts.append("cache")
    if has_talker_mask:
        dirty_parts.append("mask")

    if not dirty_parts:
        return None

    return (
        "fresh init_duplex_decoding_state found unexpected model-level talker state "
        f"({'/'.join(dirty_parts)}); previous cleanup may have been skipped"
    )
