"""Local system-prompt map for duplex worker session initialization."""

from __future__ import annotations

from typing import Literal

PromptLanguage = Literal["eng", "kor"]
SystemPromptStyle = Literal["generic", "raon", "raon_persona", "raon_persona_context", "custom"]

CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE = {
    "eng:full_duplex:speak-first": "You are engaging in real-time conversation.",
    "eng:full_duplex:listen-first": "You are engaging in real-time conversation.",
    "eng:duplex_instruct:speak-first": "You are engaging in real-time conversation.",
    "eng:duplex_instruct:listen-first": "You are engaging in real-time conversation.",
    "kor:full_duplex:speak-first": "당신은 실시간 대화에 참여하고 있습니다.",
    "kor:full_duplex:listen-first": "당신은 실시간 대화에 참여하고 있습니다.",
    "kor:duplex_instruct:speak-first": "당신은 실시간 대화에 참여하고 있습니다.",
    "kor:duplex_instruct:listen-first": "당신은 실시간 대화에 참여하고 있습니다.",
}

SYSTEM_PROMPT_BASE_BY_LANGUAGE: dict[PromptLanguage, str] = {
    "eng": "You are engaging in real-time conversation.",
    "kor": "당신은 실시간 대화에 참여하고 있습니다.",
}

SYSTEM_PROMPT_STYLES: tuple[SystemPromptStyle, ...] = (
    "generic",
    "raon",
    "raon_persona",
    "raon_persona_context",
    "custom",
)


def get_duplex_system_message_key(
    language: PromptLanguage,
    channel: Literal["full_duplex", "duplex_instruct"],
    speak_first: bool,
) -> str:
    speak_mode = "speak-first" if speak_first else "listen-first"
    return f"{language}:{channel}:{speak_mode}"


def _parse_prompt_key(prompt_text: str) -> tuple[str, str, str] | None:
    """Parse canonical/shorthand duplex prompt keys.

    Returns ``(language, channel, speak_mode)`` for recognized duplex prompt
    keys, or ``None`` for free-form prompt text.
    """
    if prompt_text in CHANNEL_DUPLEX_TO_SYSTEM_MESSAGE:
        language, channel, speak_mode = prompt_text.split(":", 2)
        return language, channel, speak_mode

    parts = [part.strip() for part in prompt_text.split(":") if part.strip()]
    if len(parts) == 2:
        language = "eng"
        channel, speak_mode = parts
    elif len(parts) == 3:
        language, channel, speak_mode = parts
    else:
        return None

    if channel not in {"full_duplex", "duplex_instruct"}:
        return None
    if speak_mode not in {"speak-first", "listen-first"}:
        return None

    return language, channel, speak_mode


def _normalize_language(language: str | None, default: PromptLanguage = "eng") -> PromptLanguage:
    if language == "kor":
        return "kor"
    if language == "eng":
        return "eng"
    return default


def _normalize_style(style: str | None) -> SystemPromptStyle:
    if style in SYSTEM_PROMPT_STYLES:
        return style
    return "generic"


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _ensure_terminal_punctuation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if stripped[-1] in ".!?":
        return stripped
    return stripped + "."


def resolve_prompt_language(
    prompt_text: str,
    prompt_role: str,
    *,
    default: PromptLanguage = "eng",
) -> PromptLanguage:
    """Resolve the language embedded in a canonical/shorthand duplex prompt key."""
    if prompt_role != "system":
        return default

    parsed = _parse_prompt_key(prompt_text)
    if parsed is None:
        return default

    return _normalize_language(parsed[0], default=default)


def build_system_prompt(
    *,
    language: PromptLanguage = "eng",
    system_prompt_style: str | None = None,
    system_prompt_persona: str | None = None,
    system_prompt_context: str | None = None,
    custom_system_prompt: str | None = None,
) -> str:
    """Build a structured system prompt for the V2 duplex model."""
    language = _normalize_language(language)
    style = _normalize_style(system_prompt_style)
    base = SYSTEM_PROMPT_BASE_BY_LANGUAGE[language]
    persona = _clean_optional_text(system_prompt_persona)
    context = _clean_optional_text(system_prompt_context)
    custom = _clean_optional_text(custom_system_prompt)

    if style == "custom":
        return _ensure_terminal_punctuation(custom) if custom else base

    if style == "generic":
        return base

    if style == "raon":
        if language == "kor":
            return f"{base} 당신은 도움이 되는 어시스턴트입니다."
        return f"{base} You are a helpful assistant."

    if not persona:
        return build_system_prompt(
            language=language,
            system_prompt_style="raon",
        )

    if language == "kor":
        prompt = f"{base} 당신은 도움이 되는 어시스턴트이고, {persona}"
    else:
        prompt = f"{base} You are a helpful assistant, {persona}."

    prompt = _ensure_terminal_punctuation(prompt)

    if style == "raon_persona_context" and context:
        return f"{prompt} {_ensure_terminal_punctuation(context)}"

    return prompt


def resolve_prompt(
    prompt_text: str,
    prompt_role: str,
    *,
    prompt_language: str | None = None,
    system_prompt_style: str | None = None,
    system_prompt_persona: str | None = None,
    system_prompt_context: str | None = None,
    custom_system_prompt: str | None = None,
) -> str:
    """Resolve canonical or shorthand system-prompt keys to prompt text."""
    if prompt_role != "system":
        return prompt_text

    parsed = _parse_prompt_key(prompt_text)
    if parsed is None:
        return prompt_text

    language = _normalize_language(prompt_language, default=_normalize_language(parsed[0]))
    return build_system_prompt(
        language=language,
        system_prompt_style=system_prompt_style,
        system_prompt_persona=system_prompt_persona,
        system_prompt_context=system_prompt_context,
        custom_system_prompt=custom_system_prompt,
    )


def resolve_speak_first(prompt_text: str, prompt_role: str) -> bool:
    """Resolve the duplex start-turn mode from a canonical/shorthand prompt key.

    The new duplex model no longer encodes speak/listen mode in the system
    message text, so runtime callers must carry this bit separately.
    """
    if prompt_role != "system":
        return False

    parsed = _parse_prompt_key(prompt_text)
    if parsed is None:
        return False

    return parsed[2] == "speak-first"
