"""Lightweight public API for the vendored Raon runtime."""

from __future__ import annotations

from importlib import import_module

_EXPORT_TO_MODULE = {
    "DuplexDecodingState": ".inference",
    "DuplexGenerateResult": ".inference",
    "GenerateOutput": ".inference",
    "QwenDuplexInferenceModel": ".inference",
    "RaonDecodingState": ".inference",
    "RaonGenerateResult": ".inference",
    "RaonInferenceModel": ".inference",
    "extract_predicted_text": ".inference",
    "QwenDuplexModel": ".model",
    "QwenDuplexModelConfig": ".model",
    "QwenDuplexModelOutput": ".model",
    "RaonConfig": ".model",
    "RaonModel": ".model",
    "RaonModelOutput": ".model",
    "DuplexProcessor": ".processor",
    "RaonProcessor": ".processor",
    "SGLangRaonModel": ".sglang_backend",
    "RaonMachineState": ".state_machine",
    "RaonPhase": ".state_machine",
    "RaonStateConfig": ".state_machine",
    "RaonStateManager": ".state_machine",
}


def __getattr__(name: str) -> object:
    if name not in _EXPORT_TO_MODULE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_TO_MODULE[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "DuplexDecodingState",
    "DuplexGenerateResult",
    "DuplexProcessor",
    "GenerateOutput",
    "QwenDuplexInferenceModel",
    "QwenDuplexModel",
    "QwenDuplexModelConfig",
    "QwenDuplexModelOutput",
    "RaonConfig",
    "RaonDecodingState",
    "RaonGenerateResult",
    "RaonInferenceModel",
    "RaonMachineState",
    "RaonModel",
    "RaonModelOutput",
    "RaonPhase",
    "RaonProcessor",
    "RaonStateConfig",
    "RaonStateManager",
    "SGLangRaonModel",
    "extract_predicted_text",
]
