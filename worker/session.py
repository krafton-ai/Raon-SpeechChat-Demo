"""Per-session state and decode-step logic for the Raon worker.

Adapts the original session logic for multi-session use:
- No exclusive lock — multiple sessions share one model
- Per-session state: raw_input_bytes buffer, RaonDecodingState (KV cache),
  audio encoder/decoder caches
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from proto.close_reasons import INTERNAL_ERROR
from proto.config import SessionConfig
from proto.messages import Frame
from proto.prompt_map import resolve_prompt

if TYPE_CHECKING:
    import torch
    from raon_runtime.inference import RaonDecodingState
    from raon_runtime.sglang_backend import SGLangRaonModel

logger = logging.getLogger(__name__)


_FATAL_CUDA_ERROR_MARKERS = (
    "cuda error",
    "device-side assert",
    "acceleratorerror",
    "cublas",
    "cudnn",
    "illegal memory access",
    "operation not supported on global/shared address space",
)


def is_fatal_cuda_error(exc: BaseException) -> bool:
    """Return True when an exception indicates CUDA context corruption."""
    message = str(exc).lower()
    return any(marker in message for marker in _FATAL_CUDA_ERROR_MARKERS)


def _sanitize_prompt_tokens(
    tokens: list[object],
    *,
    text_vocab_size: int,
    fallback_token_id: int,
) -> tuple[list[int], int]:
    """Ensure prompt tokens stay in text-token range expected by Raon init."""
    if text_vocab_size <= 0:
        normalized = [int(tok) for tok in tokens] if tokens else [0]
        return normalized, 0

    safe_fallback = int(fallback_token_id)
    if safe_fallback < 0 or safe_fallback >= text_vocab_size:
        safe_fallback = max(0, text_vocab_size - 1)

    normalized: list[int] = []
    replaced = 0
    for raw_token in tokens:
        try:
            token = int(raw_token)
        except (TypeError, ValueError):
            token = safe_fallback
            replaced += 1
        if token < 0 or token >= text_vocab_size:
            token = safe_fallback
            replaced += 1
        normalized.append(token)

    if not normalized:
        normalized = [safe_fallback]
        replaced += 1

    return normalized, replaced


def _encode_single_token(tokenizer: object, token_text: str) -> int | None:
    """Encode a single special token string to its ID, or None if not found."""
    try:
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
    except TypeError:
        try:
            token_ids = tokenizer.encode(token_text)
        except Exception:
            return None
    except Exception:
        return None
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def _safe_int_attr(obj: object, name: str, default: int | None = None) -> int | None:
    """Read model attributes defensively when optional token IDs are absent/malformed."""
    value = getattr(obj, name, None)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_ignored_token_ids(
    model: SGLangRaonModel,
    tokenizer: object,
) -> tuple[set[int], int | None, int | None]:
    """Build the set of special token IDs that should be filtered from text output.

    Returns (ignored_ids, sil_token_id, audio_start_token_id).
    """
    epad_id = _encode_single_token(tokenizer, "<|fim_middle|>")
    if epad_id is None:
        epad_id = _safe_int_attr(model, "duplex_end_pad_token_id", 0) or 0

    dpad_id = _encode_single_token(tokenizer, "<|fim_prefix|>")
    if dpad_id is None:
        dpad_id = _safe_int_attr(model, "duplex_pad_token_id", 0) or 0

    sil_id = _encode_single_token(tokenizer, "<|audio_output_sil|>")
    if sil_id is None:
        sil_id = _safe_int_attr(model, "duplex_sil_token_id", None)

    bc_id = _encode_single_token(tokenizer, "<|audio_output_backchannel|>")
    if bc_id is None and getattr(model, "use_backchannel_token", False):
        bc_id = _safe_int_attr(model, "duplex_bc_token_id", None)

    ignored: set[int] = set()
    for attr in (
        "audio_output_token_id",
        "audio_input_token_id",
        "im_start_token_id",
        "audio_start_token_id",
    ):
        val = _safe_int_attr(model, attr, None)
        if val is not None:
            ignored.add(val)
    ignored.add(epad_id)
    ignored.add(dpad_id)
    if sil_id is not None:
        ignored.add(sil_id)
    if bc_id is not None:
        ignored.add(bc_id)

    speaker_token_id = _safe_int_attr(model, "speaker_token_id", None)
    if speaker_token_id is not None:
        ignored.add(speaker_token_id)

    eos_token_id = _safe_int_attr(model, "eos_token_id", None)
    if eos_token_id is not None:
        ignored.add(eos_token_id)

    audio_start_id = _safe_int_attr(model, "audio_start_token_id", None)

    return ignored, sil_id, audio_start_id


def _decode_text_tokens(
    token_ids: list[int],
    tokenizer: object,
    *,
    text_vocab_size: int,
    ignored_token_ids: set[int],
    sil_token_id: int | None,
    audio_start_token_id: int | None,
) -> str:
    """Decode text tokens, filtering out audio codes and special tokens.

    Ported from live_demo_server.py decode_text_tokens.
    """
    pieces: list[str] = []
    text_buffer: list[int] = []

    def flush_text_buffer() -> None:
        if not text_buffer:
            return
        pieces.append(tokenizer.decode(text_buffer, skip_special_tokens=False))
        text_buffer.clear()

    for token_id in token_ids:
        if (
            (sil_token_id is not None and token_id == sil_token_id)
            or (audio_start_token_id is not None and token_id == audio_start_token_id)
        ):
            flush_text_buffer()
            pieces.append("\n")
            continue
        if token_id in ignored_token_ids:
            flush_text_buffer()
            continue
        if token_id < text_vocab_size:
            text_buffer.append(token_id)

    flush_text_buffer()
    return "".join(pieces)


def _token_repr(tokenizer: object, token_id: int) -> str:
    """Return a readable representation for a single emitted token ID."""
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        try:
            token_text = convert(token_id)
        except Exception:
            token_text = None
        else:
            if token_text is not None:
                return str(token_text)

    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        for value in ([token_id], token_id):
            try:
                decoded = decode(value, skip_special_tokens=False)
            except TypeError:
                try:
                    decoded = decode(value)
                except Exception:
                    continue
            except Exception:
                continue
            if isinstance(decoded, str) and decoded:
                return decoded

    return f"<id:{token_id}>"


def _format_seq_delta(
    token_ids: list[int],
    tokenizer: object,
    *,
    frame_index: int,
    sequence_len: int,
    text_delta: str,
) -> str:
    """Render a stable one-line trace for newly emitted model tokens."""
    token_texts = [_token_repr(tokenizer, int(token_id)) for token_id in token_ids]
    return (
        f"frame={frame_index} seq_len={sequence_len} token_count={len(token_ids)} "
        f"ids={list(map(int, token_ids))} tokens={token_texts!r} text={text_delta!r}"
    )


def _is_recoverable_decode_error(exc: BaseException) -> bool:
    """Return True for non-fatal decode-step failures that can emit silence."""
    if isinstance(exc, AssertionError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return "sil-no-audio" in message and "assert" in message
    return False


@dataclass
class SessionState:
    """Mutable per-session state managed by RaonWorkerSession."""

    session_id: str
    config: SessionConfig

    # Raw PCM byte buffer (float32 LE)
    raw_input_bytes: bytearray = field(default_factory=bytearray)

    # Model decode state (KV cache, sequences, audio codes, etc.)
    decoding_state: RaonDecodingState | None = None
    prompt_token_ids: list[int] = field(default_factory=list)
    speaker_embeds: object | None = None

    # Track sequence length for text delta extraction
    last_sequence_len: int = 0

    # Text token filtering (populated during init)
    ignored_text_token_ids: set[int] = field(default_factory=set)
    sil_token_id: int | None = None
    audio_start_token_id: int | None = None

    # Timing
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    # Stats
    frames_in: int = 0
    frames_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    dropped_input_frames: int = 0
    dropped_input_bytes: int = 0
    backlog_soft_events: int = 0
    backlog_hard_events: int = 0
    max_time_behind_seconds: float = 0.0
    decode_errors: int = 0
    consecutive_decode_errors: int = 0
    close_requested_reason: str | None = None
    close_reason: str | None = None

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    def touch(self) -> None:
        self.last_activity = time.time()


class RaonWorkerSession:
    """Manages a single Raon session's lifecycle against a shared model.

    This is a stateless helper — all mutable state lives in SessionState.
    The model reference is passed in per-call so multiple sessions can share it.
    """

    @staticmethod
    def init(
        model: SGLangRaonModel,
        session_config: SessionConfig,
        tokenizer: object,
        speaker_embeds: object | None = None,
    ) -> SessionState:
        """Initialize a new session: prepare prompt state before decode-state prefill."""
        from transformers import Qwen2TokenizerFast

        tok = tokenizer
        assert isinstance(tok, Qwen2TokenizerFast)

        state = SessionState(
            session_id=session_config.session_id,
            config=session_config,
        )

        prompt_text = resolve_prompt(
            session_config.prompt,
            session_config.prompt_role,
            prompt_language=session_config.prompt_language,
            system_prompt_style=session_config.system_prompt_style,
            system_prompt_persona=session_config.system_prompt_persona,
            system_prompt_context=session_config.system_prompt_context,
            custom_system_prompt=session_config.custom_system_prompt,
        )
        if prompt_text != session_config.prompt:
            logger.info("resolved prompt key=%s → %r", session_config.prompt, prompt_text)

        # Build prompt tokens
        messages = [{"role": session_config.prompt_role, "content": prompt_text}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if isinstance(text, list):
            tokens = text
        else:
            tokens = tok.encode(text)

        text_vocab_size = int(getattr(model, "text_vocab_size", None) or getattr(model, "vocab_size", 0) or 0)
        fallback_token_id = int(
            getattr(
                model,
                "duplex_end_pad_token_id",
                max(0, text_vocab_size - 1),
            )
        )
        tokens, replaced_tokens = _sanitize_prompt_tokens(
            list(tokens),
            text_vocab_size=text_vocab_size,
            fallback_token_id=fallback_token_id,
        )
        if replaced_tokens:
            logger.warning(
                "prompt_token_sanitized session_id=%s replaced=%d total=%d text_vocab_size=%d fallback=%d",
                session_config.session_id,
                replaced_tokens,
                len(tokens),
                text_vocab_size,
                fallback_token_id,
            )

        state.prompt_token_ids = tokens
        state.speaker_embeds = speaker_embeds

        if speaker_embeds is not None:
            logger.info(
                "speaker_embeds_debug session_id=%s shape=%s dtype=%s device=%s "
                "speaker_token_id=%s model_hidden=%s",
                session_config.session_id,
                list(speaker_embeds.shape),
                speaker_embeds.dtype,
                speaker_embeds.device,
                getattr(model.get_model(), "speaker_token_id", "MISSING"),
                getattr(model.get_model(), "config", {}).hidden_size
                if hasattr(getattr(model.get_model(), "config", None), "hidden_size")
                else "unknown",
            )
        # Repetition penalty is disabled: causes CUDA scatter index out of bounds
        # when force_audio_output slices logits to codebook-only but sequences
        # still contain full-range text token IDs. See inference.py:1546-1554.
        if session_config.sampling.repetition_penalty > 1.0:
            logger.warning(
                "repetition_penalty=%.2f ignored (known CUDA crash bug) session_id=%s",
                session_config.sampling.repetition_penalty,
                session_config.session_id,
            )
        if False:  # disabled until duplex runtime fix
            try:
                from transformers import RepetitionPenaltyLogitsProcessor
                state.decoding_state.logits_processor.append(
                    RepetitionPenaltyLogitsProcessor(penalty=sampling.repetition_penalty)
                )
                logger.info("repetition_penalty=%.2f added", sampling.repetition_penalty)
            except Exception as exc:
                logger.warning(
                    "repetition_penalty unavailable session_id=%s reason=%s",
                    session_config.session_id,
                    exc,
                )

        # Build text-token filter sets once per session
        ignored, sil_id, audio_start_id = _resolve_ignored_token_ids(model, tok)
        state.ignored_text_token_ids = ignored
        state.sil_token_id = sil_id
        state.audio_start_token_id = audio_start_id

        logger.info(
            "session_init session_id=%s seq_len=%d speak_first=%s prompt_style=%s speaker_mode=%s speaker_embeds=%s ignored_tokens=%d",
            session_config.session_id,
            len(state.prompt_token_ids),
            session_config.speak_first,
            session_config.system_prompt_style,
            session_config.speaker_mode,
            speaker_embeds is not None,
            len(state.ignored_text_token_ids),
        )
        return state

    @staticmethod
    def _ensure_decoding_state(
        model: SGLangRaonModel,
        state: SessionState,
    ) -> None:
        """Allocate the per-session decode state when the session becomes runnable."""
        import torch

        if state.decoding_state is not None:
            return

        t0 = time.time()
        prompt_tensor = torch.tensor([state.prompt_token_ids], device=model.device, dtype=torch.long)
        attention_mask = torch.ones_like(prompt_tensor)
        sampling = state.config.sampling
        init_kwargs = dict(
            sequences=prompt_tensor,
            attention_mask=attention_mask,
            do_sample=sampling.do_sample,
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            eos_penalty=sampling.eos_penalty,
            code_temperature=sampling.code_temperature,
            code_top_k=sampling.code_top_k,
            sil_penalty=sampling.sil_penalty,
            bc_penalty=sampling.bc_penalty,
            audio_encoder_chunk_frames=sampling.audio_encoder_chunk_frames,
            speak_first=state.config.speak_first,
            speaker_embeds=state.speaker_embeds,
        )
        sig = inspect.signature(model.init_duplex_decoding_state)
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )
        if not accepts_var_kwargs:
            accepted = set(sig.parameters.keys())
            init_kwargs = {k: v for k, v in init_kwargs.items() if k in accepted}
        with torch.inference_mode():
            state.decoding_state = model.init_duplex_decoding_state(**init_kwargs)
        state.last_sequence_len = int(state.decoding_state.sequences.shape[1])
        logger.info(
            "session_prefill_done session_id=%s seq_len=%d speak_first=%s elapsed_ms=%.1f",
            state.session_id,
            state.last_sequence_len,
            state.config.speak_first,
            (time.time() - t0) * 1000.0,
        )

    @dataclass(slots=True)
    class FeedAudioResult:
        dropped_bytes: int = 0
        dropped_frames: int = 0
        backlog_bytes: int = 0
        backlog_frames: float = 0.0
        time_behind_seconds: float = 0.0
        soft_backlog: bool = False
        hard_backlog: bool = False
        hard_action: str = "none"  # none | degrade | close

    @staticmethod
    def _drop_oldest_aligned(
        state: SessionState,
        target_drop_bytes: int,
    ) -> tuple[int, int]:
        """Drop oldest input bytes aligned to frame boundary."""
        if target_drop_bytes <= 0 or not state.raw_input_bytes:
            return 0, 0

        frame_bytes = max(1, state.config.audio.frame_size * 4)
        drop = min(target_drop_bytes, len(state.raw_input_bytes))
        drop -= drop % frame_bytes
        if drop <= 0:
            return 0, 0

        del state.raw_input_bytes[:drop]
        return drop, drop // frame_bytes

    @staticmethod
    def feed_audio(state: SessionState, pcm_bytes: bytes) -> FeedAudioResult:
        """Buffer incoming PCM bytes and apply backlog/overload policy."""
        result = RaonWorkerSession.FeedAudioResult()

        state.raw_input_bytes.extend(pcm_bytes)
        state.bytes_in += len(pcm_bytes)
        state.touch()

        audio_cfg = state.config.audio
        frame_bytes = max(1, audio_cfg.frame_size * 4)

        max_bytes = audio_cfg.max_buffer_bytes
        if max_bytes > 0 and len(state.raw_input_bytes) > max_bytes:
            dropped, dropped_frames = RaonWorkerSession._drop_oldest_aligned(
                state=state,
                target_drop_bytes=len(state.raw_input_bytes) - max_bytes,
            )
            result.dropped_bytes += dropped
            result.dropped_frames += dropped_frames

        backlog_bytes = len(state.raw_input_bytes)
        bytes_per_second = max(1, int(audio_cfg.sampling_rate * 4))
        backlog_seconds = backlog_bytes / bytes_per_second
        soft = audio_cfg.soft_backlog_seconds > 0 and backlog_seconds > audio_cfg.soft_backlog_seconds
        hard = audio_cfg.hard_backlog_seconds > 0 and backlog_seconds > audio_cfg.hard_backlog_seconds

        result.soft_backlog = soft
        result.hard_backlog = hard

        if hard:
            action = (audio_cfg.hard_backlog_action or "").strip().lower()
            if action not in {"degrade", "close"}:
                action = "degrade"
            result.hard_action = action
            if action == "degrade":
                target_seconds = max(0.0, audio_cfg.degrade_target_seconds)
                target_bytes = int(target_seconds * bytes_per_second)
                if backlog_bytes > target_bytes:
                    dropped, dropped_frames = RaonWorkerSession._drop_oldest_aligned(
                        state=state,
                        target_drop_bytes=backlog_bytes - target_bytes,
                    )
                    result.dropped_bytes += dropped
                    result.dropped_frames += dropped_frames

        result.backlog_bytes = len(state.raw_input_bytes)
        result.backlog_frames = result.backlog_bytes / frame_bytes
        result.time_behind_seconds = result.backlog_bytes / bytes_per_second
        return result

    @staticmethod
    def step(
        model: SGLangRaonModel,
        state: SessionState,
        tokenizer: object,
    ) -> list[Frame]:
        """Run one decode step. Returns output frames (audio, text, trace)."""
        import torch
        from transformers import Qwen2TokenizerFast

        tok = tokenizer
        assert isinstance(tok, Qwen2TokenizerFast)

        audio_cfg = state.config.audio
        frame_size = audio_cfg.frame_size
        needed_bytes = frame_size * 4  # float32

        if len(state.raw_input_bytes) < needed_bytes:
            return []

        if state.decoding_state is None:
            RaonWorkerSession._ensure_decoding_state(model=model, state=state)

        # Read one frame of PCM from buffer
        chunk_bytes = bytes(state.raw_input_bytes[:needed_bytes])
        del state.raw_input_bytes[:needed_bytes]
        pcm_view = np.frombuffer(chunk_bytes, dtype=np.float32)
        pcm = pcm_view.copy() if hasattr(pcm_view, "copy") else np.array(pcm_view, dtype=np.float32)
        chunk_length = len(pcm)

        # Pad if short
        if chunk_length < frame_size:
            pcm = np.pad(pcm, (0, frame_size - chunk_length))

        state.frames_in += 1

        # Apply input processing
        if audio_cfg.input_gain != 1.0:
            pcm = pcm * audio_cfg.input_gain
        if audio_cfg.input_clip > 0:
            pcm = np.clip(pcm, -audio_cfg.input_clip, audio_cfg.input_clip)

        # Silence gate
        if audio_cfg.silence_rms_threshold > 0:
            rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
            if rms < audio_cfg.silence_rms_threshold:
                pcm = np.zeros_like(pcm)

        # Run model decode step
        t_tensor = time.time()
        chunk_tensor = torch.from_numpy(pcm).to(
            device=model.device, dtype=model.dtype
        )[None, :]
        t_tensor_done = time.time()

        frames: list[Frame] = []
        sid = state.session_id

        try:
            t_decode = time.time()
            with torch.inference_mode():
                state.decoding_state, audio_output = model.duplex_decoding_step(
                    state=state.decoding_state,
                    audio_input=chunk_tensor,
                )
            t_decode_done = time.time()
            state.consecutive_decode_errors = 0
        except Exception as exc:
            if _is_recoverable_decode_error(exc):
                state.decode_errors += 1
                state.consecutive_decode_errors += 1
                log_fn = logger.error if state.consecutive_decode_errors >= 3 else logger.warning
                log_fn(
                    "decode_step_recoverable_error session_id=%s frame=%d consecutive=%d total=%d: %s",
                    state.session_id, state.frames_in,
                    state.consecutive_decode_errors, state.decode_errors, exc,
                )
                if state.consecutive_decode_errors >= 3:
                    state.close_requested_reason = INTERNAL_ERROR
                silence = np.zeros(audio_cfg.frame_size, dtype=np.float32)
                frames.append(Frame.audio(silence, session_id=sid))
                if state.consecutive_decode_errors == 1:
                    frames.append(Frame.error(f"decode error: {exc}", session_id=sid))
                state.frames_out += 1
                state.bytes_out += len(silence) * 4
                state.touch()
                return frames

            if is_fatal_cuda_error(exc):
                logger.error(
                    "decode_step_fatal_cuda session_id=%s frame=%d: %s",
                    state.session_id,
                    state.frames_in,
                    exc,
                )
            raise

        # Extract text delta
        sequences = state.decoding_state.sequences
        new_token_slice = sequences[0, state.last_sequence_len:]
        if hasattr(new_token_slice, "tolist"):
            new_tokens = new_token_slice.tolist()
        else:
            new_tokens = [int(token_id) for token_id in new_token_slice]
        state.last_sequence_len = int(sequences.shape[1])
        text_delta = _decode_text_tokens(
            new_tokens,
            tok,
            text_vocab_size=int(getattr(model, "text_vocab_size", None) or getattr(model, "vocab_size", 0)),
            ignored_token_ids=state.ignored_text_token_ids,
            sil_token_id=state.sil_token_id,
            audio_start_token_id=state.audio_start_token_id,
        )
        if new_tokens:
            frames.append(
                Frame.seq_delta(
                    _format_seq_delta(
                        new_tokens,
                        tok,
                        frame_index=state.frames_in,
                        sequence_len=state.last_sequence_len,
                        text_delta=text_delta,
                    ),
                    session_id=sid,
                )
            )
        if text_delta:
            frames.append(Frame.text(text_delta, session_id=sid))

        # Process output audio
        output_np = audio_output[0].detach().float().cpu().numpy().astype(np.float32)

        if audio_cfg.output_gain != 1.0:
            output_np = output_np * audio_cfg.output_gain
        if audio_cfg.output_clip > 0:
            output_np = np.clip(output_np, -audio_cfg.output_clip, audio_cfg.output_clip)

        frames.append(Frame.audio(output_np, session_id=sid))

        if state.frames_out == 0:
            logger.info(
                "session_first_output session_id=%s frames_in=%d frames_out=%d decode_ms=%.1f backlog_frames=%.1f",
                sid,
                state.frames_in,
                state.frames_out + 1,
                (t_decode_done - t_decode) * 1000.0,
                len(state.raw_input_bytes) / max(1, needed_bytes),
            )

        state.frames_out += 1
        state.bytes_out += len(output_np) * 4
        state.touch()

        # Profiling: log every 50 frames
        if state.frames_out % 50 == 1:
            tensor_ms = (t_tensor_done - t_tensor) * 1000
            decode_ms = (t_decode_done - t_decode) * 1000
            total_ms = (time.time() - t_tensor) * 1000
            logger.info(
                "PROFILE session=%s frame=%d tensor=%.1fms decode=%.1fms total=%.1fms",
                sid, state.frames_out, tensor_ms, decode_ms, total_ms,
            )

        return frames

    @staticmethod
    def close(
        model: SGLangRaonModel,
        state: SessionState,
        *,
        skip_gpu_cleanup: bool = False,
    ) -> None:
        """Free KV cache and clean up session resources."""
        if state.decoding_state is not None:
            if not skip_gpu_cleanup:
                try:
                    import torch
                    with torch.inference_mode():
                        model.free_duplex_decoding_state(state.decoding_state)
                except Exception as exc:
                    if is_fatal_cuda_error(exc):
                        logger.warning(
                            "session_close_skipped_fatal_cuda session_id=%s: %s",
                            state.session_id,
                            exc,
                        )
                    else:
                        logger.exception(
                            "session_close_error session_id=%s", state.session_id
                        )
            state.decoding_state = None
        state.raw_input_bytes.clear()
        logger.info(
            "session_closed session_id=%s frames_in=%d frames_out=%d",
            state.session_id,
            state.frames_in,
            state.frames_out,
        )
