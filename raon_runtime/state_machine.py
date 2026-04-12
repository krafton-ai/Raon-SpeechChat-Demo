"""Unified Mealy state machine for duplex sequence construction and inference."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from .special_tokens import (
    AUDIO_END,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_OUTPUT_SIL,
    AUDIO_START,
    IM_END,
    IM_START,
    LOSS_IGNORE_INDEX,
    SPEAKER_EMBEDDING_PLACEHOLDER,
)

if TYPE_CHECKING:
    from .inference import RaonInferenceModel
    from .processor import RaonProcessor


class RaonPhase(enum.Enum):
    """Two-phase duplex state: silence or active speech."""

    SIL = "SIL"
    SPEECH = "SPEECH"


@dataclass
class RaonMachineState:
    """Current Mealy machine state with last emitted frame tokens.

    Tracks the current phase (SIL or SPEECH) and the token sequence
    emitted in the most recent frame, enabling the state manager to
    determine valid transitions and input token counts.
    """

    phase: RaonPhase
    last_frame_tokens: list[int]

    @property
    def num_input_tokens(self) -> int:
        """Return the number of tokens in the last emitted frame."""
        return len(self.last_frame_tokens)

    @property
    def emitted_audio(self) -> bool:
        """Return True if the last frame included an audio output token."""
        return AUDIO_OUTPUT_PLACEHOLDER.id in self.last_frame_tokens or AUDIO_START.id in self.last_frame_tokens


@dataclass(frozen=True)
class RaonStateConfig:
    """Immutable configuration for RaonStateManager."""

    use_duplex_end_pad: bool = False
    use_sil_token: bool = False
    no_audio_in_sil: bool = False
    sequence_mode: Literal["tua", "uta"] | None = None
    duplex_pad_token_id: int = AUDIO_OUTPUT_PAD.id
    duplex_end_pad_token_id: int = AUDIO_OUTPUT_END_PAD.id
    duplex_sil_token_id: int = AUDIO_OUTPUT_SIL.id
    use_backchannel_token: bool = False
    duplex_bc_token_id: int = AUDIO_OUTPUT_BC.id

    @property
    def condensed_silence(self) -> bool:
        """Return False because condensed silence mode has been removed."""
        return False

    @property
    def effective_sequence_mode(self) -> Literal["tua", "uta"]:
        """Return the effective sequence mode (defaulting to 'tua')."""
        return self.sequence_mode or "tua"


# Structural tokens that must never be sampled as text predictions.
_BLOCKED_STRUCTURAL: frozenset[int] = frozenset(
    {
        AUDIO_INPUT_PLACEHOLDER.id,
        AUDIO_OUTPUT_PLACEHOLDER.id,
        AUDIO_START.id,
        AUDIO_END.id,
        IM_START.id,
        IM_END.id,
        SPEAKER_EMBEDDING_PLACEHOLDER.id,
    }
)


class RaonStateManager:
    """Unified Mealy state machine for duplex sequence construction and inference.

    Encapsulates the sequence-construction logic shared by runtime decoding
    and prompt construction. Provides methods for preamble generation,
    state transitions, logit masking, and token emission.
    """

    def __init__(self, config: RaonStateConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_processor(cls, processor: RaonProcessor) -> RaonStateManager:
        """Create a RaonStateManager from a RaonProcessor's configuration.

        Args:
            processor: Configured RaonProcessor instance.

        Returns:
            RaonStateManager with matching configuration.
        """
        config = RaonStateConfig(
            use_duplex_end_pad=getattr(processor, "use_duplex_end_pad", False),
            use_sil_token=getattr(processor, "use_sil_token", False),
            no_audio_in_sil=getattr(processor, "no_audio_in_sil", False),
            sequence_mode=_resolve_sequence_mode_from_processor(processor),
            duplex_pad_token_id=getattr(processor, "duplex_pad_token_id", AUDIO_OUTPUT_PAD.id),
            duplex_end_pad_token_id=getattr(processor, "duplex_end_pad_token_id", AUDIO_OUTPUT_END_PAD.id),
            duplex_sil_token_id=getattr(processor, "duplex_sil_token_id", AUDIO_OUTPUT_SIL.id),
            use_backchannel_token=getattr(processor, "use_backchannel_token", False),
            duplex_bc_token_id=getattr(processor, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id),
        )
        return cls(config)

    @classmethod
    def from_inference_model(cls, model: RaonInferenceModel) -> RaonStateManager:
        """Create a RaonStateManager from an inference model's configuration.

        Args:
            model: RaonInferenceModel instance.

        Returns:
            RaonStateManager with matching configuration.
        """
        sequence_mode = getattr(model, "sequence_mode", None)
        if sequence_mode is not None and sequence_mode not in ("tua", "uta"):
            raise ValueError(f"Unsupported sequence_mode '{sequence_mode}'.")

        config = RaonStateConfig(
            use_duplex_end_pad=getattr(model, "use_duplex_end_pad", False),
            use_sil_token=getattr(model, "use_sil_token", False),
            no_audio_in_sil=getattr(model, "no_audio_in_sil", False),
            sequence_mode=sequence_mode,
            duplex_pad_token_id=AUDIO_OUTPUT_PAD.id,
            duplex_end_pad_token_id=AUDIO_OUTPUT_END_PAD.id,
            duplex_sil_token_id=getattr(model, "duplex_sil_token_id", AUDIO_OUTPUT_SIL.id),
            use_backchannel_token=getattr(model, "use_backchannel_token", False),
            duplex_bc_token_id=getattr(model, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id),
        )
        return cls(config)

    # ------------------------------------------------------------------
    # Preamble and initial state
    # ------------------------------------------------------------------

    def preamble_tokens(self, speak_first: bool) -> list[int]:
        """Return the token preamble for duplex decoding initialization.

        Args:
            speak_first: Whether the assistant speaks first.

        Returns:
            List of token IDs for the preamble sequence.
        """
        a_s = AUDIO_START.id
        _ = speak_first
        # Unified preamble: [IM_START, AUDIO_START]. Speak-first now forces the
        # first [U] prediction to EPAD at runtime instead of changing preamble.
        return [IM_START.id, a_s]

    def initial_forced_prediction_id(self, speak_first: bool) -> int | None:
        """Return the token ID that must be forced for the first [U] prediction.

        The v2 duplex model no longer encodes speak/listen mode in the system
        prompt. Runtime initialization must therefore force the first text-side
        prediction explicitly:

        - speak-first: force EPAD to enter onset/speech mode
        - listen-first: force SIL to remain in silence mode

        Returns:
            Token ID to force, or None when no explicit first-step override is
            configured.
        """
        cfg = self._config
        if speak_first:
            if cfg.use_duplex_end_pad:
                return cfg.duplex_end_pad_token_id
            return None

        if cfg.use_sil_token:
            return cfg.duplex_sil_token_id
        return None

    def initial_state_from_sample(
        self,
        speak_first: bool,
        emitted_tokens: list[int],
    ) -> RaonMachineState:
        """Determine initial machine state from the first forward pass output.

        After running the initial forward pass and generating the first frame,
        this method determines whether we are in SPEECH or SIL phase based on
        what tokens were emitted.

        Args:
            speak_first: Whether the assistant speaks first.
            emitted_tokens: Token IDs emitted by the first frame.

        Returns:
            RaonMachineState reflecting the initial phase.
        """
        epad_id = self._config.duplex_end_pad_token_id
        a_s = AUDIO_START.id
        aop = AUDIO_OUTPUT_PLACEHOLDER.id

        if speak_first:
            return RaonMachineState(phase=RaonPhase.SPEECH, last_frame_tokens=emitted_tokens)

        # Check if the first frame transitioned to speech (EPAD onset).
        has_audio = aop in emitted_tokens or a_s in emitted_tokens
        has_epad = epad_id in emitted_tokens
        if has_epad and has_audio:
            return RaonMachineState(phase=RaonPhase.SPEECH, last_frame_tokens=emitted_tokens)

        # Otherwise, determine phase from whether audio was emitted.
        if has_audio:
            return RaonMachineState(phase=RaonPhase.SPEECH, last_frame_tokens=emitted_tokens)

        return RaonMachineState(phase=RaonPhase.SIL, last_frame_tokens=emitted_tokens)

    # ------------------------------------------------------------------
    # Mealy transition
    # ------------------------------------------------------------------

    def transition(
        self,
        state: RaonMachineState,
        predicted_id: int,
    ) -> tuple[RaonMachineState, list[int], bool]:
        """Compute the next state and emitted frame tokens from a text prediction.

        Implements the Mealy transition table for the duplex state machine.

        Args:
            state: Current machine state.
            predicted_id: Predicted text token ID from the language model.

        Returns:
            Tuple of (new_state, frame_tokens, emitted_audio).
        """
        cfg = self._config
        aip = AUDIO_INPUT_PLACEHOLDER.id
        aop = AUDIO_OUTPUT_PLACEHOLDER.id
        sil_id = cfg.duplex_sil_token_id
        epad_id = cfg.duplex_end_pad_token_id
        pad_id = cfg.duplex_pad_token_id
        bc_id = cfg.duplex_bc_token_id
        is_uta = cfg.effective_sequence_mode == "uta"

        is_sil_prediction = predicted_id == sil_id

        if state.phase == RaonPhase.SIL:
            if is_sil_prediction:
                # SIL -> SIL: stay in silence.
                tokens = [aip, aop]
                return RaonMachineState(RaonPhase.SIL, tokens), tokens, True

            if cfg.use_duplex_end_pad and predicted_id == epad_id:
                # SIL -> SPEECH via EPAD onset.
                tokens = [aip, epad_id, aop] if is_uta else [epad_id, aip, aop]
                return RaonMachineState(RaonPhase.SPEECH, tokens), tokens, True

            if cfg.use_backchannel_token and predicted_id == bc_id:
                # SIL -> SPEECH via BC onset.
                tokens = [aip, bc_id, aop] if is_uta else [bc_id, aip, aop]
                return RaonMachineState(RaonPhase.SPEECH, tokens), tokens, True

            # SIL -> SPEECH via direct text.
            if is_uta:
                tokens = [aip, predicted_id, aop]
            else:
                tokens = [predicted_id, aip, aop]
            return RaonMachineState(RaonPhase.SPEECH, tokens), tokens, True

        # state.phase == RaonPhase.SPEECH
        if is_sil_prediction:
            # SPEECH -> SIL.
            tokens = [aip, aop]
            return RaonMachineState(RaonPhase.SIL, tokens), tokens, True

        if predicted_id == pad_id:
            # SPEECH -> SPEECH (PAD).
            tokens = [aip, aop]
            return RaonMachineState(RaonPhase.SPEECH, tokens), tokens, True

        if predicted_id == epad_id:
            # SPEECH -> SPEECH (EPAD).
            if is_uta:
                tokens = [aip, epad_id, aop]
            else:
                tokens = [epad_id, aip, aop]
            return RaonMachineState(RaonPhase.SPEECH, tokens), tokens, True

        # SPEECH -> SPEECH (text token).
        if is_uta:
            tokens = [aip, predicted_id, aop]
        else:
            tokens = [predicted_id, aip, aop]
        return RaonMachineState(RaonPhase.SPEECH, tokens), tokens, True

    # ------------------------------------------------------------------
    # Logit masking
    # ------------------------------------------------------------------

    def apply_logit_mask(
        self,
        user_logits: torch.Tensor,
        state: RaonMachineState,
        vocab_size: int,
    ) -> torch.Tensor:
        """Mask logits to enforce valid state-machine transitions.

        Args:
            user_logits: Text logits from the language model.
                Shape: [1, 1, vocab_size]. Dtype: float.
            state: Current machine state.
            vocab_size: Text vocabulary size for masking ranges.

        Returns:
            Masked logits with invalid tokens set to -inf.
                Shape: [1, 1, vocab_size]. Dtype: float.
        """
        cfg = self._config
        sil_id = cfg.duplex_sil_token_id
        epad_id = cfg.duplex_end_pad_token_id
        pad_id = cfg.duplex_pad_token_id
        bc_id = cfg.duplex_bc_token_id
        onset_ids = {epad_id}
        if cfg.use_backchannel_token:
            onset_ids.add(bc_id)

        user_logits = user_logits.clone()
        mask = torch.full_like(user_logits, float("-inf"))
        max_token_id = mask.shape[-1]

        def _allow_token(token_id: int) -> None:
            if token_id < max_token_id:
                mask[:, :, token_id] = 0.0

        if state.phase == RaonPhase.SIL:
            # SILENCE: only SIL, EPAD, or BC allowed.
            _allow_token(sil_id)
            if cfg.use_duplex_end_pad:
                _allow_token(epad_id)
            if cfg.use_backchannel_token:
                _allow_token(bc_id)
        elif state.phase == RaonPhase.SPEECH:
            context_token = self._extract_context_token(state)

            if context_token is not None and context_token in onset_ids:
                # After EPAD/BC onset: only text tokens allowed.
                mask[:, :, :vocab_size] = 0.0
                for block_id in _BLOCKED_STRUCTURAL | {sil_id, pad_id} | onset_ids:
                    if block_id < mask.shape[-1]:
                        mask[:, :, block_id] = float("-inf")
            elif context_token is not None and context_token not in (_BLOCKED_STRUCTURAL | onset_ids | {pad_id, sil_id}):
                # After text: text + PAD + EPAD + SIL allowed (BC only from SIL phase).
                mask[:, :, :vocab_size] = 0.0
                _allow_token(pad_id)
                _allow_token(epad_id)
                _allow_token(sil_id)
                for block_id in _BLOCKED_STRUCTURAL:
                    if block_id < mask.shape[-1]:
                        mask[:, :, block_id] = float("-inf")
            else:
                # PAD frame (no context token): PAD + EPAD + SIL allowed.
                _allow_token(pad_id)
                _allow_token(epad_id)
                _allow_token(sil_id)

        return user_logits + mask

    def _extract_context_token(self, state: RaonMachineState) -> int | None:
        """Extract the text or EPAD context token from the last frame.

        For 3-token frames, returns the text/EPAD token. For 2-token or
        1-token frames, returns None.

        Args:
            state: Current machine state.

        Returns:
            Context token ID, or None if the frame has no text/EPAD context.
        """
        tokens = state.last_frame_tokens
        if len(tokens) != 3:
            return None

        is_uta = self._config.effective_sequence_mode == "uta"
        if is_uta:
            # UTA: [AIP, text/EPAD, AOP/A_S]
            return tokens[1]
        else:
            # TUA: [text/EPAD, AIP, AOP/A_S]
            return tokens[0]

    # ------------------------------------------------------------------
    # Sequence emission
    # ------------------------------------------------------------------

    def emit_sequence(
        self,
        frames: list[_FrameInfoProtocol],
        speak_first: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Emit input_ids and unshifted_labels from the frame state array.

        Builds shifted_labels alongside input_ids, then converts to unshifted
        via ``unshifted = [IGN] + shifted[:-1]``.

        Args:
            frames: Per-frame state and label annotations.
            speak_first: Whether the assistant speaks first.

        Returns:
            Tuple of (input_ids, unshifted_labels).
        """
        cfg = self._config
        is_uta = cfg.effective_sequence_mode == "uta"
        ign = LOSS_IGNORE_INDEX
        aop = AUDIO_OUTPUT_PLACEHOLDER.id
        aip = AUDIO_INPUT_PLACEHOLDER.id
        a_s = AUDIO_START.id
        pad_id = cfg.duplex_pad_token_id
        epad_id = cfg.duplex_end_pad_token_id
        sil_id = cfg.duplex_sil_token_id
        bc_id = cfg.duplex_bc_token_id

        input_ids: list[int] = []
        shifted_labels: list[int] = []

        # Preamble.
        first_u_label = frames[0].text_label if frames else ign
        _speak_first_onset = (
            speak_first
            and cfg.use_duplex_end_pad
            and frames
            and frames[0].phase == RaonPhase.SPEECH
            and frames[0].text_token is not None
        )
        if _speak_first_onset:
            # Speak-first: [IM_S, EPAD, A_S] preamble.
            input_ids.extend([IM_START.id, epad_id, a_s])
            shifted_labels.extend([ign, frames[0].text_token, aop])  # type: ignore[list-item]
        elif (
            is_uta
            and cfg.use_duplex_end_pad
            and frames
            and frames[0].phase == RaonPhase.SPEECH
            and frames[0].text_token is not None
        ):
            # UTA special case: first frame is text, no pre-text frames.
            input_ids.extend([IM_START.id, epad_id, a_s])
            shifted_labels.extend([ign, frames[0].text_token, aop])  # type: ignore[list-item]
        else:
            input_ids.extend([IM_START.id, a_s])
            shifted_labels.extend([first_u_label, aop])

        def _append_3tok(frame_token: int, text_label: int) -> None:
            """Append a 3-token frame (text/EPAD + [U] + [A])."""
            if is_uta:
                input_ids.extend([aip, frame_token, aop])
            else:
                input_ids.extend([frame_token, aip, aop])
            shifted_labels.extend([ign, text_label, aop])

        def _append_2tok(text_label: int) -> None:
            """Append a 2-token frame ([U] + [A])."""
            input_ids.extend([aip, aop])
            shifted_labels.extend([text_label, aop])

        # First frame already emitted in preamble for speak-first onset or UTA+EPAD special case.
        skip_first = _speak_first_onset or (
            is_uta
            and cfg.use_duplex_end_pad
            and frames
            and frames[0].phase == RaonPhase.SPEECH
            and frames[0].text_token is not None
        )

        for f_idx, frame in enumerate(frames):
            if skip_first and f_idx == 0:
                continue

            tl = frame.text_label
            is_bc_onset_frame = cfg.use_backchannel_token and frame.text_token is None and tl == bc_id
            is_onset_frame = frame.text_token is None and tl not in {pad_id, sil_id, epad_id, bc_id}

            if is_bc_onset_frame:
                if is_uta:
                    input_ids.extend([aip, bc_id, aop])
                else:
                    input_ids.extend([bc_id, aip, aop])
                shifted_labels.extend([ign, tl, aop])
            elif is_onset_frame:
                if is_uta:
                    input_ids.extend([aip, epad_id, aop])
                else:
                    input_ids.extend([epad_id, aip, aop])
                shifted_labels.extend([ign, tl, aop])
            elif frame.phase == RaonPhase.SIL:
                # Regular silence frame.
                _append_2tok(tl)
            elif frame.phase == RaonPhase.SPEECH:
                if frame.text_token is not None:
                    # SPEECH_TEXT: 3-token frame.
                    if is_uta:
                        input_ids.extend([aip, frame.text_token, aop])
                    else:
                        input_ids.extend([frame.text_token, aip, aop])
                    shifted_labels.extend([ign, tl, aop])
                else:
                    # SPEECH_PAD: 2-token frame.
                    input_ids.extend([aip, aop])
                    shifted_labels.extend([tl, aop])

        unshifted_labels = [ign] + shifted_labels[:-1]
        return input_ids, unshifted_labels


# ------------------------------------------------------------------
# Protocol for FrameInfo (avoids circular import with duplex_dataset)
# ------------------------------------------------------------------


class _FrameInfoProtocol:
    """Structural protocol for FrameInfo objects used by the state manager.

    This is not enforced at runtime; it documents the expected interface.
    """

    phase: RaonPhase
    text_token: int | None
    text_label: int


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------


def _resolve_sequence_mode_from_processor(
    processor: RaonProcessor,
) -> Literal["tua", "uta"] | None:
    """Return the explicit sequence mode override if configured.

    Args:
        processor: RaonProcessor instance.

    Returns:
        Sequence mode string or None if not configured.
    """
    sequence_mode = getattr(processor, "sequence_mode", None)
    if sequence_mode is None:
        return None
    if sequence_mode not in ("tua", "uta"):
        raise ValueError(f"Unsupported sequence_mode '{sequence_mode}'.")
    return sequence_mode  # type: ignore[return-value]

__all__ = [
    "RaonMachineState",
    "RaonPhase",
    "RaonStateConfig",
    "RaonStateManager",
]
