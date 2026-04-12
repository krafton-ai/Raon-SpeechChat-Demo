"""Inference logic for the duplex model, including TTS generation, full-duplex streaming, and RAS."""

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict, cast
from typing import Any as TextModelCache

import torch
import torch.nn.functional as F
from torch import nn
from torch._logging._internal import set_logs
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import trange
from transformers import DynamicCache, LogitsProcessorList, StaticCache, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

from worker.session_leak_logging import cleanup_resource_fields, log_session_leak_detail, log_session_leak_summary

from .audio_encoder.concurrent_audio_decoder import ConcurrentAudioDecoder
from .audio_encoder.streaming_mimi import (
    MimiConv1dPaddingCache,
    MimiConvTranspose1dPaddingCache,
    StaticMimiConv1dPaddingCache,
)
from .cleanup import (
    cleanup_failed_duplex_init,
    free_duplex_state_best_effort,
    release_transient_streaming_state,
)
from .voxtral import VoxtralStreamingState, VoxtralWrapper
from .modules import EmbeddingAdaptorOutput
from .special_tokens import (
    AUDIO_END,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    IM_END,
    IM_START,
    PAD,
)
from .state_machine import RaonMachineState, RaonPhase, RaonStateManager
from .talker_state_warning import get_talker_state_warning
from .util.delay import undelay_audio_codes

if TYPE_CHECKING:
    from .model import RaonModel
    from .modules import AudioDecoderOutput, AudioEncoderOutput, AudioTokenizerOutput

set_logs(recompiles=True)
logger = logging.getLogger(__name__)


class GenerateOutput(TypedDict):
    """Output container for TTS and speech generation.

    sequences: Generated token IDs. Shape: [batch_size, seq_length]. Dtype: long.
    audio_codes: Generated audio codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
    audio_codes_mask: Mask for valid audio positions. Shape: [batch_size, num_frames]. Dtype: bool.
    audio: Decoded waveform. Shape: [batch_size, num_samples]. Dtype: float.
    audio_lengths: Per-sequence audio lengths. Shape: [batch_size]. Dtype: long.
    """

    sequences: torch.Tensor
    audio_codes: torch.Tensor | None
    audio_codes_mask: torch.Tensor | None
    audio: torch.Tensor | None
    audio_lengths: torch.Tensor | None


class RaonGenerateResult(NamedTuple):
    """Result of batched or single duplex generation.

    audio_output: Decoded waveform. Shape: [1, num_samples]. Dtype: float.
    predicted_text: Decoded text prediction, or None if no tokenizer available.
    sequences: Full generated token sequence. Shape: [1, seq_length]. Dtype: long.
    """

    audio_output: torch.Tensor
    predicted_text: str | None
    sequences: torch.Tensor | None = None


def extract_predicted_text(
    sequences: torch.Tensor,
    tokenizer: Any,
    text_vocab_size: int,
    initial_seq_len: int = 0,
    eos_token_id: int | None = None,
    sil_token_id: int | None = None,
    audio_start_token_id: int | None = None,
    ignored_token_ids: set[int] | None = None,
) -> str:
    """Extract and decode predicted text tokens from a generated sequence.

    Filters out special tokens and audio tokens, keeping only text tokens
    (those with IDs below text_vocab_size). SIL and audio-start tokens are
    converted to line breaks to preserve turn structure.

    Args:
        sequences: Full sequence tensor.
            Shape: [batch_size, seq_length] or [seq_length]. Dtype: long.
        tokenizer: HuggingFace tokenizer for decoding.
        text_vocab_size: Tokens below this threshold are text tokens.
        initial_seq_len: Length of initial prompt to skip.
        eos_token_id: EOS token ID to exclude from decoding.
        sil_token_id: SIL token ID to convert into line breaks.
        audio_start_token_id: Audio-start token ID to convert into line breaks.
        ignored_token_ids: Additional special token IDs to skip.

    Returns:
        Decoded text string from predicted tokens.
    """
    if sequences.dim() == 2:
        seq = sequences[0]
    else:
        seq = sequences

    new_tokens = seq[initial_seq_len:].tolist()

    ignored = set(ignored_token_ids or ())
    if eos_token_id is not None:
        ignored.add(eos_token_id)

    pieces: list[str] = []
    text_buffer: list[int] = []

    def flush_text_buffer() -> None:
        if not text_buffer:
            return
        try:
            pieces.append(tokenizer.decode(text_buffer))
        except Exception:
            pieces.append(f"[decode error: {len(text_buffer)} tokens]")
        text_buffer.clear()

    for token_id in new_tokens:
        if (sil_token_id is not None and token_id == sil_token_id) or (
            audio_start_token_id is not None and token_id == audio_start_token_id
        ):
            flush_text_buffer()
            pieces.append("\n")
            continue
        if token_id in ignored:
            flush_text_buffer()
            continue
        if token_id < text_vocab_size:
            text_buffer.append(token_id)

    flush_text_buffer()
    return "".join(pieces).strip()

AudioInputEncoderCache = VoxtralStreamingState


@dataclass
class RaonDecodingState:
    """Mutable state for full-duplex streaming decoding.

    Tracks sequences, attention masks, audio codes, KV cache, encoder cache,
    decoder stream ID, sampling config, and inline/acoustic-delay state.
    """

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    audio_codes: torch.Tensor
    audio_codes_mask: torch.Tensor
    past_key_values: TextModelCache
    audio_input_encoder_cache: AudioInputEncoderCache
    audio_decoder_stream_id: int
    do_sample: bool
    logits_processor: LogitsProcessorList
    num_code_groups: int = 8
    delay_buffer: torch.Tensor | None = None
    delay_buffer_position: int = 0
    # For acoustic delay processing: stores semantic codes from previous frame
    semantic_buffer: torch.Tensor | None = None
    # Penalty to subtract from eos logit (higher = longer responses)
    eos_penalty: float = 0.0
    # Penalty to subtract from SIL logit (higher = less silence).
    sil_penalty: float = 0.0
    # Penalty to subtract from BC logit in SIL phase (positive = suppress, negative = boost).
    bc_penalty: float = 0.0
    # Pre-computed speaker embedding for voice conditioning
    speaker_embeds: torch.Tensor | None = None
    # Mealy machine state for the duplex state manager
    machine_state: RaonMachineState | None = None
    # Per-session talker KV cache and cumulative attention mask (moved from model-level to per-session).
    talker_past_key_values: DynamicCache | None = None
    talker_attention_mask: torch.Tensor | None = None
    # Number of remaining frames where SIL is forced (listen-first warmup).
    forced_sil_remaining: int = 0

    def _reset(self) -> None:
        """Reset the decoding state for reuse."""
        device = self.sequences.device
        self.sequences = torch.zeros(1, 0, dtype=torch.long, device=device)
        self.attention_mask = torch.zeros(1, 0, dtype=torch.long, device=device)
        self.audio_codes = torch.zeros(1, 0, self.num_code_groups, dtype=torch.long, device=device)
        self.audio_codes_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
        self.delay_buffer_position = 0

        if self.delay_buffer is not None:
            self.delay_buffer.zero_()

        if self.semantic_buffer is not None:
            self.semantic_buffer = None
        self.talker_past_key_values = None
        self.talker_attention_mask = None

        if self.machine_state is not None:
            self.machine_state = RaonMachineState(
                phase=RaonPhase.SIL,
                last_frame_tokens=[AUDIO_INPUT_PLACEHOLDER.id, AUDIO_OUTPUT_PLACEHOLDER.id],
            )

        self.audio_input_encoder_cache.reset()


class RaonInferenceModel(ABC):
    """Abstract base class for duplex inference: TTS generation, full-duplex streaming, and RAS."""

    vocab_size: int
    codebook_size: int
    use_duplex_end_pad: bool
    sequence_mode: Literal["tua", "uta"] | None
    use_sil_token: bool
    no_audio_in_sil: bool
    duplex_sil_token_id: int
    use_backchannel_token: bool
    duplex_bc_token_id: int
    num_code_groups: int
    sampling_rate: int
    frame_rate: float
    delays: list[int]
    max_delay: int
    tokenizer: Any | None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert hasattr(self, "vocab_size"), "Model must have vocab_size attribute."
        assert hasattr(self, "codebook_size"), "Model must have codebook_size attribute."
        assert hasattr(self, "use_duplex_end_pad"), "Model must have use_duplex_end_pad attribute."
        assert hasattr(self, "num_code_groups"), "Model must have num_code_groups attribute."
        assert hasattr(self, "sampling_rate"), "Model must have sampling_rate attribute."
        assert hasattr(self, "frame_rate"), "Model must have frame_rate attribute."
        assert hasattr(self, "delays"), "Model must have delays attribute."
        assert hasattr(self, "max_delay"), "Model must have max_delay attribute."
        assert hasattr(self, "sequence_mode"), "Model must have sequence_mode attribute."
        assert hasattr(self, "use_sil_token"), "Model must have use_sil_token attribute."
        assert hasattr(self, "duplex_sil_token_id"), "Model must have duplex_sil_token_id attribute."
        assert hasattr(self, "use_backchannel_token"), "Model must have use_backchannel_token attribute."
        assert hasattr(self, "duplex_bc_token_id"), "Model must have duplex_bc_token_id attribute."
        self.concurrent_audio_decoder: ConcurrentAudioDecoder | None = None
        self._state_manager: RaonStateManager = RaonStateManager.from_inference_model(self)
        self._audio_input_outer_calls = 0
        self._audio_input_zero_embed_calls = 0
        self._audio_input_warmup_calls = 0
        self._audio_input_outer_wrapper_compiled = False

    def _get_sequence_mode(self) -> Literal["tua", "uta"]:
        """Return the configured duplex frame ordering.

        Delegates to state manager configuration. Kept for backward
        compatibility with sglang_backend and tests.
        """
        return self._state_manager._config.effective_sequence_mode

    def _apply_state_machine_logit_mask(
        self,
        user_logits: torch.Tensor,
        sequences: torch.Tensor,
        machine_state: RaonMachineState | None = None,
    ) -> torch.Tensor:
        """Mask logits to enforce valid state-machine transitions.

        Delegates to ``RaonStateManager.apply_logit_mask`` when a machine_state
        is provided. Falls back to sequence-based inference for backward
        compatibility.

        Args:
            user_logits: Text logits. Shape: [1, 1, vocab_size]. Dtype: float.
            sequences: Current token sequence. Shape: [1, seq_len]. Dtype: long.
            machine_state: Current Mealy machine state, or None for legacy mode.

        Returns:
            Masked logits with invalid tokens set to -inf.
        """
        if machine_state is not None:
            return self._state_manager.apply_logit_mask(user_logits, machine_state, self.vocab_size)

        # Legacy fallback: infer state from sequences.
        state = self._infer_machine_state_from_sequences(sequences)
        return self._state_manager.apply_logit_mask(user_logits, state, self.vocab_size)

    def _infer_machine_state_from_sequences(self, sequences: torch.Tensor) -> RaonMachineState:
        """Reconstruct a RaonMachineState from trailing sequence tokens.

        Used for backward compatibility when no explicit machine_state is
        available (e.g. sglang_backend).

        .. warning::
            This heuristic cannot distinguish SPEECH_PAD ``[AIP, AOP]`` from
            SIL ``[AIP, AOP]`` — both are 2-token frames with identical tokens.
            When an explicit ``machine_state`` is available, prefer passing it
            directly to avoid incorrect phase inference.

        Args:
            sequences: Current token sequence. Shape: [1, seq_len]. Dtype: long.

        Returns:
            Inferred RaonMachineState.
        """
        context_token = self._get_standard_mode_text_context_token(sequences)
        if context_token is not None:
            # 3-token frame: always SPEECH (text or EPAD context present).
            last3 = [
                int(sequences[0, -3].item()),
                int(sequences[0, -2].item()),
                int(sequences[0, -1].item()),
            ]
            return RaonMachineState(phase=RaonPhase.SPEECH, last_frame_tokens=last3)

        # 2-token frame: cannot reliably distinguish SPEECH_PAD from SIL.
        # Default to SIL which is safe for logit masking (SIL allows SIL + EPAD;
        # SPEECH_PAD allows PAD + EPAD + SIL — the SIL mask is a subset).
        if sequences.shape[1] >= 2:
            last2 = [int(sequences[0, -2].item()), int(sequences[0, -1].item())]
            return RaonMachineState(phase=RaonPhase.SIL, last_frame_tokens=last2)

        return RaonMachineState(
            phase=RaonPhase.SIL,
            last_frame_tokens=[AUDIO_INPUT_PLACEHOLDER.id, AUDIO_OUTPUT_PLACEHOLDER.id],
        )

    def _build_standard_mode_frame_input_ids(
        self,
        predicted_token_id: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, bool]:
        """Build the next standard-mode frame using the configured token ordering."""
        state = RaonMachineState(
            phase=RaonPhase.SPEECH,
            last_frame_tokens=[AUDIO_INPUT_PLACEHOLDER.id, AUDIO_OUTPUT_PLACEHOLDER.id],
        )

        new_state, frame_tokens, emitted_audio = self._state_manager.transition(state, predicted_token_id)
        return torch.tensor([frame_tokens], device=device), emitted_audio

    def _use_condensed_silence(self) -> bool:
        """Return False because condensed silence mode has been removed."""
        return False

    def _ends_with_condensed_silence_step(self, sequences: torch.Tensor) -> bool:
        """Return False because condensed silence mode has been removed."""
        return False

    def _get_standard_mode_text_context_token(self, sequences: torch.Tensor) -> int | None:
        """Return the text or EPAD token from the trailing standard-mode frame."""
        if sequences.shape[1] < 3:
            return None

        if self._state_manager._config.effective_sequence_mode == "uta":
            if int(sequences[0, -3].item()) != AUDIO_INPUT_PLACEHOLDER.id:
                return None
            if int(sequences[0, -1].item()) not in (AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id):
                return None
            return int(sequences[0, -2].item())

        if int(sequences[0, -2].item()) != AUDIO_INPUT_PLACEHOLDER.id:
            return None
        if int(sequences[0, -1].item()) not in (AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id):
            return None
        return int(sequences[0, -3].item())

    @abstractmethod
    def inference_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        speaker_embeds: torch.Tensor | None = None,
        use_cache: bool | None = False,
        past_key_values: TextModelCache = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def tokenize_audio(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        return_mimi_features: bool = False,
        encoder_past_key_values: StaticCache | None = None,
        conv_padding_cache: StaticMimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> "AudioTokenizerOutput": ...

    @abstractmethod
    def get_audio_input_embeds(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        encoder_past_key_values: StaticCache | None = None,
        conv_padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> "AudioEncoderOutput": ...

    @abstractmethod
    def decode_audio(
        self,
        audio_codes: torch.Tensor,
        decoder_past_key_values: StaticCache | None = None,
        conv_padding_cache: StaticMimiConv1dPaddingCache | None = None,
        conv_transpose_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> "AudioDecoderOutput":
        """Decode discrete audio codes into waveform.

        Implementations must return an AudioDecoderOutput with audio waveform.

        Args:
            audio_codes: Discrete codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            decoder_past_key_values: Decoder KV cache for streaming.
            conv_padding_cache: Conv padding cache for streaming.
            conv_transpose_padding_cache: Conv-transpose padding cache for streaming.
            use_streaming: Whether to use streaming mode.

        Returns:
            AudioDecoderOutput with decoded audio waveform.
        """
        ...

    @abstractmethod
    def generate_audio_codes(
        self,
        talker_last_hidden_state: torch.Tensor,
        first_code_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
        allow_audio_end: bool = True,
    ) -> torch.Tensor:
        """Generate audio codes from talker hidden state using the code predictor.

        Implementations may use first_code_sampler to sample the first code;
        remaining codes are typically generated greedily.

        Args:
            talker_last_hidden_state: Hidden states from the language model.
                Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            first_code_sampler: Optional callable to sample first code from logits.
                Input logits shape: [batch_size, codebook_size]. Returns: [batch_size, 1]. Dtype: long.
            allow_audio_end: If False, suppress AUDIO_END sampling for duplex-style decoding.

        Returns:
            Generated audio codes. Shape: [batch_size, num_code_groups]. Dtype: long.
        """
        ...

    @abstractmethod
    def get_proj_code(self) -> nn.Linear:
        """Return the audio code projection layer."""
        ...

    @abstractmethod
    def get_model(self) -> "RaonModel":
        """Return the underlying RaonModel."""
        ...

    def _compute_speaker_embeds(
        self,
        speaker_audio: torch.Tensor,
        speaker_audio_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute speaker embeddings from raw audio using the model's speaker encoder.

        Handles both pretrained (ECAPA-TDNN) and mimi-based speaker encoders.

        Args:
            speaker_audio: Raw waveform at model sampling rate.
                Shape: [batch_size, num_samples]. Dtype: float.
            speaker_audio_lengths: Valid sample count per batch element.
                Shape: [batch_size]. Dtype: long.

        Returns:
            Speaker embeddings. Shape: [batch_size, 1, hidden_size]. Dtype: float.
        """
        from .modules import PretrainedSpeakerEncoder

        model = self.get_model()
        assert model.speaker_encoder is not None, "_compute_speaker_embeds requires a model with a speaker encoder."

        if speaker_audio_lengths is None:
            speaker_audio_lengths = torch.full(
                (speaker_audio.shape[0],), speaker_audio.shape[1], device=speaker_audio.device, dtype=torch.long
            )

        if isinstance(model.speaker_encoder, PretrainedSpeakerEncoder):
            return model.speaker_encoder(speaker_audio, speaker_audio_lengths)

        tokenizer_output = self.tokenize_audio(
            audio=speaker_audio,
            audio_lengths=speaker_audio_lengths,
            return_mimi_features=True,
        )
        return model.speaker_encoder(
            tokenizer_output.mimi_features,
            mask=tokenizer_output.audio_codes_mask,
        )

    @abstractmethod
    def init_past_key_values(
        self,
        batch_size: int,
        max_sequence_length: int,
        prev_cache: TextModelCache | None = None,
    ) -> TextModelCache:
        """Allocate or reuse KV cache for incremental decoding.

        Args:
            batch_size: Number of sequences in the batch.
            max_sequence_length: Maximum sequence length to support.
            prev_cache: Optional previous cache to reuse.

        Returns:
            Initialized KV cache for use with inference_forward.
        """
        ...

    @abstractmethod
    def free_past_key_values(self, past_key_values: TextModelCache) -> None:
        """Release KV cache resources."""
        ...

    def start_concurrent_audio_decoder(self, timeout: float = 5.0) -> None:
        if self.concurrent_audio_decoder is None:
            self.concurrent_audio_decoder = ConcurrentAudioDecoder(self)

        if not self.concurrent_audio_decoder.is_running:
            self.concurrent_audio_decoder.start(timeout=timeout)

    def stop_concurrent_audio_decoder(self, timeout: float | None = 5.0) -> None:
        """Stop the background audio decoder worker and release resources."""
        if self.concurrent_audio_decoder is not None:
            self.concurrent_audio_decoder.stop(timeout=timeout)
            self.concurrent_audio_decoder = None

    def create_audio_decoder_stream(self) -> int:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        return self.concurrent_audio_decoder.create_stream()

    def _destroy_audio_decoder_stream(self, stream_id: int) -> None:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        self.concurrent_audio_decoder.destroy_stream(stream_id)

    def reset_audio_decoder_stream(self, stream_id: int) -> None:
        """Reset decoder conv state for a stream at speech onset.

        Clears conv padding caches so the next decoded frame starts with fresh
        state (no artifacts from init placeholder codes or previous utterance).
        """
        if self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running:
            self.concurrent_audio_decoder.reset_stream(stream_id)

    def get_silence_codes(self, device: torch.device) -> torch.Tensor:
        """Return cached silence audio codes for keeping the decoder warm during SIL frames.

        Encodes a single frame of zero audio through Mimi to obtain the codes that
        represent true silence, then caches the result for reuse.

        Args:
            device: Target device for the returned tensor.

        Returns:
            Silence codes. Shape: [num_code_groups]. Dtype: long.
        """
        if not hasattr(self, "_silence_codes") or self._silence_codes is None:
            samples_per_frame = int(self.sampling_rate / self.frame_rate)
            silence_audio = torch.zeros(1, 1, samples_per_frame, device=device)
            silence_lengths = torch.tensor([samples_per_frame], device=device)
            with torch.no_grad():
                result = self.tokenize_audio(
                    audio=silence_audio,
                    audio_lengths=silence_lengths,
                    num_code_groups=self.num_code_groups,
                )
            self._silence_codes = result.audio_codes[0, 0].to(device)  # [num_code_groups]
        return self._silence_codes.to(device)

    def push_audio_codes(self, audio_codes: torch.Tensor, stream_id: int) -> None:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        assert audio_codes.ndim == 1 and audio_codes.shape[0] == self.num_code_groups, (
            f"Expected 1D audio codes with shape `[{self.num_code_groups}]` but got `{audio_codes.shape}`."
        )
        self.concurrent_audio_decoder.push_audio_codes(stream_id, audio_codes[None, None])

    def pull_audio(self, stream_id: int) -> torch.Tensor:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        results = self.concurrent_audio_decoder.drain_to(max_pending=1, stream_id=stream_id)
        assert len(results) == 1, f"Expected exactly one audio result but got `{len(results)}`."
        _, decoded_audio = results[0]
        return decoded_audio

    def _drain_audio_decoding_queue(self, stream_id: int) -> list[tuple[int, torch.Tensor]]:
        assert self.concurrent_audio_decoder is not None and self.concurrent_audio_decoder.is_running, (
            "Concurrent audio decoder must be running."
        )
        return self.concurrent_audio_decoder.drain_to(max_pending=0, stream_id=stream_id)

    def free_duplex_decoding_state(self, state: RaonDecodingState) -> None:
        fields = cleanup_resource_fields(
            stream_id=state.audio_decoder_stream_id,
            has_past_key_values=state.past_key_values is not None,
            has_audio_cache=state.audio_input_encoder_cache is not None,
            has_talker_cache=state.talker_past_key_values is not None,
            has_talker_mask=state.talker_attention_mask is not None,
        )
        log_session_leak_summary(
            logger,
            "runtime_free_state_start",
            **fields,
            seq_len=state.sequences.shape[1],
            audio_frames=state.audio_codes.shape[1],
        )
        free_duplex_state_best_effort(
            state=state,
            drain_audio_decoding_queue=self._drain_audio_decoding_queue,
            destroy_audio_decoder_stream=self._destroy_audio_decoder_stream,
            free_past_key_values=self.free_past_key_values,
        )
        log_session_leak_summary(logger, "runtime_free_state_done", **fields)

    def init_audio_encoder_cache(
        self,
        prev_cache: AudioInputEncoderCache | None = None,
    ) -> AudioInputEncoderCache:
        """Initialize or reset the Voxtral audio input encoder cache for streaming."""
        model = self.get_model()
        assert isinstance(model.audio_encoder, VoxtralWrapper)
        if prev_cache is not None:
            prev_cache.reset()
            return prev_cache
        return model.audio_encoder.init_streaming_state()

    def _streaming_get_audio_input_embeds_voxtral(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        streaming_state: VoxtralStreamingState,
    ) -> tuple[torch.Tensor, torch.Tensor, VoxtralStreamingState]:
        """Get audio input embeddings using Voxtral causal streaming encoder.

        Runs the Voxtral wrapper in streaming mode (one chunk at a time) and
        passes the output through the input adaptor.

        Args:
            audio: Raw audio frame. Shape: [1, num_samples]. Dtype: float.
            audio_lengths: Per-sample lengths. Shape: [1]. Dtype: long.
            streaming_state: Current Voxtral streaming state.

        Returns:
            Tuple of (audio_embeds, audio_embeds_mask, updated_streaming_state).
            audio_embeds: Shape [1, num_frames, hidden_size]. Dtype: float.
            audio_embeds_mask: Shape [1, num_frames]. Dtype: bool.
        """
        model = self.get_model()
        assert isinstance(model.audio_encoder, VoxtralWrapper)
        del audio_lengths
        self._audio_input_outer_calls += 1

        encoder_outputs = model.audio_encoder.forward_streaming_chunk(audio, streaming_state)

        assert encoder_outputs.embeds is not None
        audio_embeds = encoder_outputs.embeds
        updated_state = encoder_outputs.streaming_state
        assert isinstance(updated_state, VoxtralStreamingState)

        audio_embeds_mask = torch.ones(
            audio_embeds.shape[:2],
            dtype=torch.bool,
            device=audio_embeds.device,
        )

        assert model.input_adaptor is not None, "input_adaptor is unavailable when supports_audio_input is False."
        adaptor_outputs = model.input_adaptor(audio_embeds, mask=audio_embeds_mask)
        assert isinstance(adaptor_outputs, EmbeddingAdaptorOutput)
        assert (audio_embeds := adaptor_outputs.outputs_embeds) is not None
        assert (audio_embeds_mask := adaptor_outputs.mask) is not None  # type: ignore
        if audio_embeds.shape[1] == 0:
            self._audio_input_zero_embed_calls += 1

        return audio_embeds, audio_embeds_mask, updated_state

    def get_audio_runtime_stats(self) -> dict[str, object]:
        """Expose audio-path runtime counters for worker health and log review."""
        model = self.get_model()
        encoder_stats: dict[str, object] | None = None
        if isinstance(model.audio_encoder, VoxtralWrapper):
            encoder_stats = cast(dict[str, object], model.audio_encoder.get_streaming_runtime_stats())
        return {
            "outer_wrapper_mode": "eager",
            "outer_wrapper_compiled": self._audio_input_outer_wrapper_compiled,
            "outer_calls": self._audio_input_outer_calls,
            "outer_zero_embed_calls": self._audio_input_zero_embed_calls,
            "warmup_calls": self._audio_input_warmup_calls,
            "encoder": encoder_stats,
        }

    def compile_audio_modules(self, duplex: bool = True, max_sequence_length: int = 8192) -> RaonDecodingState | None:
        """Enable CUDA-graph code prediction and warm the duplex streaming path.

        Args:
            duplex: If True, run duplex warmup; else only code predictor.
            max_sequence_length: Max sequence length for KV cache during warmup.

        Returns:
            RaonDecodingState after warmup if duplex=True; None otherwise.
        """
        warnings.filterwarnings("ignore", message="Logical operators 'and' and 'or' are deprecated")

        model = self.get_model()
        code_predictor = model.code_predictor
        assert code_predictor is not None, "compile_audio_modules requires a model with audio output support."
        code_predictor.enable_predict_codes_cuda_graph()

        # Compile the Voxtral encoder forward for CUDA graph capture.
        if isinstance(model.audio_encoder, VoxtralWrapper):
            model.audio_encoder.compile_encoder()

        if duplex:
            self._audio_input_outer_wrapper_compiled = False
            logger.info(
                "audio_input_path configured with eager outer wrapper and compiled Voxtral transformer core"
            )

            set_logs(recompiles=False)
            with torch.inference_mode():
                device = self.get_model().device
                dtype = self.get_model().dtype
                samples_per_frame = int(self.sampling_rate / self.frame_rate)

                safe_vocab = min(self.vocab_size, 1000)
                state = self.init_duplex_decoding_state(
                    sequences=torch.randint(0, safe_vocab, (1, 1), device=device),
                    attention_mask=torch.ones(1, 1, dtype=torch.long, device=device),
                    do_sample=False,
                    max_sequence_length=max_sequence_length,
                )

                for step in trange(8, desc="Warmup 1/3", mininterval=0):
                    state, _ = self.duplex_decoding_step(
                        state=state,
                        audio_input=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                    )

                self._drain_audio_decoding_queue(state.audio_decoder_stream_id)

                state = self.init_duplex_decoding_state(
                    sequences=torch.randint(0, safe_vocab, (1, 20), device=device),
                    attention_mask=torch.ones(1, 20, dtype=torch.long, device=device),
                    do_sample=False,
                    max_sequence_length=max_sequence_length,
                    prev_state=state,
                )
                for step in trange(8, desc="Warmup 2/3", mininterval=0):
                    state, _ = self.duplex_decoding_step(
                        state=state,
                        audio_input=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                    )

                self._drain_audio_decoding_queue(state.audio_decoder_stream_id)

                audio_input_encoder_cache: AudioInputEncoderCache = state.audio_input_encoder_cache
                # Acquire a pool cache for the encoder-only warmup so that
                # torch.compile traces the SlidingWindowVoxtralKVCache path
                # using the exact same Python objects that real sessions will use.
                if isinstance(audio_input_encoder_cache, VoxtralStreamingState) and isinstance(
                    model.audio_encoder, VoxtralWrapper
                ):
                    pool_available_before = len(model.audio_encoder._cache_available)
                    audio_input_encoder_cache = model.audio_encoder.init_streaming_state()
                    log_session_leak_detail(
                        logger,
                        "warmup_transient_cache_acquired",
                        has_audio_cache=True,
                        pool_slot=getattr(audio_input_encoder_cache, "_pool_idx", None),
                        pool_available_before=pool_available_before,
                        pool_available_after=len(model.audio_encoder._cache_available),
                    )

                try:
                    for _ in trange(256, desc="Warmup 3/3", mininterval=0):
                        self._audio_input_warmup_calls += 1
                        _, _, audio_input_encoder_cache = self._streaming_get_audio_input_embeds_voxtral(
                            audio=torch.randn(1, samples_per_frame, device=device, dtype=dtype),
                            audio_lengths=torch.tensor([samples_per_frame], device=device),
                            streaming_state=audio_input_encoder_cache,
                        )
                finally:
                    release_transient_streaming_state(
                        audio_input_encoder_cache,
                        state.audio_input_encoder_cache,
                    )

                self.free_duplex_decoding_state(state)
                set_logs(recompiles=True)
                return state

        return None

    @staticmethod
    @torch.inference_mode()
    def _apply_repetition_aware_sampling(
        sampled_ids: torch.Tensor,
        logits: torch.Tensor,
        audio_codes: torch.Tensor,
        window_size: int,
        repetition_threshold: float,
    ) -> torch.Tensor:
        batch_size = sampled_ids.shape[0]
        result_ids = sampled_ids.clone()
        first_group_codes = audio_codes[:, :, 0]

        for b in range(batch_size):
            sampled_token = sampled_ids[b, 0].item()
            codes_seq = first_group_codes[b]

            window_start = max(0, codes_seq.shape[0] - window_size)
            window = codes_seq[window_start:]
            if window.numel() == 0:
                continue

            repetition_count = (window == sampled_token).sum().item()
            repetition_ratio = repetition_count / window.numel()
            if repetition_ratio > repetition_threshold:
                probs = F.softmax(logits[b], dim=-1, dtype=torch.float32)
                probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
                new_id = torch.multinomial(probs, num_samples=1)
                result_ids[b, 0] = new_id[0]

        return result_ids

    @staticmethod
    def _make_audio_code_sampler(
        sequences: torch.Tensor,
        logits_processor: LogitsProcessorList,
        audio_codes: torch.Tensor,
        ras_enabled: bool,
        ras_window_size: int,
        ras_repetition_threshold: float,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def sample_audio_code(logits: torch.Tensor) -> torch.Tensor:
            processed_logits = logits_processor(input_ids=sequences, scores=logits)  # type: ignore
            probs = F.softmax(processed_logits, dim=-1, dtype=torch.float32)
            probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
            sampled_ids = torch.multinomial(probs, num_samples=1)
            if ras_enabled and audio_codes.shape[1] > 0:
                sampled_ids = RaonInferenceModel._apply_repetition_aware_sampling(
                    sampled_ids=sampled_ids,
                    logits=logits,
                    audio_codes=audio_codes,
                    window_size=ras_window_size,
                    repetition_threshold=ras_repetition_threshold,
                )
            return sampled_ids

        return sample_audio_code

    @torch.inference_mode()
    def _sample_from_logits(
        self,
        sequences: torch.Tensor,
        logits: torch.Tensor,
        force_audio_output: bool,
        force_text_output: bool,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
    ) -> torch.Tensor:
        """
        Sample text/output-control tokens from logits.

        Args:
            sequences: Input sequences tensor
            logits: Logits tensor from model
            force_audio_output: If True, only sample from audio tokens
            force_text_output: If True, only sample from text tokens
            do_sample: If True, use sampling; otherwise use argmax
            logits_processor: LogitsProcessorList for processing logits
        """
        if force_audio_output:
            return torch.full((logits.shape[0], 1), fill_value=AUDIO_OUTPUT_PAD.id, dtype=torch.long, device=logits.device)

        if force_text_output:
            logits[..., AUDIO_OUTPUT_PAD.id] = torch.finfo(logits.dtype).tiny

        if do_sample:
            processed_logits = logits_processor(input_ids=sequences, scores=logits[:, -1])  # type: ignore
            probs = F.softmax(processed_logits, dim=-1, dtype=torch.float32)
            probs = probs.clamp_min(torch.finfo(probs.dtype).tiny)
            return torch.multinomial(probs, num_samples=1)
        else:
            return logits[:, -1:].argmax(dim=-1)

    @torch.inference_mode()
    def _update_sequences_and_generate_audio_codes(
        self,
        new_logits: torch.Tensor,
        new_last_hidden_state: torch.Tensor,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        is_complete: torch.Tensor,
        pad_token_id: int,
        force_audio_output: bool,
        force_text_output: bool,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        is_generating_audio: torch.Tensor | None = None,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        next_is_generating_audio = is_generating_audio.clone() if is_generating_audio is not None else None
        if is_generating_audio is None:
            new_ids = self._sample_from_logits(
                sequences=sequences,
                logits=new_logits,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                do_sample=do_sample,
                logits_processor=logits_processor,
            )
        else:
            new_ids = torch.full(
                (new_logits.shape[0], 1),
                fill_value=pad_token_id,
                dtype=torch.long,
                device=new_logits.device,
            )
            text_mode_mask = ~is_generating_audio
            if text_mode_mask.any():
                sampled_text_ids = self._sample_from_logits(
                    sequences=sequences[text_mode_mask],
                    logits=new_logits[text_mode_mask],
                    force_audio_output=False,
                    force_text_output=True,
                    do_sample=do_sample,
                    logits_processor=logits_processor,
                )
                new_ids[text_mode_mask] = sampled_text_ids
            if is_generating_audio.any():
                new_ids[is_generating_audio] = AUDIO_OUTPUT_PAD.id

        new_ids[is_complete, -1] = pad_token_id
        is_complete |= new_ids[:, -1] == IM_END.id
        if next_is_generating_audio is not None:
            assert is_generating_audio is not None
            start_audio_mask = (~is_complete) & (~is_generating_audio) & (new_ids[:, -1] == AUDIO_START.id)
            next_is_generating_audio[start_audio_mask] = True
            is_audio_output = (~is_complete) & is_generating_audio
        else:
            is_audio_output = (~is_complete) & (new_ids[:, -1] == AUDIO_OUTPUT_PAD.id)
            new_ids[is_audio_output, -1] = AUDIO_OUTPUT_PLACEHOLDER.id

        final_audio_output_mask = is_audio_output.clone()
        sequences_with_new_ids = torch.cat((sequences, new_ids), dim=1)
        attention_mask = F.pad(attention_mask, (0, 1), value=1)
        audio_codes_mask = torch.cat((audio_codes_mask, final_audio_output_mask[:, None]), dim=1)
        audio_codes = F.pad(audio_codes, (0, 0, 0, 1))

        if is_audio_output.any():
            first_code_sampler = None
            if do_sample:
                first_code_sampler = self._make_audio_code_sampler(
                    sequences=sequences_with_new_ids[is_audio_output],
                    logits_processor=logits_processor,
                    audio_codes=audio_codes[is_audio_output, :-1],
                    ras_enabled=ras_enabled,
                    ras_window_size=ras_window_size,
                    ras_repetition_threshold=ras_repetition_threshold,
                )

            generated_audio_codes = self.generate_audio_codes(
                talker_last_hidden_state=new_last_hidden_state[is_audio_output, -1:],
                first_code_sampler=first_code_sampler,
            )
            generated_audio_end_mask = generated_audio_codes[:, 0] == self.codebook_size
            new_ids[is_audio_output, -1] = AUDIO_OUTPUT_PLACEHOLDER.id
            if generated_audio_end_mask.any():
                local_non_end_mask = ~generated_audio_end_mask
                global_non_end_mask = is_audio_output.clone()
                global_non_end_mask[is_audio_output] = local_non_end_mask
                final_audio_output_mask = global_non_end_mask
                audio_end_global_mask = is_audio_output.clone()
                audio_end_global_mask[is_audio_output] = generated_audio_end_mask
                new_ids[audio_end_global_mask, -1] = AUDIO_END.id
                if next_is_generating_audio is not None:
                    next_is_generating_audio[audio_end_global_mask] = False
                if local_non_end_mask.any():
                    audio_codes[global_non_end_mask, -1] = generated_audio_codes[local_non_end_mask]
            else:
                audio_codes[is_audio_output, -1] = generated_audio_codes

        audio_codes_mask[:, -1] = final_audio_output_mask
        sequences = torch.cat((sequences, new_ids), dim=1)
        if next_is_generating_audio is not None:
            next_is_generating_audio[is_complete] = False

        return sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, next_is_generating_audio

    @torch.inference_mode()
    def _generation_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        audio_input: torch.Tensor | None,
        audio_output: torch.Tensor | None,
        audio_input_lengths: torch.Tensor | None,
        audio_output_lengths: torch.Tensor | None,
        audio_output_codes: torch.Tensor | None,
        pad_token_id: int,
        force_audio_output: bool,
        force_text_output: bool,
        disable_eos_on_first_output: bool,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        max_sequence_length: int,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        speaker_embeds: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        TextModelCache,
    ]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = attention_mask.cumsum(dim=1) - 1

        past_key_values = self.init_past_key_values(batch_size=input_ids.shape[0], max_sequence_length=max_sequence_length)
        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
        talker_last_hidden_state, text_logits = self.inference_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            audio_input=audio_input,
            audio_output=audio_output,
            audio_input_lengths=audio_input_lengths,
            audio_output_lengths=audio_output_lengths,
            audio_output_codes=audio_output_codes,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=speaker_embeds,
            use_cache=True,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        if disable_eos_on_first_output:
            text_logits[..., IM_END.id] = torch.finfo(text_logits.dtype).tiny

        sequences = input_ids
        audio_codes = torch.zeros(
            input_ids.shape[0],
            0,
            self.num_code_groups,
            dtype=torch.long,
            device=input_ids.device,
        )
        audio_codes_mask = torch.zeros(input_ids.shape[0], 0, dtype=torch.bool, device=input_ids.device)
        is_complete = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if force_audio_output:
            is_generating_audio: torch.Tensor | None = torch.ones(
                input_ids.shape[0],
                dtype=torch.bool,
                device=input_ids.device,
            )
        elif force_text_output:
            is_generating_audio = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        else:
            is_generating_audio = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio = (
            self._update_sequences_and_generate_audio_codes(
                new_logits=text_logits,
                new_last_hidden_state=talker_last_hidden_state,
                sequences=sequences,
                attention_mask=attention_mask,
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                is_complete=is_complete,
                pad_token_id=pad_token_id,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                do_sample=do_sample,
                logits_processor=logits_processor,
                is_generating_audio=is_generating_audio,
                ras_enabled=ras_enabled,
                ras_window_size=ras_window_size,
                ras_repetition_threshold=ras_repetition_threshold,
            )
        )
        return (
            sequences,
            attention_mask,
            audio_codes,
            audio_codes_mask,
            is_complete,
            is_generating_audio,
            past_key_values,
        )

    @torch.inference_mode()
    def _decoding_step(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        is_complete: torch.Tensor,
        past_key_values: StaticCache,
        pad_token_id: int,
        force_audio_output: bool,
        force_text_output: bool,
        is_generating_audio: torch.Tensor | None,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        supports_audio_output = bool(getattr(self, "supports_audio_output", True))
        cache_position = attention_mask.sum(dim=1, keepdim=False) - 1  # [batch_size]
        talker_last_hidden_state, text_logits = self.inference_forward(
            input_ids=sequences[:, -1:],
            position_ids=cache_position.unsqueeze(1),
            attention_mask=attention_mask,
            audio_output_codes=audio_codes[:, -1:] if supports_audio_output else None,
            audio_output_codes_mask=audio_codes_mask[:, -1:] if supports_audio_output else None,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
        return self._update_sequences_and_generate_audio_codes(
            new_logits=text_logits,
            new_last_hidden_state=talker_last_hidden_state,
            sequences=sequences,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            is_complete=is_complete,
            pad_token_id=pad_token_id,
            force_audio_output=force_audio_output,
            force_text_output=force_text_output,
            do_sample=do_sample,
            logits_processor=logits_processor,
            is_generating_audio=is_generating_audio,
            ras_enabled=ras_enabled,
            ras_window_size=ras_window_size,
            ras_repetition_threshold=ras_repetition_threshold,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        pad_token_id: int = PAD.id,
        num_code_groups: int | None = None,
        force_audio_output: bool = False,
        force_text_output: bool = False,
        disable_eos_on_first_output: bool = True,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.8,
        disable_tqdm: bool = False,
        ras_enabled: bool = False,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        speaker_embeds: torch.Tensor | None = None,
        speaker_audio: torch.Tensor | None = None,
        speaker_audio_lengths: torch.Tensor | None = None,
    ) -> GenerateOutput:
        """Generate audio/text sequences with optional Repetition Aware Sampling (RAS).

        RAS is based on VALL-E 2 paper (https://arxiv.org/abs/2406.05370):
        - Refines nucleus sampling by accounting for token repetition in decoding history
        - Stabilizes decoding and circumvents infinite loop issues
        - When repetition ratio exceeds threshold, switches to random sampling

        Args:
            input_ids: Tokenized input sequence.
                Shape: [batch_size, seq_len]. Dtype: long.
            attention_mask: Mask indicating valid positions.
                Shape: [batch_size, seq_len]. Dtype: long.
            audio_input: Raw input audio waveform.
                Shape: [batch_size, num_samples]. Dtype: float.
            audio_output: Raw output audio waveform.
                Shape: [batch_size, num_samples]. Dtype: float.
            audio_input_lengths: Valid sample count per input audio.
                Shape: [batch_size]. Dtype: long.
            audio_output_lengths: Valid sample count per output audio.
                Shape: [batch_size]. Dtype: long.
            audio_output_codes: Pre-tokenized output audio codes.
                Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_input_embeds: Pre-computed audio input embeddings.
                Shape: [batch_size, num_frames, feature_dim]. Dtype: float.
            audio_input_embeds_mask: Mask for valid audio input positions.
                Shape: [batch_size, num_frames]. Dtype: bool.
            max_new_tokens: Maximum number of new tokens to generate.
            pad_token_id: Token ID used for padding.
            num_code_groups: Number of audio code groups to generate.
            force_audio_output: If True, always generate audio output tokens.
            force_text_output: If True, suppress audio output tokens.
            disable_eos_on_first_output: If True, prevent EOS on the first generated token.
            do_sample: If True, use nucleus sampling; otherwise use greedy decoding.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.
            top_p: Top-p (nucleus) filtering parameter.
            disable_tqdm: If True, suppress the progress bar.
            ras_enabled: Enable Repetition Aware Sampling.
            ras_window_size: Window size K for calculating repetition ratio.
            ras_repetition_threshold: Threshold for triggering random resampling.
            speaker_embeds: Pre-computed speaker embeddings. Mutually exclusive with speaker_audio.
                Shape: [batch_size, 1, hidden_size]. Dtype: float.
            speaker_audio: Raw speaker reference audio. The model's speaker encoder will compute
                embeddings internally. Mutually exclusive with speaker_embeds.
                Shape: [batch_size, num_samples]. Dtype: float.
            speaker_audio_lengths: Valid sample count per speaker audio element.
                Shape: [batch_size]. Dtype: long.

        Returns:
            GenerateOutput containing generated sequences, audio codes, and decoded audio.
        """
        if speaker_audio is not None and speaker_embeds is None:
            speaker_embeds = self._compute_speaker_embeds(speaker_audio, speaker_audio_lengths)

        if num_code_groups is None:
            num_code_groups = self.num_code_groups

        assert num_code_groups <= self.num_code_groups, (
            f"Expected `num_code_groups` to be at most `{self.num_code_groups}` but got `{num_code_groups}`."
        )
        if not bool(getattr(self, "supports_audio_output", True)):
            force_audio_output = False
            force_text_output = True

        logits_processor = LogitsProcessorList()
        if do_sample and temperature and temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(temperature=temperature))
        if do_sample and top_k and top_k > 0:
            logits_processor.append(TopKLogitsWarper(top_k=top_k))
        if do_sample and top_p and top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(top_p=top_p))

        (
            sequences,
            attention_mask,
            audio_codes,
            audio_codes_mask,
            is_complete,
            is_generating_audio,
            past_key_values,
        ) = self._generation_prefill(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input=audio_input,
            audio_output=audio_output,
            audio_input_lengths=audio_input_lengths,
            audio_output_lengths=audio_output_lengths,
            audio_output_codes=audio_output_codes,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=speaker_embeds,
            pad_token_id=pad_token_id,
            force_audio_output=force_audio_output,
            force_text_output=force_text_output,
            disable_eos_on_first_output=disable_eos_on_first_output,
            do_sample=do_sample,
            logits_processor=logits_processor,
            max_sequence_length=8 * (1 + (input_ids.shape[1] + max_new_tokens) // 8),
            ras_enabled=ras_enabled,
            ras_window_size=ras_window_size,
            ras_repetition_threshold=ras_repetition_threshold,
        )
        for _ in trange(max_new_tokens - 1, disable=disable_tqdm):
            if is_complete.all():
                break

            sequences, attention_mask, audio_codes, audio_codes_mask, is_complete, is_generating_audio = self._decoding_step(
                sequences=sequences,
                attention_mask=attention_mask,
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                is_complete=is_complete,
                past_key_values=past_key_values,
                pad_token_id=pad_token_id,
                force_audio_output=force_audio_output,
                force_text_output=force_text_output,
                is_generating_audio=is_generating_audio,
                do_sample=do_sample,
                logits_processor=logits_processor,
                ras_enabled=ras_enabled,
                ras_window_size=ras_window_size,
                ras_repetition_threshold=ras_repetition_threshold,
            )

        audio = None
        audio_lengths = None
        if not force_text_output and audio_codes_mask.any():
            contiguous_audio_sequences = pad_sequence(
                [seq[mask] for seq, mask in zip(audio_codes, audio_codes_mask, strict=True)],
                batch_first=True,
                padding_value=0,
            )
            # Realign delayed codes before decoding
            if self.max_delay > 0:
                contiguous_audio_sequences = undelay_audio_codes(self.delays, contiguous_audio_sequences, padding_value=0)
            audio = self.decode_audio(audio_codes=contiguous_audio_sequences).audio
            audio_lengths = (audio_codes_mask.float().sum(dim=1) * self.sampling_rate / self.frame_rate).floor().long()

        self.free_past_key_values(past_key_values)

        return {
            "sequences": sequences,
            "audio_codes": audio_codes,
            "audio_codes_mask": audio_codes_mask,
            "audio": audio,
            "audio_lengths": audio_lengths,
        }

    def _pad_audio_input(self, audio_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        sampling_rate = self.sampling_rate
        frame_rate: float = self.frame_rate
        assert (samples_per_frame := int(sampling_rate / frame_rate)) == sampling_rate / frame_rate, (
            f"Expected `sampling_rate / frame_rate` to be an integer but got `{sampling_rate / frame_rate}`."
        )

        if audio_input.shape[1] < samples_per_frame:
            audio_input = F.pad(audio_input, (0, samples_per_frame - audio_input.shape[1]))
            print(
                f"WARNING: Duplex decoding uses {samples_per_frame} samples per frame, "
                f"but {audio_input.shape[1]} samples were input. "
                "The input audio has been padded accordingly."
            )
        elif audio_input.shape[1] > samples_per_frame:
            audio_input = audio_input[:, :samples_per_frame]
            print(
                f"WARNING: Duplex decoding uses {samples_per_frame} samples per frame, "
                f"but {audio_input.shape[1]} samples were input. "
                "The input audio has been truncated accordingly."
            )

        audio_input_lengths = torch.tensor([audio_input.shape[1]], device=audio_input.device)
        return audio_input, audio_input_lengths

    @torch.inference_mode()
    def _update_duplex_sequences_and_generate_audio_codes(
        self,
        new_logits: torch.Tensor,
        new_last_hidden_state: torch.Tensor,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        eos_penalty: float = 0.0,
        sil_penalty: float = 0.0,
        bc_penalty: float = 0.0,
        defer_audio_code_generation: bool = False,
        machine_state: RaonMachineState | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
        torch.Tensor | None,
        RaonMachineState | None,
    ]:
        if new_logits.shape[0] != 1:
            raise NotImplementedError(f"Only batch size 1 is supported but got `{new_logits.shape[0]}`.")

        # Text prediction is always read from the token before [A].
        user_logits = new_logits[:, -2:-1, : self.vocab_size]
        # Apply pad penalty to reduce pad probability (encourage longer responses)
        if eos_penalty > 0:
            user_logits = user_logits.clone()
            user_logits[:, :, AUDIO_OUTPUT_PAD.id] -= eos_penalty
        if sil_penalty > 0 and getattr(self, "use_sil_token", False):
            user_logits = user_logits.clone()
            user_logits[:, :, self.duplex_sil_token_id] -= sil_penalty
        if (
            bc_penalty != 0
            and getattr(self, "use_backchannel_token", False)
            and machine_state is not None
            and machine_state.phase == RaonPhase.SIL
        ):
            user_logits = user_logits.clone()
            user_logits[:, :, self.duplex_bc_token_id] -= bc_penalty

        # Apply state-machine logit masking to enforce valid transitions.
        if self.use_duplex_end_pad:
            user_logits = self._apply_state_machine_logit_mask(user_logits, sequences, machine_state=machine_state)

        if do_sample:
            user_logits = logits_processor(input_ids=sequences, scores=user_logits[:, -1])  # type: ignore
            user_probs = F.softmax(user_logits, dim=-1, dtype=torch.float32)
            user_probs = user_probs.clamp_min(torch.finfo(user_probs.dtype).tiny)
            text_or_eos_id = torch.multinomial(user_probs, num_samples=1)
        else:
            text_or_eos_id = user_logits[:, -1:].argmax(dim=-1)

        predicted_token_id = int(text_or_eos_id.item())

        # Cross-validate text prediction (SIL vs non-SIL) against audio prediction
        # (audio_end vs non-audio_end) to prevent premature silence transitions.
        # Always generate audio codes so we can check for audio_end, then decide.
        if machine_state is not None:
            is_in_speech = machine_state.phase == RaonPhase.SPEECH
        else:
            is_in_speech = True
        text_predicts_sil = self.use_sil_token and predicted_token_id == self.duplex_sil_token_id

        deferred_hidden: torch.Tensor | None = None
        audio_end_predicted = False

        if is_in_speech and not defer_audio_code_generation:
            # Generate audio codes with audio_end enabled so we can cross-validate
            # the text prediction (SIL) against the audio prediction (audio_end).
            first_code_sampler = None
            if do_sample:
                first_code_sampler = self._make_audio_code_sampler(
                    sequences=sequences,
                    logits_processor=logits_processor,
                    audio_codes=audio_codes,
                    ras_enabled=False,
                    ras_window_size=40,
                    ras_repetition_threshold=0.1,
                )
            new_audio_codes = self.generate_audio_codes(
                talker_last_hidden_state=new_last_hidden_state[:, -1:],
                first_code_sampler=first_code_sampler,
                allow_audio_end=True,
            )
            new_audio_codes = new_audio_codes.clone()
            audio_end_predicted = bool((new_audio_codes[:, 0] == self.codebook_size).any())
            if audio_end_predicted:
                new_audio_codes[new_audio_codes[:, 0] == self.codebook_size, 0] = 0

            # Cross-validation: text and audio must agree on silence transition.
            if text_predicts_sil and not audio_end_predicted:
                # Text says SIL but audio says continue → override to PAD (keep speaking).
                predicted_token_id = AUDIO_OUTPUT_PAD.id
                text_predicts_sil = False
            elif not text_predicts_sil and audio_end_predicted:
                # Text says continue but audio says end → override to SIL (stop speaking).
                predicted_token_id = self.duplex_sil_token_id
                text_predicts_sil = True

        new_machine_state: RaonMachineState | None = None
        if machine_state is not None:
            new_machine_state, frame_tokens_list, emitted_audio = self._state_manager.transition(
                machine_state, predicted_token_id
            )
            input_ids = torch.tensor([frame_tokens_list], device=sequences.device)
        else:
            input_ids, emitted_audio = self._build_standard_mode_frame_input_ids(
                predicted_token_id,
                sequences.device,
            )

        if emitted_audio:
            if defer_audio_code_generation:
                deferred_hidden = new_last_hidden_state[:, -1:]
            elif is_in_speech:
                # Audio codes were already generated above for cross-validation.
                audio_codes = torch.cat((audio_codes, new_audio_codes[None]), dim=1)
                audio_codes_mask = torch.cat(
                    (
                        audio_codes_mask,
                        torch.tensor([[True]], device=audio_codes.device, dtype=torch.bool),
                    ),
                    dim=1,
                )
            else:
                # First frame from silence (onset) — generate fresh codes.
                first_code_sampler = None
                if do_sample:
                    first_code_sampler = self._make_audio_code_sampler(
                        sequences=sequences,
                        logits_processor=logits_processor,
                        audio_codes=audio_codes,
                        ras_enabled=False,
                        ras_window_size=40,
                        ras_repetition_threshold=0.1,
                    )
                new_audio_codes = self.generate_audio_codes(
                    talker_last_hidden_state=new_last_hidden_state[:, -1:],
                    first_code_sampler=first_code_sampler,
                    allow_audio_end=False,
                )
                new_audio_codes = new_audio_codes.clone()
                if (new_audio_codes[:, 0] == self.codebook_size).any():
                    new_audio_codes[new_audio_codes[:, 0] == self.codebook_size, 0] = 0
                audio_codes = torch.cat((audio_codes, new_audio_codes[None]), dim=1)
                audio_codes_mask = torch.cat(
                    (
                        audio_codes_mask,
                        torch.tensor([[True]], device=audio_codes.device, dtype=torch.bool),
                    ),
                    dim=1,
                )

        sequences = torch.cat((sequences, input_ids), dim=1)
        attention_mask = F.pad(attention_mask, (0, input_ids.shape[1]), value=1)
        return (
            input_ids,
            sequences,
            attention_mask,
            audio_codes,
            audio_codes_mask,
            emitted_audio,
            deferred_hidden,
            new_machine_state,
        )

    @torch.inference_mode()
    def duplex_decoding_step(
        self,
        state: RaonDecodingState,
        audio_input: torch.Tensor,
    ) -> tuple[RaonDecodingState, torch.Tensor]:
        """Run one duplex decoding step: encode user audio, predict tokens/codes, push codes, pull waveform.

        Args:
            state: Current duplex decoding state.
            audio_input: One frame of user audio. Shape: [1, num_samples_per_frame]. Dtype: float.

        Returns:
            Tuple of (updated_state, decoded_audio).
            decoded_audio: Decoded waveform for this frame. Shape: [1, num_samples_per_frame]. Dtype: float.
        """
        if state.sequences.shape[0] != 1:
            raise NotImplementedError(f"Only batch size 1 is supported but got `{state.sequences.shape[0]}`.")

        last_token = int(state.sequences[0, -1].item())
        valid_last_tokens = {AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id}
        if last_token not in valid_last_tokens:
            raise ValueError(f"Last token must be one of `{sorted(valid_last_tokens)}` but got `{state.sequences[0, -1]}`.")

        prev_audio_codes_length = state.audio_codes.shape[1]

        audio_input, audio_input_lengths = self._pad_audio_input(audio_input=audio_input)

        audio_input_embeds, audio_input_embeds_mask, audio_input_encoder_cache = (
            self._streaming_get_audio_input_embeds_voxtral(
                audio=audio_input,
                audio_lengths=audio_input_lengths,
                streaming_state=state.audio_input_encoder_cache,
            )
        )

        # Check whether the trailing frame contains text or EPAD.
        if state.machine_state is not None:
            num_input_tokens = state.machine_state.num_input_tokens
        else:
            context_token = self._get_standard_mode_text_context_token(state.sequences)
            num_input_tokens = 3 if context_token is not None else 2
        context_token = self._get_standard_mode_text_context_token(state.sequences)

        full_position_ids = state.attention_mask.cumsum(dim=1) - 1
        seq_len = state.attention_mask.sum(dim=1)[0].item()
        cache_position = torch.arange(
            seq_len - num_input_tokens, seq_len, device=state.sequences.device
        )

        step_audio_codes = state.audio_codes[:, -1:] if state.audio_codes.shape[1] > 0 else None
        step_audio_codes_mask = state.audio_codes_mask[:, -1:] if state.audio_codes.shape[1] > 0 else None

        # Restore per-session talker state before inference.
        self._talker_past_key_values = state.talker_past_key_values
        self._talker_attention_mask = state.talker_attention_mask

        talker_last_hidden_state, text_logits = self.inference_forward(
            input_ids=state.sequences[:, -num_input_tokens:],
            attention_mask=None,
            position_ids=full_position_ids[:, -num_input_tokens:],
            audio_output_codes=step_audio_codes,
            audio_output_codes_mask=step_audio_codes_mask,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=state.speaker_embeds,
            use_cache=True,
            past_key_values=state.past_key_values,
            cache_position=cache_position,
        )

        # Force SIL for the remaining listen-first warmup frames.
        if state.forced_sil_remaining > 0 and self.use_sil_token:
            forced_logits = torch.full_like(text_logits, fill_value=-1e9)
            forced_logits[:, -2, self.duplex_sil_token_id] = 0.0  # Force SIL
            text_logits = forced_logits

        # Standard mode (with optional EPAD support)
        _, sequences, attention_mask, audio_codes, audio_codes_mask, emitted_audio, _, new_machine_state = (
            self._update_duplex_sequences_and_generate_audio_codes(
                new_logits=text_logits,
                new_last_hidden_state=talker_last_hidden_state,
                sequences=state.sequences,
                attention_mask=state.attention_mask,
                audio_codes=state.audio_codes,
                audio_codes_mask=state.audio_codes_mask,
                do_sample=state.do_sample,
                logits_processor=state.logits_processor,
                eos_penalty=state.eos_penalty,
                sil_penalty=state.sil_penalty,
                bc_penalty=state.bc_penalty,
                machine_state=state.machine_state,
            )
        )

        expected_audio_codes = prev_audio_codes_length + int(emitted_audio)
        valid_trailing_tokens = {AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id}
        assert int(sequences[0, -1].item()) in valid_trailing_tokens, (
            f"Last token must be one of `{sorted(valid_trailing_tokens)}` but got `{sequences[0, -1]}`."
        )
        assert audio_codes.shape[1] == expected_audio_codes, (
            f"Expected `{expected_audio_codes}` audio codes but got `{audio_codes.shape[1]}`."
        )

        # Handle acoustic delay if configured.
        # When no audio is emitted (silence step), clear the semantic buffer so the
        # post-silence frame is treated as a fresh "first frame" rather than combining
        # stale semantic state with fresh acoustic codes.
        # Push zero codes during silence to keep the decoder conv state warm,
        # avoiding the need for a hard reset at speech onset.
        new_semantic_buffer = None
        if not emitted_audio:
            # Push true silence codes to keep decoder conv state warm.
            silence_codes = self.get_silence_codes(sequences.device)
            self.push_audio_codes(audio_codes=silence_codes, stream_id=state.audio_decoder_stream_id)
            decoded_audio = self.pull_audio(state.audio_decoder_stream_id)
            if decoded_audio.device != audio_input.device:
                decoded_audio = decoded_audio.to(audio_input.device)
        elif self.max_delay > 0:
            # With delay: semantic codes are predicted for current frame,
            # acoustic codes are predicted for previous frame
            current_codes = audio_codes[0, -1]  # [num_code_groups]
            semantic_code = current_codes[0:1]  # Current frame semantic
            acoustic_codes = current_codes[1:]  # Previous frame acoustic

            if state.semantic_buffer is None:
                # First frame: buffer semantic, output silence (or skip)
                new_semantic_buffer = semantic_code
                # Output zeros for first frame (no valid acoustic yet)
                output_codes = torch.zeros_like(current_codes)
                output_codes[0] = semantic_code[0]
            else:
                # Combine: previous semantic + current acoustic prediction
                output_codes = torch.cat([state.semantic_buffer, acoustic_codes], dim=0)
                new_semantic_buffer = semantic_code

            self.push_audio_codes(audio_codes=output_codes, stream_id=state.audio_decoder_stream_id)
        else:
            # No delay: push codes directly
            self.push_audio_codes(audio_codes=audio_codes[0, -1], stream_id=state.audio_decoder_stream_id)

        if emitted_audio:
            decoded_audio = self.pull_audio(state.audio_decoder_stream_id)
            if decoded_audio.device != audio_input.device:
                decoded_audio = decoded_audio.to(audio_input.device)

        updated_state = RaonDecodingState(
            sequences=sequences,
            attention_mask=attention_mask,
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            past_key_values=state.past_key_values,
            audio_input_encoder_cache=audio_input_encoder_cache,
            audio_decoder_stream_id=state.audio_decoder_stream_id,
            do_sample=state.do_sample,
            logits_processor=state.logits_processor,
            num_code_groups=state.num_code_groups,
            semantic_buffer=new_semantic_buffer,
            eos_penalty=state.eos_penalty,
            sil_penalty=state.sil_penalty,
            bc_penalty=state.bc_penalty,
            speaker_embeds=state.speaker_embeds,
            machine_state=new_machine_state if new_machine_state is not None else state.machine_state,
            talker_past_key_values=self._talker_past_key_values,
            talker_attention_mask=self._talker_attention_mask,
            forced_sil_remaining=max(0, state.forced_sil_remaining - 1),
        )

        return updated_state, decoded_audio

    def init_duplex_decoding_state(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.8,
        max_sequence_length: int = 8192,
        prev_state: RaonDecodingState | None = None,
        eos_penalty: float = 0.0,
        sil_penalty: float = 0.0,
        bc_penalty: float = 0.0,
        speaker_embeds: torch.Tensor | None = None,
        speak_first: bool = False,
    ) -> RaonDecodingState:
        """Initialize duplex decoding state and run the first frame to obtain [U][A] prompt.

        Args:
            sequences: Initial text tokens (system prompt). Shape: [1, seq_length]. Dtype: long.
            attention_mask: Mask for valid positions. Shape: [1, seq_length]. Dtype: long.
            do_sample: Whether to sample (vs. greedy).
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p filtering.
            max_sequence_length: Max sequence length for KV cache.
            prev_state: Previous state to reuse caches (e.g. from warmup).
            eos_penalty: Penalty to subtract from pad/eos logit to encourage longer output.

        Returns:
            RaonDecodingState ready for duplex_decoding_step.
        """
        warning_message = get_talker_state_warning(
            has_talker_cache=self._talker_past_key_values is not None,
            has_talker_mask=self._talker_attention_mask is not None,
            is_reuse_init=prev_state is not None,
        )
        if warning_message is not None:
            logger.warning(warning_message)
        self.start_concurrent_audio_decoder()

        # Auto-append speaker token so update_inputs_embeds can inject the embedding.
        if speaker_embeds is not None:
            speaker_token_id = getattr(self.get_model(), "speaker_token_id", None)
            if speaker_token_id is None:
                raise ValueError("speaker_embeds is provided but model.speaker_token_id is None.")
            if not (sequences == speaker_token_id).any():
                speaker_token = torch.full(
                    (sequences.shape[0], 1),
                    fill_value=speaker_token_id,
                    dtype=sequences.dtype,
                    device=sequences.device,
                )
                sequences = torch.cat((sequences, speaker_token), dim=1)
                if attention_mask is not None:
                    attention_mask = F.pad(attention_mask, (0, 1), value=1)

        if sequences.shape[0] != 1:
            raise NotImplementedError(f"Only batch size 1 is supported but got `{sequences.shape[0]}`.")

        if self.max_delay > 1:
            raise NotImplementedError(
                f"Duplex decoding only supports acoustic_delay of 0 or 1, "
                f"got max_delay={self.max_delay}. semantic_buffer assumes single-step delay."
            )
        if self.max_delay > 0 and self.delays[0] != 0:
            raise ValueError(
                f"Semantic codebook (index 0) must have delay=0 for duplex decoding, "
                f"got delays[0]={self.delays[0]}. The semantic_buffer logic assumes "
                f"delays=[0, N, N, ..., N]."
            )

        if attention_mask is None:
            attention_mask = torch.ones_like(sequences)

        audio_special_token_ids = {
            AUDIO_INPUT_PLACEHOLDER.id,
            AUDIO_OUTPUT_PLACEHOLDER.id,
            AUDIO_START.id,
            AUDIO_END.id,
            AUDIO_OUTPUT_PAD.id,
            AUDIO_OUTPUT_END_PAD.id,
        }
        if self.use_sil_token:
            audio_special_token_ids.add(self.duplex_sil_token_id)
        if self.use_backchannel_token:
            audio_special_token_ids.add(self.duplex_bc_token_id)
        _audio_ids = torch.tensor(sorted(audio_special_token_ids), device=sequences.device)
        assert not torch.isin(sequences, _audio_ids).any() and (attention_mask == 1).all(), (
            "All `sequences` must be text tokens and all `attention_mask` values must be 1. "
            f"`{sequences=}`, `{attention_mask=}`."
        )

        logits_processor = LogitsProcessorList()
        if do_sample and temperature and temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(temperature=temperature))
        if do_sample and top_k and top_k > 0:
            logits_processor.append(TopKLogitsWarper(top_k=top_k))
        if do_sample and top_p and top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(top_p=top_p))

        past_key_values: TextModelCache | None = None
        audio_input_encoder_cache: AudioInputEncoderCache | None = None
        audio_decoder_stream_id: int | None = None

        try:
            if prev_state is not None:
                past_key_values = self.init_past_key_values(
                    batch_size=1,
                    max_sequence_length=max_sequence_length,
                    prev_cache=prev_state.past_key_values,
                )
                audio_input_encoder_cache = self.init_audio_encoder_cache(prev_cache=prev_state.audio_input_encoder_cache)
                self._drain_audio_decoding_queue(stream_id=prev_state.audio_decoder_stream_id)
                self._destroy_audio_decoder_stream(prev_state.audio_decoder_stream_id)
            else:
                past_key_values = self.init_past_key_values(batch_size=1, max_sequence_length=max_sequence_length)
                audio_input_encoder_cache = self.init_audio_encoder_cache()

            audio_decoder_stream_id = self.create_audio_decoder_stream()

            audio_codes = torch.zeros(1, 0, self.num_code_groups, dtype=torch.long, device=sequences.device)
            audio_codes_mask = torch.zeros(1, 0, dtype=torch.bool, device=sequences.device)

            # Initial machine state: standard init starts in SIL.
            init_machine_state = RaonMachineState(
                phase=RaonPhase.SIL,
                last_frame_tokens=[IM_START.id, AUDIO_START.id],
            )

            input_ids = torch.cat(
                [
                    sequences,
                    torch.tensor(
                        [[IM_START.id, AUDIO_START.id]],
                        device=sequences.device,
                    ),
                ],
                dim=1,
            )
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)

            talker_last_hidden_state, text_logits = self.inference_forward(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=position_ids,
                speaker_embeds=speaker_embeds,
                use_cache=True,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

            # The v2 duplex model no longer encodes speak/listen mode in the system
            # prompt text. Force the first [U] prediction explicitly:
            # speak-first -> EPAD, listen-first -> SIL.
            forced_initial_prediction_id = self._state_manager.initial_forced_prediction_id(speak_first)
            if forced_initial_prediction_id is not None:
                forced_logits = torch.full_like(text_logits, fill_value=-1e9)
                forced_logits[:, -2, forced_initial_prediction_id] = 0.0
                text_logits = forced_logits

            _, sequences, attention_mask, audio_codes, audio_codes_mask, emitted_audio, _, init_machine_state = (
                self._update_duplex_sequences_and_generate_audio_codes(
                    new_logits=text_logits,
                    new_last_hidden_state=talker_last_hidden_state,
                    sequences=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    audio_codes=audio_codes,
                    audio_codes_mask=audio_codes_mask,
                    do_sample=do_sample,
                    logits_processor=logits_processor,
                    eos_penalty=eos_penalty,
                    sil_penalty=sil_penalty,
                    bc_penalty=bc_penalty,
                    machine_state=init_machine_state,
                )
            )

            valid_last_tokens = {AUDIO_OUTPUT_PLACEHOLDER.id, AUDIO_START.id}
            assert int(sequences[0, -1].item()) in valid_last_tokens, (
                f"Last token must be one of `{sorted(valid_last_tokens)}` but got `{sequences[0, -1]}`."
            )
            assert audio_codes.shape[1] == int(emitted_audio), (
                f"Expected `{int(emitted_audio)}` audio codes but got `{audio_codes.shape[1]}`."
            )
            # When acoustic delay is active, the first frame is a placeholder:
            # only the semantic code (CB0) is valid; acoustic codes (CB1-7) are zeros
            # because no previous frame exists to provide delayed acoustic predictions.
            initial_semantic_buffer = None
            if not emitted_audio:
                initial_semantic_buffer = None
            elif self.max_delay > 0:
                first_codes = audio_codes[0, -1]
                semantic_code = first_codes[0:1]
                output_codes = torch.zeros_like(first_codes)
                output_codes[0] = semantic_code[0]
                initial_semantic_buffer = semantic_code
                self.push_audio_codes(audio_codes=output_codes, stream_id=audio_decoder_stream_id)
            else:
                self.push_audio_codes(audio_codes=audio_codes[0, -1], stream_id=audio_decoder_stream_id)

            # When listen-first, the first frame is already forced to SIL above.
            # Force one additional frame so the model sees at least 2 SIL frames
            # before it is allowed to predict freely.
            forced_sil_remaining = 1 if (not speak_first and self.use_sil_token) else 0

            state = RaonDecodingState(
                sequences=sequences,
                attention_mask=attention_mask,
                audio_codes=audio_codes,
                audio_codes_mask=audio_codes_mask,
                past_key_values=past_key_values,
                audio_input_encoder_cache=audio_input_encoder_cache,
                audio_decoder_stream_id=audio_decoder_stream_id,
                do_sample=do_sample,
                logits_processor=logits_processor,
                num_code_groups=self.num_code_groups,
                semantic_buffer=initial_semantic_buffer,
                eos_penalty=eos_penalty,
                sil_penalty=sil_penalty,
                bc_penalty=bc_penalty,
                speaker_embeds=speaker_embeds,
                machine_state=init_machine_state,
                talker_past_key_values=self._talker_past_key_values,
                talker_attention_mask=self._talker_attention_mask,
                forced_sil_remaining=forced_sil_remaining,
            )
            return state
        except Exception:
            cleanup_failed_duplex_init(
                stream_id=audio_decoder_stream_id,
                past_key_values=past_key_values,
                audio_input_encoder_cache=audio_input_encoder_cache,
                drain_audio_decoding_queue=self._drain_audio_decoding_queue,
                destroy_audio_decoder_stream=self._destroy_audio_decoder_stream,
                free_past_key_values=self.free_past_key_values,
            )
            raise

    @torch.inference_mode()
    def duplex_generate_with_fixed_input(
        self,
        sequences: torch.Tensor,
        audio_input: torch.Tensor,
        speaker_embeds: torch.Tensor | None = None,
        prev_state: RaonDecodingState | None = None,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.8,
        disable_tqdm: bool = False,
        sample_id: str | None = None,
        eos_penalty: float = 0.0,
        sequence_viz_path: str | None = None,
        speak_first: bool = False,
    ) -> torch.Tensor:
        """Full-duplex generate: process fixed user audio frame-by-frame and return assistant audio.

        Args:
            sequences: Initial text tokens (system prompt). Shape: [1, seq_length]. Dtype: long.
            audio_input: Full user audio. Shape: [1, num_samples] or [num_samples]. Dtype: float.
            prev_state: Previous state from warmup (e.g. compile_audio_modules).
            do_sample: Whether to sample (vs. greedy).
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p filtering.
            disable_tqdm: Disable progress bar.
            sample_id: Reserved for backward-compatible callers; ignored by the runtime.
            eos_penalty: Penalty to subtract from pad/eos logit.

        Returns:
            Generated assistant audio waveform. Shape: [1, num_output_samples]. Dtype: float.
        """
        _ = sample_id, sequence_viz_path

        assert (samples_per_frame := int(self.sampling_rate / self.frame_rate)) == self.sampling_rate / self.frame_rate, (
            f"Expected `sampling_rate / frame_rate` to be an integer but got {self.sampling_rate / self.frame_rate}."
        )

        audio_input_length = audio_input.shape[-1]
        if audio_input.ndim == 1:
            audio_input = audio_input[None]

        assert audio_input.shape == (
            1,
            audio_input_length,
        ), f"Expected audio shape `(1, {audio_input_length})` but got `{audio_input.shape}`."

        assert audio_input_length >= samples_per_frame, (
            f"Expected `audio_input` to have at least {samples_per_frame} samples but got {audio_input_length}."
        )

        state = self.init_duplex_decoding_state(
            sequences=sequences,
            attention_mask=torch.ones_like(sequences),
            do_sample=do_sample,
            prev_state=prev_state,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_penalty=eos_penalty,
            speaker_embeds=speaker_embeds,
            speak_first=speak_first,
        )

        audio_output_frames: list[torch.Tensor] = []

        for i in trange(
            0,
            audio_input_length - samples_per_frame + 1,
            samples_per_frame,
            mininterval=0,
            desc="Duplex Generation",
            disable=disable_tqdm,
        ):
            audio_input_frame = audio_input[:, i : i + samples_per_frame]
            state, audio_output_frame = self.duplex_decoding_step(state=state, audio_input=audio_input_frame)
            audio_output_frames.append(audio_output_frame)

        self.free_duplex_decoding_state(state)
        return torch.cat(audio_output_frames, dim=1)

DuplexGenerateResult = RaonGenerateResult
DuplexDecodingState = RaonDecodingState

__all__ = [
    "DuplexDecodingState",
    "DuplexGenerateResult",
    "GenerateOutput",
    "RaonDecodingState",
    "RaonGenerateResult",
    "RaonInferenceModel",
    "extract_predicted_text",
]
