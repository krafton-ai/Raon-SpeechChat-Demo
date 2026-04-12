"""Wrapper that adapts the Voxtral Realtime encoder for the duplex model pipeline.

Takes raw audio waveforms, computes mel spectrograms, runs the Voxtral encoder
and multi-modal projector, and returns ``CausalAudioEncoderOutput`` compatible
with the duplex model's audio encoder interface.
"""

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from transformers import WhisperFeatureExtractor
from transformers.cache_utils import DynamicCache

from ..audio_encoder.output import CausalAudioEncoderOutput

from .model import (
    SlidingWindowVoxtralKVCache,
    StaticVoxtralConv1dPaddingCache,
    VoxtralRealtimeCausalConv1d,
    VoxtralRealtimeConv1dPaddingCache,
    VoxtralRealtimeEncoder,
    VoxtralRealtimeEncoderConfig,
    VoxtralRealtimeMultiModalProjector,
    _make_fixed_sliding_window_mask,
)

DOWNSAMPLE_FACTOR = 4
ENCODER_STRIDE = 2
MEL_FRAMES_PER_ADAPTER_TOKEN = DOWNSAMPLE_FACTOR * ENCODER_STRIDE  # 8
logger = logging.getLogger(__name__)


@dataclass
class VoxtralStreamingState:
    """Streaming state for the Voxtral encoder.

    Holds the KV cache, conv1d padding cache, and mel feature buffer needed
    to run the encoder incrementally at 12.5Hz (80ms per adapter token).
    """

    kv_cache: DynamicCache | SlidingWindowVoxtralKVCache
    padding_cache: VoxtralRealtimeConv1dPaddingCache
    stft_cache: torch.Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.float32), repr=False)
    running_max: torch.Tensor = field(
        default_factory=lambda: torch.tensor(float("-inf"), dtype=torch.float32), repr=False
    )
    mel_buffer: torch.Tensor | None = field(default=None, repr=False)

    def reset(self) -> None:
        """Reset all caches for session reuse without reallocating tensors.

        If this state was acquired from a ``VoxtralWrapper`` cache pool, the
        pool slot is released so it can be acquired by the next session.
        """
        if hasattr(self.kv_cache, "reset"):
            self.kv_cache.reset()
        if hasattr(self.padding_cache, "reset"):
            self.padding_cache.reset()
        self.stft_cache = self.stft_cache.new_zeros((0,))
        self.running_max = self.running_max.new_full((), float("-inf"))
        self.mel_buffer = None
        # Release pool slot if acquired from a pool.
        pool_idx = getattr(self, "_pool_idx", None)
        pool_owner = getattr(self, "_pool_owner", None)
        if pool_idx is not None and pool_owner is not None:
            if pool_idx not in pool_owner._cache_available:
                pool_owner._cache_available.append(pool_idx)


def _load_encoder_and_projector_state_dicts(
    pretrained_model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Load ``audio_tower`` and ``multi_modal_projector`` weights from safetensors.

    Handles both sharded (``model.safetensors.index.json``) and single-file
    checkpoints, and works with both local directories and HuggingFace Hub IDs.

    Args:
        pretrained_model_name_or_path: HuggingFace model ID or local path.
        dtype: Target dtype for the loaded tensors.

    Returns:
        Tuple of (encoder_state_dict, projector_state_dict) with prefixes
        stripped (e.g. ``audio_tower.layers.0.`` becomes ``layers.0.``).
    """
    import json
    import os

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    is_local = os.path.isdir(pretrained_model_name_or_path)

    def resolve(filename: str) -> str:
        if is_local:
            return os.path.join(pretrained_model_name_or_path, filename)
        return hf_hub_download(repo_id=pretrained_model_name_or_path, filename=filename)

    weight_map: dict[str, str] | None = None
    try:
        index_path = resolve("model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        shard_files = sorted(set(weight_map.values()))  # type: ignore
    except Exception:
        shard_files = ["model.safetensors"]

    encoder_prefix = "audio_tower."
    projector_prefix = "multi_modal_projector."
    encoder_state_dict: dict[str, torch.Tensor] = {}
    projector_state_dict: dict[str, torch.Tensor] = {}

    for shard_name in shard_files:
        if weight_map is not None:
            shard_keys = [k for k, v in weight_map.items() if v == shard_name]
            has_relevant = any(k.startswith(encoder_prefix) or k.startswith(projector_prefix) for k in shard_keys)
            if not has_relevant:
                continue

        shard_path = resolve(shard_name)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(encoder_prefix):
                    stripped = key[len(encoder_prefix) :]
                    encoder_state_dict[stripped] = f.get_tensor(key).to(dtype)
                elif key.startswith(projector_prefix):
                    stripped = key[len(projector_prefix) :]
                    projector_state_dict[stripped] = f.get_tensor(key).to(dtype)

    return encoder_state_dict, projector_state_dict


def _collect_conv_layer_specs(
    encoder: VoxtralRealtimeEncoder,
) -> list[tuple[str, int, int]]:
    """Walk the encoder's causal conv1d layers and return specs for static allocation.

    Returns:
        List of ``(cache_key, in_channels, left_pad)`` tuples, one per causal
        conv1d layer in the encoder's embedder.
    """
    specs: list[tuple[str, int, int]] = []
    for module in encoder.modules():
        if isinstance(module, VoxtralRealtimeCausalConv1d):
            specs.append((module.cache_key, module.in_channels, module.left_pad))
    return specs


class VoxtralWrapper(nn.Module):
    """Wrapper that runs the Voxtral Realtime encoder on raw audio waveforms.

    Handles resampling, mel feature extraction, encoder + projector forward,
    and streaming state management.
    """

    config: VoxtralRealtimeEncoderConfig

    def __init__(
        self,
        config: VoxtralRealtimeEncoderConfig,
        feature_extractor: WhisperFeatureExtractor,
        encoder: VoxtralRealtimeEncoder,
        projector: VoxtralRealtimeMultiModalProjector | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.projector = projector
        self.register_buffer(
            "_mel_filters",
            torch.as_tensor(np.array(feature_extractor.mel_filters), dtype=torch.float32).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "_stft_window",
            torch.hann_window(feature_extractor.n_fft, periodic=True, dtype=torch.float32),
            persistent=False,
        )

        self._use_static_cache = False
        self._cache_pool: list[tuple[SlidingWindowVoxtralKVCache, StaticVoxtralConv1dPaddingCache]] | None = None
        self._cache_available: list[int] = []
        self._compiled_transformer: object | None = None
        self._streaming_calls = 0
        self._streaming_resample_calls = 0
        self._streaming_zero_feature_calls = 0
        self._streaming_zero_adapter_token_calls = 0
        self._streaming_compiled_transformer_calls = 0
        self._streaming_eager_transformer_calls = 0
        self._streaming_emitted_adapter_tokens = 0
        self.input_sample_rate = 24000
        self.encoder_sample_rate: int = feature_extractor.sampling_rate
        self.frame_rate = 12.5

        if config.skip_projector:
            self.hidden_size = config.hidden_size * config.downsample_factor
        else:
            self.hidden_size = config.projector_output_size or config.hidden_size

        self._min_encoder_samples: int = feature_extractor.n_fft
        self.config.sampling_rate = self.input_sample_rate

    @classmethod
    def from_config(
        cls,
        config: VoxtralRealtimeEncoderConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "VoxtralWrapper":
        """Create a VoxtralWrapper with randomly-initialized weights.

        Args:
            config: Encoder config specifying architecture dimensions.
            dtype: Parameter dtype.

        Returns:
            Initialized ``VoxtralWrapper`` with random weights.
        """
        feature_extractor = WhisperFeatureExtractor(
            feature_size=config.num_mel_bins,
            sampling_rate=16000,
        )
        encoder = VoxtralRealtimeEncoder(config)
        encoder.to(dtype)  # type: ignore
        projector: VoxtralRealtimeMultiModalProjector | None = None
        if not config.skip_projector:
            projector = VoxtralRealtimeMultiModalProjector(config)
            projector.to(dtype)  # type: ignore
        return cls(config=config, feature_extractor=feature_extractor, encoder=encoder, projector=projector)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "mistralai/Voxtral-Mini-4B-Realtime-2602",
        config: VoxtralRealtimeEncoderConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "VoxtralWrapper":
        """Load a VoxtralWrapper from a pretrained HuggingFace checkpoint.

        Reads the config directly from ``config.json`` (no ``AutoConfig``
        dependency on the ``voxtral_realtime`` model type) and loads only the
        ``audio_tower.*`` and ``multi_modal_projector.*`` weights from the
        safetensors shards.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path.
            config: Optional override config. If None, derived from checkpoint.
            dtype: Parameter dtype.

        Returns:
            Initialized ``VoxtralWrapper`` with pretrained encoder and projector.
        """
        if config is None:
            config = VoxtralRealtimeEncoderConfig.from_pretrained(pretrained_model_name_or_path)

        wrapper = cls.from_config(config, dtype=dtype)

        encoder_state_dict, projector_state_dict = _load_encoder_and_projector_state_dicts(
            pretrained_model_name_or_path, dtype=dtype
        )

        wrapper.encoder.load_state_dict(encoder_state_dict, strict=False)
        if wrapper.projector is not None:
            wrapper.projector.load_state_dict(projector_state_dict)

        return wrapper

    @property
    def output_embedding_scale(self) -> float:
        """Return the scalar applied after the downsampling projector."""
        return self.config.output_embedding_scale

    @property
    def device(self) -> torch.device:
        """Return the device of the encoder parameters."""
        return next(self.encoder.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the encoder parameters."""
        return next(self.encoder.parameters()).dtype

    def compute_expected_output_length(self, num_samples: int) -> int:
        """Compute the expected number of output frames for a given number of audio samples.

        Args:
            num_samples: Number of input audio samples at ``input_sample_rate``.

        Returns:
            Expected number of adapter output frames at 12.5Hz.
        """
        samples_per_frame = int(self.input_sample_rate / self.frame_rate)
        return math.ceil(num_samples / samples_per_frame)

    def get_streaming_runtime_stats(self) -> dict[str, int | bool]:
        """Return lightweight counters for the streaming encoder path."""
        return {
            "calls": self._streaming_calls,
            "resample_calls": self._streaming_resample_calls,
            "zero_feature_calls": self._streaming_zero_feature_calls,
            "zero_adapter_token_calls": self._streaming_zero_adapter_token_calls,
            "compiled_transformer_calls": self._streaming_compiled_transformer_calls,
            "eager_transformer_calls": self._streaming_eager_transformer_calls,
            "emitted_adapter_tokens": self._streaming_emitted_adapter_tokens,
            "compiled_transformer_enabled": self._compiled_transformer is not None,
            "static_cache_enabled": self._use_static_cache,
        }

    def prepare_streaming_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize one streaming chunk before it reaches the encoder core.

        The eager outer wrapper owns channel folding, dtype/device normalization,
        and optional resampling so the compiled transformer core only sees stable
        mel chunks and cache tensors.
        """
        if audio.ndim == 3:
            if audio.shape[1] == 2:
                audio = audio.mean(dim=1, keepdim=True)
            if audio.shape[1] != 1:
                raise ValueError(f"Expected mono or stereo streaming audio, got shape {tuple(audio.shape)}.")
            audio = audio.squeeze(1)
        elif audio.ndim != 2:
            raise ValueError(f"Expected streaming audio rank 2 or 3, got shape {tuple(audio.shape)}.")

        if audio.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for streaming audio, got shape {tuple(audio.shape)}.")

        audio = audio.to(device=self.device, dtype=torch.float32).contiguous()

        if self.input_sample_rate != self.encoder_sample_rate:
            self._streaming_resample_calls += 1
            audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=self.input_sample_rate,
                new_freq=self.encoder_sample_rate,
            ).contiguous()

        return audio

    def _extract_features(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract Whisper mel features from raw audio waveforms.

        Uses the standard centered STFT from the ``WhisperFeatureExtractor``.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - batched_features: Padded mel spectrogram.
                    Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
                - audio_feature_lengths: Per-sample mel frame counts.
                    Shape: [batch_size]. Dtype: long.
        """
        if audio.shape[1] == 2:
            audio = audio.mean(dim=1, keepdim=True)
        audio = audio.squeeze(1)

        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long)
            audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

        if self.input_sample_rate != self.encoder_sample_rate:
            audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=self.input_sample_rate,
                new_freq=self.encoder_sample_rate,
            )
            if audio_lengths is not None:
                audio_lengths = torch.floor(audio_lengths.float() * self.encoder_sample_rate / self.input_sample_rate).to(
                    torch.long
                )
                audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

        n_fft: int = self.feature_extractor.n_fft
        hop_length: int = self.feature_extractor.hop_length

        if audio_lengths is None:
            audio_lengths = torch.full((audio.shape[0],), audio.shape[-1], dtype=torch.long, device=audio.device)

        effective_lengths = torch.maximum(audio_lengths, torch.full_like(audio_lengths, n_fft))
        target_len = int(effective_lengths.max().item())

        if audio.shape[-1] < target_len:
            audio = F.pad(audio, (0, target_len - audio.shape[-1]))
        elif audio.shape[-1] > target_len:
            audio = audio[:, :target_len]

        sample_indices = torch.arange(target_len, device=audio.device).unsqueeze(0)
        sample_mask = sample_indices < effective_lengths.unsqueeze(1)
        waveform = audio.to(device=self.device, dtype=torch.float32) * sample_mask.to(dtype=torch.float32)

        window = self._stft_window
        stft = torch.stft(waveform, n_fft, hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        mel_filters = self._mel_filters
        mel_spec = mel_filters.T @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_val - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        feature_attention_mask = sample_mask[:, ::hop_length]
        if target_len % hop_length != 0:
            feature_attention_mask = feature_attention_mask[:, :-1]

        audio_feature_lengths = feature_attention_mask.sum(dim=1).to(dtype=torch.long, device=self.device)
        batched_features = log_spec.to(device=self.device, dtype=self.dtype)
        return batched_features, audio_feature_lengths

    def _extract_features_causal(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract mel features using a left-padded causal STFT.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths.
                Shape: [batch_size]. Dtype: long.

        Returns:
            A tuple of:
                - batched_features: Padded mel spectrogram.
                    Shape: [batch_size, num_mel_bins, max_frames]. Dtype: float.
                - audio_feature_lengths: Per-sample mel frame counts.
                    Shape: [batch_size]. Dtype: long.
        """
        if audio.shape[1] == 2:
            audio = audio.mean(dim=1, keepdim=True)
        audio = audio.squeeze(1)

        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(device=audio.device, dtype=torch.long)
            audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

        if self.input_sample_rate != self.encoder_sample_rate:
            audio = torchaudio.functional.resample(
                waveform=audio,
                orig_freq=self.input_sample_rate,
                new_freq=self.encoder_sample_rate,
            )
            if audio_lengths is not None:
                audio_lengths = torch.floor(audio_lengths.float() * self.encoder_sample_rate / self.input_sample_rate).to(
                    torch.long
                )
                audio_lengths = audio_lengths.clamp(min=0, max=audio.shape[-1])

        if audio_lengths is None:
            if audio.shape[-1] < self._min_encoder_samples:
                audio = F.pad(audio, (0, self._min_encoder_samples - audio.shape[-1]))
            audio_lengths = torch.full((audio.shape[0],), audio.shape[-1], dtype=torch.long, device=audio.device)

        n_fft: int = self.feature_extractor.n_fft
        hop_length: int = self.feature_extractor.hop_length
        mel_filters = self._mel_filters
        window = self._stft_window

        all_log_specs: list[torch.Tensor] = []
        frame_counts: list[int] = []

        for waveform, length in zip(audio, audio_lengths, strict=True):
            waveform_f32 = waveform[: int(length.item())].to(device=self.device, dtype=torch.float32)
            if waveform_f32.numel() < n_fft:
                waveform_f32 = F.pad(waveform_f32, (0, n_fft - waveform_f32.numel()))

            padded = F.pad(waveform_f32, (n_fft // 2, 0))
            stft = torch.stft(padded, n_fft, hop_length, window=window, center=False, return_complex=True)
            magnitudes = stft[..., :-1].abs() ** 2
            mel_spec = mel_filters.T @ magnitudes
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()

            per_frame_max = log_spec.max(dim=0, keepdim=True)[0]
            running_max = torch.cummax(per_frame_max, dim=1)[0]
            log_spec = torch.maximum(log_spec, running_max.expand_as(log_spec) - 8.0)
            log_spec = (log_spec + 4.0) / 4.0

            all_log_specs.append(log_spec)
            frame_counts.append(log_spec.shape[1])

        batched_features = nn.utils.rnn.pad_sequence([s.T for s in all_log_specs], batch_first=True).permute(0, 2, 1)
        batched_features = batched_features.to(device=self.device, dtype=self.dtype)
        audio_feature_lengths = torch.tensor(frame_counts, dtype=torch.long, device=self.device)
        return batched_features, audio_feature_lengths

    def _extract_features_causal_streaming(
        self,
        audio: torch.Tensor,
        stft_cache: torch.Tensor,
        running_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract causal mel features for a single streaming chunk.

        Args:
            audio: Raw audio chunk (mono, at encoder sample rate).
                Shape: [1, num_samples]. Dtype: float.
            stft_cache: Leftover waveform from previous chunk.
                Shape: [num_leftover_samples]. Dtype: float. Empty on first call.
            running_max: Running maximum of per-frame log-mel maxima.

        Returns:
            Tuple of (packed_features, audio_feature_lengths, new_stft_cache, new_running_max).
        """
        n_fft: int = self.feature_extractor.n_fft
        hop_length: int = self.feature_extractor.hop_length
        mel_filters = self._mel_filters
        window = self._stft_window

        waveform = audio[0].to(device=self.device, dtype=torch.float32).contiguous()

        is_first_chunk = stft_cache.numel() == 0
        if is_first_chunk:
            waveform = F.pad(waveform, (n_fft // 2, 0))
        else:
            waveform = torch.cat([stft_cache, waveform], dim=0)

        total_samples = waveform.shape[0]

        if total_samples < n_fft:
            packed_features = torch.zeros(mel_filters.shape[0], 0, device=self.device, dtype=self.dtype)
            audio_feature_lengths = torch.tensor([0], dtype=torch.long, device=self.device)
            return packed_features, audio_feature_lengths, waveform, running_max

        num_frames = (total_samples - n_fft) // hop_length + 1
        if num_frames <= 1:
            packed_features = torch.zeros(mel_filters.shape[0], 0, device=self.device, dtype=self.dtype)
            audio_feature_lengths = torch.tensor([0], dtype=torch.long, device=self.device)
            return packed_features, audio_feature_lengths, waveform, running_max

        emit_frames = num_frames - 1
        consumed = (emit_frames - 1) * hop_length + n_fft

        stft = torch.stft(waveform[:consumed], n_fft, hop_length, window=window, center=False, return_complex=True)
        magnitudes = stft.abs() ** 2
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        # Vectorized running-max normalization (replaces per-frame Python loop).
        per_frame_max = log_spec.max(dim=0)[0]  # [num_frames]
        running_max_tensor = running_max.to(device=per_frame_max.device, dtype=per_frame_max.dtype).expand_as(per_frame_max)
        running_max_vals = torch.cummax(torch.maximum(per_frame_max, running_max_tensor), dim=0)[0]
        new_running_max = running_max_vals[-1].clone()
        log_spec = torch.maximum(log_spec, (running_max_vals - 8.0).unsqueeze(0).expand_as(log_spec))
        log_spec = (log_spec + 4.0) / 4.0

        leftover_start = emit_frames * hop_length
        new_stft_cache = waveform[leftover_start:].clone()

        packed_features = log_spec.to(device=self.device, dtype=self.dtype)
        audio_feature_lengths = torch.tensor([emit_frames], dtype=torch.long, device=self.device)
        return packed_features, audio_feature_lengths, new_stft_cache, new_running_max

    def _frame_stack_and_project(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """Frame-stack encoder tokens by downsample_factor, then optionally project and scale.

        When ``config.skip_projector`` is True the frame-stacked tensor is
        returned directly without projection or scaling.

        Args:
            encoder_hidden: Encoder output.
                Shape: [batch_size, num_encoder_tokens, hidden_size]. Dtype: float.

        Returns:
            Adapter embeddings.
            Shape: [batch_size, num_adapter_tokens, hidden_size * downsample_factor]
            (skip_projector) or [batch_size, num_adapter_tokens, projector_output_size].
            Dtype: float.
        """
        stacked = encoder_hidden.reshape(
            encoder_hidden.shape[0], -1, self.config.hidden_size * self.config.downsample_factor
        )
        if self.config.skip_projector:
            return stacked
        assert self.projector is not None
        projected = self.projector(stacked)
        if self.output_embedding_scale != 1.0:
            projected = projected * self.output_embedding_scale
        return projected

    def init_streaming_state(self) -> VoxtralStreamingState:
        """Create an initial streaming state for frame-by-frame encoding.

        When the encoder has been compiled via ``compile_encoder()``, acquires a
        pre-allocated cache set from the pool so that ``torch._dynamo`` sees the
        same Python objects across sessions (no identity-guard recompilation).
        Falls back to ``DynamicCache`` when compilation is not active.

        Returns:
            A fresh ``VoxtralStreamingState`` ready for the first ``forward`` call.
        """
        if self._use_static_cache and self._cache_pool is not None and self._cache_available:
            idx = self._cache_available.pop()
            kv_cache, padding_cache = self._cache_pool[idx]
            kv_cache.reset()
            padding_cache.reset()
            state = VoxtralStreamingState(
                kv_cache=kv_cache,
                padding_cache=padding_cache,
                stft_cache=torch.zeros(0, device=self.device, dtype=torch.float32),
                running_max=torch.full((), float("-inf"), device=self.device, dtype=torch.float32),
            )
            state._pool_idx = idx  # type: ignore[attr-defined]
            state._pool_owner = self  # type: ignore[attr-defined]
            return state
        if self._use_static_cache:
            return self.init_static_streaming_state()
        return VoxtralStreamingState(
            kv_cache=DynamicCache(),
            padding_cache=VoxtralRealtimeConv1dPaddingCache(),
            stft_cache=torch.zeros(0, device=self.device, dtype=torch.float32),
            running_max=torch.full((), float("-inf"), device=self.device, dtype=torch.float32),
        )

    def init_static_streaming_state(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> VoxtralStreamingState:
        """Create a streaming state with pre-allocated static caches for torch.compile.

        Uses ``SlidingWindowVoxtralKVCache`` and ``StaticVoxtralConv1dPaddingCache``
        with all tensor addresses marked static to avoid torch._dynamo recompilation.

        Args:
            batch_size: Batch size (always 1 for streaming).
            device: Target device.  Defaults to the wrapper's device.
            dtype: Target dtype.  Defaults to the wrapper's dtype.

        Returns:
            A ``VoxtralStreamingState`` ready for compiled streaming.
        """
        device = device or self.device
        dtype = dtype or self.dtype
        config = self.config

        kv_cache = SlidingWindowVoxtralKVCache(
            num_layers=config.num_hidden_layers,
            batch_size=batch_size,
            num_kv_heads=config.num_key_value_heads,
            window_size=config.sliding_window,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
        )

        conv_layer_specs = _collect_conv_layer_specs(self.encoder)
        padding_cache = StaticVoxtralConv1dPaddingCache(
            layer_specs=conv_layer_specs,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )

        return VoxtralStreamingState(
            kv_cache=kv_cache,
            padding_cache=padding_cache,
            stft_cache=torch.zeros(0, device=device, dtype=torch.float32),
            running_max=torch.full((), float("-inf"), device=device, dtype=torch.float32),
        )

    def init_streaming_state_pool(
        self,
        pool_size: int,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[VoxtralStreamingState]:
        """Pre-allocate a pool of static streaming states for compile-friendly reuse.

        Each state uses ``SlidingWindowVoxtralKVCache`` and
        ``StaticVoxtralConv1dPaddingCache``.  The same Python objects are reused
        across sessions (via ``reset()``) to avoid ``torch._dynamo`` guard failures
        on object identity.

        Args:
            pool_size: Number of states to pre-allocate (typically ``max_sessions``).
            batch_size: Batch size per state (always 1 for streaming).
            device: Target device.
            dtype: Target dtype.

        Returns:
            List of ``VoxtralStreamingState`` instances.
        """
        return [
            self.init_static_streaming_state(batch_size, device, dtype)
            for _ in range(pool_size)
        ]

    def compile_encoder(self, max_sessions: int = 2) -> None:
        """Enable sliding-window cache and compile the transformer layers.

        Pre-allocates a pool of ``max_sessions`` cache sets so that the same
        Python objects (and tensor addresses) are reused across sessions,
        preventing ``torch._dynamo`` identity-guard recompilation.

        Only the transformer core (RoPE + 32 layers + final norm) is compiled.
        The embedder's conv1d padding cache is excluded from the compiled graph
        because its per-session tensors trigger identity guards.

        Args:
            max_sessions: Number of cache sets to pre-allocate (one per
                concurrent session slot).
        """
        self._use_static_cache = True
        device = self.device
        dtype = self.dtype
        config = self.config
        conv_specs = _collect_conv_layer_specs(self.encoder)

        # Pre-allocate cache pool — same objects reused across sessions.
        self._cache_pool = []
        for _ in range(max_sessions):
            kv = SlidingWindowVoxtralKVCache(
                num_layers=config.num_hidden_layers,
                batch_size=1,
                num_kv_heads=config.num_key_value_heads,
                window_size=config.sliding_window,
                head_dim=config.head_dim,
                dtype=dtype,
                device=device,
            )
            pad = StaticVoxtralConv1dPaddingCache(
                layer_specs=conv_specs,
                batch_size=1,
                dtype=dtype,
                device=device,
            )
            self._cache_pool.append((kv, pad))
        self._cache_available = list(range(max_sessions))

        # Compile the transformer core (excludes embedder + padding cache).
        self._compiled_transformer = torch.compile(
            self.encoder.transformer_forward,
            fullgraph=True,
            dynamic=False,
            mode="reduce-overhead",
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        encoder_past_key_values: object = None,
        padding_cache: object = None,
        use_streaming: bool | None = None,
        causal: bool = False,
        streaming_state: VoxtralStreamingState | None = None,
    ) -> CausalAudioEncoderOutput:
        """Encode raw audio waveforms into hidden state embeddings.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples] for batch mode or
                [1, num_samples] / [1, num_channels, num_samples] for streaming.
            audio_lengths: Valid sample lengths.
                Shape: [batch_size]. Dtype: long.
            encoder_past_key_values: Must be None (legacy, not supported).
            padding_cache: Must be None (legacy, not supported).
            use_streaming: Must be False (legacy, not supported).
            causal: If True, use causal feature extraction.
            streaming_state: If provided, run in incremental streaming mode.

        Returns:
            ``CausalAudioEncoderOutput`` with encoder + projector embeddings.
        """
        assert encoder_past_key_values is None, "VoxtralWrapper: encoder_past_key_values must be None."
        assert padding_cache is None, "VoxtralWrapper: padding_cache must be None."
        assert not use_streaming, "VoxtralWrapper: use_streaming must be False."

        if streaming_state is not None:
            return self.forward_streaming_chunk(audio, streaming_state)

        assert 1 <= audio.shape[1] <= 2, f"Number of audio channels must be 1 or 2, got {audio.shape[1]}."
        return self._forward_batch(audio, audio_lengths, causal=causal)

    def forward_streaming_chunk(
        self,
        audio: torch.Tensor,
        streaming_state: VoxtralStreamingState,
    ) -> CausalAudioEncoderOutput:
        """Run the eager streaming entrypoint around the compiled encoder core."""
        self._streaming_calls += 1
        prepared_audio = self.prepare_streaming_audio(audio)
        return self._forward_streaming(prepared_audio, streaming_state)

    def _forward_streaming(
        self,
        audio: torch.Tensor,
        streaming_state: VoxtralStreamingState,
    ) -> CausalAudioEncoderOutput:
        """Streaming forward: process one chunk at 12.5Hz.

        Args:
            audio: Prepared mono audio chunk at encoder sample rate.
                Shape: [1, num_samples]. Dtype: float32.
            streaming_state: Current streaming state.

        Returns:
            ``CausalAudioEncoderOutput`` with new adapter embeddings and updated state.
        """
        if audio.ndim != 2 or audio.shape[0] != 1:
            raise ValueError(f"Expected prepared streaming audio with shape [1, num_samples], got {tuple(audio.shape)}.")

        packed_features, audio_feature_lengths, new_stft_cache, new_running_max = self._extract_features_causal_streaming(
            audio, streaming_state.stft_cache, streaming_state.running_max
        )

        new_state = VoxtralStreamingState(
            kv_cache=streaming_state.kv_cache,
            padding_cache=streaming_state.padding_cache,
            stft_cache=new_stft_cache,
            running_max=new_running_max,
            mel_buffer=streaming_state.mel_buffer,
        )
        # Preserve pool metadata for cache release.
        if hasattr(streaming_state, "_pool_idx"):
            new_state._pool_idx = streaming_state._pool_idx  # type: ignore[attr-defined]
            new_state._pool_owner = streaming_state._pool_owner  # type: ignore[attr-defined]

        num_feature_frames = int(audio_feature_lengths[0].item())
        if num_feature_frames == 0:
            self._streaming_zero_feature_calls += 1
            return CausalAudioEncoderOutput(
                embeds=torch.zeros(1, 0, self.hidden_size, device=self.device, dtype=self.dtype),
                streaming_state=new_state,
            )

        # Append new mel frames to the buffer.
        # Shape: [num_mel_bins, new_frames] -> [1, num_mel_bins, new_frames]
        new_mel = packed_features.unsqueeze(0)
        if new_state.mel_buffer is not None:
            mel_all = torch.cat([new_state.mel_buffer, new_mel], dim=2)
        else:
            mel_all = new_mel

        # Process complete groups of MEL_FRAMES_PER_ADAPTER_TOKEN mel frames.
        total_mel_frames = mel_all.shape[2]
        num_complete_groups = total_mel_frames // MEL_FRAMES_PER_ADAPTER_TOKEN
        consumed_mel = num_complete_groups * MEL_FRAMES_PER_ADAPTER_TOKEN

        if num_complete_groups == 0:
            self._streaming_zero_adapter_token_calls += 1
            new_state.mel_buffer = mel_all
            return CausalAudioEncoderOutput(
                embeds=torch.zeros(1, 0, self.hidden_size, device=self.device, dtype=self.dtype),
                streaming_state=new_state,
            )

        # Feed mel frames through encoder in ENCODER_STRIDE chunks.
        mel_to_process = mel_all[:, :, :consumed_mel]
        all_adapter_tokens: list[torch.Tensor] = []

        use_sliding_window = isinstance(new_state.kv_cache, SlidingWindowVoxtralKVCache)

        for group_idx in range(num_complete_groups):
            step_enc_tokens: list[torch.Tensor] = []
            for sub in range(DOWNSAMPLE_FACTOR):
                mel_start = group_idx * MEL_FRAMES_PER_ADAPTER_TOKEN + sub * ENCODER_STRIDE
                chunk = mel_to_process[:, :, mel_start : mel_start + ENCODER_STRIDE]

                if use_sliding_window:
                    kv_cache = new_state.kv_cache
                    assert isinstance(kv_cache, SlidingWindowVoxtralKVCache)

                    # Run embedder OUTSIDE the compiled graph (it touches
                    # padding_cache whose tensor identity varies per session).
                    inputs_embeds = self.encoder.embedder(
                        chunk, new_state.padding_cache
                    )

                    # Pre-compute position_ids and mask outside the compiled
                    # graph to avoid Python int guards.
                    position_ids = torch.tensor(
                        [[kv_cache._total_seen_tokens]], device=self.device, dtype=torch.long
                    )
                    mask = _make_fixed_sliding_window_mask(
                        valid_len=kv_cache.get_kv_len() + 1,
                        window_size=kv_cache._window_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    kv_cache.step()

                    # Compiled transformer core: RoPE + 32 layers + norm.
                    transformer_fn = self._compiled_transformer or self.encoder.transformer_forward
                    if self._compiled_transformer is not None:
                        self._streaming_compiled_transformer_calls += 1
                    else:
                        self._streaming_eager_transformer_calls += 1
                    hidden = transformer_fn(
                        inputs_embeds, position_ids, mask, kv_cache
                    )
                    if self._compiled_transformer is not None:
                        hidden = hidden.clone()
                    step_enc_tokens.append(hidden)
                else:
                    self._streaming_eager_transformer_calls += 1
                    out = self.encoder(
                        input_features=chunk,
                        past_key_values=new_state.kv_cache,
                        padding_cache=new_state.padding_cache,
                        use_cache=True,
                        use_padding_cache=True,
                    )
                    new_state.kv_cache = out.past_key_values
                    new_state.padding_cache = out.padding_cache
                    step_enc_tokens.append(out.last_hidden_state)

            enc_group = torch.cat(step_enc_tokens, dim=1)
            adapter_token = self._frame_stack_and_project(enc_group)
            all_adapter_tokens.append(adapter_token)

        # Save leftover mel frames.
        if consumed_mel < total_mel_frames:
            new_state.mel_buffer = mel_all[:, :, consumed_mel:]
        else:
            new_state.mel_buffer = None

        embeds = torch.cat(all_adapter_tokens, dim=1)
        self._streaming_emitted_adapter_tokens += int(embeds.shape[1])
        return CausalAudioEncoderOutput(
            embeds=embeds,
            streaming_state=new_state,
        )

    def _forward_batch(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor | None,
        causal: bool = False,
    ) -> CausalAudioEncoderOutput:
        """Non-streaming batch forward.

        Args:
            audio: Raw audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths.
                Shape: [batch_size]. Dtype: long.
            causal: Whether to use causal feature extraction.

        Returns:
            ``CausalAudioEncoderOutput`` with adapter embeddings.
        """
        num_samples = audio.shape[2]
        expected_output_length = self.compute_expected_output_length(num_samples)

        if causal:
            mel_features, mel_lengths = self._extract_features_causal(audio, audio_lengths)
        else:
            mel_features, mel_lengths = self._extract_features(audio, audio_lengths)

        # Ensure at least MEL_FRAMES_PER_ADAPTER_TOKEN mel frames so the encoder
        # produces at least one adapter token.  Very short audio (< ~125 ms in
        # causal mode) can yield fewer frames; right-pad with zeros to reach the
        # minimum.  The padding is silence and gets masked out downstream.
        max_mel = mel_features.shape[2]
        if max_mel < MEL_FRAMES_PER_ADAPTER_TOKEN:
            import warnings

            warnings.warn(
                f"[VoxtralWrapper] Short audio: {max_mel} mel frames < {MEL_FRAMES_PER_ADAPTER_TOKEN} required. "
                f"Right-padding with silence. audio_samples={num_samples}, causal={causal}",
                stacklevel=2,
            )
            mel_features = F.pad(mel_features, (0, MEL_FRAMES_PER_ADAPTER_TOKEN - max_mel))
            max_mel = MEL_FRAMES_PER_ADAPTER_TOKEN

        # Truncate mel to a multiple of MEL_FRAMES_PER_ADAPTER_TOKEN for clean frame-stacking.
        usable_mel = (max_mel // MEL_FRAMES_PER_ADAPTER_TOKEN) * MEL_FRAMES_PER_ADAPTER_TOKEN
        mel_features = mel_features[:, :, :usable_mel]

        enc_out = self.encoder(
            input_features=mel_features,
            use_cache=False,
            use_padding_cache=False,
        )
        enc_hidden = enc_out.last_hidden_state
        embeds = self._frame_stack_and_project(enc_hidden)

        actual_output_length = embeds.shape[1]
        if actual_output_length > expected_output_length:
            embeds = embeds[:, :expected_output_length]
        elif actual_output_length < expected_output_length:
            embeds = F.pad(embeds, (0, 0, 0, expected_output_length - actual_output_length))

        return CausalAudioEncoderOutput(embeds=embeds)
