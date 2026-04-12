"""Voxtral Realtime encoder and projector, vendored from HuggingFace transformers.

These classes are copied from the upstream ``transformers.models.voxtral_realtime``
module (transformers >= 5.2) and adapted to run on transformers 4.57.x without
requiring a version upgrade.  Only the encoder and multi-modal projector are
included; the text decoder and generation logic are not needed.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ROPE_INIT_FUNCTIONS: dict[str, Any] = {}


class VoxtralRealtimeEncoderConfig(PretrainedConfig):
    """Configuration for the Voxtral Realtime audio encoder.

    Stores both the encoder architecture parameters and the projector/downsample
    settings needed to reconstruct the full audio pipeline.
    """

    model_type = "voxtral_realtime_encoder"

    def __init__(
        self,
        hidden_size: int = 1280,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        activation_function: str = "gelu",
        num_mel_bins: int = 128,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1500,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        sliding_window: int = 750,
        head_dim: int = 64,
        downsample_factor: int = 4,
        projector_hidden_act: str = "gelu",
        projector_output_size: int | None = None,
        output_embedding_scale: float = 1.0,
        skip_projector: bool = False,
        attn_implementation: str = "eager",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.activation_function = activation_function
        self.num_mel_bins = num_mel_bins
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.downsample_factor = downsample_factor
        self.projector_hidden_act = projector_hidden_act
        self.projector_output_size = projector_output_size
        self.output_embedding_scale = output_embedding_scale
        self.skip_projector = skip_projector
        self._attn_implementation = attn_implementation

        # Aliases expected by the encoder layers.
        self.encoder_layers = num_hidden_layers
        self.encoder_attention_heads = num_attention_heads

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "VoxtralRealtimeEncoderConfig":
        """Load config from a Voxtral Realtime checkpoint.

        Reads ``config.json`` and extracts the ``audio_config`` sub-dict along
        with top-level ``downsample_factor``, ``projector_hidden_act``, and
        ``text_config.hidden_size`` (used as ``projector_output_size``).

        Works with both the full ``voxtral_realtime`` model config and a
        standalone ``voxtral_realtime_encoder`` config.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path.

        Returns:
            Populated ``VoxtralRealtimeEncoderConfig``.
        """
        import json
        import os

        from huggingface_hub import hf_hub_download

        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json",
            )

        with open(config_path) as f:
            full_config = json.load(f)

        # If this is the full model config, extract the audio sub-config.
        if "audio_config" in full_config:
            audio_cfg = full_config["audio_config"]
            downsample_factor = full_config.get("downsample_factor", 4)
            projector_hidden_act = full_config.get("projector_hidden_act", "gelu")
            text_hidden_size = full_config.get("text_config", {}).get("hidden_size")
        else:
            # Standalone encoder config (e.g. saved by us).
            audio_cfg = full_config
            downsample_factor = audio_cfg.get("downsample_factor", 4)
            projector_hidden_act = audio_cfg.get("projector_hidden_act", "gelu")
            text_hidden_size = audio_cfg.get("projector_output_size")

        # The upstream rope_theta lives inside rope_parameters.
        rope_params = audio_cfg.get("rope_parameters") or {}
        rope_theta = rope_params.get("rope_theta", audio_cfg.get("rope_theta", 10000.0))

        return cls(
            hidden_size=audio_cfg.get("hidden_size", 1280),
            intermediate_size=audio_cfg.get("intermediate_size", 5120),
            num_hidden_layers=audio_cfg.get("num_hidden_layers", 32),
            num_attention_heads=audio_cfg.get("num_attention_heads", 32),
            num_key_value_heads=audio_cfg.get("num_key_value_heads"),
            activation_function=audio_cfg.get("activation_function", "gelu"),
            num_mel_bins=audio_cfg.get("num_mel_bins", 128),
            initializer_range=audio_cfg.get("initializer_range", 0.02),
            attention_dropout=audio_cfg.get("attention_dropout", 0.0),
            hidden_act=audio_cfg.get("hidden_act", "silu"),
            max_position_embeddings=audio_cfg.get("max_position_embeddings", 1500),
            rms_norm_eps=audio_cfg.get("rms_norm_eps", 1e-5),
            rope_theta=rope_theta,
            sliding_window=audio_cfg.get("sliding_window", 750),
            head_dim=audio_cfg.get("head_dim", 64),
            downsample_factor=downsample_factor,
            projector_hidden_act=projector_hidden_act,
            projector_output_size=text_hidden_size,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Conv1d padding cache (for streaming)
# ---------------------------------------------------------------------------


class VoxtralRealtimeConv1dCacheLayer:
    """Cache for a single causal Conv1d layer's left-padding state."""

    def __init__(self) -> None:
        self.cache: torch.Tensor | None = None
        self.is_initialized: bool = False

    def lazy_initialization(
        self,
        hidden_states: torch.Tensor,
        conv_module: "VoxtralRealtimeCausalConv1d",
    ) -> None:
        """Initialize the cache on first use."""
        self.left_pad = conv_module.left_pad
        self.in_channels = conv_module.in_channels
        self.cache = torch.zeros(
            hidden_states.shape[0],
            self.in_channels,
            self.left_pad,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self.is_initialized = True

    def update(
        self,
        hidden_states: torch.Tensor,
        conv_module: "VoxtralRealtimeCausalConv1d | None" = None,
    ) -> torch.Tensor:
        """Return the current padding and update the cache with new states."""
        if not self.is_initialized and conv_module is not None:
            self.lazy_initialization(hidden_states, conv_module)
        elif not self.is_initialized:
            raise ValueError("Cache not initialized. Provide conv_module on first call.")

        assert self.cache is not None
        if self.left_pad > 0:
            shortfall = max(0, self.left_pad - hidden_states.shape[-1])
            if shortfall > 0:
                padding_states = torch.cat([self.cache[:, :, -shortfall:], hidden_states], dim=-1)
            else:
                padding_states = hidden_states[:, :, -self.left_pad :]
        else:
            padding_states = torch.empty(
                hidden_states.shape[0],
                self.in_channels,
                0,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        current_cache = self.cache.clone()
        self.cache.copy_(padding_states)
        return current_cache


class VoxtralRealtimeConv1dPaddingCache:
    """Container for per-layer conv1d padding caches used during streaming."""

    def __init__(self) -> None:
        self.layers: dict[str, VoxtralRealtimeConv1dCacheLayer] = {}

    def update(
        self,
        hidden_states: torch.Tensor,
        cache_key: str,
        conv_module: "VoxtralRealtimeCausalConv1d",
    ) -> torch.Tensor:
        """Pad hidden_states using cached left-padding for the given layer."""
        if cache_key not in self.layers:
            self.layers[cache_key] = VoxtralRealtimeConv1dCacheLayer()
        padding_states = self.layers[cache_key].update(hidden_states, conv_module)
        return torch.cat([padding_states, hidden_states], dim=-1)


class StaticVoxtralKVCache(Cache):
    """Pre-allocated KV cache for torch.compile-friendly Voxtral streaming.

    All key/value tensors are allocated upfront at ``max_cache_len`` and updated
    in-place.  Tensor addresses are marked static via ``torch._dynamo`` to
    prevent recompilation when the same buffers are reused across steps.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        max_cache_len: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Skip Cache.__init__ which requires layers/layer_class_to_replicate
        # in newer transformers versions.  We manage our own KV buffers directly.
        self._max_len = max_cache_len
        self._seen_tokens = 0
        self.num_layers = num_layers

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        for _ in range(num_layers):
            k = torch.zeros(batch_size, num_kv_heads, max_cache_len, head_dim, dtype=dtype, device=device)
            v = torch.zeros(batch_size, num_kv_heads, max_cache_len, head_dim, dtype=dtype, device=device)
            self.key_cache.append(k)
            self.value_cache.append(v)
            torch._dynamo.mark_static_address(k)
            torch._dynamo.mark_static_address(v)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new KV entries at the current position and return the active slice.

        Args:
            key_states: New key states.
                Shape: [batch_size, num_kv_heads, new_tokens, head_dim]. Dtype: float.
            value_states: New value states.
                Shape: [batch_size, num_kv_heads, new_tokens, head_dim]. Dtype: float.
            layer_idx: Index of the transformer layer.
            cache_kwargs: Unused, kept for Cache interface compatibility.

        Returns:
            Tuple of (key_cache, value_cache) sliced to the active length.
            Each shape: [batch_size, num_kv_heads, seen_tokens + new_tokens, head_dim].
        """
        new_tokens = key_states.shape[2]
        start = self._seen_tokens

        self.key_cache[layer_idx][:, :, start : start + new_tokens, :].copy_(key_states)
        self.value_cache[layer_idx][:, :, start : start + new_tokens, :].copy_(value_states)

        end = start + new_tokens
        # Increment the global counter after the last layer writes.
        if layer_idx == self.num_layers - 1:
            self._seen_tokens = end

        return (
            self.key_cache[layer_idx][:, :, :end, :],
            self.value_cache[layer_idx][:, :, :end, :],
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the number of tokens currently stored."""
        return self._seen_tokens

    def get_max_cache_shape(self) -> list[int]:
        """Return the maximum cache length."""
        return [self._max_len]

    def reset(self) -> None:
        """Zero out all caches and reset the position counter."""
        self._seen_tokens = 0
        for k, v in zip(self.key_cache, self.value_cache):
            k.zero_()
            v.zero_()


class SlidingWindowVoxtralKVCache(Cache):
    """Fixed-size sliding window KV cache for torch.compile-compatible Voxtral streaming.

    Pre-allocates key/value tensors of shape
    ``[batch_size, num_kv_heads, window_size, head_dim]`` and manages them with an
    always-shift strategy: every ``update()`` call shifts the buffer left by one
    position and writes the new token at the rightmost slot.  The full buffer is
    always returned (never a slice), keeping the shape constant across all steps
    and enabling CUDA graph capture via ``torch.compile(dynamic=False)``.

    During warmup (first ``window_size`` steps), valid data fills from right to
    left while the left side contains zeros.  The caller is responsible for
    building an attention mask that blocks these invalid positions.  After warmup,
    all positions are valid and the mask is all-attend.

    Counter bookkeeping (``total_seen_tokens``, ``valid_len``) is managed
    externally by the caller to keep the compiled graph free of data-dependent
    Python int guards.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        window_size: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # Skip Cache.__init__ which requires layers/layer_class_to_replicate
        # in newer transformers versions.  We manage our own KV buffers directly.
        self._window_size = window_size
        self._num_layers = num_layers
        self._total_seen_tokens = 0
        self._valid_len = 0

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        for _ in range(num_layers):
            k = torch.zeros(batch_size, num_kv_heads, window_size, head_dim, dtype=dtype, device=device)
            v = torch.zeros(batch_size, num_kv_heads, window_size, head_dim, dtype=dtype, device=device)
            self.key_cache.append(k)
            self.value_cache.append(v)
            torch._dynamo.mark_static_address(k)
            torch._dynamo.mark_static_address(v)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shift buffer left and write new KV at the end; return the full buffer.

        The shift-then-write pattern has no data-dependent branches, making it
        safe for ``torch.compile(fullgraph=True)``.

        Args:
            key_states: New key states.
                Shape: [batch_size, num_kv_heads, 1, head_dim]. Dtype: float.
            value_states: New value states.
                Shape: [batch_size, num_kv_heads, 1, head_dim]. Dtype: float.
            layer_idx: Index of the transformer layer.
            cache_kwargs: Unused, kept for Cache interface compatibility.

        Returns:
            Tuple of (key_cache, value_cache), each the full pre-allocated buffer.
            Shape: [batch_size, num_kv_heads, window_size, head_dim].
        """
        k_buf = self.key_cache[layer_idx]
        v_buf = self.value_cache[layer_idx]

        # Always shift left by 1 and write at the end.  During warmup this
        # shifts zeros into zeros on the left, which is harmless since the
        # attention mask blocks those positions.  .clone() guarantees
        # correctness for overlapping source/destination regions.
        k_buf[:, :, :-1, :].copy_(k_buf[:, :, 1:, :].clone())
        k_buf[:, :, -1:, :].copy_(key_states)
        v_buf[:, :, :-1, :].copy_(v_buf[:, :, 1:, :].clone())
        v_buf[:, :, -1:, :].copy_(value_states)

        return k_buf, v_buf

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the total number of tokens seen (for RoPE position IDs)."""
        return self._total_seen_tokens

    def get_kv_len(self) -> int:
        """Return how many buffer positions contain valid data (for mask building)."""
        return self._valid_len

    def get_max_cache_shape(self) -> list[int]:
        """Return the fixed window size."""
        return [self._window_size]

    def step(self) -> None:
        """Advance counters by one token.  Called by the wrapper BEFORE the encoder."""
        self._total_seen_tokens += 1
        self._valid_len = min(self._valid_len + 1, self._window_size)

    def reset(self) -> None:
        """Zero all buffers and reset counters for session reuse."""
        self._total_seen_tokens = 0
        self._valid_len = 0
        for k, v in zip(self.key_cache, self.value_cache):
            k.zero_()
            v.zero_()


class StaticVoxtralConv1dPaddingCache(VoxtralRealtimeConv1dPaddingCache):
    """Pre-allocated conv1d padding cache with static tensor addresses.

    Mirrors ``VoxtralRealtimeConv1dPaddingCache`` but pre-allocates all layer
    buffers at construction time so that ``torch._dynamo`` can mark their
    addresses as static and avoid recompilation.
    """

    def __init__(
        self,
        layer_specs: list[tuple[str, int, int]],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Initialize with pre-allocated buffers.

        Args:
            layer_specs: List of (cache_key, in_channels, left_pad) per conv layer.
            batch_size: Batch size (always 1 for streaming).
            dtype: Tensor dtype.
            device: Tensor device.
        """
        super().__init__()
        for cache_key, in_channels, left_pad in layer_specs:
            layer = VoxtralRealtimeConv1dCacheLayer()
            layer.left_pad = left_pad
            layer.in_channels = in_channels
            layer.cache = torch.zeros(batch_size, in_channels, left_pad, dtype=dtype, device=device)
            layer.is_initialized = True
            torch._dynamo.mark_static_address(layer.cache)
            self.layers[cache_key] = layer

    def reset(self) -> None:
        """Zero all padding caches without re-allocating."""
        for layer in self.layers.values():
            if layer.cache is not None:
                layer.cache.zero_()


# ---------------------------------------------------------------------------
# Encoder output
# ---------------------------------------------------------------------------


@dataclass
class VoxtralRealtimeEncoderOutput(BaseModelOutputWithPast):
    """Output type for the Voxtral encoder, adding a padding cache field."""

    padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None


# ---------------------------------------------------------------------------
# Rotary embedding
# ---------------------------------------------------------------------------


class VoxtralRealtimeRotaryEmbedding(nn.Module):
    """RoPE implementation for the Voxtral encoder."""

    inv_freq: torch.Tensor

    def __init__(self, config: VoxtralRealtimeEncoderConfig, device: torch.device | None = None) -> None:
        super().__init__()
        self.config = config
        dim = config.head_dim
        base = config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for rotary position embeddings."""
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Causal Conv1d
# ---------------------------------------------------------------------------


class VoxtralRealtimeCausalConv1d(nn.Conv1d):
    """Causal Conv1d that supports streaming via a padding cache."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        cache_key: str,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        self.cache_key = cache_key

    @cached_property
    def left_pad(self) -> int:
        """Number of left-padding samples needed for causal convolution."""
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        """Run causal conv1d, using padding_cache if in streaming mode."""
        if padding_cache is not None:
            x = padding_cache.update(x, self.cache_key, self)
        else:
            x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


# ---------------------------------------------------------------------------
# RMS Norm
# ---------------------------------------------------------------------------


class VoxtralRealtimeRMSNorm(nn.Module):
    """RMS normalization layer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary helpers
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class VoxtralRealtimeAttention(nn.Module):
    """Multi-headed attention with RoPE and sliding-window causal masking."""

    def __init__(self, config: VoxtralRealtimeEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """Run multi-head attention with RoPE."""
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.training and self.attention_dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
        return self.o_proj(attn_output)


class VoxtralRealtimeSdpaAttention(VoxtralRealtimeAttention):
    """SDPA-backed attention using torch.nn.functional.scaled_dot_product_attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """Run multi-head attention via SDPA with RoPE.

        Args:
            hidden_states: Input to the attention layer.
                Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
            position_embeddings: Tuple of (cos, sin) rotary embeddings.
                Each shape: [batch_size, seq_len, head_dim]. Dtype: float.
            attention_mask: Additive causal mask applied before softmax.
                Shape: [1, 1, seq_len, total_seq_len]. Dtype: float.
            past_key_values: KV cache for streaming inference.

        Returns:
            Attention output. Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
        """
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Convert additive float mask to bool mask for SDPA (True = attend, False = mask).
        # SDPA expects attn_mask as either a boolean mask or an additive float mask.
        # The existing mask is already additive (0.0 = attend, large negative = mask),
        # so we pass it directly.
        sdpa_mask = attention_mask

        # Handle GQA by expanding KV heads to match query heads.
        if self.num_key_value_groups > 1:
            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)

        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1).contiguous()
        return self.o_proj(attn_output)


class VoxtralRealtimeFlashAttention2(VoxtralRealtimeAttention):
    """Flash Attention 2 backed attention using flash_attn.flash_attn_func."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """Run multi-head attention via Flash Attention 2 with RoPE.

        Args:
            hidden_states: Input to the attention layer.
                Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
            position_embeddings: Tuple of (cos, sin) rotary embeddings.
                Each shape: [batch_size, seq_len, head_dim]. Dtype: float.
            attention_mask: Unused; Flash Attention handles causality natively.
            past_key_values: KV cache for streaming inference.

        Returns:
            Attention output. Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
        """
        try:
            from flash_attn import flash_attn_func  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "flash_attn is required for flash_attention_2 backend. Install it with: pip install flash-attn"
            ) from exc

        bsz, seq_len, _ = hidden_states.shape
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        # Projections: [batch_size, seq_len, num_heads, head_dim] (no transpose for flash_attn).
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE: temporarily transpose to [batch_size, num_heads, seq_len, head_dim].
        cos, sin = position_embeddings
        query_states_t = query_states.transpose(1, 2)
        key_states_t = key_states.transpose(1, 2)
        query_states_t, key_states_t = _apply_rotary_pos_emb(query_states_t, key_states_t, cos, sin)
        # Transpose back to [batch_size, seq_len, num_heads, head_dim].
        query_states = query_states_t.transpose(1, 2)
        key_states = key_states_t.transpose(1, 2)

        if past_key_values is not None:
            # Cache expects [batch_size, num_heads, seq_len, head_dim].
            key_states_c, value_states_c = past_key_values.update(
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                self.layer_idx,
            )
            # Transpose back to [batch_size, seq_len, num_heads, head_dim].
            key_states = key_states_c.transpose(1, 2)
            value_states = value_states_c.transpose(1, 2)

        dropout_p = self.attention_dropout if self.training else 0.0
        # flash_attn_func expects [batch_size, seq_len, num_heads, head_dim].
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout_p,
            softmax_scale=self.scaling,
            causal=True,
            window_size=(self.config.sliding_window - 1, 0),
        )

        # Reshape: [batch_size, seq_len, num_heads * head_dim].
        attn_output = attn_output.reshape(bsz, attn_output.shape[1], -1).contiguous()
        return self.o_proj(attn_output)


VOXTRAL_ATTENTION_CLASSES: dict[str, type[VoxtralRealtimeAttention]] = {
    "eager": VoxtralRealtimeAttention,
    "sdpa": VoxtralRealtimeSdpaAttention,
    "flash_attention_2": VoxtralRealtimeFlashAttention2,
}


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class VoxtralRealtimeMLP(nn.Module):
    """Gated MLP (SwiGLU-style) used in encoder layers."""

    def __init__(self, config: VoxtralRealtimeEncoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated MLP."""
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Embedder (mel -> hidden via two causal convs)
# ---------------------------------------------------------------------------


class VoxtralRealtimeEmbedder(nn.Module):
    """Front-end: two causal Conv1d layers mapping mel bins to hidden_size at 50Hz."""

    def __init__(self, config: VoxtralRealtimeEncoderConfig) -> None:
        super().__init__()
        self.conv1 = VoxtralRealtimeCausalConv1d(config.num_mel_bins, config.hidden_size, kernel_size=3, cache_key="conv1")
        self.conv2 = VoxtralRealtimeCausalConv1d(
            config.hidden_size, config.hidden_size, kernel_size=3, stride=2, cache_key="conv2"
        )

    def forward(
        self,
        input_features: torch.Tensor,
        padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        """Convert mel spectrogram to encoder input embeddings.

        Args:
            input_features: Mel spectrogram.
                Shape: [batch_size, num_mel_bins, num_frames]. Dtype: float.
            padding_cache: Optional streaming conv padding cache.

        Returns:
            Embeddings. Shape: [batch_size, num_encoder_tokens, hidden_size]. Dtype: float.
        """
        x = F.gelu(self.conv1(input_features, padding_cache=padding_cache))
        x = F.gelu(self.conv2(x, padding_cache=padding_cache))
        return x.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------


class VoxtralRealtimeEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm attention and MLP."""

    def __init__(self, config: VoxtralRealtimeEncoderConfig, layer_idx: int) -> None:
        super().__init__()
        attn_cls = VOXTRAL_ATTENTION_CLASSES.get(config._attn_implementation, VoxtralRealtimeAttention)
        self.self_attn = attn_cls(config, layer_idx)
        self.self_attn_layer_norm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layer_norm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = VoxtralRealtimeMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """Run one encoder layer."""
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Sliding-window causal mask builder
# ---------------------------------------------------------------------------


def _make_sliding_window_causal_mask(
    seq_len: int,
    past_len: int,
    sliding_window: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a [1, 1, seq_len, past_len + seq_len] causal attention mask.

    Positions outside the sliding window are masked with a large negative value.
    """
    total_len = past_len + seq_len
    mask = torch.full((seq_len, total_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    for i in range(seq_len):
        abs_pos = past_len + i
        start = 0
        if sliding_window is not None:
            start = max(0, abs_pos - sliding_window + 1)
        mask[i, start : abs_pos + 1] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


def _make_fixed_sliding_window_mask(
    valid_len: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a fixed-shape ``[1, 1, 1, window_size]`` mask for a sliding-window buffer.

    The buffer is filled from right to left during warmup: valid entries occupy
    positions ``[window_size - valid_len, window_size)``.  Positions to the left
    of the valid region are masked with a large negative value so that
    ``softmax`` assigns them zero weight.

    After warmup (``valid_len == window_size``), the mask is all zeros (full
    attend).

    Args:
        valid_len: Number of buffer positions that contain real data.
        window_size: Fixed size of the KV buffer.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Attention mask. Shape: [1, 1, 1, window_size]. Dtype: float.
    """
    mask = torch.zeros(1, 1, 1, window_size, dtype=dtype, device=device)
    if valid_len < window_size:
        mask[:, :, :, : window_size - valid_len] = torch.finfo(dtype).min
    return mask


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class VoxtralRealtimeEncoder(PreTrainedModel):
    """Voxtral Realtime audio encoder (causal transformer over mel spectrograms).

    Produces hidden states at 50Hz (one token per 2 mel frames).  Supports
    streaming via KV cache and conv1d padding cache.
    """

    config_class = VoxtralRealtimeEncoderConfig  # type: ignore[assignment]
    main_input_name = "input_features"
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config: VoxtralRealtimeEncoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embedder = VoxtralRealtimeEmbedder(config)
        self.layers = nn.ModuleList(
            [VoxtralRealtimeEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = VoxtralRealtimeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = VoxtralRealtimeRotaryEmbedding(config)
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        padding_cache: VoxtralRealtimeConv1dPaddingCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        use_padding_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> VoxtralRealtimeEncoderOutput:
        """Run the encoder on mel features or pre-computed embeddings.

        Args:
            input_features: Log-mel spectrogram.
                Shape: [batch_size, num_mel_bins, num_frames]. Dtype: float.
            past_key_values: KV cache for streaming.
            padding_cache: Conv1d padding cache for streaming.
            inputs_embeds: Pre-computed embeddings (alternative to input_features).
                Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
            use_cache: Whether to use/return KV cache.
            use_padding_cache: Whether to use/return padding cache.
            position_ids: Pre-computed position IDs for RoPE.
                Shape: [batch_size, seq_len]. Dtype: long.
                When provided, skips computing from cache (avoids dynamo int guards).
            attention_mask: Pre-computed attention mask.
                Shape: [1, 1, seq_len, kv_len]. Dtype: float.
                When provided, skips building from cache state.

        Returns:
            VoxtralRealtimeEncoderOutput with last_hidden_state, past_key_values,
            and padding_cache.
        """
        if (input_features is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_features or inputs_embeds.")

        if use_padding_cache and padding_cache is None:
            padding_cache = VoxtralRealtimeConv1dPaddingCache()

        if inputs_embeds is None:
            inputs_embeds = self.embedder(input_features, padding_cache if use_padding_cache else None)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # Position IDs: use caller-provided tensor or compute from cache state.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens  # type: ignore
            position_ids = position_ids.unsqueeze(0)

        # Attention mask: use caller-provided tensor or build from config.
        if attention_mask is not None:
            causal_mask = attention_mask
        elif self.config._attn_implementation == "flash_attention_2":
            # Flash Attention handles causality and sliding window natively.
            causal_mask = None
        else:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            causal_mask = _make_sliding_window_causal_mask(
                seq_len=inputs_embeds.shape[1],  # type: ignore
                past_len=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                device=inputs_embeds.device,  # type: ignore
                dtype=inputs_embeds.dtype,  # type: ignore
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values if use_cache else None,
            )

        hidden_states = self.norm(hidden_states)
        return VoxtralRealtimeEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            padding_cache=padding_cache if use_padding_cache else None,
        )

    def transformer_forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Cache,
    ) -> torch.Tensor:
        """Run only the transformer layers (RoPE + attention/MLP + final norm).

        This method excludes the embedder and all cache/mask setup logic, making
        it safe for ``torch.compile(fullgraph=True)`` when the KV cache tensors
        have stable addresses (via ``torch._dynamo.mark_static_address``).

        Args:
            inputs_embeds: Embedder output.
                Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
            position_ids: Absolute position IDs for RoPE.
                Shape: [batch_size, seq_len]. Dtype: long.
            attention_mask: Pre-built attention mask.
                Shape: [1, 1, seq_len, kv_len]. Dtype: float.
            past_key_values: KV cache (``SlidingWindowVoxtralKVCache``).

        Returns:
            Encoder output hidden states.
            Shape: [batch_size, seq_len, hidden_size]. Dtype: float.
        """
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )
        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Multi-modal projector (frame-stack + MLP)
# ---------------------------------------------------------------------------


class VoxtralRealtimeMultiModalProjector(nn.Module):
    """Frame-stacks encoder tokens by downsample_factor, then projects via MLP.

    Reduces the 50Hz encoder output to 12.5Hz adapter embeddings matching the
    LLM hidden size.
    """

    def __init__(self, config: VoxtralRealtimeEncoderConfig) -> None:
        super().__init__()
        output_size = config.projector_output_size or config.hidden_size
        self.linear_1 = nn.Linear(config.hidden_size * config.downsample_factor, output_size, bias=False)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(output_size, output_size, bias=False)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Project frame-stacked encoder features.

        Args:
            audio_features: Frame-stacked encoder output.
                Shape: [batch_size, num_adapter_tokens, hidden_size * downsample_factor].
                Dtype: float.

        Returns:
            Projected embeddings.
            Shape: [batch_size, num_adapter_tokens, output_size]. Dtype: float.
        """
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
