"""Reusable sub-modules: EmbeddingAdaptor, code predictor, speaker encoders, and dtype utilities."""

import logging
import os
from dataclasses import dataclass
from typing import Any, overload

import torch
import torch.nn.functional as F
from einops import einsum
from torch import nn
from transformers import (
    Cache,
    GenerationMixin,
    PretrainedConfig,
    Qwen3Config,
    Qwen3Model,
    Qwen3OmniMoePreTrainedModel,
    Qwen3OmniMoeTalkerCodePredictorModel,
    StaticCache,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorOutputWithPast
from transformers.utils.generic import ModelOutput

from .audio_encoder.streaming_mimi import MimiConv1dPaddingCache, MimiConvTranspose1dPaddingCache

logger = logging.getLogger(__name__)


def _ensure_torchaudio_speechbrain_compat() -> None:
    """Patch Torchaudio APIs that older SpeechBrain releases still expect.

    Torchaudio 2.9 removed the old backend-listing interface, but SpeechBrain's
    import path still probes it during module import. Returning an empty backend
    list preserves SpeechBrain's warning path without blocking runtime import.
    """
    try:
        import torchaudio
    except Exception:
        return

    if not hasattr(torchaudio, "list_audio_backends"):
        setattr(torchaudio, "list_audio_backends", lambda: [])

    if not hasattr(torchaudio, "set_audio_backend"):
        setattr(torchaudio, "set_audio_backend", lambda _backend: None)


def _get_module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


@overload
def cast_float_inputs(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor: ...
@overload
def cast_float_inputs(tensor: None, target_dtype: torch.dtype) -> None: ...


def cast_float_inputs(tensor: torch.Tensor | None, target_dtype: torch.dtype) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.is_floating_point() and tensor.dtype != target_dtype:
        return tensor.to(target_dtype)
    return tensor


@overload
def cast_to_module_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor: ...
@overload
def cast_to_module_dtype(tensor: None, module: nn.Module) -> None: ...


def cast_to_module_dtype(tensor: torch.Tensor | None, module: nn.Module) -> torch.Tensor | None:
    return cast_float_inputs(tensor, _get_module_dtype(module))


@dataclass
class AudioEncoderOutput(ModelOutput):
    audio_embeds: torch.Tensor | None = None
    audio_embeds_mask: torch.Tensor | None = None
    encoder_cache: Any = None


@dataclass
class _PredictCodesCudaGraphRunner:
    static_inputs_embeds: torch.Tensor
    static_outputs: torch.Tensor
    past_key_values: StaticCache
    graph: torch.cuda.CUDAGraph


@dataclass
class AudioTokenizerOutput(ModelOutput):
    audio_codes: torch.Tensor | None = None
    audio_codes_mask: torch.Tensor | None = None
    encoder_cache: tuple[Cache, MimiConv1dPaddingCache] | None = None
    mimi_features: torch.Tensor | None = None  # [batch_size, num_frames, 512] pre-quantization features


@dataclass
class AudioDecoderOutput(ModelOutput):
    audio: torch.Tensor
    decoder_cache: tuple[Cache, MimiConv1dPaddingCache, MimiConvTranspose1dPaddingCache] | None = None


class EmbeddingAdaptorConfig(PretrainedConfig):
    model_type = "embedding_adaptor"

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 4096,
        output_time_scale: float = 1.0,
        num_layers: int = 1,
        hidden_size: int | None = None,
        decoder_config: dict[str, Any] | Qwen3Config | None = None,
        use_post_norm: bool = False,
        norm_eps: float = 1e-6,
        post_norm_init_scale: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.output_time_scale = output_time_scale
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_post_norm = use_post_norm
        self.norm_eps = norm_eps
        self.post_norm_init_scale = post_norm_init_scale

        # Parse decoder_config for transformer adaptor mode
        if isinstance(decoder_config, dict):
            decoder_config = Qwen3Config(**decoder_config)
        self.decoder_config = decoder_config


@dataclass
class EmbeddingAdaptorOutput(ModelOutput):
    outputs_embeds: torch.Tensor
    mask: torch.Tensor | None = None


class EmbeddingAdaptor(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        output_time_scale: float = 1.0,
        num_layers: int = 1,
        hidden_size: int | None = None,
        decoder_config: Qwen3Config | None = None,
        use_post_norm: bool = False,
        norm_eps: float = 1e-6,
        post_norm_init_scale: float | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.output_time_scale = output_time_scale
        self.decoder_config = decoder_config

        if output_time_scale >= 1:
            scale = int(output_time_scale)
            assert scale == output_time_scale, (
                f"`output_time_scale` must be an integer when >= 1, got `{output_time_scale}`."
            )
            proj_input_size = input_size
            final_output_size = output_size * scale
        else:
            scale = int(1 / output_time_scale)
            assert scale == 1 / output_time_scale, (
                f"`1/output_time_scale` must be an integer when < 1, got `{output_time_scale}`."
            )
            proj_input_size = input_size * scale
            final_output_size = output_size

        # Check if we should use transformer mode
        if decoder_config is not None:
            # Transformer adaptor mode
            self.is_linear = False
            decoder_hidden_size = decoder_config.hidden_size
            self.input_proj = nn.Linear(
                proj_input_size,
                int(decoder_hidden_size * output_time_scale),
                bias=False,
                dtype=dtype,
            )
            self.decoder = Qwen3Model._from_config(decoder_config, dtype=dtype)
            # Remove unused embedding layer to save memory
            del self.decoder.embed_tokens
            self.decoder.embed_tokens = None  # type: ignore
            self.output_proj = nn.Linear(decoder_hidden_size, output_size, bias=False, dtype=dtype)
        elif num_layers == 1:
            # MLP mode (1 layer)
            self.is_linear = True
            self.proj = nn.Linear(proj_input_size, final_output_size, bias=False, dtype=dtype)
        elif num_layers == 2:
            # MLP mode (2 layers)
            self.is_linear = True
            hidden = hidden_size or final_output_size
            self.proj = nn.Sequential(
                nn.Linear(proj_input_size, hidden, bias=False, dtype=dtype),
                nn.GELU(),
                nn.Linear(hidden, final_output_size, bias=False, dtype=dtype),
            )
        else:
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")

        self.post_norm = nn.RMSNorm(output_size, eps=norm_eps, dtype=dtype) if use_post_norm else None
        if self.post_norm is not None and post_norm_init_scale is not None:
            self.post_norm.weight.data.fill_(post_norm_init_scale)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EmbeddingAdaptorOutput:
        batch_size, seq_length, _ = inputs.shape

        if self.output_time_scale >= 1:
            scale = int(self.output_time_scale)

            if self.is_linear:
                # MLP mode
                outputs_embeds = self.proj(inputs)
            else:
                # Transformer mode
                inputs_embeds = self.input_proj(inputs)
                # Convert mask to attention mask format if provided
                attention_mask = mask.to(inputs_embeds.dtype) if mask is not None else None
                decoder_outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                outputs_embeds = self.output_proj(decoder_outputs.last_hidden_state)

            outputs_embeds = outputs_embeds.view(batch_size, seq_length * scale, self.output_size)

            if mask is not None:
                output_mask = mask.repeat_interleave(scale, dim=1)
            else:
                output_mask = None
        else:
            scale = int(1 / self.output_time_scale)
            remainder = seq_length % scale
            if remainder != 0:
                padding_length = scale - remainder
                last_embed = inputs[:, -1:].expand(-1, padding_length, -1)
                inputs = torch.cat([inputs, last_embed], dim=1)
                if mask is not None:
                    mask = F.pad(mask, (0, padding_length), value=False)

            new_seq_length = inputs.shape[1] // scale
            inputs = inputs.view(batch_size, new_seq_length, scale * self.input_size)

            if self.is_linear:
                # MLP mode
                outputs_embeds = self.proj(inputs)
            else:
                # Transformer mode
                inputs_embeds = self.input_proj(inputs)
                attention_mask = mask.to(inputs_embeds.dtype) if mask is not None else None
                decoder_outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                outputs_embeds = self.output_proj(decoder_outputs.last_hidden_state)

            if mask is not None:
                output_mask = mask.view(batch_size, new_seq_length, scale).any(dim=-1)
            else:
                output_mask = None

        if self.post_norm is not None:
            outputs_embeds = self.post_norm(outputs_embeds)

        return EmbeddingAdaptorOutput(outputs_embeds=outputs_embeds, mask=output_mask)


class RaonCodePredictorModelForConditionalGeneration(Qwen3OmniMoePreTrainedModel, GenerationMixin):  # type: ignore
    """Code predictor for autoregressive audio code generation with fused codec embedding."""

    config_class: type[Qwen3OmniMoeTalkerCodePredictorConfig] = Qwen3OmniMoeTalkerCodePredictorConfig  # type: ignore[assignment]

    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig):
        super().__init__(config)
        self.num_code_groups = config.num_code_groups
        self.model = Qwen3OmniMoeTalkerCodePredictorModel._from_config(config, dtype=self.dtype)
        self._predict_codes_cuda_graph_enabled = False
        self._predict_codes_cuda_graph_runners: dict[
            tuple[int | None, tuple[int, ...], torch.dtype],
            _PredictCodesCudaGraphRunner,
        ] = {}
        self._predict_codes_cuda_graph_captures = 0
        self._predict_codes_cuda_graph_replays = 0
        self._predict_codes_cuda_graph_fallbacks = 0
        self._predict_codes_eager_calls = 0
        input_embeddings = self.model.get_input_embeddings()
        assert isinstance(input_embeddings, nn.ModuleList), "Expected input embeddings to be a ModuleList."
        weights: list[torch.Tensor] = []
        for i in range(self.num_code_groups):
            embed = input_embeddings[i - 1]
            assert isinstance(embed, nn.Embedding)
            weights.append(embed.weight)

        fused_code_embed_weight = torch.cat(weights)
        self.codec_embedding = nn.Embedding(
            fused_code_embed_weight.shape[0],
            fused_code_embed_weight.shape[1],
            dtype=fused_code_embed_weight.dtype,
        )
        with torch.no_grad():
            self.codec_embedding.weight.copy_(fused_code_embed_weight)

        del self.model.codec_embedding
        self.vocab_size = config.vocab_size
        self.fused_lm_head = nn.Parameter(
            torch.randn(
                self.num_code_groups - 1,
                self.vocab_size,
                self.config.hidden_size,
                dtype=self.dtype,
            )
        )
        self.post_init()

    def enable_predict_codes_cuda_graph(self) -> None:
        """Enable lazy CUDA-graph capture for predict_codes() on CUDA inputs."""
        self._predict_codes_cuda_graph_enabled = True

    def _get_predict_codes_cuda_graph_key(
        self,
        inputs_embeds: torch.Tensor,
    ) -> tuple[int | None, tuple[int, ...], torch.dtype]:
        return (inputs_embeds.device.index, tuple(inputs_embeds.shape), inputs_embeds.dtype)

    def _predict_codes_eager(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        self._predict_codes_eager_calls += 1
        past_key_values = StaticCache(self.config, max_cache_len=self.num_code_groups)
        return self.generate_greedy(inputs_embeds=inputs_embeds, past_key_values=past_key_values)

    def _create_predict_codes_cuda_graph_runner(self, inputs_embeds: torch.Tensor) -> _PredictCodesCudaGraphRunner:
        static_inputs_embeds = torch.empty_like(inputs_embeds)
        static_outputs = torch.empty(
            (inputs_embeds.shape[0], self.num_code_groups - 1),
            dtype=torch.int64,
            device=inputs_embeds.device,
        )
        past_key_values = StaticCache(self.config, max_cache_len=self.num_code_groups)

        warmup_stream = torch.cuda.Stream(device=inputs_embeds.device)
        current_stream = torch.cuda.current_stream(device=inputs_embeds.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream):
            static_inputs_embeds.copy_(inputs_embeds)
            for _ in range(3):
                past_key_values.reset()
                static_outputs.copy_(self.generate_greedy(static_inputs_embeds, past_key_values))
        current_stream.wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            past_key_values.reset()
            static_outputs.copy_(self.generate_greedy(static_inputs_embeds, past_key_values))

        self._predict_codes_cuda_graph_captures += 1
        logger.info(
            "Captured CUDA graph for code predictor shape=%s device=%s total_captures=%d",
            tuple(inputs_embeds.shape),
            inputs_embeds.device,
            self._predict_codes_cuda_graph_captures,
        )
        return _PredictCodesCudaGraphRunner(
            static_inputs_embeds=static_inputs_embeds,
            static_outputs=static_outputs,
            past_key_values=past_key_values,
            graph=graph,
        )

    def _predict_codes_cuda_graph(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        key = self._get_predict_codes_cuda_graph_key(inputs_embeds)
        runner = self._predict_codes_cuda_graph_runners.get(key)
        if runner is None:
            runner = self._create_predict_codes_cuda_graph_runner(inputs_embeds)
            self._predict_codes_cuda_graph_runners[key] = runner

        runner.static_inputs_embeds.copy_(inputs_embeds)
        self._predict_codes_cuda_graph_replays += 1
        runner.graph.replay()
        return runner.static_outputs.clone()

    def get_predict_codes_runtime_stats(self) -> dict[str, int | bool]:
        """Expose CUDA-graph capture/replay counters for worker health."""
        return {
            "cuda_graph_enabled": self._predict_codes_cuda_graph_enabled,
            "runner_count": len(self._predict_codes_cuda_graph_runners),
            "captures": self._predict_codes_cuda_graph_captures,
            "replays": self._predict_codes_cuda_graph_replays,
            "fallbacks": self._predict_codes_cuda_graph_fallbacks,
            "eager_calls": self._predict_codes_eager_calls,
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        generation_steps: int | None = None,
        **kwargs: Any,
    ) -> Qwen3OmniMoeTalkerCodePredictorOutputWithPast:
        inputs_embeds = cast_to_module_dtype(inputs_embeds, self)
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            generation_steps = inputs_embeds.shape[1] - 2
        else:
            assert input_ids is not None and generation_steps is not None, f"{input_ids=}, {generation_steps=}"
            inputs_embeds = self.get_input_embeddings()(input_ids + generation_steps * self.vocab_size)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        logits = F.linear(outputs.last_hidden_state, self.fused_lm_head[generation_steps])
        return Qwen3OmniMoeTalkerCodePredictorOutputWithPast(
            logits=logits,  # type: ignore
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            generation_steps=generation_steps + 1,
        )

    def parallel_forward(self, hidden_embeds: torch.Tensor, audio_codes: torch.Tensor) -> torch.Tensor:
        """Predict all code groups in parallel given hidden states and teacher-forced codes.

        Args:
            hidden_embeds: Hidden states from the LM. Shape: [batch_size, hidden_size].
                Dtype: float.
            audio_codes: Teacher-forced audio codes (all but last group).
                Shape: [batch_size, num_code_groups]. Dtype: long.

        Returns:
            Logits for the next code group at each position. Shape: [batch_size,
            num_code_groups - 1, vocab_size]. Dtype: float.
        """
        hidden_embeds = cast_to_module_dtype(hidden_embeds, self)
        generation_step = torch.arange(self.config.num_code_groups - 1, device=audio_codes.device)
        audio_code_embeds = self.codec_embedding(audio_codes[:, :-1] + generation_step * self.vocab_size)
        inputs_embeds = torch.cat((hidden_embeds[:, None], audio_code_embeds), dim=1).contiguous()
        last_hidden_state = self.model(inputs_embeds=inputs_embeds).last_hidden_state
        logits: torch.Tensor = einsum(last_hidden_state[:, 1:], self.fused_lm_head, "b s h, s c h -> b s c")
        return logits

    def generate_greedy(self, inputs_embeds: torch.Tensor, past_key_values: StaticCache) -> torch.Tensor:
        """Generate audio codes greedily given initial embeddings and KV cache.

        Args:
            inputs_embeds: Initial input embeddings. Shape: [batch_size, seq_length,
                hidden_size]. Dtype: float.
            past_key_values: StaticCache holding past KV for incremental decoding.

        Returns:
            Greedily sampled code sequence. Shape: [batch_size, num_code_groups - 1].
            Dtype: long.
        """
        cache_position = torch.arange(2, device=inputs_embeds.device)
        optional_input_ids: torch.Tensor | None = None
        optional_inputs_embeds: torch.Tensor | None = inputs_embeds
        sequences = torch.empty(
            (inputs_embeds.shape[0], self.num_code_groups - 1),
            dtype=torch.int64,
            device=inputs_embeds.device,
        )
        for i in range(self.num_code_groups - 1):
            logits: torch.Tensor = self(
                input_ids=optional_input_ids,
                inputs_embeds=optional_inputs_embeds,
                past_key_values=past_key_values,
                cache_position=cache_position,
                generation_steps=i,
            ).logits
            optional_inputs_embeds = None
            optional_input_ids = logits[:, -1:].argmax(dim=-1)
            cache_position = cache_position[-1:] + 1
            sequences[:, i] = optional_input_ids[:, -1]

        return sequences

    def _update_model_kwargs_for_generation(  # type: ignore
        self,
        outputs: Qwen3OmniMoeTalkerCodePredictorOutputWithPast,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        model_kwargs["generation_steps"] = outputs.generation_steps
        return model_kwargs

    def predict_codes(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Predict full audio code sequence from input embeddings via greedy generation.

        Args:
            inputs_embeds: Input embeddings. Shape: [batch_size, seq_length, hidden_size].
                Dtype: float.

        Returns:
            Predicted audio codes. Shape: [batch_size, num_code_groups - 1]. Dtype: long.
        """
        inputs_embeds = cast_to_module_dtype(inputs_embeds, self)
        assert inputs_embeds is not None
        inputs_embeds = inputs_embeds.contiguous()

        if (
            self._predict_codes_cuda_graph_enabled
            and inputs_embeds.is_cuda
            and not torch.is_grad_enabled()
        ):
            try:
                return self._predict_codes_cuda_graph(inputs_embeds)
            except Exception:
                self._predict_codes_cuda_graph_fallbacks += 1
                logger.exception(
                    "Falling back to eager code predictor after CUDA graph failure; disabling future capture"
                )
                self._predict_codes_cuda_graph_enabled = False
                self._predict_codes_cuda_graph_runners.clear()

        return self._predict_codes_eager(inputs_embeds)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the fused codec embedding layer."""
        return self.codec_embedding


class SpeakerEncoderConfig(PretrainedConfig):
    """Configuration for SpeakerEncoder: input/output sizes, attention heads, and frame window."""

    model_type = "speaker_encoder"

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 4096,
        num_heads: int = 8,
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        frame_rate: float = 12.5,
        encoder_type: str = "from_scratch",
        pretrained_model_id: str | None = None,
        pretrained_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        self.frame_rate = frame_rate
        self.encoder_type = encoder_type
        self.pretrained_model_id = pretrained_model_id
        self.pretrained_dim = pretrained_dim


class SpeakerEncoder(nn.Module):
    """Attention-pooling speaker encoder that produces a fixed-size speaker embedding from variable-length mimi features.

    Uses a learned query token with multi-head attention to pool over the
    time dimension, followed by a projection MLP. Inputs longer than
    ``max_frames`` are truncated for the deployed service runtime.
    """

    def __init__(self, config: SpeakerEncoderConfig, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.min_frames = int(config.min_seconds * config.frame_rate)
        self.max_frames = int(config.max_seconds * config.frame_rate)

        self.query = nn.Parameter(torch.randn(1, 1, config.input_size, dtype=dtype) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=config.input_size,
            num_heads=config.num_heads,
            batch_first=True,
            dtype=dtype,
        )

        self.post_attn = nn.Sequential(
            nn.LayerNorm(config.input_size, dtype=dtype),
            nn.Linear(config.input_size, config.input_size, dtype=dtype),
            nn.GELU(),
            nn.Linear(config.input_size, config.output_size, bias=False, dtype=dtype),
        )

    def forward(
        self,
        mimi_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute speaker embedding from mimi features via attention pooling.

        Args:
            mimi_features: Pre-quantization mimi features. Shape: [batch_size,
                num_frames, feature_dim]. Dtype: float.
            mask: Optional validity mask. Shape: [batch_size, num_frames]. Dtype: bool.

        Returns:
            Speaker embedding. Shape: [batch_size, 1, output_size]. Dtype: float.
        """
        B, T, D = mimi_features.shape

        if T > self.max_frames:
            mimi_features = mimi_features[:, : self.max_frames, :]
            if mask is not None:
                mask = mask[:, : self.max_frames]

        query = self.query.expand(B, -1, -1)
        key_padding_mask = ~mask if mask is not None else None

        pooled, _ = self.attn(
            query=query,
            key=mimi_features,
            value=mimi_features,
            key_padding_mask=key_padding_mask,
        )

        output = self.post_attn(pooled)

        return output


PRETRAINED_ENCODER_TYPES = {"ecapa_tdnn"}


class PretrainedSpeakerEncoder(nn.Module):
    """Frozen pretrained speaker encoder with a trainable projection.

    This encoder consumes raw 24kHz audio, resamples internally to 16kHz, runs
    a frozen SpeechBrain ECAPA model, and projects the pretrained embedding to
    the duplex hidden size.
    """

    def __init__(self, config: SpeakerEncoderConfig, dtype: torch.dtype | None = None) -> None:
        """Initialize ECAPA speaker encoder wrapper.

        Args:
            config: Speaker encoder configuration.
            dtype: Dtype for the trainable projection layer.
        """
        super().__init__()
        assert config.encoder_type in PRETRAINED_ENCODER_TYPES, (
            f"Unknown pretrained encoder_type: {config.encoder_type}. Expected one of {PRETRAINED_ENCODER_TYPES}."
        )
        assert config.pretrained_dim is not None, (
            f"The `pretrained_dim` attribute must be set for encoder_type={config.encoder_type}."
        )

        self.encoder_type = config.encoder_type
        self.pretrained_model_id = config.pretrained_model_id
        self.pretrained_dim = config.pretrained_dim
        self.output_size = config.output_size
        self.source_sample_rate = 24000
        self.target_sample_rate = 16000
        self.min_seconds = config.min_seconds
        self.max_seconds = config.max_seconds

        self.projection = nn.Linear(config.pretrained_dim, config.output_size, bias=False, dtype=dtype)

        # Backend is loaded lazily on first forward to avoid redundant loads.
        self._backend: Any = None
        self._backend_device = torch.device("cpu")

    def _load_backend(self) -> None:
        """Load frozen ECAPA backend on CPU.

        Multi-node safe loading strategy:

        1. Global rank 0 downloads to a shared filesystem cache via
           ``huggingface_hub.snapshot_download``, then writes a ``.done``
           sentinel.  All other ranks poll for the sentinel (10 min timeout).
        2. Each rank copies the shared cache to its own node-local directory
           (``/tmp/speechbrain_ecapa_rank{local_rank}_pid{pid}``) and loads with
           ``source == savedir`` so SpeechBrain finds all files in place and
           skips symlink creation — avoiding cross-rank filesystem races.

        The shared cache root is read from ``SPEECHBRAIN_ECAPA_SAVEDIR`` env
        var, defaulting to ``/tmp/speechbrain_ecapa_cache``.
        """
        if self.encoder_type != "ecapa_tdnn":
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        import shutil
        import time

        _ensure_torchaudio_speechbrain_compat()
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        model_id = self.pretrained_model_id or "speechbrain/spkrec-ecapa-voxceleb"
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        default_cache = "/tmp/speechbrain_ecapa_cache"
        fallback_cache = "/tmp/speechbrain_ecapa_cache"
        cache_root = os.environ.get("SPEECHBRAIN_ECAPA_SAVEDIR", default_cache)
        try:
            os.makedirs(cache_root, exist_ok=True)
        except OSError:
            cache_root = fallback_cache
            os.makedirs(cache_root, exist_ok=True)

        shared_source = os.path.join(cache_root, model_id.replace("/", "_"))
        done_file = f"{shared_source}.done"

        # Step 1: global rank 0 downloads to the shared cache.
        if rank == 0 and not os.path.exists(done_file):
            os.makedirs(shared_source, exist_ok=True)
            from huggingface_hub import snapshot_download  # type: ignore

            snapshot_download(model_id, local_dir=shared_source)
            with open(done_file, "w") as f:
                f.write("done")

        if rank != 0:
            for _ in range(600):
                if os.path.exists(done_file):
                    break
                time.sleep(1)

        # Step 2: each rank copies to its own node-local directory and loads
        # with source == savedir so SpeechBrain finds all files in place and
        # skips symlink creation (avoiding cross-rank filesystem races).
        # Include PID to avoid collisions between Ray workers sharing the same
        # LOCAL_RANK (e.g. multiple inference workers on the same node).
        local_dir = f"/tmp/speechbrain_ecapa_rank{local_rank}_pid{os.getpid()}"
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        shutil.copytree(
            shared_source,
            local_dir,
            ignore=shutil.ignore_patterns(".cache", ".git*"),
        )

        backend = EncoderClassifier.from_hparams(
            source=local_dir,
            savedir=local_dir,
            run_opts={"device": "cpu"},
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        object.__setattr__(self, "_backend", backend)
        if hasattr(self._backend, "parameters"):
            for param in self._backend.parameters():
                param.requires_grad = False

        self._backend_device = torch.device("cpu")

    def _ensure_backend_device(self) -> None:
        """Move backend to the projection device if needed."""
        target_device = self.projection.weight.device
        if self._backend is None:
            self._load_backend()
        if self._backend_device == target_device:
            return

        # SpeechBrain Pretrained keeps modules in backend.mods.
        self._backend.device = str(target_device)
        for mod in self._backend.mods.values():
            mod.to(target_device)
        self._backend_device = target_device

    def _extract_embedding(self, audio_16k: torch.Tensor, lengths_16k: torch.Tensor) -> torch.Tensor:
        """Extract frozen ECAPA embeddings from 16kHz audio.

        Args:
            audio_16k: Resampled mono audio. Shape: [batch_size, num_samples_16k]. Dtype: float.
            lengths_16k: Valid sample lengths. Shape: [batch_size]. Dtype: long.

        Returns:
            Pretrained embedding. Shape: [batch_size, pretrained_dim]. Dtype: float.
        """
        min_samples_16k = 1600
        batch_size = audio_16k.shape[0]
        if audio_16k.shape[1] < min_samples_16k:
            return torch.zeros(
                batch_size,
                self.pretrained_dim,
                device=audio_16k.device,
                dtype=audio_16k.dtype,
            )

        lengths_16k = lengths_16k.clamp(min=min_samples_16k)
        reference_length = max(1, int(audio_16k.shape[1]))
        wav_lens = lengths_16k.float() / float(reference_length)
        wav_lens = wav_lens.clamp(max=1.0)
        autocast_dtype: torch.dtype | None = None
        if audio_16k.device.type == "cuda" and self.projection.weight.dtype in {torch.float16, torch.bfloat16}:
            autocast_dtype = self.projection.weight.dtype

        with torch.no_grad():
            with torch.autocast(
                device_type=audio_16k.device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                embeddings = self._backend.encode_batch(audio_16k, wav_lens)

        return embeddings.squeeze(1)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute speaker embedding from raw 24kHz audio.

        Args:
            audio: Raw mono waveform at 24kHz. Shape: [batch_size, num_samples]. Dtype: float.
            audio_lengths: Valid sample lengths. Shape: [batch_size]. Dtype: long.

        Returns:
            Speaker embedding. Shape: [batch_size, 1, output_size]. Dtype: float.
        """
        self._ensure_backend_device()
        import torchaudio.functional

        audio_input = audio
        if (
            audio_input.device.type == "cuda"
            and self.projection.weight.dtype in {torch.float16, torch.bfloat16}
            and audio_input.dtype != self.projection.weight.dtype
        ):
            audio_input = audio_input.to(dtype=self.projection.weight.dtype)
        elif audio_input.dtype not in {torch.float32, torch.float64, torch.float16, torch.bfloat16}:
            audio_input = audio_input.float()

        audio_16k = torchaudio.functional.resample(
            audio_input,
            orig_freq=self.source_sample_rate,
            new_freq=self.target_sample_rate,
        )
        lengths_16k = (audio_lengths.float() * self.target_sample_rate / self.source_sample_rate).long()

        max_samples = int(self.max_seconds * self.target_sample_rate)
        if audio_16k.shape[1] > max_samples:
            audio_16k = audio_16k[:, :max_samples]
            lengths_16k = lengths_16k.clamp(max=max_samples)

        raw_embedding = self._extract_embedding(audio_16k, lengths_16k)
        raw_embedding = raw_embedding.to(dtype=self.projection.weight.dtype)
        projected = self.projection(raw_embedding)
        return projected.unsqueeze(1)


def is_pretrained_speaker_encoder(encoder: SpeakerEncoder | PretrainedSpeakerEncoder) -> bool:
    """Check if encoder is a pretrained speaker encoder that takes raw audio input.

    Pretrained speaker encoders accept ``(audio, audio_lengths)`` rather than
    ``(mimi_features, mask)`` like the from-scratch ``SpeakerEncoder``.
    """
    return isinstance(encoder, PretrainedSpeakerEncoder)


class ThinkerToTalkerProjection(nn.Module):
    """Projection from thinker hidden states to talker input space.

    Supports two modes:
    - ``"linear"``: RMSNorm (optional) followed by a single linear layer (no bias).
    - ``"mlp"``: Optional RMSNorm followed by a two-layer MLP with SiLU activation
      and bias, matching the ``Qwen3OmniMoeTalkerResizeMLP`` design from Qwen3-Omni.

    Args:
        thinker_hidden_size: Dimension of thinker hidden states.
        talker_hidden_size: Dimension of talker input.
        intermediate_size: Hidden dimension for the MLP mode. Required when
            ``mode="mlp"``, ignored for ``"linear"``.
        mode: Projection type — ``"linear"`` or ``"mlp"``.
        use_norm: If True, apply RMSNorm before projection (both modes).
        rms_norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        thinker_hidden_size: int,
        talker_hidden_size: int,
        intermediate_size: int | None = None,
        mode: str = "linear",
        use_norm: bool = True,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.norm: nn.RMSNorm | None = nn.RMSNorm(thinker_hidden_size, eps=rms_norm_eps) if use_norm else None
        if mode == "mlp":
            assert intermediate_size is not None, "intermediate_size is required for mlp mode."
            self.linear_fc1 = nn.Linear(thinker_hidden_size, intermediate_size, bias=True)
            self.linear_fc2 = nn.Linear(intermediate_size, talker_hidden_size, bias=True)
            self.act_fn = nn.SiLU()
            self.linear = None
        else:
            self.linear = nn.Linear(thinker_hidden_size, talker_hidden_size, bias=False)
            self.linear_fc1 = None
            self.linear_fc2 = None
            self.act_fn = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project thinker hidden states to talker input space.

        Args:
            hidden_states: Thinker hidden states.
                Shape: [batch_size, seq_len, thinker_hidden_size]. Dtype: float.

        Returns:
            Projected hidden states. Shape: [batch_size, seq_len, talker_hidden_size]. Dtype: float.
        """
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.mode == "mlp":
            return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))
        return self.linear(hidden_states)


def build_speaker_encoder(
    config: SpeakerEncoderConfig,
    dtype: torch.dtype | None = None,
) -> SpeakerEncoder | PretrainedSpeakerEncoder:
    """Build speaker encoder instance from configuration.

    Args:
        config: Speaker encoder configuration.
        dtype: Target floating-point dtype.

    Returns:
        Speaker encoder module for from-scratch or pretrained ECAPA mode.
    """
    encoder_type = getattr(config, "encoder_type", "from_scratch")
    if encoder_type == "from_scratch":
        return SpeakerEncoder(config, dtype=dtype)
    return PretrainedSpeakerEncoder(config, dtype=dtype)
