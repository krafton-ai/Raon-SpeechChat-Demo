"""Core duplex model definition for inference and audio codec integration."""

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional
from torch import nn
from transformers import (
    Cache,
    DynamicCache,
    MimiConfig,
    PretrainedConfig,
    PreTrainedModel,
    Qwen3Config,
    Qwen3Model,
    StaticCache,
)
from transformers.models.mimi.modeling_mimi import MimiEncoderOutput
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorConfig
from transformers.utils.generic import ModelOutput

from .audio_encoder.output import CausalAudioEncoderOutput
from .audio_encoder.streaming_mimi import (
    MimiConv1dPaddingCache,
    MimiConvTranspose1dPaddingCache,
    StreamingMimiDecoderOutput,
    StreamingMimiModel,
)
from .inference import RaonInferenceModel
from .modules import (
    AudioDecoderOutput,
    AudioEncoderOutput,
    AudioTokenizerOutput,
    EmbeddingAdaptor,
    EmbeddingAdaptorConfig,
    EmbeddingAdaptorOutput,
    PretrainedSpeakerEncoder,
    RaonCodePredictorModelForConditionalGeneration,
    SpeakerEncoder,
    SpeakerEncoderConfig,
    ThinkerToTalkerProjection,
    build_speaker_encoder,
    cast_float_inputs,
    cast_to_module_dtype,
    is_pretrained_speaker_encoder,
)
from .special_tokens import (
    AUDIO_OUTPUT_BC,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_OUTPUT_SIL,
    SPEAKER_EMBEDDING_PLACEHOLDER,
)
from .voxtral import VoxtralRealtimeEncoderConfig, VoxtralWrapper

TEXT_MODEL_CONFIGS: dict[str, type[PretrainedConfig]] = {
    Qwen3Config.model_type: Qwen3Config
}
TEXT_MODELS: dict[str, type[PreTrainedModel]] = {
    Qwen3Config.model_type: Qwen3Model
}


class RaonConfig(PretrainedConfig):
    """Configuration class for RaonModel."""

    model_type = "qwen_duplex"
    has_no_defaults_at_init = True
    text_model_config: PretrainedConfig
    audio_encoder_config: VoxtralRealtimeEncoderConfig
    audio_tokenizer_config: MimiConfig
    input_adaptor_config: EmbeddingAdaptorConfig
    output_adaptor_config: EmbeddingAdaptorConfig
    code_predictor_config: Qwen3OmniMoeTalkerCodePredictorConfig
    speaker_encoder_config: SpeakerEncoderConfig | None
    # Note: speaker_encoder_config is intentionally excluded from sub_configs.
    # It is optional (can be None), and transformers' _get_dtype unconditionally
    # calls sub_config.dtype on every entry, which crashes on None.
    # Deserialization from dict is handled in __init__ instead.
    sub_configs = {
        "text_model_config": PretrainedConfig,
        "audio_encoder_config": PretrainedConfig,
        "audio_tokenizer_config": PretrainedConfig,
        "input_adaptor_config": EmbeddingAdaptorConfig,
        "output_adaptor_config": EmbeddingAdaptorConfig,
        "code_predictor_config": Qwen3OmniMoeTalkerCodePredictorConfig,
    }

    def __init__(
        self,
        *,
        text_model_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_encoder_config: dict[str, Any] | VoxtralRealtimeEncoderConfig | None = None,
        audio_tokenizer_config: dict[str, Any] | MimiConfig | None = None,
        input_adaptor_config: dict[str, Any] | EmbeddingAdaptorConfig | None = None,
        output_adaptor_config: dict[str, Any] | EmbeddingAdaptorConfig | None = None,
        code_predictor_config: dict[str, Any] | Qwen3OmniMoeTalkerCodePredictorConfig | None = None,
        speaker_encoder_config: dict[str, Any] | SpeakerEncoderConfig | None = None,
        num_talker_layers: int = 0,
        supports_audio_input: bool = True,
        supports_audio_output: bool = True,
        duplex_pad_token_id: int = AUDIO_OUTPUT_PAD.id,
        duplex_end_pad_token_id: int = AUDIO_OUTPUT_END_PAD.id,
        duplex_sil_token_id: int = AUDIO_OUTPUT_SIL.id,
        duplex_bc_token_id: int = AUDIO_OUTPUT_BC.id,
        use_duplex_end_pad: bool = False,  # default: disabled
        use_sil_token: bool = False,
        use_backchannel_token: bool = False,
        no_audio_in_sil: bool = False,
        sequence_mode: Literal["tua", "uta"] | None = None,
        acoustic_delay: list[int] | int | None = None,
        aut_is_causal: bool = False,
        proj_code_bias: bool = False,
        accept_hidden_layer: int = -1,
        talker_config: dict[str, Any] | PretrainedConfig | None = None,
        thinker_to_talker_pre_norm: bool = False,
        input_num_code_groups: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        assert text_model_config is not None, "RaonConfig: `text_model_config` is required."
        assert audio_encoder_config is not None, "RaonConfig: `audio_encoder_config` is required."
        assert audio_tokenizer_config is not None, "RaonConfig: `audio_tokenizer_config` is required."
        assert input_adaptor_config is not None, "RaonConfig: `input_adaptor_config` is required."
        assert output_adaptor_config is not None, "RaonConfig: `output_adaptor_config` is required."
        assert code_predictor_config is not None, "RaonConfig: `code_predictor_config` is required."

        if isinstance(text_model_config, dict):
            model_type = text_model_config.get("model_type", Qwen3Config.model_type)
            text_model_config = TEXT_MODEL_CONFIGS[model_type](**text_model_config)

        if isinstance(audio_encoder_config, dict):
            model_type = audio_encoder_config.get("model_type", VoxtralRealtimeEncoderConfig.model_type)
            if model_type != VoxtralRealtimeEncoderConfig.model_type:
                raise ValueError(
                    "RaonConfig only supports voxtral_realtime_encoder audio input for the deployed service. "
                    f"Got audio_encoder_config.model_type={model_type!r}."
                )
            audio_encoder_config = VoxtralRealtimeEncoderConfig(**audio_encoder_config)

        if isinstance(audio_tokenizer_config, dict):
            model_type = audio_tokenizer_config.get("model_type", MimiConfig.model_type)
            if model_type != MimiConfig.model_type:
                raise ValueError(
                    "RaonConfig only supports Mimi audio tokenization for the deployed service. "
                    f"Got audio_tokenizer_config.model_type={model_type!r}."
                )
            audio_tokenizer_config = MimiConfig(**audio_tokenizer_config)

        if isinstance(input_adaptor_config, dict):
            input_adaptor_config = EmbeddingAdaptorConfig(**input_adaptor_config)

        if isinstance(output_adaptor_config, dict):
            output_adaptor_config = EmbeddingAdaptorConfig(**output_adaptor_config)

        if isinstance(code_predictor_config, dict):
            code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(**code_predictor_config)

        if isinstance(speaker_encoder_config, dict):
            speaker_encoder_config = SpeakerEncoderConfig(**speaker_encoder_config)

        if isinstance(talker_config, dict):
            talker_model_type = talker_config.get("model_type", Qwen3Config.model_type)
            talker_config = TEXT_MODEL_CONFIGS[talker_model_type](**talker_config)

        # Auto-derive talker_config from text_model_config for older checkpoints that
        # have num_talker_layers but no explicit talker_config.
        if talker_config is None and supports_audio_output and num_talker_layers > 0:
            talker_dict = (
                text_model_config.to_dict() if isinstance(text_model_config, PretrainedConfig) else dict(text_model_config)
            )
            talker_dict["num_hidden_layers"] = num_talker_layers
            # Truncate per-layer config lists (e.g. layer_types) to match num_talker_layers.
            if "layer_types" in talker_dict:
                talker_dict["layer_types"] = talker_dict["layer_types"][:num_talker_layers]
            talker_model_type = talker_dict.get("model_type", Qwen3Config.model_type)
            talker_config = TEXT_MODEL_CONFIGS[talker_model_type](**talker_dict)

        assert isinstance(audio_encoder_config, VoxtralRealtimeEncoderConfig), (
            "audio_encoder_config must be VoxtralRealtimeEncoderConfig."
        )
        assert isinstance(audio_tokenizer_config, MimiConfig), "audio_tokenizer_config must be MimiConfig."
        assert isinstance(input_adaptor_config, EmbeddingAdaptorConfig), (
            "input_adaptor_config must be EmbeddingAdaptorConfig."
        )
        assert isinstance(output_adaptor_config, EmbeddingAdaptorConfig), (
            "output_adaptor_config must be EmbeddingAdaptorConfig."
        )
        assert isinstance(code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig), (
            "code_predictor_config must be Qwen3OmniMoeTalkerCodePredictorConfig."
        )
        assert isinstance(text_model_config, PretrainedConfig), "text_model_config must be PretrainedConfig."
        assert speaker_encoder_config is None or isinstance(speaker_encoder_config, SpeakerEncoderConfig), (
            "speaker_encoder_config must be None or SpeakerEncoderConfig."
        )

        self.text_model_config = text_model_config
        self.audio_encoder_config = audio_encoder_config
        self.audio_tokenizer_config = audio_tokenizer_config
        self.input_adaptor_config = input_adaptor_config
        self.output_adaptor_config = output_adaptor_config
        self.code_predictor_config = code_predictor_config
        self.speaker_encoder_config = speaker_encoder_config
        self.num_talker_layers = num_talker_layers
        self.supports_audio_input = supports_audio_input
        self.supports_audio_output = supports_audio_output
        self.duplex_pad_token_id = duplex_pad_token_id
        self.duplex_end_pad_token_id = duplex_end_pad_token_id
        self.duplex_sil_token_id = duplex_sil_token_id
        self.duplex_bc_token_id = duplex_bc_token_id
        self.use_duplex_end_pad = use_duplex_end_pad
        self.use_sil_token = use_sil_token
        self.use_backchannel_token = use_backchannel_token
        self.no_audio_in_sil = no_audio_in_sil
        if sequence_mode not in (None, "tua", "uta"):
            raise ValueError(f"RaonConfig: Unsupported sequence_mode '{sequence_mode}'.")
        self.sequence_mode = sequence_mode
        self.acoustic_delay = acoustic_delay
        self.aut_is_causal = aut_is_causal
        self.proj_code_bias = proj_code_bias
        self.accept_hidden_layer = accept_hidden_layer
        self.talker_config = talker_config
        self.thinker_to_talker_pre_norm = thinker_to_talker_pre_norm
        self.input_num_code_groups = input_num_code_groups

        if supports_audio_output:
            assert talker_config is not None, (
                "RaonConfig: `talker_config` is required when audio output is enabled."
            )
            assert num_talker_layers > 0, (
                "RaonConfig: `num_talker_layers` must be positive when audio output is enabled."
            )

    def _get_non_default_generation_parameters(self) -> dict[str, Any]:
        return {}

    def to_diff_dict(self) -> dict[str, Any]:
        """Return config as a dict suitable for diffing."""
        return self.to_dict()


@dataclass
class RaonModelOutput(ModelOutput):
    """Output container for the deployed inference forward pass."""

    talker_last_hidden_state: torch.Tensor | None = None
    text_logits: torch.Tensor | None = None
    past_key_values: Cache | None = None


class RaonModel(PreTrainedModel, RaonInferenceModel):
    """Core duplex model combining text language model with audio codec."""

    _tied_weights_keys: list[str] = []  # type: ignore
    config_class: type[RaonConfig] = RaonConfig  # type: ignore
    config: RaonConfig

    def __init__(self, config: RaonConfig) -> None:
        super().__init__(config)
        assert config.text_model_config is not None, "Config text_model_config is required."
        assert config.audio_encoder_config is not None, "Config audio_encoder_config is required."
        assert config.audio_tokenizer_config is not None, "Config audio_tokenizer_config is required."
        assert config.input_adaptor_config is not None, "Config input_adaptor_config is required."
        assert config.output_adaptor_config is not None, "Config output_adaptor_config is required."
        assert config.code_predictor_config is not None, "Config code_predictor_config is required."
        assert config.text_model_config.vocab_size is not None, "text_model_config.vocab_size is required."
        assert config.audio_tokenizer_config.codebook_size is not None, "audio_tokenizer_config.codebook_size is required."
        assert config.code_predictor_config.num_code_groups is not None, "code_predictor_config.num_code_groups is required."
        assert config.audio_tokenizer_config.sampling_rate is not None, "audio_tokenizer_config.sampling_rate is required."
        assert config.text_model_config.hidden_size is not None, "text_model_config.hidden_size is required."
        assert config.code_predictor_config.hidden_size is not None, "code_predictor_config.hidden_size is required."
        if getattr(config, "supports_audio_output", True):
            assert config.talker_config is not None, "talker_config is required when audio output is enabled."
            assert config.num_talker_layers > 0, "num_talker_layers must be positive when audio output is enabled."

        self.config = config
        self.hidden_size = int(config.text_model_config.hidden_size)
        self.vocab_size = int(config.text_model_config.vocab_size)
        self.codebook_size = config.audio_tokenizer_config.codebook_size
        self.audio_lm_head_vocab_size = self.codebook_size + 1
        self.num_talker_layers = config.num_talker_layers
        self.supports_audio_input = getattr(config, "supports_audio_input", True)
        self.supports_audio_output = getattr(config, "supports_audio_output", True)
        self.duplex_pad_token_id = getattr(config, "duplex_pad_token_id", AUDIO_OUTPUT_PAD.id)
        self.duplex_end_pad_token_id = getattr(config, "duplex_end_pad_token_id", AUDIO_OUTPUT_END_PAD.id)
        self.duplex_sil_token_id = getattr(config, "duplex_sil_token_id", AUDIO_OUTPUT_SIL.id)
        self.duplex_bc_token_id = getattr(config, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id)
        self.use_duplex_end_pad = getattr(config, "use_duplex_end_pad", False)
        self.use_sil_token = getattr(config, "use_sil_token", False)
        self.use_backchannel_token = getattr(config, "use_backchannel_token", False)
        self.no_audio_in_sil = getattr(config, "no_audio_in_sil", False)
        self.sequence_mode = getattr(config, "sequence_mode", None)

        self.num_code_groups = config.code_predictor_config.num_code_groups
        self.input_num_code_groups = getattr(config, "input_num_code_groups", None) or self.num_code_groups
        self.sampling_rate = config.audio_tokenizer_config.sampling_rate
        assert (frame_rate := config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
            "audio_tokenizer_config._frame_rate is required."
        )
        self.frame_rate = frame_rate

        # Create thinker text_model: num_hidden_layers IS the thinker count (talker is separate).
        total_layers = int(config.text_model_config.num_hidden_layers)
        num_thinker_layers = total_layers
        thinker_text_config = deepcopy(config.text_model_config)
        thinker_text_config.num_hidden_layers = num_thinker_layers
        if hasattr(thinker_text_config, "layer_types") and thinker_text_config.layer_types:
            thinker_text_config.layer_types = thinker_text_config.layer_types[:num_thinker_layers]
        self.text_model = TEXT_MODELS[thinker_text_config.model_type]._from_config(
            thinker_text_config,
            dtype=self.dtype,
        )
        # Propagate attention backend from text model to audio modules before construction.
        # The attention class is selected at __init__ time, so config must be set first.
        _attn_impl = getattr(self.text_model.config, "_attn_implementation", None) or getattr(
            config, "_attn_implementation", None
        )
        if self.supports_audio_input:
            if _attn_impl is not None and isinstance(config.audio_encoder_config, VoxtralRealtimeEncoderConfig):
                config.audio_encoder_config._attn_implementation = _attn_impl
            self.audio_encoder: VoxtralWrapper | None = VoxtralWrapper.from_config(
                config=config.audio_encoder_config,
                dtype=self.dtype,
            )
        else:
            self.audio_encoder = None

        self.aut_is_causal = getattr(config, "aut_is_causal", False)

        if self.supports_audio_output:
            if _attn_impl is not None and isinstance(config.audio_tokenizer_config, MimiConfig):
                config.audio_tokenizer_config._attn_implementation = _attn_impl
            self.audio_tokenizer: StreamingMimiModel | None = StreamingMimiModel._from_config(
                cast(MimiConfig, config.audio_tokenizer_config),
                dtype=self.dtype,
            )
        else:
            self.audio_tokenizer = None
        if self.supports_audio_input:
            self.input_adaptor: EmbeddingAdaptor | None = EmbeddingAdaptor(
                input_size=config.input_adaptor_config.input_size,
                output_size=config.input_adaptor_config.output_size,
                output_time_scale=config.input_adaptor_config.output_time_scale,
                num_layers=getattr(config.input_adaptor_config, "num_layers", 1),
                hidden_size=getattr(config.input_adaptor_config, "hidden_size", None),
                decoder_config=getattr(config.input_adaptor_config, "decoder_config", None),
                use_post_norm=getattr(config.input_adaptor_config, "use_post_norm", False),
                norm_eps=getattr(config.input_adaptor_config, "norm_eps", 1e-6),
                post_norm_init_scale=getattr(config.input_adaptor_config, "post_norm_init_scale", None),
                dtype=self.dtype,
            )
        else:
            self.input_adaptor = None
        if self.supports_audio_output:
            self.output_adaptor: EmbeddingAdaptor | None = EmbeddingAdaptor(
                input_size=config.output_adaptor_config.input_size,
                output_size=config.output_adaptor_config.output_size,
                output_time_scale=config.output_adaptor_config.output_time_scale,
                num_layers=getattr(config.output_adaptor_config, "num_layers", 1),
                hidden_size=getattr(config.output_adaptor_config, "hidden_size", None),
                decoder_config=getattr(config.output_adaptor_config, "decoder_config", None),
                use_post_norm=getattr(config.output_adaptor_config, "use_post_norm", False),
                norm_eps=getattr(config.output_adaptor_config, "norm_eps", 1e-6),
                post_norm_init_scale=getattr(config.output_adaptor_config, "post_norm_init_scale", None),
                dtype=self.dtype,
            )
        else:
            self.output_adaptor = None
        # Create separate talker model and thinker-to-talker projection (audio output only).
        rms_norm_eps = getattr(config.text_model_config, "rms_norm_eps", 1e-6)
        if self.supports_audio_output:
            resolved_talker_config = config.talker_config
            self.talker: PreTrainedModel | None = TEXT_MODELS[resolved_talker_config.model_type]._from_config(
                resolved_talker_config,
                dtype=self.dtype,
            )
            # Talker only receives inputs_embeds (from thinker_to_talker_proj), never input_ids.
            self.talker.embed_tokens = None  # type: ignore
            talker_hidden_size = int(resolved_talker_config.hidden_size)
            projection_mode = getattr(config, "thinker_to_talker_projection_mode", "linear")
            projection_intermediate_size = getattr(config, "thinker_to_talker_intermediate_size", None)
            if projection_mode == "mlp" and projection_intermediate_size is None:
                projection_intermediate_size = int(resolved_talker_config.intermediate_size)
            self.thinker_to_talker_proj: ThinkerToTalkerProjection | None = ThinkerToTalkerProjection(
                thinker_hidden_size=self.hidden_size,
                talker_hidden_size=talker_hidden_size,
                intermediate_size=projection_intermediate_size,
                mode=projection_mode,
                use_norm=getattr(config, "thinker_to_talker_pre_norm", False),
                rms_norm_eps=rms_norm_eps,
            )
        else:
            self.talker = None
            self.thinker_to_talker_proj = None
            talker_hidden_size = self.hidden_size

        # Resolve accept_hidden_layer: -1 means last thinker layer.
        accept_hidden_layer = getattr(config, "accept_hidden_layer", -1)
        if accept_hidden_layer < 0:
            accept_hidden_layer = num_thinker_layers + accept_hidden_layer
        self.accept_hidden_layer = accept_hidden_layer
        self.thinker_capture_layer_index = (
            accept_hidden_layer  # Index of the thinker layer whose output feeds the talker and audio head.
        )
        self.lm_head = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.vocab_size,
            bias=False,
            dtype=self.dtype,
        )
        if self.supports_audio_output:
            self.audio_lm_head: nn.Linear | None = nn.Linear(
                in_features=talker_hidden_size,
                out_features=self.audio_lm_head_vocab_size,
                bias=False,
                dtype=self.dtype,
            )
            self.proj_code: nn.Linear | None = nn.Linear(
                in_features=talker_hidden_size,
                out_features=config.code_predictor_config.hidden_size,
                bias=config.proj_code_bias,
                dtype=self.dtype,
            )
            self.code_predictor: RaonCodePredictorModelForConditionalGeneration | None = (
                RaonCodePredictorModelForConditionalGeneration._from_config(
                    config.code_predictor_config,
                    dtype=self.dtype,
                )
            )
        else:
            self.audio_lm_head = None
            self.proj_code = None
            self.code_predictor = None

        self.accepted_thinker_hidden_states: torch.Tensor | None = None
        self.register_thinker_capture_hook()

        # Migration hook: remap old unified state_dict (32-layer text_model) to
        # separated thinker (28-layer text_model) + talker (4-layer talker).
        if self.talker is not None:
            self._register_state_dict_migration_hook(num_thinker_layers, config.num_talker_layers)

        # Speaker encoder (optional, for speaker-conditioned TTS)
        if self.supports_audio_output and config.speaker_encoder_config is not None:
            self.speaker_encoder: SpeakerEncoder | PretrainedSpeakerEncoder | None = build_speaker_encoder(
                config.speaker_encoder_config,
                dtype=self.dtype,
            )
            self.is_pretrained_speaker_encoder = is_pretrained_speaker_encoder(self.speaker_encoder)
            self.speaker_token_id: int | None = SPEAKER_EMBEDDING_PLACEHOLDER.id
        else:
            self.speaker_encoder = None
            self.is_pretrained_speaker_encoder = False
            self.speaker_token_id = None

        # Defensive cleanup: when audio output is disabled, ensure all output-side
        # audio modules are hard-disabled even if something assigned them earlier.
        if not self.supports_audio_output:
            self._disable_audio_output_modules()
        if not self.supports_audio_input:
            self._disable_audio_input_modules()

        acoustic_delay = getattr(config, "acoustic_delay", None)
        if acoustic_delay is None:
            self.delays = [0] * self.num_code_groups
        elif isinstance(acoustic_delay, list):
            assert len(acoustic_delay) == self.num_code_groups, (
                f"Expected {self.num_code_groups} delays, got {len(acoustic_delay)}."
            )
            self.delays = acoustic_delay
        elif isinstance(acoustic_delay, int):
            self.delays = [0] + [acoustic_delay] * (self.num_code_groups - 1)
        else:
            raise ValueError(f"Invalid acoustic_delay type: {type(acoustic_delay)}")

        self.max_delay = max(self.delays)

        if self.config.input_adaptor_config.output_time_scale != 1:
            raise NotImplementedError("Only `output_time_scale == 1` is supported.")

        RaonInferenceModel.__init__(self)

    def _disable_audio_output_modules(self) -> None:
        """Force-disable all audio-output and speaker-output modules."""
        self.audio_tokenizer = None
        self.output_adaptor = None
        self.audio_lm_head = None
        self.proj_code = None
        self.code_predictor = None
        self.speaker_encoder = None
        self.is_pretrained_speaker_encoder = False
        self.speaker_token_id = None

    def _disable_audio_input_modules(self) -> None:
        """Force-disable all audio-input modules."""
        self.audio_encoder = None
        self.input_adaptor = None

    def register_thinker_capture_hook(self) -> None:
        """Register a forward hook on the accept_hidden_layer to capture unnormalized hidden states."""

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            self.accepted_thinker_hidden_states = output[0] if isinstance(output, tuple) else output

        cast(list[nn.Module], self.text_model.layers)[self.accept_hidden_layer].register_forward_hook(hook)

    def _register_state_dict_migration_hook(
        self,
        num_thinker_layers: int,
        num_talker_layers: int,
    ) -> None:
        """Register a pre-hook that migrates old unified state_dicts to separated thinker+talker format.

        Old format: ``text_model.layers.{0..N-1}`` where the last ``num_talker_layers`` are talker.
        New format: ``text_model.layers.{0..T-1}`` (thinker) + ``talker.layers.{0..K-1}`` (talker).

        Args:
            num_thinker_layers: Number of thinker layers in the new text_model.
            num_talker_layers: Number of talker layers in the new talker model.
        """

        def migrate(state_dict: dict[str, Any], prefix: str, *_args: Any, **_kwargs: Any) -> None:
            # Check if migration is needed: old state_dict has text_model.layers.{num_thinker_layers}
            sentinel = f"{prefix}text_model.layers.{num_thinker_layers}."
            needs_migration = any(k.startswith(sentinel) for k in state_dict)
            if not needs_migration:
                return

            # Move talker layers: text_model.layers.{T..T+K-1} → talker.layers.{0..K-1}
            for i in range(num_talker_layers):
                old_prefix = f"{prefix}text_model.layers.{num_thinker_layers + i}."
                new_prefix = f"{prefix}talker.layers.{i}."
                for key in list(state_dict):
                    if key.startswith(old_prefix):
                        state_dict[key.replace(old_prefix, new_prefix, 1)] = state_dict.pop(key)

            # Move text_model.norm → talker.norm (was talker's final norm in unified model).
            old_norm = f"{prefix}text_model.norm.weight"
            new_talker_norm = f"{prefix}talker.norm.weight"
            thinker_norm_src = f"{prefix}text_output_norm.weight"
            if old_norm in state_dict and new_talker_norm not in state_dict:
                state_dict[new_talker_norm] = state_dict.pop(old_norm)
                # Restore thinker norm from text_output_norm (which held the thinker copy).
                if thinker_norm_src in state_dict:
                    state_dict[old_norm] = state_dict.pop(thinker_norm_src)

            # Drop talker embed_tokens keys — talker only uses inputs_embeds, never input_ids.
            for key in list(state_dict):
                if key.startswith(f"{prefix}talker.embed_tokens."):
                    del state_dict[key]

            # Drop legacy text_output_norm keys (module removed; text logits now use text_model.norm).
            for key in list(state_dict):
                if key.startswith(f"{prefix}text_output_norm."):
                    del state_dict[key]

        self._register_load_state_dict_pre_hook(migrate)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the text model input embedding layer."""
        assert isinstance(self.text_model.embed_tokens, nn.Embedding), "text_model.embed_tokens must be nn.Embedding."
        return self.text_model.embed_tokens

    def get_proj_code(self) -> nn.Linear:
        """Return the audio code projection layer."""
        assert self.proj_code is not None, "proj_code is unavailable when supports_audio_output is False."
        return self.proj_code

    def _validate_audio_output_inputs(
        self,
        audio_output: torch.Tensor | None,
        audio_output_codes: torch.Tensor | None,
        audio_output_codes_mask: torch.Tensor | None,
    ) -> None:
        """Validate that audio-output inputs are only used when audio output is supported."""
        if self.supports_audio_output:
            return
        if audio_output is not None or audio_output_codes is not None or audio_output_codes_mask is not None:
            raise ValueError(
                "Audio output is disabled (`supports_audio_output=False`), but audio-output inputs were provided."
            )

    def _validate_audio_input_inputs(
        self,
        input_ids: torch.Tensor | None,
        audio_input: torch.Tensor | None,
        audio_input_embeds: torch.Tensor | None,
        audio_input_embeds_mask: torch.Tensor | None,
    ) -> None:
        """Validate that audio-input inputs are only used when audio input is supported."""
        if self.supports_audio_input:
            return
        _ = input_ids, audio_input
        if audio_input_embeds is not None or audio_input_embeds_mask is not None:
            raise ValueError("Audio input is disabled (`supports_audio_input=False`), but audio-input inputs were provided.")

    def get_model(self) -> Self:
        """Return self as the model instance."""
        return self

    def tie_weights(
        self,
        missing_keys: set[str] | None = None,
        recompute_mapping: bool = False,
    ) -> None: ...

    @property
    def all_tied_weights_keys(self) -> dict[str, str]:
        return {}

    def get_audio_input_embeds(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        encoder_past_key_values: StaticCache | None = None,
        conv_padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> AudioEncoderOutput:
        """Encode raw audio to input embeddings via audio encoder and input adaptor.

        Args:
            audio: Raw waveform. Shape: [batch_size, num_channels, num_samples] or
                [batch_size, num_samples] or [num_samples]. Dtype: float. None to return empty output.
            audio_lengths: Valid length per sample. Shape: [batch_size]. Dtype: long.
            sampling_rate: Sample rate of input audio; resampled if different from encoder.
            num_code_groups: Number of code groups (unused, for API compatibility).
            encoder_past_key_values: Cached encoder KV for streaming.
            conv_padding_cache: Cached conv padding for streaming encoder.
            use_streaming: Whether to use streaming mode.

        Returns:
            AudioEncoderOutput with audio_embeds (Shape: [batch_size, num_frames, feature_dim].
            Dtype: float.), audio_embeds_mask (Shape: [batch_size, num_frames]. Dtype: bool.), and
            encoder_cache if streaming.
        """
        if audio is None:
            return AudioEncoderOutput()
        if self.audio_encoder is None:
            raise RuntimeError("audio_encoder is unavailable when supports_audio_input is False.")
        assert self.input_adaptor is not None, "input_adaptor is unavailable when supports_audio_input is False."

        audio = cast_to_module_dtype(audio, self.audio_encoder)
        if audio.ndim == 1:
            audio = audio[None, None]
        elif audio.ndim == 2:
            audio = audio[:, None]

        assert audio.ndim == 3, "Audio tensor must have 3 dimensions [batch_size, num_channels, num_samples]."

        if sampling_rate is not None and sampling_rate != self.audio_encoder.config.sampling_rate:
            assert self.audio_encoder.config.sampling_rate is not None, (
                "audio_encoder.config.sampling_rate is required for resampling."
            )
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sampling_rate,
                new_freq=self.audio_encoder.config.sampling_rate,
            )

        encoder_kwargs: dict[str, Any] = {
            "encoder_past_key_values": encoder_past_key_values,
            "padding_cache": conv_padding_cache,
            "use_streaming": use_streaming,
        }
        if isinstance(self.audio_encoder, VoxtralWrapper):
            encoder_kwargs["causal"] = self.aut_is_causal
            encoder_kwargs["audio_lengths"] = audio_lengths

        encoder_outputs = self.audio_encoder(audio, **encoder_kwargs)

        assert (audio_embeds := encoder_outputs.embeds) is not None, "Encoder outputs must contain embeds."

        assert isinstance(encoder_outputs, CausalAudioEncoderOutput), "encoder_outputs must be CausalAudioEncoderOutput."
        encoder_cache: tuple[Cache, MimiConv1dPaddingCache] | None = None
        if encoder_outputs.encoder_past_key_values is not None and encoder_outputs.padding_cache is not None:
            assert isinstance(encoder_outputs.encoder_past_key_values, Cache), "encoder_past_key_values must be Cache."
            assert isinstance(encoder_outputs.padding_cache, MimiConv1dPaddingCache), (
                "padding_cache must be MimiConv1dPaddingCache."
            )
            encoder_cache = (
                encoder_outputs.encoder_past_key_values,
                encoder_outputs.padding_cache,
            )

        if audio_lengths is not None:
            indices = torch.arange(audio.shape[-1], device=audio.device)
            audio_embeds_mask = (indices[None] < audio_lengths[:, None]).long()

            assert (encoder_sampling_rate := self.config.audio_tokenizer_config.sampling_rate) is not None, (
                "audio_tokenizer_config.sampling_rate is required."
            )
            assert (frame_rate := self.config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
                "audio_tokenizer_config._frame_rate is required."
            )
            assert (samples_per_frame := int(encoder_sampling_rate / frame_rate)) == encoder_sampling_rate / frame_rate, (
                "samples_per_frame must divide evenly."
            )
            padded_audio_mask = F.pad(
                audio_embeds_mask,
                (0, audio_embeds.shape[1] * samples_per_frame - audio_embeds_mask.shape[1]),
            )
            audio_embeds_mask = padded_audio_mask.view(-1, audio_embeds.shape[1], samples_per_frame).any(dim=-1)
        else:
            audio_embeds_mask = torch.ones(
                audio_embeds.shape[:2],
                dtype=torch.bool,
                device=audio_embeds.device,
            )

        adaptor_outputs = self.input_adaptor(audio_embeds, mask=audio_embeds_mask)
        assert isinstance(adaptor_outputs, EmbeddingAdaptorOutput), "adaptor_outputs must be EmbeddingAdaptorOutput."
        assert (audio_embeds := adaptor_outputs.outputs_embeds) is not None, "adaptor outputs_embeds is required."
        assert (audio_embeds_mask := adaptor_outputs.mask) is not None, "adaptor mask is required."  # type: ignore

        return AudioEncoderOutput(
            audio_embeds=audio_embeds,
            audio_embeds_mask=audio_embeds_mask,
            encoder_cache=encoder_cache,
        )

    def tokenize_audio(
        self,
        audio: torch.Tensor | None = None,
        audio_lengths: torch.Tensor | None = None,
        sampling_rate: int | None = None,
        num_code_groups: int = 8,
        encoder_past_key_values: StaticCache | None = None,
        conv_padding_cache: MimiConv1dPaddingCache | None = None,
        use_streaming: bool | None = None,
        return_mimi_features: bool = False,
    ) -> AudioTokenizerOutput:
        if audio is None:
            return AudioTokenizerOutput()
        if self.audio_tokenizer is None:
            raise RuntimeError("audio_tokenizer is unavailable when supports_audio_output is False.")
        dtype_module: nn.Module = self.audio_encoder if self.audio_encoder is not None else self.audio_tokenizer
        target_sampling_rate = (
            self.audio_encoder.config.sampling_rate
            if self.audio_encoder is not None
            else self.audio_tokenizer.config.sampling_rate
        )
        audio = cast_to_module_dtype(audio, dtype_module)
        if audio.ndim == 1:
            audio = audio[None, None]
        elif audio.ndim == 2:
            audio = audio[:, None]

        assert audio.ndim == 3, "Audio tensor must have 3 dimensions [batch_size, num_channels, num_samples]."

        if sampling_rate is not None and sampling_rate != target_sampling_rate:
            assert target_sampling_rate is not None, "sampling_rate is required for resampling."
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sampling_rate,
                new_freq=target_sampling_rate,
            )

        audio_mask = None
        if audio_lengths is not None:
            indices = torch.arange(audio.shape[-1], device=audio.device)
            audio_mask = (indices[None] < audio_lengths[:, None]).long()

        outputs = self.audio_tokenizer.encode(
            audio,
            padding_mask=audio_mask,
            num_quantizers=num_code_groups,
            encoder_past_key_values=encoder_past_key_values,
            padding_cache=conv_padding_cache,
            use_streaming=use_streaming,
            return_dict=True,
        )
        assert isinstance(outputs, MimiEncoderOutput), "tokenizer encode output must be MimiEncoderOutput."
        encoder_cache: tuple[Cache, MimiConv1dPaddingCache] | None = None
        if outputs.encoder_past_key_values is not None and outputs.padding_cache is not None:
            assert isinstance(outputs.encoder_past_key_values, Cache), "encoder_past_key_values must be Cache."
            assert isinstance(outputs.padding_cache, MimiConv1dPaddingCache), "padding_cache must be MimiConv1dPaddingCache."
            encoder_cache = (
                outputs.encoder_past_key_values,
                outputs.padding_cache,
            )

        assert outputs.audio_codes is not None, "tokenizer encode output must contain audio_codes."
        audio_codes = outputs.audio_codes.view(outputs.audio_codes.shape[-3:]).transpose(1, 2)
        if audio_mask is not None:
            assert (encoder_sampling_rate := self.config.audio_tokenizer_config.sampling_rate) is not None, (
                "audio_tokenizer_config.sampling_rate is required."
            )
            assert (frame_rate := self.config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
                "audio_tokenizer_config._frame_rate is required."
            )
            assert (samples_per_frame := int(encoder_sampling_rate / frame_rate)) == encoder_sampling_rate / frame_rate, (
                "samples_per_frame must divide evenly."
            )
            padded_audio_mask = F.pad(
                audio_mask,
                (0, audio_codes.shape[1] * samples_per_frame - audio_mask.shape[1]),
            )
            audio_codes_mask = padded_audio_mask.view(-1, audio_codes.shape[1], samples_per_frame).any(dim=-1)
        else:
            audio_codes_mask = torch.ones(
                audio_codes.shape[:2],
                dtype=torch.bool,
                device=audio_codes.device,
            )

        # Optionally return mimi_features for speaker embedding
        mimi_features = None
        if return_mimi_features:
            # Get latent features by decoding the audio codes (quantizer decode)
            # Shape: [batch_size, 512, num_frames] -> transpose to [batch_size, num_frames, 512]
            mimi_features = self.audio_tokenizer.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)

        return AudioTokenizerOutput(
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            encoder_cache=encoder_cache,
            mimi_features=mimi_features,
        )

    def tokenize_audio_segments(
        self,
        audio: torch.Tensor,
        segments: list[tuple[int, int]],
        num_code_groups: int = 8,
        return_mimi_features: bool = False,
    ) -> AudioTokenizerOutput:
        """Tokenize speech segments independently via batched Mimi encoding.

        Each segment is encoded independently (no cross-segment conv context).
        Used in no_audio_in_sil mode where SIL frames have no [A] token.

        Args:
            audio: Raw waveform. Shape: [1, num_samples] or [1, 1, num_samples]. Dtype: float.
            segments: List of (start_sample, end_sample) tuples for utterance regions.
            num_code_groups: Number of codec groups for Mimi quantizer.
            return_mimi_features: Whether to return pre-quantization features.

        Returns:
            AudioTokenizerOutput with codes for speech frames only (concatenated).
        """
        if not segments:
            return AudioTokenizerOutput()

        if audio.ndim == 3:
            audio = audio.squeeze(1)

        assert self.config.audio_tokenizer_config.sampling_rate is not None
        assert self.config.audio_tokenizer_config._frame_rate is not None  # type: ignore
        sr = self.config.audio_tokenizer_config.sampling_rate
        fr = self.config.audio_tokenizer_config._frame_rate  # type: ignore

        seg_audios = [audio[0, s:e] for s, e in segments]
        seg_lengths = torch.tensor([s.shape[0] for s in seg_audios], device=audio.device)

        max_len = int(seg_lengths.max().item())
        batched = torch.stack([F.pad(s, (0, max_len - s.shape[0])) for s in seg_audios]).to(audio.dtype)

        batched_out = self.tokenize_audio(
            audio=batched,
            audio_lengths=seg_lengths,
            num_code_groups=num_code_groups,
            return_mimi_features=return_mimi_features,
        )
        if batched_out.audio_codes is None:
            return AudioTokenizerOutput()

        all_codes = []
        all_masks = []
        all_features = []
        for i, length in enumerate(seg_lengths):
            n_frames = math.ceil(length.item() * fr / sr)
            n_frames = min(n_frames, batched_out.audio_codes.shape[1])
            all_codes.append(batched_out.audio_codes[i : i + 1, :n_frames])
            if batched_out.audio_codes_mask is not None:
                all_masks.append(batched_out.audio_codes_mask[i : i + 1, :n_frames])
            if batched_out.mimi_features is not None:
                all_features.append(batched_out.mimi_features[i : i + 1, :n_frames])

        audio_codes = torch.cat(all_codes, dim=1)
        audio_codes_mask = torch.cat(all_masks, dim=1) if all_masks else None
        mimi_features = torch.cat(all_features, dim=1) if all_features else None

        if audio_codes_mask is None:
            audio_codes_mask = torch.ones(audio_codes.shape[:2], dtype=torch.bool, device=audio_codes.device)

        return AudioTokenizerOutput(
            audio_codes=audio_codes,
            audio_codes_mask=audio_codes_mask,
            mimi_features=mimi_features,
        )

    def _get_audio_output_embeds(
        self,
        audio_codes: torch.Tensor,
        audio_codes_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode audio codes to embeddings via the output adaptor.

        Args:
            audio_codes: Audio codec codes.
                Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_codes_mask: Valid frame mask.
                Shape: [batch_size, num_frames]. Dtype: bool.

        Returns:
            Tuple of adapted embeddings and mask.
        """
        assert self.audio_tokenizer is not None, "audio_tokenizer is unavailable when supports_audio_output is False."
        assert self.output_adaptor is not None, "output_adaptor is unavailable when supports_audio_output is False."
        assert audio_codes.ndim == 3 and audio_codes_mask.ndim == 2, (
            "audio_codes must have 3 dims and audio_codes_mask must have 2 dims."
        )

        if self.input_num_code_groups < self.num_code_groups:
            audio_codes = audio_codes[:, :, : self.input_num_code_groups]

        latent_features = self.audio_tokenizer.quantizer.decode(audio_codes.transpose(1, 2)).transpose(1, 2)
        adaptor_outputs = self.output_adaptor(latent_features, mask=audio_codes_mask)
        return adaptor_outputs.outputs_embeds, adaptor_outputs.mask

    def _insert_audio_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
        audio_embeds_mask: torch.Tensor | None,
        audio_token_id: int,
    ) -> torch.Tensor:
        audio_mask = (input_ids == audio_token_id)[..., None].expand_as(inputs_embeds)
        audio_embeds = cast_float_inputs(audio_embeds, inputs_embeds.dtype)
        if audio_embeds_mask is not None:
            audio_embeds = audio_embeds[audio_embeds_mask]
        else:
            audio_embeds = audio_embeds.view(-1, audio_embeds.shape[-1])

        assert audio_mask.sum() == audio_embeds.numel(), (
            f"Number of masked positions must match audio_embeds element count. "
            f"audio_mask.sum()={audio_mask.sum()}, audio_embeds.numel()={audio_embeds.numel()}, "
            f"audio_token_id={audio_token_id}, input_ids_shape={input_ids.shape}, "
            f"audio_embeds_shape_before_mask={audio_embeds.shape}, "
            f"input_ids_tokens={input_ids[0].tolist() if input_ids.shape[0] == 1 else 'batch>1'}"
        )
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)
        return inputs_embeds

    def update_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        speaker_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Insert audio input, audio output, and speaker embeddings into inputs_embeds at placeholder positions.

        Args:
            inputs_embeds: Base embeddings from text tokens. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            input_ids: Token IDs for locating placeholders. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input_embeds: Encoded audio input embeddings. Shape: [batch_size, num_frames, hidden_size].
                Dtype: float.
            audio_input_embeds_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            audio_output_codes: Discrete audio output codes. Shape: [batch_size, num_frames, num_code_groups].
                Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            speaker_embeds: Speaker conditioning embeddings. Shape: [batch_size, 1, hidden_size]. Dtype: float.

        Returns:
            Updated inputs_embeds with all placeholders filled.
            Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
        """
        if audio_output_codes is not None:
            assert input_ids is not None, "`input_ids` required when `audio_output_codes` is provided."
            assert audio_output_codes_mask is not None, (
                "`audio_output_codes_mask` required when `audio_output_codes` is provided."
            )

        if audio_input_embeds is not None:
            assert input_ids is not None, "`input_ids` required when `audio_input_embeds` is provided."
            assert audio_input_embeds_mask is not None, (
                "`audio_input_embeds_mask` required when `audio_input_embeds` is provided."
            )

        if (
            input_ids is not None
            and audio_input_embeds is not None
            and audio_input_embeds_mask is not None
            and audio_input_embeds_mask.any()
        ):
            inputs_embeds = self._insert_audio_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                audio_embeds=audio_input_embeds,
                audio_embeds_mask=audio_input_embeds_mask,
                audio_token_id=AUDIO_INPUT_PLACEHOLDER.id,
            )

        if (
            input_ids is not None
            and audio_output_codes is not None
            and audio_output_codes_mask is not None
            and audio_output_codes_mask.any()
        ):
            audio_output_embeds, audio_output_embeds_mask = self._get_audio_output_embeds(
                audio_codes=audio_output_codes,
                audio_codes_mask=audio_output_codes_mask,
            )
            inputs_embeds = self._insert_audio_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                audio_embeds=audio_output_embeds,
                audio_embeds_mask=audio_output_embeds_mask,
                audio_token_id=AUDIO_OUTPUT_PLACEHOLDER.id,
            )

        # Insert speaker embedding at <|speaker_embedding_placeholder|> position
        if input_ids is not None and speaker_embeds is not None and self.speaker_token_id is not None:
            speaker_mask = input_ids == self.speaker_token_id
            if speaker_mask.any():
                inputs_embeds = self._insert_audio_embeds(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    audio_embeds=speaker_embeds,
                    audio_embeds_mask=None,  # Single token per sample, no mask needed
                    audio_token_id=self.speaker_token_id,
                )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        audio_input: torch.Tensor | None = None,
        audio_output: torch.Tensor | None = None,
        audio_input_lengths: torch.Tensor | None = None,
        audio_output_lengths: torch.Tensor | None = None,
        speaker_encoder_audio: torch.Tensor | None = None,
        speaker_encoder_audio_lengths: torch.Tensor | None = None,
        audio_output_codes: torch.Tensor | None = None,
        audio_output_codes_mask: torch.Tensor | None = None,
        audio_input_embeds: torch.Tensor | None = None,
        audio_input_embeds_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        use_speaker_embedding: bool = False,
        speaker_embeds: torch.Tensor | None = None,
        audio_output_segments: list[tuple[int, int]] | None = None,
        debug_mode: bool = False,
        debug_step: int | None = None,
        **kwargs: Any,
    ) -> RaonModelOutput:
        """Run the deployed inference forward pass.

        Args:
            input_ids: Token IDs. Shape: [batch_size, seq_length]. Dtype: long.
            attention_mask: Valid position mask. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input: Raw input audio. Shape: [batch_size, num_channels, num_samples] or [batch_size, num_samples].
                Dtype: float.
            audio_output: Raw output audio for tokenization. Same shapes as audio_input.
            audio_input_lengths: Valid sample lengths for audio_input. Shape: [batch_size]. Dtype: long.
            audio_output_lengths: Valid sample lengths for audio_output. Shape: [batch_size]. Dtype: long.
            speaker_encoder_audio: Unchunked audio for pretrained speaker encoder.
                Shape: [num_speakers, num_samples]. Dtype: float.
            speaker_encoder_audio_lengths: Valid sample lengths for speaker_encoder_audio.
                Shape: [num_speakers]. Dtype: long.
            audio_output_codes: Pre-tokenized audio codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            audio_input_embeds: Pre-computed audio input embeddings. Shape: [batch_size, num_frames, hidden_size].
                Dtype: float.
            audio_input_embeds_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            labels: Unsupported. Loss computation has been removed from the deployed runtime.
            position_ids: Position indices. Shape: [batch_size, seq_length]. Dtype: long.
            past_key_values: KV cache for generation.
            inputs_embeds: Pre-computed input embeddings. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            use_cache: Whether to return KV cache.
            cache_position: Position indices for static cache.
            use_speaker_embedding: Whether to compute speaker embeddings from `speaker_encoder_audio`.
            speaker_embeds: Pre-computed speaker embeddings. Shape: [num_speakers, 1, feature_dim]. Dtype: float.
            audio_output_segments: Optional segmented assistant audio boundaries for code extraction.
            debug_mode: Unsupported.
            debug_step: Unsupported.
            **kwargs: Passed to text model.

        Returns:
            RaonModelOutput with talker hidden states, text logits, and optional KV cache.
        """
        if labels is not None:
            raise NotImplementedError(
                "Loss computation has been removed from RaonModel; `labels` are unsupported "
                "in the deployed service runtime."
            )
        if debug_mode or debug_step is not None:
            raise NotImplementedError("Training/debug forward helpers have been removed from RaonModel.")

        self._validate_audio_output_inputs(
            audio_output=audio_output,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
        )
        self._validate_audio_input_inputs(
            input_ids=input_ids,
            audio_input=audio_input,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
        )
        speaker_encoder = self.speaker_encoder
        need_speaker_embedding = (
            audio_output is not None and use_speaker_embedding and speaker_embeds is None and speaker_encoder is not None
        )

        if self.supports_audio_output and audio_output_codes is None:
            if audio_output_segments is not None and audio_output is not None:
                # Utterance-segmented encoding: each speech segment is encoded
                # independently so codes align with [A] positions (no SIL frames).
                audio_output_inputs = self.tokenize_audio_segments(
                    audio=audio_output,
                    segments=audio_output_segments,
                    num_code_groups=self.num_code_groups,
                    return_mimi_features=False,
                )
            else:
                audio_output_inputs = self.tokenize_audio(
                    audio=audio_output,
                    audio_lengths=audio_output_lengths,
                    num_code_groups=self.num_code_groups,
                    return_mimi_features=False,
                )
            audio_output_codes = audio_output_inputs.audio_codes
            audio_output_codes_mask = audio_output_inputs.audio_codes_mask

        if need_speaker_embedding:
            assert self.is_pretrained_speaker_encoder, (
                "Non-fast speaker embedding path is deprecated. "
                "Enable pretrained speaker encoder with speaker_encoder_audio inputs."
            )
            assert speaker_encoder_audio is not None, (
                "speaker_encoder_audio is required when use_speaker_embedding is enabled."
            )
            assert speaker_encoder is not None, "speaker_encoder is required when use_speaker_embedding is enabled."
            assert speaker_encoder_audio_lengths is not None, (
                "speaker_encoder_audio_lengths is required when speaker_encoder_audio is provided."
            )
            assert speaker_encoder_audio.shape[0] == speaker_encoder_audio_lengths.shape[0], (
                "speaker_encoder_audio and speaker_encoder_audio_lengths must have matching batch size. "
                f"Got `{speaker_encoder_audio.shape[0]=}` and `{speaker_encoder_audio_lengths.shape[0]=}`."
            )
            speaker_embeds = speaker_encoder(speaker_encoder_audio, speaker_encoder_audio_lengths)

        if self.supports_audio_input and audio_input_embeds is None and audio_input is not None:
            audio_input_outputs = self.get_audio_input_embeds(
                audio=audio_input,
                audio_lengths=audio_input_lengths,
            )
            audio_input_embeds = audio_input_outputs.audio_embeds
            audio_input_embeds_mask = audio_input_outputs.audio_embeds_mask

        if inputs_embeds is None:
            assert input_ids is not None, "input_ids is required when inputs_embeds is None."
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
            assert inputs_embeds is not None, "get_input_embeddings must return non-None."
            inputs_embeds = self.update_inputs_embeds(
                inputs_embeds=inputs_embeds,
                input_ids=input_ids,
                audio_output_codes=audio_output_codes,
                audio_output_codes_mask=audio_output_codes_mask,
                audio_input_embeds=audio_input_embeds,
                audio_input_embeds_mask=audio_input_embeds_mask,
                speaker_embeds=speaker_embeds,
            )
        else:
            assert input_ids is None and audio_output_codes is None and audio_input_embeds is None, (
                "When inputs_embeds is provided, input_ids, audio_output_codes, and audio_input_embeds must be None."
            )

        inputs_embeds = cast_to_module_dtype(inputs_embeds, self.text_model)
        text_outputs = self.text_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        assert self.accepted_thinker_hidden_states is not None, (
            "accepted_thinker_hidden_states must be set by thinker capture hook."
        )
        assert text_outputs.last_hidden_state is not None, "text model must return last_hidden_state."

        accepted_hidden = self.accepted_thinker_hidden_states
        self.accepted_thinker_hidden_states = None

        # Text logits from thinker output (post-norm from text_model.norm, like standard LLM).
        text_logits = self.lm_head(text_outputs.last_hidden_state)

        # Talker forward: project thinker hidden → talker → audio hidden states.
        if self.talker is not None and self.thinker_to_talker_proj is not None:
            talker_input = self.thinker_to_talker_proj(accepted_hidden)
            talker_cache = getattr(self, "_talker_past_key_values", None)
            talker_outputs = self.talker(
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=talker_input,
                past_key_values=talker_cache,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            if use_cache:
                self._talker_past_key_values = talker_outputs.past_key_values
            talker_last_hidden_state = talker_outputs.last_hidden_state
        else:
            # STT-only: no separate talker, use thinker output directly.
            talker_last_hidden_state = text_outputs.last_hidden_state

        return RaonModelOutput(
            talker_last_hidden_state=talker_last_hidden_state,
            text_logits=text_logits,
            past_key_values=text_outputs.past_key_values,
        )

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
        past_key_values: DynamicCache | StaticCache | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference forward pass and return talker hidden state and text logits for decoding.

        Args:
            input_ids: Token IDs. Shape: [batch_size, seq_length]. Dtype: long.
            attention_mask: Valid position mask. Shape: [batch_size, seq_length]. Dtype: long.
            position_ids: Position indices. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input: Raw input audio. Shape: [batch_size, num_channels, num_samples]. Dtype: float.
            audio_output: Raw output audio. Same shape as audio_input.
            audio_input_lengths: Valid sample lengths. Shape: [batch_size]. Dtype: long.
            audio_output_lengths: Valid sample lengths. Shape: [batch_size]. Dtype: long.
            audio_output_codes: Pre-tokenized output codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_output_codes_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            audio_input_embeds: Pre-computed audio input embeddings. Shape: [batch_size, num_frames, hidden_size].
                Dtype: float.
            audio_input_embeds_mask: Valid frame mask. Shape: [batch_size, num_frames]. Dtype: bool.
            speaker_embeds: Speaker conditioning. Shape: [batch_size, num_frames, feature_dim]. Dtype: float.
            use_cache: Whether to use KV cache.
            past_key_values: KV cache for incremental decoding.
            cache_position: Explicit cache write positions, bypassing the StaticCache
                heuristic that can miscount when a key vector is all zeros.
                Shape: [seq_length]. Dtype: long.

        Returns:
            Tuple of (talker_last_hidden_state, text_logits). talker_last_hidden_state Shape: [batch_size, seq_length,
            hidden_size]. Dtype: float. text_logits Shape: [batch_size, seq_length, vocab_size]. Dtype: float.
        """
        outputs: RaonModelOutput = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            audio_input=audio_input,
            audio_output=audio_output,
            audio_input_lengths=audio_input_lengths,
            audio_output_lengths=audio_output_lengths,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=speaker_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        assert isinstance(talker_last_hidden_state := outputs.talker_last_hidden_state, torch.Tensor), (
            "forward must return talker_last_hidden_state as a tensor."
        )
        assert isinstance(text_logits := outputs.text_logits, torch.Tensor), "forward must return text_logits as a tensor."
        return talker_last_hidden_state, text_logits

    def generate_audio_codes(
        self,
        talker_last_hidden_state: torch.Tensor,
        first_code_sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
        allow_audio_end: bool = True,
    ) -> torch.Tensor:
        """Generate autoregressive audio codes from the last talker hidden state.

        Args:
            talker_last_hidden_state: Talker hidden states. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            first_code_sampler: Optional callable to sample first code from logits; else argmax.
            allow_audio_end: If False, suppress AUDIO_END sampling for duplex-style decoding.

        Returns:
            Generated audio codes. Shape: [batch_size, num_generated_frames, num_code_groups]. Dtype: long.
        """
        assert self.audio_lm_head is not None, "audio_lm_head is unavailable when supports_audio_output is False."
        assert self.proj_code is not None, "proj_code is unavailable when supports_audio_output is False."
        assert self.code_predictor is not None, "code_predictor is unavailable when supports_audio_output is False."

        first_code_logits: torch.Tensor = self.audio_lm_head(talker_last_hidden_state[:, -1])
        if not allow_audio_end:
            first_code_logits = first_code_logits.clone()
            first_code_logits[..., self.codebook_size] = torch.finfo(first_code_logits.dtype).min

        if first_code_sampler is not None:
            first_code = first_code_sampler(first_code_logits)
        else:
            first_code = first_code_logits.argmax(dim=-1, keepdim=True)

        audio_end_mask = first_code[:, 0] == self.codebook_size
        safe_first_code = first_code.clamp_max(self.codebook_size - 1)

        hidden_embeds = self.proj_code(talker_last_hidden_state[:, -1:])
        inputs_embeds = torch.cat(
            (hidden_embeds, self.code_predictor.get_input_embeddings()(safe_first_code)),
            dim=1,
        )
        sequences = self.code_predictor.predict_codes(inputs_embeds=inputs_embeds)
        sequences = torch.cat((first_code, sequences.to(first_code.device)), dim=1)
        if audio_end_mask.any():
            sequences[audio_end_mask, 1:] = 0
        return sequences

    def decode_audio(
        self,
        audio_codes: torch.Tensor,
        decoder_past_key_values: Cache | None = None,
        conv_padding_cache: MimiConv1dPaddingCache | None = None,
        conv_transpose_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
        use_streaming: bool | None = None,
    ) -> AudioDecoderOutput:
        """Decode discrete audio codes to waveform via the audio tokenizer decoder.

        Args:
            audio_codes: Discrete audio codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            decoder_past_key_values: Cached decoder KV for streaming.
            conv_padding_cache: Cached conv padding for streaming decoder.
            conv_transpose_padding_cache: Cached conv transpose padding for streaming.
            use_streaming: Whether to use streaming decode mode.

        Returns:
            AudioDecoderOutput with audio waveform (Shape: [batch_size, num_samples]. Dtype: float.) and decoder_cache.
        """
        assert self.audio_tokenizer is not None, "audio_tokenizer is unavailable when supports_audio_output is False."
        outputs = self.audio_tokenizer.decode(
            audio_codes.transpose(1, 2),
            decoder_past_key_values=decoder_past_key_values,
            conv1d_padding_cache=conv_padding_cache,
            convtranspose1d_padding_cache=conv_transpose_padding_cache,
            use_streaming=use_streaming,
            return_dict=True,
        )
        assert isinstance(outputs, StreamingMimiDecoderOutput), "tokenizer decode output must be StreamingMimiDecoderOutput."
        assert (audio_values := outputs.audio_values) is not None, "decode output must contain audio_values."
        audio = audio_values.view(audio_values.shape[0], audio_values.shape[2])

        if isinstance(outputs.decoder_past_key_values, Cache):
            decoder_past_key_values = outputs.decoder_past_key_values

        if isinstance(outputs.conv1d_padding_cache, MimiConv1dPaddingCache):
            conv_padding_cache = outputs.conv1d_padding_cache

        if isinstance(outputs.convtranspose1d_padding_cache, MimiConvTranspose1dPaddingCache):
            conv_transpose_padding_cache = outputs.convtranspose1d_padding_cache

        decoder_cache = None
        if (
            decoder_past_key_values is not None
            and conv_padding_cache is not None
            and conv_transpose_padding_cache is not None
        ):
            decoder_cache = (
                decoder_past_key_values,
                conv_padding_cache,
                conv_transpose_padding_cache,
            )

        return AudioDecoderOutput(audio=audio, decoder_cache=decoder_cache)

    def init_past_key_values(
        self,
        batch_size: int,
        max_sequence_length: int,
        prev_cache: Cache | None = None,
    ) -> Cache:
        """Initialize or reset KV cache for text model incremental decoding.

        When the model has a separate talker, also initializes a
        DynamicCache for the talker model stored as ``_talker_past_key_values``.

        Args:
            batch_size: Batch size for the cache.
            max_sequence_length: Maximum sequence length to cache.
            prev_cache: Existing cache to reset and reuse; if None, creates a new StaticCache.

        Returns:
            Initialized StaticCache ready for incremental decoding.
        """
        # Initialize talker KV cache for the separate HF talker model.
        if self.talker is not None:
            self._talker_past_key_values: DynamicCache | None = DynamicCache()

        if prev_cache is not None:
            prev_cache.reset()
            return prev_cache

        return StaticCache(
            self.config.text_model_config,
            max_cache_len=max_sequence_length,
        )

    def free_past_key_values(self, past_key_values: Cache) -> None:
        """Release KV cache resources including talker KV cache."""
        # Clean up talker KV cache.
        if hasattr(self, "_talker_past_key_values"):
            self._talker_past_key_values = None

    def _set_attention_implementation(
        self,
        attn_implementation: Literal["eager", "sdpa", "flash_attention_2"],
    ) -> list[str]:
        """Set attention implementation on whichever config-bearing submodules are present."""
        updated_components: list[str] = []

        def _set_component_attn(component: object, component_name: str) -> None:
            config = getattr(component, "config", None)
            if config is not None:
                config._attn_implementation = attn_implementation  # type: ignore[attr-defined]
                updated_components.append(component_name)

        def _set_nested_component_attn(
            component: object,
            component_name: str,
            nested_attr: str,
        ) -> None:
            _set_component_attn(component, component_name)
            nested_component = getattr(component, nested_attr, None)
            if nested_component is not None:
                _set_component_attn(nested_component, f"{component_name}.{nested_attr}")

        text_model = getattr(self, "text_model", None)
        if text_model is not None:
            _set_component_attn(text_model, "text_model")

        code_predictor = getattr(self, "code_predictor", None)
        if code_predictor is not None:
            _set_nested_component_attn(code_predictor, "code_predictor", "model")

        audio_encoder = getattr(self, "audio_encoder", None)
        encoder = getattr(audio_encoder, "encoder", None)
        if encoder is not None:
            _set_component_attn(encoder, "audio_encoder")

        audio_tokenizer = getattr(self, "audio_tokenizer", None)
        if audio_tokenizer is not None:
            _set_nested_component_attn(audio_tokenizer, "audio_tokenizer", "model")

        return updated_components

    def _compile_text_model(self) -> None:
        """Compile the text model forward pass with torch.compile for faster execution."""
        self.text_model.forward = torch.compile(
            self.text_model.forward,
            fullgraph=False,
            dynamic=None,
            backend="inductor",
            mode="default",
        )

__all__ = [
    "RaonConfig",
    "RaonModel",
    "RaonModelOutput",
    "TEXT_MODEL_CONFIGS",
    "TEXT_MODELS",
]
