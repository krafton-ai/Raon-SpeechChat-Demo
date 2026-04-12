from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import PreTrainedModel

from ..audio_encoder.streaming_mimi import StreamingMimiModel
from ..model import TEXT_MODELS, RaonConfig, RaonModel
from ..modules import (
    EmbeddingAdaptor,
    RaonCodePredictorModelForConditionalGeneration,
    ThinkerToTalkerProjection,
    build_speaker_encoder,
    is_pretrained_speaker_encoder,
)
from ..special_tokens import AUDIO_OUTPUT_BC, AUDIO_OUTPUT_SIL, SPEAKER_EMBEDDING_PLACEHOLDER
from ..voxtral import VoxtralRealtimeEncoderConfig, VoxtralWrapper


def load_raon_model_without_text_model(
    path: str | Path,
    device: str,
    dtype: torch.dtype,
) -> RaonModel:
    checkpoint_path = Path(path)
    config_path = checkpoint_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing runtime config: {config_path}")
    config = RaonConfig(**json.loads(config_path.read_text(encoding="utf-8")))

    state_dict: dict = {}
    for safetensor_file in checkpoint_path.glob("*.safetensors"):
        state_dict.update(load_file(safetensor_file))

    new_model = object.__new__(RaonModel)
    PreTrainedModel.__init__(new_model, config)
    new_model.config = config
    new_model.hidden_size = config.text_model_config.hidden_size
    new_model.vocab_size = config.text_model_config.vocab_size
    new_model.codebook_size = config.audio_tokenizer_config.codebook_size
    new_model.audio_lm_head_vocab_size = new_model.codebook_size + 1
    new_model.supports_audio_input = getattr(config, "supports_audio_input", True)
    new_model.supports_audio_output = getattr(config, "supports_audio_output", True)
    new_model.use_duplex_end_pad = getattr(config, "use_duplex_end_pad", False)
    new_model.use_sil_token = getattr(config, "use_sil_token", False)
    new_model.use_backchannel_token = getattr(config, "use_backchannel_token", False)
    new_model.no_audio_in_sil = getattr(config, "no_audio_in_sil", False)
    new_model.sequence_mode = getattr(config, "sequence_mode", None)
    new_model.duplex_sil_token_id = getattr(config, "duplex_sil_token_id", AUDIO_OUTPUT_SIL.id)
    new_model.duplex_bc_token_id = getattr(config, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id)
    new_model.num_code_groups = config.code_predictor_config.num_code_groups
    new_model.input_num_code_groups = getattr(config, "input_num_code_groups", None) or new_model.num_code_groups
    new_model.sampling_rate = config.audio_tokenizer_config.sampling_rate
    assert (frame_rate := config.audio_tokenizer_config._frame_rate) is not None  # type: ignore[attr-defined]
    new_model.frame_rate = frame_rate

    if new_model.supports_audio_output and config.speaker_encoder_config is not None:
        new_model.speaker_encoder = build_speaker_encoder(config.speaker_encoder_config, dtype=dtype)
        new_model.is_pretrained_speaker_encoder = is_pretrained_speaker_encoder(new_model.speaker_encoder)
        new_model.speaker_token_id = SPEAKER_EMBEDDING_PLACEHOLDER.id
    else:
        new_model.speaker_encoder = None
        new_model.is_pretrained_speaker_encoder = False
        new_model.speaker_token_id = None

    acoustic_delay = getattr(config, "acoustic_delay", None)
    if acoustic_delay is None:
        new_model.delays = [0] * new_model.num_code_groups
    elif isinstance(acoustic_delay, list):
        new_model.delays = acoustic_delay
    elif isinstance(acoustic_delay, int):
        new_model.delays = [0] + [acoustic_delay] * (new_model.num_code_groups - 1)
    else:
        new_model.delays = [0] * new_model.num_code_groups
    new_model.max_delay = max(new_model.delays)

    new_model.text_model = None  # type: ignore[assignment]
    if new_model.supports_audio_input:
        new_model.audio_encoder = VoxtralWrapper.from_config(config=config.audio_encoder_config, dtype=dtype)  # type: ignore[assignment]
    else:
        new_model.audio_encoder = None
    new_model.aut_is_causal = getattr(config, "aut_is_causal", False)
    if new_model.supports_audio_output:
        new_model.audio_tokenizer = StreamingMimiModel._from_config(config.audio_tokenizer_config, dtype=dtype)  # type: ignore[assignment]
    else:
        new_model.audio_tokenizer = None
    if new_model.supports_audio_input:
        new_model.input_adaptor = EmbeddingAdaptor(
            input_size=config.input_adaptor_config.input_size,
            output_size=config.input_adaptor_config.output_size,
            output_time_scale=config.input_adaptor_config.output_time_scale,
            num_layers=getattr(config.input_adaptor_config, "num_layers", 1),
            hidden_size=getattr(config.input_adaptor_config, "hidden_size", None),
            decoder_config=getattr(config.input_adaptor_config, "decoder_config", None),
            use_post_norm=getattr(config.input_adaptor_config, "use_post_norm", False),
            norm_eps=getattr(config.input_adaptor_config, "norm_eps", 1e-6),
            post_norm_init_scale=getattr(config.input_adaptor_config, "post_norm_init_scale", None),
            dtype=dtype,
        )
    else:
        new_model.input_adaptor = None
    if new_model.supports_audio_output:
        new_model.output_adaptor = EmbeddingAdaptor(
            input_size=config.output_adaptor_config.input_size,
            output_size=config.output_adaptor_config.output_size,
            output_time_scale=config.output_adaptor_config.output_time_scale,
            num_layers=getattr(config.output_adaptor_config, "num_layers", 1),
            hidden_size=getattr(config.output_adaptor_config, "hidden_size", None),
            decoder_config=getattr(config.output_adaptor_config, "decoder_config", None),
            use_post_norm=getattr(config.output_adaptor_config, "use_post_norm", False),
            norm_eps=getattr(config.output_adaptor_config, "norm_eps", 1e-6),
            post_norm_init_scale=getattr(config.output_adaptor_config, "post_norm_init_scale", None),
            dtype=dtype,
        )
    else:
        new_model.output_adaptor = None
    new_model.lm_head = None  # type: ignore[assignment]

    num_talker_layers = config.num_talker_layers
    num_thinker_layers = config.text_model_config.num_hidden_layers
    rms_norm_eps = getattr(config.text_model_config, "rms_norm_eps", 1e-6)
    new_model.num_talker_layers = num_talker_layers
    accept_hidden_layer = getattr(config, "accept_hidden_layer", -1)
    if accept_hidden_layer < 0:
        accept_hidden_layer = num_thinker_layers + accept_hidden_layer
    new_model.accept_hidden_layer = accept_hidden_layer
    new_model.thinker_capture_layer_index = accept_hidden_layer

    talker_hidden_size = (
        int(config.talker_config.hidden_size)
        if config.talker_config is not None
        else int(config.text_model_config.hidden_size)
    )

    if new_model.supports_audio_output:
        new_model.proj_code = torch.nn.Linear(
            talker_hidden_size,
            config.code_predictor_config.hidden_size,
            bias=getattr(config, "proj_code_bias", False),
            dtype=dtype,
        )
        new_model.code_predictor = RaonCodePredictorModelForConditionalGeneration._from_config(  # type: ignore[assignment]
            config.code_predictor_config,
            dtype=dtype,
        )
        new_model.audio_lm_head = torch.nn.Linear(
            talker_hidden_size,
            new_model.audio_lm_head_vocab_size,
            bias=False,
            dtype=dtype,
        )
    else:
        new_model.proj_code = None
        new_model.code_predictor = None
        new_model.audio_lm_head = None

    if new_model.supports_audio_output and config.talker_config is not None:
        projection_mode = getattr(config, "thinker_to_talker_projection_mode", "linear")
        projection_intermediate_size = getattr(config, "thinker_to_talker_intermediate_size", None)
        if projection_mode == "mlp" and projection_intermediate_size is None:
            projection_intermediate_size = int(config.talker_config.intermediate_size)
        new_model.thinker_to_talker_proj = ThinkerToTalkerProjection(
            thinker_hidden_size=int(config.text_model_config.hidden_size),
            talker_hidden_size=talker_hidden_size,
            intermediate_size=projection_intermediate_size,
            mode=projection_mode,
            use_norm=getattr(config, "thinker_to_talker_pre_norm", False),
            rms_norm_eps=rms_norm_eps,
        ).to(dtype=dtype)
        talker = TEXT_MODELS[config.talker_config.model_type]._from_config(config.talker_config, dtype=dtype)
        talker.embed_tokens = None
        new_model.talker = talker
    else:
        new_model.thinker_to_talker_proj = None
        new_model.talker = None

    audio_encoder_state = {
        k.replace("audio_encoder.", ""): v for k, v in state_dict.items() if k.startswith("audio_encoder.")
    }
    audio_tokenizer_state = {
        k.replace("audio_tokenizer.", ""): v for k, v in state_dict.items() if k.startswith("audio_tokenizer.")
    }
    input_adaptor_state = {
        k.replace("input_adaptor.", ""): v for k, v in state_dict.items() if k.startswith("input_adaptor.")
    }
    output_adaptor_state = {
        k.replace("output_adaptor.", ""): v for k, v in state_dict.items() if k.startswith("output_adaptor.")
    }
    adaptor_state = {k.replace("adaptor.", ""): v for k, v in state_dict.items() if k.startswith("adaptor.")}
    proj_code_state = {k.replace("proj_code.", ""): v for k, v in state_dict.items() if k.startswith("proj_code.")}
    code_predictor_state = {
        k.replace("code_predictor.", ""): v for k, v in state_dict.items() if k.startswith("code_predictor.")
    }
    speaker_encoder_state = {
        k.replace("speaker_encoder.", ""): v for k, v in state_dict.items() if k.startswith("speaker_encoder.")
    }

    if audio_encoder_state and new_model.audio_encoder is not None:
        new_model.audio_encoder.load_state_dict(audio_encoder_state)
    if audio_tokenizer_state and new_model.audio_tokenizer is not None:
        new_model.audio_tokenizer.load_state_dict(audio_tokenizer_state)
    if input_adaptor_state and new_model.input_adaptor is not None:
        new_model.input_adaptor.load_state_dict(input_adaptor_state)
    if output_adaptor_state and new_model.output_adaptor is not None:
        new_model.output_adaptor.load_state_dict(output_adaptor_state)
    if adaptor_state:
        if new_model.input_adaptor is not None:
            new_model.input_adaptor.load_state_dict(adaptor_state)
        if new_model.output_adaptor is not None:
            new_model.output_adaptor.load_state_dict(adaptor_state)
    audio_lm_head_state = {
        k.replace("audio_lm_head.", ""): v for k, v in state_dict.items() if k.startswith("audio_lm_head.")
    }
    if proj_code_state and new_model.proj_code is not None:
        new_model.proj_code.load_state_dict(proj_code_state)
    if audio_lm_head_state and new_model.audio_lm_head is not None:
        new_model.audio_lm_head.load_state_dict(audio_lm_head_state)
    if code_predictor_state and new_model.code_predictor is not None:
        new_model.code_predictor.load_state_dict(code_predictor_state)
    if new_model.speaker_encoder is not None and speaker_encoder_state:
        new_model.speaker_encoder.load_state_dict(speaker_encoder_state)
    thinker_to_talker_proj_state = {
        k.replace("thinker_to_talker_proj.", ""): v
        for k, v in state_dict.items()
        if k.startswith("thinker_to_talker_proj.")
    }
    if thinker_to_talker_proj_state and new_model.thinker_to_talker_proj is not None:
        new_model.thinker_to_talker_proj.load_state_dict(thinker_to_talker_proj_state)
    talker_state = {k.replace("talker.", ""): v for k, v in state_dict.items() if k.startswith("talker.")}
    if talker_state and new_model.talker is not None:
        new_model.talker.load_state_dict(talker_state, strict=False)

    return new_model.to(device)  # type: ignore[return-value]


__all__ = ["load_raon_model_without_text_model"]
