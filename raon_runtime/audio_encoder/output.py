from dataclasses import dataclass

import torch
from transformers.cache_utils import Cache
from transformers.models.mimi.modeling_mimi import MimiConv1dPaddingCache
from transformers.utils.generic import ModelOutput


@dataclass
class CausalAudioEncoderOutput(ModelOutput):
    embeds: torch.Tensor | None = None
    encoder_past_key_values: Cache | None = None
    padding_cache: MimiConv1dPaddingCache | None = None
    streaming_state: object | None = None
