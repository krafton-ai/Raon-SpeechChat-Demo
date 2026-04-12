from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.mimi import MimiConfig, MimiModel
from transformers.models.mimi.modeling_mimi import (
    MimiConv1d,
    MimiConv1dPaddingCache,
    MimiConvTranspose1d,
    MimiResnetBlock,
)
from transformers.utils.generic import ModelOutput
from transformers.utils.import_utils import is_torchdynamo_compiling


class StaticMimiConv1dPaddingCache(MimiConv1dPaddingCache):
    def __init__(self, per_layer_padding: list[int], padding_cache: list[torch.Tensor]) -> None:
        self.per_layer_padding = per_layer_padding
        self.padding_cache: list[torch.Tensor | None] = padding_cache  # type: ignore[assignment]
        self.is_initialized = True
        if not is_torchdynamo_compiling():
            for i in range(len(padding_cache)):
                torch._dynamo.mark_static_address(self.padding_cache[i])

    def update(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        assert self.is_initialized
        padding = self.per_layer_padding[layer_idx]
        cache_item = self.padding_cache[layer_idx]
        assert cache_item is not None
        current_cache = cache_item.clone()
        cache_item.copy_(hidden_states[:, :, hidden_states.shape[2] - padding :])
        return current_cache

    def reset(self) -> None:
        self.is_initialized = False

    def initialize(self, padding_cache: list[torch.Tensor]) -> None:
        for i in range(len(self.padding_cache)):
            cache_item = self.padding_cache[i]
            assert cache_item is not None
            cache_item.copy_(padding_cache[i])

        self.is_initialized = True


class MimiConvTranspose1dPaddingCache:
    """
    Padding cache for MimiConvTranspose1d causal convolutions in order to support streaming via cache padding.

    A padding cache is a list of cached partial hidden states for each convolution layer.
    Hidden states are cached from the previous call to the MimiConvTranspose1d forward pass, given the padding size.
    """

    def __init__(
        self,
        num_layers: int,
        per_layer_padding: list[torch.Tensor],
        per_layer_in_channels: list[int],
    ):
        # ensure correct number of layers for each arg
        from_args_num_layers = {len(per_layer_padding), len(per_layer_in_channels)}

        if len(from_args_num_layers) != 1 or from_args_num_layers.pop() != num_layers:
            raise ValueError(
                f"Expected `num_layers` ({num_layers}) values in `per_layer_padding`, "
                "`per_layer_padding_mode` and `per_layer_in_channels`"
            )
        self.per_layer_padding = per_layer_padding
        self.per_layer_in_channels = per_layer_in_channels
        self.per_layer_is_init = [True] * num_layers
        self.padding_cache: list[torch.Tensor | None] = [None] * num_layers

    def update(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        batch_size, dtype, device = (
            hidden_states.shape[0],
            hidden_states.dtype,
            hidden_states.device,
        )
        padding = int(self.per_layer_padding[layer_idx].long().item())
        in_channels = self.per_layer_in_channels[layer_idx]

        cached = self.padding_cache[layer_idx]
        if cached is None:
            current_cache = torch.zeros(
                batch_size,
                in_channels,
                padding,
                device=device,
                dtype=dtype,
            )
        else:
            current_cache = cached

        if padding > 0:
            padding_states = hidden_states[:, :, -padding:]
        else:
            padding_states = torch.empty(batch_size, in_channels, padding, dtype=dtype, device=device)
        self.padding_cache[layer_idx] = padding_states

        return current_cache


@dataclass
class StreamingMimiOutput(ModelOutput):
    audio_codes: torch.LongTensor | None = None
    audio_values: torch.FloatTensor | None = None
    encoder_past_key_values: Cache | list[torch.FloatTensor] | None = None
    decoder_past_key_values: Cache | list[torch.FloatTensor] | None = None
    conv1d_padding_cache: MimiConv1dPaddingCache | None = None
    convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None


@dataclass
class StreamingMimiDecoderOutput(ModelOutput):
    audio_values: torch.FloatTensor | None = None
    decoder_past_key_values: Cache | list[torch.FloatTensor] | None = None
    conv1d_padding_cache: MimiConv1dPaddingCache | None = None
    convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None


class StreamingMimiConvTranspose1d(MimiConvTranspose1d):
    def __init__(
        self,
        config: MimiConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(config, in_channels, out_channels, kernel_size, stride, groups, bias)

        self.in_channels = in_channels
        self.layer_idx = layer_idx
        kernel_size_tensor = torch.tensor(self.conv.kernel_size[0], dtype=torch.int64)
        stride_tensor = torch.tensor(self.conv.stride[0], dtype=torch.int64)
        padding_total = kernel_size_tensor - stride_tensor

        self.register_buffer("stride", stride_tensor, persistent=False)
        self.register_buffer("kernel_size", kernel_size_tensor, persistent=False)
        self.register_buffer("padding_total", padding_total, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_cache: MimiConvTranspose1dPaddingCache | None = None,
    ) -> torch.Tensor:
        if not self.causal and padding_cache is not None:
            raise ValueError("`padding_cache` is only defined for causal convolutions.")
        if self.causal and padding_cache is not None:
            assert self.layer_idx is not None
            layer_padding_cache = padding_cache.update(hidden_states, self.layer_idx)
            padding_len = padding_cache.per_layer_padding[self.layer_idx]
            extra_padding = padding_len - layer_padding_cache.shape[-1]
            if extra_padding > 0:
                layer_padding_cache = nn.functional.pad(
                    layer_padding_cache,
                    (int(extra_padding), 0),
                    mode="constant",
                    value=0,
                )
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=-1)
            padding_left = layer_padding_cache.shape[-1] * self.stride + self.padding_left  # type: ignore
        else:
            padding_left = self.padding_left
        hidden_states = self.conv(hidden_states)

        end = hidden_states.shape[-1] - self.padding_right
        hidden_states = hidden_states[..., padding_left:end]

        return hidden_states


class StreamingMimiDecoder(nn.Module):
    """SEANet decoder as used by Mimi."""

    def __init__(self, config: MimiConfig):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model: list[nn.Module] = [
            MimiConv1d(
                config,
                config.hidden_size,
                scaling * config.num_filters,
                config.kernel_size,
            )
        ]
        mimiconv1d_layer_names = ["layers.0"]
        mimiconvtranspose1d_layer_names: list[str] = []

        # Upsample to raw audio scale
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            # Add  upsampling layers
            model += [nn.ELU()]
            mimiconvtranspose1d_layer_names.append(f"layers.{len(model)}")
            model += [
                StreamingMimiConvTranspose1d(
                    config,
                    current_scale,
                    current_scale // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            # Add residual layers
            for j in range(config.num_residual_layers):
                mimiconv1d_layer_names.extend([f"layers.{len(model)}.block.{1}", f"layers.{len(model)}.block.{3}"])
                model += [MimiResnetBlock(config, current_scale // 2, (config.dilation_growth_rate**j, 1))]  # type: ignore
            scaling //= 2

        # Add final layers
        model += [nn.ELU()]
        mimiconv1d_layer_names.append(f"layers.{len(model)}")
        model += [
            MimiConv1d(
                config,
                config.num_filters,
                config.audio_channels,
                config.last_kernel_size,
            )
        ]
        self.layers = nn.ModuleList(model)

        self._mimiconv1d_layer_names = mimiconv1d_layer_names
        self._mimiconvtranspose1d_layer_names = mimiconvtranspose1d_layer_names

        # initialize layer_idx for MimiConv1d submodules, necessary for padding_cache
        for layer_idx, layername in enumerate(self._mimiconv1d_layer_names):
            conv_layer = self.get_submodule(layername)
            conv_layer.layer_idx = layer_idx  # type: ignore
        # initialize layer_idx for MimiConvTranspose1d submodules, necessary for padding_cache
        for layer_idx, layername in enumerate(self._mimiconvtranspose1d_layer_names):
            convtranspose_layer = self.get_submodule(layername)
            convtranspose_layer.layer_idx = layer_idx  # type: ignore

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv1d_padding_cache: MimiConv1dPaddingCache | None = None,
        convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, (MimiConv1d, MimiResnetBlock)):
                hidden_states = layer(hidden_states, padding_cache=conv1d_padding_cache)
            elif isinstance(layer, MimiConvTranspose1d):
                hidden_states = layer(hidden_states, padding_cache=convtranspose1d_padding_cache)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class StreamingMimiModel(MimiModel):
    def __init__(self, config: MimiConfig):
        super().__init__(config)
        self.decoder = StreamingMimiDecoder(config)
        self.upsample = StreamingMimiConvTranspose1d(
            config,
            config.hidden_size,
            config.hidden_size,
            kernel_size=2 * int(config.encodec_frame_rate / config.frame_rate),
            stride=2,
            bias=False,
            groups=config.upsample_groups,
            layer_idx=len(self.decoder._mimiconvtranspose1d_layer_names),
        )
        # targets = [self.encoder, self.downsample, self.decoder]
        targets = [self.decoder]
        for target in targets:
            for module in target.modules():
                if isinstance(module, MimiConv1d):
                    module.forward = partial(self.mimi_conv1d_forward, module)  # type: ignore[method-assign]

    def mimi_conv1d_forward(
        self,
        module: MimiConv1d,
        hidden_states: torch.Tensor,
        padding_cache: MimiConv1dPaddingCache | None = None,
    ) -> torch.Tensor:
        extra_padding = module._get_extra_padding_for_conv1d(hidden_states)

        if not module.causal and padding_cache is not None:
            raise ValueError("`padding_cache` is not supported for non-causal convolutions.")

        if module.causal and padding_cache is not None:
            assert module.layer_idx is not None
            layer_padding_cache = padding_cache.update(hidden_states, module.layer_idx)
            assert layer_padding_cache is not None
            hidden_states = torch.cat([layer_padding_cache, hidden_states], dim=2)
            assert not isinstance(module.padding_total, nn.Module)
            hidden_states = module._pad1d(
                hidden_states,
                (
                    max(0, module.padding_total - layer_padding_cache.shape[2]),  # type: ignore
                    extra_padding,  # type: ignore
                ),
                mode=module.pad_mode,
            )

        elif module.causal and padding_cache is None:
            hidden_states = module._pad1d(
                hidden_states,
                (module.padding_total, extra_padding),  # type: ignore
                mode=module.pad_mode,
            )

        else:
            hidden_states = module._pad1d(
                hidden_states,
                (module.padding_left, module.padding_right + extra_padding),  # type: ignore
                mode=module.pad_mode,
            )

        hidden_states = module.conv(hidden_states)
        return hidden_states

    def _decode_frame(  # type: ignore[override]
        self,
        codes: torch.Tensor,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        conv1d_padding_cache: MimiConv1dPaddingCache | None = None,
        convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
        return_dict: bool | None = None,
    ) -> tuple[
        torch.Tensor,
        Cache | list[torch.FloatTensor] | None,
        MimiConv1dPaddingCache | None,
        MimiConvTranspose1dPaddingCache | None,
    ]:
        embeddings = self.quantizer.decode(codes)

        assert self.upsample is not None, "_decode_frame: `self.upsample` is None."
        embeddings = self.upsample(embeddings, padding_cache=convtranspose1d_padding_cache)
        decoder_outputs = self.decoder_transformer(
            embeddings.transpose(1, 2),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=return_dict,
        )
        if return_dict:
            past_key_values = decoder_outputs.get("past_key_values")
        elif len(decoder_outputs) > 1:
            past_key_values = decoder_outputs[1]
        embeddings = decoder_outputs[0].transpose(1, 2)
        outputs = self.decoder(
            embeddings,
            conv1d_padding_cache=conv1d_padding_cache,
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,
        )
        return outputs, past_key_values, conv1d_padding_cache, convtranspose1d_padding_cache

    def decode(  # type: ignore
        self,
        audio_codes: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        decoder_past_key_values: Cache | list[torch.FloatTensor] | None = None,
        conv1d_padding_cache: MimiConv1dPaddingCache | None = None,
        convtranspose1d_padding_cache: MimiConvTranspose1dPaddingCache | None = None,
        use_streaming: bool | None = True,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | StreamingMimiDecoderOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_streaming = use_streaming if use_streaming is not None else self.config.use_streaming

        if use_streaming and conv1d_padding_cache is None:
            per_layer_padding, per_layer_padding_mode, per_layer_in_channels = (
                [],
                [],
                [],
            )
            for layer_name in self.decoder._mimiconv1d_layer_names:
                per_layer_padding.append(self.decoder.get_submodule(layer_name).padding_total)  # type: ignore
                per_layer_padding_mode.append(self.decoder.get_submodule(layer_name).pad_mode)
                per_layer_in_channels.append(self.decoder.get_submodule(layer_name).in_channels)  # type: ignore

            conv1d_padding_cache = MimiConv1dPaddingCache(
                num_layers=len(self.decoder._mimiconv1d_layer_names),
                per_layer_padding=per_layer_padding,  # type: ignore
                per_layer_padding_mode=per_layer_padding_mode,
                per_layer_in_channels=per_layer_in_channels,
            )

        if use_streaming and convtranspose1d_padding_cache is None:
            convtranspose_per_layer_padding: list[torch.Tensor] = []
            convtranspose_per_layer_in_channels: list[int] = []
            for layer_name in self.decoder._mimiconvtranspose1d_layer_names:
                k = self.decoder.get_submodule(layer_name).kernel_size
                s = self.decoder.get_submodule(layer_name).stride
                if k % s == 0:  # type: ignore
                    padding_tmp = (k / s - 1) * s  # type: ignore
                else:
                    padding_tmp = torch.floor(k / s) * s  # type: ignore
                convtranspose_per_layer_padding.append(padding_tmp)
                convtranspose_per_layer_in_channels.append(self.decoder.get_submodule(layer_name).in_channels)  # type: ignore

            assert self.upsample is not None
            k = self.upsample.kernel_size
            s = self.upsample.stride
            if k % s == 0:  # type: ignore
                padding_tmp = (k / s - 1) * s  # type: ignore
            else:
                padding_tmp = torch.floor(k / s) * s  # type: ignore

            convtranspose_per_layer_padding.append(padding_tmp)
            convtranspose_per_layer_in_channels.append(self.upsample.in_channels)  # type: ignore

            convtranspose1d_padding_cache = MimiConvTranspose1dPaddingCache(
                num_layers=len(self.decoder._mimiconvtranspose1d_layer_names) + 1,
                per_layer_padding=convtranspose_per_layer_padding,
                per_layer_in_channels=convtranspose_per_layer_in_channels,
            )

        (
            audio_values,
            decoder_past_key_values,
            conv1d_padding_cache,
            convtranspose1d_padding_cache,
        ) = self._decode_frame(
            audio_codes,
            past_key_values=decoder_past_key_values,
            conv1d_padding_cache=conv1d_padding_cache,
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,
            return_dict=return_dict,
        )

        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        if not return_dict:
            return (  # type: ignore
                audio_values,
                decoder_past_key_values,
                conv1d_padding_cache,
                convtranspose1d_padding_cache,
            )
        return StreamingMimiDecoderOutput(
            audio_values=audio_values,  # type: ignore
            decoder_past_key_values=decoder_past_key_values,
            conv1d_padding_cache=conv1d_padding_cache,
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        num_quantizers: int | None = None,
        audio_codes: torch.Tensor | None = None,
        encoder_past_key_values: Cache | list[torch.FloatTensor] | None = None,
        decoder_past_key_values: Cache | list[torch.FloatTensor] | None = None,
        return_dict: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | StreamingMimiOutput:
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        if audio_codes is None:
            encoder_outputs = self.encode(
                input_values,
                padding_mask,
                num_quantizers,
                encoder_past_key_values,
                return_dict=return_dict,
            )
            audio_codes = encoder_outputs[0]
            if return_dict:
                encoder_past_key_values = encoder_outputs.get("past_key_values")  # type: ignore[union-attr]
            elif len(encoder_outputs) > 1:
                encoder_past_key_values = encoder_outputs[1]  # type: ignore[assignment]

        decoder_outputs = self.decode(audio_codes, padding_mask, decoder_past_key_values, return_dict=return_dict)
        audio_values = decoder_outputs[0]
        if return_dict:
            decoder_past_key_values = decoder_outputs.get("past_key_values")  # type: ignore[union-attr]
            conv1d_padding_cache = decoder_outputs.get("conv1d_padding_cache")  # type: ignore[union-attr]
            convtranspose1d_padding_cache = decoder_outputs.get("convtranspose1d_padding_cache")  # type: ignore[union-attr]
        elif len(decoder_outputs) > 1:
            decoder_past_key_values = decoder_outputs[1]  # type: ignore[assignment]
            conv1d_padding_cache = decoder_outputs[2]  # type: ignore[misc]
            convtranspose1d_padding_cache = decoder_outputs[3]  # type: ignore[misc]

        if not return_dict:
            return (  # type: ignore
                audio_codes,
                audio_values,
                encoder_past_key_values,
                decoder_past_key_values,
                conv1d_padding_cache,
                convtranspose1d_padding_cache,
            )

        return StreamingMimiOutput(
            audio_codes=audio_codes,  # type: ignore
            audio_values=audio_values,  # type: ignore
            encoder_past_key_values=encoder_past_key_values,
            decoder_past_key_values=decoder_past_key_values,
            conv1d_padding_cache=conv1d_padding_cache,  # type: ignore
            convtranspose1d_padding_cache=convtranspose1d_padding_cache,  # type: ignore
        )

    def get_input_embeddings(self):
        """Return None as audio models don't have traditional input embeddings."""
        return None

    def set_input_embeddings(self, value):
        """No-op as audio models don't have traditional input embeddings."""
        pass
