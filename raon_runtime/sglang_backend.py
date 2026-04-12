"""SGLang-compatible inference backend with custom KV-cache management."""

import logging
import time
import types
from dataclasses import dataclass, field
from typing import Any, cast

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod  # type: ignore
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # type: ignore
from sglang.srt.managers.schedule_batch import ModelWorkerBatch  # type: ignore
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch  # type: ignore
from sglang.srt.model_executor.model_runner import (  # type: ignore
    ForwardMode,
    LogitsProcessorOutput,
    ModelConfig,
    ModelRunner,
    ReqToTokenPool,
    ServerArgs,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo  # type: ignore
from torch import nn
from torch._logging._internal import set_logs
from transformers import DynamicCache

from .inference import RaonGenerateResult, RaonInferenceModel, extract_predicted_text
from .model import RaonModel
from .modules import AudioDecoderOutput, AudioEncoderOutput, AudioTokenizerOutput
from .special_tokens import (
    AUDIO_END,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_START,
    IM_START,
)
from .state_machine import RaonMachineState, RaonPhase
from .util.runtime_checkpoint import load_raon_model_without_text_model


@dataclass
class SGLangDecodingMetadata:
    """Per-request decoding state for SGLang KV-cache: pool indices, sequence lengths, and cache offsets."""

    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: Any
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    max_sequence_length: int
    batch_size: int
    sampling_info: SamplingBatchInfo
    device: str

    def _get_last_loc(self, prefix_lens: torch.Tensor) -> torch.Tensor:
        """Return the last allocated physical KV slot per request, or ``-1`` if empty."""
        last_loc = torch.full((self.batch_size,), -1, dtype=torch.int64, device=self.device)
        for row_index in range(self.batch_size):
            prefix_len = int(prefix_lens[row_index].item())
            if prefix_len == 0:
                continue

            req_pool_idx = int(self.req_pool_indices[row_index].item())
            last_loc[row_index] = self.req_to_token_pool.req_to_token[req_pool_idx, prefix_len - 1].to(torch.int64)
        return last_loc

    def free_allocated_kv_slots(self) -> None:
        """Free any thinker KV slots currently owned by this metadata object."""
        active_indices: list[torch.Tensor] = []
        for row_index in range(self.batch_size):
            seq_len = int(self.seq_lens[row_index].item())
            if seq_len == 0:
                continue

            req_pool_idx = int(self.req_pool_indices[row_index].item())
            active_indices.append(self.req_to_token_pool.req_to_token[req_pool_idx, :seq_len].to(torch.int64))
            self.req_to_token_pool.req_to_token[req_pool_idx, :seq_len] = 0

        if active_indices:
            self.token_to_kv_pool_allocator.free(torch.cat(active_indices))

    def update(
        self,
        attention_mask: torch.Tensor,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update sequence lengths and cache locations after a forward pass.

        Args:
            attention_mask: Mask indicating valid tokens.
                Shape: [batch_size, seq_length]. Dtype: long or bool.
            is_prefill: Whether this forward is an extend/prefill step rather than decode.

        Returns:
            Tuple of (old_seq_lengths, new_seq_lengths, out_cache_loc).
            old_seq_lengths: Previous sequence lengths. Shape: [batch_size]. Dtype: long.
            new_seq_lengths: Updated sequence lengths. Shape: [batch_size]. Dtype: long.
            out_cache_loc: Flat cache indices for written tokens.
                Shape: [num_valid_tokens]. Dtype: long.
        """
        old_seq_lengths = self.seq_lens.clone()
        extend_lens = attention_mask.to(torch.long).sum(dim=1)
        new_seq_lengths = old_seq_lengths + extend_lens

        total_new_tokens = int(extend_lens.sum().item())
        if total_new_tokens == 0:
            self.seq_lens = new_seq_lengths
            return old_seq_lengths, new_seq_lengths, torch.empty(0, dtype=torch.int64, device=self.device)

        allocator = self.token_to_kv_pool_allocator
        page_size = int(getattr(allocator, "page_size", 1))
        out_cache_loc: torch.Tensor | None
        if page_size == 1:
            out_cache_loc = allocator.alloc(total_new_tokens)
        else:
            last_loc = self._get_last_loc(old_seq_lengths)
            if is_prefill:
                out_cache_loc = allocator.alloc_extend(
                    prefix_lens=old_seq_lengths,
                    prefix_lens_cpu=old_seq_lengths.cpu(),
                    seq_lens=new_seq_lengths,
                    seq_lens_cpu=new_seq_lengths.cpu(),
                    last_loc=last_loc,
                    extend_num_tokens=total_new_tokens,
                )
            else:
                out_cache_loc = allocator.alloc_decode(
                    seq_lens=new_seq_lengths,
                    seq_lens_cpu=new_seq_lengths.cpu(),
                    last_loc=last_loc,
                )

        if out_cache_loc is None:
            available_size = allocator.available_size() if hasattr(allocator, "available_size") else "unknown"
            mode = "prefill" if is_prefill else "decode"
            raise RuntimeError(
                "Failed to allocate thinker KV slots for "
                f"{mode}: requested={total_new_tokens}, page_size={page_size}, available_size={available_size}."
            )
        out_cache_loc = out_cache_loc.to(torch.int64)

        start_idx = 0
        req_to_token_dtype = self.req_to_token_pool.req_to_token.dtype
        for req_pool_idx, old_seq_length, extend_len in zip(
            self.req_pool_indices,
            old_seq_lengths,
            extend_lens,
            strict=True,
        ):
            extend_len_int = int(extend_len.item())
            if extend_len_int == 0:
                continue

            end_idx = start_idx + extend_len_int
            req_pool_idx_int = int(req_pool_idx.item())
            old_seq_length_int = int(old_seq_length.item())
            new_seq_length_int = old_seq_length_int + extend_len_int
            self.req_to_token_pool.req_to_token[req_pool_idx_int, old_seq_length_int:new_seq_length_int] = (
                out_cache_loc[start_idx:end_idx].to(req_to_token_dtype)
            )
            start_idx = end_idx

        assert start_idx == total_new_tokens, "Cache slot writeback consumed an unexpected number of tokens."
        self.seq_lens = new_seq_lengths
        return old_seq_lengths, new_seq_lengths, out_cache_loc

    def reset(self) -> None:
        """Reset sequence lengths to zero for reuse of the metadata object."""
        self.free_allocated_kv_slots()
        self.seq_lens.zero_()


class SGLangRaonModel(RaonInferenceModel):
    """SGLang-compatible inference backend for duplex TTS and full-duplex streaming with custom KV-cache management."""

    vocab_size: int
    codebook_size: int
    num_code_groups: int
    sampling_rate: int
    frame_rate: float

    def __init__(
        self,
        path: str,
        dtype: str = "auto",
        mem_fraction_static: float = 0.75,
        disable_cuda_graph: bool | None = None,
        cuda_graph_max_bs: int | None = None,
        # Reduce SGLang continuous batching / pool sizing (optional).
        # These correspond to common SGLang server args; we set them on ServerArgs if present.
        max_running_requests: int | None = None,
        prefill_max_requests: int | None = None,
        max_total_tokens: int | None = None,
        max_prefill_tokens: int | None = None,
        chunked_prefill_size: int | None = None,
        gpu_id: int = 0,
        tp_rank: int = 0,
        tp_size: int = 1,
        moe_ep_rank: int = 0,
        moe_ep_size: int = 1,
        pp_rank: int = 0,
        pp_size: int = 1,
        nccl_port: int = 0,
        max_allocated_req_pool_indices: int = 1024,
    ) -> None:
        """Initialize the SGLang duplex model by loading the duplex submodel and text model runner from a bundle path."""
        from pathlib import Path

        bundle_path = Path(path)
        runtime_model_path = bundle_path / "raon_runtime"
        if not runtime_model_path.is_dir():
            runtime_model_path = bundle_path / "duplex_model"
        if not runtime_model_path.is_dir():
            raise FileNotFoundError(
                f"Expected a runtime model directory at {bundle_path / 'raon_runtime'} or {bundle_path / 'duplex_model'}."
            )
        self.device = f"cuda:{gpu_id}"
        self.dtype = getattr(torch, dtype, torch.bfloat16)

        self.raon_duplex_model = load_raon_model_without_text_model(
            path=str(runtime_model_path),
            device=self.device,
            dtype=self.dtype,
        )

        self.text_model_runner = self._load_text_model_runner(
            path=str(bundle_path / "text_model"),
            dtype=dtype,
            mem_fraction_static=mem_fraction_static,
            disable_cuda_graph=disable_cuda_graph,
            cuda_graph_max_bs=cuda_graph_max_bs,
            max_running_requests=max_running_requests,
            prefill_max_requests=prefill_max_requests,
            max_total_tokens=max_total_tokens,
            max_prefill_tokens=max_prefill_tokens,
            chunked_prefill_size=chunked_prefill_size,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
        )

        assert isinstance(lm_head := self.text_model_runner.model.lm_head, ParallelLMHead), (
            "LM head must be a ParallelLMHead."
        )
        assert isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod), (
            "LM head quant_method must be UnquantizedEmbeddingMethod."
        )
        assert isinstance(lm_head_weight := lm_head.weight, torch.Tensor), "LM head weight must be a tensor."
        self.lm_head_weight = lm_head_weight

        self.thinker_capture_layer_index = self.raon_duplex_model.thinker_capture_layer_index
        self.accepted_thinker_hidden_states: torch.Tensor | None = None
        self.register_thinker_capture_hook()

        self.supports_audio_output = bool(getattr(self.raon_duplex_model, "supports_audio_output", True))
        if self.supports_audio_output:
            assert self.raon_duplex_model.talker is not None, "Separated talker is required."
            assert self.raon_duplex_model.thinker_to_talker_proj is not None, "thinker_to_talker_proj is required."
        self._talker_past_key_values: DynamicCache | None = None
        self._talker_attention_mask: torch.Tensor | None = None

        self.vocab_size = self.raon_duplex_model.vocab_size
        self.codebook_size = self.raon_duplex_model.codebook_size
        self.use_duplex_end_pad = self.raon_duplex_model.use_duplex_end_pad
        self.num_code_groups = self.raon_duplex_model.num_code_groups
        self.sampling_rate = self.raon_duplex_model.sampling_rate
        self.frame_rate = self.raon_duplex_model.frame_rate
        self.delays = self.raon_duplex_model.delays
        self.max_delay = self.raon_duplex_model.max_delay
        self.sequence_mode = self.raon_duplex_model.sequence_mode
        self.use_sil_token = self.raon_duplex_model.use_sil_token
        self.use_backchannel_token = getattr(self.raon_duplex_model, "use_backchannel_token", False)
        self.no_audio_in_sil = self.raon_duplex_model.no_audio_in_sil
        self.duplex_sil_token_id = self.raon_duplex_model.duplex_sil_token_id
        self.duplex_bc_token_id = getattr(self.raon_duplex_model, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id)
        self.tokenizer = getattr(self.raon_duplex_model, "tokenizer", None)
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(str(runtime_model_path), local_files_only=True)
            except Exception:
                self.tokenizer = None

        self.max_allocated_req_pool_indices = max_allocated_req_pool_indices
        self.allocated_req_pool_indices: set[int] = set()

        RaonInferenceModel.__init__(self)

    def register_thinker_capture_hook(self) -> None:
        """Register a forward hook on the thinker capture layer to capture pre-norm hidden states.

        In SGLang/vLLM, decoder layers use fused residual-add + RMSNorm and return
        ``(hidden_states, residual)`` where ``hidden_states`` is the MLP output delta
        and ``residual`` is the accumulated residual stream.  The actual pre-norm hidden
        state (before the model's final RMSNorm) is ``hidden_states + residual``.

        In HuggingFace Transformers, each layer returns the full hidden state directly,
        so ``output[0]`` is already the pre-norm hidden state.
        """
        model = self.text_model_runner.model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        else:
            raise AttributeError("register_thinker_capture_hook: Cannot find layers in text_model_runner.model.")

        def hook(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, tuple) and len(output) == 2:
                # SGLang/vLLM returns (hidden_states, residual); pre-norm = sum.
                self.accepted_thinker_hidden_states = output[0] + output[1]
            elif isinstance(output, tuple):
                self.accepted_thinker_hidden_states = output[0]
            else:
                self.accepted_thinker_hidden_states = output

        cast(nn.Module, layers[self.thinker_capture_layer_index]).register_forward_hook(hook)

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
        past_key_values: SGLangDecodingMetadata | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single inference step; returns talker hidden states and text logits.

        Args:
            input_ids: Token IDs. Shape: [batch_size, seq_length]. Dtype: long.
            attention_mask: Mask for valid tokens. Shape: [batch_size, seq_length]. Dtype: long or bool.
            position_ids: Token positions. Shape: [batch_size, seq_length]. Dtype: long.
            audio_input: Raw waveform for input-side conditioning.
                Shape: [batch_size, num_samples]. Dtype: float.
            audio_output: Raw waveform for output conditioning.
                Shape: [batch_size, num_samples]. Dtype: float.
            audio_input_lengths: Valid sample count per batch.
                Shape: [batch_size]. Dtype: long.
            audio_output_lengths: Valid sample count per batch.
                Shape: [batch_size]. Dtype: long.
            audio_output_codes: Pre-tokenized audio codes.
                Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.
            audio_output_codes_mask: Mask for valid audio codes.
                Shape: [batch_size, num_frames]. Dtype: bool.
            audio_input_embeds: Pre-computed audio input embeddings.
                Shape: [batch_size, num_frames, feature_dim]. Dtype: float.
            audio_input_embeds_mask: Mask for valid audio input positions.
                Shape: [batch_size, num_frames]. Dtype: bool.
            speaker_embeds: Speaker embeddings.
                Shape: [batch_size, 1, speaker_dim]. Dtype: float.
            use_cache: Unused; SGLang uses metadata-based cache.
            past_key_values: Decoding metadata (KV-cache state); required.

        Returns:
            Tuple of (padded_talker_last_hidden_states, logits).
            padded_talker_last_hidden_states: Talker hidden states, padded to input shape.
                Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            logits: Text logits over vocab.
                Shape: [batch_size, seq_length, vocab_size]. Dtype: float.
        """
        assert (metadata := past_key_values) is not None, "past_key_values (decoding metadata) is required."

        # Separate the current-frame mask (for SGLang thinker) from the cumulative mask (for HF talker).
        # The thinker uses SGLang's metadata-based KV cache and only needs the current frame's mask.
        # The talker uses HF DynamicCache and needs a full-sequence mask covering all cached + current positions.
        if attention_mask is None:
            current_frame_mask = torch.ones_like(input_ids)
            attention_mask = current_frame_mask
        else:
            current_frame_mask = attention_mask[:, -input_ids.shape[1] :]
            attention_mask = current_frame_mask

        if audio_output_codes is None and self.supports_audio_output:
            assert self.raon_duplex_model.config.code_predictor_config.num_code_groups is not None, (
                "Config code_predictor num_code_groups must be set."
            )
            audio_output_inputs = self.raon_duplex_model.tokenize_audio(
                audio=audio_output,
                audio_lengths=audio_output_lengths,
                num_code_groups=self.raon_duplex_model.config.code_predictor_config.num_code_groups,
            )
            audio_output_codes = audio_output_inputs.audio_codes
            audio_output_codes_mask = audio_output_inputs.audio_codes_mask

        if audio_input_embeds is None and audio_input is not None:
            audio_input_outputs = self.raon_duplex_model.get_audio_input_embeds(
                audio=audio_input,
                audio_lengths=audio_input_lengths,
            )
            audio_input_embeds = audio_input_outputs.audio_embeds
            audio_input_embeds_mask = audio_input_outputs.audio_embeds_mask

        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = self.raon_duplex_model.update_inputs_embeds(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            audio_output_codes=audio_output_codes,
            audio_output_codes_mask=audio_output_codes_mask,
            audio_input_embeds=audio_input_embeds,
            audio_input_embeds_mask=audio_input_embeds_mask,
            speaker_embeds=speaker_embeds,
        )

        inputs = self.get_inputs_and_update_metadata(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            metadata=metadata,
        )
        runner_output = self.text_model_runner.forward(inputs)
        if hasattr(runner_output, "logits_output"):
            outputs = runner_output.logits_output
        elif isinstance(runner_output, tuple):
            outputs = runner_output[0]
        else:
            outputs = runner_output
        assert isinstance(outputs, LogitsProcessorOutput), (
            "text_model_runner.forward(...) must return a LogitsProcessorOutput or wrap one."
        )

        # Runner output is the thinker's post-norm hidden states (text_model.norm applied inside runner).
        thinker_post_norm: torch.Tensor = outputs.hidden_states

        # Capture pre-norm hidden states from hook (set by register_thinker_capture_hook).
        assert self.accepted_thinker_hidden_states is not None, (
            "accepted_thinker_hidden_states must be set by thinker capture hook."
        )
        captured_pre_norm = self.accepted_thinker_hidden_states
        self.accepted_thinker_hidden_states = None

        # Pad post-norm for text logits.
        padded_thinker = torch.zeros(
            (*input_ids.shape, thinker_post_norm.shape[-1]),
            dtype=thinker_post_norm.dtype,
            device=thinker_post_norm.device,
        )
        padded_thinker[attention_mask == 1] = thinker_post_norm
        logits = F.linear(padded_thinker, self.lm_head_weight)[..., : self.vocab_size]

        if not self.supports_audio_output:
            # Audio output disabled: skip talker computation entirely.
            # Return dummy talker hidden states (never used when force_text_output=True).
            return logits.new_zeros((*input_ids.shape, 1)), logits

        # Pad pre-norm for talker input.
        padded_captured = torch.zeros(
            (*input_ids.shape, captured_pre_norm.shape[-1]),
            dtype=captured_pre_norm.dtype,
            device=captured_pre_norm.device,
        )
        padded_captured[attention_mask == 1] = captured_pre_norm

        # Project thinker hidden states → talker input, then run talker with cache.
        # Build cumulative attention mask for the HF talker: past (all cached) + current frame.
        if self._talker_attention_mask is None:
            talker_mask = current_frame_mask
        else:
            talker_mask = torch.cat([self._talker_attention_mask, current_frame_mask], dim=1)
        self._talker_attention_mask = talker_mask

        talker_input = self.raon_duplex_model.thinker_to_talker_proj(padded_captured)
        talker_outputs = self.raon_duplex_model.talker(
            inputs_embeds=talker_input,
            attention_mask=talker_mask,
            past_key_values=self._talker_past_key_values,
            use_cache=True,
        )
        self._talker_past_key_values = talker_outputs.past_key_values

        return talker_outputs.last_hidden_state, logits

    def get_inputs_and_update_metadata(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        metadata: SGLangDecodingMetadata,
    ) -> ForwardBatch:
        """Flatten inputs for SGLang, update metadata, and build a ForwardBatch for the text model runner.

        Args:
            inputs_embeds: Combined token embeddings.
                Shape: [batch_size, seq_length, hidden_size]. Dtype: float.
            attention_mask: Mask for valid positions.
                Shape: [batch_size, seq_length]. Dtype: long or bool.
            position_ids: Token positions. Shape: [batch_size, seq_length]. Dtype: long.
            metadata: Decoding metadata to update with new sequence lengths and cache locations.

        Returns:
            ForwardBatch ready for text_model_runner.forward().
        """
        assert inputs_embeds.shape[0] == metadata.batch_size, "inputs_embeds batch size must match metadata batch_size."
        is_prefill = inputs_embeds.shape[1] > 1

        old_seq_lengths, new_seq_lengths, out_cache_loc = metadata.update(
            attention_mask=attention_mask,
            is_prefill=is_prefill,
        )
        new_seq_lengths_sum = int(new_seq_lengths.sum().item())

        flat_inputs_embeds = inputs_embeds[attention_mask == 1]
        dummy_input_ids = torch.zeros(flat_inputs_embeds.shape[0], dtype=torch.long, device=inputs_embeds.device)

        batch = ModelWorkerBatch(
            forward_mode=ForwardMode.EXTEND if is_prefill else ForwardMode.DECODE,
            input_ids=dummy_input_ids,
            req_pool_indices=metadata.req_pool_indices,
            seq_lens=new_seq_lengths,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=new_seq_lengths.cpu(),
            seq_lens_sum=new_seq_lengths_sum,
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=is_prefill,
            can_run_dp_cuda_graph=False,
            tbo_split_seq_index=None,
            global_forward_mode=None,
            extend_num_tokens=flat_inputs_embeds.shape[0] if is_prefill else None,
            extend_seq_lens=(new_seq_lengths - old_seq_lengths).cpu().tolist() if is_prefill else None,
            extend_prefix_lens=old_seq_lengths.cpu().tolist() if is_prefill else None,
            extend_logprob_start_lens=None,
            extend_input_logprob_token_ids=None,
            multimodal_inputs=None,
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=None,
            sampling_info=metadata.sampling_info,
            input_embeds=flat_inputs_embeds,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        return ForwardBatch.init_new(batch, model_runner=self.text_model_runner)

    def tokenize_audio(self, *args: Any, **kwargs: Any) -> AudioTokenizerOutput:
        """Tokenize audio into codes; delegates to the underlying duplex model.

        Typical args: audio, audio_lengths, num_code_groups. Returns audio_codes and audio_codes_mask.

        Returns:
            AudioTokenizerOutput with audio_codes (Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.),
            audio_codes_mask (Shape: [batch_size, num_frames]. Dtype: bool.), and optional mimi_features.
        """
        return self.raon_duplex_model.tokenize_audio(*args, **kwargs)

    def get_audio_input_embeds(self, *args: Any, **kwargs: Any) -> AudioEncoderOutput:
        """Compute audio input embeddings from raw waveform; delegates to the underlying duplex model.

        Typical args: audio, audio_lengths. Returns audio_embeds and audio_embeds_mask.

        Returns:
            AudioEncoderOutput with audio_embeds (Shape: [batch_size, num_frames, feature_dim]. Dtype: float.),
            audio_embeds_mask (Shape: [batch_size, num_frames]. Dtype: bool.).
        """
        return self.raon_duplex_model.get_audio_input_embeds(*args, **kwargs)

    def decode_audio(self, *args: Any, **kwargs: Any) -> AudioDecoderOutput:
        """Decode audio codes to waveform; delegates to the underlying duplex model.

        Typical first arg: audio_codes. Shape: [batch_size, num_frames, num_code_groups]. Dtype: long.

        Returns:
            AudioDecoderOutput with audio waveform.
                Shape: [batch_size, num_channels, num_samples]. Dtype: float.
        """
        return self.raon_duplex_model.decode_audio(*args, **kwargs)

    def generate_audio_codes(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Generate next-frame audio codes from talker hidden state; delegates to the underlying duplex model.

        Typical first arg: talker_last_hidden_state. Shape: [batch_size, seq_length, hidden_size]. Dtype: float.

        Returns:
            Sampled audio codes for the next frame. Shape: [batch_size, num_code_groups]. Dtype: long.
        """
        return self.raon_duplex_model.generate_audio_codes(*args, **kwargs)

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the text model's token embedding layer."""
        model = self.text_model_runner.model
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens  # type: ignore
        elif hasattr(model, "embed_tokens"):
            return model.embed_tokens  # type: ignore
        else:
            raise AttributeError("Cannot find embed_tokens in text_model_runner.model.")

    def get_proj_code(self) -> nn.Linear:
        """Return the audio code projection layer."""
        return self.raon_duplex_model.get_proj_code()

    def get_model(self) -> RaonModel:
        """Return the underlying duplex model (without the SGLang text model runner)."""
        return self.raon_duplex_model

    def init_past_key_values(
        self,
        batch_size: int,
        max_sequence_length: int,
        prev_cache: SGLangDecodingMetadata | None = None,
    ) -> SGLangDecodingMetadata:
        """Initialize or reuse decoding metadata for KV-cache tracking.

        Args:
            batch_size: Number of concurrent requests.
            max_sequence_length: Maximum sequence length for the cache.
            prev_cache: Optional existing metadata to reset and reuse.

        Returns:
            SGLangDecodingMetadata with req_pool_indices, seq_lens, and sampling info.
        """
        # Initialize talker KV cache and cumulative attention mask.
        self._talker_past_key_values = DynamicCache()
        self._talker_attention_mask = None

        assert isinstance(req_to_token_pool := self.text_model_runner.req_to_token_pool, ReqToTokenPool), (
            "req_to_token_pool must be a ReqToTokenPool instance."
        )
        req_pool_capacity = int(getattr(req_to_token_pool, "size", req_to_token_pool.req_to_token.shape[0]))
        assert self.max_allocated_req_pool_indices <= req_pool_capacity, (
            "max_allocated_req_pool_indices must not exceed req_to_token_pool size: "
            f"{self.max_allocated_req_pool_indices} > {req_pool_capacity}."
        )
        assert hasattr(self.text_model_runner, "token_to_kv_pool_allocator"), (
            "text_model_runner must expose token_to_kv_pool_allocator."
        )
        token_to_kv_pool_allocator = self.text_model_runner.token_to_kv_pool_allocator

        if prev_cache is not None:
            assert prev_cache.req_pool_indices.numel() == batch_size, (
                "prev_cache batch size must match the requested batch size for in-place reuse."
            )
            # `prev_cache` is consumed in place by the new state; callers must not free the old owner afterwards.
            prev_cache.reset()
            prev_cache.req_to_token_pool = req_to_token_pool
            prev_cache.token_to_kv_pool_allocator = token_to_kv_pool_allocator
            prev_cache.max_sequence_length = max_sequence_length
            prev_cache.batch_size = batch_size
            return prev_cache

        sampling_info = SamplingBatchInfo(
            temperatures=torch.tensor([[1.0]], device=self.device),
            top_ps=torch.tensor([1.0], device=self.device),
            top_ks=torch.tensor([0], dtype=torch.int32, device=self.device),
            min_ps=torch.tensor([0.0], device=self.device),
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=self.vocab_size,
        )

        return SGLangDecodingMetadata(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            req_pool_indices=self.allocate_req_pool_indices(batch_size),
            seq_lens=torch.zeros(batch_size, dtype=torch.long, device=self.device),
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            sampling_info=sampling_info,
            device=self.device,
        )

    def allocate_req_pool_indices(self, batch_size: int) -> torch.Tensor:
        """Allocate request pool indices from the available pool for a batch of requests.

        Args:
            batch_size: Number of indices to allocate.

        Returns:
            Tensor of req_pool indices. Shape: [batch_size]. Dtype: long.
        """
        req_to_token_pool = self.text_model_runner.req_to_token_pool
        req_pool_capacity = int(getattr(req_to_token_pool, "size", req_to_token_pool.req_to_token.shape[0]))
        alloc_limit = min(self.max_allocated_req_pool_indices, req_pool_capacity)

        indices: list[int] = []
        for i in range(alloc_limit):
            if len(indices) >= batch_size:
                break

            if i not in self.allocated_req_pool_indices:
                self.allocated_req_pool_indices.add(i)
                indices.append(i)

        assert len(indices) == batch_size, f"Failed to allocate {batch_size} req pool indices."
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def free_req_pool_indices(self, metadata: SGLangDecodingMetadata) -> None:
        """Return req_pool indices from metadata to the pool for reuse."""
        for i in metadata.req_pool_indices.tolist():
            self.allocated_req_pool_indices.discard(i)

    def free_past_key_values(self, past_key_values: SGLangDecodingMetadata) -> None:
        """Release resources associated with past_key_values (req_pool indices and talker cache)."""
        past_key_values.free_allocated_kv_slots()
        past_key_values.seq_lens.zero_()
        self.free_req_pool_indices(past_key_values)
        self._talker_past_key_values = None
        self._talker_attention_mask = None

    @staticmethod
    def _load_text_model_runner(
        path: str,
        dtype: str,
        mem_fraction_static: float,
        disable_cuda_graph: bool | None,
        cuda_graph_max_bs: int | None,
        max_running_requests: int | None,
        prefill_max_requests: int | None,
        max_total_tokens: int | None,
        max_prefill_tokens: int | None,
        chunked_prefill_size: int | None,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
    ) -> ModelRunner:
        set_logs(recompiles=False)
        model_config = ModelConfig(model_path=path, dtype=dtype)
        server_args = ServerArgs(model_path=path, dtype=dtype)

        # Ensure SGLang uses our requested memory fraction (Ray/Hydra override).
        # Some sglang versions consult `server_args.mem_fraction_static` during pool init.
        if hasattr(server_args, "mem_fraction_static"):
            server_args.mem_fraction_static = mem_fraction_static

        # Reduce/disable CUDA graph capture (avoids "Capturing batches ..." init-time memory spikes).
        if disable_cuda_graph is not None:
            for attr in ("disable_cuda_graph", "disable_cuda_graph_capture"):
                if hasattr(server_args, attr):
                    setattr(server_args, attr, bool(disable_cuda_graph))

        if cuda_graph_max_bs is not None:
            for attr in ("cuda_graph_max_bs", "cuda_graph_max_batch_size"):
                if hasattr(server_args, attr):
                    setattr(server_args, attr, int(cuda_graph_max_bs))

        # Reduce continuous batching / pool sizing (helps multi-worker-per-GPU).
        if max_running_requests is not None and hasattr(server_args, "max_running_requests"):
            server_args.max_running_requests = int(max_running_requests)

        if prefill_max_requests is not None and hasattr(server_args, "prefill_max_requests"):
            server_args.prefill_max_requests = int(prefill_max_requests)

        if max_total_tokens is not None and hasattr(server_args, "max_total_tokens"):
            server_args.max_total_tokens = int(max_total_tokens)

        if max_prefill_tokens is not None and hasattr(server_args, "max_prefill_tokens"):
            server_args.max_prefill_tokens = int(max_prefill_tokens)

        if chunked_prefill_size is not None and hasattr(server_args, "chunked_prefill_size"):
            server_args.chunked_prefill_size = int(chunked_prefill_size)

        model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=moe_ep_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
        )

        def patched_forward_extend(
            self: ModelRunner,
            forward_batch: ForwardBatch,
            skip_attn_backend_init: bool = False,
            pp_proxy_tensors: Any = None,
        ) -> tuple[LogitsProcessorOutput, bool]:
            kwargs = {}
            if self.support_pp:
                kwargs["pp_proxy_tensors"] = pp_proxy_tensors

            if forward_batch.input_embeds is not None:
                kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()

            if not self.is_generation:
                kwargs["get_embedding"] = True

            can_run_graph = (
                self.piecewise_cuda_graph_runner is not None
                and self.piecewise_cuda_graph_runner.can_run(forward_batch)
            )

            if self.piecewise_cuda_graph_runner is not None:
                if can_run_graph:
                    return self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs), can_run_graph  # type: ignore

            if not skip_attn_backend_init:
                self.attn_backend.init_forward_metadata(forward_batch)

            return (
                self.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                ),
                can_run_graph,
            )

        def patched_forward_decode(
            self: ModelRunner,
            forward_batch: ForwardBatch,
            skip_attn_backend_init: bool = False,
            pp_proxy_tensors: Any = None,
        ) -> LogitsProcessorOutput:
            if not skip_attn_backend_init:
                if getattr(self.server_args, "enable_pdmux", False):
                    self.decode_attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.attn_backend = self.decode_attn_backend
                else:
                    self.attn_backend.init_forward_metadata(forward_batch)

            kwargs = {}
            if self.support_pp:
                kwargs["pp_proxy_tensors"] = pp_proxy_tensors

            if forward_batch.input_embeds is not None:
                kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()

            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        model_runner.forward_extend = types.MethodType(patched_forward_extend, model_runner)
        model_runner.forward_decode = types.MethodType(patched_forward_decode, model_runner)

        set_logs(recompiles=True)
        return model_runner

    # ------------------------------------------------------------------
    # Batched duplex generation (n samples in parallel)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def duplex_generate_batch(
        self,
        sequences: torch.Tensor,
        audio_input: torch.Tensor,
        n: int = 4,
        speaker_embeds: torch.Tensor | None = None,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.8,
        max_sequence_length: int = 2048,
        disable_tqdm: bool = True,
        eos_penalty: float = 0.0,
        speak_first: bool = False,
    ) -> list[RaonGenerateResult]:
        """Generate n duplex responses in parallel using batched forward passes.

        Instead of calling duplex_generate_with_fixed_input() n times sequentially,
        this method runs n independent decoding states through a single batched
        inference_forward() call per frame, yielding near-n× speedup on the GPU
        bottleneck.

        Args:
            sequences: Prompt token IDs (shared across all n).
                Shape: [1, seq_length]. Dtype: long.
            audio_input: User audio waveform.
                Shape: [1, audio_length]. Dtype: float.
            n: Number of parallel responses to generate.
            speaker_embeds: Optional speaker embedding.
                Shape: [1, 1, speaker_dim]. Dtype: float.
            do_sample: Whether to sample (True) or argmax (False).
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p nucleus filtering.
            max_sequence_length: Max KV cache length.
            disable_tqdm: Disable progress bar.
            eos_penalty: Penalty subtracted from pad/eos logit.

        Returns:
            List of n RaonGenerateResult with per-sample sequences and audio.
        """
        from tqdm.auto import trange
        from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

        logger = logging.getLogger(__name__)

        assert sequences.shape[0] == 1, f"Expected batch=1 prompt, got {sequences.shape[0]}"
        samples_per_frame = int(self.sampling_rate / self.frame_rate)
        audio_input_length = audio_input.shape[-1]
        if audio_input.ndim == 1:
            audio_input = audio_input[None]
        assert audio_input.shape == (1, audio_input_length)
        assert audio_input_length >= samples_per_frame

        device = sequences.device
        use_condensed_silence = self._use_condensed_silence()

        # --- Enable CUDA-graph code prediction for batched inference (one-time cost) ---
        code_predictor = self.raon_duplex_model.code_predictor
        if not getattr(code_predictor, "_batch_cuda_graph_enabled", False):
            code_predictor.enable_predict_codes_cuda_graph()
            code_predictor._batch_cuda_graph_enabled = True  # type: ignore[attr-defined]
            logger.info("Enabled CUDA graph mode for code_predictor.predict_codes")

        # --- Build logits processor ---
        logits_processor = LogitsProcessorList()
        if do_sample and temperature and temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(temperature=temperature))
        if do_sample and top_k and top_k > 0:
            logits_processor.append(TopKLogitsWarper(top_k=top_k))
        if do_sample and top_p and top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(top_p=top_p))

        # --- Start concurrent audio decoder ---
        self.start_concurrent_audio_decoder()

        # --- Build init sequence: prompt + [im_start] (+ [audio_start] for non-SIL) ---
        if speak_first and use_condensed_silence and self.use_duplex_end_pad:
            init_suffix = torch.tensor([[IM_START.id, AUDIO_OUTPUT_END_PAD.id, AUDIO_START.id]], device=device)
        elif use_condensed_silence:
            init_suffix = torch.tensor([[IM_START.id]], device=device)
        else:
            init_suffix = torch.tensor([[IM_START.id, AUDIO_START.id]], device=device)
        init_ids = torch.cat([sequences, init_suffix], dim=1)  # [1, prompt_len + suffix]
        init_len = init_ids.shape[1]
        empty_audio_codes = torch.zeros(1, 0, self.num_code_groups, dtype=torch.long, device=device)
        empty_audio_codes_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
        init_attention_mask = torch.ones_like(init_ids)
        silence_codes = torch.zeros(self.num_code_groups, device=device, dtype=torch.long)
        initial_machine_state = RaonMachineState(
            phase=RaonPhase.SIL,
            last_frame_tokens=[IM_START.id, AUDIO_START.id],
        )

        # --- Allocate shared KV cache for n sequences ---
        metadata = self.init_past_key_values(batch_size=n, max_sequence_length=max_sequence_length)

        # --- Batched prefill: replicate prompt n times ---
        batch_ids = init_ids.expand(n, -1)  # [n, init_len]
        position_ids = torch.arange(init_len, device=device).unsqueeze(0).expand(n, -1)
        if speaker_embeds is not None and n > 1:
            speaker_embeds = speaker_embeds.expand(n, -1, -1)

        last_hidden_state, logits = self.inference_forward(
            input_ids=batch_ids,
            attention_mask=None,
            position_ids=position_ids,
            speaker_embeds=speaker_embeds,
            use_cache=True,
            past_key_values=metadata,
        )

        # --- Initialize n per-sample states ---
        states: list[_BatchSampleState] = []
        for i in range(n):
            stream_id = self.create_audio_decoder_stream()

            if speak_first and use_condensed_silence and self.use_duplex_end_pad:
                # Speak-first init: [IM_S, EPAD, A_S] already prefilled.
                # Generate first text + audio codes from prefill logits.
                s_audio_codes = empty_audio_codes
                s_audio_codes_mask = empty_audio_codes_mask
                sample_logits = logits[i : i + 1]
                sample_hidden = last_hidden_state[i : i + 1]
                sample_machine_state = initial_machine_state

                _, s_seq, s_attn, s_audio_codes, s_audio_codes_mask, initial_emitted_audio, _, sample_machine_state = (
                    self._update_duplex_sequences_and_generate_audio_codes(
                        new_logits=sample_logits,
                        new_last_hidden_state=sample_hidden,
                        sequences=init_ids.clone(),
                        attention_mask=init_attention_mask,
                        audio_codes=s_audio_codes,
                        audio_codes_mask=s_audio_codes_mask,
                        do_sample=do_sample,
                        logits_processor=logits_processor,
                        machine_state=sample_machine_state,
                    )
                )

                initial_semantic_buffer = None
                if initial_emitted_audio and self.max_delay > 0:
                    first_codes = s_audio_codes[0, -1]
                    semantic_code = first_codes[0:1]
                    output_codes = torch.zeros_like(first_codes)
                    output_codes[0] = semantic_code[0]
                    initial_semantic_buffer = semantic_code
                    self.push_audio_codes(audio_codes=output_codes, stream_id=stream_id)
                elif initial_emitted_audio:
                    self.push_audio_codes(audio_codes=s_audio_codes[0, -1], stream_id=stream_id)
                else:
                    self.push_audio_codes(audio_codes=silence_codes, stream_id=stream_id)
            elif use_condensed_silence:
                # SIL init: append [U] to start in condensed silence mode.
                sil_u = torch.tensor([[AUDIO_INPUT_PLACEHOLDER.id]], device=device)
                s_seq = torch.cat([init_ids, sil_u], dim=1)
                s_attn = torch.ones_like(s_seq)
                s_audio_codes = empty_audio_codes
                s_audio_codes_mask = empty_audio_codes_mask
                initial_semantic_buffer = None
                initial_emitted_audio = False
                sample_machine_state = initial_machine_state

                # Push silence placeholder to keep decoder pipeline initialized.
                self.push_audio_codes(audio_codes=silence_codes, stream_id=stream_id)
            else:
                s_audio_codes = empty_audio_codes
                s_audio_codes_mask = empty_audio_codes_mask
                # Per-sample logits from prefill (batch dim i).
                sample_logits = logits[i : i + 1]
                sample_hidden = last_hidden_state[i : i + 1]
                sample_machine_state = initial_machine_state
                forced_initial_prediction_id = self._state_manager.initial_forced_prediction_id(speak_first)
                if forced_initial_prediction_id is not None:
                    forced_logits = torch.full_like(sample_logits, fill_value=-1e9)
                    forced_logits[:, -2, forced_initial_prediction_id] = 0.0
                    sample_logits = forced_logits

                _, s_seq, s_attn, s_audio_codes, s_audio_codes_mask, initial_emitted_audio, _, sample_machine_state = (
                    self._update_duplex_sequences_and_generate_audio_codes(
                        new_logits=sample_logits,
                        new_last_hidden_state=sample_hidden,
                        sequences=init_ids.clone(),
                        attention_mask=init_attention_mask,
                        audio_codes=s_audio_codes,
                        audio_codes_mask=s_audio_codes_mask,
                        do_sample=do_sample,
                        logits_processor=logits_processor,
                        machine_state=sample_machine_state,
                    )
                )

                initial_semantic_buffer = None
                if initial_emitted_audio and self.max_delay > 0:
                    first_codes = s_audio_codes[0, -1]
                    semantic_code = first_codes[0:1]
                    output_codes = torch.zeros_like(first_codes)
                    output_codes[0] = semantic_code[0]
                    initial_semantic_buffer = semantic_code
                    self.push_audio_codes(audio_codes=output_codes, stream_id=stream_id)
                elif initial_emitted_audio:
                    self.push_audio_codes(audio_codes=s_audio_codes[0, -1], stream_id=stream_id)
                else:
                    self.push_audio_codes(audio_codes=silence_codes, stream_id=stream_id)

            states.append(
                _BatchSampleState(
                    sequences=s_seq,
                    attention_mask=s_attn,
                    audio_codes=s_audio_codes,
                    audio_codes_mask=s_audio_codes_mask,
                    semantic_buffer=initial_semantic_buffer,
                    audio_decoder_stream_id=stream_id,
                    logits_processor=logits_processor,
                    eos_penalty=eos_penalty,
                    emitted_audio=initial_emitted_audio,
                    machine_state=sample_machine_state,
                )
            )

        # For condensed silence mode, [U] is appended to sequences but NOT forwarded
        # through KV cache. The first frame loop iteration will process it.

        # --- Pre-encode all audio frames at once ---
        audio_input_lengths_full = torch.tensor([audio_input_length], dtype=torch.long, device=device)
        all_audio_output = self.get_audio_input_embeds(
            audio=audio_input,
            audio_lengths=audio_input_lengths_full,
        )
        all_audio_embeds = all_audio_output.audio_embeds  # [1, total_frames, feature_dim]
        all_audio_embeds_mask = all_audio_output.audio_embeds_mask  # [1, total_frames]

        # --- Frame loop ---
        initial_seq_len = sequences.shape[1]
        total_frames = all_audio_embeds.shape[1]
        t_start = time.perf_counter()

        for frame_idx in trange(total_frames, desc=f"Batch({n}) Duplex", disable=disable_tqdm, mininterval=0):
            # Index pre-computed audio embed for this frame.
            audio_embeds = all_audio_embeds[:, frame_idx : frame_idx + 1, :]  # [1, 1, feature_dim]
            audio_embeds_mask = all_audio_embeds_mask[:, frame_idx : frame_idx + 1]  # [1, 1]

            # 1. Determine per-sample input token count using ABC helpers.
            num_input_tokens_list = [
                state.machine_state.num_input_tokens if state.machine_state is not None else 2 for state in states
            ]

            # 2. Left-pad and batch inputs.
            max_input = max(num_input_tokens_list)

            batch_input_ids_list: list[torch.Tensor] = []
            batch_attn_mask_list: list[torch.Tensor] = []
            batch_position_ids_list: list[torch.Tensor] = []
            batch_audio_codes_list: list[torch.Tensor | None] = []
            batch_audio_codes_mask_list: list[torch.Tensor | None] = []

            for i in range(n):
                nit = num_input_tokens_list[i]
                ids = states[i].sequences[:, -nit:]  # [1, nit]
                pad_len = max_input - nit

                ids_padded = F.pad(ids, (pad_len, 0), value=0)
                mask_padded = F.pad(torch.ones(1, nit, dtype=torch.long, device=device), (pad_len, 0), value=0)

                full_pos = states[i].attention_mask.cumsum(dim=1) - 1
                pos = full_pos[:, -nit:]
                pos_padded = F.pad(pos, (pad_len, 0), value=0)

                # Audio codes: provide last frame's codes for embedding lookup, or None for SIL frames.
                if not states[i].emitted_audio:
                    step_codes: torch.Tensor | None = None
                    step_codes_mask: torch.Tensor | None = None
                else:
                    step_codes = states[i].audio_codes[:, -1:] if states[i].audio_codes.shape[1] > 0 else None
                    step_codes_mask = states[i].audio_codes_mask[:, -1:] if states[i].audio_codes_mask.shape[1] > 0 else None

                batch_input_ids_list.append(ids_padded)
                batch_attn_mask_list.append(mask_padded)
                batch_position_ids_list.append(pos_padded)
                batch_audio_codes_list.append(step_codes)
                batch_audio_codes_mask_list.append(step_codes_mask)

            batched_ids = torch.cat(batch_input_ids_list, dim=0)  # [n, max_input]
            batched_mask = torch.cat(batch_attn_mask_list, dim=0)
            batched_pos = torch.cat(batch_position_ids_list, dim=0)

            # Stack audio codes: use dummy zeros where None.
            dummy_codes = torch.zeros(1, 1, self.num_code_groups, dtype=torch.long, device=device)
            dummy_codes_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
            stacked_codes = torch.cat([c if c is not None else dummy_codes for c in batch_audio_codes_list], dim=0)
            stacked_codes_mask = torch.cat(
                [m if m is not None else dummy_codes_mask for m in batch_audio_codes_mask_list], dim=0
            )

            # Replicate audio embeds: [1, 1, D] -> [n, 1, D].
            batched_audio_embeds = audio_embeds.expand(n, -1, -1)
            batched_audio_embeds_mask = audio_embeds_mask.expand(n, -1) if audio_embeds_mask is not None else None

            # 3. Single batched forward.
            hidden_states, text_logits = self.inference_forward(
                input_ids=batched_ids,
                attention_mask=batched_mask,
                position_ids=batched_pos,
                audio_output_codes=stacked_codes,
                audio_output_codes_mask=stacked_codes_mask,
                audio_input_embeds=batched_audio_embeds,
                audio_input_embeds_mask=batched_audio_embeds_mask,
                speaker_embeds=speaker_embeds,
                use_cache=True,
                past_key_values=metadata,
            )

            # 4. Per-sample text sampling with deferred audio code generation.
            sample_deferred_hidden: list[torch.Tensor | None] = []
            sample_emitted: list[bool] = []
            sample_updated: list[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RaonMachineState | None]
            ] = []

            for i in range(n):
                nit = num_input_tokens_list[i]
                # Extract only valid positions for this sample (left-padded output).
                logits_i = text_logits[i : i + 1, max_input - nit :]  # [1, nit, vocab_size]
                hidden_i = hidden_states[i : i + 1, max_input - nit :]  # [1, nit, hidden_size]

                # SIL padding: condensed SIL has 1 logit position; pad to 2 so
                # _update_duplex_sequences_and_generate_audio_codes can index [-2:-1].
                if nit < 2:
                    logits_i = torch.cat([logits_i, torch.zeros_like(logits_i)], dim=1)
                    hidden_i = torch.cat([hidden_i, torch.zeros_like(hidden_i)], dim=1)

                _, seqs, attn, codes, codes_mask, emitted, deferred_hidden, new_machine_state = (
                    self._update_duplex_sequences_and_generate_audio_codes(
                        new_logits=logits_i,
                        new_last_hidden_state=hidden_i,
                        sequences=states[i].sequences,
                        attention_mask=states[i].attention_mask,
                        audio_codes=states[i].audio_codes,
                        audio_codes_mask=states[i].audio_codes_mask,
                        do_sample=do_sample,
                        logits_processor=states[i].logits_processor,
                        eos_penalty=states[i].eos_penalty,
                        defer_audio_code_generation=True,
                        machine_state=states[i].machine_state,
                    )
                )

                sample_updated.append((seqs, attn, codes, codes_mask, new_machine_state))
                sample_emitted.append(emitted)
                sample_deferred_hidden.append(deferred_hidden)

            # 5. Batched audio code generation.
            dummy_hidden = torch.zeros(1, 1, hidden_states.shape[-1], device=device, dtype=hidden_states.dtype)
            batch_hidden_list: list[torch.Tensor] = []
            for i in range(n):
                if sample_deferred_hidden[i] is not None:
                    batch_hidden_list.append(sample_deferred_hidden[i])
                else:
                    batch_hidden_list.append(dummy_hidden)

            batched_hidden = torch.cat(batch_hidden_list, dim=0)  # [n, 1, hidden_size]
            all_audio_codes_batch = self.generate_audio_codes(
                talker_last_hidden_state=batched_hidden,
                allow_audio_end=False,
            )  # [n, num_code_groups]

            # 6. Per-sample: scatter audio codes, render audio, update state.
            for i in range(n):
                seqs, attn, codes, codes_mask, new_machine_state = sample_updated[i]
                emitted = sample_emitted[i]

                # Append audio codes for samples that emitted audio.
                if emitted:
                    new_codes = all_audio_codes_batch[i : i + 1].clone()  # [1, num_code_groups]
                    audio_end_mask = new_codes[:, 0] == self.codebook_size
                    if audio_end_mask.any():
                        new_codes[audio_end_mask, 0] = 0
                    codes = torch.cat((codes, new_codes[None]), dim=1)
                    codes_mask = torch.cat(
                        (codes_mask, torch.tensor([[True]], device=device, dtype=torch.bool)),
                        dim=1,
                    )

                # Reset audio decoder conv state at speech onset (SIL → AUDIO_START).
                # In condensed silence mode, the decoder received no codes during SIL
                # frames, so its conv caches are stale.
                if (
                    use_condensed_silence
                    and states[i].machine_state is not None
                    and new_machine_state is not None
                    and states[i].machine_state.phase == RaonPhase.SIL
                    and new_machine_state.phase == RaonPhase.SPEECH
                ):
                    self.reset_audio_decoder_stream(states[i].audio_decoder_stream_id)

                # Audio rendering.
                # When no audio is emitted (silence step), clear the semantic buffer so
                # the post-silence frame is treated as a fresh "first frame" rather than
                # combining stale semantic state with fresh acoustic codes.
                new_semantic_buffer = None
                if not emitted:
                    decoded_audio = torch.zeros(1, samples_per_frame, device=device)
                elif self.max_delay > 0:
                    current_codes = codes[0, -1]  # [num_code_groups]
                    semantic_code = current_codes[0:1]
                    acoustic_codes = current_codes[1:]

                    if states[i].semantic_buffer is None:
                        # First frame: buffer semantic, output placeholder.
                        new_semantic_buffer = semantic_code
                        output_codes = torch.zeros_like(current_codes)
                        output_codes[0] = semantic_code[0]
                    else:
                        # Combine previous semantic + current acoustic.
                        output_codes = torch.cat([states[i].semantic_buffer, acoustic_codes], dim=0)
                        new_semantic_buffer = semantic_code

                    self.push_audio_codes(audio_codes=output_codes, stream_id=states[i].audio_decoder_stream_id)
                else:
                    # No delay: push codes directly.
                    self.push_audio_codes(audio_codes=codes[0, -1], stream_id=states[i].audio_decoder_stream_id)

                if emitted:
                    decoded_audio = self.pull_audio(states[i].audio_decoder_stream_id)

                # State update.
                states[i].sequences = seqs
                states[i].attention_mask = attn
                states[i].audio_codes = codes
                states[i].audio_codes_mask = codes_mask
                states[i].semantic_buffer = new_semantic_buffer
                states[i].emitted_audio = emitted
                states[i].machine_state = new_machine_state
                states[i].audio_output_frames.append(decoded_audio)

        elapsed = time.perf_counter() - t_start
        logger.info(
            f"Batch({n}) duplex generation: {total_frames} frames in {elapsed:.1f}s ({total_frames / elapsed:.1f} frames/s)"
        )

        # --- Collect results ---
        results: list[RaonGenerateResult] = []
        for i in range(n):
            audio_output = (
                torch.cat([f.to(device) for f in states[i].audio_output_frames], dim=1)
                if states[i].audio_output_frames
                else torch.zeros(1, 0, device=device)
            )

            predicted_text = None
            if self.tokenizer is not None:
                ignored_ids: set[int] = set()
                for tok in (
                    IM_START,
                    AUDIO_START,
                    AUDIO_INPUT_PLACEHOLDER,
                    AUDIO_OUTPUT_PLACEHOLDER,
                    AUDIO_OUTPUT_PAD,
                    AUDIO_OUTPUT_END_PAD,
                    AUDIO_END,
                ):
                    ignored_ids.add(tok.id)
                if self.use_backchannel_token:
                    ignored_ids.add(self.duplex_bc_token_id)
                sil_token_id = self.duplex_sil_token_id if self.use_sil_token else None
                predicted_text = extract_predicted_text(
                    sequences=states[i].sequences,
                    tokenizer=self.tokenizer,
                    text_vocab_size=self.vocab_size,
                    initial_seq_len=initial_seq_len,
                    sil_token_id=sil_token_id,
                    audio_start_token_id=AUDIO_START.id,
                    ignored_token_ids=ignored_ids,
                )

            results.append(
                RaonGenerateResult(
                    audio_output=audio_output,
                    predicted_text=predicted_text,
                    sequences=states[i].sequences.clone(),
                )
            )

            # Cleanup per-sample decoder stream.
            self._drain_audio_decoding_queue(states[i].audio_decoder_stream_id)
            self._destroy_audio_decoder_stream(states[i].audio_decoder_stream_id)

        # Free shared KV cache.
        self.free_past_key_values(metadata)

        return results


@dataclass
class _BatchSampleState:
    """Per-sample state for batched duplex generation.

    Each of the n samples maintains its own sequence, audio codes, and
    decoder stream. The heavy KV cache is shared via a single
    SGLangDecodingMetadata with batch_size=n.
    """

    sequences: torch.Tensor
    """Token IDs for this sample. Shape: [1, seq_length]. Dtype: long."""

    attention_mask: torch.Tensor
    """Mask for valid positions. Shape: [1, seq_length]. Dtype: long."""

    audio_codes: torch.Tensor
    """Generated audio codes. Shape: [1, num_frames, num_code_groups]. Dtype: long."""

    audio_codes_mask: torch.Tensor
    """Mask for valid audio code frames. Shape: [1, num_frames]. Dtype: bool."""

    semantic_buffer: torch.Tensor | None
    """Buffered semantic code for acoustic delay alignment."""

    audio_decoder_stream_id: int
    """Concurrent audio decoder stream ID for this sample."""

    logits_processor: Any
    """Per-sample logits processor (shared across all samples)."""

    eos_penalty: float
    """Penalty subtracted from pad/eos logit."""

    machine_state: RaonMachineState | None
    """Explicit duplex machine state for this sample's current trailing frame."""

    emitted_audio: bool
    """Whether the last frame produced audio codes."""

    audio_output_frames: list[torch.Tensor] = field(default_factory=list)
    """Accumulated decoded audio frames. Each element shape: [1, samples_per_frame]."""


__all__ = [
    "SGLangRaonModel",
]
