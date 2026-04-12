"""Tokenization, audio loading, message parsing, and batched collation for duplex model inputs."""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

import soundfile as sf
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2TokenizerFast

from .model import RaonConfig
from .special_tokens import (
    AUDIO_END,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_BC,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_OUTPUT_SIL,
    AUDIO_PLACEHOLDER,
    AUDIO_START,
    IM_END,
    IM_START,
    LOSS_IGNORE_INDEX,
    PRETRAINING_AUDIO_TAG,
    SPEAKER_EMBEDDING_PLACEHOLDER,
    update_tokenizer,
)


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class AudioContent(TypedDict):
    """Content item with an audio file path."""

    type: Literal["audio"]
    audio: str


class SpeakerContentRequired(TypedDict):
    """Content item for speaker-conditioned TTS; requires type only."""

    type: Literal["speaker"]


class SpeakerContent(SpeakerContentRequired, total=False):
    """Content item for speaker-conditioned TTS; optionally includes audio path for embedding."""

    audio: str


ContentItem = TextContent | AudioContent | SpeakerContent


class MultiModalMessage(TypedDict):
    """Chat message with role and list of multimodal content items (text, audio, speaker)."""

    role: str
    content: list[ContentItem]


class TextMessage(TypedDict):
    """Chat message with role and plain text content."""

    role: str
    content: str


Message = TextMessage | MultiModalMessage

AudioPreprocessor = Callable[[torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor | None]]


class RaonInputs(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    audio_input: torch.Tensor | None
    audio_output: torch.Tensor | None
    speaker_encoder_audio: NotRequired[torch.Tensor | None]
    audio_input_lengths: torch.Tensor | None
    audio_output_lengths: torch.Tensor | None
    speaker_encoder_audio_lengths: NotRequired[torch.Tensor | None]
    labels: torch.Tensor


def collapse_audio_placeholder_tokens(text: str) -> str:
    """Collapse consecutive audio placeholder tokens into a single placeholder.

    Reduces runs of adjacent output or input placeholders to one each, leaving
    non-adjacent placeholders and other text unchanged.

    Args:
        text: Input string potentially containing repeated audio placeholders.

    Returns:
        String with consecutive duplicate placeholders collapsed.
    """
    text = re.sub(
        rf"({re.escape(str(AUDIO_OUTPUT_PLACEHOLDER))})+",
        str(AUDIO_OUTPUT_PLACEHOLDER),
        text,
    )
    text = re.sub(
        rf"({re.escape(str(AUDIO_INPUT_PLACEHOLDER))})+",
        str(AUDIO_INPUT_PLACEHOLDER),
        text,
    )
    return text


class RaonProcessor:
    """Tokenizer, audio loader, and collator for the deployed duplex runtime.

    Handles chat message parsing, audio loading/resampling, tokenization with
    assistant masks, and batched collation with optional device/dtype placement.
    """

    def __init__(
        self,
        model_name_or_path: str | PathLike | None = None,
        tokenizer_path: str | PathLike | None = None,
        config_path: str | PathLike | None = None,
        tokenizer: Qwen2TokenizerFast | None = None,
        config: RaonConfig | None = None,
        max_audio_seq_length: int = 192000,
    ) -> None:
        if tokenizer is not None and config is not None:
            self.tokenizer = tokenizer
            self.config = config
            update_tokenizer(self.tokenizer)
        else:
            if tokenizer_path is None:
                tokenizer_path = model_name_or_path

            if config_path is None:
                config_path = model_name_or_path

            assert tokenizer_path is not None and config_path is not None, (
                "`model_name_or_path` or `tokenizer_path` and `config_path` must be provided."
            )

            self.tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
            update_tokenizer(self.tokenizer)
            config_file = Path(config_path) / "config.json"
            self.config = RaonConfig(**json.loads(config_file.read_text(encoding="utf-8")))

        assert isinstance(self.tokenizer.pad_token, str), "Tokenizer pad_token must be a string."
        self.pad_token = self.tokenizer.pad_token
        (self.pad_token_id,) = self.tokenizer.encode(self.pad_token)

        assert self.config.audio_tokenizer_config.sampling_rate is not None, (
            "Config audio_tokenizer_config.sampling_rate must be set."
        )
        self.sampling_rate: int = self.config.audio_tokenizer_config.sampling_rate
        assert (frame_rate := self.config.audio_tokenizer_config._frame_rate) is not None, (  # type: ignore
            "Config audio_tokenizer_config frame_rate must be set."
        )
        self.frame_rate: float = frame_rate
        self.samples_per_frame = self.sampling_rate / self.frame_rate

        assert isinstance(eos_token_id := self.tokenizer.eos_token_id, int), "Tokenizer eos_token_id must be an integer."
        self.eos_token_id = eos_token_id

        # EPAD (Early Prediction with Acoustic Delay) configuration
        self.use_duplex_end_pad: bool = getattr(self.config, "use_duplex_end_pad", False)
        self.duplex_pad_token_id: int = getattr(self.config, "duplex_pad_token_id", AUDIO_OUTPUT_PAD.id)
        self.duplex_end_pad_token_id: int = getattr(self.config, "duplex_end_pad_token_id", AUDIO_OUTPUT_END_PAD.id)
        self.use_sil_token: bool = getattr(self.config, "use_sil_token", False)
        self.no_audio_in_sil: bool = getattr(self.config, "no_audio_in_sil", False)
        self.use_backchannel_token: bool = getattr(self.config, "use_backchannel_token", False)
        self.duplex_sil_token_id: int = int(getattr(self.config, "duplex_sil_token_id", AUDIO_OUTPUT_SIL.id))
        self.duplex_bc_token_id: int = int(getattr(self.config, "duplex_bc_token_id", AUDIO_OUTPUT_BC.id))
        self.sequence_mode: Literal["tua", "uta"] | None = getattr(self.config, "sequence_mode", None)
        if self.sequence_mode not in (None, "tua", "uta"):
            raise ValueError(f"Unsupported sequence_mode '{self.sequence_mode}'.")

        self.text_lookahead: int = int(getattr(self.config, "text_lookahead", 0))
        self.has_speaker_encoder: bool = getattr(self.config, "speaker_encoder_config", None) is not None
        self.max_audio_seq_length = int(max_audio_seq_length)
        assert self.max_audio_seq_length > 0, "max_audio_seq_length must be positive."

    def _parse_message_content(self, content: str | list[ContentItem], role: str) -> tuple[str, list[str], list[str]]:
        """Parse multimodal content items into text with audio tags and collected audio paths.

        Routes audio items to user or assistant path lists based on the message role.
        User, system, and tool roles produce input audio tags; assistant role produces
        output audio tags. Speaker items insert a speaker embedding placeholder.

        Args:
            content: Plain text string or list of typed content items (text, audio, speaker).
            role: Message role determining audio tag type and path routing.

        Returns:
            Tuple of (assembled text, user audio paths, assistant audio paths).
        """
        if isinstance(content, str):
            return content, [], []

        text_parts: list[str] = []
        user_audio_paths: list[str] = []
        assistant_audio_paths: list[str] = []

        if role in ("user", "system", "tool"):
            audio_tag = f"{AUDIO_START}{AUDIO_INPUT_PLACEHOLDER}{AUDIO_END}"
        else:
            audio_tag = f"{AUDIO_START}{AUDIO_OUTPUT_PLACEHOLDER}{AUDIO_END}"

        for item in content:
            if item["type"] == "text":
                text_parts.append(item["text"])
            elif item["type"] == "audio":
                audio_path = item["audio"]
                if audio_path:
                    if role in ("user", "system", "tool"):
                        user_audio_paths.append(audio_path)
                    else:
                        assistant_audio_paths.append(audio_path)
                    text_parts.append(audio_tag)
            elif item["type"] == "speaker":
                # Speaker token for speaker-conditioned TTS
                # The speaker embedding is resolved separately from runtime audio context.
                text_parts.append(str(SPEAKER_EMBEDDING_PLACEHOLDER))

        return "".join(text_parts), user_audio_paths, assistant_audio_paths

    def process_messages(self, messages: list[Message]) -> tuple[list[TextMessage], list[str], list[str]]:
        """Convert a list of chat messages into text messages with collected audio paths.

        Iterates over messages, parsing multimodal content into plain text with audio
        placeholder tags, and accumulating user and assistant audio file paths. Warns
        if string content contains the pretraining audio tag format.

        Args:
            messages: List of chat messages, each with a role and text or multimodal content.

        Returns:
            Tuple of (processed text messages, all user audio paths, all assistant audio paths).
        """
        processed_messages: list[TextMessage] = []
        all_user_audio_paths: list[str] = []
        all_assistant_audio_paths: list[str] = []

        for message in messages:
            # Support both HF format (role/content) and dataset format (from/value).
            role = message.get("role") or {"human": "user", "gpt": "assistant"}.get(message.get("from", ""), "user")
            content = message.get("content") if "content" in message else message.get("value", "")

            if isinstance(content, str):
                if PRETRAINING_AUDIO_TAG in content:
                    print(
                        f"WARNING RaonProcessor.process_messages: Message content contains '{PRETRAINING_AUDIO_TAG}' tag "
                        "which is the pretraining format. Use the message content list format with audio parts instead for "
                        "proper audio handling.",
                    )

                processed_messages.append({"role": role, "content": content})
            else:
                text_content, user_audio_paths, assistant_audio_paths = self._parse_message_content(content, role)
                all_user_audio_paths.extend(user_audio_paths)
                all_assistant_audio_paths.extend(assistant_audio_paths)
                processed_messages.append({"role": role, "content": text_content})

        return processed_messages, all_user_audio_paths, all_assistant_audio_paths

    def load_audio(self, audio_paths: list[str]) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Load audio files, resample to the model sampling rate, and pad into a batch.

        Supports .wav files via soundfile and all other formats via torchaudio.
        Multi-channel audio is downmixed to mono. Files with different sample rates
        are resampled to ``self.sampling_rate``.

        Args:
            audio_paths: List of file paths to load. If empty, returns None.

        Returns:
            Tuple of (padded_audio, audio_lengths) where padded_audio has shape
            [num_files, max_audio_len] (dtype: float) and audio_lengths has shape
            [num_files] (dtype: long), or None if audio_paths is empty.
        """
        if not audio_paths:
            return None

        resampled_audio: list[torch.Tensor] = []

        for audio_path in audio_paths:
            if audio_path.endswith(".wav"):
                # sf.read returns [samples, channels] so dim=-1 averages across channels.
                audio_np, prev_sampling_rate = sf.read(audio_path)
                audio = torch.from_numpy(audio_np).float()
                if audio.ndim != 1:
                    audio = audio.mean(dim=-1)
            else:
                # torchaudio.load returns [channels, samples] so dim=0 averages across channels.
                audio, prev_sampling_rate = torchaudio.load(audio_path)
                if audio.ndim != 1:
                    audio = audio.mean(dim=0)

            assert audio.ndim == 1, f"Expected 1D audio after mean but got {audio.ndim=}"

            if prev_sampling_rate != self.sampling_rate:
                audio = torchaudio.functional.resample(
                    audio,
                    orig_freq=int(prev_sampling_rate),
                    new_freq=self.sampling_rate,
                )

            resampled_audio.append(audio)

        audio_lengths = torch.tensor([audio.shape[0] for audio in resampled_audio], dtype=torch.long)
        padded_audio = pad_sequence(resampled_audio, batch_first=True, padding_side="right", padding_value=0)

        return padded_audio, audio_lengths

    def expand_audio_padding(self, text: str, audio_code_lengths: torch.Tensor, pad_token: str) -> str:
        """Expand single audio placeholder tokens to match the required code lengths.

        Each occurrence of pad_token in text is replaced with pad_token repeated
        ``audio_code_lengths[i]`` times, using a two-phase replacement via an
        intermediate placeholder to avoid collisions.

        Args:
            text: Input text containing one pad_token per audio segment.
            audio_code_lengths: Number of code frames for each audio segment.
                Shape: [num_segments]. Dtype: long.
            pad_token: The placeholder token string to expand.

        Returns:
            Text with each pad_token occurrence expanded to the corresponding length.
        """
        assert audio_code_lengths.dtype == torch.long and audio_code_lengths.ndim == 1, (
            "The audio_code_lengths tensor must be 1D with dtype long."
        )

        pattern = re.escape(pad_token)
        positions = [(match.start(), match.group()) for match in re.finditer(pattern, text)]
        positions.sort(key=lambda x: x[0])

        assert len(positions) == len(audio_code_lengths), (
            f"Audio pad token count ({len(positions)}) != audio code lengths ({len(audio_code_lengths)})"
        )

        for (_, match_text), length in zip(positions, audio_code_lengths, strict=True):
            assert match_text == pad_token, "Match text must equal pad_token."
            text = text.replace(match_text, AUDIO_PLACEHOLDER * int(length.item()), 1)

        return text.replace(AUDIO_PLACEHOLDER, pad_token)

    def decode(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        input_length: int | None = None,
        output_only: bool = False,
        collapse_audio_tokens: bool = False,
        skip_special_tokens: bool = False,
        **tokenizer_decode_kwargs: Any,
    ) -> str:
        """Decode token IDs to text, optionally restricted to output or assistant response.

        Args:
            token_ids: Token IDs to decode. Shape: [batch_size, seq_length] or [seq_length].
                Dtype: long. If 2D, uses first row.
            labels: Optional labels with LOSS_IGNORE_INDEX for non-output positions.
                Shape: [batch_size, seq_length] or [seq_length]. Dtype: long. Used when
                output_only=True to extract only assistant-predicted tokens.
            input_length: Optional length of input (prompt) in tokens. Used when
                output_only=True to slice tokens from input_length onward.
            output_only: If True, decode only assistant output tokens (requires
                labels or input_length).
            collapse_audio_tokens: If True, collapse consecutive audio placeholders.
            skip_special_tokens: If True, omit special tokens from decoded text.
            **tokenizer_decode_kwargs: Passed to underlying tokenizer decode.

        Returns:
            Decoded string.
        """
        if token_ids.ndim == 2:
            token_ids = token_ids[0]

        if labels is not None and labels.ndim == 2:
            labels = labels[0]

        if output_only:
            assert labels is not None or input_length is not None, (
                "decode: `labels` or `input_length` is required when `output_only=True`."
            )
            if labels is not None:
                token_ids = token_ids[labels != LOSS_IGNORE_INDEX]
            else:
                assert input_length is not None, "input_length must be provided when labels is None."
                token_ids = token_ids[input_length:]

        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **tokenizer_decode_kwargs,
        )

        if collapse_audio_tokens:
            text = collapse_audio_placeholder_tokens(text)

        return text

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize chat-formatted text and compute assistant response labels.

        Strips empty think tags, tokenizes the text, then builds an assistant mask
        by locating ``<|im_start|>assistant`` / ``<|im_end|>`` boundaries. Tokens
        inside assistant responses get their true token ID as the label; all other
        positions receive LOSS_IGNORE_INDEX.

        Args:
            text: Chat-template-formatted text with ``<|im_start|>`` / ``<|im_end|>`` markers.

        Returns:
            Tuple of (input_ids, attention_mask, labels), each with shape [1, seq_length]
            and dtype long. attention_mask is all ones.
        """
        text = text.replace("<think></think>\n", "").replace("<think></think>", "")

        input_ids: list[int] = self.tokenizer.encode(text)
        attention_mask = [1] * len(input_ids)

        begin_assistant_ids = self.tokenizer.encode(f"{IM_START}assistant\n")
        end_turn_ids = self.tokenizer.encode(str(IM_END))

        begin_assistant_indices = [
            i
            for i in range(len(input_ids) - len(begin_assistant_ids) + 1)
            if input_ids[i : i + len(begin_assistant_ids)] == begin_assistant_ids
        ]
        end_turn_indices = [
            i for i in range(len(input_ids) - len(end_turn_ids) + 1) if input_ids[i : i + len(end_turn_ids)] == end_turn_ids
        ]

        assistant_masks: list[int] = []
        for begin_assistant_idx in begin_assistant_indices:
            begin_response_idx = begin_assistant_idx + len(begin_assistant_ids)
            if len(assistant_masks) > begin_response_idx:
                continue

            valid_end_turn_indices = [idx for idx in end_turn_indices if idx > begin_response_idx]
            if valid_end_turn_indices:
                end_response_idx = min(valid_end_turn_indices) + 1
                assert len(assistant_masks) <= begin_response_idx, (
                    f"Tokenize: Masks length exceeds begin response index. "
                    f"Got `{len(assistant_masks)=}` and `{begin_response_idx=}`."
                )
                assistant_masks.extend([0] * (begin_response_idx - len(assistant_masks)))
                assert len(assistant_masks) == begin_response_idx, (
                    f"Tokenize: Masks length does not match begin response index. "
                    f"Got `{len(assistant_masks)=}` and `{begin_response_idx=}`."
                )
                assert len(assistant_masks) <= end_response_idx, (
                    f"Tokenize: Masks length exceeds end response index. "
                    f"Got `{len(assistant_masks)=}` and `{end_response_idx=}`."
                )
                assistant_masks.extend([1] * (end_response_idx - len(assistant_masks)))

        assistant_masks.extend([0] * (len(input_ids) - len(assistant_masks)))
        assert len(assistant_masks) == len(input_ids), (
            f"Tokenize: Masks length does not match input_ids length. Got `{len(assistant_masks)=}` and `{len(input_ids)=}`."
        )

        labels = [input_ids[i] if assistant_masks[i] == 1 else LOSS_IGNORE_INDEX for i in range(len(input_ids))]

        return (
            torch.tensor([input_ids]),
            torch.tensor([attention_mask]),
            torch.tensor([labels]),
        )

    def process_single(
        self,
        messages: list[Message],
        add_generation_prompt: bool,
        audio_preprocessor: AudioPreprocessor | None,
        max_audio_chunk_length: int | None = None,
    ) -> RaonInputs:
        """Process a single conversation into RaonInputs.

        Parses messages, applies the chat template, loads and resamples audio,
        expands audio placeholders to match codec frame counts, and tokenizes
        with assistant response labels.

        Args:
            messages: List of chat messages forming one conversation.
            add_generation_prompt: If True, append generation prompt for inference.
            audio_preprocessor: Optional callback to preprocess audio tensors
                before placeholder expansion.
            max_audio_chunk_length: If set, split audio_input into chunks of at
                most this many samples. None
                disables chunking.

        Returns:
            RaonInputs with input_ids, attention_mask, labels (each shape
            [1, seq_length]), and optional audio_input/audio_output tensors.
        """
        processed_messages, user_audio_paths, assistant_audio_paths = self.process_messages(messages)

        text = self.tokenizer.apply_chat_template(
            cast(list[dict[str, str]], processed_messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(text, str), "Chat template must return a string."

        # Load and process user audio (audio_input)
        audio_input_data = self.load_audio(user_audio_paths)
        audio_input: torch.Tensor | None = None
        audio_input_lengths: torch.Tensor | None = None

        if audio_input_data is not None:
            audio_input, audio_input_lengths = audio_input_data
            if audio_preprocessor is not None:
                audio_input, audio_input_lengths = audio_preprocessor(audio_input, audio_input_lengths)

            if audio_input_lengths is None:
                audio_input_lengths = torch.full(
                    (audio_input.shape[0],), fill_value=audio_input.shape[1], device=audio_input.device
                )

            audio_input_code_lengths = (audio_input_lengths.float() / self.samples_per_frame).ceil().long()
            text = self.expand_audio_padding(text, audio_input_code_lengths, str(AUDIO_INPUT_PLACEHOLDER))

            # Chunk audio input when requested to cap per-segment latency/memory.
            if max_audio_chunk_length is not None:
                audio_input, audio_input_lengths = self._chunk_audio(
                    audio_input, audio_input_lengths, max_audio_chunk_length
                )

        # Load and process assistant audio (audio_output)
        audio_output_data = self.load_audio(assistant_audio_paths)
        audio_output: torch.Tensor | None = None
        audio_output_lengths: torch.Tensor | None = None

        if audio_output_data is not None:
            audio_output, audio_output_lengths = audio_output_data
            if audio_preprocessor is not None:
                audio_output, audio_output_lengths = audio_preprocessor(audio_output, audio_output_lengths)

            if audio_output_lengths is None:
                audio_output_lengths = torch.full(
                    (audio_output.shape[0],), fill_value=audio_output.shape[1], device=audio_output.device
                )

            audio_output_code_lengths = (audio_output_lengths.float() / self.samples_per_frame).ceil().long()
            text = self.expand_audio_padding(text, audio_output_code_lengths, str(AUDIO_OUTPUT_PLACEHOLDER))

        input_ids, attention_mask, labels = self._tokenize(text)
        has_speaker_placeholder = bool((input_ids == SPEAKER_EMBEDDING_PLACEHOLDER.id).any().item())
        if self.has_speaker_encoder and has_speaker_placeholder:
            speaker_encoder_audio, speaker_encoder_audio_lengths = self._prepare_speaker_encoder_audio(
                audio_output=audio_output,
                audio_output_lengths=audio_output_lengths,
            )
        else:
            speaker_encoder_audio, speaker_encoder_audio_lengths = None, None

        return RaonInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_input=audio_input,
            audio_output=audio_output,
            speaker_encoder_audio=speaker_encoder_audio,
            audio_input_lengths=audio_input_lengths,
            audio_output_lengths=audio_output_lengths,
            speaker_encoder_audio_lengths=speaker_encoder_audio_lengths,
            labels=labels,
        )

    @staticmethod
    def _left_pad(tensors: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        """Left-pad and stack 2D tensors into a single batch tensor.

        Args:
            tensors: List of 2D tensors, each with shape [1, seq_length_i].
            padding_value: Value used for left-padding shorter sequences.

        Returns:
            Batched tensor with shape [num_tensors, max_seq_length]. Dtype: same as input.
        """
        rows = [row for tensor in tensors for row in tensor]
        return pad_sequence(rows, batch_first=True, padding_value=padding_value, padding_side="left")

    @staticmethod
    def _optional_cat(optional_tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
        """Concatenate non-None tensors along dim 0, returning None if all are None.

        Args:
            optional_tensors: List of tensors or None values.

        Returns:
            Concatenated tensor or None if every element is None.
        """
        tensors = [tensor for tensor in optional_tensors if tensor is not None]
        if len(tensors) == 0:
            return None
        return torch.cat(tensors)

    @staticmethod
    def _optional_left_pad(optional_tensors: list[torch.Tensor | None], padding_value: int) -> torch.Tensor | None:
        """Left-pad and stack non-None 2D tensors, returning None if all are None.

        Args:
            optional_tensors: List of 2D tensors or None values.
            padding_value: Value used for left-padding shorter sequences.

        Returns:
            Batched tensor with shape [total_rows, max_seq_length] or None if all inputs are None.
        """
        tensors = [row for tensor in optional_tensors if tensor is not None for row in tensor]
        if len(tensors) == 0:
            return None
        return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="left")

    @staticmethod
    def _chunk_audio(
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        max_audio_seq_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split padded audio waveforms into chunks of at most ``max_audio_seq_length`` samples.

        Keeps runtime audio tensors within the same bounded chunk shape used by
        the deployed service pipeline.

        Args:
            audio: Padded audio waveforms.
                Shape: [num_files, max_audio_len]. Dtype: float.
            audio_lengths: True length of each waveform in samples.
                Shape: [num_files]. Dtype: long.
            max_audio_seq_length: Maximum number of samples per chunk.

        Returns:
            Tuple of (chunked_audio, chunked_lengths) where chunked_audio has
            shape [num_chunks, max_chunk_len] (dtype: float, right-padded) and
            chunked_lengths has shape [num_chunks] (dtype: long).
        """
        chunks: list[torch.Tensor] = []
        chunk_lengths: list[int] = []

        for waveform, length in zip(audio, audio_lengths, strict=True):
            seq = waveform[: int(length.item())]
            while len(seq) > max_audio_seq_length:
                chunks.append(seq[:max_audio_seq_length])
                chunk_lengths.append(max_audio_seq_length)
                seq = seq[max_audio_seq_length:]
            if len(seq) > 0:
                chunks.append(seq)
                chunk_lengths.append(len(seq))

        padded = pad_sequence(chunks, batch_first=True, padding_side="right", padding_value=0)
        return padded, torch.tensor(chunk_lengths, dtype=torch.long)

    def _prepare_speaker_encoder_audio(
        self,
        audio_output: torch.Tensor | None,
        audio_output_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Prepare fixed-width speaker audio from a random segment of each file.

        For each audio file, selects a random contiguous window of up to
        ``max_audio_seq_length`` samples from the valid region. This avoids
        always using the first chunk, reducing content leakage through the
        speaker encoder (which otherwise always sees the same audio the model
        generates first).

        Args:
            audio_output: Raw assistant audio. Shape: [num_segments, num_samples]. Dtype: float.
            audio_output_lengths: Valid lengths. Shape: [num_segments]. Dtype: long.

        Returns:
            Tuple of:
                - speaker_encoder_audio: Shape [num_segments, max_audio_seq_length]. Dtype: float.
                - speaker_encoder_audio_lengths: Shape [num_segments]. Dtype: long.
            Returns (None, None) if audio_output or lengths are missing.
        """
        if audio_output is None or audio_output_lengths is None:
            return None, None

        max_len = self.max_audio_seq_length
        out_lengths: list[int] = []
        segments: list[torch.Tensor] = []

        for idx, valid_length in enumerate(audio_output_lengths.tolist()):
            valid_length = int(min(valid_length, audio_output.shape[1]))
            if valid_length <= 0:
                out_lengths.append(0)
                segments.append(audio_output[idx, :0])
                continue

            if valid_length > max_len:
                # Random start within the valid region.
                start = random.randint(0, valid_length - max_len)
                segments.append(audio_output[idx, start : start + max_len])
                out_lengths.append(max_len)
            else:
                segments.append(audio_output[idx, :valid_length])
                out_lengths.append(valid_length)

        # Pad to the longest segment in this batch, not to max_audio_seq_length.
        batch_max_len = max(out_lengths) if out_lengths else 0
        fixed_audio = torch.zeros(
            len(segments),
            batch_max_len,
            device=audio_output.device,
            dtype=audio_output.dtype,
        )
        for idx, (segment, length) in enumerate(zip(segments, out_lengths, strict=True)):
            if length > 0:
                fixed_audio[idx, :length] = segment

        return fixed_audio, torch.tensor(out_lengths, dtype=torch.long, device=audio_output.device)

    def _collate(self, batch: list[RaonInputs]) -> RaonInputs:
        """Collate a list of single-sample RaonInputs into a batched RaonInputs.

        Left-pads input_ids, attention_mask, and labels to the longest sequence.
        Audio tensors and their lengths are concatenated or left-padded as appropriate.

        Args:
            batch: List of RaonInputs, each from a single conversation.

        Returns:
            Batched RaonInputs with consistent sequence lengths across the batch.
        """
        return RaonInputs(
            input_ids=self._left_pad([item["input_ids"] for item in batch], padding_value=self.pad_token_id),
            attention_mask=self._left_pad([item["attention_mask"] for item in batch], padding_value=0),
            labels=self._left_pad([item["labels"] for item in batch], padding_value=LOSS_IGNORE_INDEX),
            audio_input=self._optional_left_pad([item["audio_input"] for item in batch], padding_value=0),
            audio_output=self._optional_left_pad([item["audio_output"] for item in batch], padding_value=0),
            speaker_encoder_audio=self._optional_cat([item.get("speaker_encoder_audio") for item in batch]),
            audio_input_lengths=self._optional_cat([item["audio_input_lengths"] for item in batch]),
            audio_output_lengths=self._optional_cat([item["audio_output_lengths"] for item in batch]),
            speaker_encoder_audio_lengths=self._optional_cat([item.get("speaker_encoder_audio_lengths") for item in batch]),
        )

    def __call__(
        self,
        messages: list[Message] | list[list[Message]],
        add_generation_prompt: bool = False,
        audio_preprocessor: AudioPreprocessor | None = None,
        force_audio_output: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        max_audio_chunk_length: int | None = None,
    ) -> RaonInputs:
        """Process messages into RaonInputs for the deployed runtime.

        Accepts a single conversation or a batch of conversations. Loads and
        resamples audio, expands placeholders to match code lengths, tokenizes
        with assistant masks, and collates when batched.

        Args:
            messages: Single conversation (list of Message) or batch (list of
                list of Message). Supports TextMessage and MultiModalMessage.
            add_generation_prompt: If True, append generation prompt for inference.
            audio_preprocessor: Optional callback (audio, lengths) -> (audio, lengths)
                to preprocess audio before expansion.
            force_audio_output: If True, append an ``<|audio_start|>`` token to
                input_ids and a corresponding 1 to attention_mask, prompting
                the model to begin generating audio output.
            device: Optional device to move tensors to (input_ids, attention_mask,
                labels, audio_input, audio_output, length tensors).
            dtype: Optional dtype to cast audio tensors to.
            max_audio_chunk_length: If set, split audio_input into chunks of at
                most this many samples. None
                disables chunking.

        Returns:
            RaonInputs with input_ids, attention_mask, labels, optional
            audio_input/audio_output and their length tensors. Shapes follow
            batch_size and seq_length; audio tensors are [num_chunks, max_chunk_len]
            when max_audio_seq_length is set, otherwise [batch_size, audio_len].
        """
        if len(messages) > 0 and isinstance(messages[0], list):
            batched_messages = cast(list[list[Message]], messages)
            batch = [
                self.process_single(
                    conversation,
                    add_generation_prompt=add_generation_prompt,
                    audio_preprocessor=audio_preprocessor,
                    max_audio_chunk_length=max_audio_chunk_length,
                )
                for conversation in batched_messages
            ]
            result = self._collate(batch)
        else:
            conversation = cast(list[Message], messages)
            result = self.process_single(
                conversation,
                add_generation_prompt=add_generation_prompt,
                audio_preprocessor=audio_preprocessor,
                max_audio_chunk_length=max_audio_chunk_length,
            )

        if force_audio_output:
            # Strip trailing <|audio_end|> and <|im_end|> tokens so the
            # appended <|audio_start|> directly follows the audio content.
            trailing_ids = {AUDIO_END.id, int(self.tokenizer.encode(str(IM_END))[0])}
            ids = result["input_ids"]
            while ids.shape[1] > 0 and int(ids[0, -1].item()) in trailing_ids:
                ids = ids[:, :-1]
                result["attention_mask"] = result["attention_mask"][:, :-1]
            result["input_ids"] = ids

            audio_start_id = torch.tensor(
                [[AUDIO_START.id]], dtype=result["input_ids"].dtype, device=result["input_ids"].device
            )
            result["input_ids"] = torch.cat([result["input_ids"], audio_start_id], dim=1)
            result["attention_mask"] = torch.cat(
                [
                    result["attention_mask"],
                    torch.ones(1, 1, dtype=result["attention_mask"].dtype, device=result["attention_mask"].device),
                ],
                dim=1,
            )

        if device is not None:
            result["input_ids"] = result["input_ids"].to(device)
            result["attention_mask"] = result["attention_mask"].to(device)
            result["labels"] = result["labels"].to(device)
            if result["audio_output"] is not None:
                result["audio_output"] = result["audio_output"].to(device)
            if result["audio_output_lengths"] is not None:
                result["audio_output_lengths"] = result["audio_output_lengths"].to(device)
            if result["audio_input"] is not None:
                result["audio_input"] = result["audio_input"].to(device)
            if result["audio_input_lengths"] is not None:
                result["audio_input_lengths"] = result["audio_input_lengths"].to(device)
            speaker_audio = result.get("speaker_encoder_audio")
            speaker_audio_lengths = result.get("speaker_encoder_audio_lengths")
            if speaker_audio is not None:
                result["speaker_encoder_audio"] = speaker_audio.to(device)
            if speaker_audio_lengths is not None:
                result["speaker_encoder_audio_lengths"] = speaker_audio_lengths.to(device)

        if dtype is not None:
            if result["audio_output"] is not None:
                result["audio_output"] = result["audio_output"].to(dtype)
            if result["audio_input"] is not None:
                result["audio_input"] = result["audio_input"].to(dtype)
            speaker_audio = result.get("speaker_encoder_audio")
            if speaker_audio is not None:
                result["speaker_encoder_audio"] = speaker_audio.to(dtype)

        return result


DuplexProcessor = RaonProcessor

__all__ = [
    "DuplexProcessor",
    "Message",
    "RaonProcessor",
]
