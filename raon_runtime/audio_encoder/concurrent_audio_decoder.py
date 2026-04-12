import logging
import queue
from collections import defaultdict
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
import torch.multiprocessing as mp
from transformers import Cache

from worker.session_leak_logging import log_session_leak_detail

from .streaming_mimi import (
    MimiConv1dPaddingCache,
    MimiConvTranspose1dPaddingCache,
    StreamingMimiDecoderOutput,
    StreamingMimiModel,
)

StreamDecoderState: TypeAlias = tuple[Cache | None, MimiConv1dPaddingCache | None, MimiConvTranspose1dPaddingCache | None]
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..inference import RaonInferenceModel


@torch.inference_mode()
def audio_decoder_worker(
    audio_tokenizer: StreamingMimiModel,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    device: torch.device | str,
    dtype: torch.dtype,
) -> None:
    try:
        if isinstance(device, str) and device.startswith("cuda"):
            device_id = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(device_id)
        elif isinstance(device, torch.device) and device.type == "cuda":
            torch.cuda.set_device(device.index or 0)

        audio_tokenizer = audio_tokenizer.to(device=device, dtype=dtype)  # type: ignore
        audio_tokenizer.eval()
        output_queue.put(("WORKER_READY", None))

    except Exception as e:
        output_queue.put(("WORKER_ERROR", e))
        return

    stream_states: dict[int, StreamDecoderState] = {}

    while True:
        try:
            item = input_queue.get()
            if item is None:
                break

            command, stream_id, sequence_id, audio_codes = item

            match command:
                case "CREATE_STREAM":
                    stream_states[stream_id] = (None, None, None)
                case "DESTROY_STREAM":
                    if stream_id in stream_states:
                        del stream_states[stream_id]

                case "RESET_STREAM":
                    if stream_id in stream_states:
                        stream_states[stream_id] = (None, None, None)

                case "DECODE_AUDIO":
                    decoder_past_key_values, conv_padding_cache, conv_transpose_padding_cache = stream_states[stream_id]
                    audio_codes = audio_codes.to(device=device)

                    outputs = audio_tokenizer.decode(
                        audio_codes.transpose(1, 2),
                        decoder_past_key_values=decoder_past_key_values,
                        conv1d_padding_cache=conv_padding_cache,
                        convtranspose1d_padding_cache=conv_transpose_padding_cache,
                        use_streaming=True,
                        return_dict=True,
                    )
                    assert isinstance(outputs, StreamingMimiDecoderOutput)
                    assert (audio_values := outputs.audio_values) is not None
                    assert isinstance(outputs.decoder_past_key_values, Cache)
                    assert isinstance(outputs.conv1d_padding_cache, MimiConv1dPaddingCache)
                    assert isinstance(outputs.convtranspose1d_padding_cache, MimiConvTranspose1dPaddingCache)

                    audio = audio_values.view(audio_values.shape[0], audio_values.shape[2])

                    stream_states[stream_id] = (
                        outputs.decoder_past_key_values,
                        outputs.conv1d_padding_cache,
                        outputs.convtranspose1d_padding_cache,
                    )
                    output_queue.put(("DECODE_AUDIO", stream_id, sequence_id, audio.cpu()))

                case _:
                    ...

        except Exception as e:
            output_queue.put((None, None, None, e))


class ConcurrentAudioDecoder:
    def __init__(
        self,
        model: "RaonInferenceModel",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.device = device if device is not None else model.get_model().device
        self.dtype = dtype if dtype is not None else model.get_model().dtype

        self.mp_context = mp.get_context("spawn")
        self.input_queue: mp.Queue = self.mp_context.Queue()
        self.output_queue: mp.Queue = self.mp_context.Queue()
        self.process: mp.process.BaseProcess | None = None  # type: ignore
        self.stream_counter = 0
        self.sequence_counter = 0
        self.pending_sequences: dict[int, int] = {}
        self.stream_pending_counts: dict[int, int] = defaultdict(int)
        self.stream_output_queues: dict[int, queue.Queue[tuple[int, torch.Tensor]]] = defaultdict(queue.Queue)

    def _stream_log_fields(self, stream_id: int | None = None) -> dict[str, object]:
        fields: dict[str, object] = {
            "stream_id": stream_id,
            "active_streams": len(self.stream_pending_counts),
            "total_pending": self.pending_count,
        }
        if stream_id is not None:
            fields["stream_pending"] = self.get_stream_pending_count(stream_id)
            output_queue = self.stream_output_queues.get(stream_id)
            fields["output_queue_depth"] = output_queue.qsize() if output_queue is not None else None
        return fields

    def start(self, timeout: float = 5.0) -> None:
        if self.process is not None and self.process.is_alive():
            raise RuntimeError("Audio decoder worker is already running.")

        process = self.mp_context.Process(
            target=audio_decoder_worker,
            kwargs={
                "audio_tokenizer": self.model.get_model().audio_tokenizer,
                "input_queue": self.input_queue,
                "output_queue": self.output_queue,
                "device": self.device,
                "dtype": self.dtype,
            },
            daemon=True,
        )
        process.start()
        self.process = process

        try:
            signal, error = self.output_queue.get(timeout=timeout)
            if signal == "WORKER_ERROR":
                process.join(timeout=1.0)
                self.process = None
                raise RuntimeError(f"Audio decoder worker failed to initialize: {error}") from None
            elif signal != "WORKER_READY":
                self.output_queue.put((signal, error))

        except queue.Empty:
            self.process = None
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

            raise RuntimeError("Audio decoder worker failed to start (timeout waiting for ready signal)") from None

        log_session_leak_detail(logger, "audio_decoder_worker_started", **self._stream_log_fields())

    def stop(self, timeout: float | None = 5.0) -> None:
        if self.process is None:
            return

        stop_fields = self._stream_log_fields()
        self.input_queue.put(None)

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

        self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)

        self.process = None

        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except Exception:
                break

        self.pending_sequences.clear()
        self.stream_pending_counts.clear()
        self.stream_output_queues.clear()
        self.stream_counter = 0
        self.sequence_counter = 0
        log_session_leak_detail(logger, "audio_decoder_worker_stopped", **stop_fields)

    def create_stream(self) -> int:
        assert self.process is not None and self.process.is_alive()
        stream_id = self.stream_counter
        self.stream_counter += 1
        self.stream_pending_counts[stream_id] = 0
        self.input_queue.put(("CREATE_STREAM", stream_id, None, None))
        log_session_leak_detail(logger, "audio_decoder_stream_created", **self._stream_log_fields(stream_id))
        return stream_id

    def reset_stream(self, stream_id: int) -> None:
        """Reset decoder conv state for a stream. Call at speech onset to clear
        artifacts from init placeholder codes or previous utterance."""
        if self.process is None or not self.process.is_alive():
            return
        self.input_queue.put(("RESET_STREAM", stream_id, None, None))
        log_session_leak_detail(logger, "audio_decoder_stream_reset", **self._stream_log_fields(stream_id))

    def destroy_stream(self, stream_id: int) -> None:
        if self.process is None or not self.process.is_alive():
            return

        before_fields = self._stream_log_fields(stream_id)
        if stream_id in self.stream_pending_counts:
            del self.stream_pending_counts[stream_id]

        if stream_id in self.stream_output_queues:
            del self.stream_output_queues[stream_id]

        self.input_queue.put(("DESTROY_STREAM", stream_id, None, None))
        log_session_leak_detail(
            logger,
            "audio_decoder_stream_destroyed",
            **before_fields,
            active_streams_after=len(self.stream_pending_counts),
            total_pending_after=self.pending_count,
        )

    def push_audio_codes(self, stream_id: int, audio_codes: torch.Tensor) -> int:
        assert self.process is not None and self.process.is_alive()
        assert audio_codes.ndim == 3

        sequence_id = self.sequence_counter
        self.sequence_counter += 1
        self.pending_sequences[sequence_id] = stream_id
        self.stream_pending_counts[stream_id] += 1

        self.input_queue.put(("DECODE_AUDIO", stream_id, sequence_id, audio_codes.cpu()))
        return sequence_id

    def pull_audio(
        self,
        stream_id: int,
        block: bool = True,
        timeout: float | None = None,
    ) -> tuple[int, torch.Tensor] | None:
        try:
            while self.stream_output_queues[stream_id].empty():
                command, result_stream_id, sequence_id, audio_or_error = self.output_queue.get(block=block, timeout=timeout)
                if command == "DECODE_AUDIO":
                    assert isinstance(audio := audio_or_error, torch.Tensor)
                    self.stream_output_queues[result_stream_id].put((sequence_id, audio))
                    if sequence_id in self.pending_sequences:
                        del self.pending_sequences[sequence_id]

                elif isinstance(audio_or_error, Exception):
                    raise RuntimeError(f"Audio decoder worker error: {audio_or_error}")

            result = self.stream_output_queues[stream_id].get(block=block, timeout=timeout)
            if stream_id in self.stream_pending_counts:
                self.stream_pending_counts[stream_id] -= 1
            return result
        except queue.Empty:
            return None

    @property
    def pending_count(self) -> int:
        return len(self.pending_sequences)

    def get_stream_pending_count(self, stream_id: int) -> int:
        return self.stream_pending_counts.get(stream_id, 0)

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def drain_to(
        self,
        max_pending: int,
        stream_id: int,
        timeout_per_item: float = 1.0,
    ) -> list[tuple[int, torch.Tensor]]:
        results: list[tuple[int, torch.Tensor]] = []
        log_session_leak_detail(
            logger,
            "audio_decoder_stream_drain_start",
            **self._stream_log_fields(stream_id),
            max_pending=max_pending,
        )

        while self.get_stream_pending_count(stream_id) > max_pending:
            result = self.pull_audio(stream_id=stream_id, block=True, timeout=timeout_per_item)
            if result is not None:
                results.append(result)
            else:
                break

        log_session_leak_detail(
            logger,
            "audio_decoder_stream_drain_done",
            **self._stream_log_fields(stream_id),
            drained=len(results),
            max_pending=max_pending,
        )
        return results

    def __enter__(self, *args: Any, **kwargs: Any) -> "ConcurrentAudioDecoder":
        self.start()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.stop()
