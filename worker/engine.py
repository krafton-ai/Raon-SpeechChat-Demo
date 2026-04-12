"""RaonEngine — loads the SGLang model once and manages multiple sessions.

Provides sync and async APIs for session lifecycle, audio feed, and decode steps.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Awaitable

from proto.config import SessionConfig, WorkerConfig
from proto.close_reasons import (
    CLIENT_DISCONNECT,
    IDLE_TIMEOUT,
    INTERNAL_ERROR,
    OVERLOADED_BACKLOG,
    normalize_close_reason,
)
from proto.messages import Frame
from worker.session import RaonWorkerSession, SessionState, is_fatal_cuda_error
from worker.session_leak_logging import (
    engine_registry_fields,
    log_session_leak_detail,
    log_session_leak_summary,
)

if TYPE_CHECKING:
    from raon_runtime.sglang_backend import SGLangRaonModel
    from transformers import Qwen2TokenizerFast

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    rank = int(round((p / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(rank, len(ordered) - 1))]


class RaonEngine:
    """Loads SGLangRaonModel once, manages concurrent sessions."""

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._model: SGLangRaonModel | None = None
        self._tokenizer: Qwen2TokenizerFast | None = None
        self.sessions: dict[str, SessionState] = {}
        self._session_locks: dict[str, threading.Lock] = {}
        self._pending_sessions: set[str] = set()
        self._sessions_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._started_at: float = 0.0
        self._default_speaker_embeds: object | None = None
        self._default_speaker_source: str | None = None
        self._metrics_lock = threading.Lock()
        self._metric_samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=2048))
        self._metric_counters: dict[str, int] = defaultdict(int)
        self._close_reason_counts: dict[str, int] = defaultdict(int)
        self._fatal_error: str | None = None

    def _observe_metric(self, name: str, value: float) -> None:
        with self._metrics_lock:
            self._metric_samples[name].append(float(value))

    def _inc_counter(self, name: str, amount: int = 1) -> None:
        with self._metrics_lock:
            self._metric_counters[name] += int(amount)

    def _record_close_reason(self, reason: str) -> None:
        with self._metrics_lock:
            self._close_reason_counts[normalize_close_reason(reason)] += 1

    def _mark_fatal_error(self, context: str, exc: BaseException) -> None:
        message = f"{context}: {exc}"
        if self._fatal_error is None:
            self._fatal_error = message
            logger.error("worker_marked_unhealthy reason=%s", message)
        self._inc_counter("fatal_cuda_errors")

    def _metric_summary(self, name: str) -> dict[str, float]:
        with self._metrics_lock:
            values = list(self._metric_samples.get(name, ()))
        if not values:
            return {
                "count": 0,
                "avg": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
            }
        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "p99": _percentile(values, 99),
            "max": max(values),
        }

    def _metrics_snapshot(self) -> dict:
        with self._metrics_lock:
            counters = dict(self._metric_counters)
            close_reasons = dict(self._close_reason_counts)
            names = list(self._metric_samples.keys())
        sampled = {name: self._metric_summary(name) for name in names}
        return {
            "counters": counters,
            "close_reasons": close_reasons,
            "sampled": sampled,
        }

    def _runtime_stats_snapshot(self) -> dict[str, object]:
        if self._model is None:
            return {}

        runtime: dict[str, object] = {}
        audio_stats_getter = getattr(self.model, "get_audio_runtime_stats", None)
        if callable(audio_stats_getter):
            try:
                runtime["audio_input"] = audio_stats_getter()
            except Exception:
                logger.exception("failed to collect audio runtime stats")

        try:
            inner_model = self.model.get_model()
        except Exception:
            return runtime

        code_predictor = getattr(inner_model, "code_predictor", None)
        code_predictor_stats_getter = getattr(code_predictor, "get_predict_codes_runtime_stats", None)
        if callable(code_predictor_stats_getter):
            try:
                runtime["code_predictor"] = code_predictor_stats_getter()
            except Exception:
                logger.exception("failed to collect code predictor runtime stats")

        return runtime

    @property
    def model(self) -> SGLangRaonModel:
        assert self._model is not None, "Engine not initialized. Call load_model() first."
        return self._model

    @property
    def tokenizer(self) -> Qwen2TokenizerFast:
        assert self._tokenizer is not None, "Engine not initialized. Call load_model() first."
        return self._tokenizer

    def _resolve_model_dtype(self):
        import torch

        dtype = getattr(self.model, "dtype", None)
        if isinstance(dtype, torch.dtype):
            return dtype
        try:
            return next(self.model.get_model().parameters()).dtype
        except (AttributeError, StopIteration, TypeError):
            return torch.float32

    def _resolve_model_device(self):
        import torch

        device = getattr(self.model, "device", None)
        if device is not None:
            return torch.device(device)
        try:
            return next(self.model.get_model().parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return torch.device("cpu")

    def _model_has_speaker_encoder(self) -> bool:
        try:
            return bool(getattr(self.model.get_model(), "speaker_encoder", None) is not None)
        except Exception:
            return False

    def _compute_speaker_embeds_from_silence(self):
        import torch
        from raon_runtime.modules import PretrainedSpeakerEncoder

        speaker_encoder = getattr(self.model.get_model(), "speaker_encoder", None)
        if speaker_encoder is None:
            return None

        device = self._resolve_model_device()
        dtype = self._resolve_model_dtype()
        sampling_rate = int(getattr(self.model, "sampling_rate", 24000))

        # Deterministic fallback speaker reference: 1s of silence.
        audio = torch.zeros(1, sampling_rate, device=device, dtype=dtype)
        audio_lengths = torch.tensor([audio.shape[1]], device=device)

        if isinstance(speaker_encoder, PretrainedSpeakerEncoder):
            speaker_embeds = speaker_encoder(audio, audio_lengths)
        else:
            tokenized = self.model.tokenize_audio(
                audio=audio,
                audio_lengths=audio_lengths,
                return_mimi_features=True,
            )
            if tokenized.mimi_features is None or tokenized.audio_codes_mask is None:
                raise RuntimeError("Failed to compute Mimi features for default speaker embedding.")
            speaker_embeds = speaker_encoder(tokenized.mimi_features, mask=tokenized.audio_codes_mask)

        if getattr(speaker_embeds, "ndim", 0) == 2:
            speaker_embeds = speaker_embeds.unsqueeze(1)
        return speaker_embeds.to(device=device, dtype=dtype)

    def _compute_speaker_embeds_from_audio_path(self, audio_path: Path):
        import torch
        from raon_runtime.modules import PretrainedSpeakerEncoder

        try:
            import torchaudio
        except ImportError:
            logger.warning(
                "DEFAULT_SPEAKER_AUDIO_PATH was set, but torchaudio is unavailable. "
                "Install torchaudio or use DEFAULT_SPEAKER_EMBEDDING_PATH."
            )
            return None

        if not audio_path.exists():
            logger.warning("DEFAULT_SPEAKER_AUDIO_PATH does not exist: %s", audio_path)
            return None

        speaker_encoder = getattr(self.model.get_model(), "speaker_encoder", None)
        if speaker_encoder is None:
            return None

        dtype = self._resolve_model_dtype()
        device = self._resolve_model_device()
        sampling_rate = int(getattr(self.model, "sampling_rate", 24000))

        audio, sr = torchaudio.load(str(audio_path))
        if sr != sampling_rate:
            audio = torchaudio.functional.resample(audio, sr, sampling_rate)
        audio = audio.mean(dim=0, keepdim=True).to(device=device, dtype=dtype)
        audio_lengths = torch.tensor([audio.shape[1]], device=device)

        if isinstance(speaker_encoder, PretrainedSpeakerEncoder):
            speaker_embeds = speaker_encoder(audio, audio_lengths)
        else:
            tokenized = self.model.tokenize_audio(
                audio=audio,
                audio_lengths=audio_lengths,
                return_mimi_features=True,
            )
            if tokenized.mimi_features is None or tokenized.audio_codes_mask is None:
                raise RuntimeError("Failed to compute Mimi features for default speaker embedding.")
            speaker_embeds = speaker_encoder(tokenized.mimi_features, mask=tokenized.audio_codes_mask)

        if getattr(speaker_embeds, "ndim", 0) == 2:
            speaker_embeds = speaker_embeds.unsqueeze(1)
        return speaker_embeds.to(device=device, dtype=dtype)

    def _load_default_speaker_embeds(self) -> None:
        if not self._model_has_speaker_encoder():
            logger.info("model has no speaker_encoder; speaker conditioning disabled")
            self._default_speaker_embeds = None
            self._default_speaker_source = None
            return

        embed_path_raw = os.environ.get("DEFAULT_SPEAKER_EMBEDDING_PATH", "").strip()
        audio_path_raw = os.environ.get("DEFAULT_SPEAKER_AUDIO_PATH", "").strip()
        source_label = os.environ.get("DEFAULT_SPEAKER_LABEL", "").strip() or None

        speaker_embeds = None
        source = None

        if embed_path_raw:
            import torch

            embed_path = Path(embed_path_raw)
            if embed_path.exists():
                loaded = torch.load(
                    str(embed_path),
                    map_location=self._resolve_model_device(),
                    weights_only=True,
                )
                if getattr(loaded, "ndim", 0) == 2:
                    loaded = loaded.unsqueeze(1)
                speaker_embeds = loaded.to(
                    device=self._resolve_model_device(),
                    dtype=self._resolve_model_dtype(),
                )
                source = f"embedding:{embed_path}"
            else:
                logger.warning(
                    "DEFAULT_SPEAKER_EMBEDDING_PATH does not exist: %s",
                    embed_path,
                )

        if speaker_embeds is None and audio_path_raw:
            audio_path = Path(audio_path_raw)
            try:
                speaker_embeds = self._compute_speaker_embeds_from_audio_path(audio_path)
            except Exception:
                logger.exception("failed to compute default speaker embedding from audio: %s", audio_path)
                speaker_embeds = None
            if speaker_embeds is not None:
                source = f"audio:{audio_path}"

        if speaker_embeds is None:
            # Keep a deterministic default tone even when no explicit reference is configured.
            try:
                speaker_embeds = self._compute_speaker_embeds_from_silence()
                source = "silence-fallback"
            except Exception:
                logger.exception("failed to compute silence-fallback speaker embedding")
                speaker_embeds = None
                source = None

        self._default_speaker_embeds = speaker_embeds
        self._default_speaker_source = source_label or source
        logger.info(
            "default speaker ready source=%s has_embeds=%s",
            self._default_speaker_source,
            self._default_speaker_embeds is not None,
        )

    def load_model(self) -> None:
        """Load SGLang model onto the configured GPU. Blocks until ready."""
        from raon_runtime.sglang_backend import SGLangRaonModel
        from transformers import Qwen2TokenizerFast

        cfg = self.config.sglang
        logger.info(
            "loading model gpu_id=%d path=%s",
            self.config.gpu_id,
            cfg.model_path,
        )

        # When running under Ray with num_gpus=1, CUDA_VISIBLE_DEVICES is
        # already set to a single GPU. Always use gpu_id=0 (the only visible device).
        effective_gpu_id = 0 if "CUDA_VISIBLE_DEVICES" in os.environ else self.config.gpu_id

        self._model = SGLangRaonModel(
            path=cfg.model_path,
            dtype=cfg.dtype,
            mem_fraction_static=cfg.mem_fraction_static,
            disable_cuda_graph=cfg.disable_cuda_graph,
            cuda_graph_max_bs=cfg.cuda_graph_max_bs,
            max_running_requests=cfg.max_running_requests,
            max_total_tokens=cfg.max_total_tokens,
            max_prefill_tokens=cfg.max_prefill_tokens,
            chunked_prefill_size=cfg.chunked_prefill_size,
            max_allocated_req_pool_indices=cfg.max_allocated_req_pool_indices,
            gpu_id=effective_gpu_id,
        )

        # Apply runtime model overrides (matches live_demo_server.py behavior)
        # These must be set on BOTH the wrapper and the inner model
        inner_model = self._model.get_model()
        self._model.use_duplex_end_pad = True
        inner_model.use_duplex_end_pad = True

        # Enable the fastest supported attention backend on all available sub-modules.
        if hasattr(inner_model, '_set_attention_implementation'):
            attention_errors: list[str] = []
            for attn_impl in ("sdpa", "eager"):
                try:
                    updated = inner_model._set_attention_implementation(attn_impl)
                    logger.info(
                        "attention_implementation set to %s for %s",
                        attn_impl,
                        ", ".join(updated) if updated else "no submodules",
                    )
                    break
                except Exception as e:
                    attention_errors.append(f"{attn_impl}: {e}")
            else:
                logger.warning(
                    "failed to set attention_implementation after trying sdpa, eager: %s",
                    " | ".join(attention_errors),
                )
        else:
            logger.info("_set_attention_implementation not available on this model version")

        # Read sequence_mode, use_sil_token, sil_no_audio from model config (don't override)
        logger.info(
            "model config: use_duplex_end_pad=%s sequence_mode=%s use_sil_token=%s sil_no_audio=%s",
            getattr(self._model, 'use_duplex_end_pad', None),
            getattr(self._model, 'sequence_mode', None),
            getattr(self._model, 'use_sil_token', None),
            getattr(self._model, 'sil_no_audio', None),
        )

        # Load tokenizer from text_model/ subdir of the SGLang bundle
        tokenizer_path = os.path.join(cfg.model_path, "text_model")
        self._tokenizer = Qwen2TokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)

        # Start concurrent audio decoder (required for streaming decode)
        self._model.start_concurrent_audio_decoder()

        if hasattr(self._model, 'compile_audio_modules'):
            if _env_flag("FD_ENABLE_COMPILE_AUDIO_MODULES", default=True):
                try:
                    logger.info("warming audio modules (code predictor CUDA graph + Voxtral compiled core)...")
                    self._model.compile_audio_modules(duplex=True)
                    logger.info("audio modules warmed successfully runtime=%s", self._runtime_stats_snapshot())
                except Exception as e:
                    logger.warning("compile_audio_modules failed (continuing without): %s", e)
            else:
                logger.info("skipping compile_audio_modules; FD_ENABLE_COMPILE_AUDIO_MODULES is disabled")

        self._load_default_speaker_embeds()

        self._started_at = time.time()
        logger.info("model loaded gpu_id=%d", self.config.gpu_id)

    def create_session(self, session_id: str, config: SessionConfig) -> bool:
        """Create a new session. Returns False if at capacity or ID exists."""
        if self._fatal_error is not None:
            logger.error(
                "session_rejected_worker_unhealthy session_id=%s reason=%s",
                session_id,
                self._fatal_error,
            )
            return False

        with self._sessions_lock:
            if session_id in self.sessions or session_id in self._pending_sessions:
                logger.warning("session_exists session_id=%s", session_id)
                return False
            in_flight = len(self.sessions) + len(self._pending_sessions)
            if in_flight >= self.config.max_sessions:
                logger.warning(
                    "max_sessions_reached current=%d max=%d",
                    in_flight,
                    self.config.max_sessions,
                )
                return False
            session_lock = threading.Lock()
            self._session_locks[session_id] = session_lock
            self._pending_sessions.add(session_id)
            active_sessions = list(self.sessions.keys())
            pending_sessions = sorted(self._pending_sessions)

        log_session_leak_summary(
            logger,
            "create_start",
            **engine_registry_fields(
                active_sessions=active_sessions,
                pending_sessions=pending_sessions,
                session_id=session_id,
                has_state=False,
                has_lock=True,
            ),
        )
        log_session_leak_detail(
            logger,
            "create_start_registry",
            **engine_registry_fields(
                active_sessions=active_sessions,
                pending_sessions=pending_sessions,
                session_id=session_id,
                has_state=False,
                has_lock=True,
            ),
        )

        config.session_id = session_id
        speaker_embeds = None
        if config.speaker_mode == "default":
            if self._default_speaker_embeds is None:
                logger.warning(
                    "speaker_mode=default requested but no default speaker embedding is available; "
                    "falling back to speaker_mode=none"
                )
            else:
                speaker_embeds = self._default_speaker_embeds
        elif config.speaker_mode == "recorded":
            logger.warning(
                "speaker_mode=recorded requested but fd-demo worker does not store recorded speakers; "
                "falling back to default speaker"
            )
            speaker_embeds = self._default_speaker_embeds

        logger.info(
            "session_init_start session_id=%s speaker_mode=%s speaker_source=%s speaker_embeds=%s",
            session_id,
            config.speaker_mode,
            self._default_speaker_source,
            speaker_embeds is not None,
        )

        session_init_started_at = time.perf_counter()
        state: SessionState | None = None
        try:
            with session_lock, self._model_lock:
                state = RaonWorkerSession.init(
                    model=self.model,
                    session_config=config,
                    tokenizer=self.tokenizer,
                    speaker_embeds=speaker_embeds,
                )
                RaonWorkerSession._ensure_decoding_state(
                    model=self.model,
                    state=state,
                )
        except Exception as exc:
            if is_fatal_cuda_error(exc):
                self._mark_fatal_error("create_session", exc)
            if state is not None and state.decoding_state is not None:
                with session_lock, self._model_lock:
                    RaonWorkerSession.close(model=self.model, state=state)
            with self._sessions_lock:
                self._pending_sessions.discard(session_id)
                self._session_locks.pop(session_id, None)
                active_sessions = list(self.sessions.keys())
                pending_sessions = sorted(self._pending_sessions)
            log_session_leak_summary(
                logger,
                "create_failed",
                error=type(exc).__name__,
                **engine_registry_fields(
                    active_sessions=active_sessions,
                    pending_sessions=pending_sessions,
                    session_id=session_id,
                    has_state=state is not None and state.decoding_state is not None,
                    has_lock=False,
                ),
            )
            log_session_leak_detail(
                logger,
                "create_failed_registry",
                error=str(exc),
                **engine_registry_fields(
                    active_sessions=active_sessions,
                    pending_sessions=pending_sessions,
                    session_id=session_id,
                    has_state=state is not None and state.decoding_state is not None,
                    has_lock=False,
                ),
            )
            raise

        logger.info(
            "session_init_done session_id=%s elapsed_ms=%.1f",
            session_id,
            (time.perf_counter() - session_init_started_at) * 1000.0,
        )
        abandoned = False
        with self._sessions_lock:
            if session_id in self._pending_sessions:
                self._pending_sessions.remove(session_id)
                self.sessions[session_id] = state
                total = len(self.sessions)
            else:
                abandoned = True
                total = len(self.sessions)
        if abandoned:
            # Session was canceled while init was in progress.
            with session_lock, self._model_lock:
                RaonWorkerSession.close(model=self.model, state=state)
            logger.warning("session_create_abandoned session_id=%s", session_id)
            log_session_leak_summary(
                logger,
                "create_abandoned",
                session_id=session_id,
                total=total,
            )
            return False
        logger.info(
            "session_created session_id=%s total=%d",
            session_id,
            total,
        )
        log_session_leak_summary(
            logger,
            "create_done",
            session_id=session_id,
            total=total,
            pending_count=0,
        )
        log_session_leak_detail(
            logger,
            "create_done_registry",
            **engine_registry_fields(
                active_sessions=list(self.sessions.keys()),
                pending_sessions=sorted(self._pending_sessions),
                session_id=session_id,
                has_state=state is not None and state.decoding_state is not None,
                has_lock=session_id in self._session_locks,
            ),
        )
        self._inc_counter("created_sessions")
        return True

    def close_session(self, session_id: str, reason: str = CLIENT_DISCONNECT) -> None:
        """Free KV cache and remove session."""
        close_reason = normalize_close_reason(reason)
        with self._sessions_lock:
            pre_active_sessions = list(self.sessions.keys())
            pre_pending_sessions = sorted(self._pending_sessions)
            pre_has_state = session_id in self.sessions
            pre_has_lock = session_id in self._session_locks
        log_session_leak_summary(
            logger,
            "close_start",
            reason=close_reason,
            **engine_registry_fields(
                active_sessions=pre_active_sessions,
                pending_sessions=pre_pending_sessions,
                session_id=session_id,
                has_state=pre_has_state,
                has_lock=pre_has_lock,
            ),
        )
        log_session_leak_detail(
            logger,
            "close_start_registry",
            reason=close_reason,
            **engine_registry_fields(
                active_sessions=pre_active_sessions,
                pending_sessions=pre_pending_sessions,
                session_id=session_id,
                has_state=pre_has_state,
                has_lock=pre_has_lock,
            ),
        )
        with self._sessions_lock:
            state = self.sessions.pop(session_id, None)
            session_lock = self._session_locks.pop(session_id, None)
            pending = session_id in self._pending_sessions
            if pending:
                self._pending_sessions.discard(session_id)
            if state is None and pending:
                logger.info(
                    "session_cancelled_during_init session_id=%s reason=%s",
                    session_id,
                    close_reason,
                )
                log_session_leak_summary(
                    logger,
                    "close_cancelled_during_init",
                    session_id=session_id,
                    reason=close_reason,
                )
                return
        if state is None:
            log_session_leak_summary(
                logger,
                "close_missing",
                session_id=session_id,
                reason=close_reason,
            )
            return
        state.close_reason = close_reason
        if session_lock is None:
            session_lock = threading.Lock()
        skip_gpu_cleanup = self._fatal_error is not None
        with session_lock, self._model_lock:
            RaonWorkerSession.close(
                model=self.model,
                state=state,
                skip_gpu_cleanup=skip_gpu_cleanup,
            )
        self._record_close_reason(close_reason)
        self._inc_counter("closed_sessions")
        with self._sessions_lock:
            remaining = len(self.sessions)
        logger.info(
            "session_removed session_id=%s remaining=%d reason=%s",
            session_id,
            remaining,
            close_reason,
        )
        log_session_leak_summary(
            logger,
            "close_done",
            session_id=session_id,
            remaining=remaining,
            reason=close_reason,
            skip_gpu_cleanup=skip_gpu_cleanup,
        )
        log_session_leak_detail(
            logger,
            "close_done_registry",
            **engine_registry_fields(
                active_sessions=list(self.sessions.keys()),
                pending_sessions=sorted(self._pending_sessions),
                session_id=session_id,
                has_state=False,
                has_lock=session_id in self._session_locks,
            ),
        )

    def feed_audio(self, session_id: str, pcm_bytes: bytes) -> None:
        """Buffer audio for a session."""
        with self._sessions_lock:
            state = self.sessions.get(session_id)
            session_lock = self._session_locks.get(session_id)
        if state is None or session_lock is None:
            return
        with session_lock:
            feed_result = RaonWorkerSession.feed_audio(state=state, pcm_bytes=pcm_bytes)
            state.dropped_input_bytes += feed_result.dropped_bytes
            state.dropped_input_frames += feed_result.dropped_frames
            state.max_time_behind_seconds = max(
                state.max_time_behind_seconds,
                feed_result.time_behind_seconds,
            )
            if feed_result.soft_backlog:
                state.backlog_soft_events += 1
                self._inc_counter("backlog_soft_events")
                if state.backlog_soft_events % 25 == 1:
                    logger.warning(
                        "session_backlog_soft session_id=%s behind=%.3fs backlog_frames=%.1f",
                        session_id,
                        feed_result.time_behind_seconds,
                        feed_result.backlog_frames,
                    )
            if feed_result.hard_backlog:
                state.backlog_hard_events += 1
                self._inc_counter("backlog_hard_events")
                if feed_result.hard_action == "close":
                    state.close_requested_reason = OVERLOADED_BACKLOG
                logger.warning(
                    "session_backlog_hard session_id=%s action=%s behind=%.3fs backlog_frames=%.1f",
                    session_id,
                    feed_result.hard_action,
                    feed_result.time_behind_seconds,
                    feed_result.backlog_frames,
                )

        if feed_result.dropped_bytes > 0:
            self._inc_counter("dropped_input_bytes", feed_result.dropped_bytes)
            self._inc_counter("dropped_input_frames", feed_result.dropped_frames)
        self._observe_metric("input_backlog_bytes", feed_result.backlog_bytes)
        self._observe_metric("input_backlog_frames", feed_result.backlog_frames)
        self._observe_metric("time_behind_real_time_ms", feed_result.time_behind_seconds * 1000.0)

    def run_step(self, session_id: str) -> list[Frame]:
        """Run one decode step for a session. Returns output frames."""
        if self._fatal_error is not None:
            self.close_session(session_id, reason=INTERNAL_ERROR)
            return [
                Frame.error("worker unhealthy: fatal CUDA error", session_id=session_id),
                Frame.close(session_id=session_id, reason=INTERNAL_ERROR),
            ]

        with self._sessions_lock:
            state = self.sessions.get(session_id)
            session_lock = self._session_locks.get(session_id)
        if state is None or session_lock is None:
            return []
        pending_close_reason: str | None = None
        with session_lock:
            if state.close_requested_reason:
                pending_close_reason = normalize_close_reason(state.close_requested_reason)
                state.close_requested_reason = None
            if state.decoding_state is None:
                if not pending_close_reason:
                    return []
            frame_bytes = state.config.audio.frame_size * 4
            if not pending_close_reason and len(state.raw_input_bytes) < frame_bytes:
                return []
            if pending_close_reason:
                # Exit quickly on fail-fast overload close.
                pass
            else:
                errors_before = state.decode_errors
                wait_start = time.perf_counter()
                with self._model_lock:
                    lock_wait_ms = (time.perf_counter() - wait_start) * 1000.0
                    decode_start = time.perf_counter()
                    try:
                        out = RaonWorkerSession.step(
                            model=self.model,
                            state=state,
                            tokenizer=self.tokenizer,
                        )
                    except Exception as exc:
                        if is_fatal_cuda_error(exc):
                            self._mark_fatal_error("run_step", exc)
                            pending_close_reason = INTERNAL_ERROR
                            out = []
                        else:
                            raise
                    decode_step_ms = (time.perf_counter() - decode_start) * 1000.0
                self._observe_metric("_model_lock_wait_ms", lock_wait_ms)
                self._observe_metric("decode_step_ms", decode_step_ms)
                if state.decode_errors > errors_before:
                    self._inc_counter("decode_step_errors")
                if pending_close_reason is None:
                    return out

        if pending_close_reason:
            self.close_session(session_id, reason=pending_close_reason)
            return [
                Frame.error(f"session closed: {pending_close_reason}", session_id=session_id),
                Frame.close(session_id=session_id, reason=pending_close_reason),
            ]
        return []

    async def run_loop(
        self,
        session_id: str,
        send_callback: Callable[[list[Frame]], Awaitable[None]],
    ) -> None:
        """Async loop: read from buffer, decode in thread, send output frames.

        Runs until session is closed or idle timeout is reached.
        """
        with self._sessions_lock:
            state = self.sessions.get(session_id)
            session_lock = self._session_locks.get(session_id)
        if state is None:
            return
        if session_lock is None:
            return

        idle_timeout = state.config.idle_timeout_seconds
        audio_cfg = state.config.audio
        frame_bytes = audio_cfg.frame_size * 4
        close_reason = CLIENT_DISCONNECT

        try:
            while True:
                with self._sessions_lock:
                    state = self.sessions.get(session_id)
                if state is None:
                    break

                # Check idle timeout
                with session_lock:
                    idle_seconds = state.idle_seconds
                    buffered = len(state.raw_input_bytes)

                if idle_timeout > 0 and idle_seconds > idle_timeout:
                    logger.info(
                        "idle_timeout session_id=%s idle=%.1fs",
                        session_id,
                        idle_seconds,
                    )
                    close_reason = IDLE_TIMEOUT
                    break

                # Wait for enough data
                if buffered < frame_bytes:
                    await asyncio.sleep(0.001)
                    continue

                # Run decode step in thread to avoid blocking event loop
                frames = await asyncio.to_thread(self.run_step, session_id)

                if frames:
                    await send_callback(frames)

        except Exception:
            logger.exception("run_loop_error session_id=%s", session_id)
            close_reason = INTERNAL_ERROR
        finally:
            self.close_session(session_id, reason=close_reason)

    def cleanup_idle_sessions(self, max_idle_seconds: float | None = None) -> list[str]:
        """Close sessions that have been idle beyond timeout. Returns closed IDs."""
        closed: list[str] = []
        with self._sessions_lock:
            session_ids = list(self.sessions.keys())

        for session_id in session_ids:
            with self._sessions_lock:
                state = self.sessions.get(session_id)
                session_lock = self._session_locks.get(session_id)
            if state is None:
                continue
            if session_lock is None:
                continue
            with session_lock:
                idle_seconds = state.idle_seconds
            timeout = max_idle_seconds if max_idle_seconds is not None else state.config.idle_timeout_seconds
            if timeout > 0 and idle_seconds > timeout:
                self.close_session(session_id, reason=IDLE_TIMEOUT)
                closed.append(session_id)
        return closed

    def health(self) -> dict:
        """Report engine health status."""
        with self._sessions_lock:
            sessions = list(self.sessions.items())
            session_ids = [sid for sid, _ in sessions]
            session_count = len(session_ids)
            session_summaries = {
                sid: {
                    "backlog_frames": (len(state.raw_input_bytes) / max(1, state.config.audio.frame_size * 4)),
                    "time_behind_ms": state.max_time_behind_seconds * 1000.0,
                    "dropped_input_frames": state.dropped_input_frames,
                    "dropped_input_bytes": state.dropped_input_bytes,
                    "backlog_soft_events": state.backlog_soft_events,
                    "backlog_hard_events": state.backlog_hard_events,
                    "decode_errors": state.decode_errors,
                    "consecutive_decode_errors": state.consecutive_decode_errors,
                }
                for sid, state in sessions
            }
        return {
            "gpu_id": self.config.gpu_id,
            "session_count": session_count,
            "max_sessions": self.config.max_sessions,
            "sessions": session_ids,
            "uptime": time.time() - self._started_at if self._started_at else 0.0,
            "model_loaded": self._model is not None,
            "healthy": self._fatal_error is None and self._model is not None,
            "fatal_error": self._fatal_error,
            "has_default_speaker": self._default_speaker_embeds is not None,
            "default_speaker_source": self._default_speaker_source,
            "metrics": self._metrics_snapshot(),
            "runtime": self._runtime_stats_snapshot(),
            "session_stats": session_summaries,
        }
