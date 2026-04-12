"""Configuration types for fd-demo components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplingConfig:
    """Per-session sampling parameters."""
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.8
    eos_penalty: float = 0.0
    repetition_penalty: float = 1.0
    code_temperature: float = 0.8
    code_top_k: int = 10
    sil_penalty: float = 0.0
    bc_penalty: float = 0.0
    audio_encoder_chunk_frames: int = 100  # reset audio encoder KV every N frames (8s @ 12.5Hz)


@dataclass
class AudioConfig:
    """Audio processing parameters."""
    sampling_rate: int = 24000
    frame_size: int = 1920          # 80ms at 24kHz (model's native frame size)
    input_gain: float = 1.0
    input_clip: float = 100.0
    output_gain: float = 1.0
    output_clip: float = 100.0
    silence_rms_threshold: float = 0.0
    max_raw_buffer_seconds: float = 10.0
    soft_backlog_seconds: float = 0.40
    hard_backlog_seconds: float = 1.20
    hard_backlog_action: str = "degrade"   # degrade | close
    degrade_target_seconds: float = 0.30

    @property
    def frame_duration_ms(self) -> float:
        return self.frame_size / self.sampling_rate * 1000

    @property
    def max_buffer_bytes(self) -> int:
        return int(self.max_raw_buffer_seconds * self.sampling_rate * 4)  # float32


@dataclass
class SessionConfig:
    """Configuration for a single duplex session."""
    session_id: str = ""
    prompt: str = "eng:full_duplex:listen-first"
    prompt_role: str = "system"
    prompt_language: str = "eng"
    speak_first: bool = False
    system_prompt_style: str = "generic"
    system_prompt_persona: Optional[str] = None
    system_prompt_context: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    speaker_mode: str = "default"   # "none" | "default" | "recorded"
    speaker_key: Optional[str] = None
    idle_timeout_seconds: float = 15.0


@dataclass
class SGLangConfig:
    """SGLang model runner configuration."""
    model_path: str = ""
    dtype: str = "bfloat16"
    mem_fraction_static: float = 0.88
    disable_cuda_graph: Optional[bool] = True    # Disable — conflicts with duplex prefill
    cuda_graph_max_bs: Optional[int] = None       # Not used when CUDA graphs disabled
    max_running_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_prefill_tokens: Optional[int] = None
    chunked_prefill_size: Optional[int] = None
    max_allocated_req_pool_indices: int = 32


@dataclass
class WorkerConfig:
    """GPU worker configuration."""
    gpu_id: int = 0
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    max_sessions: int = 2
    warmup_on_start: bool = True


@dataclass
class GatewayConfig:
    """WebSocket gateway configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    max_connections: int = 256
    ping_interval: float = 30.0
    ping_timeout: float = 15.0


@dataclass
class ServerConfig:
    """Top-level server configuration."""
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    num_gpus: Optional[int] = None      # None = auto-detect
    workers_per_gpu: int = 1
    log_level: str = "INFO"
