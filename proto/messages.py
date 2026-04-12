"""Binary message protocol for full-duplex streaming.

Wire format: [1 byte kind] [payload bytes]
Internal (gateway↔worker) adds session routing: [36 byte session_id] [1 byte kind] [payload]
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np


class MessageKind(enum.IntEnum):
    """Message type indicators (first byte of wire frame)."""
    READY = 0x00        # Server → client: session initialized
    AUDIO = 0x01        # Bidirectional: float32 PCM samples
    TEXT = 0x02         # Server → client: UTF-8 text delta
    SEQ_TRACE = 0x03    # Server → client: full sequence trace
    SEQ_DELTA = 0x04    # Server → client: incremental trace delta
    ERROR = 0x05        # Server → client: error message
    CLOSE = 0x06        # Bidirectional: graceful session close
    PING = 0x07         # Bidirectional: keepalive
    PONG = 0x08         # Bidirectional: keepalive response


@dataclass(slots=True)
class Frame:
    """A single protocol frame."""
    kind: MessageKind
    payload: bytes
    session_id: Optional[str] = None

    # --- Client ↔ Gateway encoding (no session_id) ---

    def encode(self) -> bytes:
        return bytes([self.kind]) + self.payload

    @classmethod
    def decode(cls, data: bytes) -> Frame:
        if len(data) < 1:
            raise ValueError("Empty frame")
        return cls(kind=MessageKind(data[0]), payload=data[1:])

    # --- Gateway ↔ Worker encoding (with session_id) ---

    def encode_internal(self) -> bytes:
        sid = (self.session_id or "").encode("utf-8").ljust(36, b"\x00")[:36]
        return sid + bytes([self.kind]) + self.payload

    @classmethod
    def decode_internal(cls, data: bytes) -> Frame:
        if len(data) < 37:
            raise ValueError("Internal frame too short")
        session_id = data[:36].rstrip(b"\x00").decode("utf-8")
        return cls(
            kind=MessageKind(data[36]),
            payload=data[37:],
            session_id=session_id,
        )

    # --- Convenience constructors ---

    @classmethod
    def ready(cls, session_id: Optional[str] = None) -> Frame:
        return cls(kind=MessageKind.READY, payload=b"", session_id=session_id)

    @classmethod
    def audio(cls, pcm: np.ndarray, session_id: Optional[str] = None) -> Frame:
        return cls(
            kind=MessageKind.AUDIO,
            payload=pcm.astype(np.float32).tobytes(),
            session_id=session_id,
        )

    @classmethod
    def text(cls, content: str, session_id: Optional[str] = None) -> Frame:
        return cls(
            kind=MessageKind.TEXT,
            payload=content.encode("utf-8"),
            session_id=session_id,
        )

    @classmethod
    def seq_trace(cls, content: str, session_id: Optional[str] = None) -> Frame:
        return cls(
            kind=MessageKind.SEQ_TRACE,
            payload=content.encode("utf-8"),
            session_id=session_id,
        )

    @classmethod
    def seq_delta(cls, content: str, session_id: Optional[str] = None) -> Frame:
        return cls(
            kind=MessageKind.SEQ_DELTA,
            payload=content.encode("utf-8"),
            session_id=session_id,
        )

    @classmethod
    def error(cls, message: str, session_id: Optional[str] = None) -> Frame:
        return cls(
            kind=MessageKind.ERROR,
            payload=message.encode("utf-8"),
            session_id=session_id,
        )

    @classmethod
    def close(
        cls,
        session_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Frame:
        payload = reason.encode("utf-8") if reason else b""
        return cls(kind=MessageKind.CLOSE, payload=payload, session_id=session_id)

    # --- Payload helpers ---

    def audio_samples(self) -> np.ndarray:
        return np.frombuffer(self.payload, dtype=np.float32)

    def text_content(self) -> str:
        return self.payload.decode("utf-8")
