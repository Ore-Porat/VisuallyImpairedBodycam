"""Wire protocol between device and server (PRD section 7).

Frame channel (UDP): one datagram = fixed binary header + one JPEG. The server answers each
processed frame with a small binary ack datagram.

Control channel (WebSocket): JSON text messages with a ``type`` field.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any

FRAME_MAGIC = b"SLR1"
ACK_MAGIC = b"SLRA"

# magic, frame_id, session_id, capture_ts_ms, purpose, burst_id, width, height
FRAME_HEADER = struct.Struct("!4sIIQBHHH")
# magic, session_id, frame_id, infer_ms, server_ts_ms
ACK = struct.Struct("!4sIIfQ")

PURPOSE_FIND = 0
PURPOSE_IDENTIFY = 1
PURPOSE_NAMES = {PURPOSE_FIND: "find", PURPOSE_IDENTIFY: "identify"}
PURPOSE_CODES = {v: k for k, v in PURPOSE_NAMES.items()}

DEFAULT_MAX_DATAGRAM = 60000


class ProtocolError(ValueError):
    pass


class FrameTooLarge(ProtocolError):
    pass


@dataclass(frozen=True)
class FrameHeader:
    frame_id: int
    session_id: int
    capture_ts_ms: int
    purpose: int = PURPOSE_FIND
    burst_id: int = 0
    width: int = 0
    height: int = 0

    @property
    def purpose_name(self) -> str:
        return PURPOSE_NAMES.get(self.purpose, "find")


def pack_frame(header: FrameHeader, jpeg: bytes, max_bytes: int = DEFAULT_MAX_DATAGRAM) -> bytes:
    data = FRAME_HEADER.pack(
        FRAME_MAGIC, header.frame_id & 0xFFFFFFFF, header.session_id & 0xFFFFFFFF,
        header.capture_ts_ms, header.purpose, header.burst_id & 0xFFFF,
        header.width, header.height,
    ) + jpeg
    if len(data) > max_bytes:
        raise FrameTooLarge(f"frame datagram is {len(data)} bytes, limit {max_bytes}")
    return data


def unpack_frame(data: bytes) -> tuple[FrameHeader, bytes]:
    if len(data) < FRAME_HEADER.size:
        raise ProtocolError("datagram shorter than frame header")
    magic, frame_id, session_id, ts, purpose, burst_id, w, h = FRAME_HEADER.unpack_from(data)
    if magic != FRAME_MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if purpose not in PURPOSE_NAMES:
        raise ProtocolError(f"unknown purpose {purpose}")
    return FrameHeader(frame_id, session_id, ts, purpose, burst_id, w, h), data[FRAME_HEADER.size:]


@dataclass(frozen=True)
class FrameAck:
    session_id: int
    frame_id: int
    infer_ms: float
    server_ts_ms: int


def pack_ack(ack: FrameAck) -> bytes:
    return ACK.pack(ACK_MAGIC, ack.session_id, ack.frame_id, ack.infer_ms, ack.server_ts_ms)


def unpack_ack(data: bytes) -> FrameAck:
    if len(data) != ACK.size:
        raise ProtocolError("ack datagram has wrong size")
    magic, session_id, frame_id, infer_ms, server_ts = ACK.unpack(data)
    if magic != ACK_MAGIC:
        raise ProtocolError(f"bad ack magic {magic!r}")
    return FrameAck(session_id, frame_id, infer_ms, server_ts)


# --------------------------------------------------------------------------- WebSocket messages

# Required fields per message type. Optional fields are allowed and passed through.
DEVICE_TO_SERVER = {
    "hello": ("device_id", "session_id", "udp_port", "sw_version", "clip_manifest_hash"),
    "control": ("event", "ts"),
    "heartbeat": ("ts", "frames_sent", "acks_received"),
    "cue_started": ("cue_id", "ts"),  # extension: lets the server log true capture-to-audio latency
}
SERVER_TO_DEVICE = {
    "cue": ("cue_id", "clip", "priority", "interrupt"),
    "tone": ("freq_hz", "duration_ms", "pan", "interval_ms"),
    "stream": ("on", "max_fps", "purpose"),
    "clips_sync": ("clips",),
}
MESSAGE_FIELDS = {**DEVICE_TO_SERVER, **SERVER_TO_DEVICE}

PRIORITIES = {"info": 0, "guidance": 1, "confirmation": 2, "system": 3}


def manifest_hash(manifest: dict[str, str]) -> str:
    """Short hash of a clip manifest (name -> sha256), compared in hello to skip clip sync."""
    h = hashlib.sha256()
    for name in sorted(manifest):
        h.update(f"{name}:{manifest[name]}\n".encode())
    return h.hexdigest()[:16]


def encode(msg_type: str, **fields: Any) -> str:
    msg = {"type": msg_type, **fields}
    validate(msg)
    return json.dumps(msg, separators=(",", ":"))


def decode(text: str | bytes) -> dict[str, Any]:
    try:
        msg = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProtocolError(f"not JSON: {e}") from e
    validate(msg)
    return msg


def validate(msg: Any) -> None:
    if not isinstance(msg, dict) or "type" not in msg:
        raise ProtocolError("message must be an object with a 'type' field")
    fields = MESSAGE_FIELDS.get(msg["type"])
    if fields is None:
        raise ProtocolError(f"unknown message type {msg['type']!r}")
    missing = [f for f in fields if f not in msg]
    if missing:
        raise ProtocolError(f"{msg['type']} message missing {missing}")
