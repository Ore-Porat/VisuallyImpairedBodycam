"""Wire protocol between device and server (PRD section 7).

Frame channel (UDP): one datagram = fixed binary header + one JPEG. The server answers each
processed frame with a small binary ack datagram.

Control channel (WebSocket): JSON text messages with a ``type`` field.

How it fits in
--------------
The Raspberry Pi (device) and the laptop (server) are two separate computers talking over
Wi-Fi. This file is the shared "language" they agree on, so both sides import it. Nothing in
here touches the network itself; it only converts between Python objects and the bytes/text
that travel over the wire. The actual sending and receiving lives in slr/device/frame_sender.py,
slr/device/control_client.py and slr/server/frame_receiver.py, slr/server/main.py.

Two channels are used:

1. Video frames go over UDP. Think postcards: each one is sent on its own, fast, with no
   guarantee it arrives and no automatic retry. That is fine for video, where a late frame is
   useless anyway. Each postcard carries a small fixed-size binary header followed by one JPEG
   picture. The server replies to each frame with a tiny "ack" (acknowledgement) so the device
   can measure how many frames get through and how fast.

2. Control messages (button presses, "say this clip", "start streaming") go over a WebSocket.
   Think of an open phone line: messages are reliable, in order, and either side can talk at
   any time. These messages are JSON text.

Words you will see
------------------
- datagram: one UDP packet.
- struct: Python's tool for packing numbers into a fixed binary layout (and reading them back).
  The format string ``"!4sIIQBHHH"`` lists the fields: ``!`` = network byte order, ``4s`` = 4
  raw bytes, ``I`` = 4-byte unsigned int, ``Q`` = 8-byte unsigned int, ``B`` = 1 byte,
  ``H`` = 2-byte unsigned int, ``f`` = 4-byte float.
- magic: a few fixed bytes at the start of a packet (``SLR1``, ``SLRA``) so the receiver can
  tell at a glance that the packet is one of ours and which kind it is.
- JSON: a text format for structured data, like ``{"type": "cue", "cue_id": 7}``.
- hash: a short fingerprint of some data (here SHA-256). If two fingerprints match, the data is
  the same. Used so the device and server can check they have the same set of audio clips
  without sending them all.
- dataclass(frozen=True): a simple container for data that cannot be changed after creation.
- ``& 0xFFFFFFFF``: bitwise AND with all-ones; keeps only the low 32 bits of a number so it
  fits into a 4-byte field even if the counter has grown beyond that.
- ``**fields``: lets a function accept any number of named arguments and collects them in a
  dictionary; ``{**a, **b}`` merges two dictionaries into a new one.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any

# Fixed byte markers at the start of each packet type: "SLR1" = frame, "SLRA" = ack.
FRAME_MAGIC = b"SLR1"
ACK_MAGIC = b"SLRA"

# Binary layouts of the two UDP packet headers (see the glossary above for the letters).
# magic, frame_id, session_id, capture_ts_ms, purpose, burst_id, width, height
FRAME_HEADER = struct.Struct("!4sIIQBHHH")
# magic, session_id, frame_id, infer_ms, server_ts_ms
ACK = struct.Struct("!4sIIfQ")

# Why a frame was sent: searching the shelf for a list item (find) or naming an item that is
# held up to the camera (identify). Stored as a 1-byte number in the header.
PURPOSE_FIND = 0
PURPOSE_IDENTIFY = 1
# Number -> word, and the reverse mapping (word -> number) built from it.
PURPOSE_NAMES = {PURPOSE_FIND: "find", PURPOSE_IDENTIFY: "identify"}
PURPOSE_CODES = {v: k for k, v in PURPOSE_NAMES.items()}

# Largest UDP packet we will send. The theoretical UDP maximum is about 65 KB; staying a bit
# below leaves room for the header and avoids edge cases on some networks.
DEFAULT_MAX_DATAGRAM = 60000


class ProtocolError(ValueError):
    """Raised when a packet or message does not follow this protocol (bad magic, missing field...)."""
    pass


class FrameTooLarge(ProtocolError):
    """Raised when a JPEG is too big for one UDP packet; the sender then lowers JPEG quality."""
    pass


@dataclass(frozen=True)
class FrameHeader:
    """The information that travels in front of each JPEG in a frame packet.

    - ``frame_id``: counts up by one per frame, so acks can be matched to frames.
    - ``session_id``: random number chosen when the device boots; lets the server ignore stale
      packets from a previous run.
    - ``capture_ts_ms``: the device's clock, in milliseconds, when the picture was taken. Used
      to measure latency.
    - ``purpose``: PURPOSE_FIND or PURPOSE_IDENTIFY (see above).
    - ``burst_id``: identify mode sends a short burst of frames; this number groups them.
    - ``width``, ``height``: picture size in pixels (0 if unknown).
    """

    frame_id: int
    session_id: int
    capture_ts_ms: int
    purpose: int = PURPOSE_FIND
    burst_id: int = 0
    width: int = 0
    height: int = 0

    @property
    def purpose_name(self) -> str:
        """The purpose as a word ("find" / "identify") instead of a number."""
        return PURPOSE_NAMES.get(self.purpose, "find")


def pack_frame(header: FrameHeader, jpeg: bytes, max_bytes: int = DEFAULT_MAX_DATAGRAM) -> bytes:
    """Build the bytes of one frame packet: binary header followed by the JPEG data.

    Used by the device before sending. Raises ``FrameTooLarge`` if the result would not fit in
    ``max_bytes``, which tells the sender to compress harder.
    """
    data = FRAME_HEADER.pack(
        FRAME_MAGIC, header.frame_id & 0xFFFFFFFF, header.session_id & 0xFFFFFFFF,
        header.capture_ts_ms, header.purpose, header.burst_id & 0xFFFF,
        header.width, header.height,
    ) + jpeg
    if len(data) > max_bytes:
        raise FrameTooLarge(f"frame datagram is {len(data)} bytes, limit {max_bytes}")
    return data


def unpack_frame(data: bytes) -> tuple[FrameHeader, bytes]:
    """The reverse of ``pack_frame``: split a received packet into (header, jpeg bytes).

    Used by the server. Raises ``ProtocolError`` if the packet is too short, does not start
    with our magic bytes, or has an unknown purpose code.
    """
    if len(data) < FRAME_HEADER.size:
        raise ProtocolError("datagram shorter than frame header")
    # unpack_from reads the header fields from the start of the data, ignoring the rest.
    magic, frame_id, session_id, ts, purpose, burst_id, w, h = FRAME_HEADER.unpack_from(data)
    if magic != FRAME_MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if purpose not in PURPOSE_NAMES:
        raise ProtocolError(f"unknown purpose {purpose}")
    # Everything after the header is the JPEG.
    return FrameHeader(frame_id, session_id, ts, purpose, burst_id, w, h), data[FRAME_HEADER.size:]


@dataclass(frozen=True)
class FrameAck:
    """The server's tiny reply to one frame.

    - ``session_id``, ``frame_id``: which frame this ack is for.
    - ``infer_ms``: how long the detector took on that frame, in milliseconds.
    - ``server_ts_ms``: the server's clock when it finished, in milliseconds.
    """

    session_id: int
    frame_id: int
    infer_ms: float
    server_ts_ms: int


def pack_ack(ack: FrameAck) -> bytes:
    """Turn a FrameAck into the bytes of an ack packet (server side)."""
    return ACK.pack(ACK_MAGIC, ack.session_id, ack.frame_id, ack.infer_ms, ack.server_ts_ms)


def unpack_ack(data: bytes) -> FrameAck:
    """Turn a received ack packet back into a FrameAck (device side). Raises ProtocolError if bad."""
    if len(data) != ACK.size:
        raise ProtocolError("ack datagram has wrong size")
    magic, session_id, frame_id, infer_ms, server_ts = ACK.unpack(data)
    if magic != ACK_MAGIC:
        raise ProtocolError(f"bad ack magic {magic!r}")
    return FrameAck(session_id, frame_id, infer_ms, server_ts)


# --------------------------------------------------------------------------- WebSocket messages
# Every WebSocket message is a JSON object with a "type" field plus the fields listed below.
# The tables are used by ``validate`` to reject malformed messages early, on both sides.

# Required fields per message type. Optional fields are allowed and passed through.
DEVICE_TO_SERVER = {
    # Sent once after connecting: who am I, which UDP port I listen on for acks, and a
    # fingerprint of my cached audio clips (so the server knows whether to send new ones).
    "hello": ("device_id", "session_id", "udp_port", "sw_version", "clip_manifest_hash"),
    # A button / switch / dial event, e.g. {"event": "next_item", "ts": 123.4}.
    "control": ("event", "ts"),
    # Periodic "still alive" with counters used for the loss statistics.
    "heartbeat": ("ts", "frames_sent", "acks_received"),
    "cue_started": ("cue_id", "ts"),  # extension: lets the server log true capture-to-audio latency
}
SERVER_TO_DEVICE = {
    # Play this speech clip. priority / interrupt decide whether it may cut off what is playing.
    "cue": ("cue_id", "clip", "priority", "interrupt"),
    # Play a beep (pan = left/right balance, interval = repeat rate).
    "tone": ("freq_hz", "duration_ms", "pan", "interval_ms"),
    # Start or stop sending camera frames, and why (find / identify).
    "stream": ("on", "max_fps", "purpose"),
    # Here are audio clips you are missing (name -> data).
    "clips_sync": ("clips",),
}
# Both directions merged into one lookup table.
MESSAGE_FIELDS = {**DEVICE_TO_SERVER, **SERVER_TO_DEVICE}

# Importance of a cue, lowest to highest. A higher-priority clip may interrupt a lower one;
# the device's audio player uses these numbers to decide.
PRIORITIES = {"info": 0, "guidance": 1, "confirmation": 2, "system": 3}


def manifest_hash(manifest: dict[str, str]) -> str:
    """Short hash of a clip manifest (name -> sha256), compared in hello to skip clip sync.

    The names are sorted first so the same set of clips always gives the same hash regardless
    of the order they were listed in. Only the first 16 hex characters are kept; that is
    plenty to tell two manifests apart.
    """
    h = hashlib.sha256()
    for name in sorted(manifest):
        h.update(f"{name}:{manifest[name]}\n".encode())
    return h.hexdigest()[:16]


def encode(msg_type: str, **fields: Any) -> str:
    """Build a WebSocket message as JSON text, e.g. ``encode("control", event="enter", ts=1.0)``.

    The message is validated first, so a programming mistake (missing field) is caught on the
    sending side. ``separators=(",", ":")`` removes spaces to keep the text compact.
    """
    msg = {"type": msg_type, **fields}
    validate(msg)
    return json.dumps(msg, separators=(",", ":"))


def decode(text: str | bytes) -> dict[str, Any]:
    """Parse received JSON text into a dictionary and check it is a valid message.

    Raises ``ProtocolError`` if the text is not JSON or the message is malformed.
    """
    try:
        msg = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # "from e" keeps the original error attached for debugging.
        raise ProtocolError(f"not JSON: {e}") from e
    validate(msg)
    return msg


def validate(msg: Any) -> None:
    """Check that ``msg`` is a dictionary with a known ``type`` and all its required fields.

    Returns nothing on success; raises ``ProtocolError`` describing the problem otherwise.
    """
    if not isinstance(msg, dict) or "type" not in msg:
        raise ProtocolError("message must be an object with a 'type' field")
    fields = MESSAGE_FIELDS.get(msg["type"])
    if fields is None:
        raise ProtocolError(f"unknown message type {msg['type']!r}")
    # Collect every required field name that is absent from the message.
    missing = [f for f in fields if f not in msg]
    if missing:
        raise ProtocolError(f"{msg['type']} message missing {missing}")
