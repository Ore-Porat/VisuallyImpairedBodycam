"""Tests for the wire protocol: how the device and the laptop pack messages into bytes.

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly (with an ``AssertionError``) if the behavior is wrong.

What is being tested here?
    ``slr/common/protocol.py`` defines two kinds of messages:
      * Frames: camera pictures sent from the Pi to the laptop over UDP. Each one is a
        small fixed-size binary header (frame number, session id, timestamp, ...)
        followed by the JPEG bytes.
      * Acks and control messages: small replies and button events. Acks are binary;
        control / cue messages are JSON text sent over a WebSocket.
    These tests check that packing and then unpacking gives back exactly what we started
    with ("round trip"), and that garbage input is rejected instead of crashing the server.

Words you will see:
    assert        "Check that this is true; if not, the test fails."
    pytest.raises A block that PASSES only if the code inside raises the named error.
    parametrize   Runs the same test function once per listed input value.
    round trip    pack -> unpack should return the original data unchanged.
"""

import pytest

from slr.common import protocol as p


def test_frame_round_trip():
    """A frame header plus JPEG bytes survives being packed to bytes and unpacked again."""
    # Arrange: build a header with every field filled in, plus a tiny fake JPEG payload.
    h = p.FrameHeader(frame_id=42, session_id=0xDEADBEEF, capture_ts_ms=1_700_000_000_123,
                      purpose=p.PURPOSE_IDENTIFY, burst_id=7, width=640, height=480)
    # Act: pack to bytes (what goes over the network), then unpack.
    data = p.pack_frame(h, b"\xff\xd8jpeg\xff\xd9")
    h2, payload = p.unpack_frame(data)
    # Assert: nothing was lost or changed.
    assert h2 == h
    assert payload == b"\xff\xd8jpeg\xff\xd9"
    assert h2.purpose_name == "identify"


def test_header_size_is_fixed():
    """The binary header is always 27 bytes; the receiver relies on that to split header/JPEG."""
    assert p.FRAME_HEADER.size == 27


def test_frame_size_limit():
    """A frame that would exceed the UDP packet limit is refused with FrameTooLarge."""
    h = p.FrameHeader(1, 1, 0)
    limit = 1000
    # Exactly at the limit is fine ...
    ok = p.pack_frame(h, b"x" * (limit - p.FRAME_HEADER.size), max_bytes=limit)
    assert len(ok) == limit
    # ... one byte over is an error (the sender then lowers JPEG quality and retries).
    with pytest.raises(p.FrameTooLarge):
        p.pack_frame(h, b"x" * (limit - p.FRAME_HEADER.size + 1), max_bytes=limit)


# parametrize: this one test runs three times, once with each of the listed byte strings.
@pytest.mark.parametrize("data", [b"", b"SLR1", b"XXXX" + b"\0" * 40])
def test_bad_frames_rejected(data):
    """Empty, truncated, or wrong-magic packets raise ProtocolError instead of crashing."""
    with pytest.raises(p.ProtocolError):
        p.unpack_frame(data)


def test_unknown_purpose_rejected():
    """A header whose 'purpose' byte is not a known value (find / identify) is rejected."""
    # bytearray lets us edit individual bytes of an otherwise valid packet.
    data = bytearray(p.pack_frame(p.FrameHeader(1, 1, 0), b"j"))
    data[20] = 9  # purpose byte
    with pytest.raises(p.ProtocolError):
        p.unpack_frame(bytes(data))


def test_ack_round_trip():
    """An ack (the laptop's 'got frame N' reply) packs and unpacks without change."""
    ack = p.FrameAck(session_id=5, frame_id=9, infer_ms=12.5, server_ts_ms=123456789)
    assert p.unpack_ack(p.pack_ack(ack)) == ack
    # A packet that is only the 4-byte magic with nothing after it is invalid.
    with pytest.raises(p.ProtocolError):
        p.unpack_ack(b"SLRA")


def test_ws_messages_validate():
    """JSON control messages are checked for required fields and known types."""
    # encode builds the JSON text for a message type; decode parses and validates it.
    text = p.encode("control", event="enter", ts=1.0)
    assert p.decode(text) == {"type": "control", "event": "enter", "ts": 1.0}
    # A "cue" message needs more fields than just cue_id, so this must be refused.
    with pytest.raises(p.ProtocolError):
        p.encode("cue", cue_id=1)  # missing fields
    # Unknown message type and plain non-JSON text are both refused.
    with pytest.raises(p.ProtocolError):
        p.decode('{"type": "bogus"}')
    with pytest.raises(p.ProtocolError):
        p.decode("not json")


def test_optional_fields_pass_through():
    """Optional fields (like burst size for identify mode) are kept, not stripped."""
    msg = p.decode(p.encode("stream", on=True, max_fps=10, purpose="identify", burst=5, burst_id=3))
    assert msg["burst"] == 5 and msg["burst_id"] == 3
