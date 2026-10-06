import pytest

from slr.common import protocol as p


def test_frame_round_trip():
    h = p.FrameHeader(frame_id=42, session_id=0xDEADBEEF, capture_ts_ms=1_700_000_000_123,
                      purpose=p.PURPOSE_IDENTIFY, burst_id=7, width=640, height=480)
    data = p.pack_frame(h, b"\xff\xd8jpeg\xff\xd9")
    h2, payload = p.unpack_frame(data)
    assert h2 == h
    assert payload == b"\xff\xd8jpeg\xff\xd9"
    assert h2.purpose_name == "identify"


def test_header_size_is_fixed():
    assert p.FRAME_HEADER.size == 27


def test_frame_size_limit():
    h = p.FrameHeader(1, 1, 0)
    limit = 1000
    ok = p.pack_frame(h, b"x" * (limit - p.FRAME_HEADER.size), max_bytes=limit)
    assert len(ok) == limit
    with pytest.raises(p.FrameTooLarge):
        p.pack_frame(h, b"x" * (limit - p.FRAME_HEADER.size + 1), max_bytes=limit)


@pytest.mark.parametrize("data", [b"", b"SLR1", b"XXXX" + b"\0" * 40])
def test_bad_frames_rejected(data):
    with pytest.raises(p.ProtocolError):
        p.unpack_frame(data)


def test_unknown_purpose_rejected():
    data = bytearray(p.pack_frame(p.FrameHeader(1, 1, 0), b"j"))
    data[20] = 9  # purpose byte
    with pytest.raises(p.ProtocolError):
        p.unpack_frame(bytes(data))


def test_ack_round_trip():
    ack = p.FrameAck(session_id=5, frame_id=9, infer_ms=12.5, server_ts_ms=123456789)
    assert p.unpack_ack(p.pack_ack(ack)) == ack
    with pytest.raises(p.ProtocolError):
        p.unpack_ack(b"SLRA")


def test_ws_messages_validate():
    text = p.encode("control", event="enter", ts=1.0)
    assert p.decode(text) == {"type": "control", "event": "enter", "ts": 1.0}
    with pytest.raises(p.ProtocolError):
        p.encode("cue", cue_id=1)  # missing fields
    with pytest.raises(p.ProtocolError):
        p.decode('{"type": "bogus"}')
    with pytest.raises(p.ProtocolError):
        p.decode("not json")


def test_optional_fields_pass_through():
    msg = p.decode(p.encode("stream", on=True, max_fps=10, purpose="identify", burst=5, burst_id=3))
    assert msg["burst"] == 5 and msg["burst_id"] == 3
