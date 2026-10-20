"""Adaptive JPEG quality (PRD 7.1) and one-in-flight pacing.

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    ``slr/device/frame_sender.py`` runs on the Pi and sends camera frames to the laptop.
      * ``JpegQuality``: a UDP packet can only hold so many bytes. If a compressed frame
        is too big, the sender lowers the JPEG quality (75 -> 65 -> 55 ...) until it
        fits, and slowly raises it again when frames become small enough.
      * ``FrameSender``: sends ONE frame at a time and waits for the laptop's "ack"
        (acknowledgement) before sending the next, so a slow Wi-Fi link never fills up
        with stale pictures. Identify mode asks for a short "burst" of N frames and then
        streaming stops by itself.
    "PRD 7.1" is a section of the product spec, PRD.pdf.

Words you will see:
    fake encoder  A stand-in for the real JPEG compressor. Its output size is a simple
                  function of the quality number, so the test can predict exactly what
                  the quality logic will do.
    async test    A test written with ``async def``; pytest-asyncio runs it in an event
                  loop so it can ``await`` real network sends and receives on localhost.
    assert        "Check that this is true; if not, the test fails."
"""

import asyncio
import socket

import numpy as np

from slr.common import protocol as p
from slr.common.config import StreamingConfig
from slr.device.frame_sender import FrameSender, JpegQuality


def fake_encoder(sizes):
    """Encoder whose output size depends only on quality: sizes[q] bytes."""
    calls = []

    # enc(image, quality) returns pretend-JPEG bytes; `calls` records every quality tried.
    def enc(img, q):
        calls.append(q)
        return b"x" * sizes(q)

    return enc, calls


def test_quality_steps_down_until_it_fits():
    """Quality drops in steps of 10 until the frame fits, then stays at that level."""
    cfg = StreamingConfig(max_datagram_bytes=10000)
    jq = JpegQuality(cfg)
    enc, calls = fake_encoder(lambda q: q * 200)  # q75 -> 15000 bytes, q45 -> 9000
    data = jq.encode(None, enc)
    assert calls == [75, 65, 55, 45]
    assert len(data) == 9000 and jq.quality == 45
    calls.clear()
    jq.encode(None, enc)
    assert calls == [45]  # keeps the lower quality for later frames


def test_quality_gives_up_below_minimum():
    """If even the minimum quality (40) is too big, the frame is skipped (returns None)."""
    jq = JpegQuality(StreamingConfig(max_datagram_bytes=1000))
    enc, calls = fake_encoder(lambda q: 5000)
    assert jq.encode(None, enc) is None
    assert calls[-1] == 40


def test_quality_recovers_when_frames_are_small_again():
    """After jpeg_recover_frames small frames in a row, quality steps back up by 10."""
    cfg = StreamingConfig(max_datagram_bytes=10000, jpeg_recover_frames=3)
    jq = JpegQuality(cfg)
    big = True
    # The lambda reads `big` each time it is called, so flipping the variable below
    # changes the encoder's behavior mid-test.
    enc, _ = fake_encoder(lambda q: q * 200 if big else 1000)
    jq.encode(None, enc)
    assert jq.quality == 45
    big = False
    for _ in range(3):
        jq.encode(None, enc)
    assert jq.quality == 55
    for _ in range(6):
        jq.encode(None, enc)
    assert jq.quality == 75


class FakeCamera:
    """A camera that always returns a small black image (64 x 48 pixels)."""

    def capture(self):
        return np.zeros((48, 64, 3), np.uint8)


async def test_one_frame_in_flight_and_burst():
    """Over a real localhost UDP socket: wait for an ack before the next frame; bursts end."""
    # Arrange: a plain UDP socket plays the laptop. Port 0 = "any free port".
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.setblocking(False)
    port = server.getsockname()[1]
    cfg = StreamingConfig(max_fps=50, ack_timeout_ms=200)
    # The encoder here always returns the same tiny fake JPEG.
    sender = FrameSender(cfg, FakeCamera(), lambda img, q: b"\xff\xd8jpeg", session_id=77)
    await sender.start("127.0.0.1", port)
    loop = asyncio.get_running_loop()
    try:
        # Act: start streaming in Find mode and grab the first frame.
        sender.set_stream(True, max_fps=50, purpose="find")
        data, addr = await asyncio.wait_for(loop.sock_recvfrom(server, 70000), 2)
        h, _ = p.unpack_frame(data)
        assert h.session_id == 77 and h.purpose == p.PURPOSE_FIND
        # no ack yet: nothing else is sent before the 200 ms ack timeout
        try:
            await asyncio.wait_for(loop.sock_recvfrom(server, 70000), 0.12)
            raise AssertionError("second frame sent while one was in flight")
        except asyncio.TimeoutError:
            pass
        # ack it: the next frame follows quickly
        server.sendto(p.pack_ack(p.FrameAck(77, h.frame_id, 5.0, 0)), addr)
        data, _ = await asyncio.wait_for(loop.sock_recvfrom(server, 70000), 0.15)
        assert p.unpack_frame(data)[0].frame_id == h.frame_id + 1
        assert sender.acks_received == 1

        # identify burst of 3 frames, acked promptly
        sender.set_stream(True, max_fps=50, purpose="identify", burst=3, burst_id=9)
        got = []
        while len(got) < 3:
            data, addr = await asyncio.wait_for(loop.sock_recvfrom(server, 70000), 1)
            hh, _ = p.unpack_frame(data)
            # A leftover Find frame may still arrive; only count Identify frames.
            if hh.purpose == p.PURPOSE_IDENTIFY:
                got.append(hh)
            server.sendto(p.pack_ack(p.FrameAck(77, hh.frame_id, 5.0, 0)), addr)
        assert all(x.burst_id == 9 for x in got)
        await asyncio.sleep(0.1)
        assert not sender.streaming  # burst finished
    finally:
        # Always release the sockets, even if an assert failed.
        sender.close()
        server.close()
