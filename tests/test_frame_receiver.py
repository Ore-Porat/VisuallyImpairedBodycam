"""Tests for the laptop's UDP frame receiver (``slr/server/frame_receiver.py``).

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    Camera frames arrive from the Pi over UDP, which does not guarantee order or delivery.
    The server keeps only ONE frame waiting (a ``FrameSlot``): while the detector is busy
    with a picture, newer frames replace the waiting one, and frames that arrive late
    (older than one we already handled) are thrown away. This keeps guidance based on the
    newest picture instead of a backlog. For every frame it processes, the server sends a
    tiny "ack" back so the device knows it can send the next one.

Words you will see:
    FrameSlot     The one-frame waiting room described above.
    offer()       Try to put a frame in the slot; returns False if it was rejected.
    take()        Remove and return the waiting frame (None if empty).
    stats         Counters: superseded (replaced by a newer frame), stale (arrived too
                  late), invalid (garbage bytes).
    async test    A test written with ``async def``; pytest-asyncio runs it inside an
                  event loop so it can ``await`` network operations.
    assert        "Check that this is true; if not, the test fails."
"""

import asyncio
import socket

from slr.common import protocol as p
from slr.server.frame_receiver import FrameReceiver, FrameSlot, ReceivedFrame


def rf(frame_id, session=1):
    """Build a minimal received frame with the given frame number and session id."""
    return ReceivedFrame(p.FrameHeader(frame_id, session, 0), b"jpeg", ("127.0.0.1", 1), 0.0)


def test_newest_frame_wins_while_busy():
    """If three frames arrive before the detector is free, only the newest is kept."""
    slot = FrameSlot()
    for i in (1, 2, 3):
        assert slot.offer(rf(i))
    assert slot.take().header.frame_id == 3
    assert slot.stats.superseded == 2
    assert slot.take() is None


def test_out_of_order_frame_dropped():
    """A frame numbered lower than the one already waiting is ignored."""
    slot = FrameSlot()
    slot.offer(rf(5))
    assert not slot.offer(rf(4))
    assert slot.take().header.frame_id == 5
    assert slot.stats.stale == 1


def test_frame_older_than_last_processed_dropped():
    """Frames older than (or equal to) the last one processed are ignored too."""
    slot = FrameSlot()
    slot.offer(rf(10))
    slot.take()
    assert not slot.offer(rf(9))
    assert not slot.offer(rf(10))  # duplicate
    assert slot.offer(rf(11))
    assert slot.stats.stale == 2


def test_new_session_resets_ordering():
    """When the device reboots it starts a new session and frame numbers restart at 1."""
    slot = FrameSlot()
    slot.offer(rf(100, session=1))
    slot.take()
    assert slot.offer(rf(1, session=2))  # device rebooted: frame ids restart
    assert slot.take().header.session_id == 2


async def test_receiver_processes_and_acks_over_localhost():
    """A real UDP socket on this computer: garbage is counted, a good frame is acked."""
    seen = []

    # This stands in for the detector. It records which frames it saw and pretends the
    # detection took 7.5 ms (that number is echoed back in the ack).
    async def process(frame):
        seen.append(frame.header.frame_id)
        return 7.5

    # Start the receiver on port 0, which means "pick any free port".
    rx = FrameReceiver(process)
    await rx.start("127.0.0.1", 0)
    port = rx.transport.get_extra_info("sockname")[1]
    # A plain UDP socket plays the role of the device.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2)
    try:
        # Act: send some garbage, then one valid frame (frame 1 of session 9).
        sock.sendto(b"garbage", ("127.0.0.1", port))
        sock.sendto(p.pack_frame(p.FrameHeader(1, 9, 123), b"\xff\xd8"), ("127.0.0.1", port))
        # Wait for the ack. sock.recv blocks, so run it in a helper thread (run_in_executor)
        # to keep the async event loop free for the receiver to do its work.
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, sock.recv, 100)
        ack = p.unpack_ack(data)
        # Assert: the ack names the right frame and the processing time we reported.
        assert (ack.session_id, ack.frame_id, ack.infer_ms) == (9, 1, 7.5)
        assert seen == [1]
        assert rx.slot.stats.invalid == 1
    finally:
        # Always clean up sockets, even if an assert above failed.
        sock.close()
        rx.close()
