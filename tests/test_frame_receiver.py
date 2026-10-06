import asyncio
import socket

from slr.common import protocol as p
from slr.server.frame_receiver import FrameReceiver, FrameSlot, ReceivedFrame


def rf(frame_id, session=1):
    return ReceivedFrame(p.FrameHeader(frame_id, session, 0), b"jpeg", ("127.0.0.1", 1), 0.0)


def test_newest_frame_wins_while_busy():
    slot = FrameSlot()
    for i in (1, 2, 3):
        assert slot.offer(rf(i))
    assert slot.take().header.frame_id == 3
    assert slot.stats.superseded == 2
    assert slot.take() is None


def test_out_of_order_frame_dropped():
    slot = FrameSlot()
    slot.offer(rf(5))
    assert not slot.offer(rf(4))
    assert slot.take().header.frame_id == 5
    assert slot.stats.stale == 1


def test_frame_older_than_last_processed_dropped():
    slot = FrameSlot()
    slot.offer(rf(10))
    slot.take()
    assert not slot.offer(rf(9))
    assert not slot.offer(rf(10))  # duplicate
    assert slot.offer(rf(11))
    assert slot.stats.stale == 2


def test_new_session_resets_ordering():
    slot = FrameSlot()
    slot.offer(rf(100, session=1))
    slot.take()
    assert slot.offer(rf(1, session=2))  # device rebooted: frame ids restart
    assert slot.take().header.session_id == 2


async def test_receiver_processes_and_acks_over_localhost():
    seen = []

    async def process(frame):
        seen.append(frame.header.frame_id)
        return 7.5

    rx = FrameReceiver(process)
    await rx.start("127.0.0.1", 0)
    port = rx.transport.get_extra_info("sockname")[1]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2)
    try:
        sock.sendto(b"garbage", ("127.0.0.1", port))
        sock.sendto(p.pack_frame(p.FrameHeader(1, 9, 123), b"\xff\xd8"), ("127.0.0.1", port))
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, sock.recv, 100)
        ack = p.unpack_ack(data)
        assert (ack.session_id, ack.frame_id, ack.infer_ms) == (9, 1, 7.5)
        assert seen == [1]
        assert rx.slot.stats.invalid == 1
    finally:
        sock.close()
        rx.close()
