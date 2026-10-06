"""UDP frame channel, server side (PRD 7.1).

Newest frame wins: the server keeps one pending slot. A frame older than the last one processed
(same session) is dropped, and if several frames arrive while the detector is busy only the
newest is processed. After processing, a small ack goes back to the address the frame came from.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from slr.common import protocol as p

log = logging.getLogger(__name__)


@dataclass
class ReceivedFrame:
    header: p.FrameHeader
    jpeg: bytes
    addr: tuple
    recv_ts: float  # server wall clock, seconds


@dataclass
class ReceiverStats:
    received: int = 0
    processed: int = 0
    invalid: int = 0
    stale: int = 0
    superseded: int = 0
    last_infer_ms: float = 0.0
    session_frames: dict = field(default_factory=dict)


class FrameSlot:
    """Pure newest-frame-wins bookkeeping (no sockets, so it is easy to test)."""

    def __init__(self) -> None:
        self.pending: ReceivedFrame | None = None
        self.session_id: int | None = None
        self.last_taken_id: int = -1
        self.stats = ReceiverStats()

    def offer(self, frame: ReceivedFrame) -> bool:
        """Returns True if the frame was kept as the pending frame."""
        h = frame.header
        self.stats.received += 1
        if h.session_id != self.session_id:
            # a device restart: new session, forget the old ordering
            self.session_id = h.session_id
            self.last_taken_id = -1
            if self.pending is not None:
                self.stats.superseded += 1
            self.pending = None
        if h.frame_id <= self.last_taken_id:
            self.stats.stale += 1
            return False
        if self.pending is not None:
            if h.frame_id <= self.pending.header.frame_id:
                self.stats.stale += 1
                return False
            self.stats.superseded += 1
        self.pending = frame
        return True

    def take(self) -> ReceivedFrame | None:
        frame, self.pending = self.pending, None
        if frame is not None:
            self.last_taken_id = frame.header.frame_id
        return frame


Processor = Callable[[ReceivedFrame], Awaitable[float]]  # returns inference time in ms


class FrameReceiver(asyncio.DatagramProtocol):
    def __init__(self, process: Processor):
        self.slot = FrameSlot()
        self._process = process
        self._wake = asyncio.Event()
        self.transport: asyncio.DatagramTransport | None = None
        self._task: asyncio.Task | None = None

    # asyncio protocol callbacks
    def connection_made(self, transport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        try:
            header, jpeg = p.unpack_frame(data)
        except p.ProtocolError as e:
            self.slot.stats.invalid += 1
            log.debug("dropping invalid datagram from %s: %s", addr, e)
            return
        if self.slot.offer(ReceivedFrame(header, jpeg, addr, time.time())):
            self._wake.set()

    def error_received(self, exc) -> None:  # e.g. ICMP port unreachable on Windows
        log.debug("UDP error: %s", exc)

    async def start(self, host: str, port: int) -> None:
        loop = asyncio.get_running_loop()
        await loop.create_datagram_endpoint(lambda: self, local_addr=(host, port))
        self._task = asyncio.create_task(self._run(), name="frame-processor")
        log.info("UDP frame channel listening on %s:%d", host, port)

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            frame = self.slot.take()
            if frame is None:
                continue
            try:
                infer_ms = await self._process(frame)
            except Exception:
                log.exception("frame processing failed")
                continue
            self.slot.stats.processed += 1
            self.slot.stats.last_infer_ms = infer_ms
            self.send_ack(frame, infer_ms)

    def send_ack(self, frame: ReceivedFrame, infer_ms: float) -> None:
        if self.transport is None:
            return
        ack = p.FrameAck(frame.header.session_id, frame.header.frame_id, float(infer_ms),
                         int(time.time() * 1000))
        self.transport.sendto(p.pack_ack(ack), frame.addr)

    def close(self) -> None:
        if self._task:
            self._task.cancel()
        if self.transport:
            self.transport.close()
