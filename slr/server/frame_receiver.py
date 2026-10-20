"""UDP frame channel, server side (PRD 7.1).

Newest frame wins: the server keeps one pending slot. A frame older than the last one processed
(same session) is dropped, and if several frames arrive while the detector is busy only the
newest is processed. After processing, a small ack goes back to the address the frame came from.

WHAT THIS FILE DOES
-------------------
The Raspberry Pi sends camera pictures (JPEG files) to the laptop over the network. This file
is the laptop's "mailbox" for those pictures. It:

1. listens on a UDP port for incoming packets,
2. unpacks each packet into a header (frame number, size, purpose...) and the JPEG bytes,
3. keeps only the newest picture (old ones are useless for live guidance),
4. hands that picture to a "processor" function (the detector + guidance engine in main.py),
5. sends a tiny acknowledgement ("ack") back to the Pi so it knows how things are going.

HOW IT FITS IN
--------------
Pi (``slr/device/frame_sender.py``) --UDP--> ``FrameReceiver`` here --> ``main.py`` processes it.
The packet format itself is defined in ``slr/common/protocol.py``.

WORDS YOU WILL SEE
------------------
- UDP: a network "postcard" service. Each packet (a "datagram") is sent on its own with no
  guarantee it arrives or arrives in order. It is fast and simple, which is exactly right for
  live video where a late frame is worthless anyway.
- datagram: one UDP packet.
- ack: short for acknowledgement, a small reply saying "got frame N".
- session id: a random number the Pi picks each time it starts, so the laptop can tell a
  restart apart from the previous run (frame numbers start over from 0).
- asyncio: Python's framework for doing many waiting-heavy tasks (network, timers) in one
  thread. ``async def`` marks a function that can pause with ``await`` while waiting.
- stale / superseded: a frame older than one already handled / a frame replaced by a newer one.
- dataclass: a class that is just a bundle of named fields.
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
    """One picture as it arrived: parsed header, the raw JPEG bytes, who sent it, and when."""

    header: p.FrameHeader
    jpeg: bytes
    addr: tuple
    recv_ts: float  # server wall clock, seconds


@dataclass
class ReceiverStats:
    """Counters for the debug view, the session log and ``slr bench``.

    received:   packets that parsed correctly.
    processed:  frames actually run through the detector.
    invalid:    packets that could not be parsed.
    stale:      frames older than one already seen (dropped).
    superseded: frames that were waiting but got replaced by a newer one before processing.
    """

    received: int = 0
    processed: int = 0
    invalid: int = 0
    stale: int = 0
    superseded: int = 0
    last_infer_ms: float = 0.0
    # ``field(default_factory=dict)`` gives each instance its own empty dict.
    session_frames: dict = field(default_factory=dict)


class FrameSlot:
    """Pure newest-frame-wins bookkeeping (no sockets, so it is easy to test).

    Think of a single parking space: a new frame parks there, kicking out whoever was waiting.
    ``take`` empties the space when the processor is ready for the next frame.
    """

    def __init__(self) -> None:
        """Start empty, with no session seen yet."""
        self.pending: ReceivedFrame | None = None
        self.session_id: int | None = None
        # Highest frame id handed to the processor so far in this session (-1 = none yet).
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
            # Arrived after a newer frame was already processed: too old to be useful.
            self.stats.stale += 1
            return False
        if self.pending is not None:
            if h.frame_id <= self.pending.header.frame_id:
                # Older than the frame already waiting.
                self.stats.stale += 1
                return False
            # The waiting frame is replaced by this newer one.
            self.stats.superseded += 1
        self.pending = frame
        return True

    def take(self) -> ReceivedFrame | None:
        """Remove and return the waiting frame (or None), remembering its id."""
        # Swap in one line: grab the pending frame and clear the slot.
        frame, self.pending = self.pending, None
        if frame is not None:
            self.last_taken_id = frame.header.frame_id
        return frame


# A "Processor" is any async function that takes a ReceivedFrame and returns a float.
Processor = Callable[[ReceivedFrame], Awaitable[float]]  # returns inference time in ms


class FrameReceiver(asyncio.DatagramProtocol):
    """The UDP listener. asyncio calls the ``*_received`` methods for us whenever data arrives.

    A background task (``_run``) waits for a wake-up signal, takes the newest frame from the
    slot, runs the processor on it and sends the ack.
    """

    def __init__(self, process: Processor):
        """process: the async function that handles one frame and returns inference time in ms."""
        self.slot = FrameSlot()
        self._process = process
        # An Event is a flag one task can set and another can wait for.
        self._wake = asyncio.Event()
        self.transport: asyncio.DatagramTransport | None = None
        self._task: asyncio.Task | None = None

    # asyncio protocol callbacks
    def connection_made(self, transport) -> None:
        """Called by asyncio once the socket is open; keep the transport to send acks later."""
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        """Called by asyncio for every incoming UDP packet. Parse it and offer it to the slot."""
        try:
            header, jpeg = p.unpack_frame(data)
        except p.ProtocolError as e:
            # Garbage or a packet from some other program: count it and move on.
            self.slot.stats.invalid += 1
            log.debug("dropping invalid datagram from %s: %s", addr, e)
            return
        if self.slot.offer(ReceivedFrame(header, jpeg, addr, time.time())):
            # Wake the processing loop (``_run``).
            self._wake.set()

    def error_received(self, exc) -> None:  # e.g. ICMP port unreachable on Windows
        """Called by asyncio on a socket error. Harmless for UDP, so just log it."""
        log.debug("UDP error: %s", exc)

    async def start(self, host: str, port: int) -> None:
        """Open the UDP socket on host:port and start the background processing task."""
        loop = asyncio.get_running_loop()
        # ``lambda: self`` tells asyncio to use this very object as the protocol handler.
        await loop.create_datagram_endpoint(lambda: self, local_addr=(host, port))
        self._task = asyncio.create_task(self._run(), name="frame-processor")
        log.info("UDP frame channel listening on %s:%d", host, port)

    async def _run(self) -> None:
        """Forever: wait for a frame, process it, ack it. Errors in processing are logged, not fatal."""
        while True:
            await self._wake.wait()
            self._wake.clear()
            frame = self.slot.take()
            if frame is None:
                continue
            try:
                # ``await`` pauses here (letting other tasks run) until processing finishes.
                infer_ms = await self._process(frame)
            except Exception:
                log.exception("frame processing failed")
                continue
            self.slot.stats.processed += 1
            self.slot.stats.last_infer_ms = infer_ms
            self.send_ack(frame, infer_ms)

    def send_ack(self, frame: ReceivedFrame, infer_ms: float) -> None:
        """Send the small ack packet back to the address the frame came from."""
        if self.transport is None:
            return
        ack = p.FrameAck(frame.header.session_id, frame.header.frame_id, float(infer_ms),
                         int(time.time() * 1000))
        self.transport.sendto(p.pack_ack(ack), frame.addr)

    def close(self) -> None:
        """Stop the background task and close the socket."""
        if self._task:
            self._task.cancel()
        if self.transport:
            self.transport.close()
