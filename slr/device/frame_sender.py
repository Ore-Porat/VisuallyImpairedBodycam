"""UDP frame channel, device side (PRD 7.1).

WHAT THIS FILE DOES
-------------------
Sends camera pictures to the laptop server, one at a time, over UDP. The server looks at each
picture with its object detector and sends back a tiny "ack" (acknowledgement) saying "got
frame number N at time T". This file uses those acks to pace itself and to measure the
network.

One frame in flight: send a new frame when the previous ack arrives or after ack_timeout_ms,
never faster than max_fps, always capturing a fresh frame at send time. Lost frames are skipped,
never resent. JPEG quality steps down when a frame does not fit in one datagram and back up once
frames are comfortably small again.

HOW IT FITS IN
--------------
``slr/device/main.py`` creates one ``FrameSender`` with the camera from ``camera.py``. The
server, over the WebSocket (``control_client.py``), tells it when to stream and why ("find" a
list item, "identify" what is held, or a short "burst" of frames). ``PRD 7.1`` is the section of
the product spec (PRD.pdf) that defines this channel; FR-56 is the "video weak" requirement.

WORDS YOU WILL SEE
------------------
- UDP: a network protocol like sending postcards. Each message ("datagram") is sent once with
  no delivery guarantee and no ordering. That is fine for live video: a lost picture is
  replaced by a newer one a moment later, which beats waiting to resend an old one.
- datagram: one UDP message. We keep each frame small enough to fit in ONE datagram.
- ack: the server's short reply confirming it received a frame.
- in flight: sent but not yet acknowledged.
- RTT: round-trip time, how long a frame + its ack take to go there and back.
- clock offset: the difference between the server's clock and the device's clock, estimated
  from the acks so latency can be measured end to end.
- JPEG quality: 0..100; lower = smaller file, blurrier picture.
- asyncio / async def / await: Python's way to wait for network events without blocking.
  ``await`` pauses this task and lets others run until the awaited thing is ready.
- asyncio.Event: a flag that tasks can wait on until another task sets it.
- executor / thread pool: a helper thread. Capturing and JPEG-encoding a frame takes tens of
  milliseconds, so it runs on a worker thread and the network code stays responsive.
- EMA (exponential moving average): a running average that gives recent values more weight.
"""

# Allows modern type hints like ``int | None`` on older Python versions.
from __future__ import annotations

import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from slr.common import protocol as p
from slr.common.config import StreamingConfig
from slr.device.camera import CameraError

log = logging.getLogger(__name__)


class JpegQuality:
    """Adaptive JPEG quality (pure apart from the injected encoder).

    Keeps every frame under the one-datagram size limit. If a frame is too big, re-encode at a
    lower quality (in steps). After enough consecutive small frames, nudge the quality back up.
    "Injected encoder" means the actual compression function is passed in, so tests can use a
    fake one.
    """

    def __init__(self, cfg: StreamingConfig, overhead: int = p.FRAME_HEADER.size):
        """``cfg`` is the streaming section of config/device.yaml; ``overhead`` is the size of the
        header that travels in front of the JPEG bytes in each datagram."""
        self.cfg = cfg
        self.limit = cfg.max_datagram_bytes - overhead
        self.quality = cfg.jpeg_quality_start
        self._small_streak = 0   # consecutive frames comfortably under the limit

    def encode(self, img: np.ndarray, encoder: Callable[[np.ndarray, int], bytes]) -> bytes | None:
        """Returns JPEG bytes that fit in one datagram, or None if even the minimum quality is too big."""
        c = self.cfg
        q = self.quality
        # Encode; if too big, lower the quality one step and try again, down to the minimum.
        while True:
            data = encoder(img, q)
            if len(data) <= self.limit:
                break
            if q <= c.jpeg_quality_min:
                self.quality = c.jpeg_quality_min
                return None
            q = max(c.jpeg_quality_min, q - c.jpeg_quality_step)
        # We had to drop quality this frame: restart the "small frames" count.
        if q < self.quality:
            self._small_streak = 0
        self.quality = q
        # Recovery: below the starting quality and the frame is well under the limit? Count it,
        # and after jpeg_recover_frames such frames step the quality back up one notch.
        if q < c.jpeg_quality_start and len(data) < c.jpeg_recover_ratio * self.limit:
            self._small_streak += 1
            if self._small_streak >= c.jpeg_recover_frames:
                self.quality = min(c.jpeg_quality_start, q + c.jpeg_quality_step)
                self._small_streak = 0
        else:
            self._small_streak = 0
        return data


# Inheriting from asyncio.DatagramProtocol lets asyncio call our ``connection_made``,
# ``datagram_received`` and ``error_received`` methods when UDP things happen.
class FrameSender(asyncio.DatagramProtocol):
    """Captures frames, JPEG-encodes them and sends them to the server over UDP, paced by acks."""

    def __init__(self, cfg: StreamingConfig, camera, encoder, session_id: int | None = None,
                 on_video_weak: Callable[[], None] | None = None,
                 on_camera_error: Callable[[str], None] | None = None,
                 can_warn: Callable[[], bool] = lambda: True):
        """
        - cfg: streaming settings (max fps, ack timeout, datagram size, JPEG quality rules).
        - camera: an object with ``capture()`` (see camera.py), or None if the camera failed.
        - encoder: function (image, quality) -> JPEG bytes (see camera.make_jpeg_encoder).
        - session_id: random number tagging this boot; acks for other sessions are ignored.
        - on_video_weak: called once when acks stop arriving for a while (FR-56).
        - on_camera_error: called with a reason string when capturing fails.
        - can_warn: function returning whether the "video weak" warning makes sense right now
          (main.py passes "is the WebSocket connected?"; if not, a different warning applies).
        """
        self.cfg = cfg
        self.camera = camera
        self.encoder = encoder
        # getrandbits(32) gives a random 32-bit number if the caller did not supply one.
        self.session_id = session_id if session_id is not None else random.getrandbits(32)
        self.quality = JpegQuality(cfg)
        self.on_video_weak = on_video_weak
        self.on_camera_error = on_camera_error
        self.can_warn = can_warn
        self.transport: asyncio.DatagramTransport | None = None   # the UDP socket wrapper
        self.server_addr: tuple[str, int] | None = None           # (host, port) to send to
        # One worker thread for capture + encode, so they never block the network loop.
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera")

        # Streaming state, controlled by the server's "stream" messages.
        self.streaming = False
        self.purpose = p.PURPOSE_FIND          # why we are streaming (find / identify / ...)
        self.max_fps = cfg.max_fps
        self.burst_left: int | None = None     # frames still to send in a burst, or None
        self.burst_id = 0
        self.frame_id = 0                      # increments per frame sent
        self._in_flight: int | None = None     # frame_id awaiting an ack, or None
        self._last_send = 0.0
        # Smoothed capture + JPEG-encode time. The sender starts work this long before the
        # next send deadline so max_fps includes that work instead of adding it on top.
        self._capture_encode_s = 0.0
        self._send_wall: dict[int, float] = {}   # frame_id -> wall-clock time it was sent
        self._wake = asyncio.Event()           # set when streaming turns on
        self._ack = asyncio.Event()            # set when the in-flight frame is acked
        self._stream_started = 0.0
        self._weak_warned = False

        # Statistics reported in heartbeats and used by `slr bench`.
        self.frames_sent = 0
        self.acks_received = 0
        self.last_ack_at: float | None = None
        self.clock_offset_ms: float | None = None   # server_clock - device_clock
        self.rtt_ms: float | None = None
        self.dropped_too_big = 0
        self.last_jpeg_bytes = 0
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------ lifecycle

    async def start(self, server_host: str, server_port: int) -> None:
        """Open the UDP socket and start the two background tasks (sender loop and watchdog)."""
        loop = asyncio.get_running_loop()
        self.server_addr = (server_host, server_port)
        # Bind to any local address, port 0 = "pick a free port for me".
        await loop.create_datagram_endpoint(lambda: self, local_addr=("0.0.0.0", 0))
        self._tasks = [asyncio.create_task(self._run(), name="frame-sender"),
                       asyncio.create_task(self._watchdog(), name="video-watchdog")]

    @property
    def local_port(self) -> int:
        """The UDP port we are listening on for acks (told to the server in "hello")."""
        return self.transport.get_extra_info("sockname")[1] if self.transport else 0

    def close(self) -> None:
        """Stop the tasks, close the socket, stop the worker thread."""
        for t in self._tasks:
            t.cancel()
        if self.transport:
            self.transport.close()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def connection_made(self, transport) -> None:
        """asyncio calls this once the UDP socket is ready."""
        self.transport = transport

    def error_received(self, exc) -> None:
        """asyncio calls this for socket-level errors; they are not fatal for UDP."""
        log.debug("UDP error: %s", exc)

    # ------------------------------------------------------------------ control

    def set_stream(self, on: bool, max_fps: float | None = None, purpose: str = "find",
                   burst: int | None = None, burst_id: int = 0) -> None:
        """Start or stop streaming, as commanded by the server.

        - on: True to stream, False to stop.
        - max_fps: frame-rate cap for this request (None = the configured default).
        - purpose: "find", "identify", ... (encoded into each frame header for the server).
        - burst: send exactly this many frames then stop (used for identify); None = continuous.
        - burst_id: the server's number for this burst, echoed in the frame headers.
        """
        if on and self.camera is None:
            if self.on_camera_error:
                self.on_camera_error("no camera")
            return
        was = self.streaming
        self.streaming = on
        if not on:
            self.burst_left = None
            return
        self.purpose = p.PURPOSE_CODES.get(purpose, p.PURPOSE_FIND)
        self.max_fps = max_fps or self.cfg.max_fps
        self.burst_left = burst
        self.burst_id = burst_id if burst else 0
        if not was or burst:
            self._stream_started = time.monotonic()
            self._weak_warned = False
            self._in_flight = None  # a new request starts immediately
        # Wake up the sender loop, which sleeps while streaming is off.
        self._wake.set()

    # ------------------------------------------------------------------ acks

    def datagram_received(self, data: bytes, addr) -> None:
        """asyncio calls this for every UDP packet we receive: the server's acks."""
        try:
            ack = p.unpack_ack(data)
        except p.ProtocolError:
            return
        if ack.session_id != self.session_id:
            return
        now_wall = time.time()
        self.acks_received += 1
        self.last_ack_at = time.monotonic()
        self._weak_warned = False
        # If we remember when this frame was sent, update the RTT and clock-offset estimates.
        sent = self._send_wall.pop(ack.frame_id, None)
        if sent is not None:
            rtt = (now_wall - sent) * 1000
            # The server stamped its clock when it got the frame; compare with the midpoint of
            # our send and receive times (``* 500`` = average of the two, times 1000 for ms).
            offset = ack.server_ts_ms - (sent + now_wall) * 500
            # Exponential moving averages: 80% old estimate, 20% new measurement.
            self.rtt_ms = rtt if self.rtt_ms is None else 0.8 * self.rtt_ms + 0.2 * rtt
            self.clock_offset_ms = offset if self.clock_offset_ms is None else \
                0.8 * self.clock_offset_ms + 0.2 * offset
        # This is the ack we were waiting for: release the sender loop.
        if ack.frame_id == self._in_flight:
            self._in_flight = None
            self._ack.set()

    # ------------------------------------------------------------------ main loop

    async def _run(self) -> None:
        """The sender loop: wait for the previous ack (or timeout), respect max_fps, capture,
        encode, send. Runs forever as a background task."""
        loop = asyncio.get_running_loop()
        while True:
            # Streaming off: sleep until set_stream wakes us.
            if not self.streaming:
                self._wake.clear()
                await self._wake.wait()
                continue
            # at most one frame in flight: wait for its ack or the ack timeout
            deadline = self._last_send + self.cfg.ack_timeout_ms / 1000
            while self._in_flight is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._in_flight = None  # lost: skip it, never resend
                    break
                self._ack.clear()
                try:
                    await asyncio.wait_for(self._ack.wait(), remaining)
                except asyncio.TimeoutError:
                    pass
            # Frame-rate pacing. The target is one frame interval after the last send.
            # Start capture/encode early enough that the work finishes at the target time;
            # otherwise a fixed sleep + capture + encode would turn 10 fps into ~7 fps.
            target = self._last_send + 1 / max(self.max_fps, 0.1)
            start = target - self._capture_encode_s
            now = time.monotonic()
            if start > now:
                await asyncio.sleep(start - now)
            if not self.streaming:
                continue
            # Capture and encode on the worker thread (both are slow, CPU-bound jobs).
            work_start = time.monotonic()
            try:
                img = await loop.run_in_executor(self.pool, self.camera.capture)
                capture_ms = int(time.time() * 1000)
                jpeg = await loop.run_in_executor(self.pool, self.quality.encode, img, self.encoder)
            except CameraError as e:
                log.error("camera error: %s", e)
                self.streaming = False
                if self.on_camera_error:
                    self.on_camera_error(str(e))
                continue
            # Smooth the estimate over several frames so one slow frame does not skew pacing.
            elapsed = time.monotonic() - work_start
            self._capture_encode_s = (elapsed if self._capture_encode_s <= 0
                                      else 0.7 * self._capture_encode_s + 0.3 * elapsed)
            if jpeg is None:
                self.dropped_too_big += 1
                log.warning("frame too large even at minimum JPEG quality; skipped")
                self._last_send = time.monotonic()
                continue
            # Capture may have finished a little early; wait for the exact send target.
            gap = target - time.monotonic()
            if gap > 0:
                await asyncio.sleep(gap)
            if not self.streaming:
                continue
            self._send(img, jpeg, capture_ms)

    def _send(self, img: np.ndarray, jpeg: bytes, capture_ms: int) -> None:
        """Wrap the JPEG in a header and send it as one UDP datagram; update the counters."""
        # ``& 0xFFFFFFFF`` wraps the counter around at 32 bits (the size of the header field).
        self.frame_id = (self.frame_id + 1) & 0xFFFFFFFF
        h, w = img.shape[:2]
        header = p.FrameHeader(self.frame_id, self.session_id, capture_ms, self.purpose,
                               self.burst_id, w, h)
        data = p.pack_frame(header, jpeg, self.cfg.max_datagram_bytes)
        if self.transport is None or self.server_addr is None:
            return
        self.transport.sendto(data, self.server_addr)
        self.frames_sent += 1
        self.last_jpeg_bytes = len(jpeg)
        self._in_flight = self.frame_id
        self._last_send = time.monotonic()
        self._send_wall[self.frame_id] = time.time()
        # Keep the send-time table small: forget the oldest entry once it is over 100 long.
        if len(self._send_wall) > 100:
            self._send_wall.pop(next(iter(self._send_wall)))
        # Count down a burst and stop automatically when it is complete.
        if self.burst_left is not None:
            self.burst_left -= 1
            if self.burst_left <= 0:
                self.streaming = False
                self.burst_left = None

    async def _watchdog(self) -> None:
        """FR-56: streaming but no ack for video_weak_s while the WebSocket is up -> warn once."""
        while True:
            await asyncio.sleep(0.25)
            if not self.streaming or self._weak_warned:
                continue
            # Measure from whichever is later: the last ack, or when streaming started.
            last = max(self.last_ack_at or 0.0, self._stream_started)
            if time.monotonic() - last >= self.cfg.video_weak_s and self.can_warn():
                self._weak_warned = True
                log.warning("no frame acks for %.1f s", self.cfg.video_weak_s)
                if self.on_video_weak:
                    self.on_video_weak()
