"""UDP frame channel, device side (PRD 7.1).

One frame in flight: send a new frame when the previous ack arrives or after ack_timeout_ms,
never faster than max_fps, always capturing a fresh frame at send time. Lost frames are skipped,
never resent. JPEG quality steps down when a frame does not fit in one datagram and back up once
frames are comfortably small again.
"""

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
    """Adaptive JPEG quality (pure apart from the injected encoder)."""

    def __init__(self, cfg: StreamingConfig, overhead: int = p.FRAME_HEADER.size):
        self.cfg = cfg
        self.limit = cfg.max_datagram_bytes - overhead
        self.quality = cfg.jpeg_quality_start
        self._small_streak = 0

    def encode(self, img: np.ndarray, encoder: Callable[[np.ndarray, int], bytes]) -> bytes | None:
        """Returns JPEG bytes that fit in one datagram, or None if even the minimum quality is too big."""
        c = self.cfg
        q = self.quality
        while True:
            data = encoder(img, q)
            if len(data) <= self.limit:
                break
            if q <= c.jpeg_quality_min:
                self.quality = c.jpeg_quality_min
                return None
            q = max(c.jpeg_quality_min, q - c.jpeg_quality_step)
        if q < self.quality:
            self._small_streak = 0
        self.quality = q
        if q < c.jpeg_quality_start and len(data) < c.jpeg_recover_ratio * self.limit:
            self._small_streak += 1
            if self._small_streak >= c.jpeg_recover_frames:
                self.quality = min(c.jpeg_quality_start, q + c.jpeg_quality_step)
                self._small_streak = 0
        else:
            self._small_streak = 0
        return data


class FrameSender(asyncio.DatagramProtocol):
    def __init__(self, cfg: StreamingConfig, camera, encoder, session_id: int | None = None,
                 on_video_weak: Callable[[], None] | None = None,
                 on_camera_error: Callable[[str], None] | None = None,
                 can_warn: Callable[[], bool] = lambda: True):
        self.cfg = cfg
        self.camera = camera
        self.encoder = encoder
        self.session_id = session_id if session_id is not None else random.getrandbits(32)
        self.quality = JpegQuality(cfg)
        self.on_video_weak = on_video_weak
        self.on_camera_error = on_camera_error
        self.can_warn = can_warn
        self.transport: asyncio.DatagramTransport | None = None
        self.server_addr: tuple[str, int] | None = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera")

        self.streaming = False
        self.purpose = p.PURPOSE_FIND
        self.max_fps = cfg.max_fps
        self.burst_left: int | None = None
        self.burst_id = 0
        self.frame_id = 0
        self._in_flight: int | None = None
        self._last_send = 0.0
        self._send_wall: dict[int, float] = {}
        self._wake = asyncio.Event()
        self._ack = asyncio.Event()
        self._stream_started = 0.0
        self._weak_warned = False

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
        loop = asyncio.get_running_loop()
        self.server_addr = (server_host, server_port)
        await loop.create_datagram_endpoint(lambda: self, local_addr=("0.0.0.0", 0))
        self._tasks = [asyncio.create_task(self._run(), name="frame-sender"),
                       asyncio.create_task(self._watchdog(), name="video-watchdog")]

    @property
    def local_port(self) -> int:
        return self.transport.get_extra_info("sockname")[1] if self.transport else 0

    def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self.transport:
            self.transport.close()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def connection_made(self, transport) -> None:
        self.transport = transport

    def error_received(self, exc) -> None:
        log.debug("UDP error: %s", exc)

    # ------------------------------------------------------------------ control

    def set_stream(self, on: bool, max_fps: float | None = None, purpose: str = "find",
                   burst: int | None = None, burst_id: int = 0) -> None:
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
        self._wake.set()

    # ------------------------------------------------------------------ acks

    def datagram_received(self, data: bytes, addr) -> None:
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
        sent = self._send_wall.pop(ack.frame_id, None)
        if sent is not None:
            rtt = (now_wall - sent) * 1000
            offset = ack.server_ts_ms - (sent + now_wall) * 500
            self.rtt_ms = rtt if self.rtt_ms is None else 0.8 * self.rtt_ms + 0.2 * rtt
            self.clock_offset_ms = offset if self.clock_offset_ms is None else \
                0.8 * self.clock_offset_ms + 0.2 * offset
        if ack.frame_id == self._in_flight:
            self._in_flight = None
            self._ack.set()

    # ------------------------------------------------------------------ main loop

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
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
            gap = self._last_send + 1 / max(self.max_fps, 0.1) - time.monotonic()
            if gap > 0:
                await asyncio.sleep(gap)
            if not self.streaming:
                continue
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
            if jpeg is None:
                self.dropped_too_big += 1
                log.warning("frame too large even at minimum JPEG quality; skipped")
                self._last_send = time.monotonic()
                continue
            self._send(img, jpeg, capture_ms)

    def _send(self, img: np.ndarray, jpeg: bytes, capture_ms: int) -> None:
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
        if len(self._send_wall) > 100:
            self._send_wall.pop(next(iter(self._send_wall)))
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
            last = max(self.last_ack_at or 0.0, self._stream_started)
            if time.monotonic() - last >= self.cfg.video_weak_s and self.can_warn():
                self._weak_warned = True
                log.warning("no frame acks for %.1f s", self.cfg.video_weak_s)
                if self.on_video_weak:
                    self.on_video_weak()
