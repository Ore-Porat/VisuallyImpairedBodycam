"""WebSocket control channel, device side (PRD 7.2, FR-51..53).

Connects, says hello, sends heartbeats every 2 s, and reconnects with exponential backoff when
the link drops. Clip sync: compares the server's manifest with the local cache and downloads
only changed clips over HTTP.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Awaitable, Callable

import websockets
from websockets.asyncio.client import connect

from slr.common import protocol as p
from slr.common.config import DeviceConfig

log = logging.getLogger(__name__)


class ClipCache:
    """Local copy of the server's clips: data/device_clips/<name>.wav plus manifest.json."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        try:
            self.manifest: dict[str, str] = json.loads(self.manifest_path.read_text())
        except (OSError, ValueError):
            self.manifest = {}
        # forget entries whose file vanished
        self.manifest = {n: h for n, h in self.manifest.items() if self.path(n).exists()}

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.wav"

    def has(self, name: str) -> bool:
        return name in self.manifest and self.path(name).exists()

    def manifest_hash(self) -> str:
        return p.manifest_hash(self.manifest)

    def store(self, name: str, sha256: str, data: bytes) -> None:
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"checksum mismatch for clip {name}")
        tmp = self.path(name).with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(self.path(name))
        self.manifest[name] = sha256

    def save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest))


def http_get(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


class ControlClient:
    def __init__(self, cfg: DeviceConfig, session_id: int, udp_port: Callable[[], int],
                 cache: ClipCache,
                 on_message: Callable[[dict], Awaitable[None]],
                 on_connected: Callable[[bool], Awaitable[None]],
                 on_disconnected: Callable[[], Awaitable[None]],
                 heartbeat_fields: Callable[[], dict]):
        self.cfg = cfg
        self.session_id = session_id
        self.udp_port = udp_port
        self.cache = cache
        self.on_message = on_message
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        self.heartbeat_fields = heartbeat_fields
        self.host = cfg.server.host
        self.ws_port = cfg.server.ws_port
        self.http_port = cfg.server.http_port
        self.ws = None
        self.connected = asyncio.Event()
        self.ever_connected = False
        self._sync_task: asyncio.Task | None = None

    @property
    def http_base(self) -> str:
        return f"http://{self.host}:{self.http_port}"

    async def send(self, msg_type: str, **fields) -> bool:
        ws = self.ws
        if ws is None:
            return False
        try:
            await ws.send(p.encode(msg_type, **fields))
            return True
        except websockets.ConnectionClosed:
            return False

    async def run(self) -> None:
        delay = self.cfg.connection.reconnect_initial_s
        while True:
            try:
                async with connect(f"ws://{self.host}:{self.ws_port}", open_timeout=3,
                                   ping_interval=2, ping_timeout=4, max_size=2 ** 22) as ws:
                    self.ws = ws
                    delay = self.cfg.connection.reconnect_initial_s
                    await self.send("hello", device_id=self.cfg.device_id, session_id=self.session_id,
                                    udp_port=self.udp_port(), sw_version=_version(),
                                    clip_manifest_hash=self.cache.manifest_hash())
                    self.connected.set()
                    reconnect = self.ever_connected
                    self.ever_connected = True
                    log.info("connected to server %s:%d", self.host, self.ws_port)
                    await self.on_connected(reconnect)
                    hb = asyncio.create_task(self._heartbeat())
                    try:
                        async for text in ws:
                            try:
                                msg = p.decode(text)
                            except p.ProtocolError as e:
                                log.warning("bad message from server: %s", e)
                                continue
                            if msg["type"] == "clips_sync":
                                self._start_sync(msg["clips"])
                            else:
                                await self.on_message(msg)
                    finally:
                        hb.cancel()
            except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as e:
                log.debug("connection attempt failed: %s", e)
            was_connected = self.ws is not None
            self.ws = None
            self.connected.clear()
            if was_connected:
                log.warning("connection to server lost; reconnecting")
                await self.on_disconnected()
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.cfg.connection.reconnect_max_s)

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.cfg.connection.heartbeat_s)
            await self.send("heartbeat", ts=time.time(), **self.heartbeat_fields())

    # ------------------------------------------------------------------ clips (FR-53)

    def _start_sync(self, clips: list[dict]) -> None:
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
        self._sync_task = asyncio.create_task(self._sync(clips))

    async def _sync(self, clips: list[dict]) -> None:
        loop = asyncio.get_running_loop()
        todo = [c for c in clips if self.cache.manifest.get(c["name"]) != c["sha256"]
                or not self.cache.path(c["name"]).exists()]
        log.info("clip sync: %d of %d clips to download", len(todo), len(clips))
        for c in todo:
            try:
                data = await loop.run_in_executor(None, http_get, self.http_base + c["url"],
                                                  self.cfg.connection.http_timeout_s)
                self.cache.store(c["name"], c["sha256"], data)
            except Exception as e:
                log.warning("could not download clip %s: %s", c["name"], e)
        self.cache.save_manifest()

    async def fetch_clip(self, name: str) -> Path | None:
        """Download one clip right now (a cue arrived for a clip we don't have yet)."""
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(None, http_get, f"{self.http_base}/clips/{name}.wav",
                                              self.cfg.connection.http_timeout_s)
        except Exception as e:
            log.warning("could not fetch clip %s: %s", name, e)
            return None
        self.cache.store(name, hashlib.sha256(data).hexdigest(), data)
        self.cache.save_manifest()
        return self.cache.path(name)


def _version() -> str:
    from slr import __version__

    return __version__
