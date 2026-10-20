"""WebSocket control channel, device side (PRD 7.2, FR-51..53).

WHAT THIS FILE DOES
-------------------
The device and the laptop server talk over two links. This file is the device's end of the
"control" link, a WebSocket: an open two-way phone line over the Wi-Fi hotspot. Over it the
device sends button presses and heartbeats, and the server sends back cues ("say this clip"),
tones and "start/stop streaming video" commands. (Video frames themselves go over the other
link, UDP, in ``frame_sender.py``.)

Connects, says hello, sends heartbeats every 2 s, and reconnects with exponential backoff when
the link drops. Clip sync: compares the server's manifest with the local cache and downloads
only changed clips over HTTP.

HOW IT FITS IN
--------------
``slr/device/main.py`` builds one ``ControlClient`` and one ``ClipCache`` and hands the client a
few callback functions (``on_message``, ``on_connected``, ``on_disconnected``). The client runs
forever in ``run()``; main.py reacts to whatever it reports. FR-51..53 are requirement numbers
from the product spec, PRD.pdf (connection, local clips, clip sync).

WORDS YOU WILL SEE
------------------
- WebSocket: a persistent two-way connection over the network. Unlike a web page request it
  stays open, so either side can send at any moment. Messages here are small JSON texts.
- JSON: a text format for structured data, e.g. ``{"type": "cue", "clip": "pasta_found"}``.
- HTTP: the ordinary web protocol; used here only to download clip files from the server.
- heartbeat: a tiny "I'm still here" message sent every couple of seconds so both sides notice
  quickly when the link dies.
- exponential backoff: after a failed connection attempt, wait 0.5 s, then 1 s, 2 s, 4 s...
  (capped) so a server that is down is not hammered with attempts.
- asyncio / async def / await: Python's way of doing many slow things (network waits) in one
  thread. ``async def`` marks a function that can pause; ``await`` is where it pauses until the
  slow thing finishes, letting other tasks run meanwhile.
- callback: a function passed in so that this code can call it when something happens.
- manifest: a list of clip names with a checksum for each, so we can tell which have changed.
- sha256: a checksum; a fingerprint of a file's bytes. Same bytes = same fingerprint.
- cache: a local copy kept so we do not have to download the same thing again.
"""

# Allows modern type hints like ``asyncio.Task | None`` on older Python versions.
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
    """Local copy of the server's clips: data/device_clips/<name>.wav plus manifest.json.

    The server renders speech clips ("Pasta found") with a text-to-speech voice. The device
    keeps copies on its SD card so playback is instant and does not depend on the network.
    ``manifest.json`` records the sha256 checksum of every clip we hold, so a sync can skip
    clips that have not changed.
    """

    def __init__(self, directory: Path):
        """Open (or create) the cache folder and load the manifest if there is one."""
        self.dir = directory
        # parents=True creates missing parent folders; exist_ok=True means no error if present.
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        try:
            self.manifest: dict[str, str] = json.loads(self.manifest_path.read_text())
        except (OSError, ValueError):
            # No manifest yet, or a corrupt one: start empty.
            self.manifest = {}
        # forget entries whose file vanished
        self.manifest = {n: h for n, h in self.manifest.items() if self.path(n).exists()}

    def path(self, name: str) -> Path:
        """Where the WAV file for clip ``name`` lives."""
        return self.dir / f"{name}.wav"

    def has(self, name: str) -> bool:
        """True if we hold this clip (listed in the manifest AND the file really exists)."""
        return name in self.manifest and self.path(name).exists()

    def manifest_hash(self) -> str:
        """A single checksum over the whole manifest, sent in "hello" so the server can tell at
        a glance whether the device is up to date."""
        return p.manifest_hash(self.manifest)

    def store(self, name: str, sha256: str, data: bytes) -> None:
        """Save downloaded clip bytes, after checking they match the expected checksum.

        The file is first written as ``<name>.part`` and then renamed, so a crash mid-write
        can never leave a half-written WAV that looks valid.
        """
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError(f"checksum mismatch for clip {name}")
        tmp = self.path(name).with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(self.path(name))
        self.manifest[name] = sha256

    def save_manifest(self) -> None:
        """Write the manifest to disk."""
        self.manifest_path.write_text(json.dumps(self.manifest))


def http_get(url: str, timeout: float) -> bytes:
    """Download ``url`` and return the raw bytes (a plain, blocking web request)."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


class ControlClient:
    """Keeps the WebSocket to the server alive and routes messages both ways."""

    def __init__(self, cfg: DeviceConfig, session_id: int, udp_port: Callable[[], int],
                 cache: ClipCache,
                 on_message: Callable[[dict], Awaitable[None]],
                 on_connected: Callable[[bool], Awaitable[None]],
                 on_disconnected: Callable[[], Awaitable[None]],
                 heartbeat_fields: Callable[[], dict]):
        """
        - cfg: the whole device configuration (config/device.yaml).
        - session_id: a random number chosen at boot that tags this run of the device.
        - udp_port: a function returning the local UDP port the frame sender is using (a function
          rather than a number because the port is only known once the sender has started).
        - cache: the ClipCache to keep in sync with the server.
        - on_message: async function called with every decoded server message except clip sync.
        - on_connected: async function called after each successful connect; its argument is
          True when this is a RE-connect (so main.py can say "Connected" only on recovery).
        - on_disconnected: async function called when an established link drops.
        - heartbeat_fields: function returning extra numbers (frames sent, acks...) to include
          in each heartbeat for the server's statistics.
        """
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
        self.ws = None                      # the live WebSocket, or None while disconnected
        # An asyncio.Event is a flag other tasks can wait on; "set" means connected.
        self.connected = asyncio.Event()
        self.ever_connected = False
        self._sync_task: asyncio.Task | None = None

    # ``@property`` lets callers write ``client.http_base`` (no parentheses) as if it were a
    # plain attribute, while the value is computed on the fly.
    @property
    def http_base(self) -> str:
        """The server's web address for clip downloads, e.g. http://10.42.0.1:8767."""
        return f"http://{self.host}:{self.http_port}"

    async def send(self, msg_type: str, **fields) -> bool:
        """Send one JSON message to the server. Returns False if not connected.

        ``**fields`` collects any extra keyword arguments into a dict, so callers can write
        ``send("control", event="enter", ts=...)``.
        """
        ws = self.ws
        if ws is None:
            return False
        try:
            await ws.send(p.encode(msg_type, **fields))
            return True
        except websockets.ConnectionClosed:
            return False

    async def run(self) -> None:
        """Main loop, runs for the life of the program: connect, talk, reconnect on failure.

        Each pass of the outer ``while True`` is one connection attempt. Inside it we say hello,
        start the heartbeat task, and read messages until the link breaks.
        """
        delay = self.cfg.connection.reconnect_initial_s
        while True:
            try:
                # ``async with`` opens the connection and guarantees it is closed when the block
                # ends, however it ends. ping_interval/ping_timeout make the library itself detect
                # a dead link within a few seconds.
                async with connect(f"ws://{self.host}:{self.ws_port}", open_timeout=3,
                                   ping_interval=2, ping_timeout=4, max_size=2 ** 22) as ws:
                    self.ws = ws
                    # Connected, so reset the backoff delay for next time.
                    delay = self.cfg.connection.reconnect_initial_s
                    # First message: introduce ourselves (who we are, which UDP port to send acks
                    # to, which software version, and a fingerprint of our clip cache).
                    await self.send("hello", device_id=self.cfg.device_id, session_id=self.session_id,
                                    udp_port=self.udp_port(), sw_version=_version(),
                                    clip_manifest_hash=self.cache.manifest_hash())
                    self.connected.set()
                    reconnect = self.ever_connected
                    self.ever_connected = True
                    log.info("connected to server %s:%d", self.host, self.ws_port)
                    await self.on_connected(reconnect)
                    # Heartbeats run as a separate background task alongside the read loop.
                    hb = asyncio.create_task(self._heartbeat())
                    try:
                        # ``async for`` waits for each incoming message in turn.
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
                        # Whatever happened, stop the heartbeat task for this connection.
                        hb.cancel()
            except (OSError, asyncio.TimeoutError, websockets.WebSocketException) as e:
                log.debug("connection attempt failed: %s", e)
            # We get here when the connection ended or the attempt failed.
            was_connected = self.ws is not None
            self.ws = None
            self.connected.clear()
            if was_connected:
                log.warning("connection to server lost; reconnecting")
                await self.on_disconnected()
            # Wait before retrying, doubling the wait each time up to the configured maximum.
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.cfg.connection.reconnect_max_s)

    async def _heartbeat(self) -> None:
        """Every ``heartbeat_s`` seconds send a small status message to the server."""
        while True:
            await asyncio.sleep(self.cfg.connection.heartbeat_s)
            await self.send("heartbeat", ts=time.time(), **self.heartbeat_fields())

    # ------------------------------------------------------------------ clips (FR-53)

    def _start_sync(self, clips: list[dict]) -> None:
        """The server sent its clip list: start (or restart) a background download task."""
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
        self._sync_task = asyncio.create_task(self._sync(clips))

    async def _sync(self, clips: list[dict]) -> None:
        """Download every clip whose checksum differs from ours (or whose file is missing).

        ``clips`` is a list of dicts like ``{"name": ..., "sha256": ..., "url": ...}``.
        Downloads are blocking web requests, so each one is run in a worker thread via
        ``run_in_executor`` so the WebSocket keeps being serviced meanwhile.
        """
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
        """Download one clip right now (a cue arrived for a clip we don't have yet).

        Returns the local path on success, or None if the download failed.
        """
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
    """The software version string from slr/__init__.py (imported lazily to avoid import cycles)."""
    from slr import __version__

    return __version__
