"""Server orchestration: UDP frames in -> detector -> guidance -> cues out over the WebSocket.

WHAT THIS FILE DOES
-------------------
This is the laptop program that ``slr server`` (and ``slr sim``) runs. It is the "conductor":
it does not make decisions itself, it wires the other server modules together and keeps them
running at the same time:

- ``FrameReceiver``  (frame_receiver.py)  receives camera pictures from the Pi over UDP
- ``Detector``       (detector.py)        finds grocery items in each picture
- ``ServerCore``     (core.py)            runs the guidance engine and produces messages
- WebSocket server   (this file)          the two-way control line to the Pi: button events
                                          come in, cues/tones/stream commands go out
- ``ClipStore``      (clips.py)           renders speech clips the Pi will download
- ``HttpServer``     (list_web.py)        serves the clips and the list editor web page
- ``DebugView``      (debug_view.py)      the developer window
- ``SessionLog``     (session_log.py)     the JSONL diary of everything that happened

The flow for one camera frame: UDP packet arrives -> ``_process_frame`` runs the detector in a
worker thread -> ``dispatch`` feeds a ``Frame`` event to the engine -> any resulting messages are
queued -> ``_sender`` makes sure the speech clip exists and sends the message to the Pi.

WORDS YOU WILL SEE
------------------
- asyncio: Python's way to do many "wait for something" jobs in one thread. ``async def``
  marks a function that can pause at ``await`` while waiting for the network, a timer, or a
  worker thread, letting other tasks run meanwhile. ``asyncio.create_task`` starts such a
  function running in the background.
- event loop: the asyncio scheduler that juggles all those tasks.
- executor / thread pool: a worker thread for slow, blocking work (the detector, TTS). The
  event loop "runs it in the executor" and awaits the result.
- UDP: fast "postcard" network packets with no delivery guarantee (used for pictures).
- WebSocket: a persistent two-way connection, like an open phone line (used for messages).
- JSON: text format for the messages, e.g. {"type": "cue", "text": "Pasta found"}.
- heartbeat: a small message the Pi sends every couple of seconds with its counters and clock,
  so we can estimate packet loss and the clock difference between the two machines.
- mDNS: a way for the Pi to find the laptop on the network by name instead of IP address.
- callback: a function passed to someone else to be called later.
- FR-xx: requirement numbers from the product spec, ``PRD.pdf``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import websockets
from websockets.asyncio.server import ServerConnection, serve

from slr import __version__
from slr.common import protocol as p
from slr.common.config import ServerConfig, item_classes, load_classes
from slr.server import guidance as g
from slr.server.clips import ClipStore, PhraseCatalog, make_tts, voice_tag
from slr.server.core import ServerCore
from slr.server.detector import Detector, decode_jpeg, load_detector, timed_detect
from slr.server.frame_receiver import FrameReceiver, ReceivedFrame
from slr.server.list_web import HttpServer, make_handler
from slr.server.session import ListError, ShoppingList
from slr.server.session_log import SessionLog

log = logging.getLogger(__name__)


class ServerApp:
    """The whole laptop server as one object. ``start()`` brings everything up, ``stop()`` tears
    it down. In between, asyncio tasks and the receiver callbacks do all the work."""

    def __init__(self, cfg: ServerConfig, detector: Detector | None = None, debug_view: bool | None = None,
                 on_debug_key: Callable[[str], None] | None = None, tick_hz: float = 10.0):
        """cfg:          the parsed ``config/server.yaml``.
        detector:     a ready-made detector (tests pass a ScriptedDetector); None = load from cfg.
        debug_view:   force the debug window on/off; None = whatever the config says.
        on_debug_key: function called for key presses in the debug window (sim mode uses it).
        tick_hz:      how many times per second to send a Tick to the engine (for time-outs).
        """
        self.cfg = cfg
        self.detector = detector
        self.want_debug = cfg.debug_view.enabled if debug_view is None else debug_view
        self.on_debug_key = on_debug_key
        self.tick_hz = tick_hz
        # The current device connection (only one device at a time).
        self.ws: ServerConnection | None = None
        # The device's "hello" message (id, capabilities...).
        self.device: dict = {}
        self.clock_offset_s = 0.0          # server_clock - device_clock, from heartbeats
        # Estimated fraction of frames lost on the way (0..1), from heartbeat counters.
        self.loss: float | None = None
        self._hb_prev: tuple[int, int] | None = None
        # Timestamps of recently processed frames, used to compute FPS for the debug view.
        self._frame_times: list[float] = []
        self._tasks: list[asyncio.Task] = []
        # Outgoing messages wait here until ``_sender`` picks them up.
        self.out: asyncio.Queue[dict] = asyncio.Queue()
        self.debug = None
        self.http: HttpServer | None = None
        self.advertiser = None
        # Set once start() has finished (tests wait on it).
        self.ready = asyncio.Event()
        # Set once the background clip pre-rendering has finished.
        self.prerendered = threading.Event()

    # ------------------------------------------------------------------ setup

    async def start(self) -> None:
        """Load everything and open all the network ports. Called once."""
        cfg = self.cfg
        self.loop = asyncio.get_running_loop()
        # Class id -> spoken name, from config/classes.yaml.
        self.classes = load_classes(cfg.path(cfg.model.classes_file))
        self.tts = make_tts(cfg.tts, cfg)
        log.info("TTS engine: %s", self.tts.name)
        self.catalog = PhraseCatalog.load(cfg.root / "config" / "phrases.yaml", self.classes,
                                          voice_tag(cfg.tts, self.tts))
        self.clips = ClipStore(cfg.path(cfg.paths.clips_dir), self.tts)
        # The list (and its web editor) only offers real items: the hand class is left out.
        self.item_classes = item_classes(self.classes, cfg.guidance.non_item_classes)
        self.shopping = ShoppingList(cfg.path(cfg.paths.shopping_list), self.item_classes)
        for name in self.shopping.unknown_items():
            log.warning("shopping list item %r is not a class of the loaded model", name)
        self.session_log = SessionLog(cfg.path(cfg.paths.logs_dir))
        self.core = ServerCore(cfg.guidance, self.catalog, self.shopping, self.session_log)
        self.session_log.write("server_start", version=__version__, model=cfg.model.weights,
                               backend=cfg.model.backend)

        # One dedicated worker thread for the detector (loading and running a model is slow
        # and must not freeze the event loop).
        self.det_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")
        if self.detector is None:
            self.detector = await self.loop.run_in_executor(self.det_pool, load_detector, cfg)
        if hasattr(self.detector, "warmup"):
            await self.loop.run_in_executor(self.det_pool, self.detector.warmup)
        # Warn if the classes file mentions things the model cannot actually detect.
        model_names = set(getattr(self.detector, "names", []))
        missing = [c for c in self.classes if model_names and c not in model_names]
        if missing:
            log.warning("classes file lists %d classes the model does not know, e.g. %s",
                        len(missing), missing[:5])

        # pre-render list phrases now, every class in the background (FR-60)
        threading.Thread(target=self._prerender, name="prerender", daemon=True).start()

        # Open the UDP picture channel and the WebSocket control channel.
        self.receiver = FrameReceiver(self._process_frame)
        await self.receiver.start(cfg.network.host, cfg.network.udp_port)
        self.ws_server = await serve(self._ws_handler, cfg.network.host, cfg.network.ws_port,
                                     max_size=2 ** 20, ping_interval=2, ping_timeout=4)
        log.info("WebSocket control channel on %s:%d", cfg.network.host, cfg.network.ws_port)

        # The small HTTP server: clip downloads for the device + the list editor page.
        handler = make_handler(self._page_data, self._list_op_threadsafe, self.clips.manifest, self.clips.path)
        self.http = HttpServer(cfg.network.host, cfg.network.http_port, handler)
        self.http.start()
        log.info("clips + list editor on http://%s:%d/", cfg.network.host, cfg.network.http_port)

        if cfg.network.mdns:
            # registration can take ~10 s on some networks; never hold up startup for it
            threading.Thread(target=self._advertise, name="mdns", daemon=True).start()

        if self.want_debug:
            # Imported here so a headless server never needs OpenCV's GUI parts.
            from slr.server.debug_view import DebugView

            self.debug = DebugView(cfg.debug_view.window_name, cfg.path(cfg.paths.captured_dir),
                                   cfg.guidance.conf_find, self.on_debug_key)
            self.debug.start()

        # Two forever-running background tasks: the message sender and the tick generator.
        self._tasks += [asyncio.create_task(self._sender(), name="sender"),
                        asyncio.create_task(self._ticker(), name="ticker")]
        self.ready.set()

    @property
    def ports(self) -> tuple[int, int, int]:
        """Actually bound (ws, udp, http) ports; differs from config when it asked for port 0."""
        ws = self.ws_server.sockets[0].getsockname()[1]
        udp = self.receiver.transport.get_extra_info("sockname")[1]
        return ws, udp, self.http.port

    def _advertise(self) -> None:
        """Announce this server on the local network via mDNS so the Pi can find it by name."""
        from slr.common.discovery import Advertiser

        n = self.cfg.network
        self.advertiser = Advertiser(n.ws_port, n.udp_port, n.http_port)

    def _prerender(self) -> None:
        """Background thread: render speech clips for the list first, then for every class."""
        try:
            n = self.clips.prerender(self.catalog, self.catalog.prerender_set(self.shopping.items,
                                                                              all_classes=False))
            n += self.clips.prerender(self.catalog, self.catalog.prerender_set(all_classes=True))
            if n:
                log.info("pre-rendered %d clips", n)
        except Exception:
            log.exception("clip pre-rendering failed")
        finally:
            # ``finally`` runs whether or not there was an error: always signal completion.
            self.prerendered.set()

    async def stop(self) -> None:
        """Shut everything down in reverse order."""
        for t in self._tasks:
            t.cancel()
        self.receiver.close()
        self.ws_server.close()
        if self.http:
            self.http.stop()
        if self.advertiser:
            self.advertiser.close()
        if self.debug:
            self.debug.stop()
        self.session_log.write("server_stop")
        self.session_log.close()
        self.det_pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ engine plumbing

    def dispatch(self, event: g.Event, origin_ts: float | None = None) -> None:
        """Feed one event to the engine, queue its messages, refresh the debug window."""
        for msg in self.core.handle(event, time.monotonic(), origin_ts):
            self.out.put_nowait(msg)
        if self.debug:
            item = self.core.state.current_item
            self.debug.update(mode=self.core.state.mode.value, target=self.core.state.target,
                              last_cue=self.core.last_cue_text, item=item.name if item else "")

    async def _ticker(self) -> None:
        """Background task: send a Tick ``tick_hz`` times per second so time-outs can fire.

        Every 5th tick also checks whether the shopping list file was edited from the command
        line and reloads it if so.
        """
        n = 0
        while True:
            await asyncio.sleep(1 / self.tick_hz)
            self.dispatch(g.Tick())
            n += 1
            if n % 5 == 0 and self.shopping.changed_on_disk():  # CLI edited the list
                self.shopping.load()
                self.core.reload_list()
                log.info("shopping list reloaded from disk")
                threading.Thread(target=self._prerender, daemon=True).start()

    async def _sender(self) -> None:
        """Background task: take messages off the queue, render clips if needed, send to the Pi."""
        while True:
            msg = await self.out.get()
            if msg["type"] == "cue":
                try:
                    # Make sure the WAV exists (renders it in a worker thread if not cached).
                    await self.loop.run_in_executor(None, self.clips.ensure, msg["clip"], msg["text"])
                except Exception:
                    log.exception("TTS failed for %r", msg["text"])
                log.info("cue: %s", msg["text"])
            # Log every outgoing message (without the redundant "type" key).
            self.session_log.write(msg["type"], **{k: v for k, v in msg.items() if k != "type"})
            ws = self.ws
            if ws is None:
                # No device connected: the message is simply dropped.
                continue
            try:
                await ws.send(json.dumps(msg, separators=(",", ":")))
            except websockets.ConnectionClosed:
                pass

    # ------------------------------------------------------------------ frames

    def _detect(self, jpeg: bytes):
        """Runs in the detector thread: decode the JPEG and run the model. Returns
        (image, detections, inference ms); image is None if the JPEG was corrupt."""
        img = decode_jpeg(jpeg)
        if img is None:
            return None, [], 0.0
        dets, ms = timed_detect(self.detector, img)
        return img, dets, ms

    async def _process_frame(self, fr: ReceivedFrame) -> float:
        """Called by the FrameReceiver for each new picture. Returns inference time in ms."""
        img, dets, infer_ms = await self.loop.run_in_executor(self.det_pool, self._detect, fr.jpeg)
        h = fr.header
        # Convert the Pi's capture time into the laptop's clock using the heartbeat offset.
        capture_ts = h.capture_ts_ms / 1000 + self.clock_offset_s
        self.session_log.write(
            "frame", frame_id=h.frame_id, session_id=h.session_id, purpose=h.purpose_name,
            burst_id=h.burst_id, capture_ts=round(capture_ts, 4), recv_ts=round(fr.recv_ts, 4),
            infer_ms=round(infer_ms, 1), jpeg_bytes=len(fr.jpeg), size=[h.width, h.height],
            detections=[d.to_dict() for d in dets])
        self.dispatch(g.Frame(tuple(dets), h.purpose_name, h.burst_id), origin_ts=capture_ts)
        # Keep the last 2 seconds of frame times to show frames-per-second in the debug view.
        now = time.time()
        self._frame_times = [t for t in self._frame_times if now - t < 2.0] + [now]
        if self.debug and img is not None:
            self.debug.update(image=img, detections=dets, infer_ms=infer_ms,
                              fps=len(self._frame_times) / 2.0, loss=self.loss)
        return infer_ms

    # ------------------------------------------------------------------ control channel

    async def _ws_handler(self, ws: ServerConnection) -> None:
        """Runs for the lifetime of one device connection: read messages until it closes."""
        # Only one device at a time: a new connection replaces the old one.
        old, self.ws = self.ws, ws
        if old is not None:
            await old.close(reason="replaced by a newer device connection")
        log.info("device connected from %s", ws.remote_address)
        try:
            # ``async for`` waits for each incoming text message in turn.
            async for text in ws:
                try:
                    msg = p.decode(text)
                except p.ProtocolError as e:
                    log.warning("bad message from device: %s", e)
                    continue
                await self._on_message(ws, msg)
        except websockets.ConnectionClosed:
            pass
        finally:
            # Only clean up if this is still the active connection (not an already replaced one).
            if self.ws is ws:
                self.ws = None
                self.core.device_disconnected()
                self.session_log.write("device_disconnected")
                log.info("device disconnected")

    async def _on_message(self, ws: ServerConnection, msg: dict) -> None:
        """Handle one message from the device, by its "type"."""
        kind = msg["type"]
        if kind == "hello":
            # First message after connecting: who the device is and which clips it has.
            self.device = msg
            self._hb_prev = None
            self.session_log.write("hello", **{k: v for k, v in msg.items() if k != "type"})
            # If the device's clip set differs from ours, send it the full list so it can
            # download whatever is missing or outdated.
            if msg.get("clip_manifest_hash") != self.clips.manifest_hash():
                clips = [{"name": n, "sha256": s, "url": f"/clips/{n}.wav"}
                         for n, s in sorted(self.clips.manifest().items())]
                await ws.send(p.encode("clips_sync", clips=clips, manifest_hash=self.clips.manifest_hash()))
        elif kind == "control":
            # A button / switch / dial event: log it and feed it to the engine.
            self.session_log.write("control", event=msg["event"], value=msg.get("value"), device_ts=msg["ts"])
            self.dispatch(g.Control(msg["event"], msg.get("value")))
        elif kind == "heartbeat":
            # Periodic counters from the device: update the clock offset and loss estimate.
            if "clock_offset_ms" in msg:
                self.clock_offset_s = float(msg["clock_offset_ms"]) / 1000
            sent, received = int(msg["frames_sent"]), self.receiver.slot.stats.received
            if self._hb_prev is not None:
                # Compare how many frames the Pi sent vs how many we got since last heartbeat.
                ds, dr = sent - self._hb_prev[0], received - self._hb_prev[1]
                if ds > 0:
                    self.loss = max(0.0, min(1.0, 1 - dr / ds))
            self._hb_prev = (sent, received)
            self.session_log.write("heartbeat", frames_sent=sent, acks_received=msg["acks_received"],
                                   server_received=received, loss=self.loss,
                                   clock_offset_ms=msg.get("clock_offset_ms"))
        elif kind == "cue_started":
            # The device began playing a cue: compute camera-to-speech latency if we know
            # which frame triggered it.
            origin = self.core.cue_origin.pop(msg["cue_id"], None)
            latency = None
            if origin is not None:
                latency = (float(msg["ts"]) + self.clock_offset_s) - origin
                if self.debug:
                    self.debug.update(latency_ms=latency * 1000)
            self.session_log.write("cue_started", cue_id=msg["cue_id"],
                                   latency_ms=round(latency * 1000, 1) if latency is not None else None)

    # ------------------------------------------------------------------ list editor (FR-4)

    def _page_data(self):
        """Data for the web page: (items, current index, class id -> spoken name)."""
        return list(self.shopping.items), self.shopping.current, self.item_classes

    def _list_op_threadsafe(self, op: str, arg: str) -> str:
        """Called from the HTTP server's thread: run ``_list_op`` on the event loop and wait."""
        fut = asyncio.run_coroutine_threadsafe(self._list_op(op, arg), self.loop)
        return fut.result(timeout=10)

    async def _list_op(self, op: str, arg: str) -> str:
        """Apply one list edit from the web page ("add", "remove", "clear", "reset", "current").

        Returns a human-readable confirmation message shown on the page.
        """
        sp = self.shopping
        if op == "add":
            cls = sp.add(arg)
            msg = f"Added {self.classes.get(cls, cls)}."
        elif op == "remove":
            cls = sp.remove(arg)
            msg = f"Removed {self.classes.get(cls, cls)}."
        elif op == "clear":
            sp.clear()
            msg = "Removed every item."
        elif op == "reset":
            sp.reset()
            msg = "Every item is marked not found."
        elif op == "current":
            idx = int(arg)
            if not 0 <= idx < len(sp.items):
                raise ListError("no such item")
            sp.set_state(tuple(sp.items), idx)
            msg = f"Current item is now {self.classes.get(sp.items[idx].name, sp.items[idx].name)}."
        else:
            raise ListError(f"unknown list operation {op!r}")
        # Tell the engine about the new list and render any new phrases in the background.
        self.core.reload_list()
        threading.Thread(target=self._prerender, daemon=True).start()
        return msg


async def run_server(cfg: ServerConfig, debug_view: bool | None = None) -> None:
    """Entry point used by ``slr server``: start the app and run until interrupted (Ctrl+C)."""
    app = ServerApp(cfg, debug_view=debug_view)
    await app.start()
    try:
        # Wait forever on an Event nobody ever sets; Ctrl+C cancels it.
        await asyncio.Event().wait()
    finally:
        await app.stop()
