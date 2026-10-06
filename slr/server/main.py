"""Server orchestration: UDP frames in -> detector -> guidance -> cues out over the WebSocket."""

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
from slr.common.config import ServerConfig, load_classes
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
    def __init__(self, cfg: ServerConfig, detector: Detector | None = None, debug_view: bool | None = None,
                 on_debug_key: Callable[[str], None] | None = None, tick_hz: float = 10.0):
        self.cfg = cfg
        self.detector = detector
        self.want_debug = cfg.debug_view.enabled if debug_view is None else debug_view
        self.on_debug_key = on_debug_key
        self.tick_hz = tick_hz
        self.ws: ServerConnection | None = None
        self.device: dict = {}
        self.clock_offset_s = 0.0          # server_clock - device_clock, from heartbeats
        self.loss: float | None = None
        self._hb_prev: tuple[int, int] | None = None
        self._frame_times: list[float] = []
        self._tasks: list[asyncio.Task] = []
        self.out: asyncio.Queue[dict] = asyncio.Queue()
        self.debug = None
        self.http: HttpServer | None = None
        self.advertiser = None
        self.ready = asyncio.Event()
        self.prerendered = threading.Event()

    # ------------------------------------------------------------------ setup

    async def start(self) -> None:
        cfg = self.cfg
        self.loop = asyncio.get_running_loop()
        self.classes = load_classes(cfg.path(cfg.model.classes_file))
        self.tts = make_tts(cfg.tts, cfg)
        log.info("TTS engine: %s", self.tts.name)
        self.catalog = PhraseCatalog.load(cfg.root / "config" / "phrases.yaml", self.classes,
                                          voice_tag(cfg.tts, self.tts))
        self.clips = ClipStore(cfg.path(cfg.paths.clips_dir), self.tts)
        self.shopping = ShoppingList(cfg.path(cfg.paths.shopping_list), self.classes)
        for name in self.shopping.unknown_items():
            log.warning("shopping list item %r is not a class of the loaded model", name)
        self.session_log = SessionLog(cfg.path(cfg.paths.logs_dir))
        self.core = ServerCore(cfg.guidance, self.catalog, self.shopping, self.session_log)
        self.session_log.write("server_start", version=__version__, model=cfg.model.weights,
                               backend=cfg.model.backend)

        self.det_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector")
        if self.detector is None:
            self.detector = await self.loop.run_in_executor(self.det_pool, load_detector, cfg)
        if hasattr(self.detector, "warmup"):
            await self.loop.run_in_executor(self.det_pool, self.detector.warmup)
        model_names = set(getattr(self.detector, "names", []))
        missing = [c for c in self.classes if model_names and c not in model_names]
        if missing:
            log.warning("classes file lists %d classes the model does not know, e.g. %s",
                        len(missing), missing[:5])

        # pre-render list phrases now, every class in the background (FR-60)
        threading.Thread(target=self._prerender, name="prerender", daemon=True).start()

        self.receiver = FrameReceiver(self._process_frame)
        await self.receiver.start(cfg.network.host, cfg.network.udp_port)
        self.ws_server = await serve(self._ws_handler, cfg.network.host, cfg.network.ws_port,
                                     max_size=2 ** 20, ping_interval=2, ping_timeout=4)
        log.info("WebSocket control channel on %s:%d", cfg.network.host, cfg.network.ws_port)

        handler = make_handler(self._page_data, self._list_op_threadsafe, self.clips.manifest, self.clips.path)
        self.http = HttpServer(cfg.network.host, cfg.network.http_port, handler)
        self.http.start()
        log.info("clips + list editor on http://%s:%d/", cfg.network.host, cfg.network.http_port)

        if cfg.network.mdns:
            # registration can take ~10 s on some networks; never hold up startup for it
            threading.Thread(target=self._advertise, name="mdns", daemon=True).start()

        if self.want_debug:
            from slr.server.debug_view import DebugView

            self.debug = DebugView(cfg.debug_view.window_name, cfg.path(cfg.paths.captured_dir),
                                   cfg.guidance.conf_find, self.on_debug_key)
            self.debug.start()

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
        from slr.common.discovery import Advertiser

        n = self.cfg.network
        self.advertiser = Advertiser(n.ws_port, n.udp_port, n.http_port)

    def _prerender(self) -> None:
        try:
            n = self.clips.prerender(self.catalog, self.catalog.prerender_set(self.shopping.items,
                                                                              all_classes=False))
            n += self.clips.prerender(self.catalog, self.catalog.prerender_set(all_classes=True))
            if n:
                log.info("pre-rendered %d clips", n)
        except Exception:
            log.exception("clip pre-rendering failed")
        finally:
            self.prerendered.set()

    async def stop(self) -> None:
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
        for msg in self.core.handle(event, time.monotonic(), origin_ts):
            self.out.put_nowait(msg)
        if self.debug:
            item = self.core.state.current_item
            self.debug.update(mode=self.core.state.mode.value, target=self.core.state.target,
                              last_cue=self.core.last_cue_text, item=item.name if item else "")

    async def _ticker(self) -> None:
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
        while True:
            msg = await self.out.get()
            if msg["type"] == "cue":
                try:
                    await self.loop.run_in_executor(None, self.clips.ensure, msg["clip"], msg["text"])
                except Exception:
                    log.exception("TTS failed for %r", msg["text"])
                log.info("cue: %s", msg["text"])
            self.session_log.write(msg["type"], **{k: v for k, v in msg.items() if k != "type"})
            ws = self.ws
            if ws is None:
                continue
            try:
                await ws.send(json.dumps(msg, separators=(",", ":")))
            except websockets.ConnectionClosed:
                pass

    # ------------------------------------------------------------------ frames

    def _detect(self, jpeg: bytes):
        img = decode_jpeg(jpeg)
        if img is None:
            return None, [], 0.0
        dets, ms = timed_detect(self.detector, img)
        return img, dets, ms

    async def _process_frame(self, fr: ReceivedFrame) -> float:
        img, dets, infer_ms = await self.loop.run_in_executor(self.det_pool, self._detect, fr.jpeg)
        h = fr.header
        capture_ts = h.capture_ts_ms / 1000 + self.clock_offset_s
        self.session_log.write(
            "frame", frame_id=h.frame_id, session_id=h.session_id, purpose=h.purpose_name,
            burst_id=h.burst_id, capture_ts=round(capture_ts, 4), recv_ts=round(fr.recv_ts, 4),
            infer_ms=round(infer_ms, 1), jpeg_bytes=len(fr.jpeg), size=[h.width, h.height],
            detections=[d.to_dict() for d in dets])
        self.dispatch(g.Frame(tuple(dets), h.purpose_name, h.burst_id), origin_ts=capture_ts)
        now = time.time()
        self._frame_times = [t for t in self._frame_times if now - t < 2.0] + [now]
        if self.debug and img is not None:
            self.debug.update(image=img, detections=dets, infer_ms=infer_ms,
                              fps=len(self._frame_times) / 2.0, loss=self.loss)
        return infer_ms

    # ------------------------------------------------------------------ control channel

    async def _ws_handler(self, ws: ServerConnection) -> None:
        old, self.ws = self.ws, ws
        if old is not None:
            await old.close(reason="replaced by a newer device connection")
        log.info("device connected from %s", ws.remote_address)
        try:
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
            if self.ws is ws:
                self.ws = None
                self.core.device_disconnected()
                self.session_log.write("device_disconnected")
                log.info("device disconnected")

    async def _on_message(self, ws: ServerConnection, msg: dict) -> None:
        kind = msg["type"]
        if kind == "hello":
            self.device = msg
            self._hb_prev = None
            self.session_log.write("hello", **{k: v for k, v in msg.items() if k != "type"})
            if msg.get("clip_manifest_hash") != self.clips.manifest_hash():
                clips = [{"name": n, "sha256": s, "url": f"/clips/{n}.wav"}
                         for n, s in sorted(self.clips.manifest().items())]
                await ws.send(p.encode("clips_sync", clips=clips, manifest_hash=self.clips.manifest_hash()))
        elif kind == "control":
            self.session_log.write("control", event=msg["event"], value=msg.get("value"), device_ts=msg["ts"])
            self.dispatch(g.Control(msg["event"], msg.get("value")))
        elif kind == "heartbeat":
            if "clock_offset_ms" in msg:
                self.clock_offset_s = float(msg["clock_offset_ms"]) / 1000
            sent, received = int(msg["frames_sent"]), self.receiver.slot.stats.received
            if self._hb_prev is not None:
                ds, dr = sent - self._hb_prev[0], received - self._hb_prev[1]
                if ds > 0:
                    self.loss = max(0.0, min(1.0, 1 - dr / ds))
            self._hb_prev = (sent, received)
            self.session_log.write("heartbeat", frames_sent=sent, acks_received=msg["acks_received"],
                                   server_received=received, loss=self.loss,
                                   clock_offset_ms=msg.get("clock_offset_ms"))
        elif kind == "cue_started":
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
        return list(self.shopping.items), self.shopping.current, self.classes

    def _list_op_threadsafe(self, op: str, arg: str) -> str:
        fut = asyncio.run_coroutine_threadsafe(self._list_op(op, arg), self.loop)
        return fut.result(timeout=10)

    async def _list_op(self, op: str, arg: str) -> str:
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
        self.core.reload_list()
        threading.Thread(target=self._prerender, daemon=True).start()
        return msg


async def run_server(cfg: ServerConfig, debug_view: bool | None = None) -> None:
    app = ServerApp(cfg, debug_view=debug_view)
    await app.start()
    try:
        await asyncio.Event().wait()
    finally:
        await app.stop()
