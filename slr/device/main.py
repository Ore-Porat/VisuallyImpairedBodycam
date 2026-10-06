"""Device app: thin client on the Pi (or in sim mode on the laptop).

Captures frames, reads controls, plays audio. All decisions live on the server.
"""

from __future__ import annotations

import asyncio
import logging
import random
import subprocess
import time
from pathlib import Path

from slr.common.config import ControlsConfig, DeviceConfig
from slr.common.protocol import PRIORITIES
from slr.device.audio import SYSTEM, AudioPlayer, QueuedCue
from slr.device.camera import CameraError, make_jpeg_encoder, open_camera
from slr.device.control_client import ClipCache, ControlClient
from slr.device.controls import KeyboardControls, make_controls
from slr.device.frame_sender import FrameSender
from slr.device.volume_dial import (DialThread, StepVolume, VolumeCurve, VolumeFilter,
                                    level_to_gain)

log = logging.getLogger(__name__)


class DeviceApp:
    def __init__(self, cfg: DeviceConfig, ccfg: ControlsConfig, sim: bool = False, camera=None):
        self.cfg = cfg
        self.ccfg = ccfg
        self.sim = sim
        self.session_id = random.getrandbits(32)  # new per boot (PRD 7.1)
        self.camera = camera
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stopping = asyncio.Event()
        self.controls = None
        self.dial: DialThread | None = None
        self.step_volume: StepVolume | None = None
        self.volume_level = cfg.volume.initial_level
        self.system_dir = cfg.path(cfg.paths.system_clips)
        self.state_dir = cfg.path(cfg.paths.state_dir)

    # ------------------------------------------------------------------ audio helpers

    def play_local(self, key: str, priority: int = SYSTEM) -> None:
        """System clips ship with the device and work with no server (FR-52)."""
        path = self.system_dir / f"{key}.wav"
        if not path.exists():
            log.warning("system clip %s missing (run `slr clips build-system`)", key)
            return
        self.audio.play(QueuedCue(clip=key, priority=priority, interrupt=True, path=path))

    def _on_cue_started(self, cue: QueuedCue) -> None:
        if cue.cue_id is not None:
            ts = time.time()
            self.loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self.client.send("cue_started", cue_id=cue.cue_id, ts=ts)))

    def _apply_volume(self, level: int) -> None:
        self.volume_level = level
        self.audio.set_gain(level_to_gain(level, self.cfg.volume.min_db, self.cfg.volume.max_db))

    def _announce_volume(self, level: int) -> None:
        self.play_local(f"volume_{level}", PRIORITIES["confirmation"])
        asyncio.ensure_future(self.client.send("control", event="volume_set", value=level, ts=time.time()))

    # ------------------------------------------------------------------ startup

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.audio = AudioPlayer(self.cfg.audio, on_started=self._on_cue_started)
        self._apply_volume(self.volume_level)
        self.play_local("starting_up")

        if self.cfg.server.host == "auto":
            from slr.common.discovery import discover

            found = await self.loop.run_in_executor(None, discover, self.cfg.server.discovery_timeout_s)
            if found:
                host, ws, udp, http = found
                self.cfg.server.host, self.cfg.server.ws_port = host, ws
                self.cfg.server.udp_port, self.cfg.server.http_port = udp, http
                log.info("found server via mDNS at %s", host)
            else:
                log.error("no server found via mDNS; set server.host in config/device.yaml")
                self.cfg.server.host = "127.0.0.1"

        if self.camera is None:
            try:
                self.camera = await self.loop.run_in_executor(None, open_camera, self.cfg.camera, self.cfg.root)
            except CameraError as e:
                log.error("camera: %s", e)
                self.play_local("camera_error")  # FR-55: keep controls and audio working

        self.cache = ClipCache(self.cfg.path(self.cfg.paths.clip_cache))
        self.sender = FrameSender(self.cfg.streaming, self.camera, make_jpeg_encoder(), self.session_id,
                                  on_video_weak=lambda: self.play_local("video_weak"),
                                  on_camera_error=self._camera_error,
                                  can_warn=lambda: self.client.connected.is_set())
        await self.sender.start(self.cfg.server.host, self.cfg.server.udp_port)
        self.client = ControlClient(self.cfg, self.session_id, lambda: self.sender.local_port, self.cache,
                                    self._on_message, self._on_connected, self._on_disconnected,
                                    self._heartbeat_fields)
        self._start_controls(volume_buttons=self._start_volume())
        client_task = asyncio.create_task(self.client.run(), name="control-client")
        tasks = [client_task]
        if self.step_volume is not None:
            tasks.append(asyncio.create_task(self._step_volume_loop(), name="volume"))
        try:
            await self.stopping.wait()
        finally:
            for t in tasks:
                t.cancel()
            self.close()

    def _camera_error(self, reason: str) -> None:
        self.loop.call_soon_threadsafe(self.play_local, "camera_error")

    def _start_volume(self) -> bool:
        """Start the dial if there is one. Returns True if volume buttons are needed (FR-46)."""
        vc = self.cfg.volume
        if vc.source == "adc":
            try:
                from slr.device.volume_dial import Ads1115Reader

                reader = Ads1115Reader(vc.adc)
                filt = VolumeFilter(VolumeCurve(vc.calibration), vc)
                self.dial = DialThread(
                    reader, filt, vc.sample_hz,
                    on_level=lambda lvl: self.loop.call_soon_threadsafe(self._apply_volume, lvl),
                    on_announce=lambda lvl: self.loop.call_soon_threadsafe(self._announce_volume, lvl))
                self.dial.start()
                return False
            except Exception as e:
                log.warning("no ADS1115 volume dial (%s); falling back to two volume buttons", e)
        self.step_volume = StepVolume(vc)
        self.step_volume.level = self.volume_level
        return vc.source != "keyboard"

    def _start_controls(self, volume_buttons: bool) -> None:
        self.controls = make_controls(self.ccfg, self._emit_threadsafe, self.sim, volume_buttons)
        self.controls.start()
        if isinstance(self.controls, KeyboardControls):
            print("Sim keys: " + self.controls.help_text())

    async def _step_volume_loop(self) -> None:
        while True:
            await asyncio.sleep(0.05)
            lvl = self.step_volume.poll(time.monotonic())
            if lvl is not None:
                self._announce_volume(lvl)

    def close(self) -> None:
        if self.controls:
            self.controls.stop()
        if self.dial:
            self.dial.stop()
        self.sender.close()
        if self.camera is not None and hasattr(self.camera, "close"):
            self.camera.close()
        self.audio.close()

    # ------------------------------------------------------------------ controls

    def _emit_threadsafe(self, event: str) -> None:
        self.loop.call_soon_threadsafe(self._on_control_event, event)

    def inject_key(self, key: str) -> None:
        """Sim: keys pressed in the server's debug window."""
        if isinstance(self.controls, KeyboardControls):
            self.controls.inject(key)

    def _on_control_event(self, event: str) -> None:
        log.info("control: %s", event)
        if event in ("volume_up", "volume_down"):
            if self.step_volume is not None:
                lvl = self.step_volume.step(1 if event == "volume_up" else -1, time.monotonic())
                if lvl is not None:
                    self._apply_volume(lvl)
            return
        if event == "shutdown":
            asyncio.ensure_future(self._shutdown())
            return
        position = getattr(getattr(self.controls, "switch", None), "position", None)
        if self.ccfg.mode_switch.toggle_latching and position == "toggle" \
                and event not in ("mode_on", "mode_off", "identify"):
            return  # FR-43: latched in TOGGLE, the device waits until the switch goes back to ON
        if not self.client.connected.is_set():
            if event != "mode_off":
                self.play_local("not_connected", PRIORITIES["confirmation"])
            return
        asyncio.ensure_future(self.client.send("control", event=event, ts=time.time()))

    async def _shutdown(self) -> None:
        self.play_local("shutting_down")
        await self.client.send("control", event="shutdown", ts=time.time())
        await self.loop.run_in_executor(None, self.audio.wait_idle, 3.0)
        cmd = self.cfg.shutdown_command
        if cmd:
            log.warning("running shutdown command: %s", " ".join(cmd))
            subprocess.run(cmd, check=False)
        self.stopping.set()

    # ------------------------------------------------------------------ server messages

    async def _on_connected(self, reconnect: bool) -> None:
        if reconnect:
            self.play_local("connected")
        # tell the server where the switch is (on boot and after every reconnect, FR-42)
        for ev in self.controls.current_events():
            await self.client.send("control", event=ev, ts=time.time())
        await self.client.send("control", event="volume_set", value=self.volume_level, ts=time.time())
        flag = self.state_dir / "first_run_done"
        if not flag.exists():  # FR-57: spoken tutorial on first boot
            await self.client.send("control", event="help", ts=time.time())
            self.state_dir.mkdir(parents=True, exist_ok=True)
            flag.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

    async def _on_disconnected(self) -> None:
        self.sender.set_stream(False)
        self.audio.stop_tone()
        self.play_local("connection_lost")

    def _heartbeat_fields(self) -> dict:
        f = {"frames_sent": self.sender.frames_sent, "acks_received": self.sender.acks_received}
        if self.sender.clock_offset_ms is not None:
            f["clock_offset_ms"] = round(self.sender.clock_offset_ms, 1)
        if self.sender.rtt_ms is not None:
            f["rtt_ms"] = round(self.sender.rtt_ms, 1)
        return f

    async def _on_message(self, msg: dict) -> None:
        kind = msg["type"]
        if kind == "cue":
            name = msg["clip"]
            path = self.cache.path(name) if self.cache.has(name) else await self.client.fetch_clip(name)
            if path is not None:
                self.audio.play(QueuedCue(clip=name, priority=int(msg["priority"]),
                                          interrupt=bool(msg["interrupt"]), cue_id=msg["cue_id"], path=path))
        elif kind == "tone":
            if msg["interval_ms"] is None:
                self.audio.stop_tone()
            else:
                self.audio.set_tone(msg["freq_hz"], msg["duration_ms"], msg["pan"], msg["interval_ms"])
        elif kind == "stream":
            self.sender.set_stream(bool(msg["on"]), msg.get("max_fps"), msg.get("purpose", "find"),
                                   msg.get("burst"), int(msg.get("burst_id") or 0))


async def run_device(cfg: DeviceConfig, ccfg: ControlsConfig, sim: bool = False) -> None:
    await DeviceApp(cfg, ccfg, sim).run()
