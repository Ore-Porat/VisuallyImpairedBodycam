"""Device app: thin client on the Pi (or in sim mode on the laptop).

WHAT THIS FILE DOES
-------------------
This is the "conductor" of the device side. It creates every other device part (camera, audio
player, controls, volume dial, UDP frame sender, WebSocket control client), wires them together
with callbacks, and runs until the user shuts the device down.

Captures frames, reads controls, plays audio. All decisions live on the server: the device
never decides what to say or when to stream; it reports button presses and does what the
laptop tells it. That keeps the Pi Zero's workload tiny.

HOW IT FITS IN
--------------
``slr device`` (see slr/cli.py) loads config/device.yaml and config/controls.yaml and calls
``run_device`` at the bottom of this file. ``slr sim`` does the same on a laptop with the
``sim`` overrides (webcam, keyboard, speakers). Requirement numbers such as FR-52 refer to the
product spec, PRD.pdf.

WORDS YOU WILL SEE
------------------
- asyncio / event loop: Python's system for juggling many waiting jobs (network, timers) in one
  thread. ``async def`` functions can pause at ``await``; the "event loop" runs them in turns.
- task: an ``async`` function scheduled to run on the event loop alongside others.
- callback: a function handed to another part so it can call us when something happens.
- thread-safe: safe to call from a different thread. The audio, controls and dial run on their
  own threads, so anything they report must be passed to the event loop with
  ``loop.call_soon_threadsafe(...)`` rather than touching shared state directly.
- cue: a server message meaning "play this speech clip with this priority".
- system clip: a WAV that ships with the device ("Starting up", "Volume 5") and works offline.
- mDNS: a way to find the server on the local network by name instead of a fixed IP address.
- lambda: a tiny unnamed function written in place, e.g. ``lambda lvl: do(lvl)``.
"""

# Allows modern type hints like ``DialThread | None`` on older Python versions.
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
    """Everything that runs on the device, glued together. One instance per boot."""

    def __init__(self, cfg: DeviceConfig, ccfg: ControlsConfig, sim: bool = False, camera=None):
        """
        - cfg: parsed config/device.yaml.
        - ccfg: parsed config/controls.yaml.
        - sim: True when running on a laptop as a simulated device.
        - camera: an optional ready-made camera (tests pass a scripted one); normally None and
          opened from the config in ``run``.
        """
        self.cfg = cfg
        self.ccfg = ccfg
        self.sim = sim
        self.session_id = random.getrandbits(32)  # new per boot (PRD 7.1)
        self.camera = camera
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stopping = asyncio.Event()        # set when it is time to exit ``run``
        self.controls = None
        self.dial: DialThread | None = None    # the MCP3008 volume dial, if one was found
        self.step_volume: StepVolume | None = None   # fallback: +/- keys or two buttons
        self.volume_level = cfg.volume.initial_level
        self.system_dir = cfg.path(cfg.paths.system_clips)
        self.state_dir = cfg.path(cfg.paths.state_dir)

    # ------------------------------------------------------------------ audio helpers

    def play_local(self, key: str, priority: int = SYSTEM) -> None:
        """System clips ship with the device and work with no server (FR-52).

        ``key`` is the clip's file name without ".wav", e.g. "starting_up" or "volume_7".
        """
        path = self.system_dir / f"{key}.wav"
        if not path.exists():
            log.warning("system clip %s missing (run `slr clips build-system`)", key)
            return
        self.audio.play(QueuedCue(clip=key, priority=priority, interrupt=True, path=path))

    def _on_cue_started(self, cue: QueuedCue) -> None:
        """Audio player callback (from the audio thread): tell the server a cue began playing,
        so it can measure how long a cue took from camera to ear."""
        if cue.cue_id is not None:
            ts = time.time()
            # Hop from the audio thread onto the event loop, then schedule the send.
            self.loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self.client.send("cue_started", cue_id=cue.cue_id, ts=ts)))

    def _apply_volume(self, level: int) -> None:
        """Set the loudness for level 0..10 (0 = mute) without announcing it."""
        self.volume_level = level
        self.audio.set_gain(level_to_gain(level, self.cfg.volume.min_db, self.cfg.volume.max_db))

    def _announce_volume(self, level: int) -> None:
        """Say "Volume N" in the headphones and tell the server the new level."""
        self.play_local(f"volume_{level}", PRIORITIES["confirmation"])
        asyncio.ensure_future(self.client.send("control", event="volume_set", value=level, ts=time.time()))

    # ------------------------------------------------------------------ startup

    async def run(self) -> None:
        """Start everything in order, then wait until shutdown is requested.

        Order: audio first (so "Starting up" plays immediately), then find the server, open
        the camera, create the frame sender and control client, start the controls and volume,
        and finally hand over to the control client's reconnect loop.
        """
        self.loop = asyncio.get_running_loop()
        self.audio = AudioPlayer(self.cfg.audio, on_started=self._on_cue_started)
        self._apply_volume(self.volume_level)
        self.play_local("starting_up")

        # Server address "auto": look for the laptop on the network by its mDNS name.
        if self.cfg.server.host == "auto":
            from slr.common.discovery import discover

            # discover() blocks for a few seconds, so run it on a helper thread.
            found = await self.loop.run_in_executor(None, discover, self.cfg.server.discovery_timeout_s)
            if found:
                host, ws, udp, http = found
                self.cfg.server.host, self.cfg.server.ws_port = host, ws
                self.cfg.server.udp_port, self.cfg.server.http_port = udp, http
                log.info("found server via mDNS at %s", host)
            else:
                log.error("no server found via mDNS; set server.host in config/device.yaml")
                self.cfg.server.host = "127.0.0.1"

        # Open the camera (slow on the Pi, so also on a helper thread). If it fails the device
        # still runs: controls and audio keep working and "Camera error" is spoken.
        if self.camera is None:
            try:
                self.camera = await self.loop.run_in_executor(None, open_camera, self.cfg.camera, self.cfg.root)
            except CameraError as e:
                log.error("camera: %s", e)
                self.play_local("camera_error")  # FR-55: keep controls and audio working

        self.cache = ClipCache(self.cfg.path(self.cfg.paths.clip_cache))
        # The frame sender gets small callback functions for its two warnings. ``can_warn`` only
        # allows "video weak" while the WebSocket is up (otherwise "connection lost" applies).
        self.sender = FrameSender(self.cfg.streaming, self.camera, make_jpeg_encoder(), self.session_id,
                                  on_video_weak=lambda: self.play_local("video_weak"),
                                  on_camera_error=self._camera_error,
                                  can_warn=lambda: self.client.connected.is_set())
        await self.sender.start(self.cfg.server.host, self.cfg.server.udp_port)
        self.client = ControlClient(self.cfg, self.session_id, lambda: self.sender.local_port, self.cache,
                                    self._on_message, self._on_connected, self._on_disconnected,
                                    self._heartbeat_fields)
        # Start the volume first: whether the dial works decides if fallback buttons are needed.
        self._start_controls(volume_buttons=self._start_volume())
        client_task = asyncio.create_task(self.client.run(), name="control-client")
        tasks = [client_task]
        if self.step_volume is not None:
            tasks.append(asyncio.create_task(self._step_volume_loop(), name="volume"))
        try:
            # Block here until _shutdown() sets the flag.
            await self.stopping.wait()
        finally:
            for t in tasks:
                t.cancel()
            self.close()

    def _camera_error(self, reason: str) -> None:
        """Frame sender callback (may come from any thread): speak "Camera error"."""
        self.loop.call_soon_threadsafe(self.play_local, "camera_error")

    def _start_volume(self) -> bool:
        """Start the dial if there is one. Returns True if volume buttons are needed (FR-46)."""
        vc = self.cfg.volume
        if vc.source == "adc":
            try:
                from slr.device.volume_dial import Mcp3008Reader

                reader = Mcp3008Reader(vc.adc)
                filt = VolumeFilter(VolumeCurve(vc.calibration), vc)
                # The dial thread reports levels from its own thread; the lambdas hop onto the
                # event loop before touching the audio player.
                self.dial = DialThread(
                    reader, filt, vc.sample_hz,
                    on_level=lambda lvl: self.loop.call_soon_threadsafe(self._apply_volume, lvl),
                    on_announce=lambda lvl: self.loop.call_soon_threadsafe(self._announce_volume, lvl))
                self.dial.start()
                return False
            except Exception as e:
                log.warning("no MCP3008 volume dial (%s); falling back to two volume buttons", e)
        # No dial: volume changes in steps from +/- keys (sim) or two physical buttons.
        self.step_volume = StepVolume(vc)
        self.step_volume.level = self.volume_level
        return vc.source != "keyboard"

    def _start_controls(self, volume_buttons: bool) -> None:
        """Create the buttons/switch backend (GPIO or keyboard) and start it."""
        self.controls = make_controls(self.ccfg, self._emit_threadsafe, self.sim, volume_buttons)
        self.controls.start()
        if isinstance(self.controls, KeyboardControls):
            print("Sim keys: " + self.controls.help_text())

    async def _step_volume_loop(self) -> None:
        """Every 50 ms ask the step-volume helper whether a settled level should be announced."""
        while True:
            await asyncio.sleep(0.05)
            lvl = self.step_volume.poll(time.monotonic())
            if lvl is not None:
                self._announce_volume(lvl)

    def close(self) -> None:
        """Stop every part cleanly (threads, sockets, camera, audio)."""
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
        """The ``emit`` function given to the controls backend; it runs on the polling thread, so
        hand the event to the event loop."""
        self.loop.call_soon_threadsafe(self._on_control_event, event)

    def inject_key(self, key: str) -> None:
        """Sim: keys pressed in the server's debug window."""
        if isinstance(self.controls, KeyboardControls):
            self.controls.inject(key)

    def _on_control_event(self, event: str) -> None:
        """Handle one named event from a button, switch or key.

        Volume and shutdown are handled locally; everything else is forwarded to the server,
        unless we are not connected (then "Not connected" is spoken instead).
        """
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
        # getattr(obj, name, default) reads an attribute if it exists; the keyboard backend has
        # no ``switch``, so this yields None there.
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
        """Say "Shutting down", tell the server, let the clip finish, then power off the Pi."""
        self.play_local("shutting_down")
        await self.client.send("control", event="shutdown", ts=time.time())
        await self.loop.run_in_executor(None, self.audio.wait_idle, 3.0)
        # On the Pi this is ``sudo shutdown -h now``; in sim mode the list is empty (just quit).
        cmd = self.cfg.shutdown_command
        if cmd:
            log.warning("running shutdown command: %s", " ".join(cmd))
            subprocess.run(cmd, check=False)
        self.stopping.set()

    # ------------------------------------------------------------------ server messages

    async def _on_connected(self, reconnect: bool) -> None:
        """Control client callback after each successful connection to the server."""
        if reconnect:
            self.play_local("connected")
        # tell the server where the switch is (on boot and after every reconnect, FR-42)
        for ev in self.controls.current_events():
            await self.client.send("control", event=ev, ts=time.time())
        await self.client.send("control", event="volume_set", value=self.volume_level, ts=time.time())
        # A marker file remembers that the tutorial has been played once on this device.
        flag = self.state_dir / "first_run_done"
        if not flag.exists():  # FR-57: spoken tutorial on first boot
            await self.client.send("control", event="help", ts=time.time())
            self.state_dir.mkdir(parents=True, exist_ok=True)
            flag.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

    async def _on_disconnected(self) -> None:
        """Control client callback when the link drops: stop video and beeps, say so."""
        self.sender.set_stream(False)
        self.audio.stop_tone()
        self.play_local("connection_lost")

    def _heartbeat_fields(self) -> dict:
        """Numbers to include in every heartbeat so the server can log network health."""
        f = {"frames_sent": self.sender.frames_sent, "acks_received": self.sender.acks_received}
        if self.sender.clock_offset_ms is not None:
            f["clock_offset_ms"] = round(self.sender.clock_offset_ms, 1)
        if self.sender.rtt_ms is not None:
            f["rtt_ms"] = round(self.sender.rtt_ms, 1)
        return f

    async def _on_message(self, msg: dict) -> None:
        """Handle one message from the server: a cue (speech), a tone (beeps) or a stream command."""
        kind = msg["type"]
        if kind == "cue":
            name = msg["clip"]
            # Use the cached clip if we have it, otherwise download it right now.
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
    """Entry point used by the ``slr device`` and ``slr sim`` commands: build the app and run it."""
    await DeviceApp(cfg, ccfg, sim).run()
