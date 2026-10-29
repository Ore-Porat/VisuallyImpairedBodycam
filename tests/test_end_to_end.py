"""Full loopback: real ServerApp + DeviceApp over UDP and WebSocket on localhost (FR-70 code paths).

Uses a scripted detector, beep TTS, a folder camera and the null audio backend, so it runs
anywhere without a webcam, GPU or speakers.

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    This is the "whole product" test. It starts the real laptop server and the real
    device program in the same process, lets them talk over real network sockets on
    this computer (127.0.0.1), presses the buttons, and checks that the right sentences
    would have been spoken and that the shopping list was updated. Only the parts that
    need hardware are swapped for stand-ins:
      * detector  -> ScriptedDetector (pre-written detections, see tests/conftest.py)
      * camera    -> FileCamera reading the tiny frames from the frames_dir fixture
      * speech    -> "beep" text-to-speech (no voice model needed)
      * speakers  -> the "null" audio backend (records what it would have played)
    FR-70 is a requirement number from the product spec, PRD.pdf.

Words you will see:
    async test    A test written with ``async def``; pytest-asyncio runs it in an event
                  loop so the server and device can run at the same time.
    fixture       A helper pytest builds before a test (``configs``, ``frames_dir``,
                  ``scripted_detector``).
    lambda        A tiny one-line function, used here to describe a condition to wait for.
    spy           A wrapper around a real function that records what passes through it.
    assert        "Check that this is true; if not, the test fails."
"""

import asyncio

import pytest

from slr.common.config import load_controls_config, load_device_config, load_server_config
from slr.device.camera import FileCamera
from slr.device.main import DeviceApp
from slr.server.main import ServerApp


async def wait_for(cond, timeout=10.0):
    """Keep checking cond() every 50 ms until it is true, or fail after timeout seconds."""
    end = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > end:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.05)


@pytest.fixture
def configs(tmp_path):
    """Server, device and controls configs tuned for a self-contained test run."""
    scfg = load_server_config()
    # Port 0 means "pick any free port" so tests never clash with a running server.
    scfg.network.ws_port = scfg.network.udp_port = scfg.network.http_port = 0
    scfg.network.host = "127.0.0.1"
    scfg.network.mdns = False
    scfg.tts.engines = ["beep"]
    # This test checks the manual Enter flow and the on-demand stream.
    scfg.guidance.stream_in_idle = False
    scfg.guidance.auto_find_on_mode_on = False
    # Keep every file the server writes inside the throw-away temporary folder.
    scfg.paths.shopping_list = str(tmp_path / "list.yaml")
    scfg.paths.clips_dir = str(tmp_path / "clips")
    scfg.paths.logs_dir = str(tmp_path / "logs")
    scfg.debug_view.enabled = False
    # sim=True: laptop-style device (webcam/keyboard); then silence the audio entirely.
    dcfg = load_device_config(sim=True)
    dcfg.audio.backend = "null"
    dcfg.paths.clip_cache = str(tmp_path / "device_clips")
    dcfg.paths.state_dir = str(tmp_path / "state")
    ccfg = load_controls_config()
    return scfg, dcfg, ccfg


async def test_sim_pipeline_find_and_check_off(configs, frames_dir, scripted_detector):
    """Find pasta end to end: search, found, within reach, check off; clips reach the device."""
    scfg, dcfg, ccfg = configs
    # Arrange: start the server, put two items on the list, and tell the device which ports
    # the server ended up on (they were chosen at random because we asked for port 0).
    server = ServerApp(scfg, detector=scripted_detector, debug_view=False)
    await server.start()
    server.shopping.add("pasta")
    server.shopping.add("coffee")
    server.core.reload_list()
    ws, udp, http = server.ports
    dcfg.server.ws_port, dcfg.server.udp_port, dcfg.server.http_port = ws, udp, http

    # Start the device as a background task so the test can keep running alongside it.
    device = DeviceApp(dcfg, ccfg, sim=True, camera=FileCamera(str(frames_dir), 160, 120))
    task = asyncio.create_task(device.run())
    spoken = []
    try:
        # Wait until the device has connected and the server has heard "mode_on".
        await wait_for(lambda: getattr(device, "client", None) is not None and device.client.connected.is_set())
        await wait_for(lambda: server.core.state.mode.value == "idle")  # mode_on from the sim switch

        # watch the server's outgoing cues
        orig = server.core.handle

        def spy(event, now, origin_ts=None):
            msgs = orig(event, now, origin_ts)
            spoken.extend(m["text"] for m in msgs if m["type"] == "cue")
            return msgs

        server.core.handle = spy
        # Act: press the shutter (enter) to start searching. The scripted detector then shows
        # pasta far away, then close, so the server should reach "within_reach" by itself.
        device._on_control_event("enter")
        await wait_for(lambda: server.core.state.mode.value == "within_reach", timeout=15)
        # Press enter again to check the item off.
        device._on_control_event("enter")
        await wait_for(lambda: server.core.state.mode.value == "idle")

        # Assert: the right things were said, in a sensible state, and data flowed both ways.
        assert "Searching for pasta." in spoken
        assert "Pasta found." in spoken
        assert "Within reach. Pasta straight ahead." in spoken
        assert "Pasta checked off. Next: coffee." in spoken
        assert server.shopping.items[0].status == "found"   # persisted
        assert device.sender.frames_sent > 5
        assert device.sender.acks_received > 0
        # the device actually received and "played" the clips (fetched over HTTP on demand)
        await wait_for(lambda: len(device.audio.history) >= 3)
        assert any(c.startswith("found-") for c in device.audio.history)
        assert not device.sender.streaming                  # streaming stops when idle (FR-54)
    finally:
        # Shut everything down even if an assert failed, so the next test starts clean.
        device.stopping.set()
        await asyncio.sleep(0.2)
        task.cancel()
        await server.stop()


async def test_device_reconnects_after_server_restart(configs, frames_dir, scripted_detector):
    """If the laptop server restarts, the device reconnects by itself and re-sends its mode."""
    scfg, dcfg, ccfg = configs
    server = ServerApp(scfg, detector=scripted_detector, debug_view=False)
    await server.start()
    ws, udp, http = server.ports
    dcfg.server.ws_port, dcfg.server.udp_port, dcfg.server.http_port = ws, udp, http
    # Retry quickly so the test does not have to wait for the normal 4 s back-off.
    dcfg.connection.reconnect_max_s = 0.5
    device = DeviceApp(dcfg, ccfg, sim=True, camera=FileCamera(str(frames_dir), 160, 120))
    task = asyncio.create_task(device.run())
    try:
        await wait_for(lambda: getattr(device, "client", None) is not None and device.client.connected.is_set())
        # Act: stop the server, confirm the device notices, start a new server on the SAME ports.
        await server.stop()
        await wait_for(lambda: not device.client.connected.is_set())
        scfg.network.ws_port, scfg.network.udp_port, scfg.network.http_port = ws, udp, http
        server = ServerApp(scfg, detector=scripted_detector, debug_view=False)
        await server.start()
        # Assert: the device comes back and the server learns the switch is ON again.
        await wait_for(lambda: device.client.connected.is_set(), timeout=10)
        await wait_for(lambda: server.core.state.mode.value == "idle")  # switch position re-sent
    finally:
        device.stopping.set()
        await asyncio.sleep(0.2)
        task.cancel()
        await server.stop()
