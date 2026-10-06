"""Full loopback: real ServerApp + DeviceApp over UDP and WebSocket on localhost (FR-70 code paths).

Uses a scripted detector, beep TTS, a folder camera and the null audio backend, so it runs
anywhere without a webcam, GPU or speakers.
"""

import asyncio

import pytest

from slr.common.config import load_controls_config, load_device_config, load_server_config
from slr.device.camera import FileCamera
from slr.device.main import DeviceApp
from slr.server.main import ServerApp


async def wait_for(cond, timeout=10.0):
    end = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > end:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.05)


@pytest.fixture
def configs(tmp_path):
    scfg = load_server_config()
    scfg.network.ws_port = scfg.network.udp_port = scfg.network.http_port = 0
    scfg.network.host = "127.0.0.1"
    scfg.network.mdns = False
    scfg.tts.engines = ["beep"]
    scfg.paths.shopping_list = str(tmp_path / "list.yaml")
    scfg.paths.clips_dir = str(tmp_path / "clips")
    scfg.paths.logs_dir = str(tmp_path / "logs")
    scfg.debug_view.enabled = False
    dcfg = load_device_config(sim=True)
    dcfg.audio.backend = "null"
    dcfg.paths.clip_cache = str(tmp_path / "device_clips")
    dcfg.paths.state_dir = str(tmp_path / "state")
    ccfg = load_controls_config()
    return scfg, dcfg, ccfg


async def test_sim_pipeline_find_and_check_off(configs, frames_dir, scripted_detector):
    scfg, dcfg, ccfg = configs
    server = ServerApp(scfg, detector=scripted_detector, debug_view=False)
    await server.start()
    server.shopping.add("bottle")
    server.shopping.add("cup")
    server.core.reload_list()
    ws, udp, http = server.ports
    dcfg.server.ws_port, dcfg.server.udp_port, dcfg.server.http_port = ws, udp, http

    device = DeviceApp(dcfg, ccfg, sim=True, camera=FileCamera(str(frames_dir), 160, 120))
    task = asyncio.create_task(device.run())
    spoken = []
    try:
        await wait_for(lambda: getattr(device, "client", None) is not None and device.client.connected.is_set())
        await wait_for(lambda: server.core.state.mode.value == "idle")  # mode_on from the sim switch

        # watch the server's outgoing cues
        orig = server.core.handle

        def spy(event, now, origin_ts=None):
            msgs = orig(event, now, origin_ts)
            spoken.extend(m["text"] for m in msgs if m["type"] == "cue")
            return msgs

        server.core.handle = spy
        device._on_control_event("enter")
        await wait_for(lambda: server.core.state.mode.value == "within_reach", timeout=15)
        device._on_control_event("enter")
        await wait_for(lambda: server.core.state.mode.value == "idle")

        assert "Searching for bottle." in spoken
        assert "Bottle found." in spoken
        assert "Within reach. Bottle straight ahead." in spoken
        assert "Bottle checked off. Next: cup." in spoken
        assert server.shopping.items[0].status == "found"   # persisted
        assert device.sender.frames_sent > 5
        assert device.sender.acks_received > 0
        # the device actually received and "played" the clips (fetched over HTTP on demand)
        await wait_for(lambda: len(device.audio.history) >= 3)
        assert any(c.startswith("found-") for c in device.audio.history)
        assert not device.sender.streaming                  # streaming stops when idle (FR-54)
    finally:
        device.stopping.set()
        await asyncio.sleep(0.2)
        task.cancel()
        await server.stop()


async def test_device_reconnects_after_server_restart(configs, frames_dir, scripted_detector):
    scfg, dcfg, ccfg = configs
    server = ServerApp(scfg, detector=scripted_detector, debug_view=False)
    await server.start()
    ws, udp, http = server.ports
    dcfg.server.ws_port, dcfg.server.udp_port, dcfg.server.http_port = ws, udp, http
    dcfg.connection.reconnect_max_s = 0.5
    device = DeviceApp(dcfg, ccfg, sim=True, camera=FileCamera(str(frames_dir), 160, 120))
    task = asyncio.create_task(device.run())
    try:
        await wait_for(lambda: getattr(device, "client", None) is not None and device.client.connected.is_set())
        await server.stop()
        await wait_for(lambda: not device.client.connected.is_set())
        scfg.network.ws_port, scfg.network.udp_port, scfg.network.http_port = ws, udp, http
        server = ServerApp(scfg, detector=scripted_detector, debug_view=False)
        await server.start()
        await wait_for(lambda: device.client.connected.is_set(), timeout=10)
        await wait_for(lambda: server.core.state.mode.value == "idle")  # switch position re-sent
    finally:
        device.stopping.set()
        await asyncio.sleep(0.2)
        task.cancel()
        await server.stop()
