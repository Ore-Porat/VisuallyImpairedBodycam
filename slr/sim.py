"""Sim mode (FR-70): the server and a simulated device in one process on one laptop.

Webcam + keyboard + laptop speakers. Same code paths as the real system, including UDP and the
WebSocket over localhost; only the camera, controls and audio backends differ.
"""

from __future__ import annotations

import asyncio
import logging

from slr.common.config import ControlsConfig, DeviceConfig, ServerConfig
from slr.device.main import DeviceApp
from slr.server.main import ServerApp

log = logging.getLogger(__name__)


async def run_sim(scfg: ServerConfig, dcfg: DeviceConfig, ccfg: ControlsConfig,
                  debug_view: bool | None = None, camera=None) -> None:
    device: DeviceApp | None = None

    def on_debug_key(key: str) -> None:
        if device is not None and device.loop is not None:
            device.loop.call_soon_threadsafe(device.inject_key, key)

    scfg.network.mdns = False  # everything is on localhost
    server = ServerApp(scfg, debug_view=debug_view, on_debug_key=on_debug_key)
    await server.start()
    ws, udp, http = server.ports
    dcfg.server.host = "127.0.0.1"
    dcfg.server.ws_port, dcfg.server.udp_port, dcfg.server.http_port = ws, udp, http
    device = DeviceApp(dcfg, ccfg, sim=True, camera=camera)
    device.loop = asyncio.get_running_loop()
    print("\nSLR SLR sim running. Put items on the list with `slr list add <item>` or at "
          f"http://127.0.0.1:{http}/  -  press q to quit.\n")
    try:
        await device.run()
    finally:
        await server.stop()
