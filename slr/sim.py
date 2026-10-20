"""Sim mode (FR-70): the server and a simulated device in one process on one laptop.

Webcam + keyboard + laptop speakers. Same code paths as the real system, including UDP and the
WebSocket over localhost; only the camera, controls and audio backends differ.

How it fits in
--------------
Normally the product is two computers: the Raspberry Pi inside the camera body (the "device")
and a laptop (the "server") that does the object detection. Sim mode lets you develop and demo
without the Pi: both programs run inside one Python process on your laptop, talk to each other
through the laptop's own network address (localhost), and use the webcam, keyboard and speakers
instead of the Pi camera, buttons and headphones. ``slr sim`` in the terminal ends up here.

FR-70 is the requirement number for this feature in the product spec, PRD.pdf.

Words you will see
------------------
- async / await: Python's way of running several slow, waiting-heavy jobs (network, camera,
  audio) at the same time in one program. ``async def`` marks a function that can pause while
  it waits; ``await`` is where it pauses and lets other jobs run.
- event loop: the scheduler that juggles all those async jobs. ``asyncio`` is the library.
- localhost / 127.0.0.1: the network address that means "this same computer".
- UDP: a fast, no-guarantee way to send network packets (like postcards). Used for video frames.
- WebSocket: a two-way network connection that stays open (like a phone line). Used for control
  messages and audio cues.
- mDNS: a way for devices on a local network to find each other by name without a central
  server. Not needed in sim mode because everything is on one machine.
- callback: a function you hand to someone else so they can call it later when something
  happens (here: when a key is pressed in the debug window).
- thread: a second line of execution inside the program. The debug window runs on its own
  thread, so key presses must be handed to the event loop carefully.
"""

from __future__ import annotations

import asyncio
import logging

from slr.common.config import ControlsConfig, DeviceConfig, ServerConfig
from slr.device.main import DeviceApp
from slr.server.main import ServerApp

# A logger lets this file print timestamped messages via the standard logging system.
log = logging.getLogger(__name__)


async def run_sim(scfg: ServerConfig, dcfg: DeviceConfig, ccfg: ControlsConfig,
                  debug_view: bool | None = None, camera=None) -> None:
    """Start the server, then the simulated device, and run until the user quits.

    Parameters:
    - ``scfg``: the server settings loaded from config/server.yaml.
    - ``dcfg``: the device settings from config/device.yaml, with the "sim" overrides applied.
    - ``ccfg``: button / keyboard mapping from config/controls.yaml.
    - ``debug_view``: True/False to force the debug window on or off; None = use the config.
    - ``camera``: an already-created camera object (used by tests / bench); None = build one
      from the config (normally the laptop webcam).

    Returns nothing. Called by ``cmd_sim`` in cli.py, which wraps it in ``asyncio.run``.
    """
    # The device object is created later, but the callback below needs to reach it, so we
    # declare the variable first. ``DeviceApp | None`` means "a DeviceApp or nothing yet".
    device: DeviceApp | None = None

    def on_debug_key(key: str) -> None:
        """Called by the debug window (on its own thread) when a key is pressed.

        The device's keyboard handling lives on the asyncio event loop, so we cannot call it
        directly from another thread; ``call_soon_threadsafe`` queues the call safely.
        """
        if device is not None and device.loop is not None:
            device.loop.call_soon_threadsafe(device.inject_key, key)

    # Turn off network discovery: the device will be told the address directly.
    scfg.network.mdns = False  # everything is on localhost
    # Create and start the laptop-side server (detector, guidance, WebSocket + UDP listeners).
    server = ServerApp(scfg, debug_view=debug_view, on_debug_key=on_debug_key)
    await server.start()
    # The server may have been given port 0 ("pick any free port"), so ask it which ports it
    # actually got and point the device at them.
    ws, udp, http = server.ports
    dcfg.server.host = "127.0.0.1"
    dcfg.server.ws_port, dcfg.server.udp_port, dcfg.server.http_port = ws, udp, http
    # Create the simulated device: webcam, keyboard and speakers instead of Pi hardware.
    device = DeviceApp(dcfg, ccfg, sim=True, camera=camera)
    # Remember which event loop the device runs on, so on_debug_key can hand keys to it.
    device.loop = asyncio.get_running_loop()
    print("\nSLR SLR sim running. Put items on the list with `slr list add <item>` or at "
          f"http://127.0.0.1:{http}/  -  press q to quit.\n")
    # try/finally: whatever happens inside ``try`` (normal quit, error, Ctrl+C), the ``finally``
    # block always runs, so the server is always shut down cleanly.
    try:
        # This runs the device until the user presses q (or the process is interrupted).
        await device.run()
    finally:
        await server.stop()
