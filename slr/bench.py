"""`slr bench` (FR-75): end-to-end latency, processed FPS and UDP frame loss (NFR-1..3).

Two ways to use it:
- live loopback: start a server and a sim device on this laptop, stream frames (webcam or a
  file) at each resolution for a few seconds, and report. Compares resolutions (PRD 7.1).
- --log: analyze a session log from a real run (Pi + hotspot) for the same numbers.

This is software used to create an intuitive gui to test the model

How it fits in
--------------
The product spec (PRD.pdf) sets speed targets: a spoken cue must arrive within half a second of
the camera seeing the item (NFR-1), the server must process at least 5 frames per second
(NFR-2), and no more than 10% of video packets may be lost (NFR-3). "FR" numbers are feature
requirements, "NFR" numbers are non-functional (performance) requirements. This file measures
those numbers so the team can prove the targets are met, and prints them as a Markdown table.

The raw material is the session log: while the server runs it writes one line of JSON per
event (frame processed, cue started, heartbeat, ...) to data/logs/. ``analyze_log`` turns one
such file into summary numbers. ``bench_once`` is the live mode: it runs a whole sim session
for a few seconds just to produce such a log, then analyzes it.

Words you will see
------------------
- FPS: frames per second, how many camera pictures the server processes each second.
- latency: delay. "capture->audio latency" is the time from the camera taking a picture to the
  speaker starting to say something about it.
- p50 / p90: percentiles. p50 is the median (half the measurements are faster); p90 means 90%
  of measurements are at least this fast. p90 is used for targets because it ignores rare
  outliers but still describes the typical worst case.
- UDP: the fast, no-guarantee network channel used for video frames (like postcards). Some
  frames can get lost, which is why loss is measured.
- JSONL: a text file with one JSON object per line. The session log uses this format.
- heartbeat: a small "I am alive" message the device sends every couple of seconds, carrying
  counters (frames sent, acks received) that let us compute packet loss.
- async / await: lets the program wait for the network and camera without freezing. See sim.py
  for a longer explanation.
- list comprehension: ``[x for x in items if ...]`` builds a new list in one line by filtering
  or transforming another list.
"""

from __future__ import annotations

import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from slr.common.config import ControlsConfig, DeviceConfig, ServerConfig
from slr.server.session_log import read_log


def pct(values: list[float], q: float) -> float | None:
    """Return the q-th percentile of ``values`` (e.g. q=90 for p90), or None if the list is empty.

    This is the simple "nearest rank" method: sort the numbers and pick the one at position
    q% of the way through. No interpolation between neighbours.
    """
    if not values:
        return None
    vals = sorted(values)
    # Index of the chosen value. min/max keep it inside the list even for q=0 or q=100.
    k = min(len(vals) - 1, max(0, int(round(q / 100 * (len(vals) - 1)))))
    return vals[k]


def analyze_log(path: Path) -> dict:
    """Read one session log file (JSONL) and compute the benchmark numbers.

    ``path``: the log file, e.g. data/logs/session_2026-10-06.jsonl.

    Returns a dictionary with frame count, FPS, inference time percentiles, end-to-end latency
    percentiles, UDP loss fraction and average JPEG size. Any number that could not be measured
    is None (printed as "-" by ``format_report``).
    """
    recs = read_log(path)
    # Each record has a "kind" saying what happened. Pull out the kinds we need.
    frames = [r for r in recs if r["kind"] == "frame"]
    # How long the neural network took on each frame, in milliseconds.
    infer = [r["infer_ms"] for r in frames]
    # True capture-to-audio latency, logged when the device reports a cue started playing.
    latency = [r["latency_ms"] for r in recs if r["kind"] == "cue_started" and r.get("latency_ms") is not None]
    # Capture-to-decision: network travel time (receive time minus capture time, seconds -> ms)
    # plus the model's inference time.
    e2e_frame = [(r["recv_ts"] - r["capture_ts"]) * 1000 + r["infer_ms"] for r in frames]
    hb = [r for r in recs if r["kind"] == "heartbeat"]
    loss = None
    # Packet loss = 1 - (frames the server got) / (frames the device sent), measured between
    # the first and last heartbeat so start-up noise is excluded.
    if len(hb) >= 2:
        sent = hb[-1]["frames_sent"] - hb[0]["frames_sent"]
        got = hb[-1]["server_received"] - hb[0]["server_received"]
        loss = max(0.0, 1 - got / sent) if sent > 0 else None
    fps = None
    if len(frames) >= 2:
        # processed FPS while streaming: count frames in 1 s windows that had any frames
        by_sec: dict[int, int] = {}
        for r in frames:
            by_sec[int(r["t"])] = by_sec.get(int(r["t"]), 0) + 1
        # Drop the slowest quarter of the seconds (partial first/last seconds, warm-up), unless
        # that would leave nothing, then average the rest.
        full = sorted(by_sec.values())[len(by_sec) // 4:] or list(by_sec.values())
        fps = statistics.mean(full)
    # Size of each compressed frame, if it was logged (lets you see the effect of resolution).
    sizes = [r["jpeg_bytes"] for r in frames if "jpeg_bytes" in r]
    return {
        "frames": len(frames),
        "fps": fps,
        "infer_ms_p50": pct(infer, 50), "infer_ms_p90": pct(infer, 90),
        "frame_to_decision_ms_p50": pct(e2e_frame, 50), "frame_to_decision_ms_p90": pct(e2e_frame, 90),
        "cue_latency_ms_p50": pct(latency, 50), "cue_latency_ms_p90": pct(latency, 90),
        "cues_measured": len(latency),
        "udp_loss": loss,
        "jpeg_kb_avg": statistics.mean(sizes) / 1024 if sizes else None,
    }


def format_report(rows: list[tuple[str, dict]]) -> str:
    """Turn one or more ``analyze_log`` results into a Markdown table.

    ``rows``: a list of (name, numbers) pairs, one per run (e.g. one per resolution).
    Returns the table as a single string, followed by a reminder of the PRD targets.
    """
    def f(v, fmt="{:.0f}"):
        """Format a number with ``fmt``, or print "-" when the value is missing (None)."""
        return "-" if v is None else fmt.format(v)

    # Table header lines (Markdown uses | to separate columns and |---| for the divider).
    out = ["| run | frames | FPS | infer p50/p90 ms | capture->decision p50/p90 ms | "
           "capture->audio p50/p90 ms | UDP loss | JPEG KB |",
           "|---|---|---|---|---|---|---|---|"]
    # One table row per run. The f-strings (f"...{value}...") fill values into the text.
    for name, r in rows:
        out.append(
            f"| {name} | {r['frames']} | {f(r['fps'], '{:.1f}')} | "
            f"{f(r['infer_ms_p50'])}/{f(r['infer_ms_p90'])} | "
            f"{f(r['frame_to_decision_ms_p50'])}/{f(r['frame_to_decision_ms_p90'])} | "
            f"{f(r['cue_latency_ms_p50'])}/{f(r['cue_latency_ms_p90'])} ({r['cues_measured']}) | "
            f"{f(r['udp_loss'] * 100 if r['udp_loss'] is not None else None, '{:.1f}%')} | "
            f"{f(r['jpeg_kb_avg'], '{:.1f}')} |")
    out.append("")
    out.append("Targets: capture->audio p90 <= 500 ms (NFR-1), FPS >= 5 (NFR-2), UDP loss <= 10% (NFR-3).")
    return "\n".join(out)


async def bench_once(scfg: ServerConfig, dcfg: DeviceConfig, ccfg: ControlsConfig, width: int, height: int,
                     duration: float, target: str, detector=None, camera=None) -> dict:
    """Run one live benchmark at one camera resolution and return its numbers.

    It starts a throw-away server and sim device (like ``slr sim``), presses "enter" to start a
    Find for ``target``, streams for ``duration`` seconds, shuts everything down, and analyzes
    the session log that was produced.

    Parameters:
    - ``scfg``, ``dcfg``, ``ccfg``: server, device and controls settings. They are copied, so
      the caller's objects are not modified.
    - ``width``, ``height``: camera resolution to test, in pixels.
    - ``duration``: how many seconds to stream.
    - ``target``: shopping list item to search for (hold one in view to get cue latency).
    - ``detector``: an already-loaded model (so it is loaded once, not once per resolution).
    - ``camera``: an already-created camera object for this resolution.

    Returns the dictionary from ``analyze_log``. Called by ``cmd_bench`` in cli.py.
    """
    # Imported here rather than at the top so that merely importing this file does not pull in
    # the heavy server/device modules (and ``--log`` mode never needs them).
    from slr.device.main import DeviceApp
    from slr.server.main import ServerApp

    # ``with tempfile.TemporaryDirectory()`` creates a scratch folder and deletes it again when
    # the block ends, so the benchmark never touches the real shopping list or logs.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Work on copies of the configs so the caller's settings stay untouched.
        scfg = scfg.model_copy(deep=True)
        # Port 0 = "let the operating system pick any free port". Avoids clashes with a real
        # server that might be running at the same time.
        scfg.network.ws_port = scfg.network.udp_port = scfg.network.http_port = 0
        scfg.network.mdns = False
        scfg.paths.shopping_list = str(tmp / "list.yaml")
        scfg.paths.logs_dir = str(tmp / "logs")
        scfg.debug_view.enabled = False
        # a long reach hold keeps the session in guidance mode, so frame-triggered cues keep coming
        scfg.guidance.reach_hold_s = 1e9
        dcfg = dcfg.model_copy(deep=True)
        dcfg.camera.width, dcfg.camera.height = width, height
        # No sound during a benchmark.
        dcfg.audio.backend = "null"
        dcfg.paths.state_dir = str(tmp / "state")
        dcfg.paths.clip_cache = str(tmp / "clips")
        (tmp / "state").mkdir()
        # Pretend the device has booted before, so it skips the first-run help speech.
        (tmp / "state" / "first_run_done").write_text("bench")

        server = ServerApp(scfg, detector=detector, debug_view=False)
        await server.start()
        # On a brand-new install the server renders speech clips in the background. Wait for
        # that to finish, otherwise the timing numbers would include clip rendering.
        if not server.prerendered.is_set():
            print("  waiting for speech clips to finish rendering (first run only) ...")
            await asyncio.get_running_loop().run_in_executor(None, server.prerendered.wait)
        # Put the target on the (temporary) shopping list and tell the guidance engine.
        server.shopping.add(target)
        server.core.reload_list()
        # Point the device at the ports the server actually got.
        ws, udp, http = server.ports
        dcfg.server.host = "127.0.0.1"
        dcfg.server.ws_port, dcfg.server.udp_port, dcfg.server.http_port = ws, udp, http
        device = DeviceApp(dcfg, ccfg, sim=True, camera=camera)
        # Start the device as a background task so this function can keep going.
        task = asyncio.create_task(device.run())
        try:
            # Wait up to 10 s (100 x 0.1 s) for the device to connect to the server.
            for _ in range(100):
                await asyncio.sleep(0.1)
                if getattr(device, "client", None) and device.client.connected.is_set():
                    break
            await asyncio.sleep(0.5)
            # Simulate pressing the "enter" button: starts the Find for the target item.
            device._on_control_event("enter")
            # Let it stream for the requested time.
            await asyncio.sleep(duration)
        finally:
            # Always tear down, even if something above failed.
            device.stopping.set()
            await asyncio.sleep(0.3)
            task.cancel()
            # Grab the log path before stopping the server (stop closes the file).
            log_path = server.session_log.path
            await server.stop()
        result = analyze_log(log_path)
    return result


def parse_resolutions(text: str) -> list[tuple[int, int]]:
    """Turn a command-line string like ``"640x480,320x240"`` into ``[(640, 480), (320, 240)]``."""
    out = []
    for part in text.split(","):
        w, h = part.lower().split("x")
        out.append((int(w), int(h)))
    return out
