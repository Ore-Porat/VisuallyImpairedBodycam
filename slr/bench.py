"""`slr bench` (FR-75): end-to-end latency, processed FPS and UDP frame loss (NFR-1..3).

Two ways to use it:
- live loopback: start a server and a sim device on this laptop, stream frames (webcam or a
  file) at each resolution for a few seconds, and report. Compares resolutions (PRD 7.1).
- --log: analyze a session log from a real run (Pi + hotspot) for the same numbers.

This is software used to create an intuitive gui to test the model
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
    if not values:
        return None
    vals = sorted(values)
    k = min(len(vals) - 1, max(0, int(round(q / 100 * (len(vals) - 1)))))
    return vals[k]


def analyze_log(path: Path) -> dict:
    recs = read_log(path)
    frames = [r for r in recs if r["kind"] == "frame"]
    infer = [r["infer_ms"] for r in frames]
    latency = [r["latency_ms"] for r in recs if r["kind"] == "cue_started" and r.get("latency_ms") is not None]
    e2e_frame = [(r["recv_ts"] - r["capture_ts"]) * 1000 + r["infer_ms"] for r in frames]
    hb = [r for r in recs if r["kind"] == "heartbeat"]
    loss = None
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
        full = sorted(by_sec.values())[len(by_sec) // 4:] or list(by_sec.values())
        fps = statistics.mean(full)
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
    def f(v, fmt="{:.0f}"):
        return "-" if v is None else fmt.format(v)

    out = ["| run | frames | FPS | infer p50/p90 ms | capture->decision p50/p90 ms | "
           "capture->audio p50/p90 ms | UDP loss | JPEG KB |",
           "|---|---|---|---|---|---|---|---|"]
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
    from slr.device.main import DeviceApp
    from slr.server.main import ServerApp

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scfg = scfg.model_copy(deep=True)
        scfg.network.ws_port = scfg.network.udp_port = scfg.network.http_port = 0
        scfg.network.mdns = False
        scfg.paths.shopping_list = str(tmp / "list.yaml")
        scfg.paths.logs_dir = str(tmp / "logs")
        scfg.debug_view.enabled = False
        # a long reach hold keeps the session in guidance mode, so frame-triggered cues keep coming
        scfg.guidance.reach_hold_s = 1e9
        dcfg = dcfg.model_copy(deep=True)
        dcfg.camera.width, dcfg.camera.height = width, height
        dcfg.audio.backend = "null"
        dcfg.paths.state_dir = str(tmp / "state")
        dcfg.paths.clip_cache = str(tmp / "clips")
        (tmp / "state").mkdir()
        (tmp / "state" / "first_run_done").write_text("bench")

        server = ServerApp(scfg, detector=detector, debug_view=False)
        await server.start()
        if not server.prerendered.is_set():
            print("  waiting for speech clips to finish rendering (first run only) ...")
            await asyncio.get_running_loop().run_in_executor(None, server.prerendered.wait)
        server.shopping.add(target)
        server.core.reload_list()
        ws, udp, http = server.ports
        dcfg.server.host = "127.0.0.1"
        dcfg.server.ws_port, dcfg.server.udp_port, dcfg.server.http_port = ws, udp, http
        device = DeviceApp(dcfg, ccfg, sim=True, camera=camera)
        task = asyncio.create_task(device.run())
        try:
            for _ in range(100):
                await asyncio.sleep(0.1)
                if getattr(device, "client", None) and device.client.connected.is_set():
                    break
            await asyncio.sleep(0.5)
            device._on_control_event("enter")
            await asyncio.sleep(duration)
        finally:
            device.stopping.set()
            await asyncio.sleep(0.3)
            task.cancel()
            log_path = server.session_log.path
            await server.stop()
        result = analyze_log(log_path)
    return result


def parse_resolutions(text: str) -> list[tuple[int, int]]:
    out = []
    for part in text.split(","):
        w, h = part.lower().split("x")
        out.append((int(w), int(h)))
    return out
