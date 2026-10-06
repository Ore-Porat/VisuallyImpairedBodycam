"""Replay (FR-73): feed a recorded video or folder of frames through the server logic with a
scripted control sequence, on a virtual clock, and output the cue timeline.

Script (YAML):

    fps: 10                # frame rate of the recording (virtual clock step)
    duration: 20           # seconds to run (default: last event + 10 s)
    list: [pasta, coffee]  # shopping list for the run (the real list is not touched)
    events:
      - {t: 0.0, event: mode_on}
      - {t: 0.5, event: enter}
      - {t: 9.0, event: enter}

Frames are only consumed while the engine has streaming switched on, just like the device.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from slr.common.config import ServerConfig, load_classes, read_yaml
from slr.server import guidance as g
from slr.server.clips import PhraseCatalog
from slr.server.core import ServerCore
from slr.server.session import ShoppingList


def run_replay(cfg: ServerConfig, source: str | Path, script: dict, detector, camera=None) -> list[dict]:
    classes = load_classes(cfg.path(cfg.model.classes_file))
    catalog = PhraseCatalog.load(cfg.root / "config" / "phrases.yaml", classes)
    fps = float(script.get("fps", 10))
    events = sorted(script.get("events", []), key=lambda e: float(e["t"]))
    duration = float(script.get("duration", (float(events[-1]["t"]) if events else 0) + 10))

    if camera is None:
        from slr.device.camera import FileCamera

        camera = FileCamera(str(source), 0, 0, resize=False)

    timeline: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        shopping = ShoppingList(Path(tmp) / "list.yaml", classes)
        for item in script.get("list", []):
            shopping.add(item)
        core = ServerCore(cfg.guidance, catalog, shopping)

        streaming = False
        purpose, burst_left, burst_id = "find", None, 0
        frame_no = 0
        dt = 1.0 / fps
        steps = int(round(duration * fps))
        ei = 0

        def apply(msgs: list[dict], t: float, prev_mode: g.Mode) -> None:
            nonlocal streaming, purpose, burst_left, burst_id
            if core.state.mode != prev_mode:
                timeline.append({"t": round(t, 2), "type": "state", "mode": core.state.mode.value})
            for m in msgs:
                rec = {"t": round(t, 2), "type": m["type"]}
                if m["type"] == "cue":
                    rec.update(key=m["key"], text=m["text"])
                elif m["type"] == "tone":
                    rec.update(pan=m["pan"], interval_ms=m["interval_ms"])
                elif m["type"] == "stream":
                    rec.update(on=m["on"], purpose=m["purpose"])
                    streaming = m["on"]
                    purpose = m["purpose"]
                    burst_left = m.get("burst")
                    burst_id = m.get("burst_id", 0)
                timeline.append(rec)

        for i in range(steps + 1):
            t = i * dt
            while ei < len(events) and float(events[ei]["t"]) <= t + 1e-9:
                ev = events[ei]
                prev = core.state.mode
                apply(core.handle(g.Control(ev["event"], ev.get("value")), t), t, prev)
                ei += 1
            if streaming:
                img = camera.capture()
                frame_no += 1
                dets = tuple(detector.detect(img))
                prev = core.state.mode
                apply(core.handle(g.Frame(dets, purpose, burst_id if purpose == "identify" else 0), t), t, prev)
                if burst_left is not None:
                    burst_left -= 1
                    if burst_left <= 0:
                        streaming, burst_left = False, None
            prev = core.state.mode
            apply(core.handle(g.Tick(), t), t, prev)
    return timeline


def load_script(path: str | Path) -> dict:
    return read_yaml(Path(path))


def format_timeline(timeline: list[dict]) -> str:
    lines = []
    for r in timeline:
        if r["type"] == "cue":
            lines.append(f"{r['t']:7.2f}s  SAY    {r['text']}")
        elif r["type"] == "state":
            lines.append(f"{r['t']:7.2f}s  STATE  {r['mode']}")
        elif r["type"] == "stream":
            lines.append(f"{r['t']:7.2f}s  STREAM {'on' if r['on'] else 'off'} ({r['purpose']})")
        elif r["type"] == "tone":
            lines.append(f"{r['t']:7.2f}s  TONE   pan={r['pan']} interval={r['interval_ms']}")
    return "\n".join(lines)
