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

WHAT THIS FILE DOES
-------------------
A way to test the whole decision pipeline WITHOUT the Raspberry Pi, a webcam or real time.
You give it a recording (a video file or a folder of pictures) and a short YAML script saying
which buttons were pressed at which second. It then pretends to be the device: it advances a
fake clock one frame at a time, runs the detector on each picture, pushes events through the
same ``ServerCore`` the live server uses, and collects every spoken cue, tone and mode change
with its timestamp. Run it with ``slr replay <video> --script s.yaml``.

Because the clock is simulated, the same recording and script always give the same timeline,
which makes it great for regression tests ("did my tweak to the thresholds change what gets
said?").

WORDS YOU WILL SEE
------------------
- FR-73: requirement number in the product spec, ``PRD.pdf``.
- virtual clock: time is a number we increment ourselves (``t = i * dt``), not the wall clock.
- timeline: the list of result records, each with a time ``t`` and a ``type``.
- ``nonlocal``: lets an inner function change variables that belong to the outer function.
- ``with tempfile.TemporaryDirectory()``: a scratch folder that is deleted automatically at
  the end of the ``with`` block.
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
    """Run the recording through the engine and return the timeline.

    cfg:      server configuration (guidance thresholds, class file...).
    source:   video file or folder of images.
    script:   the parsed YAML script (see the module docstring).
    detector: any detector object (real model or ``ScriptedDetector``).
    camera:   optional frame source; by default a ``FileCamera`` reading ``source``.
    Returns a list of dicts like {"t": 1.2, "type": "cue", "key": "found", "text": "..."}.
    """
    classes = load_classes(cfg.path(cfg.model.classes_file))
    catalog = PhraseCatalog.load(cfg.root / "config" / "phrases.yaml", classes)
    fps = float(script.get("fps", 10))
    # Sort the scripted button presses by time so we can walk through them in order.
    events = sorted(script.get("events", []), key=lambda e: float(e["t"]))
    duration = float(script.get("duration", (float(events[-1]["t"]) if events else 0) + 10))

    if camera is None:
        from slr.device.camera import FileCamera

        camera = FileCamera(str(source), 0, 0, resize=False)

    timeline: list[dict] = []
    # Use a throw-away folder for the shopping list so the user's real list is never touched.
    with tempfile.TemporaryDirectory() as tmp:
        shopping = ShoppingList(Path(tmp) / "list.yaml", classes)
        for item in script.get("list", []):
            shopping.add(item)
        core = ServerCore(cfg.guidance, catalog, shopping)

        # These mirror what the real device would be doing in response to "stream" messages.
        streaming = False
        purpose, burst_left, burst_id = "find", None, 0
        frame_no = 0
        dt = 1.0 / fps
        steps = int(round(duration * fps))
        # ei = index of the next scripted event still to apply.
        ei = 0

        def apply(msgs: list[dict], t: float, prev_mode: g.Mode) -> None:
            """Record the engine's output messages at time ``t`` and act on "stream" ones."""
            # ``nonlocal`` lets us update the outer function's variables from in here.
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
                    # Behave like the device: start/stop consuming frames, track bursts.
                    rec.update(on=m["on"], purpose=m["purpose"])
                    streaming = m["on"]
                    purpose = m["purpose"]
                    burst_left = m.get("burst")
                    burst_id = m.get("burst_id", 0)
                timeline.append(rec)

        # The main loop: one iteration per frame period of the virtual clock.
        for i in range(steps + 1):
            t = i * dt
            # 1) Apply every scripted button press whose time has come (1e-9 absorbs float error).
            while ei < len(events) and float(events[ei]["t"]) <= t + 1e-9:
                ev = events[ei]
                prev = core.state.mode
                apply(core.handle(g.Control(ev["event"], ev.get("value")), t), t, prev)
                ei += 1
            # 2) If the engine asked for frames, feed it one.
            if streaming:
                img = camera.capture()
                frame_no += 1
                dets = tuple(detector.detect(img))
                prev = core.state.mode
                apply(core.handle(g.Frame(dets, purpose, burst_id if purpose == "identify" else 0), t), t, prev)
                # An identify burst stops by itself after the requested number of frames.
                if burst_left is not None:
                    burst_left -= 1
                    if burst_left <= 0:
                        streaming, burst_left = False, None
            # 3) Always tick so time-outs fire.
            prev = core.state.mode
            apply(core.handle(g.Tick(), t), t, prev)
    return timeline


def load_script(path: str | Path) -> dict:
    """Read the YAML script file into a dict."""
    return read_yaml(Path(path))


def format_timeline(timeline: list[dict]) -> str:
    """Turn the timeline into readable lines like ``   1.20s  SAY    Pasta found.``."""
    lines = []
    for r in timeline:
        # ``{r['t']:7.2f}`` formats the time right-aligned in 7 characters with 2 decimals.
        if r["type"] == "cue":
            lines.append(f"{r['t']:7.2f}s  SAY    {r['text']}")
        elif r["type"] == "state":
            lines.append(f"{r['t']:7.2f}s  STATE  {r['mode']}")
        elif r["type"] == "stream":
            lines.append(f"{r['t']:7.2f}s  STREAM {'on' if r['on'] else 'off'} ({r['purpose']})")
        elif r["type"] == "tone":
            lines.append(f"{r['t']:7.2f}s  TONE   pan={r['pan']} interval={r['interval_ms']}")
    return "\n".join(lines)
