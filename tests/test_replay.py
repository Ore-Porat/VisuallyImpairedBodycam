"""Replay regression test (FR-73): recorded frames + scripted controls -> cue timeline."""

import json

from slr.common.config import load_server_config
from slr.common.types import Detection
from slr.server.detector import ScriptedDetector
from slr.server.replay import format_timeline, run_replay


def cue_keys(timeline):
    return [r["key"] for r in timeline if r["type"] == "cue"]


def test_find_flow_replay(frames_dir, scripted_detector):
    cfg = load_server_config()
    script = {"fps": 10, "duration": 8, "list": ["bottle", "cup"],
              "events": [{"t": 0.0, "event": "mode_on"}, {"t": 0.2, "event": "enter"},
                         {"t": 7.0, "event": "enter"}]}
    tl = run_replay(cfg, frames_dir, script, scripted_detector)
    keys = cue_keys(tl)
    assert keys[:2] == ["ready", "searching"]
    assert keys.index("found") < keys.index("within_reach") < keys.index("checked_off_next")
    texts = [r["text"] for r in tl if r["type"] == "cue"]
    assert "Bottle found." in texts
    assert "Left. Eye level." in texts
    assert "Within reach. Bottle straight ahead." in texts
    assert "Bottle checked off. Next: cup." in texts
    # streaming only while searching
    streams = [(r["t"], r["on"]) for r in tl if r["type"] == "stream"]
    assert streams[0][1] is True and streams[-1][1] is False
    assert "SAY    Bottle found." in format_timeline(tl)


def test_identify_replay(frames_dir):
    cfg = load_server_config()
    det = ScriptedDetector(frames=[[Detection.centered("cup", 0.9, 0.5, 0.5, 0.3, 0.3)]])
    script = {"fps": 10, "duration": 3, "list": ["cup"],
              "events": [{"t": 0.0, "event": "mode_on"}, {"t": 0.5, "event": "identify"}]}
    tl = run_replay(cfg, frames_dir, script, det)
    assert cue_keys(tl) == ["ready", "identifying", "identified_on_list"]
    assert det.calls == 5  # exactly one burst


def test_scripted_detections_file(tmp_path, frames_dir):
    path = tmp_path / "dets.json"
    path.write_text(json.dumps({"names": ["cup"], "frames": [[{"cls": "cup", "conf": 0.9,
                                                               "box": [0.4, 0.4, 0.6, 0.6]}]]}))
    det = ScriptedDetector(path)
    assert det.detect(None)[0].cls == "cup"
