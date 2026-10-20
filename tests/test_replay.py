"""Replay regression test (FR-73): recorded frames + scripted controls -> cue timeline.

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    "Replay" (``slr/server/replay.py``, command ``slr replay``) feeds a folder of recorded
    frames and a script of button presses through the real server logic, without a camera
    or a device, and produces a timeline of what the device would have said and when.
    These tests check that the Find flow and the Identify flow produce the expected
    sequence of spoken cues. FR-73 is a requirement number from the product spec, PRD.pdf.

Words you will see:
    fixture       A helper pytest builds for the test; ``frames_dir`` and
                  ``scripted_detector`` come from tests/conftest.py.
    timeline      A list of dictionaries, one per event ("cue" = something spoken,
                  "stream" = camera streaming turned on or off), each with a time ``t``.
    assert        "Check that this is true; if not, the test fails."
"""

import json

from slr.common.config import load_server_config
from slr.common.types import Detection
from slr.server.detector import ScriptedDetector
from slr.server.replay import format_timeline, run_replay


def cue_keys(timeline):
    """Pull out just the phrase names (e.g. 'found', 'within_reach') of the spoken cues."""
    return [r["key"] for r in timeline if r["type"] == "cue"]


def test_find_flow_replay(frames_dir, scripted_detector):
    """A full Find: switch on, press enter, walk up to pasta, press enter to check it off."""
    # Arrange: real server config, a list of two items, and three scripted button presses.
    cfg = load_server_config()
    script = {"fps": 10, "duration": 8, "list": ["pasta", "coffee"],
              "events": [{"t": 0.0, "event": "mode_on"}, {"t": 0.2, "event": "enter"},
                         {"t": 7.0, "event": "enter"}]}
    # Act: run the replay. The scripted detector shows pasta far left, then close and centered.
    tl = run_replay(cfg, frames_dir, script, scripted_detector)
    keys = cue_keys(tl)
    # Assert: the cues come in the right order ...
    assert keys[:2] == ["ready", "searching"]
    assert keys.index("found") < keys.index("within_reach") < keys.index("checked_off_next")
    # ... and the exact spoken sentences are right (names come from config/classes.yaml).
    texts = [r["text"] for r in tl if r["type"] == "cue"]
    assert "Pasta found." in texts
    assert "Left. Eye level." in texts
    assert "Within reach. Pasta straight ahead." in texts
    assert "Pasta checked off. Next: coffee." in texts
    # streaming only while searching
    streams = [(r["t"], r["on"]) for r in tl if r["type"] == "stream"]
    assert streams[0][1] is True and streams[-1][1] is False
    # The human-readable printout of the timeline shows the same cue.
    assert "SAY    Pasta found." in format_timeline(tl)


def test_identify_replay(frames_dir):
    """Identify mode: one burst of 5 frames showing coffee ends in 'This is coffee. It's on your list.'"""
    cfg = load_server_config()
    # A detector that always sees one centered coffee box.
    det = ScriptedDetector(frames=[[Detection.centered("coffee", 0.9, 0.5, 0.5, 0.3, 0.3)]])
    script = {"fps": 10, "duration": 3, "list": ["coffee"],
              "events": [{"t": 0.0, "event": "mode_on"}, {"t": 0.5, "event": "identify"}]}
    tl = run_replay(cfg, frames_dir, script, det)
    assert cue_keys(tl) == ["ready", "identifying", "identified_on_list"]
    assert det.calls == 5  # exactly one burst


def test_scripted_detections_file(tmp_path, frames_dir):
    """A scripted detector can be loaded from a JSON file (used by `slr replay`)."""
    path = tmp_path / "dets.json"
    # Write a JSON file describing one frame with one coffee box.
    path.write_text(json.dumps({"names": ["coffee"], "frames": [[{"cls": "coffee", "conf": 0.9,
                                                               "box": [0.4, 0.4, 0.6, 0.6]}]]}))
    det = ScriptedDetector(path)
    # detect(None): the scripted detector ignores the image, so None is fine.
    assert det.detect(None)[0].cls == "coffee"
