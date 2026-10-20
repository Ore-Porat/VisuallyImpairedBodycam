"""Shared test setup that pytest loads automatically.

What is this file?
    pytest (the tool that runs our tests) looks for a file called ``conftest.py`` in the
    tests folder and imports it before any test runs. Anything defined here is available to
    every test file without an import. It is the place for "fixtures": small helpers that
    build the objects a test needs (a folder of images, a fake detector, ...).

Words you will see:
    fixture      A function marked with ``@pytest.fixture``. When a test lists its name as a
                 parameter, pytest calls the fixture first and passes the result into the test.
    tmp_path     A fixture that pytest itself provides: an empty temporary folder that is
                 thrown away after the test. Tests use it so they never touch real files.
    ScriptedDetector
                 A stand-in for the real YOLO object detector. Instead of looking at pixels,
                 it returns a pre-written list of "detections" for each frame it is asked
                 about. That lets the tests run on any computer, with no GPU or camera.
    Detection    One detected box: which class (pasta, coffee, ...), how confident the
                 detector is (0..1), and where the box is in the picture (0..1 of the frame).
"""

from pathlib import Path

import numpy as np
import pytest

from slr.common.types import Detection
from slr.server.detector import ScriptedDetector

# The folder that contains the whole project (one level above tests/). Tests use it to find
# the real config files, for example ROOT / "config" / "phrases.yaml".
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def frames_dir(tmp_path):
    """A folder of small blank frames (the scripted detector ignores pixels)."""
    # cv2 is OpenCV, the image library. It is imported here (not at the top) so that test
    # files which never use this fixture do not need OpenCV installed.
    import cv2

    d = tmp_path / "frames"
    d.mkdir()
    # Write 5 tiny JPEG images (160 x 120 pixels, 3 color channels), each a different shade
    # of gray. The file names f000.jpg ... f004.jpg sort in order, like a video would.
    for i in range(5):
        cv2.imwrite(str(d / f"f{i:03d}.jpg"), np.full((120, 160, 3), 40 * i, np.uint8))
    return d


def approach_script_detections(cls="pasta", n_far=10, n_near=40):
    """Target far on the left, then centered and big (within reach).

    Returns a list with one entry per frame. Each entry is the list of boxes the fake
    detector should "see" in that frame. The story it tells:
      * 5 frames with nothing visible,
      * n_far frames with a small box on the left (the item is far away),
      * n_near frames with a large box in the middle (the item is right in front of us).
    This is the same journey a shopper makes when walking up to a shelf item.
    """
    # Detection.centered(class, confidence, center_x, center_y, width, height); all 0..1.
    far = [Detection.centered(cls, 0.9, 0.2, 0.5, 0.1, 0.15)]
    near = [Detection.centered(cls, 0.9, 0.5, 0.5, 0.2, 0.45)]
    # [[]] * 5 means "five empty frames"; [far] * n_far repeats the far frame n_far times.
    return [[]] * 5 + [far] * n_far + [near] * n_near


@pytest.fixture
def scripted_detector():
    """A fake detector that plays back the approach story above, frame by frame."""
    return ScriptedDetector(frames=approach_script_detections(), names=["pasta", "coffee"])
