from pathlib import Path

import numpy as np
import pytest

from slr.common.types import Detection
from slr.server.detector import ScriptedDetector

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def frames_dir(tmp_path):
    """A folder of small blank frames (the scripted detector ignores pixels)."""
    import cv2

    d = tmp_path / "frames"
    d.mkdir()
    for i in range(5):
        cv2.imwrite(str(d / f"f{i:03d}.jpg"), np.full((120, 160, 3), 40 * i, np.uint8))
    return d


def approach_script_detections(cls="pasta", n_far=10, n_near=40):
    """Target far on the left, then centered and big (within reach)."""
    far = [Detection.centered(cls, 0.9, 0.2, 0.5, 0.1, 0.15)]
    near = [Detection.centered(cls, 0.9, 0.5, 0.5, 0.2, 0.45)]
    return [[]] * 5 + [far] * n_far + [near] * n_near


@pytest.fixture
def scripted_detector():
    return ScriptedDetector(frames=approach_script_detections(), names=["pasta", "coffee"])
