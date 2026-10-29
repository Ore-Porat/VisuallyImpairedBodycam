"""Camera freshness logic that does not need a physical webcam.

The device must never send a picture that was already handed out (PRD 7.1). These tests build
an ``OpenCVCamera`` without opening hardware and exercise the condition-variable helper that
enforces that guarantee.
"""

import threading

import numpy as np

from slr.device.camera import OpenCVCamera


def test_wait_frame_times_out_instead_of_returning_stale():
    """No new frame -> None, even when an older frame is still in memory."""
    cam = object.__new__(OpenCVCamera)          # skip __init__: no webcam needed
    cam._cond = threading.Condition()
    cam._latest = np.zeros((2, 2, 3), dtype=np.uint8)
    cam._seq = 1

    assert cam._wait_frame(after=1, timeout=0.01) is None
    frame = cam._wait_frame(after=0, timeout=0.01)
    assert frame is not None and frame.shape == (2, 2, 3)
