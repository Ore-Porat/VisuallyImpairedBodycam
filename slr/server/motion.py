"""Estimate global camera motion between two frames (the bodycam is worn on the chest).

WHY THIS EXISTS
---------------
The guidance engine tracks one shopping item. When the shopper turns their head or walks,
the item leaves the picture for a moment even though it is still on the shelf. Without extra
information the engine cannot tell that apart from "the item is gone", so it would say
"Lost it." during a perfectly normal movement.

The server already decodes every JPEG (``slr/server/main.py``), so it can measure how far the
whole picture moved since the previous frame. That number is passed to the guidance engine
as ``Frame.motion``; the engine then freezes the lost timer while the view is moving.

HOW
---
Downscale to a small greyscale image (cheap), then use phase correlation
(``cv2.phaseCorrelate``) to find the translation that best aligns the two pictures. Phase
correlation is the right tool here: it is a single FFT-based pass, it is robust to brightness
changes, and it returns a confidence (``response``) so we can ignore frames with no texture.
On a laptop this costs about 1-2 ms per 640x480 frame, next to 20-40 ms of inference.

This measures translation, which is what a head turn / body movement mostly produces at the
scale of one frame. Rotation is not modelled; for the "is the view moving" gate that is
enough. When OpenCV is unavailable, or the estimate is not trustworthy, ``update`` returns
0.0 ("no evidence of motion") so the engine falls back to its normal behaviour.
"""

from __future__ import annotations

import numpy as np


class MotionEstimator:
    """Frame-to-frame picture shift, as a fraction of the frame (0.0 = still).

    ``update`` is stateful: it remembers the previous small greyscale frame. It must be
    called in frame order, which is guaranteed because the server runs detection on a single
    worker thread.
    """

    def __init__(self, width: int = 160, min_response: float = 0.05):
        self.width = width
        # Below this phase-correlation confidence the pictures have too little common
        # texture (blank wall, lens covered) to say anything, so report "still".
        self.min_response = min_response
        self._prev: np.ndarray | None = None
        self._window: np.ndarray | None = None
        self._source_shape: tuple[int, int] | None = None

    def reset(self) -> None:
        """Forget the previous frame (e.g. after a stream restart or a resolution change)."""
        self._prev = None
        self._window = None
        self._source_shape = None

    def update(self, image: np.ndarray | None) -> float:
        """Return the global shift since the previous call, in frame fractions.

        image: a decoded BGR or greyscale frame, or None for a corrupt JPEG.
        """
        if image is None or getattr(image, "size", 0) == 0:
            return 0.0
        try:
            import cv2
        except ImportError:
            # OpenCV is required for decoding too, but keep the server usable without it.
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        height, width = gray.shape[:2]
        if width <= 1 or height <= 1:
            return 0.0
        # A different camera resolution (or a restart at another size) makes the previous
        # frame incomparable: start over instead of reporting a bogus shift.
        if self._source_shape is not None and self._source_shape != (height, width):
            self._prev = None
        self._source_shape = (height, width)

        # Resize to a fixed width so the cost does not depend on the camera resolution.
        scale = self.width / float(width)
        small = cv2.resize(gray, (self.width, max(2, int(round(height * scale)))),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        prev, self._prev = self._prev, small
        if prev is None or prev.shape != small.shape:
            return 0.0
        # A flat picture has no features to correlate; correlating it gives nonsense.
        if float(small.std()) < 1.0 or float(prev.std()) < 1.0:
            return 0.0

        if self._window is None or self._window.shape != small.shape:
            self._window = cv2.createHanningWindow((small.shape[1], small.shape[0]),
                                                   cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(prev, small, self._window)
        if not np.isfinite(dx) or not np.isfinite(dy) or response < self.min_response:
            # Not enough agreement between the frames: treat as "no evidence".
            return 0.0
        # Convert pixels to fractions of the frame so the threshold is resolution independent.
        return float(np.hypot(dx / small.shape[1], dy / small.shape[0]))
