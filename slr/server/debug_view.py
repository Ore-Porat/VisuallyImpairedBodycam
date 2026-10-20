"""Live debug window on the laptop (FR-71): frame, every box with class + confidence, the target
highlighted, state, last cue, latency and UDP loss. Press f to save the frame for training (FR-74).

The window runs in its own thread (OpenCV GUI calls must stay on one thread). Other keys are
forwarded to ``on_key`` so in sim mode the window can also drive the simulated device.

WHAT THIS FILE DOES
-------------------
A window on the laptop screen that shows what the system "sees" and "thinks": the latest camera
picture with rectangles drawn around detected items (green and thick = the item we are looking
for, orange and thin = anything else), faint grey guide lines marking the "straight ahead" band
and the shelf zones, and a black status bar underneath with the current mode, timings, and the
last sentence spoken. It is purely for developers; the shopper never sees it.

HOW IT FITS IN
--------------
``main.py`` calls ``DebugView.update(...)`` whenever it has a new frame, detections or state.
The window's own thread redraws about 30 times a second. Key presses in the window go back to
``main.py`` via the ``on_key`` callback (used by ``slr sim`` to simulate device buttons).

WORDS YOU WILL SEE
------------------
- FR-71 / FR-74: requirement numbers in the product spec, ``PRD.pdf``.
- OpenCV (``cv2``): the image library we use to draw and to show the window.
- thread: a separate line of execution running at the same time as the rest of the server.
  OpenCV windows must be created and drawn from one single thread, hence ``_run``.
- lock: guards ``self.info`` so the server thread and the window thread never read/write it
  at the same moment.
- callback: a function handed to someone else so they can call it later (``on_key``).
- numpy array: an image is a grid of pixels stored as a numpy array (height x width x 3 colors).
- latency / e2e: end-to-end time from camera capture to the device starting to speak.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from slr.common.types import Detection

log = logging.getLogger(__name__)

# cv2.waitKeyEx codes for arrow keys (Windows, Linux/GTK, macOS)
ARROWS = {2424832: "left", 2555904: "right", 65361: "left", 65363: "right", 63234: "left", 63235: "right"}


@dataclass
class DebugInfo:
    """Everything the window shows, in one bundle. ``update`` fills it, ``_render`` reads it."""

    image: np.ndarray | None = None
    detections: list[Detection] = field(default_factory=list)
    target: str | None = None
    mode: str = ""
    last_cue: str = ""
    infer_ms: float = 0.0
    latency_ms: float | None = None
    loss: float | None = None
    fps: float = 0.0
    item: str = ""


class DebugView:
    """The debug window. Create it, call ``start()``, then ``update(...)`` as things change."""

    def __init__(self, window_name: str, captured_dir: Path, conf_find: float,
                 on_key: Callable[[str], None] | None = None):
        """window_name:  title of the window.
        captured_dir: folder where pressing "f" saves frames (for training data).
        conf_find:    the confidence threshold; boxes of the target below it are not highlighted.
        on_key:       optional function called with a key name ("space", "n", "left"...) for
                      every key press that the window does not handle itself.
        """
        self.window_name = window_name
        self.captured_dir = captured_dir
        self.conf_find = conf_find
        self.on_key = on_key
        self.info = DebugInfo()
        self._lock = threading.Lock()
        # A flag the server sets to ask the window thread to exit.
        self._stop = threading.Event()
        # daemon=True: this thread must not keep the program alive when everything else exits.
        self._thread = threading.Thread(target=self._run, name="debug-view", daemon=True)

    def start(self) -> None:
        """Open the window (in its own thread)."""
        self._thread.start()

    def stop(self) -> None:
        """Ask the window thread to close the window and exit."""
        self._stop.set()

    def update(self, **kw) -> None:
        """Set any DebugInfo fields by name, e.g. ``update(mode="guiding", fps=9.5)``."""
        with self._lock:
            # ``setattr(obj, "name", value)`` is the same as ``obj.name = value`` with a
            # variable field name.
            for k, v in kw.items():
                setattr(self.info, k, v)

    def _render(self) -> np.ndarray:
        """Draw the current picture, boxes, guide lines and status bar into one image."""
        import cv2

        # Take a snapshot copy under the lock so drawing does not block the server thread.
        with self._lock:
            info = DebugInfo(**vars(self.info))
        # No frame yet? Show a black 640x480 placeholder.
        img = info.image.copy() if info.image is not None else np.zeros((480, 640, 3), np.uint8)
        h, w = img.shape[:2]
        for d in info.detections:
            is_target = d.cls == info.target and d.conf >= self.conf_find
            # Colors are (blue, green, red) in OpenCV: green for the target, orange otherwise.
            color = (0, 220, 0) if is_target else (0, 180, 255)
            # Convert the 0..1 box corners into pixel coordinates.
            p1 = (int(d.x1 * w), int(d.y1 * h))
            p2 = (int(d.x2 * w), int(d.y2 * h))
            cv2.rectangle(img, p1, p2, color, 4 if is_target else 1)
            # Label just above the box (but never above the top edge of the picture).
            cv2.putText(img, f"{d.cls} {d.conf:.2f}", (p1[0], max(12, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # zone guides: straight-ahead band and shelf lines
        for x in (0.42, 0.58):
            cv2.line(img, (int(x * w), 0), (int(x * w), h), (90, 90, 90), 1)
        for y in (0.33, 0.66):
            cv2.line(img, (0, int(y * h)), (w, int(y * h)), (90, 90, 90), 1)
        # Format optional numbers as "-" when unknown.
        lat = f"{info.latency_ms:.0f} ms" if info.latency_ms is not None else "-"
        loss = f"{info.loss * 100:.1f}%" if info.loss is not None else "-"
        lines = [
            f"mode: {info.mode}   target: {info.target or '-'}   item: {info.item}",
            f"infer {info.infer_ms:.0f} ms   e2e {lat}   fps {info.fps:.1f}   udp loss {loss}",
            f"cue: {info.last_cue}",
            "f = save frame",
        ]
        # A black bar, 22 pixels per text line, drawn under the picture.
        bar = np.zeros((22 * len(lines) + 8, w, 3), np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(bar, line, (6, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        # Stack picture and bar vertically into one image.
        return np.vstack([img, bar])

    def save_frame(self) -> Path | None:
        """Save the current (undrawn) camera picture as a JPEG for training. Returns its path."""
        import cv2

        with self._lock:
            img = None if self.info.image is None else self.info.image.copy()
        if img is None:
            return None
        self.captured_dir.mkdir(parents=True, exist_ok=True)
        path = self.captured_dir / datetime.now().strftime("frame_%Y%m%d_%H%M%S_%f.jpg")
        cv2.imwrite(str(path), img)
        log.info("saved %s", path)
        return path

    def _run(self) -> None:
        """The window thread: redraw, read a key, repeat until stopped or the window is closed."""
        import cv2

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            # Headless machine (no screen): give up quietly, the server keeps running.
            log.warning("debug view unavailable (no display?): %s", e)
            return
        while not self._stop.is_set():
            cv2.imshow(self.window_name, self._render())
            # Wait up to 30 ms for a key press; -1 means no key.
            key = cv2.waitKeyEx(30)
            if key == -1:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break  # user closed the window
                continue
            # Translate the key code into a name: arrows via the table, letters via chr().
            name = ARROWS.get(key)
            if name is None and 0 <= key < 256:
                name = "space" if key == 32 else chr(key).lower()
            if name == "f":
                self.save_frame()
            elif name and self.on_key:
                self.on_key(name)
            # Give other threads a chance to run.
            time.sleep(0)
        cv2.destroyWindow(self.window_name)
