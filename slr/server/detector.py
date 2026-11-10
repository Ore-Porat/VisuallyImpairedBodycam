"""Object detector backends. All return list[Detection] with normalized boxes.

- fast_grocery (default): YOLO-World + CLIP for products, EgoHands for hands.
- scripted: detections read from a JSON file, for tests and replay without a model.

Runs on the laptop only. Never import this module on the Raspberry Pi (PRD rule 4).

WHAT THIS FILE DOES
-------------------
This is the "eyes" of the product. Given one camera picture, a detector returns a list of
``Detection`` objects: for each grocery item it recognises, the class name (e.g. "pasta"), a
confidence score and a rectangle ("bounding box") saying where in the picture it is.

``load_detector`` picks the backend named by ``config/server.yaml``. The real backend lives
in ``fast_detector.py``; this module keeps the shared protocol, the test/replay backend and
the JPEG helper so those do not need the heavy ML imports.

HOW IT FITS IN
--------------
``main.py`` decodes each JPEG from the Pi into an image and calls ``detector.detect(image)``.
The resulting list goes to the guidance engine as a ``Frame`` event.

WORDS YOU WILL SEE
------------------
- bounding box: the rectangle around a detected object. We store it "normalized": x and y
  as fractions 0..1 of the picture width/height, so it does not matter what resolution was used.
- confidence: 0..1, how sure the model is.
- inference: running the model on one image. ``infer_ms`` is how long that took.
- numpy array (``np.ndarray``): the standard Python type for an image: a grid of pixel values.
- CUDA / MPS: GPU acceleration on NVIDIA cards / Apple silicon. ``cpu`` is the slow fallback.
- Protocol (typing): "anything with these methods counts", without needing inheritance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from slr.common.config import ModelConfig, ServerConfig
from slr.common.types import Detection


class Detector(Protocol):
    """What every detector backend must provide: a ``names`` list and a ``detect`` method."""

    names: list[str]

    def detect(self, image: np.ndarray) -> list[Detection]: ...


def pick_device(requested: str) -> str:
    """Decide where to run the model: the configured device, or the best available if "auto".

    Returns "cuda:0" (NVIDIA GPU), "mps" (Apple GPU) or "cpu".
    """
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        # torch not installed at all: CPU it is.
        pass
    return "cpu"


class ScriptedDetector:
    """Detections from a JSON file: {"names": [...], "frames": [[{cls, conf, box}, ...], ...]}.

    Frame i of the input gets entry i (the last entry repeats). Used by replay tests.
    No neural network at all: the "answers" are written down in advance, which makes tests
    fast and predictable.
    """

    def __init__(self, path: str | Path | None = None, frames: list[list[Detection]] | None = None,
                 names: list[str] | None = None):
        """Give either ``path`` (a JSON file) or ``frames`` (a list of per-frame detection lists)."""
        if path is not None:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            frames = [[Detection.from_dict(d) for d in fr] for fr in data["frames"]]
            names = data.get("names", names)
        self.frames = frames or [[]]
        # If no names given, collect every class that appears in the frames.
        self.names = list(names or sorted({d.cls for fr in self.frames for d in fr}))
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return the scripted detections for the next frame (the image itself is ignored)."""
        fr = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return list(fr)


def load_detector(cfg: ServerConfig) -> Detector:
    """Build the detector named by ``model.backend`` in ``config/server.yaml``."""
    m: ModelConfig = cfg.model
    if m.backend == "fast_grocery":
        # Imported here so the Pi never imports torch/transformers/ultralytics.
        from slr.server.fast_detector import FastGroceryDetector

        return FastGroceryDetector(cfg, m)
    if m.backend == "scripted":
        if not m.scripted_file:
            raise ValueError("scripted backend needs model.scripted_file")
        return ScriptedDetector(cfg.path(m.scripted_file))
    raise ValueError(f"unknown detector backend {m.backend}")


def decode_jpeg(jpeg: bytes) -> np.ndarray | None:
    """Turn raw JPEG bytes (as received from the Pi) into an image array. None if corrupt."""
    import cv2

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def timed_detect(detector: Detector, image: np.ndarray) -> tuple[list[Detection], float]:
    """Run ``detector.detect`` and also measure how long it took, in milliseconds."""
    t0 = time.perf_counter()
    dets = detector.detect(image)
    return dets, (time.perf_counter() - t0) * 1000
