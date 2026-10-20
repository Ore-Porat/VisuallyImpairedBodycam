"""Object detector backends (PRD 8.1). All return list[Detection] with normalized boxes.

- ultralytics (default): our trained .pt / ONNX / OpenVINO model, or stock COCO weights for M1.
- darknet_opencv: original GrocerEye-style YOLOv3 .cfg + .weights through cv2.dnn.
- scripted: detections read from a JSON file, for tests and replay without a model.

Runs on the laptop only. Never import this module on the Raspberry Pi (PRD rule 4).

WHAT THIS FILE DOES
-------------------
This is the "eyes" of the product. Given one camera picture, a detector returns a list of
``Detection`` objects: for each grocery item it recognises, the class name (e.g. "pasta"), a
confidence score and a rectangle ("bounding box") saying where in the picture it is.

There are three interchangeable "backends" (ways to do the same job). Any object with a
``detect(image)`` method and a ``names`` list counts as a detector (see the ``Detector``
protocol). ``load_detector`` picks one based on ``config/server.yaml``.

HOW IT FITS IN
--------------
``main.py`` decodes each JPEG from the Pi into an image and calls ``detector.detect(image)``.
The resulting list goes to the guidance engine as a ``Frame`` event.

WORDS YOU WILL SEE
------------------
- YOLO: "You Only Look Once", a family of fast neural networks for object detection.
  ``ultralytics`` is the Python library that runs modern YOLO models (YOLO11 etc.).
- weights / .pt file: the trained model, i.e. the millions of numbers the network learned.
- COCO: a standard public dataset (people, cups, bottles...). Stock YOLO weights trained on it
  are handy for debugging without our grocery model.
- bounding box: the rectangle around a detected object. We store it "normalized": x and y
  as fractions 0..1 of the picture width/height, so it does not matter what resolution was used.
- confidence: 0..1, how sure the model is. ``min_conf`` throws away weak guesses early.
- inference: running the model on one image. ``infer_ms`` is how long that took.
- numpy array (``np.ndarray``): the standard Python type for an image: a grid of pixel values.
- CUDA / MPS: GPU acceleration on NVIDIA cards / Apple silicon. ``cpu`` is the slow fallback.
- NMS (non-maximum suppression): removing duplicate overlapping boxes for the same object.
- Protocol (typing): "anything with these methods counts", without needing inheritance.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from slr.common.config import ModelConfig, ServerConfig
from slr.common.types import Detection

log = logging.getLogger(__name__)


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


class UltralyticsDetector:
    """The default backend: runs a YOLO model through the ``ultralytics`` library."""

    def __init__(self, weights: str, device: str = "auto", imgsz: int = 640,
                 min_conf: float = 0.25, half: bool = False):
        """Load the model.

        weights:  path to a .pt/.onnx file, or a stock name like "yolo11n.pt" (auto-downloaded).
        device:   "auto", "cpu", "cuda:0", "mps" ...
        imgsz:    the picture is resized to this many pixels on its long side before inference.
        min_conf: boxes below this confidence are discarded.
        half:     use 16-bit floats (faster) when on an NVIDIA GPU.
        """
        # Imported here (not at the top) so the Pi never needs ultralytics installed.
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = pick_device(device)
        self.imgsz = imgsz
        self.min_conf = min_conf
        self.half = half and self.device.startswith("cuda")
        # The model knows its class names as either a dict {index: name} or a list; normalise
        # to a list in index order.
        names = self.model.names
        self.names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        log.info("ultralytics model %s on %s (%d classes)", weights, self.device, len(self.names))

    def warmup(self) -> None:
        """Run one dummy black image so the first real frame is not slow (lazy GPU setup)."""
        self.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run the model on one image and convert the results into normalized Detections."""
        # ``**extra`` unpacks the dict as extra keyword arguments (adds half=True only if wanted).
        extra = {"half": True} if self.half else {}
        res = self.model.predict(image, imgsz=self.imgsz, conf=self.min_conf, device=self.device,
                                 verbose=False, **extra)[0]
        h, w = image.shape[:2]
        out = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return out
        # Pull the results off the GPU into plain numpy arrays: corners, confidences, class ids.
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        # ``zip`` walks the three arrays side by side. Divide by width/height to normalize.
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            out.append(Detection(self.names[k], float(c), float(x1 / w), float(y1 / h),
                                 float(x2 / w), float(y2 / h)))
        return out


class DarknetOpenCVDetector:
    """Original GrocerEye YOLOv3 weights, if the team obtains them.

    Uses OpenCV's built-in neural network module (``cv2.dnn``) instead of ultralytics.
    """

    def __init__(self, cfg_path: str, weights_path: str, names: list[str], imgsz: int = 416,
                 min_conf: float = 0.25, nms: float = 0.45):
        """cfg_path / weights_path: the Darknet .cfg and .weights files. names: class names
        in the model's order. nms: overlap threshold for removing duplicate boxes."""
        import cv2

        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        try:
            # Ask for GPU acceleration; silently stays on CPU if OpenCV was built without CUDA.
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            pass
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.names = names
        self.imgsz = imgsz
        self.min_conf = min_conf
        self.nms = nms

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run YOLOv3 through cv2.dnn, then NMS, and return normalized Detections."""
        cv2 = self.cv2
        # A "blob" is the image resized, scaled to 0..1 and reordered the way the network wants.
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        boxes, confs, ids = [], [], []
        # Each output row is: center x, center y, width, height, objectness, then one score
        # per class. All already as fractions of the image.
        for out in outs:
            for row in out:
                scores = row[5:]
                k = int(np.argmax(scores))
                c = float(scores[k] * row[4])
                if c < self.min_conf:
                    continue
                cx, cy, bw, bh = row[:4]
                boxes.append([cx - bw / 2, cy - bh / 2, bw, bh])
                confs.append(c)
                ids.append(k)
        # Non-maximum suppression: keep only the best of each cluster of overlapping boxes.
        keep = cv2.dnn.NMSBoxes(boxes, confs, self.min_conf, self.nms) if boxes else []
        out = []
        for i in np.array(keep).flatten():
            x, y, bw, bh = boxes[i]
            # Clamp to the picture edges.
            out.append(Detection(self.names[ids[i]], confs[i], max(0.0, x), max(0.0, y),
                                 min(1.0, x + bw), min(1.0, y + bh)))
        return out


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
    if m.backend == "ultralytics":
        weights = m.weights
        if cfg.path(weights).exists():
            weights = str(cfg.path(weights))
        # else: a bare name like "yolo11n.pt" that ultralytics downloads once and caches
        return UltralyticsDetector(weights, m.device, m.imgsz, m.min_conf, m.half)
    if m.backend == "darknet_opencv":
        if not (m.darknet_cfg and m.darknet_weights):
            raise ValueError("darknet_opencv backend needs model.darknet_cfg and model.darknet_weights")
        from slr.common.config import load_classes

        names = list(load_classes(cfg.path(m.classes_file)))
        return DarknetOpenCVDetector(str(cfg.path(m.darknet_cfg)), str(cfg.path(m.darknet_weights)),
                                     names, m.imgsz, m.min_conf)
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
