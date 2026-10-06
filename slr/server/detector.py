"""Object detector backends (PRD 8.1). All return list[Detection] with normalized boxes.

- ultralytics (default): our trained .pt / ONNX / OpenVINO model, or stock COCO weights for M1.
- darknet_opencv: original GrocerEye-style YOLOv3 .cfg + .weights through cv2.dnn.
- scripted: detections read from a JSON file, for tests and replay without a model.

Runs on the laptop only. Never import this module on the Raspberry Pi (PRD rule 4).
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
    names: list[str]

    def detect(self, image: np.ndarray) -> list[Detection]: ...


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class UltralyticsDetector:
    def __init__(self, weights: str, device: str = "auto", imgsz: int = 640,
                 min_conf: float = 0.25, half: bool = False):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = pick_device(device)
        self.imgsz = imgsz
        self.min_conf = min_conf
        self.half = half and self.device.startswith("cuda")
        names = self.model.names
        self.names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        log.info("ultralytics model %s on %s (%d classes)", weights, self.device, len(self.names))

    def warmup(self) -> None:
        self.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    def detect(self, image: np.ndarray) -> list[Detection]:
        extra = {"half": True} if self.half else {}
        res = self.model.predict(image, imgsz=self.imgsz, conf=self.min_conf, device=self.device,
                                 verbose=False, **extra)[0]
        h, w = image.shape[:2]
        out = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            out.append(Detection(self.names[k], float(c), float(x1 / w), float(y1 / h),
                                 float(x2 / w), float(y2 / h)))
        return out


class DarknetOpenCVDetector:
    """Original GrocerEye YOLOv3 weights, if the team obtains them."""

    def __init__(self, cfg_path: str, weights_path: str, names: list[str], imgsz: int = 416,
                 min_conf: float = 0.25, nms: float = 0.45):
        import cv2

        self.cv2 = cv2
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        try:
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
        cv2 = self.cv2
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        boxes, confs, ids = [], [], []
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
        keep = cv2.dnn.NMSBoxes(boxes, confs, self.min_conf, self.nms) if boxes else []
        out = []
        for i in np.array(keep).flatten():
            x, y, bw, bh = boxes[i]
            out.append(Detection(self.names[ids[i]], confs[i], max(0.0, x), max(0.0, y),
                                 min(1.0, x + bw), min(1.0, y + bh)))
        return out


class ScriptedDetector:
    """Detections from a JSON file: {"names": [...], "frames": [[{cls, conf, box}, ...], ...]}.

    Frame i of the input gets entry i (the last entry repeats). Used by replay tests.
    """

    def __init__(self, path: str | Path | None = None, frames: list[list[Detection]] | None = None,
                 names: list[str] | None = None):
        if path is not None:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            frames = [[Detection.from_dict(d) for d in fr] for fr in data["frames"]]
            names = data.get("names", names)
        self.frames = frames or [[]]
        self.names = list(names or sorted({d.cls for fr in self.frames for d in fr}))
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        fr = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return list(fr)


def load_detector(cfg: ServerConfig) -> Detector:
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
    import cv2

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def timed_detect(detector: Detector, image: np.ndarray) -> tuple[list[Detection], float]:
    t0 = time.perf_counter()
    dets = detector.detect(image)
    return dets, (time.perf_counter() - t0) * 1000
