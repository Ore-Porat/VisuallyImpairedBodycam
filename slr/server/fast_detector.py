"""Fast zero-shot grocery detector: YOLO-World boxes + CLIP labels + EgoHands hands.

This is the server-side port of ``p6k_egohands/scripts/infer_fast.py``. It runs
in-process, once, in the server's detector thread. No subprocess is started per
frame.

One frame goes through:

1. YOLO-World proposes generic product boxes (can, bottle, box, bag...).
2. Each box is expanded by 10%, cropped, and padded to a square with grey.
3. CLIP names the crop with one of the grocery classes (or a reject class).
4. Boxes whose CLIP label is unknown, background, hand, or person are dropped.
5. Class-aware NMS removes duplicate boxes.
6. EgoHands adds ``hand`` boxes, which are kept for the guidance engine.

Heavy packages (``torch``, ``transformers``, ``ultralytics``, OpenCV, Pillow) are
imported lazily so the Raspberry Pi never imports this module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from slr.common.config import ModelConfig, ServerConfig
from slr.common.types import Detection
from slr.server.detector import pick_device
from slr.server.grocery_taxonomy import (
    CLASS_DESCRIPTIONS,
    DETECTOR_CLASSES,
    DETECTOR_KEYWORDS,
    HAND_CLASS,
    REJECT_CLASSES,
)

log = logging.getLogger(__name__)

# Labels whose CLIP top-1 is already close enough to override the generic detector
# label. Mirrors infer_fast.py: a "box" can become soda only if CLIP is this sure.
DETECTOR_KEYWORD_BYPASS = 0.995


def _pooled(output):
    """Return the pooled embedding from either an output object or a bare tensor.

    Transformers versions differ: some return ``BaseModelOutputWithPooling`` and
    others return the tensor directly from ``get_*_features``.
    """
    return output.pooler_output if hasattr(output, "pooler_output") else output


def model_dtype_inputs(inputs, model):
    """Cast floating-point inputs to the model's dtype (FP16 on CUDA)."""
    import torch

    dtype = next(model.parameters()).dtype
    if dtype == torch.float16:
        return {
            key: value.half() if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in inputs.items()
        }
    return inputs


def prepare_crop(frame: np.ndarray, box, margin: float = 0.10) -> np.ndarray | None:
    """Expand a detector box, crop it, and pad it to a square with grey.

    The 10% margin and square grey pad match the classifier-development crops and
    stop bottles/bags from being squashed into a rectangle. The grey value is
    ``(114, 114, 114)`` in BGR, the same for every channel.
    """
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(round(bw * margin)), int(round(bh * margin))
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(width, x2 + mx)
    y2 = min(height, y2 + my)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(
        crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )


def resolve_label(best_label: str, best_conf: float, detector_label: str,
                  cls_conf: float) -> str:
    """Turn CLIP's best guess into a final class id, or ``"unknown"``.

    A crop is rejected when CLIP picked a reject class, when confidence is below
    ``cls_conf``, or when the specific label contradicts the generic detector label
    (for example "soda" from a bare "box"). Very confident CLIP labels bypass the
    last check.
    """
    if best_label in REJECT_CLASSES or best_conf < cls_conf:
        return "unknown"
    keywords = DETECTOR_KEYWORDS.get(best_label)
    if (keywords is not None
            and not any(keyword in detector_label for keyword in keywords)
            and best_conf < DETECTOR_KEYWORD_BYPASS):
        return "unknown"
    return best_label


def class_aware_nms(products: list[dict], iou_threshold: float) -> list[dict]:
    """Keep the highest-confidence box per class, removing overlapping duplicates."""
    if not products:
        return []
    import torch
    from torchvision.ops import nms

    groups: dict[str, list[dict]] = {}
    for product in products:
        groups.setdefault(product["label"], []).append(product)

    kept: list[dict] = []
    for items in groups.values():
        boxes = torch.tensor([item["box"] for item in items], dtype=torch.float32)
        scores = torch.tensor([item["score"] for item in items], dtype=torch.float32)
        for index in nms(boxes, scores, iou_threshold).tolist():
            kept.append(items[index])
    kept.sort(key=lambda item: item["score"], reverse=True)
    return kept


def classifier_text_features(clip_processor, clip_model, labels: list[str],
                             descriptions: dict[str, str], device: str):
    """Embed every class prompt once, normalized for cosine similarity."""
    import torch
    import torch.nn.functional as F

    texts = [descriptions[name] for name in labels]
    with torch.no_grad():
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
        inputs = model_dtype_inputs(inputs, clip_model)
        features = _pooled(clip_model.get_text_features(**inputs))
        features = F.normalize(features, dim=-1)
    return features


def classify_batch(clip_processor, clip_model, crops: list[np.ndarray], text_features,
                   labels: list[str], device: str, batch_size: int) -> list[list[tuple[str, float]]]:
    """Run CLIP on a list of BGR crops and return each crop's top-5 labels."""
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image

    results: list[list[tuple[str, float]]] = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size]
        images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in batch]
        with torch.no_grad():
            inputs = clip_processor(images=images, return_tensors="pt").to(device)
            inputs = model_dtype_inputs(inputs, clip_model)
            features = _pooled(clip_model.get_image_features(**inputs))
            features = F.normalize(features, dim=-1)
            probs = ((features @ text_features.T) * 100).softmax(dim=-1)
        for row in probs:
            top_probs, top_indices = row.topk(min(5, len(labels)))
            results.append([
                (labels[int(index)], float(prob))
                for prob, index in zip(top_probs, top_indices)
            ])
    return results


class FastGroceryDetector:
    """The shipped detector: YOLO-World products, CLIP labels, and EgoHands hands."""

    names: list[str]

    def __init__(self, cfg: ServerConfig, m: ModelConfig):
        """Load both TensorRT engines and CLIP once. Runs in the detector thread."""
        from transformers import CLIPModel, CLIPProcessor
        from ultralytics import YOLO

        self.cfg = cfg
        self.m = m
        self.device = pick_device(m.device)
        self.detector_path = cfg.path(m.fast_detector_engine)
        self.hand_path = cfg.path(m.fast_hand_engine) if m.fast_hand_engine else None

        if not self.detector_path.is_file():
            raise ValueError(
                f"fast_detector_engine not found: {self.detector_path}. "
                "Copy or export the engine as described in models/tensorrt/README.md."
            )
        if self.hand_path is not None and not self.hand_path.is_file():
            raise ValueError(
                f"fast_hand_engine not found: {self.hand_path}. "
                "Copy or export the engine as described in models/tensorrt/README.md, "
                "or set fast_hand_engine: null to disable hands."
            )

        detector_suffix = self.detector_path.suffix.lower()
        detector_baked = detector_suffix in {".engine", ".onnx"}
        if detector_baked and not self.device.startswith("cuda"):
            raise ValueError(
                "TensorRT engines require CUDA, but the selected device is "
                f"{self.device!r}. Run from the project nix-shell so "
                "LD_LIBRARY_PATH includes /run/opengl-driver/lib, and set "
                "model.device: auto (or cuda:0)."
            )

        self.detector = YOLO(str(self.detector_path))
        if not detector_baked:
            # A plain .pt YOLO-World model has no baked classes; configure it the
            # same way export_tensorrt.py does before exporting the engine.
            self.detector.set_classes(list(DETECTOR_CLASSES))
            log.info("configured YOLO-World detector classes %s", list(DETECTOR_CLASSES))

        self.hand_model = None
        if self.hand_path is not None:
            hand_suffix = self.hand_path.suffix.lower()
            if hand_suffix in {".engine", ".onnx"} and not self.device.startswith("cuda"):
                raise ValueError(
                    "The EgoHands TensorRT engine requires CUDA, but the selected "
                    f"device is {self.device!r}."
                )
            self.hand_model = YOLO(str(self.hand_path))

        self.clip_processor = CLIPProcessor.from_pretrained(m.fast_clip_model)
        self.clip_model = CLIPModel.from_pretrained(m.fast_clip_model).to(self.device).eval()
        self.clip_half = bool(m.fast_clip_half and self.device.startswith("cuda"))
        if self.clip_half:
            self.clip_model = self.clip_model.half()

        self.labels = list(CLASS_DESCRIPTIONS) + list(REJECT_CLASSES)
        descriptions = {**CLASS_DESCRIPTIONS, **REJECT_CLASSES}
        self.text_features = classifier_text_features(
            self.clip_processor, self.clip_model, self.labels, descriptions, self.device
        )
        self.names = list(CLASS_DESCRIPTIONS) + [HAND_CLASS]

        log.info(
            "fast grocery detector: %s on %s; CLIP %s (%s); hands %s",
            self.detector_path.name, self.device, m.fast_clip_model,
            "FP16" if self.clip_half else "FP32",
            self.hand_path.name if self.hand_path is not None else "off",
        )

    def warmup(self) -> None:
        """Run one dummy frame and one dummy crop so the first real frame is fast."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._detect_products(dummy)
        if self.hand_model is not None:
            self._detect_hands(dummy)
        # Even a blank frame may produce no boxes, so explicitly warm CLIP.
        self._classify_crops([np.full((224, 224, 3), 114, dtype=np.uint8)])

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return product and hand detections for one BGR frame, normalized 0..1."""
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        products = class_aware_nms(self._detect_products(image), self.m.fast_nms_iou)

        out: list[Detection] = []
        for product in products:
            if product["label"] == "unknown":
                continue
            x1, y1, x2, y2 = product["box"]
            out.append(Detection(
                product["label"], float(product["score"]),
                x1 / width, y1 / height, x2 / width, y2 / height,
            ))
        out.extend(self._detect_hands(image))
        return out

    def _classify_crops(self, crops: list[np.ndarray]) -> list[list[tuple[str, float]]]:
        if not crops:
            return []
        return classify_batch(
            self.clip_processor, self.clip_model, crops, self.text_features,
            self.labels, self.device, self.m.fast_clip_batch,
        )

    def _detect_products(self, image: np.ndarray) -> list[dict]:
        """Run YOLO-World, crop each box, and let CLIP assign a label."""
        result = self.detector.predict(
            image,
            imgsz=self.m.fast_detector_imgsz,
            conf=self.m.fast_detector_conf,
            iou=self.m.fast_detector_iou,
            max_det=self.m.fast_max_det,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        height, width = image.shape[:2]
        crops: list[np.ndarray] = []
        candidates: list[dict] = []
        for box, score, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            x1, y1, x2, y2 = (int(round(float(v))) for v in box)
            x1, x2 = max(0, x1), min(width - 1, x2)
            y1, y2 = max(0, y1), min(height - 1, y2)
            if x2 - x1 < self.m.fast_min_box or y2 - y1 < self.m.fast_min_box:
                continue
            detector_label = str(result.names[int(class_id)]).lower()
            if HAND_CLASS in detector_label:
                # Hands come from the dedicated EgoHands backend.
                continue
            crop = prepare_crop(image, (x1, y1, x2, y2), 0.10)
            if crop is None:
                continue
            crops.append(crop)
            candidates.append({
                "box": [x1, y1, x2, y2],
                "detector_score": float(score),
                "detector_label": detector_label,
            })

        tops = self._classify_crops(crops)
        products: list[dict] = []
        for candidate, top in zip(candidates, tops):
            best_label, best_conf = top[0]
            candidate.update({
                "label": resolve_label(
                    best_label, best_conf, candidate["detector_label"], self.m.fast_clip_conf
                ),
                "score": best_conf,
                "top": top,
            })
            products.append(candidate)
        return products

    def _detect_hands(self, image: np.ndarray) -> list[Detection]:
        """Run EgoHands and return its boxes as ``hand`` detections."""
        if self.hand_model is None:
            return []
        result = self.hand_model.predict(
            image,
            imgsz=self.m.fast_hand_imgsz,
            conf=self.m.fast_hand_conf,
            iou=0.5,
            max_det=self.m.fast_max_hands,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        height, width = image.shape[:2]
        hands: list[Detection] = []
        for box, score in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = (int(round(float(v))) for v in box)
            x1, x2 = max(0, x1), min(width - 1, x2)
            y1, y2 = max(0, y1), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            hands.append(Detection(
                HAND_CLASS, float(score),
                x1 / width, y1 / height, x2 / width, y2 / height,
            ))
        return hands[: self.m.fast_max_hands]
