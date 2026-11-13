"""Fast zero-shot grocery detector: YOLO-World boxes + CLIP verification + EgoHands hands.

This is the server-side port of ``p6k_egohands/scripts/infer_fast.py``. It runs
in-process, once, in the server's detector thread. No subprocess is started per
frame.

WHAT CHANGED IN PHASE 1
-----------------------
The old pipeline classified each crop against a fixed list of 29 labels with a softmax.
Softmax always picks something, so "not on the list" had to be faked with reject classes and
a hand-maintained YOLO-keyword compatibility map. That map is gone.

The detector now asks one question per item on the shopping list:

    "does this crop match this item?"

Each item has its own CLIP vector (its prompts, averaged). Similarity is a plain cosine
score, not a probability. An item is accepted when its score clears ``fast_item_min_score``;
when two or more items are on the list, the winner must also lead the runner-up by
``fast_item_margin``. A close call produces no detection at all, which is safer than naming
the wrong product.

Prompt quality is what makes this work. Every item ships several prompts, and a box that
YOLO calls a "can" is also compared with "a can of Coke" built from that label. The label is
never emitted as an item; it only chooses which prompt text to use.

One frame goes through:

1. YOLO-World proposes generic product boxes (can, bottle, box, bag...).
2. Each box is expanded by 10%, cropped, and padded to a square with grey.
3. CLIP embeds the crop and compares it with the current list's prompt vectors.
4. ``choose_item`` applies the threshold and margin; rejected boxes are dropped.
5. Class-aware NMS removes duplicate boxes.
6. EgoHands adds ``hand`` boxes, which are kept for the guidance engine.

Heavy packages (``torch``, ``transformers``, ``ultralytics``, OpenCV, Pillow) are
imported lazily so the Raspberry Pi never imports this module.

WORDS YOU WILL SEE
------------------
- cosine similarity: the angle between two CLIP vectors, -1..1. Roughly 0.25-0.40 for a
  match, lower for an unrelated product. It is not a probability.
- prompt bank: the text vectors for the items currently on the shopping list. It is rebuilt
  only when the list (or a prompt) changes.
- threshold / margin: the two acceptance numbers, both calibrated by
  ``training/calibrate_items.py`` (see docs/clip-item-verification.md).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from slr.common.catalog import CatalogItem, box_prompt, item_prompts
from slr.common.config import ModelConfig, ServerConfig
from slr.common.types import Detection
from slr.server.detector import pick_device
from slr.server.grocery_taxonomy import DETECTOR_CLASSES, HAND_CLASS

log = logging.getLogger(__name__)

# Container words used to turn YOLO's generic box label into a better prompt:
# a box labelled "can" is compared with "a can of Coke" (for a custom item, "a photo of a
# can of Coke") as well as the curated prompts.
# "product" is not a container, so it is left out.
PROMPT_LABELS: tuple[str, ...] = tuple(label for label in DETECTOR_CLASSES if label != "product")



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


# --------------------------------------------------------------------------- prompt bank


@dataclass(frozen=True)
class PromptBank:
    """CLIP text vectors for the items currently on the shopping list.

    ids:        the item ids, in the same order as the rows of every matrix.
    min_scores: the acceptance threshold for each item (the catalog entry's own value, or
                the global setting).
    any:        ``(n_items, dim)`` vectors averaged over all of an item's prompts.
    by_label:   for each YOLO box label ("can", "bottle", ...), ``(n_items, dim)`` vectors
                built from the curated prompts plus "a can of X". A crop labelled ``can``
                is scored against this matrix, so the prompt matches what is visible.
    key:        identity of the inputs, used to avoid rebuilding on every frame.
    """

    ids: tuple[str, ...]
    min_scores: tuple[float, ...]
    any: np.ndarray
    by_label: dict[str, np.ndarray] = field(default_factory=dict)
    key: tuple = ()

    @property
    def empty(self) -> bool:
        return not self.ids


def prompt_bank_key(items: Sequence[CatalogItem], min_score: float, from_box: bool) -> tuple:
    """Everything that changes the text vectors (and therefore requires a rebuild)."""
    return (tuple((it.id, it.spoken, it.prompts, it.min_score, it.preset) for it in items),
            float(min_score), bool(from_box))


def _mean_normalized(features: np.ndarray, rows: list[int]) -> np.ndarray:
    """Average rows and normalize the result (so a cosine is a dot product)."""
    mean = features[rows].mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm else mean


def build_prompt_bank(items: Sequence[CatalogItem], encode: Callable[[list[str]], np.ndarray],
                      min_score: float, from_box: bool = True) -> PromptBank:
    """Build the text vectors for ``items``.

    ``encode`` turns a list of prompt strings into a ``(n, dim)`` matrix of normalized CLIP
    text vectors. It is a parameter so the grouping and averaging can be tested without
    loading CLIP.
    """
    unique: list[str] = []
    ids_by_text: dict[str, int] = {}

    if not items:
        return PromptBank(ids=(), min_scores=(), any=np.zeros((0, 0), dtype=np.float32),
                          key=prompt_bank_key((), min_score, from_box))

    def text_id(text: str) -> int:
        if text not in ids_by_text:
            ids_by_text[text] = len(unique)
            unique.append(text)
        return ids_by_text[text]

    # Per item, per group name, the text ids that make up that vector.
    groups: list[dict[str, list[int]]] = []
    for item in items:
        # Presets as written; custom items always worded as photos (catalog.item_prompts).
        base = list(item_prompts(item))
        labels = PROMPT_LABELS if from_box else ()
        generated = [text_id(box_prompt(item, label)) for label in labels]
        per_item: dict[str, list[int]] = {"__all__": [text_id(t) for t in base] + generated}
        for label in labels:
            per_item[label] = [text_id(t) for t in base] + [text_id(box_prompt(item, label))]
        groups.append(per_item)

    features = np.asarray(encode(unique), dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(unique):
        raise ValueError("encode() must return one row per prompt")

    def matrix(group: str) -> np.ndarray:
        return np.stack([_mean_normalized(features, per_item[group]) for per_item in groups])

    by_label = {label: matrix(label) for label in PROMPT_LABELS if from_box}
    return PromptBank(
        ids=tuple(item.id for item in items),
        min_scores=tuple(float(item.min_score) if item.min_score is not None else float(min_score)
                         for item in items),
        any=matrix("__all__"),
        by_label=by_label,
        key=prompt_bank_key(items, min_score, from_box),
    )


def score_crops(image_features: np.ndarray, bank: PromptBank,
                labels: Sequence[str]) -> np.ndarray:
    """Cosine score of every crop against every list item, shape ``(n_crops, n_items)``.

    A crop's YOLO label selects the prompt variant for that container when one exists:
    a "can" box is compared with "a can of Coke", not only with the generic prompts.
    """
    scores = image_features @ bank.any.T
    if bank.by_label:
        for label in set(labels):
            matrix = bank.by_label.get(label)
            if matrix is None:
                continue
            rows = [i for i, value in enumerate(labels) if value == label]
            scores[rows] = image_features[rows] @ matrix.T
    return scores


def choose_item(scores: np.ndarray, ids: Sequence[str], min_scores: Sequence[float],
                margin: float) -> tuple[str, float] | None:
    """Pick the list item this crop matches, or None.

    Rules:
    - the best score must clear that item's own threshold;
    - with two or more items, it must also lead the runner-up by ``margin``. A close call
      returns None: saying nothing is safer than naming the wrong product.
    """
    if not ids or len(scores) == 0:
        return None
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    if scores[best] < min_scores[best]:
        return None
    if len(ids) >= 2 and float(scores[best] - scores[order[1]]) < margin:
        return None
    return str(ids[best]), float(scores[best])


# --------------------------------------------------------------------------- detector


class FastGroceryDetector:
    """The shipped detector: YOLO-World proposals, CLIP verification, EgoHands hands."""

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

        # Acceptance numbers (see docs/clip-item-verification.md and the calibration report).
        self.min_score = float(m.fast_item_min_score)
        self.margin = float(m.fast_item_margin)
        self.from_box = bool(m.fast_prompt_from_box)

        # The current shopping list and its text vectors. ``set_items`` may be called from
        # the event loop while the detector thread is working, hence the lock.
        self._lock = threading.Lock()
        self._items: tuple[CatalogItem, ...] = ()
        self._bank: PromptBank | None = None
        self.names = [HAND_CLASS]

        log.info(
            "fast grocery detector: %s on %s; CLIP %s (%s); hands %s",
            self.detector_path.name, self.device, m.fast_clip_model,
            "FP16" if self.clip_half else "FP32",
            self.hand_path.name if self.hand_path is not None else "off",
        )

    # ------------------------------------------------------------------ list wiring

    def set_items(self, items: Sequence[CatalogItem]) -> None:
        """Tell the detector which items are on the shopping list.

        Cheap: only stores the tuple and updates ``names``. The text vectors are built on the
        detector thread at the start of the next frame, so the event loop never runs CLIP.
        """
        with self._lock:
            self._items = tuple(items)
            self.names = [item.id for item in items] + [HAND_CLASS]

    def _bank_for_current_items(self) -> PromptBank | None:
        """Return the cached prompt bank, rebuilding it if the list or a prompt changed."""
        with self._lock:
            items = self._items
            cached = self._bank
        if not items:
            return None
        key = prompt_bank_key(items, self.min_score, self.from_box)
        if cached is not None and cached.key == key:
            return cached
        bank = build_prompt_bank(items, self._encode_texts, self.min_score, self.from_box)
        with self._lock:
            self._bank = bank
        log.info("prompt bank rebuilt for %d list item(s): %s", len(bank.ids), list(bank.ids))
        return bank

    # ------------------------------------------------------------------ model calls

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """CLIP text vectors, normalized and on the CPU (the bank is small)."""
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            inputs = self.clip_processor(text=texts, return_tensors="pt",
                                         padding=True).to(self.device)
            inputs = model_dtype_inputs(inputs, self.clip_model)
            features = _pooled(self.clip_model.get_text_features(**inputs))
            features = F.normalize(features, dim=-1)
        return features.float().cpu().numpy()

    def _embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """CLIP image vectors, normalized, shape ``(n_crops, dim)``."""
        import cv2
        import torch
        import torch.nn.functional as F
        from PIL import Image

        out: list[np.ndarray] = []
        for start in range(0, len(crops), self.m.fast_clip_batch):
            batch = crops[start:start + self.m.fast_clip_batch]
            images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in batch]
            with torch.no_grad():
                inputs = self.clip_processor(images=images, return_tensors="pt").to(self.device)
                inputs = model_dtype_inputs(inputs, self.clip_model)
                features = _pooled(self.clip_model.get_image_features(**inputs))
                features = F.normalize(features, dim=-1)
            out.append(features.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)

    def warmup(self) -> None:
        """Run one dummy frame and one dummy crop so the first real frame is fast."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._detect_products(dummy)          # builds the prompt bank if a list is loaded
        if self.hand_model is not None:
            self._detect_hands(dummy)
        # Even a frame with no boxes must warm the CLIP image path.
        self._embed_crops([np.full((224, 224, 3), 114, dtype=np.uint8)])

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return product and hand detections for one BGR frame, normalized 0..1."""
        if image is None or image.size == 0:
            return []
        products = self._detect_products(image)
        return products + self._detect_hands(image)

    def _detect_products(self, image: np.ndarray) -> list[Detection]:
        """Run YOLO-World, verify each box against the list, and keep the matches."""
        bank = self._bank_for_current_items()
        if bank is None or bank.empty:
            # No list, no targets: skip YOLO and CLIP entirely. Hands still work.
            return []

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
            candidates.append({"box": [x1, y1, x2, y2], "detector_label": detector_label})

        if not crops:
            return []
        features = self._embed_crops(crops)
        scores = score_crops(features, bank, [c["detector_label"] for c in candidates])

        products: list[dict] = []
        for candidate, row in zip(candidates, scores):
            chosen = choose_item(row, bank.ids, bank.min_scores, self.margin)
            if chosen is None:
                continue
            label, confidence = chosen
            products.append({
                "label": label,
                "score": confidence,
                "box": candidate["box"],
            })
        return [
            Detection(product["label"], float(product["score"]),
                      product["box"][0] / width, product["box"][1] / height,
                      product["box"][2] / width, product["box"][3] / height)
            for product in class_aware_nms(products, self.m.fast_nms_iou)
        ]

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
