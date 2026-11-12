"""Fast zero-shot grocery detector: prompts, label filtering, crops and config."""

import numpy as np
import pytest

from slr.common.config import load_classes, load_server_config
from slr.server.detector import load_detector
from slr.server.fast_detector import class_aware_nms, prepare_crop, resolve_label
from slr.server.grocery_taxonomy import (
    CLASS_DESCRIPTIONS,
    DETECTOR_CLASSES,
    DETECTOR_KEYWORDS,
    FAST_CLASS_NAMES,
    REJECT_CLASSES,
)

from conftest import ROOT


def test_taxonomy_matches_classes_file():
    """Every shopping-list class has a CLIP prompt, and only hand is added separately."""
    classes = load_classes(ROOT / "config" / "classes.yaml")
    assert set(FAST_CLASS_NAMES) == set(classes)
    assert set(CLASS_DESCRIPTIONS) == set(classes) - {"hand"}
    assert set(DETECTOR_KEYWORDS) <= set(CLASS_DESCRIPTIONS)
    assert set(REJECT_CLASSES) == {"background", "hand", "person"}


def test_cookies_has_prompt_and_keywords():
    """The VIB shopping list has cookies; the fast model needs a prompt and box keywords."""
    assert CLASS_DESCRIPTIONS["cookies"] == "a photo of a package of cookies"
    assert DETECTOR_KEYWORDS["cookies"] == ("box", "package", "bag")


def test_generic_detector_classes():
    """The YOLO-World classes are the generic boxes used for non-engine .pt models too."""
    assert DETECTOR_CLASSES == (
        "product", "package", "can", "bottle", "box", "bag", "jar", "container",
    )


def test_resolve_label_rules():
    """CLIP labels are filtered by reject classes, confidence and detector agreement."""
    # Reject class and low confidence are always unknown.
    assert resolve_label("background", 0.99, "box", 0.8) == "unknown"
    assert resolve_label("hand", 0.99, "box", 0.8) == "unknown"
    assert resolve_label("chips", 0.50, "bag", 0.8) == "unknown"
    # Specific label contradicts the generic detector label.
    assert resolve_label("soda", 0.90, "box", 0.8) == "unknown"
    # Matching detector keyword passes.
    assert resolve_label("soda", 0.90, "can", 0.8) == "soda"
    assert resolve_label("cookies", 0.90, "bag", 0.8) == "cookies"
    # A very confident CLIP label bypasses the keyword check.
    assert resolve_label("soda", 0.999, "box", 0.8) == "soda"


def test_prepare_crop_expands_and_pads_square():
    """Crops get a 10% margin and a grey square pad, matching the classifier training data."""
    pytest.importorskip("cv2")
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)
    crop = prepare_crop(frame, (30, 40, 70, 60), 0.10)
    assert crop is not None
    assert crop.shape == (48, 48, 3)
    # 40x20 box -> 48x24 crop; the 12 rows above and below are grey padding.
    assert tuple(crop[0, 0]) == (114, 114, 114)
    assert tuple(crop[-1, 0]) == (114, 114, 114)


def test_class_aware_nms_keeps_best_per_class():
    """Overlapping boxes of the same class collapse; different classes are independent."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    products = [
        {"label": "chips", "score": 0.90, "box": [0, 0, 10, 10]},
        {"label": "chips", "score": 0.80, "box": [1, 1, 11, 11]},
        {"label": "soda", "score": 0.70, "box": [0, 0, 10, 10]},
    ]
    kept = class_aware_nms(products, 0.30)
    assert [(p["label"], p["score"]) for p in kept] == [("chips", 0.90), ("soda", 0.70)]


def test_shipped_config_uses_fast_grocery():
    """The shipped server.yaml points at the S detector and EgoHands engines."""
    cfg = load_server_config()
    assert cfg.model.backend == "fast_grocery"
    assert cfg.model.fast_detector_engine == "models/tensorrt/shelf_world_s.engine"
    assert cfg.model.fast_hand_engine == "models/tensorrt/hand_egohands_960.engine"
    assert cfg.model.fast_hand_imgsz == 960
    assert cfg.model.fast_hand_conf == 0.55
    assert cfg.model.fast_clip_half is True
    assert cfg.model.active_model == cfg.model.fast_detector_engine


def test_missing_engine_is_a_clear_error():
    """A missing detector engine fails before any model download or CUDA work."""
    cfg = load_server_config()
    cfg.model.fast_detector_engine = "models/tensorrt/does-not-exist.engine"
    with pytest.raises(ValueError, match="fast_detector_engine not found"):
        load_detector(cfg)
