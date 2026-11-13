"""Fast zero-shot grocery detector: prompt bank, scoring, crops and config.

What is being tested?
    The detector no longer classifies a crop against a fixed label list with a softmax. It
    scores the crop against each item on the shopping list and accepts it only when the score
    clears a threshold (and, with two or more items, leads the runner-up by a margin). These
    tests pin that decision logic without loading CLIP or TensorRT.
"""

import numpy as np
import pytest

from slr.common.catalog import CatalogItem
from slr.common.config import ModelConfig, load_server_config
from slr.server.detector import load_detector
from slr.server.fast_detector import (
    PromptBank,
    build_prompt_bank,
    choose_item,
    class_aware_nms,
    prepare_crop,
    score_crops,
)
from slr.server.grocery_taxonomy import DETECTOR_CLASSES, HAND_CLASS

from conftest import ROOT


def item(item_id, prompts=(), spoken=None, min_score=None):
    """A small catalog entry for the tests."""
    return CatalogItem(id=item_id, spoken=spoken or item_id,
                       prompts=tuple(prompts), min_score=min_score)


def fake_encoder(texts):
    """Deterministic 'CLIP': a 3-D vector from words that appear in the prompt.

    This lets us check grouping and averaging without a model. The values are arbitrary;
    only their relative direction matters.
    """
    rows = []
    for text in texts:
        lowered = text.lower()
        rows.append([
            1.0 if "can" in lowered else 0.0,
            1.0 if "bottle" in lowered else 0.0,
            1.0 if "red" in lowered else 0.5,
        ])
    return np.array(rows, dtype=np.float32)


# --------------------------------------------------------------------------- vocabulary


def test_detector_classes_are_generic_boxes():
    """The YOLO-World vocabulary is generic containers, never item labels."""
    assert DETECTOR_CLASSES == (
        "product", "package", "can", "bottle", "box", "bag", "jar", "container",
    )
    assert "soda" not in DETECTOR_CLASSES and "pasta" not in DETECTOR_CLASSES
    assert HAND_CLASS == "hand"


# --------------------------------------------------------------------------- prompt bank


def test_prompt_bank_matches_the_item_order():
    items = [item("coke", ["a red can of Coke"]), item("sprite", ["a green can of Sprite"])]
    bank = build_prompt_bank(items, fake_encoder, min_score=0.3)
    assert bank.ids == ("coke", "sprite")
    assert bank.any.shape[0] == 2
    assert not bank.empty


def test_prompt_bank_has_a_variant_for_every_container():
    """A box YOLO calls 'can' gets a prompt that says 'a can of X'."""
    bank = build_prompt_bank([item("coke", ["a red can of Coke"])], fake_encoder, 0.3)
    assert set(bank.by_label) == {"package", "can", "bottle", "box", "bag", "jar", "container"}
    assert bank.by_label["can"].shape == (1, bank.any.shape[1])


def test_prompts_are_averaged_and_normalized():
    """The item vector is the normalized mean of its prompts, so cosine = dot product."""
    bank = build_prompt_bank([item("coke", ["a red can of Coke", "a can of Coca-Cola"])],
                             fake_encoder, 0.3)
    vector = bank.any[0]
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)


def test_box_prompts_can_be_disabled():
    bank = build_prompt_bank([item("coke", ["a red can of Coke"])], fake_encoder, 0.3,
                             from_box=False)
    assert bank.by_label == {}


def test_an_item_without_prompts_gets_a_generated_one():
    bank = build_prompt_bank([item("blue_gatorade")], fake_encoder, 0.3)
    assert bank.ids == ("blue_gatorade",)
    assert bank.any.shape[0] == 1


def test_custom_items_are_prompted_as_photos_presets_as_written():
    """Custom items read as photos ("a photo of a can of X"); presets keep their wording."""
    seen = []

    def recording(texts):
        seen.extend(texts)
        return fake_encoder(texts)

    custom = CatalogItem(id="kombucha", spoken="kombucha", prompts=("a glass bottle",),
                         preset=False)
    build_prompt_bank([custom, item("coke", ["a red can of Coke"])], recording, 0.3)
    mine = [s for s in seen if "kombucha" in s or "glass bottle" in s]
    assert mine and all(s.startswith("a photo of") for s in mine)
    assert "a photo of a can of kombucha" in seen
    assert "a red can of Coke" in seen and "a can of coke" in seen


def test_per_item_threshold_overrides_the_global_one():
    bank = build_prompt_bank(
        [item("coke", ["a can"], min_score=0.41), item("sprite", ["a can"])],
        fake_encoder, min_score=0.3,
    )
    assert bank.min_scores == (0.41, 0.3)


def test_bank_key_changes_when_a_prompt_changes():
    """The cache must notice an edited prompt, not only a different item id."""
    from slr.server.fast_detector import prompt_bank_key

    a = [item("coke", ["a can of Coke"])]
    b = [item("coke", ["a red can of Coke"])]
    assert prompt_bank_key(a, 0.3, True) != prompt_bank_key(b, 0.3, True)
    assert prompt_bank_key(a, 0.3, True) == prompt_bank_key([item("coke", ["a can of Coke"])],
                                                            0.3, True)


# --------------------------------------------------------------------------- scoring


def test_score_crops_uses_the_box_label_variant():
    """A 'can' crop is scored against the can prompts; other labels use the generic ones."""
    bank = PromptBank(
        ids=("a", "b"),
        min_scores=(0.2, 0.2),
        any=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        by_label={"can": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)},
    )
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = score_crops(features, bank, ["can", "bottle"])
    assert np.allclose(scores[0], [0.0, 1.0])      # can variant flips the order
    assert np.allclose(scores[1], [0.0, 1.0])      # bottle falls back to the any matrix


# --------------------------------------------------------------------------- decision rule


def test_choose_item_accepts_a_clear_winner():
    assert choose_item(np.array([0.40, 0.20]), ("a", "b"), (0.3, 0.3), 0.02) == ("a", 0.40)


def test_choose_item_rejects_a_close_call():
    """Coke vs Sprite too close to call: say nothing rather than guess."""
    assert choose_item(np.array([0.310, 0.305]), ("coke", "sprite"), (0.3, 0.3), 0.02) is None


def test_choose_item_rejects_below_the_threshold():
    assert choose_item(np.array([0.29, 0.10]), ("a", "b"), (0.30, 0.30), 0.0) is None


def test_choose_item_respects_a_per_item_threshold():
    """The top box still has to clear its own threshold, not just the global one."""
    assert choose_item(np.array([0.31, 0.20]), ("a", "b"), (0.40, 0.10), 0.0) is None
    assert choose_item(np.array([0.41, 0.20]), ("a", "b"), (0.40, 0.10), 0.0) == ("a", 0.41)


def test_choose_item_with_one_item_ignores_the_margin():
    """There is no runner-up to compare with, so only the threshold decides."""
    assert choose_item(np.array([0.31]), ("a",), (0.30,), 0.5) == ("a", 0.31)


def test_choose_item_with_an_empty_list_is_none():
    assert choose_item(np.array([]), (), (), 0.02) is None


# --------------------------------------------------------------------------- boxes


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


# --------------------------------------------------------------------------- config


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


def test_shipped_thresholds_match_the_calibration():
    """The cosine-scale numbers come from reports/clip_item_calibration.md."""
    cfg = load_server_config()
    assert cfg.model.fast_item_min_score == 0.30
    assert cfg.model.fast_item_margin == 0.02
    assert cfg.model.fast_prompt_from_box is True
    # The softmax probability threshold is gone; the name would silently mistune the model.
    assert "fast_clip_conf" not in ModelConfig.model_fields
    assert cfg.guidance.conf_find == 0.30
    assert cfg.guidance.conf_identify == 0.33


def test_missing_engine_is_a_clear_error():
    """A missing detector engine fails before any model download or CUDA work."""
    cfg = load_server_config()
    cfg.model.fast_detector_engine = "models/tensorrt/does-not-exist.engine"
    with pytest.raises(ValueError, match="fast_detector_engine not found"):
        load_detector(cfg)
