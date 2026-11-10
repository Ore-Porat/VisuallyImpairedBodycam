"""Labels and CLIP prompts for the fast zero-shot grocery detector.

The fast pipeline has no trained grocery classifier. YOLO-World proposes generic
product boxes, and CLIP names each crop by comparing its image embedding with the
text embeddings below. ``config/classes.yaml`` uses the same class ids and adds
``hand``, which is supplied by the separate EgoHands detector rather than CLIP.

Words you will see
------------------
- CLIP: a model that places images and text in the same vector space, so "a photo
  of a soda can" can be compared with a crop of a can.
- prompt: the sentence CLIP compares against each crop.
- reject class: a prompt that means "not a grocery item" (room background, a hand,
  a person). If CLIP picks one of these, the crop is labelled ``unknown``.
- detector keyword: YOLO-World's generic box label (can, bottle, box, bag...). A
  specific CLIP label must be compatible with it, which rejects room objects that
  happen to look like a product.
"""

from __future__ import annotations

# One prompt per grocery class. The order matches config/classes.yaml (minus hand),
# so the classifier output can be compared directly with the shopping list.
CLASS_DESCRIPTIONS: dict[str, str] = {
    "beans": "a photo of canned beans",
    "cake": "a photo of a cake",
    "candy": "a photo of a candy package",
    "cereal": "a photo of a cereal box",
    "chips": "a photo of a bag of potato chips",
    "chocolate": "a photo of a chocolate bar",
    "coffee": "a photo of a coffee package",
    "corn": "a photo of canned corn",
    "fish": "a photo of canned fish",
    "flour": "a photo of a bag of flour",
    "honey": "a photo of a jar of honey",
    "jam": "a photo of a jar of jam",
    "juice": "a photo of a bottle of juice",
    "milk": "a photo of a milk carton",
    "nuts": "a photo of a package of nuts",
    "oil": "a photo of a bottle of oil",
    "pasta": "a photo of a pasta package",
    "rice": "a photo of a bag of rice",
    "soda": "a photo of a soda can",
    "spices": "a photo of a spice jar",
    "sugar": "a photo of a bag of sugar",
    "tea": "a photo of a tea box",
    "tomato_sauce": "a photo of a can of tomato sauce",
    "vinegar": "a photo of a bottle of vinegar",
    "water": "a photo of a water bottle",
    "cookies": "a photo of a package of cookies",
}

# Prompts that are not shopping items. A crop whose best CLIP label is one of these
# is hidden as "unknown"; hands are drawn separately by the hand detector.
REJECT_CLASSES: dict[str, str] = {
    "background": "a photo of a room background",
    "hand": "a photo of a human hand",
    "person": "a photo of a person",
}

# Compatibility between a CLIP label and YOLO-World's generic detector label.
# If the detector called the box "box", it cannot become "soda" unless CLIP is
# almost certain (the bypass threshold lives in fast_detector.resolve_label).
DETECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "soda": ("can", "bottle"),
    "water": ("bottle", "can", "carton"),
    "juice": ("bottle", "can", "carton"),
    "oil": ("bottle", "can"),
    "vinegar": ("bottle", "can"),
    "milk": ("carton", "bottle", "can"),
    "chips": ("bag", "package", "packet", "box", "chips"),
    "candy": ("bag", "package", "packet", "box"),
    "chocolate": ("bar", "package", "box"),
    "coffee": ("package", "bag", "box", "jar"),
    "tea": ("box", "package", "bag"),
    "cereal": ("box", "package"),
    "pasta": ("package", "bag", "box"),
    "flour": ("bag", "package"),
    "rice": ("bag", "package"),
    "sugar": ("bag", "package"),
    "nuts": ("bag", "package", "jar"),
    "beans": ("can", "jar", "package"),
    "corn": ("can", "jar"),
    "fish": ("can", "jar"),
    "tomato_sauce": ("can", "jar", "bottle"),
    "spices": ("jar", "bottle", "package"),
    "honey": ("jar", "bottle"),
    "jam": ("jar", "bottle"),
    "cookies": ("box", "package", "bag"),
}

# Generic classes baked into the shipped YOLO-World TensorRT engine. They are also
# used to configure a plain .pt YOLO-World model, which has no baked labels.
DETECTOR_CLASSES: tuple[str, ...] = (
    "product", "package", "can", "bottle", "box", "bag", "jar", "container",
)

# The hand class is reported by the EgoHands model, never by CLIP.
HAND_CLASS = "hand"

# Every class the fast detector can return, in shopping-list order.
FAST_CLASS_NAMES: tuple[str, ...] = (*CLASS_DESCRIPTIONS, HAND_CLASS)
