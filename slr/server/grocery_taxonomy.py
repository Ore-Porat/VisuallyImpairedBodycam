"""The generic box vocabulary YOLO-World proposes.

WHAT THIS FILE IS FOR
---------------------
YOLO-World is open-vocabulary: it draws a box around something and calls it one of the
generic words below ("can", "bottle", "box"...). Those words are NOT item labels. The
server never says "soda" or "package" to the shopper; it takes the box, crops it, and asks
CLIP whether the crop matches an item on the shopping list (see ``fast_detector.py``).

So this list is deliberately small and generic. It is the outer bound on what can be
detected at all: if YOLO-World does not propose a box around an object, no prompt can find
it. Adding a grocery category here would not help; adding an item goes in
``config/catalog.yaml`` (or the shopping-list page).

WORDS YOU WILL SEE
------------------
- box vocabulary: the generic nouns the detector was exported with. The shipped TensorRT
  engine has them baked in; a plain ``.pt`` model is configured with them at load time.
- ``HAND_CLASS``: the one non-generic class, produced by the separate EgoHands detector.
"""

from __future__ import annotations

# Generic classes baked into the shipped YOLO-World TensorRT engine. They are also
# used to configure a plain .pt YOLO-World model, which has no baked labels.
DETECTOR_CLASSES: tuple[str, ...] = (
    "product", "package", "can", "bottle", "box", "bag", "jar", "container",
)

# The hand class is reported by the EgoHands model, never by CLIP.
HAND_CLASS = "hand"
