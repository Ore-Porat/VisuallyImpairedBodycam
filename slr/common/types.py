"""Small value types shared by the server, the guidance engine and the tests.

A "value type" is a tiny container for a few related numbers or words, like a labelled box.
Two live here:

- ``Detection``: one object the camera model spotted in a frame (what it is, how sure the model
  is, and where the box around it sits in the picture).
- ``ListItem``: one line of the shopping list (the item name and whether it was found yet).

Both halves of the product (laptop server and Pi device) and the unit tests import these, so
they live in the shared ``slr/common`` package.

Words you will see
------------------
- dataclass: a Python shortcut that turns a plain list of fields into a class with a ready-made
  constructor, printing and equality. ``frozen=True`` means the object cannot be changed after
  it is created, which keeps the guidance logic predictable.
- property: a method that you read like a plain attribute, e.g. ``det.cx`` instead of
  ``det.cx()``. Used here for values that are computed from the stored fields.
- normalized coordinates: positions written as a fraction of the picture size (0.0 = left/top
  edge, 1.0 = right/bottom edge) instead of pixels, so the same numbers work for any camera
  resolution.
- IoU (Intersection over Union): a standard 0..1 score for how much two boxes overlap.
  1.0 = identical boxes, 0.0 = no overlap at all.
- type hints (``-> float``, ``cls: str``): notes about what kind of value a function takes or
  returns. Python does not enforce them; they are documentation for people and tools.
- ``from __future__ import annotations``: lets a class mention its own name in type hints
  (e.g. ``other: Detection`` inside ``Detection``) before the class is fully defined.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detected box. Coordinates are normalized to 0..1 of the frame.

    Fields:
    - ``cls``: the class id the model assigned, e.g. ``"pasta"`` (a key in config/classes.yaml).
    - ``conf``: the model's confidence, 0.0 (no idea) to 1.0 (certain).
    - ``x1, y1``: top-left corner of the box as a fraction of frame width / height.
    - ``x2, y2``: bottom-right corner, same units.

    The server's detector creates these from the model output, and the guidance engine reads
    them to decide what to say ("slightly left", "eye level", ...).
    """

    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    # The four properties below are convenience values computed from the corners.
    @property
    def cx(self) -> float:
        """Horizontal center of the box (0 = left edge of the frame, 1 = right edge)."""
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        """Vertical center of the box (0 = top of the frame, 1 = bottom)."""
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        """Width of the box as a fraction of the frame width."""
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        """Height of the box as a fraction of the frame height (bigger = closer to the camera)."""
        return self.y2 - self.y1

    def iou(self, other: Detection) -> float:
        """How much this box overlaps ``other``, as Intersection over Union (0..1).

        Used by the guidance engine to decide whether a box in the new frame is "the same
        object" as the box it was tracking in the previous frame.
        """
        # Width and height of the overlapping rectangle. max(0, ...) clamps to zero when the
        # boxes do not touch at all (the subtraction would otherwise go negative).
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        # Area covered by either box = area A + area B - the part counted twice.
        union = self.w * self.h + other.w * other.h - inter
        # Guard against dividing by zero for degenerate (zero-area) boxes.
        return inter / union if union > 0 else 0.0

    @staticmethod
    def centered(cls: str, conf: float, cx: float, cy: float, w: float, h: float) -> Detection:
        """Build a detection from a center point and size (handy in tests).

        ``@staticmethod`` means you call it on the class, ``Detection.centered(...)``, without
        needing an existing Detection first.
        """
        return Detection(cls, conf, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def to_dict(self) -> dict:
        """Turn the detection into a plain dictionary, ready to be written as JSON.

        Used by the session log and by scripted-detection files. Numbers are rounded to 4
        decimals to keep the files small.
        """
        return {"cls": self.cls, "conf": round(self.conf, 4),
                "box": [round(v, 4) for v in (self.x1, self.y1, self.x2, self.y2)]}

    @staticmethod
    def from_dict(d: dict) -> Detection:
        """The reverse of ``to_dict``: rebuild a Detection from a dictionary read from JSON."""
        # Unpack the 4-number list into four separate variables.
        x1, y1, x2, y2 = d["box"]
        return Detection(d["cls"], float(d["conf"]), x1, y1, x2, y2)


# The two possible states of a shopping list entry. Using named constants instead of typing the
# strings everywhere avoids typos like "pendng".
PENDING = "pending"
FOUND = "found"


@dataclass(frozen=True)
class ListItem:
    """One shopping list entry (FR-5).

    FR-5 is a requirement number from the product spec, PRD.pdf.

    - ``name``: the class id of the item, e.g. ``"pasta"``.
    - ``status``: ``PENDING`` (still to find) or ``FOUND`` (checked off). New items start pending.
    """

    name: str
    status: str = PENDING
