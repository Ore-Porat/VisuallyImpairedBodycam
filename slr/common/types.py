"""Small value types shared by the server, the guidance engine and the tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detected box. Coordinates are normalized to 0..1 of the frame."""

    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def iou(self, other: Detection) -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        inter = ix * iy
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def centered(cls: str, conf: float, cx: float, cy: float, w: float, h: float) -> Detection:
        """Build a detection from a center point and size (handy in tests)."""
        return Detection(cls, conf, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def to_dict(self) -> dict:
        return {"cls": self.cls, "conf": round(self.conf, 4),
                "box": [round(v, 4) for v in (self.x1, self.y1, self.x2, self.y2)]}

    @staticmethod
    def from_dict(d: dict) -> Detection:
        x1, y1, x2, y2 = d["box"]
        return Detection(d["cls"], float(d["conf"]), x1, y1, x2, y2)


PENDING = "pending"
FOUND = "found"


@dataclass(frozen=True)
class ListItem:
    """One shopping list entry (FR-5)."""

    name: str
    status: str = PENDING
