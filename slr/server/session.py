"""Shopping list: ordered class names with pending/found status, persisted to YAML (FR-1..6)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

from slr.common.types import FOUND, PENDING, ListItem


class ListError(ValueError):
    pass


def normalize_name(name: str, known: dict[str, str]) -> str:
    """Accept a class id or its spoken name, case-insensitively; return the class id."""
    raw = " ".join(name.split()).lower()
    for cls, spoken in known.items():
        if raw in (cls.lower(), spoken.lower(), cls.lower().replace("_", " ")):
            return cls
    raise ListError(
        f"'{name}' is not an item the model knows. Valid items: " + ", ".join(sorted(known))
    )


class ShoppingList:
    def __init__(self, path: Path, known_classes: dict[str, str]):
        self.path = Path(path)
        self.known = known_classes
        self.items: list[ListItem] = []
        self.current = 0
        self._mtime: float | None = None
        self.load()

    # ----------------------------------------------------------------- persistence

    def load(self) -> None:
        if not self.path.exists():
            self.items, self.current = [], 0
            self._mtime = None
            return
        with open(self.path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = []
        for entry in data.get("items") or []:
            if isinstance(entry, str):
                entry = {"name": entry}
            status = entry.get("status", PENDING)
            items.append(ListItem(str(entry["name"]), FOUND if status == FOUND else PENDING))
        self.items = items
        self.current = int(data.get("current", 0) or 0)
        self._clamp()
        self._mtime = self.path.stat().st_mtime

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"current": self.current,
                "items": [{"name": it.name, "status": it.status} for it in self.items]}
        # atomic write so a crash never leaves a half-written list
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("# SLR SLR shopping list. Edit with `slr list ...` or the web editor.\n")
            yaml.safe_dump(data, f, sort_keys=False)
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime

    def changed_on_disk(self) -> bool:
        """True if someone else (the CLI) edited the file since we last loaded or saved it."""
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return self._mtime is not None
        return mtime != self._mtime

    def _clamp(self) -> None:
        if not self.items:
            self.current = 0
        else:
            self.current = min(max(self.current, 0), len(self.items) - 1)

    # ----------------------------------------------------------------- edits

    def add(self, name: str) -> str:
        cls = normalize_name(name, self.known)
        self.items.append(ListItem(cls))
        self.save()
        return cls

    def remove(self, name: str) -> str:
        try:
            cls = normalize_name(name, self.known)
        except ListError:
            cls = name
        for i, it in enumerate(self.items):
            if it.name == cls:
                del self.items[i]
                if i < self.current:
                    self.current -= 1
                self._clamp()
                self.save()
                return cls
        raise ListError(f"'{name}' is not on the list")

    def clear(self) -> None:
        self.items, self.current = [], 0
        self.save()

    def reset(self) -> None:
        """FR-6: mark every item pending again and start from the top."""
        self.items = [ListItem(it.name, PENDING) for it in self.items]
        self.current = 0
        self.save()

    def set_state(self, items: tuple[ListItem, ...], current: int) -> None:
        """Store the engine's view (after an item was checked off) if it changed."""
        if list(items) == self.items and current == self.current:
            return
        self.items, self.current = list(items), current
        self._clamp()
        self.save()

    def unknown_items(self) -> list[str]:
        return [it.name for it in self.items if it.name not in self.known]

    def describe(self) -> str:
        if not self.items:
            return "Shopping list is empty."
        lines = []
        for i, it in enumerate(self.items):
            marker = ">" if i == self.current else " "
            mark = "x" if it.status == FOUND else " "
            spoken = self.known.get(it.name, it.name)
            lines.append(f"{marker} {i + 1}. [{mark}] {spoken} ({it.status})")
        found = sum(1 for it in self.items if it.status == FOUND)
        lines.append(f"{found} of {len(self.items)} found.")
        return "\n".join(lines)
