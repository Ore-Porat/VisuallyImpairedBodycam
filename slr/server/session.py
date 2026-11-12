"""Shopping list: ordered class names with pending/found status, persisted to YAML (FR-1..6).

WHAT THIS FILE DOES
-------------------
The shopper's list lives in a small text file (``data/shopping_list.yaml``). This file reads it,
lets other code add / remove / reset items, and writes it back safely. Every item has a name
(a model "class id" like ``tomato_sauce``) and a status: pending (still to find) or found.

Item names are not limited to the presets. With a ``Catalog`` (the normal server setup), a name
the catalog does not know is created as a user item with a generated prompt; ``data/catalog_user.yaml``
keeps it for next time. The old strict mode (a plain id -> spoken dict) is still supported for
tests and old callers.

HOW IT FITS IN
--------------
Three things edit the list: the command line (``slr list add pasta``), the browser page from
``list_web.py``, and the guidance engine when it checks an item off. The server watches the
file's modification time so edits made from the command line are picked up while it runs.

WORDS YOU WILL SEE
------------------
- FR-1..6: requirement numbers in the product spec, ``PRD.pdf``.
- YAML: a human-friendly text format for data (indentation and ``key: value`` lines).
- class id: the internal name the detector uses for an item, e.g. ``tomato_sauce``. The
  "spoken name" is what we say out loud, e.g. "tomato sauce". ``known`` maps id -> spoken name.
- atomic write: write to a temporary file first, then rename it over the real file in one step,
  so a crash halfway through can never leave a half-written list.
- mtime: a file's "modified time" as recorded by the operating system.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

from slr.common.catalog import Catalog
from slr.common.types import FOUND, PENDING, ListItem


class ListError(ValueError):
    """Raised for user mistakes like adding an unknown item. Callers show the message."""

    pass


def normalize_name(name: str, known: dict[str, str] | Catalog) -> str:
    """Accept a class id or its spoken name, case-insensitively; return the class id.

    name:  what the user typed, e.g. "Tomato Sauce", "tomato_sauce" or "tomato sauce".
    known: dict of class id -> spoken name, or a Catalog. A Catalog also understands aliases.
    Raises ListError if nothing matches. (``ShoppingList.add`` creates new user items when a
    Catalog is supplied; this function is the strict lookup used by remove.)
    """
    # A Catalog knows ids, spoken names and aliases.
    if isinstance(known, Catalog):
        item = known.resolve(name)
        if item is None:
            raise ListError(f"'{name}' is not an item in the catalog")
        return item.id
    # Collapse extra spaces and lower-case so "Tomato   Sauce" matches "tomato sauce".
    raw = " ".join(name.split()).lower()
    for cls, spoken in known.items():
        if raw in (cls.lower(), spoken.lower(), cls.lower().replace("_", " ")):
            return cls
    raise ListError(
        f"'{name}' is not an item the model knows. Valid items: " + ", ".join(sorted(known))
    )


class ShoppingList:
    """The list in memory plus load/save to its YAML file. Keeps ``current`` (the active index)."""

    def __init__(self, path: Path, known_classes: dict[str, str] | Catalog):
        """Load the list from ``path`` (a missing file just means an empty list).

        known_classes: class id -> spoken name, or a Catalog. With a Catalog, a name that is
        not known yet is created as a user item (free text) instead of being rejected.
        """
        self.path = Path(path)
        self.catalog = known_classes if isinstance(known_classes, Catalog) else None
        self.known = (known_classes.item_map() if self.catalog is not None
                      else dict(known_classes))
        self.items: list[ListItem] = []
        self.current = 0
        # Modification time of the file when we last read or wrote it (see changed_on_disk).
        self._mtime: float | None = None
        self.load()

    def refresh_known(self) -> None:
        """Refresh the id -> spoken name map after the catalog gained an item."""
        if self.catalog is not None:
            self.known = self.catalog.item_map()

    # ----------------------------------------------------------------- persistence

    def load(self) -> None:
        """(Re)read the YAML file into ``items`` and ``current``."""
        if not self.path.exists():
            self.items, self.current = [], 0
            self._mtime = None
            return
        # ``with open(...) as f`` opens the file and closes it automatically afterwards.
        with open(self.path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = []
        for entry in data.get("items") or []:
            # Accept both the short form ("pasta") and the full form ({name: pasta, status: found}).
            if isinstance(entry, str):
                entry = {"name": entry}
            status = entry.get("status", PENDING)
            items.append(ListItem(str(entry["name"]), FOUND if status == FOUND else PENDING))
        self.items = items
        self.current = int(data.get("current", 0) or 0)
        self._clamp()
        self._mtime = self.path.stat().st_mtime

    def save(self) -> None:
        """Write ``items`` and ``current`` to the YAML file atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"current": self.current,
                "items": [{"name": it.name, "status": it.status} for it in self.items]}
        # atomic write so a crash never leaves a half-written list
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("# SLR SLR shopping list. Edit with `slr list ...` or the web editor.\n")
            yaml.safe_dump(data, f, sort_keys=False)
        # Rename the finished temp file over the real one in a single step.
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime

    def changed_on_disk(self) -> bool:
        """True if someone else (the CLI) edited the file since we last loaded or saved it."""
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            # The file vanished: that counts as a change only if we had one before.
            return self._mtime is not None
        return mtime != self._mtime

    def _clamp(self) -> None:
        """Keep ``current`` inside 0 .. len(items)-1 (or 0 for an empty list)."""
        if not self.items:
            self.current = 0
        else:
            self.current = min(max(self.current, 0), len(self.items) - 1)

    # ----------------------------------------------------------------- edits

    def add(self, name: str) -> str:
        """Append an item (by id or spoken name) and save. Returns the class id.

        With a Catalog, a name that is not known is created as a user item with generated
        prompts. This is how free text becomes a real, persistent item.
        """
        if self.catalog is not None:
            cls = self.catalog.resolve_or_create(name).id
            self.refresh_known()
        else:
            cls = normalize_name(name, self.known)
        self.items.append(ListItem(cls))
        self.save()
        return cls

    def remove(self, name: str) -> str:
        """Remove the first item with this name and save. Returns the class id."""
        if self.catalog is not None:
            item = self.catalog.resolve(name)
            cls = item.id if item is not None else name
        else:
            try:
                cls = normalize_name(name, self.known)
            except ListError:
                # Not a known class; still allow removing it by its raw name (e.g. a stale entry).
                cls = name
        for i, it in enumerate(self.items):
            if it.name == cls:
                del self.items[i]
                # If we removed something above the current item, shift current up by one so
                # it still points at the same item.
                if i < self.current:
                    self.current -= 1
                self._clamp()
                self.save()
                return cls
        raise ListError(f"'{name}' is not on the list")

    def clear(self) -> None:
        """Remove every item and save."""
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

    def uncurated_items(self) -> list[str]:
        """List items that use a generated prompt (no hand-written prompts in the catalog).

        They still work: the detector builds "a photo of X" and, for a box label, "a can of
        X". Accuracy is lower, so the UI marks them and the server logs them at startup.
        """
        if self.catalog is None:
            return []
        out: list[str] = []
        for listed in self.items:
            item = self.catalog.item(listed.name)
            if item is None or not item.prompts:
                out.append(listed.name)
        return out

    def describe(self) -> str:
        """A multi-line text view of the list for ``slr list show``.

        ``>`` marks the current item, ``[x]`` marks found items.
        """
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
