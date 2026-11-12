"""The item catalog: preset items and their CLIP prompts, plus items the user adds.

WHAT THIS FILE DOES
-------------------
The shopping list stores *ids* such as ``pasta`` or ``coke``. This file turns an id into the
two things the rest of the program needs:

- ``spoken``: the words the voice says, e.g. "tomato sauce".
- ``prompts``: the sentences CLIP compares a camera crop against, e.g.
  "a can of Coca-Cola". Good prompts are the difference between "not sure" and a correct
  label, so each preset ships several and they are averaged.

It also owns the *only* place new item names are created. You can type an item the presets
do not know ("blue Gatorade"); it gets an id, a spoken name and a generated prompt, and is
saved to ``data/catalog_user.yaml`` so it is still there next time.

WHAT IS A PRESET AND WHAT IS NOT
--------------------------------
``config/catalog.yaml`` is the shipped menu: a curated set of common groceries with good
prompts. It is read-only. ``data/catalog_user.yaml`` is written by the program and survives
updates. A user entry with the same id overrides the preset (that is how you fix a bad
prompt without editing the shipped file).

This module is deliberately free of model code: it only handles text and YAML, so the CLI,
the web page, the server and the training scripts can all use it without importing torch.

WORDS YOU WILL SEE
------------------
- id: the internal name used everywhere, e.g. ``tomato_sauce`` (no spaces).
- spoken name: what the voice says, e.g. "tomato sauce" (spaces allowed).
- alias: another way to type an item, e.g. "coca cola" for ``coke``.
- prompt: the sentence CLIP is given, e.g. ``a photo of a pasta package``.
- non-item class: a detector class that is not a grocery item (``hand``). It is kept in the
  catalog so names can be looked up, but it can never be added to the shopping list.
- slug: the id made from a name: lower case, spaces become underscores.
- atomic write: write a temp file, then rename it over the real one, so a crash cannot leave
  a half-written file.
"""

from __future__ import annotations

import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

# The entry keys a catalog file may use. Anything else is a typo and is rejected loudly.
ENTRY_KEYS = {"spoken", "prompts", "aliases", "non_item", "min_score"}

# Prompt used when an item has no curated prompts (every user-added item, at first).
# The detector adds container-specific variants ("a can of Coke") on top of this at runtime.
GENERIC_PROMPT = "a photo of {spoken}"

# Safety bounds for prompts edited on the web page: enough for several templates, short
# enough that a pasted essay cannot slow the CLIP text encoder to a crawl.
MAX_PROMPTS = 12
MAX_PROMPT_CHARS = 300


class CatalogError(ValueError):
    """Raised for a malformed catalog file, or a name that cannot become an id."""

    pass


@dataclass(frozen=True)
class CatalogItem:
    """One addable item.

    id:       internal name, e.g. ``tomato_sauce``.
    spoken:   what the voice says, e.g. "tomato sauce".
    prompts:  curated CLIP prompts. Empty means "generate one from the spoken name".
    aliases:  extra spellings accepted for this item.
    non_item: True for detector classes that are not groceries (``hand``).
    preset:   True when it came from ``config/catalog.yaml``.
    min_score: optional per-item acceptance threshold; None uses the global setting.
    """

    id: str
    spoken: str
    prompts: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    non_item: bool = False
    preset: bool = True
    min_score: float | None = None


def normalize_key(name: str) -> str:
    """A comparison key: case, spacing, underscores and hyphens do not matter.

    ``"Coca-Cola"``, ``"coca_cola"`` and ``"  COCA   COLA "`` all become ``"coca cola"``.
    Accents are stripped so a file saved on a phone still matches ("café" -> "cafe").
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[-_]+", " ", text)
    return " ".join(text.split()).lower()


def slugify(name: str) -> str:
    """Turn a spoken name into a safe id. ``"Coca Cola"`` -> ``"coca_cola"``.

    Raises CatalogError for a name with no letters or digits (e.g. "!!!").
    """
    text = normalize_key(name)
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not slug:
        raise CatalogError(f"{name!r} is not a usable item name")
    # Keep ids short enough to read on the debug screen and in logs.
    return slug[:40].strip("_")


def default_prompts(spoken: str) -> tuple[str, ...]:
    """The prompt an item gets when nobody wrote a better one."""
    return (GENERIC_PROMPT.format(spoken=spoken),)


@dataclass
class Catalog:
    """Preset items plus the user's own, merged. Look up by id, spoken name or alias.

    preset:    id -> CatalogItem from ``config/catalog.yaml``.
    user:      id -> CatalogItem from ``data/catalog_user.yaml`` (overrides a preset).
    user_path: where ``save_user`` writes. None keeps user items in memory only (tests).
    """

    preset: dict[str, CatalogItem] = field(default_factory=dict)
    user: dict[str, CatalogItem] = field(default_factory=dict)
    user_path: Path | None = None
    # Built once from the merged items; maps normalize_key(...) -> id.
    _by_key: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # Modification time of the user file when last read/written (see changed_on_disk).
    _mtime: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._reindex()
        self._remember_mtime()

    # ------------------------------------------------------------------ lookups

    @property
    def items(self) -> tuple[CatalogItem, ...]:
        """Every item, presets first (in file order), then user-only items."""
        merged = dict(self.preset)
        merged.update(self.user)
        return tuple(merged.values())

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Ids that may go on the shopping list (no ``hand``)."""
        return tuple(it.id for it in self.items if not it.non_item)

    def item(self, item_id: str) -> CatalogItem | None:
        """Look up by exact id (the form the shopping list stores)."""
        return self.user.get(item_id) or self.preset.get(item_id)

    def resolve(self, name: str) -> CatalogItem | None:
        """Find an item by id, spoken name or alias; None if the name is new."""
        item_id = self._by_key.get(normalize_key(name))
        return self.item(item_id) if item_id else None

    def spoken(self, item_id: str) -> str:
        """The words for an id, falling back to a readable form of the id itself."""
        item = self.item(item_id)
        return item.spoken if item is not None else item_id.replace("_", " ")

    def spoken_map(self) -> dict[str, str]:
        """id -> spoken name for every entry, including non-item classes.

        This keeps the old ``load_classes`` contract (and therefore ``config/classes.yaml``)
        working unchanged.
        """
        return {it.id: it.spoken for it in self.items}

    def item_map(self) -> dict[str, str]:
        """id -> spoken name for addable items only (the shopping list's menu)."""
        return {it.id: it.spoken for it in self.items if not it.non_item}

    def prompts_for(self, item: CatalogItem | str) -> tuple[str, ...]:
        """Curated prompts, or one generated from the spoken name."""
        it = self.item(item) if isinstance(item, str) else item
        if it is None:
            raise CatalogError(f"unknown catalog item {item!r}")
        return it.prompts or default_prompts(it.spoken)

    def items_for(self, item_ids: list[str] | tuple[str, ...]) -> tuple[CatalogItem, ...]:
        """Catalog entries for a shopping list, synthesizing ones that are not in the catalog.

        A list can contain an id that is not in the catalog (an older list, or one written by
        hand). Synthesizing keeps the detector running: the item gets a generated prompt
        instead of crashing the server.
        """
        out: list[CatalogItem] = []
        for item_id in item_ids:
            item = self.item(item_id)
            if item is None:
                item = CatalogItem(id=item_id, spoken=item_id.replace("_", " "), preset=False)
            out.append(item)
        return tuple(out)

    def is_preset(self, item_id: str) -> bool:
        """True when the item is curated (so it has hand-written prompts)."""
        it = self.item(item_id)
        return bool(it is not None and it.preset)

    # ------------------------------------------------------------------ edits

    def resolve_or_create(self, name: str) -> CatalogItem:
        """Find the item, or create and save a user item for a new name.

        This is how free text becomes a real item: "blue Gatorade" -> id ``blue_gatorade``
        with a generated prompt. The entry is written to ``user_path`` immediately so a
        restart does not lose it.
        """
        found = self.resolve(name)
        if found is not None:
            return found
        item_id = slugify(name)
        existing = self.item(item_id)
        if existing is not None:              # same slug, different spelling
            return existing
        spoken = " ".join(str(name).split())  # tidy spacing, keep the user's case
        item = CatalogItem(id=item_id, spoken=spoken, prompts=(), preset=False)
        self.user[item_id] = item
        self._reindex()
        self.save_user()
        return item

    def save_user(self) -> None:
        """Write the user catalog atomically (no-op when there is no path)."""
        if self.user_path is None:
            return
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"items": {it.id: _entry_to_dict(it) for it in self.user.values()}}
        fd, tmp = tempfile.mkstemp(dir=self.user_path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("# Items added from the shopping-list page or `slr list add`.\n"
                    "# Prompts can be edited: better sentences give better detection.\n")
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        os.replace(tmp, self.user_path)
        self._remember_mtime()

    def changed_on_disk(self) -> bool:
        """True when another process (the CLI) edited the user catalog since we loaded it."""
        if self.user_path is None:
            return False
        try:
            mtime = self.user_path.stat().st_mtime
        except FileNotFoundError:
            return self._mtime is not None
        return mtime != self._mtime

    def reload_user(self) -> None:
        """Re-read the user file (after the CLI edited it) and rebuild the lookup."""
        if self.user_path is None:
            return
        self.user = (read_catalog_file(self.user_path, preset=False)
                     if self.user_path.exists() else {})
        self._reindex()
        self._remember_mtime()

    def _remember_mtime(self) -> None:
        """Record the user file's timestamp so ``changed_on_disk`` only sees outside edits."""
        try:
            self._mtime = self.user_path.stat().st_mtime if self.user_path else None
        except FileNotFoundError:
            self._mtime = None

    def set_prompts(self, item_id: str, prompts: list[str] | tuple[str, ...],
                    spoken: str | None = None) -> CatalogItem:
        """Replace an item's prompts and save the user catalog.

        A preset is never edited in place: saving for a preset creates a user entry that
        overrides it, so `config/catalog.yaml` stays a clean, versioned menu. An item that is
        not in the catalog at all (a stale shopping-list id) is created as a user item.
        """
        cleaned = tuple(" ".join(str(p).split()) for p in prompts if str(p).strip())
        if not item_id or slugify(item_id) != item_id:
            raise CatalogError(f"{item_id!r} is not a valid item id")
        if not cleaned:
            raise CatalogError("an item needs at least one prompt")
        if len(cleaned) > MAX_PROMPTS:
            raise CatalogError(f"use at most {MAX_PROMPTS} prompts")
        for prompt in cleaned:
            if len(prompt) > MAX_PROMPT_CHARS:
                raise CatalogError(f"a prompt must be at most {MAX_PROMPT_CHARS} characters")

        current = self.item(item_id)
        if current is None:
            item = CatalogItem(id=item_id, spoken=spoken or item_id.replace("_", " "),
                               prompts=cleaned, preset=False)
        else:
            item = replace(current, prompts=cleaned, preset=False)
        self.user[item_id] = item
        self._reindex()
        self.save_user()
        return item

    def restore_prompts(self, item_id: str) -> None:
        """Drop a user override, restoring the preset (or the generated prompt).

        For an item that only exists because the user added it, this removes the entry;
        ``items_for`` still synthesizes a generated-prompt entry if it stays on the list.
        """
        if item_id in self.user:
            del self.user[item_id]
            self._reindex()
            self.save_user()

    def _reindex(self) -> None:
        """Rebuild the name -> id lookup after any change."""
        by_key: dict[str, str] = {}
        for it in self.items:
            # setdefault keeps the first spelling, so a preset alias wins over a user alias.
            for key in (it.id, it.spoken, *it.aliases):
                by_key.setdefault(normalize_key(key), it.id)
            by_key.setdefault(normalize_key(it.id.replace("_", " ")), it.id)
        self._by_key = by_key


def _entry_to_dict(item: CatalogItem) -> dict:
    """The YAML form of a user item, omitting empty fields."""
    out: dict = {"spoken": item.spoken}
    if item.prompts:
        out["prompts"] = list(item.prompts)
    if item.aliases:
        out["aliases"] = list(item.aliases)
    if item.non_item:
        out["non_item"] = True
    if item.min_score is not None:
        out["min_score"] = item.min_score
    return out


def _parse_entry(item_id: str, value, path: Path, preset: bool) -> CatalogItem:
    """Turn one YAML entry into a CatalogItem, with clear errors for bad files."""
    if isinstance(value, str):
        # Legacy shape from config/classes.yaml:   pasta: pasta
        return CatalogItem(id=item_id, spoken=value, preset=preset)
    if not isinstance(value, dict):
        raise CatalogError(f"{path}: item {item_id!r} must be a name or a mapping")
    unknown = set(value) - ENTRY_KEYS
    if unknown:
        raise CatalogError(f"{path}: item {item_id!r} has unknown keys {sorted(unknown)}")
    spoken = value.get("spoken", item_id.replace("_", " "))
    if not isinstance(spoken, str) or not spoken.strip():
        raise CatalogError(f"{path}: item {item_id!r} needs a non-empty 'spoken'")
    prompts = value.get("prompts") or []
    aliases = value.get("aliases") or []
    for label, seq in (("prompts", prompts), ("aliases", aliases)):
        if not isinstance(seq, list) or not all(isinstance(s, str) and s.strip() for s in seq):
            raise CatalogError(f"{path}: item {item_id!r} {label} must be a list of text")
    min_score = value.get("min_score")
    if min_score is not None:
        try:
            min_score = float(min_score)
        except (TypeError, ValueError):
            raise CatalogError(f"{path}: item {item_id!r} min_score must be a number") from None
        if not 0 < min_score <= 1:
            raise CatalogError(f"{path}: item {item_id!r} min_score must be in (0, 1]")
    return CatalogItem(
        id=item_id,
        spoken=spoken.strip(),
        prompts=tuple(s.strip() for s in prompts),
        aliases=tuple(s.strip() for s in aliases),
        non_item=bool(value.get("non_item", False)),
        preset=preset,
        min_score=min_score,
    )


def read_catalog_file(path: Path, preset: bool) -> dict[str, CatalogItem]:
    """Read one catalog YAML file. Accepts the new ``items:`` block and the old ``classes:``.

    The old shape is what ``config/classes.yaml`` and ``config/classes_coco.yaml`` use:
    ``classes: {pasta: pasta}``. Keeping it readable means the migration needs no big-bang
    edit and the trained-model class files still load.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as e:
        raise CatalogError(f"catalog file not found: {path}") from e
    except yaml.YAMLError as e:
        raise CatalogError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(data, dict):
        raise CatalogError(f"{path}: expected a mapping at the top level")

    block = data.get("items")
    if block is None:
        block = data.get("classes")
    if not isinstance(block, dict) or not block:
        raise CatalogError(f"{path}: needs a non-empty 'items:' (or legacy 'classes:') mapping")

    out: dict[str, CatalogItem] = {}
    for raw_id, value in block.items():
        item_id = str(raw_id).strip()
        if not item_id:
            raise CatalogError(f"{path}: empty item id")
        out[item_id] = _parse_entry(item_id, value, path, preset)
    return out


def load_catalog(path: Path, user_path: Path | None = None) -> Catalog:
    """Load the preset catalog, then merge the user's additions on top.

    A missing user file is normal: it appears the first time somebody adds an item.
    """
    preset = read_catalog_file(Path(path), preset=True)
    user: dict[str, CatalogItem] = {}
    if user_path is not None and Path(user_path).exists():
        user = read_catalog_file(Path(user_path), preset=False)
    return Catalog(preset=preset, user=user, user_path=Path(user_path) if user_path else None)
