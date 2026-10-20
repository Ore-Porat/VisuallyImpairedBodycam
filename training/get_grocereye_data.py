"""Download the 5K Groceries detection dataset and convert it to Ultralytics YOLO format (TR-1, TR-2).

WHAT THIS FILE IS FOR
---------------------
Before we can train the object detector (the neural network the laptop server uses to spot
grocery items, see README.md and training/README.md), we need a dataset: lots of photos, each with
a list of boxes saying "there is a pasta packet here". This script fetches a public grocery
dataset from GitHub, converts its box files into the simple text format the YOLO training library
expects, and sorts the photos into three piles: train, val and test (see the glossary below).
It also writes ``config/classes.yaml``, the list of item names the device can say out loud.

The requirement numbers in brackets (TR-1, TR-2, FR-62) refer to the product spec, PRD.pdf
("TR" = training requirement, "FR" = functional requirement).

GrocerEye (github.com/bhimar/GrocerEye) trained on the Freiburg Groceries images annotated with
bounding boxes by aleksandar-aleksandrov ("5K Groceries Dataset for Object Detection"):

    https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset

4,947 PNG images (224x224), 25 classes, one Pascal VOC XML per image (labelImg), laid out as
``dataset/images/<class>/<CLASS>0000.png`` and ``dataset/annotations/<class>/<CLASS>0000.xml``.
The ``master`` branch has no split. A ``train-val`` branch exists, but its val set is only ~3% of
the images (6 per class) and it has no test set, so by default we make our own stratified
80/10/10 train/val/test split with a fixed seed (GrocerEye itself used a random 80/20 split
made in Roboflow, which was not published). ``--split auto`` uses an existing split instead
when the input has ``train/``/``val/``/``test/`` folders (missing test is carved from train).

Output (paths relative to the repo root)::

    data/grocereye/images/{train,val,test}/*.png
    data/grocereye/labels/{train,val,test}/*.txt   # YOLO: class cx cy w h, normalized 0..1
    data/grocereye/data.yaml                       # names list order == class index
    data/grocereye/split.json                      # seed, method, counts, file lists
    config/classes.yaml                            # class_id -> spoken name (hand edits kept)

Examples::

    python training/get_grocereye_data.py                       # download + convert
    python training/get_grocereye_data.py --from-archive ~/Downloads/master.zip
    python training/get_grocereye_data.py --from-dir path/to/groceries-object-detection-dataset

Only the standard library and PyYAML are needed (OpenCV is used lazily if an XML has no size).

WORDS YOU WILL SEE
------------------
- **Dataset**: a folder of images plus, for each image, a "label file" listing the boxes in it.
- **Class**: a category the model can recognize, e.g. ``pasta`` or ``tomato_sauce``. Each class
  has a number (its "class id") because the model works with numbers, not names.
- **Bounding box / box**: a rectangle around one object in an image.
- **Pascal VOC XML**: one common file format for boxes (an XML text file per image, with pixel
  coordinates). This is what the downloaded dataset uses.
- **YOLO format**: the simpler box format the Ultralytics training library wants: one ``.txt``
  per image, one line per box: ``class_id center_x center_y width height``, all as fractions
  of the image size (0..1), so they do not depend on the image's pixel size.
- **Train / val / test split**: the model learns from the *train* photos; *val* photos are used
  during training to check progress; *test* photos are only looked at at the very end to get an
  honest score. Photos must never be in more than one pile.
- **Stratified split**: splitting so that every class has roughly the same share in each pile.
- **Seed**: the starting number for the random shuffle. Same seed = same "random" result every
  time, so the split can be reproduced exactly.
- **YAML**: a human-readable text format for settings (``key: value``). ``data.yaml`` tells the
  training library where the images are and what the classes are called.
- **JSON**: another text format for structured data; ``split.json`` records how the split was
  made.
- **@dataclass**: a Python shortcut for writing a small class that mostly just holds fields.
- **argparse**: the Python module that reads ``--options`` typed on the command line.
- **Path**: a Python object representing a file or folder path (from ``pathlib``).
"""

# Lets type hints such as `list[str] | None` be written on older Python versions. Type hints are
# notes about what kind of value a function expects/returns; Python does not enforce them.
from __future__ import annotations

import argparse                    # command-line option parsing
import collections                 # defaultdict and Counter, handy containers for counting
import json                        # read/write JSON text
import random                      # the seeded shuffle used for the split
import re                          # regular expressions (text pattern matching)
import shutil                      # copy files and delete folders
import sys                         # exit codes
import tempfile                    # a throw-away folder for extracting the download
import urllib.request              # download a file from a URL
import xml.etree.ElementTree as ET # parse the Pascal VOC XML annotation files
import zipfile                     # unpack .zip archives
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# The repository root: this file is training/get_grocereye_data.py, so go up two folders.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Where the dataset lives on GitHub and the direct link to its zip download.
DATASET_REPO = "https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset"
DEFAULT_URL = DATASET_REPO + "/archive/refs/heads/master.zip"
# File extensions we treat as images.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
# The three piles of the split, in the order we always list them.
SPLITS = ("train", "val", "test")
# Folder names people use for the piles, mapped to our canonical names.
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}
# First line written into config/classes.yaml when the file does not exist yet.
CLASSES_HEADER = "# GrocerEye / Freiburg Groceries classes: class_id -> spoken name (FR-62).\n"


# --------------------------------------------------------------------------------------------
# Pure helpers (unit tested in tests/test_training_utils.py)
#
# "Pure" means these functions only compute things from their inputs; they do not touch files,
# the network or the clock. That makes them easy to test.
# --------------------------------------------------------------------------------------------

def normalize_class_name(name: str) -> str:
    """Normalize a label to a lowercase snake_case class id: ``'TOMATO SAUCE'`` -> ``'tomato_sauce'``.

    Different tools write class names differently ("Tomato Sauce", "tomato-sauce", ...). This
    makes them all identical so they can be compared. Used all over the training scripts.
    """
    # re.sub replaces every run of characters that are NOT letters/digits with one underscore,
    # then leading/trailing underscores are stripped and the result lowercased.
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()
    if not s:
        raise ValueError(f"class name {name!r} is empty after normalization")
    return s


def spoken_default(class_id: str) -> str:
    """Default spoken name for a class id: underscores become spaces, lowercased.

    ``tomato_sauce`` -> ``tomato sauce``, which is what the device will say out loud.
    """
    return class_id.replace("_", " ").strip().lower()


# A @dataclass is a class whose only job is to hold a few named values. Python writes the
# constructor for us, so `VocBox("pasta", 1, 2, 3, 4)` just works.
@dataclass
class VocBox:
    """One box from a Pascal VOC file: class name plus pixel corners (left, top, right, bottom)."""
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocAnnotation:
    """Everything one VOC XML file says about one image: its file name, size and boxes."""
    filename: str
    width: int
    height: int
    # `field(default_factory=list)` gives each annotation its own fresh empty list.
    boxes: list[VocBox] = field(default_factory=list)


def parse_voc_xml(xml_text: str) -> VocAnnotation:
    """Parse a Pascal VOC annotation. Class names are normalized; width/height may be 0 if absent.

    Takes the text of one XML file and returns a ``VocAnnotation``. Called once per image by
    ``find_samples``.
    """
    root = ET.fromstring(xml_text)
    # findtext returns the text inside a tag, or None if the tag is missing; `or ""` handles None.
    filename = (root.findtext("filename") or "").strip()
    size = root.find("size")
    width = int(float(size.findtext("width") or 0)) if size is not None else 0
    height = int(float(size.findtext("height") or 0)) if size is not None else 0
    boxes = []
    # Each <object> element in the XML describes one labeled item.
    for obj in root.iter("object"):
        name = obj.findtext("name")
        bb = obj.find("bndbox")
        if name is None or bb is None:
            continue
        boxes.append(VocBox(normalize_class_name(name),
                            float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                            float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
    return VocAnnotation(filename, width, height, boxes)


def voc_to_yolo_lines(ann: VocAnnotation, class_to_id: dict[str, int],
                      width: int | None = None, height: int | None = None) -> list[str]:
    """Convert VOC pixel boxes to YOLO lines ``"cls cx cy w h"`` (normalized, clipped to the image).

    ``width``/``height`` override the XML size (use when the XML says 0). Degenerate boxes are dropped.

    ``class_to_id`` maps a class name to its number. Returns the lines that will be written to
    the image's ``.txt`` label file.
    """
    # `width or ann.width`: use the override if given, otherwise what the XML said.
    w = width or ann.width
    h = height or ann.height
    if not w or not h:
        raise ValueError(f"no image size for {ann.filename!r}")
    lines = []
    for b in ann.boxes:
        # Clamp each coordinate into the image (0..w, 0..h) and make sure x0 <= x1, y0 <= y1
        # even if the file had them swapped.
        x0, x1 = sorted((min(max(b.xmin, 0.0), w), min(max(b.xmax, 0.0), w)))
        y0, y1 = sorted((min(max(b.ymin, 0.0), h), min(max(b.ymax, 0.0), h)))
        # Skip boxes thinner than one pixel; they are labeling mistakes.
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        # YOLO wants the box CENTER and SIZE as fractions of the image size.
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        # `:.6f` prints a number with 6 decimal places.
        lines.append(f"{class_to_id[b.name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def stratified_split(items: list[tuple[str, str]], val_frac: float, test_frac: float,
                     seed: int) -> dict[str, list[str]]:
    """Split ``(key, stratum)`` pairs into train/val/test per stratum, deterministically.

    Each stratum (class) is sorted, shuffled with ``random.Random(seed)`` and cut so that
    ``round(n * test_frac)`` go to test and ``round(n * val_frac)`` to val. Same input + seed ->
    same split, independent of input order.

    ``items`` is a list of (image key, class name) pairs. Returns ``{"train": [...keys],
    "val": [...], "test": [...]}``.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("need 0 <= val_frac, test_frac and val_frac + test_frac < 1")
    # Group the keys by class. defaultdict(list) creates an empty list for any new class.
    by_stratum: dict[str, list[str]] = collections.defaultdict(list)
    for key, stratum in items:
        by_stratum[stratum].append(key)
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    # A private random generator with a fixed seed, so results are reproducible and do not
    # disturb (or depend on) Python's global random state.
    rng = random.Random(seed)
    for stratum in sorted(by_stratum):
        keys = sorted(by_stratum[stratum])   # sort first so input order does not matter
        rng.shuffle(keys)
        n_test = round(len(keys) * test_frac)
        n_val = round(len(keys) * val_frac)
        # Slice the shuffled list: first n_test to test, next n_val to val, the rest to train.
        out["test"] += keys[:n_test]
        out["val"] += keys[n_test:n_test + n_val]
        out["train"] += keys[n_test + n_val:]
    return {s: sorted(v) for s, v in out.items()}


def detect_split(rel_path: Path) -> str | None:
    """Return 'train'/'val'/'test' if a directory component of the path names a split.

    E.g. ``valid/images/x.png`` -> ``"val"``; ``images/pasta/x.png`` -> ``None``.
    """
    # parts[:-1] = every folder in the path, without the file name itself.
    for part in rel_path.parts[:-1]:
        s = SPLIT_ALIASES.get(part.lower())
        if s:
            return s
    return None


def load_spoken_names(path: Path) -> dict[str, str]:
    """Read existing ``classes:`` mapping from a classes.yaml (empty dict if missing/invalid).

    Used so that spoken names someone edited by hand survive a re-run of this script.
    """
    if not path.exists():
        return {}
    import yaml
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    classes = data.get("classes") or {}
    # Only accept a proper mapping; convert keys and values to plain strings.
    return {str(k): str(v) for k, v in classes.items()} if isinstance(classes, dict) else {}


def _yaml_scalar(s: str) -> str:
    """Write a string as a YAML scalar, quoting only when needed.

    Plain words like ``pasta`` can be written bare. Anything odd (or a word YAML would read as a
    boolean, like ``yes``) is wrapped in double quotes.
    """
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _.'\-]*", s) and s.strip() == s and s.lower() not in (
            "yes", "no", "true", "false", "on", "off", "null", "y", "n"):
        return s
    return json.dumps(s)  # JSON strings are valid YAML double-quoted scalars


def render_classes_yaml(class_ids: list[str], existing: dict[str, str] | None = None,
                        header: str = CLASSES_HEADER) -> str:
    """Render config/classes.yaml text, keeping spoken names already present in ``existing``.

    Produces lines like ``  tomato_sauce: tomato sauce`` under a ``classes:`` heading.
    """
    existing = existing or {}
    # Make sure the header ends with exactly one newline.
    lines = [header if header.endswith("\n") else header + "\n", "classes:\n"]
    for cid in class_ids:
        # Prefer the hand-edited spoken name; otherwise derive one from the id.
        spoken = existing.get(cid) or spoken_default(cid)
        lines.append(f"  {_yaml_scalar(cid)}: {_yaml_scalar(spoken)}\n")
    return "".join(lines)


def leading_comment_block(text: str) -> str:
    """Return the leading ``#`` comment lines of a file (used to keep a hand-edited header)."""
    out = []
    # keepends=True keeps the "\n" on each line so we can write them back unchanged.
    for line in text.splitlines(keepends=True):
        if line.startswith("#"):
            out.append(line if line.endswith("\n") else line + "\n")
        else:
            break
    return "".join(out)


# --------------------------------------------------------------------------------------------
# I/O
#
# From here on the functions touch the disk and the network.
# --------------------------------------------------------------------------------------------

def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text keeping the file's existing newline style (LF for new files).

    ``Path.write_text`` on Windows turns LF into CRLF; this keeps repo files byte-stable.

    This helper is imported by train.py, evaluate.py and pick_threshold.py too.
    """
    newline = "\n"
    if path.exists():
        # Peek at the first 4 KB of the existing file to see which line ending it uses.
        with open(path, "rb") as f:
            if b"\r\n" in f.read(4096):
                newline = "\r\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    # `with open(...) as f:` opens the file and guarantees it is closed afterwards.
    with open(path, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)


def download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` with a simple progress line.

    Writes to ``dest.part`` first and renames at the end, so a half-finished download is never
    mistaken for a complete one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n  -> {dest}")
    # Open the network connection and the output file at the same time.
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)   # size in bytes, if the server says
        done = 0
        # `:=` (the "walrus") assigns and tests in one go: read 1 MB, stop when nothing is left.
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            msg = f"{done / 1e6:.0f} MB" + (f" / {total / 1e6:.0f} MB" if total else "")
            # "\r" moves the cursor to the start of the line so the progress overwrites itself.
            print(f"\r  {msg}", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def extract_archive(archive: Path, dest: Path) -> Path:
    """Extract a .zip or .tar(.gz) archive into ``dest`` (refusing paths that escape it)."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {dest}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            # Safety check: a malicious zip could contain "../../etc/passwd". Refuse any entry
            # that would land outside the destination folder.
            for m in z.namelist():
                if not (dest / m).resolve().is_relative_to(dest.resolve()):
                    raise RuntimeError(f"unsafe path in archive: {m}")
            z.extractall(dest)
    else:
        import tarfile
        with tarfile.open(archive) as t:
            # filter="data" is tarfile's built-in protection against the same escape trick.
            t.extractall(dest, filter="data")
    return dest


@dataclass
class Sample:
    """One image + its annotation, as found on disk, plus the bookkeeping needed for the split."""
    key: str                 # unique output stem
    image: Path
    xml: Path
    ann: VocAnnotation
    stratum: str             # majority class, for stratified splitting
    source_split: str | None


def find_samples(root: Path) -> list[Sample]:
    """Find every VOC XML under ``root`` and pair it with its image.

    Walks the extracted dataset folder, parses each XML, finds the matching image file and
    returns one ``Sample`` per image. Images without a readable annotation are skipped.
    """
    # Index every image by its lowercase file name (without extension) for fast lookup.
    images_by_stem: dict[str, list[Path]] = collections.defaultdict(list)
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            images_by_stem[p.stem.lower()].append(p)

    samples, missing, used_keys = [], [], set()
    for xml in sorted(root.rglob("*.xml")):
        try:
            ann = parse_voc_xml(xml.read_text(encoding="utf-8", errors="replace"))
        except ET.ParseError:
            # Not a valid XML file; ignore it.
            continue
        if not ann.boxes and not ann.filename:
            continue
        image = _match_image(xml, ann, images_by_stem)
        if image is None:
            missing.append(xml)
            continue
        # The "stratum" used for balancing the split is the most common class in the image.
        counts = collections.Counter(b.name for b in ann.boxes)
        stratum = counts.most_common(1)[0][0] if counts else "_background"
        # Output file names must be unique; add _2, _3, ... if two images share a name.
        key = image.stem
        n = 1
        while key.lower() in used_keys:
            n += 1
            key = f"{image.stem}_{n}"
        used_keys.add(key.lower())
        samples.append(Sample(key, image, xml, ann, stratum, detect_split(xml.relative_to(root))))
    if missing:
        print(f"WARNING: {len(missing)} annotations have no matching image, e.g. {missing[0]}")
    return samples


def _match_image(xml: Path, ann: VocAnnotation, by_stem: dict[str, list[Path]]) -> Path | None:
    """Find the image file that belongs to an XML annotation, trying three strategies in order."""
    # 1) mirror path: .../annotations/<cls>/X.xml -> .../images/<cls>/X.<ext>
    parts = list(xml.parts)
    # Walk the path components from the end to find the "annotations" folder.
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() in ("annotations", "annotation", "labels", "xml"):
            base = Path(*parts[:i], "images", *parts[i + 1:-1])
            # Try the file name stored inside the XML first, then the XML's own name.
            for stem in filter(None, (Path(ann.filename).stem, xml.stem)):
                for ext in IMAGE_EXTS:
                    for cand in (base / (stem + ext), base / (stem + ext.upper())):
                        if cand.is_file():
                            return cand
            break
    # 2) next to the XML, 3) unique stem anywhere
    for stem in filter(None, (Path(ann.filename).stem, xml.stem)):
        for ext in IMAGE_EXTS:
            cand = xml.with_name(stem + ext)
            if cand.is_file():
                return cand
        hits = by_stem.get(stem.lower(), [])
        if len(hits) == 1:
            return hits[0]
        # Several images share the name: prefer the one in the same train/val/test folder.
        same_split = [h for h in hits if detect_split(h) == detect_split(xml)]
        if len(same_split) == 1:
            return same_split[0]
    return None


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) of an image, via OpenCV (lazy import).

    Only needed when an XML file does not record the image size.
    """
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read image {path}")
    # OpenCV stores images as (rows, columns, channels), i.e. (height, width, ...).
    return img.shape[1], img.shape[0]


def find_dataset_root(path: Path) -> Path:
    """Descend through single-directory wrappers (e.g. ``repo-master/``) to the data.

    A GitHub zip extracts to ``<name>-master/<files>``; this skips such lone wrapper folders.
    """
    while True:
        entries = [p for p in path.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir():
            path = entries[0]
        else:
            return path


def assign_splits(samples: list[Sample], mode: str, val_frac: float, test_frac: float,
                  seed: int) -> tuple[dict[str, list[str]], str]:
    """Return ({split: [keys]}, description of the method used).

    ``mode`` is "ours" (always make our own stratified split) or "auto" (reuse the dataset's
    own train/val/test folders when it has them, filling in any missing pile from train).
    """
    has_source = any(s.source_split for s in samples)
    if mode == "auto" and has_source:
        out = {s: [] for s in SPLITS}
        unassigned = []
        for s in samples:
            # Put each key into its folder's pile; images outside any split folder go to train.
            (out[s.source_split] if s.source_split else unassigned).append(s.key)
        out["train"] += unassigned
        method = "dataset's own split"
        if not out["test"]:
            # carve a test set out of train, sized test_frac of the whole dataset
            train = [(s.key, s.stratum) for s in samples if s.key in set(out["train"])]
            # Scale the fraction so test ends up test_frac of ALL images, not just of train.
            frac = min(0.9, test_frac * len(samples) / max(1, len(train)))
            carved = stratified_split(train, 0.0, frac, seed)
            out["train"], out["test"] = carved["train"], carved["test"]
            method += f"; test carved from train (stratified, seed {seed})"
        if not out["val"]:
            # Same trick for a missing val pile.
            train = [(s.key, s.stratum) for s in samples if s.key in set(out["train"])]
            frac = min(0.9, val_frac * len(samples) / max(1, len(train)))
            carved = stratified_split(train, 0.0, frac, seed)
            out["train"], out["val"] = carved["train"], carved["test"]
            method += f"; val carved from train (stratified, seed {seed})"
        return {k: sorted(v) for k, v in out.items()}, method
    # Default: our own stratified split of everything.
    split = stratified_split([(s.key, s.stratum) for s in samples], val_frac, test_frac, seed)
    # A lambda is a tiny one-line function; this one turns 0.1 into 10 (percent).
    pct = lambda f: round(f * 100)  # noqa: E731
    return split, (f"stratified by class {pct(1 - val_frac - test_frac)}/{pct(val_frac)}/"
                   f"{pct(test_frac)} train/val/test, seed {seed}")


def convert(root: Path, out: Path, classes_yaml: Path, split_mode: str, val_frac: float,
            test_frac: float, seed: int, source_desc: str) -> dict:
    """Convert the VOC dataset under ``root`` into Ultralytics layout under ``out``.

    This is the main worker: it finds the samples, decides the split, copies images, writes the
    YOLO label files, ``data.yaml``, ``split.json`` and ``config/classes.yaml``. Returns the
    dictionary that was written to ``split.json``.
    """
    samples = find_samples(root)
    if not samples:
        raise SystemExit(f"No Pascal VOC annotations with images found under {root}")
    # Class ids are assigned alphabetically: sorted unique names -> 0, 1, 2, ...
    class_ids = sorted({b.name for s in samples for b in s.ann.boxes})
    class_to_id = {c: i for i, c in enumerate(class_ids)}
    print(f"Found {len(samples)} annotated images, {len(class_ids)} classes: {', '.join(class_ids)}")

    split, method = assign_splits(samples, split_mode, val_frac, test_frac, seed)
    # Reverse lookup: image key -> which pile it is in.
    key_to_split = {k: s for s, keys in split.items() for k in keys}

    for sub in ("images", "labels"):  # generated content only; safe to rebuild
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    # Count how many boxes of each class land in each pile (for split.json).
    per_class = {s: collections.Counter() for s in SPLITS}
    n_boxes = 0
    # enumerate(..., 1) counts from 1 so the progress message reads naturally.
    for i, smp in enumerate(samples, 1):
        sp = key_to_split[smp.key]
        w = h = None
        if not smp.ann.width or not smp.ann.height:
            # The XML did not say how big the image is; read the file to find out.
            w, h = image_size(smp.image)
        lines = voc_to_yolo_lines(smp.ann, class_to_id, w, h)
        n_boxes += len(lines)
        # Copy the image into images/<split>/ and write its label file into labels/<split>/.
        shutil.copy2(smp.image, out / "images" / sp / (smp.key + smp.image.suffix.lower()))
        write_text(out / "labels" / sp / (smp.key + ".txt"),
                   "\n".join(lines) + ("\n" if lines else ""))
        per_class[sp].update(b.name for b in smp.ann.boxes)
        if i % 500 == 0:
            print(f"  converted {i}/{len(samples)}")

    # data.yaml: the file the Ultralytics training library reads to find everything.
    data_yaml = out / "data.yaml"
    names_block = "".join(f"  - {c}\n" for c in class_ids)
    write_text(
        data_yaml,
        "# Generated by training/get_grocereye_data.py. Paths are relative to this file.\n"
        f"# Source: {source_desc}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(class_ids)}\nnames:\n{names_block}")

    # split.json: a record of exactly how the split was made, for reproducibility (TR-2).
    info = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_desc,
        "method": method,
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "classes": class_ids,
        "counts": {s: len(split[s]) for s in SPLITS},
        "boxes": n_boxes,
        "instances_per_class": {s: dict(sorted(per_class[s].items())) for s in SPLITS},
        "files": split,
    }
    write_text(out / "split.json", json.dumps(info, indent=1) + "\n")

    # config/classes.yaml: regenerate it but keep any spoken names and header comments a human
    # has edited by hand.
    existing_text = classes_yaml.read_text(encoding="utf-8") if classes_yaml.exists() else ""
    existing = load_spoken_names(classes_yaml)
    dropped = sorted(set(existing) - set(class_ids))
    if dropped:
        print(f"WARNING: dropping classes not in the dataset from {classes_yaml}: {dropped}")
    header = leading_comment_block(existing_text) or CLASSES_HEADER
    classes_yaml.parent.mkdir(parents=True, exist_ok=True)
    write_text(classes_yaml, render_classes_yaml(class_ids, existing, header))

    print(f"\nSplit ({method}): " + ", ".join(f"{s}={info['counts'][s]}" for s in SPLITS))
    print(f"Wrote {data_yaml}\nWrote {out / 'split.json'}\nWrote {classes_yaml}")
    return info


def _rel(p: Path) -> str:
    """Show a path relative to the repo root when possible (nicer in printed messages)."""
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: read options, get the data, convert it. Returns an exit code."""
    # --- 1. Command-line options -----------------------------------------------------------
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Dataset: " + DATASET_REPO)
    # Exactly one of these three input sources may be given (the default is to download).
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--url", default=DEFAULT_URL, help="archive to download (default: %(default)s)")
    src.add_argument("--from-archive", type=Path, metavar="PATH",
                     help="use an already downloaded .zip/.tar.gz instead of downloading")
    src.add_argument("--from-dir", type=Path, metavar="PATH",
                     help="use an already extracted dataset directory")
    ap.add_argument("--out", type=Path, default=Path("data/grocereye"), help="output dir (default: %(default)s)")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/downloads"),
                    help="where the downloaded archive is kept (default: %(default)s)")
    ap.add_argument("--classes-yaml", type=Path, default=Path("config/classes.yaml"),
                    help="class_id -> spoken name file to regenerate (default: %(default)s)")
    ap.add_argument("--split", choices=("ours", "auto"), default="ours",
                    help="'ours': stratified split with --val-frac/--test-frac (default); 'auto': use the "
                         "input's train/val/test folders if present")
    ap.add_argument("--val-frac", type=float, default=0.1, help="validation fraction (default: %(default)s)")
    ap.add_argument("--test-frac", type=float, default=0.1, help="test fraction (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="split seed (default: %(default)s)")
    ap.add_argument("--keep-extracted", action="store_true",
                    help="keep the extracted archive under --cache-dir instead of a temp dir")
    args = ap.parse_args(argv)

    # Relative paths are taken from the repo root, whatever folder the terminal is in.
    resolve = lambda p: p if p.is_absolute() else REPO_ROOT / p  # noqa: E731
    out, cache, classes_yaml = resolve(args.out), resolve(args.cache_dir), resolve(args.classes_yaml)

    # --- 2. Get the raw dataset folder (download / unpack / use as-is) ----------------------
    # TemporaryDirectory creates a scratch folder that is deleted automatically at the end of
    # the `with` block, so the extracted archive does not pile up on disk.
    with tempfile.TemporaryDirectory(prefix="grocereye_") as tmp:
        if args.from_dir:
            root, desc = args.from_dir.resolve(), f"directory {args.from_dir}"
        else:
            if args.from_archive:
                archive, desc = args.from_archive.resolve(), f"archive {args.from_archive.name}"
            else:
                # Name the cached download after the URL's file name ("master.zip" is too
                # generic, so give it a descriptive name).
                archive = cache / Path(args.url.split("?")[0]).name
                if archive.name == "master.zip":
                    archive = cache / "groceries-object-detection-dataset-master.zip"
                if not archive.exists():
                    try:
                        download(args.url, archive)
                    except OSError as e:
                        print(f"Download failed: {e}\nDownload it manually from {DATASET_REPO} "
                              "(Code -> Download ZIP) and rerun with --from-archive PATH.")
                        return 1
                else:
                    print(f"Using cached {archive}")
                desc = args.url
            dest = cache / "extracted" / archive.stem if args.keep_extracted else Path(tmp)
            root = extract_archive(archive, dest)
        if not root.is_dir():
            print(f"Not a directory: {root}")
            return 1
        root = find_dataset_root(root)
        # --- 3. Convert --------------------------------------------------------------------
        convert(root, out, classes_yaml, args.split, args.val_frac, args.test_frac, args.seed, desc)
    print(f"\nNext: python training/train.py --data {_rel(out / 'data.yaml')}")
    return 0


# Standard Python idiom: run main() only when this file is executed directly, not when it is
# imported by train.py / evaluate.py / pick_threshold.py for its helper functions.
if __name__ == "__main__":
    sys.exit(main())
