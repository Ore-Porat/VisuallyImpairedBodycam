"""Download the 5K Groceries detection dataset and convert it to Ultralytics YOLO format (TR-1, TR-2).

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
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import shutil
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_REPO = "https://github.com/aleksandar-aleksandrov/groceries-object-detection-dataset"
DEFAULT_URL = DATASET_REPO + "/archive/refs/heads/master.zip"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test", "testing": "test"}
CLASSES_HEADER = "# GrocerEye / Freiburg Groceries classes: class_id -> spoken name (FR-62).\n"


# --------------------------------------------------------------------------------------------
# Pure helpers (unit tested in tests/test_training_utils.py)
# --------------------------------------------------------------------------------------------

def normalize_class_name(name: str) -> str:
    """Normalize a label to a lowercase snake_case class id: ``'TOMATO SAUCE'`` -> ``'tomato_sauce'``."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()
    if not s:
        raise ValueError(f"class name {name!r} is empty after normalization")
    return s


def spoken_default(class_id: str) -> str:
    """Default spoken name for a class id: underscores become spaces, lowercased."""
    return class_id.replace("_", " ").strip().lower()


@dataclass
class VocBox:
    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class VocAnnotation:
    filename: str
    width: int
    height: int
    boxes: list[VocBox] = field(default_factory=list)


def parse_voc_xml(xml_text: str) -> VocAnnotation:
    """Parse a Pascal VOC annotation. Class names are normalized; width/height may be 0 if absent."""
    root = ET.fromstring(xml_text)
    filename = (root.findtext("filename") or "").strip()
    size = root.find("size")
    width = int(float(size.findtext("width") or 0)) if size is not None else 0
    height = int(float(size.findtext("height") or 0)) if size is not None else 0
    boxes = []
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
    """
    w = width or ann.width
    h = height or ann.height
    if not w or not h:
        raise ValueError(f"no image size for {ann.filename!r}")
    lines = []
    for b in ann.boxes:
        x0, x1 = sorted((min(max(b.xmin, 0.0), w), min(max(b.xmax, 0.0), w)))
        y0, y1 = sorted((min(max(b.ymin, 0.0), h), min(max(b.ymax, 0.0), h)))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        lines.append(f"{class_to_id[b.name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def stratified_split(items: list[tuple[str, str]], val_frac: float, test_frac: float,
                     seed: int) -> dict[str, list[str]]:
    """Split ``(key, stratum)`` pairs into train/val/test per stratum, deterministically.

    Each stratum (class) is sorted, shuffled with ``random.Random(seed)`` and cut so that
    ``round(n * test_frac)`` go to test and ``round(n * val_frac)`` to val. Same input + seed ->
    same split, independent of input order.
    """
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("need 0 <= val_frac, test_frac and val_frac + test_frac < 1")
    by_stratum: dict[str, list[str]] = collections.defaultdict(list)
    for key, stratum in items:
        by_stratum[stratum].append(key)
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    rng = random.Random(seed)
    for stratum in sorted(by_stratum):
        keys = sorted(by_stratum[stratum])
        rng.shuffle(keys)
        n_test = round(len(keys) * test_frac)
        n_val = round(len(keys) * val_frac)
        out["test"] += keys[:n_test]
        out["val"] += keys[n_test:n_test + n_val]
        out["train"] += keys[n_test + n_val:]
    return {s: sorted(v) for s, v in out.items()}


def detect_split(rel_path: Path) -> str | None:
    """Return 'train'/'val'/'test' if a directory component of the path names a split."""
    for part in rel_path.parts[:-1]:
        s = SPLIT_ALIASES.get(part.lower())
        if s:
            return s
    return None


def load_spoken_names(path: Path) -> dict[str, str]:
    """Read existing ``classes:`` mapping from a classes.yaml (empty dict if missing/invalid)."""
    if not path.exists():
        return {}
    import yaml
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    classes = data.get("classes") or {}
    return {str(k): str(v) for k, v in classes.items()} if isinstance(classes, dict) else {}


def _yaml_scalar(s: str) -> str:
    """Write a string as a YAML scalar, quoting only when needed."""
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _.'\-]*", s) and s.strip() == s and s.lower() not in (
            "yes", "no", "true", "false", "on", "off", "null", "y", "n"):
        return s
    return json.dumps(s)  # JSON strings are valid YAML double-quoted scalars


def render_classes_yaml(class_ids: list[str], existing: dict[str, str] | None = None,
                        header: str = CLASSES_HEADER) -> str:
    """Render config/classes.yaml text, keeping spoken names already present in ``existing``."""
    existing = existing or {}
    lines = [header if header.endswith("\n") else header + "\n", "classes:\n"]
    for cid in class_ids:
        spoken = existing.get(cid) or spoken_default(cid)
        lines.append(f"  {_yaml_scalar(cid)}: {_yaml_scalar(spoken)}\n")
    return "".join(lines)


def leading_comment_block(text: str) -> str:
    """Return the leading ``#`` comment lines of a file (used to keep a hand-edited header)."""
    out = []
    for line in text.splitlines(keepends=True):
        if line.startswith("#"):
            out.append(line if line.endswith("\n") else line + "\n")
        else:
            break
    return "".join(out)


# --------------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------------

def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text keeping the file's existing newline style (LF for new files).

    ``Path.write_text`` on Windows turns LF into CRLF; this keeps repo files byte-stable.
    """
    newline = "\n"
    if path.exists():
        with open(path, "rb") as f:
            if b"\r\n" in f.read(4096):
                newline = "\r\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)


def download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` with a simple progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            msg = f"{done / 1e6:.0f} MB" + (f" / {total / 1e6:.0f} MB" if total else "")
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
            for m in z.namelist():
                if not (dest / m).resolve().is_relative_to(dest.resolve()):
                    raise RuntimeError(f"unsafe path in archive: {m}")
            z.extractall(dest)
    else:
        import tarfile
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    return dest


@dataclass
class Sample:
    key: str                 # unique output stem
    image: Path
    xml: Path
    ann: VocAnnotation
    stratum: str             # majority class, for stratified splitting
    source_split: str | None


def find_samples(root: Path) -> list[Sample]:
    """Find every VOC XML under ``root`` and pair it with its image."""
    images_by_stem: dict[str, list[Path]] = collections.defaultdict(list)
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            images_by_stem[p.stem.lower()].append(p)

    samples, missing, used_keys = [], [], set()
    for xml in sorted(root.rglob("*.xml")):
        try:
            ann = parse_voc_xml(xml.read_text(encoding="utf-8", errors="replace"))
        except ET.ParseError:
            continue
        if not ann.boxes and not ann.filename:
            continue
        image = _match_image(xml, ann, images_by_stem)
        if image is None:
            missing.append(xml)
            continue
        counts = collections.Counter(b.name for b in ann.boxes)
        stratum = counts.most_common(1)[0][0] if counts else "_background"
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
    # 1) mirror path: .../annotations/<cls>/X.xml -> .../images/<cls>/X.<ext>
    parts = list(xml.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() in ("annotations", "annotation", "labels", "xml"):
            base = Path(*parts[:i], "images", *parts[i + 1:-1])
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
        same_split = [h for h in hits if detect_split(h) == detect_split(xml)]
        if len(same_split) == 1:
            return same_split[0]
    return None


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) of an image, via OpenCV (lazy import)."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img.shape[1], img.shape[0]


def find_dataset_root(path: Path) -> Path:
    """Descend through single-directory wrappers (e.g. ``repo-master/``) to the data."""
    while True:
        entries = [p for p in path.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir():
            path = entries[0]
        else:
            return path


def assign_splits(samples: list[Sample], mode: str, val_frac: float, test_frac: float,
                  seed: int) -> tuple[dict[str, list[str]], str]:
    """Return ({split: [keys]}, description of the method used)."""
    has_source = any(s.source_split for s in samples)
    if mode == "auto" and has_source:
        out = {s: [] for s in SPLITS}
        unassigned = []
        for s in samples:
            (out[s.source_split] if s.source_split else unassigned).append(s.key)
        out["train"] += unassigned
        method = "dataset's own split"
        if not out["test"]:
            # carve a test set out of train, sized test_frac of the whole dataset
            train = [(s.key, s.stratum) for s in samples if s.key in set(out["train"])]
            frac = min(0.9, test_frac * len(samples) / max(1, len(train)))
            carved = stratified_split(train, 0.0, frac, seed)
            out["train"], out["test"] = carved["train"], carved["test"]
            method += f"; test carved from train (stratified, seed {seed})"
        if not out["val"]:
            train = [(s.key, s.stratum) for s in samples if s.key in set(out["train"])]
            frac = min(0.9, val_frac * len(samples) / max(1, len(train)))
            carved = stratified_split(train, 0.0, frac, seed)
            out["train"], out["val"] = carved["train"], carved["test"]
            method += f"; val carved from train (stratified, seed {seed})"
        return {k: sorted(v) for k, v in out.items()}, method
    split = stratified_split([(s.key, s.stratum) for s in samples], val_frac, test_frac, seed)
    pct = lambda f: round(f * 100)  # noqa: E731
    return split, (f"stratified by class {pct(1 - val_frac - test_frac)}/{pct(val_frac)}/"
                   f"{pct(test_frac)} train/val/test, seed {seed}")


def convert(root: Path, out: Path, classes_yaml: Path, split_mode: str, val_frac: float,
            test_frac: float, seed: int, source_desc: str) -> dict:
    """Convert the VOC dataset under ``root`` into Ultralytics layout under ``out``."""
    samples = find_samples(root)
    if not samples:
        raise SystemExit(f"No Pascal VOC annotations with images found under {root}")
    class_ids = sorted({b.name for s in samples for b in s.ann.boxes})
    class_to_id = {c: i for i, c in enumerate(class_ids)}
    print(f"Found {len(samples)} annotated images, {len(class_ids)} classes: {', '.join(class_ids)}")

    split, method = assign_splits(samples, split_mode, val_frac, test_frac, seed)
    key_to_split = {k: s for s, keys in split.items() for k in keys}

    for sub in ("images", "labels"):  # generated content only; safe to rebuild
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    per_class = {s: collections.Counter() for s in SPLITS}
    n_boxes = 0
    for i, smp in enumerate(samples, 1):
        sp = key_to_split[smp.key]
        w = h = None
        if not smp.ann.width or not smp.ann.height:
            w, h = image_size(smp.image)
        lines = voc_to_yolo_lines(smp.ann, class_to_id, w, h)
        n_boxes += len(lines)
        shutil.copy2(smp.image, out / "images" / sp / (smp.key + smp.image.suffix.lower()))
        write_text(out / "labels" / sp / (smp.key + ".txt"),
                   "\n".join(lines) + ("\n" if lines else ""))
        per_class[sp].update(b.name for b in smp.ann.boxes)
        if i % 500 == 0:
            print(f"  converted {i}/{len(samples)}")

    data_yaml = out / "data.yaml"
    names_block = "".join(f"  - {c}\n" for c in class_ids)
    write_text(
        data_yaml,
        "# Generated by training/get_grocereye_data.py. Paths are relative to this file.\n"
        f"# Source: {source_desc}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(class_ids)}\nnames:\n{names_block}")

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
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Dataset: " + DATASET_REPO)
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

    resolve = lambda p: p if p.is_absolute() else REPO_ROOT / p  # noqa: E731
    out, cache, classes_yaml = resolve(args.out), resolve(args.cache_dir), resolve(args.classes_yaml)

    with tempfile.TemporaryDirectory(prefix="grocereye_") as tmp:
        if args.from_dir:
            root, desc = args.from_dir.resolve(), f"directory {args.from_dir}"
        else:
            if args.from_archive:
                archive, desc = args.from_archive.resolve(), f"archive {args.from_archive.name}"
            else:
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
        convert(root, out, classes_yaml, args.split, args.val_frac, args.test_frac, args.seed, desc)
    print(f"\nNext: python training/train.py --data {_rel(out / 'data.yaml')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
