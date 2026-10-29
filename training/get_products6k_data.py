"""Convert the Products-6K grocery dataset into YOLO detection data for our 25 classes.

WHAT THIS FILE IS FOR
---------------------
Products-6K (zenodo.org/records/4428917, Georgiadis et al. 2021) is a product *recognition*
dataset from the Greek supermarket Masoutis. It was made to help visually impaired shoppers, like
our device, but it has no bounding boxes: every photo shows ONE product and the only labels are
text (product id, specific category, broad category, brand, description, mostly in Greek).

Our detector needs a box around every item and one of our 25 class ids (``config/classes.yaml``).
This script bridges the gap:

1. **Read the labels** from ``Candidate_Groundtruth.xlsx`` (and the query set's spreadsheet).
2. **Map each product to one of our classes** with the keyword rules in
   ``training/products6k_map.yaml`` ("Ελαιόλαδα" = olive oils -> ``oil``). Products that match
   nothing (detergent, biscuits, beer, ...) are left out.
3. **Draw the box automatically.** The candidate photos are studio shots on a plain white
   background, so the box is simply "everything that is not background" (no AI needed, exact).
   For photos without a plain background (the query set: products held in a hand in a real store)
   the open-vocabulary detector Grounding DINO finds the product (``--boxer auto``/``dino``,
   needs ``pip install transformers``; a GPU makes it fast).
4. **Write a YOLO dataset** with the same 25 class names and order as ``data/grocereye``, split
   train/val/test **by product**, so photos of one product never end up in both train and test.
5. **Write review sheets** (``review/sheet_*.jpg``): thumbnails with the boxes drawn, least
   certain first, so a human can spot-check the automatic boxes in a minute or two.

Output (paths relative to the repo root)::

    data/products6k/images/{train,val,test}/*.jpg
    data/products6k/labels/{train,val,test}/*.txt   # YOLO: class cx cy w h, normalized 0..1
    data/products6k/data.yaml                       # same names/order as config/classes.yaml
    data/products6k/manifest.json                   # every image: source, product, class, box, split
    data/products6k/boxes.json                      # box cache; edit it to fix or skip an image
    data/products6k/review/sheet_*.jpg              # thumbnails with boxes for spot checks

Then train on Freiburg + Products-6K together, starting from the Stage A weights::

    python training/train.py --name grocereye_p6k --model models/grocereye_a/best.pt \\
        --data data/grocereye/data.yaml --data data/products6k/data.yaml --epochs 60

Examples::

    python training/get_products6k_data.py --inspect     # what is there + how it maps, writes nothing
    python training/get_products6k_data.py               # uses data/Candidate_Set, data/Query_Set
    python training/get_products6k_data.py --download all --device cuda   # Colab: fetch everything
    python training/get_products6k_data.py --from-archive data/Query_Set.rar

LICENSE: Products-6K is CC BY-NC-ND 4.0: fine for this class project (non-commercial), but do not
share the converted copy (``data/`` is gitignored, keep it that way). Only 500 + 104 products are
public; the full 6,348 can be requested from the authors and this script handles it unchanged.

WORDS YOU WILL SEE
------------------
- **SKU / product id**: the store's number for one exact product (one brand, one size).
- **Candidate set / query set**: Products-6K's two halves. Candidate = clean studio photos (the
  store's catalogue). Query = phone photos of the product held in a hand inside the store.
- **Background box**: the box around all pixels that differ from the plain background color.
- **Grounding DINO**: a detector that finds objects described in words ("coffee. grocery
  product."), so it works on classes it was never trained on. We only use it to draw boxes.
- **Perceptual hash**: a 64-bit fingerprint of a tiny grayscale version of a photo. Near-identical
  photos (the same picture saved at two sizes) get near-identical fingerprints, so we keep one.
- **Stratified, grouped split**: products (not photos) are shuffled into train/val/test, and each
  class gets roughly the same share in every pile.
- **Normalize / fold text**: lowercase, remove accents, so "Καφές" and "ΚΑΦΕΣ" compare equal.
"""

# Lets type hints such as `list[str] | None` be written on older Python versions.
from __future__ import annotations

import argparse                    # command-line options
import collections                 # Counter / defaultdict
import csv                         # reading .csv annotation files
import hashlib                     # md5 checksums of the downloads
import json                        # manifest.json / boxes.json
import random                      # seeded sampling of background products
import re                          # text pattern matching
import shutil                      # copying files, finding programs
import subprocess                  # running unrar / 7z
import sys
import unicodedata                 # removing accents from Greek and Latin text
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# The repository root: this file is training/get_products6k_data.py, so go up two folders.
REPO_ROOT = Path(__file__).resolve().parent.parent
# Make the sibling script get_grocereye_data.py importable, whatever folder we are run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_grocereye_data import (SPLITS, download, normalize_class_name,  # noqa: E402
                                stratified_split, write_text)

ZENODO_RECORD = "https://zenodo.org/records/4428917"
# Downloadable archives and their md5 checksums, as listed on the Zenodo page.
ARCHIVES = {
    "candidate": ("Candidate_Set.rar", "b92bf54aa937f86e413da605ad62393b"),   # 110 MB
    "query": ("Query_Set.rar", "302bfb0e69131c29c01bc997379ab4b2"),           # 2.7 GB
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TABLE_EXTS = (".xlsx", ".xlsm", ".xls", ".csv")

# Loose header names for the annotation columns. A header matches when, after folding, it
# equals one of these, or contains one as a word ("Specific Category (SPC)" -> "specific").
COLUMN_ALIASES = {
    "sku": ["product id", "productid", "sku", "product sku", "id", "product code", "code"],
    "image": ["image", "image name", "image id", "file", "filename", "file name", "query image",
              "query id", "photo", "img"],
    "specific": ["specific category", "spc", "specific"],
    "broad": ["broad category", "bpc", "broad"],
    "brand": ["brand", "bc"],
    "description": ["description", "product description", "pd"],
}


# --------------------------------------------------------------------------------------------
# Pure helpers (unit tested in tests/test_products6k.py)
#
# "Pure" means these only compute things from their inputs: no files, network or clock.
# --------------------------------------------------------------------------------------------

def fold_text(text: object) -> str:
    """Lowercase, strip accents and punctuation: ``'Καφές φίλτρου!'`` -> ``'καφεσ φιλτρου'``.

    Greek final sigma (ς) becomes σ, so word forms compare equal. Words are separated by single
    spaces. ``None`` becomes ``''``.
    """
    if text is None:
        return ""
    # NFKD splits "έ" into "ε" + an accent mark; we then drop the accent marks (category "Mn").
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.casefold().replace("ς", "σ")
    # \w is any letter or digit in any alphabet; everything else becomes a space.
    return " ".join(re.sub(r"[^\w]+", " ", s).split())


@dataclass(frozen=True)
class Rule:
    """One keyword rule from products6k_map.yaml. ``cls`` None means "skip this product"."""
    cls: str | None
    keyword: str                 # folded keyword, without the trailing *
    pattern: re.Pattern


def make_rule(cls: str | None, keyword: str) -> Rule:
    """Compile a keyword: whole word (+ optional plural s/es), or word start if it ends in *."""
    prefix = keyword.strip().endswith("*")
    kw = fold_text(keyword.strip().rstrip("*"))
    if not kw:
        raise ValueError(f"empty keyword {keyword!r} for {cls}")
    # Words of a multi-word keyword may be separated by any single space in the folded text.
    body = r"\s".join(re.escape(w) for w in kw.split())
    tail = "" if prefix else r"(?:s|es)?(?!\w)"
    # (?<!\w) = "not preceded by a letter": the match must start at the beginning of a word.
    return Rule(cls, kw, re.compile(r"(?<!\w)" + body + tail))


def compile_rules(doc: dict, known_classes: list[str]) -> list[Rule]:
    """Turn the ``classes:`` and ``skip:`` sections of the map file into a list of Rules.

    Raises ValueError if a class id is not one of ``known_classes`` (our 25 ids).
    """
    known = set(known_classes)
    rules = []
    for cls, keywords in (doc.get("classes") or {}).items():
        cid = normalize_class_name(str(cls))
        if cid not in known:
            raise ValueError(f"products6k map: class {cls!r} is not in config/classes.yaml")
        for kw in keywords or []:
            rules.append(make_rule(cid, str(kw)))
    for kw in doc.get("skip") or []:
        rules.append(make_rule(None, str(kw)))
    return rules


@dataclass
class Match:
    """The winning rule for one piece of text, and where in the text it matched."""
    cls: str | None
    keyword: str
    start: int


def match_text(text: str, rules: list[Rule]) -> Match | None:
    """Best rule for ``text``: the match that starts earliest; ties go to the longest keyword.

    ``text`` must already be folded. Returns None when no rule matches.
    """
    best = None
    for r in rules:
        m = r.pattern.search(text)
        if not m:
            continue
        # Sort key: earlier start first, then longer keyword first (hence the minus sign).
        key = (m.start(), -len(r.keyword))
        if best is None or key < best[0]:
            best = (key, Match(r.cls, r.keyword, m.start()))
    return best[1] if best else None


def map_product(fields: list[tuple[str, str]], rules: list[Rule]) -> tuple[str | None, str]:
    """Pick our class for one product from its ``(field name, text)`` pairs, in priority order.

    The first field with any match decides. Returns ``(class id or None, reason)``; the reason is
    a short human-readable explanation such as ``"specific: 'καφε*'"`` or ``"no match"``.
    """
    for name, text in fields:
        m = match_text(fold_text(text), rules)
        if m is not None:
            why = f"{name}: '{m.keyword}'"
            return (m.cls, why) if m.cls else (None, "skip rule, " + why)
    return None, "no match"


def header_role(header: object, role: str) -> bool:
    """True if a spreadsheet header names the column ``role`` (see COLUMN_ALIASES)."""
    h = fold_text(header)
    if not h:
        return False
    words = set(h.split())
    for alias in COLUMN_ALIASES[role]:
        # Short aliases ("id", "pd", "bc") must be a whole word; long ones may appear anywhere.
        if h == alias or (alias in words) or (len(alias) > 3 and alias in h):
            return True
    return False


def find_columns(headers: list[object]) -> dict[str, int]:
    """Map each role (sku, specific, broad, ...) to a column index, first matching header wins.

    The ``sku`` role prefers a header that also says "product"/"sku" over a bare "id", so a
    query sheet with both "Query ID" and "Product ID" picks the product.
    """
    out: dict[str, int] = {}
    for role in COLUMN_ALIASES:
        hits = [i for i, h in enumerate(headers) if header_role(h, role)]
        if role == "sku":
            preferred = [i for i in hits if re.search(r"product|sku", fold_text(headers[i]))]
            hits = preferred or [i for i in hits if not header_role(headers[i], "image")]
        if role == "image":
            hits = [i for i in hits if i not in out.values()]
        if hits:
            out[role] = hits[0]
    return out


def parse_int(value: object) -> int | None:
    """An integer id from a cell or a string (``'0000059'`` -> 59, ``59.0`` -> 59), else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    m = re.fullmatch(r"\s*(\d+)(?:\.0+)?\s*", str(value))
    return int(m.group(1)) if m else None


def sku_from_filename(name: str) -> int | None:
    """Product id from a candidate file name: the leading digits.

    ``0000059_1.jpg`` -> 59, ``0019661-512.jpg`` -> 19661, ``0039461_IMG_5734_1.jpg`` -> 39461.
    """
    m = re.match(r"(\d+)", Path(name).name)
    return int(m.group(1)) if m else None


def image_keys(name: str) -> list[str]:
    """Ways a spreadsheet might refer to an image file: full name, stem, and number in the stem.

    ``'IMG_20200115_5.jpg'`` -> ``['img_20200115_5.jpg', 'img_20200115_5', '20200115_5']``;
    ``'Query_017.jpg'`` -> ``['query_017.jpg', 'query_017', '17']``.
    """
    p = Path(str(name).strip().replace("\\", "/"))
    keys = [p.name.lower(), p.stem.lower()]
    nums = re.findall(r"\d+", p.stem)
    if len(nums) == 1:
        keys.append(str(int(nums[0])))
    elif nums:
        keys.append("_".join(nums))
    # dict.fromkeys removes duplicates while keeping the order.
    return list(dict.fromkeys(keys))


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash of a grayscale image already resized to 9 wide x 8 tall.

    Each bit says whether a pixel is brighter than its right-hand neighbor.
    """
    g = np.asarray(gray, dtype=np.float32)
    if g.shape != (8, 9):
        raise ValueError(f"dhash needs an 8x9 array, got {g.shape}")
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return bin(a ^ b).count("1")


def _longest_run(active: np.ndarray, weights: np.ndarray, max_gap: int) -> tuple[int, int] | None:
    """Indices [start, end] of the heaviest run of True values, bridging gaps up to ``max_gap``."""
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return None
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    # The run that holds the most foreground pixels is the product; the others are specks.
    return max(runs, key=lambda r: weights[r[0]:r[1] + 1].sum())


def background_box(rgb: np.ndarray, tol: float = 30.0, border: float = 0.02,
                   min_border_bg: float = 0.6, min_area: float = 0.01
                   ) -> tuple[tuple[int, int, int, int], float] | None:
    """Box around everything that is not the plain background, for studio product photos.

    ``rgb`` is an H x W x 3 uint8 array. The background color is the median of a thin strip
    around the edges; a pixel is "product" when its color is more than ``tol`` away from it.
    Returns ``((x0, y0, x1, y1) in pixels, score)`` or None when the edges are not a plain
    background (fewer than ``min_border_bg`` of edge pixels look like background) or the box is
    tiny. The score is the fraction of edge pixels that are background (1.0 = perfectly plain).
    """
    img = np.asarray(rgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("background_box needs an H x W x 3 image")
    img = img[:, :, :3]
    h, w = img.shape[:2]
    b = max(1, int(round(min(h, w) * border)))
    # The four edge strips, flattened into one list of pixels.
    edge = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                           img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    bg = np.median(edge, axis=0)
    edge_is_bg = np.linalg.norm(edge - bg, axis=1) <= tol
    score = float(edge_is_bg.mean())
    if score < min_border_bg:
        return None
    mask = np.linalg.norm(img - bg, axis=2) > tol
    rows, cols = mask.sum(axis=1), mask.sum(axis=0)
    # Ignore rows/columns with only a few product pixels (JPEG noise, dust).
    row_run = _longest_run(rows >= max(2, 0.004 * w), rows, max(2, int(0.03 * h)))
    if row_run is None:
        return None
    y0, y1 = row_run
    cols = mask[y0:y1 + 1].sum(axis=0)
    col_run = _longest_run(cols >= max(2, 0.004 * (y1 - y0 + 1)), cols, max(2, int(0.03 * w)))
    if col_run is None:
        return None
    x0, x1 = col_run
    box = (int(x0), int(y0), int(x1) + 1, int(y1) + 1)
    if (box[2] - box[0]) * (box[3] - box[1]) < min_area * w * h:
        return None
    return box, score


def pick_box(candidates: list[tuple[float, float, float, float, float]], width: int, height: int,
             min_area: float = 0.01) -> tuple[float, float, float, float, float] | None:
    """Choose the held/featured product among detector boxes ``(x0, y0, x1, y1, score)``.

    Products-6K photos show one product up close, so big and confident wins: the box with the
    highest ``score * sqrt(area fraction)``. Boxes under ``min_area`` of the image are ignored.
    """
    best, best_val = None, -1.0
    for c in candidates:
        x0, y0, x1, y1, s = c
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / float(width * height)
        if area < min_area:
            continue
        val = s * area ** 0.5
        if val > best_val:
            best, best_val = c, val
    return best


def yolo_line(class_id: int, box: tuple[float, float, float, float]) -> str:
    """YOLO label line from a normalized corner box ``(x0, y0, x1, y1)``, clipped to 0..1."""
    x0, y0, x1, y1 = (min(max(float(v), 0.0), 1.0) for v in box)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return f"{class_id} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


def split_products(product_classes: dict[int, str], val_frac: float, test_frac: float,
                   seed: int) -> dict[int, str]:
    """Assign every product id to train/val/test, stratified by class (grouped split).

    Uses the same seeded algorithm as the Freiburg split, so the result is reproducible.
    """
    pairs = [(str(sku), cls) for sku, cls in product_classes.items()]
    split = stratified_split(pairs, val_frac, test_frac, seed)
    return {int(k): s for s, keys in split.items() for k in keys}


def safe_stem(text: str) -> str:
    """File-name-safe version of a string (letters, digits, - and _ only)."""
    return re.sub(r"[^0-9A-Za-z_-]+", "_", text).strip("_") or "img"


# --------------------------------------------------------------------------------------------
# Reading the dataset
# --------------------------------------------------------------------------------------------

@dataclass
class Product:
    """One product (SKU) and its text labels from the spreadsheet."""
    sku: int
    specific: str = ""
    broad: str = ""
    brand: str = ""
    description: str = ""

    def fields(self, order: list[str]) -> list[tuple[str, str]]:
        """``(field name, text)`` pairs in the order the map file asks for."""
        return [(f, getattr(self, f, "") or "") for f in order]


@dataclass
class Source:
    """One image file found on disk, linked to its product."""
    path: Path
    rel: str                     # path relative to the folder it was found in (cache key)
    subset: str                  # "candidate" or "query"
    sku: int | None
    link: str                    # how the product id was found


def read_table(path: Path) -> list[list[list[object]]]:
    """All sheets of a spreadsheet (.xlsx/.xls) or a .csv, as lists of rows of cell values."""
    ext = path.suffix.lower()
    if ext == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            return [list(csv.reader(f))]
    if ext in (".xlsx", ".xlsm"):
        try:
            import openpyxl
        except ImportError as e:
            raise SystemExit("Reading .xlsx needs openpyxl: pip install openpyxl") from e
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            return [[list(r) for r in ws.iter_rows(values_only=True)] for ws in wb.worksheets]
        finally:
            wb.close()
    if ext == ".xls":
        try:
            import xlrd
        except ImportError as e:
            raise SystemExit("Reading .xls needs xlrd: pip install xlrd") from e
        book = xlrd.open_workbook(str(path))
        return [[sh.row_values(i) for i in range(sh.nrows)] for sh in book.sheets()]
    raise ValueError(f"not a table: {path}")


@dataclass
class Tables:
    """Everything the annotation files say: products by id, and image -> product links."""
    products: dict[int, Product] = field(default_factory=dict)
    # (folder, image key) -> product id. Links only apply to images in the sheet's own folder,
    # so query photo "17.jpg" can never be confused with a candidate file.
    image_links: dict[tuple[str, str], int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def load_tables(roots: list[Path]) -> Tables:
    """Read every annotation spreadsheet under ``roots``.

    A sheet is used when its header row (one of the first 5 rows) has a product id column.
    If it also has an image column (a query sheet that numbers its photos), its rows link those
    images to products.
    """
    out = Tables()
    files = sorted({(str(r), p) for r in roots for p in r.rglob("*")
                    if p.suffix.lower() in TABLE_EXTS and p.is_file() and not p.name.startswith("~$")})
    for root, path in files:
        try:
            sheets = read_table(path)
        except Exception as e:  # a damaged or unrelated file should not stop the whole run
            out.notes.append(f"skipped {path.name}: {e}")
            continue
        for si, rows in enumerate(sheets):
            for hi, header in enumerate(rows[:5]):
                cols = find_columns(header)
                if "sku" in cols:
                    break
            else:
                continue
            n_prod = n_link = 0
            for row in rows[hi + 1:]:
                cell = lambda role: (row[cols[role]] if role in cols and cols[role] < len(row)  # noqa: E731
                                     else None)
                sku = parse_int(cell("sku"))
                if sku is None:
                    continue
                prod = out.products.setdefault(sku, Product(sku))
                for role in ("specific", "broad", "brand", "description"):
                    v = cell(role)
                    if v not in (None, "") and not getattr(prod, role):
                        setattr(prod, role, str(v).strip())
                n_prod += 1
                img = cell("image")
                if img not in (None, ""):
                    for k in image_keys(str(img)):
                        out.image_links.setdefault((root, k), sku)
                    n_link += 1
            colnames = ", ".join(f"{role}='{header[i]}'" for role, i in cols.items())
            out.notes.append(f"{path.name} sheet {si + 1}: {n_prod} rows, {n_link} image links ({colnames})")
    return out


def find_sources(roots: list[Path], tables: Tables) -> list[Source]:
    """Every image under ``roots``, linked to a product id.

    Linking order: (1) the spreadsheet names the image, (2) the file name starts with a product
    id that the spreadsheet knows. Images in a folder whose path says "query" are query photos.
    """
    out = []
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            # An archive may unpack into an extra folder of the same name (Candidate_Set/
            # Candidate_Set/...); drop it so boxes.json keys match on every machine.
            if rel.startswith(root.name + "/"):
                rel = rel[len(root.name) + 1:]
            subset = "query" if "query" in (root.name + "/" + rel).lower() else "candidate"
            sku, link = None, "unlinked"
            for k in image_keys(p.name):
                if (str(root), k) in tables.image_links:
                    sku, link = tables.image_links[(str(root), k)], "sheet"
                    break
            if sku is None:
                guess = sku_from_filename(p.name)
                if guess is not None and guess in tables.products:
                    sku, link = guess, "file name"
            out.append(Source(p, f"{root.name}/{rel}", subset, sku, link))
    return out


# --------------------------------------------------------------------------------------------
# Getting the archives
# --------------------------------------------------------------------------------------------

def md5sum(path: Path) -> str:
    """md5 checksum of a file, read in 1 MB chunks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def extract_rar(archive: Path, dest: Path) -> Path:
    """Unpack a .rar (or .zip) archive into ``dest`` with whichever tool this machine has.

    Tries unrar, 7z, then tar (Windows 10/11's built-in tar reads .rar), then the Python
    ``rarfile`` package. Raises SystemExit with manual instructions if none works.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(archive) as z:
            for m in z.namelist():
                if not (dest / m).resolve().is_relative_to(dest.resolve()):
                    raise RuntimeError(f"unsafe path in archive: {m}")
            z.extractall(dest)
        return dest
    print(f"Extracting {archive.name} -> {dest}")
    commands = [["unrar", "x", "-o+", "-idq", str(archive), str(dest) + "/"],
                ["7z", "x", "-y", "-bd", f"-o{dest}", str(archive)],
                ["7za", "x", "-y", "-bd", f"-o{dest}", str(archive)],
                ["tar", "-xf", str(archive), "-C", str(dest)]]
    for cmd in commands:
        if shutil.which(cmd[0]) is None:
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and any(dest.rglob("*")):
            return dest
    try:
        import rarfile
        with rarfile.RarFile(archive) as rf:
            rf.extractall(dest)
        return dest
    except Exception:
        pass
    raise SystemExit(f"Could not extract {archive}. Unpack it yourself (Windows 11: right-click -> "
                     "Extract All; or 7-Zip) into the data/ folder and run this script again.")


def fetch_archives(which: list[str], cache: Path, raw: Path) -> list[Path]:
    """Download (if needed), check and extract the Zenodo archives; returns the extracted roots."""
    roots = []
    for key in which:
        name, md5 = ARCHIVES[key]
        archive = cache / name
        if not archive.exists():
            download(f"{ZENODO_RECORD}/files/{name}?download=1", archive)
        got = md5sum(archive)
        if got != md5:
            raise SystemExit(f"{archive} is damaged (md5 {got}, expected {md5}); delete it and retry")
        dest = raw / archive.stem
        if not dest.exists() or not any(dest.rglob("*")):
            extract_rar(archive, dest)
        roots.append(dest)
    return roots


# --------------------------------------------------------------------------------------------
# Boxing
# --------------------------------------------------------------------------------------------

class DinoBoxer:
    """Grounding DINO through Hugging Face ``transformers`` (lazy import; only if needed)."""

    def __init__(self, model_id: str, device: str, box_threshold: float, text_threshold: float):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        print(f"Grounding DINO {model_id} on {device}")

    def detect(self, image, prompt: str) -> list[tuple[float, float, float, float, float]]:
        """Boxes ``(x0, y0, x1, y1, score)`` in pixels for a PIL image and a text prompt."""
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        size = [image.size[::-1]]  # (height, width)
        post = self.processor.post_process_grounded_object_detection
        try:   # transformers >= 4.51 calls it `threshold`, older versions `box_threshold`
            res = post(outputs, inputs.input_ids, threshold=self.box_threshold,
                       text_threshold=self.text_threshold, target_sizes=size)[0]
        except TypeError:
            res = post(outputs, inputs.input_ids, box_threshold=self.box_threshold,
                       text_threshold=self.text_threshold, target_sizes=size)[0]
        return [(*map(float, b.tolist()), float(s)) for b, s in zip(res["boxes"], res["scores"])]


def load_image(path: Path):
    """Open an image upright (phone photos store their rotation in EXIF) as RGB."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def compute_box(img, boxer: str, get_dino, prompt: str, subset: str = "candidate",
                sure_bg: float = 0.8) -> tuple[list[float] | None, str, float]:
    """Normalized box ``[x0, y0, x1, y1]`` for one PIL image, plus method name and score.

    ``boxer="auto"``: on candidate (studio) photos the plain-background box is used when the
    background is clearly plain (score >= ``sure_bg``). Otherwise (a gradient studio backdrop,
    or an in-store query photo, where a wall behind the hand could fool the background method)
    Grounding DINO decides; the less certain background box is only the fallback when DINO is
    not installed. ``get_dino`` returns a DinoBoxer or None and is only called when needed.
    """
    w, h = img.size
    if boxer == "full":
        return [0.0, 0.0, 1.0, 1.0], "full", 0.0
    bg = None
    if boxer == "background" or (boxer == "auto" and subset == "candidate"):
        small = img.copy()
        small.thumbnail((512, 512))   # boxes are fractions, so a small copy is enough (and fast)
        res = background_box(np.asarray(small), min_border_bg=0.5)
        if res is not None:
            (x0, y0, x1, y1), score = res
            sw, sh = small.size
            bg = ([x0 / sw, y0 / sh, x1 / sw, y1 / sh], "background", round(score, 3))
            if boxer == "background" or score >= sure_bg:
                return bg
    if boxer in ("auto", "dino"):
        dino = get_dino()
        if dino is not None:
            best = pick_box(dino.detect(img, prompt), w, h)
            if best is not None:
                x0, y0, x1, y1, s = best
                return [x0 / w, y0 / h, x1 / w, y1 / h], "dino", round(s, 3)
    return bg if bg is not None else (None, "none", 0.0)


def save_image(img, src: Path, dst: Path, max_side: int) -> None:
    """Write the training copy: the original file if it is already a small upright JPEG,
    otherwise a resized JPEG (phone photos are 3024 x 4032, far more than imgsz 320 needs)."""
    from PIL import Image
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as orig:
        # EXIF tag 274 = orientation; 1 (or missing) means the pixels are already upright.
        same = orig.size == img.size and orig.format == "JPEG" and orig.getexif().get(274, 1) == 1
    if same and max(img.size) <= max_side:
        shutil.copy2(src, dst)
        return
    img = img.copy()
    img.thumbnail((max_side, max_side))
    img.save(dst, "JPEG", quality=92)


def make_review_sheets(entries: list[dict], out_dir: Path, per_sheet: int = 30, thumb: int = 220) -> int:
    """Contact sheets of thumbnails with their boxes drawn; least certain first. Returns count.

    Green = background box, orange = Grounding DINO, gray = background-only image (no box).
    """
    from PIL import Image, ImageDraw
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    colors = {"background": (0, 170, 0), "dino": (255, 140, 0), "manual": (0, 90, 255), "full": (200, 0, 0)}
    order = sorted(entries, key=lambda e: (e["method"] != "dino", e["score"]))
    cols, label_h = 6, 28
    sheets = 0
    for start in range(0, len(order), per_sheet):
        chunk = order[start:start + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, e in enumerate(chunk):
            img = load_image(Path(e["dataset_image"]))
            img.thumbnail((thumb - 8, thumb - 8))
            ox = (i % cols) * thumb + (thumb - img.width) // 2
            oy = (i // cols) * (thumb + label_h) + (thumb - img.height) // 2
            sheet.paste(img, (ox, oy))
            if e.get("box"):
                x0, y0, x1, y1 = e["box"]
                draw.rectangle([ox + x0 * img.width, oy + y0 * img.height,
                                ox + x1 * img.width, oy + y1 * img.height],
                               outline=colors.get(e["method"], (255, 0, 0)), width=3)
            ty = (i // cols) * (thumb + label_h) + thumb
            short = {"background": "bg", "background_image": "no box"}.get(e["method"], e["method"])
            text = f"{e['class'] or 'background'}  {short} {e['score']:.2f}"
            draw.text(((i % cols) * thumb + 4, ty), text, fill="black")
            # The file name identifies the photo (the default font has no Greek letters).
            draw.text(((i % cols) * thumb + 4, ty + 13), Path(e["source"]).name[:36], fill=(90, 90, 90))
        sheet.save(out_dir / f"sheet_{sheets:03d}.jpg", quality=85)
        sheets += 1
    return sheets


# --------------------------------------------------------------------------------------------
# Main work
# --------------------------------------------------------------------------------------------

def load_class_list(classes_yaml: Path) -> list[str]:
    """Our class ids in model order (the order of config/classes.yaml = alphabetical)."""
    import yaml
    data = yaml.safe_load(classes_yaml.read_text(encoding="utf-8")) or {}
    classes = data.get("classes") or {}
    if not isinstance(classes, dict) or not classes:
        raise SystemExit(f"{classes_yaml} has no classes: mapping")
    return [str(k) for k in classes]


def load_map(path: Path, class_ids: list[str]) -> tuple[list[Rule], list[str], dict[int, str | None]]:
    """Read products6k_map.yaml: (rules, field order, per-product overrides)."""
    import yaml
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rules = compile_rules(doc, class_ids)
    fields = [str(f) for f in (doc.get("fields") or ["specific", "broad"])]
    bad = [f for f in fields if f not in ("specific", "broad", "brand", "description")]
    if bad:
        raise SystemExit(f"{path}: unknown fields {bad} (use specific, broad, brand, description)")
    overrides = {}
    for k, v in (doc.get("sku_overrides") or {}).items():
        cid = None if v in (None, "", "null") else normalize_class_name(str(v))
        if cid is not None and cid not in class_ids:
            raise SystemExit(f"{path}: sku_overrides {k}: {v!r} is not a class")
        overrides[int(k)] = cid
    return rules, fields, overrides


def classify(products: dict[int, Product], rules, fields, overrides) -> dict[int, tuple[str | None, str]]:
    """Our class (or None) and the reason, for every product."""
    out = {}
    for sku, p in products.items():
        if sku in overrides:
            out[sku] = (overrides[sku], "sku_overrides")
        else:
            out[sku] = map_product(p.fields(fields), rules)
    return out


def print_inspect(tables: Tables, sources: list[Source], mapping: dict, products: dict[int, Product]) -> None:
    """Report what was found and how every category maps, without writing anything."""
    print("\nAnnotation files:")
    for n in tables.notes:
        print("  " + n)
    by_subset = collections.Counter(s.subset for s in sources)
    linked = collections.Counter((s.subset, s.link) for s in sources)
    print(f"\nImages: {len(sources)} ({', '.join(f'{k} {v}' for k, v in sorted(by_subset.items()))})")
    for (sub, link), n in sorted(linked.items()):
        print(f"  {sub:9s} linked by {link}: {n}")
    unl = [s.rel for s in sources if s.sku is None]
    if unl:
        print(f"  e.g. unlinked: {', '.join(unl[:5])}")
    used = {s.sku for s in sources if s.sku is not None}
    imgs_per = collections.Counter(s.sku for s in sources if s.sku is not None)
    groups: dict[tuple, list[int]] = collections.defaultdict(list)
    for sku in used:
        p = products[sku]
        groups[(mapping[sku][0] or "-", p.broad, p.specific, mapping[sku][1])].append(sku)
    print("\nCategory mapping for products that have images (class | broad > specific | rule | products/images):")
    for (cls, broad, spec, why), skus in sorted(groups.items(), key=lambda kv: (kv[0][0] == "-", kv[0])):
        print(f"  {cls:13s} | {broad} > {spec} | {why} | {len(skus)}/{sum(imgs_per[s] for s in skus)}")
    per_class = collections.Counter(mapping[s][0] for s in used if mapping[s][0])
    print("\nProducts per class: " + ", ".join(f"{c} {n}" for c, n in sorted(per_class.items())))
    print(f"Mapped products: {sum(per_class.values())} of {len(used)} with images")


def convert(args: argparse.Namespace, roots: list[Path]) -> int:
    """Build the YOLO dataset. Returns a process exit code."""
    out = repo(args.out)
    class_ids = load_class_list(repo(args.classes_yaml))
    rules, fields, overrides = load_map(repo(args.map), class_ids)
    tables = load_tables(roots)
    if not tables.products:
        raise SystemExit("No annotation spreadsheet with a product id column found under: "
                         + ", ".join(map(str, roots)))
    sources = find_sources(roots, tables)
    mapping = classify(tables.products, rules, fields, overrides)
    if args.inspect:
        print_inspect(tables, sources, mapping, tables.products)
        return 0

    # --- which images to use -------------------------------------------------------------
    linked = [s for s in sources if s.sku is not None]
    mapped = {s.sku for s in linked if mapping[s.sku][0]}
    n_mapped_imgs = sum(1 for s in linked if s.sku in mapped)
    print(f"Found {len(sources)} images: {n_mapped_imgs} of products that map to our classes, "
          f"{len(linked) - n_mapped_imgs} of other products, {len(sources) - len(linked)} not "
          f"linked to any product ({len(tables.products)} products in the annotations)")
    # Background candidates: products that match no rule at all. Products dropped by a skip rule
    # (cereal bars, lemon juice, ...) look too much like a real class to be "nothing".
    unmapped = sorted({s.sku for s in linked} - mapped)
    unmapped = [k for k in unmapped if mapping[k][1] == "no match"]
    background: set[int] = set()
    if args.background_frac > 0 and unmapped:
        n_imgs = collections.Counter(s.sku for s in linked)
        want = args.background_frac * sum(n_imgs[k] for k in mapped)
        rng = random.Random(args.seed)
        rng.shuffle(unmapped)
        total = 0
        for sku in unmapped:
            if total >= want:
                break
            background.add(sku)
            total += n_imgs[sku]
    use = [s for s in linked if s.sku in mapped or s.sku in background]

    # Grouped split: products (not photos) go to train/val/test.
    strata = {sku: mapping[sku][0] or "_background" for sku in mapped | background}
    sku_split = split_products(strata, args.val_frac, args.test_frac, args.seed)

    # --- boxes ----------------------------------------------------------------------------
    cache_path = out / "boxes.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    spoken = {c: c.replace("_", " ") for c in class_ids}
    dino_state: dict = {}   # holds the loaded DinoBoxer, or None once loading has failed

    def get_dino() -> DinoBoxer | None:
        """Load Grounding DINO the first time a photo needs it."""
        if "boxer" not in dino_state:
            if args.no_dino:
                dino_state["boxer"] = None
            else:
                try:
                    dino_state["boxer"] = DinoBoxer(args.dino_model, args.device,
                                                    args.box_threshold, args.text_threshold)
                except ImportError:
                    if args.boxer == "dino":
                        raise SystemExit("--boxer dino needs: pip install transformers torch")
                    print("NOTE: transformers/torch not installed, so Grounding DINO is off: photos "
                          "without a clean plain background keep a less certain box or are left out.")
                    dino_state["boxer"] = None
        return dino_state["boxer"]

    for sub in ("images", "labels"):  # generated content only; safe to rebuild
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    entries, seen_hashes = [], collections.defaultdict(list)
    counts = collections.Counter()
    for i, src in enumerate(use, 1):
        cls = mapping[src.sku][0]
        try:
            img = load_image(src.path)
        except OSError as e:   # truncated or not really an image
            print(f"  skipping unreadable {src.rel}: {e}")
            counts["unreadable"] += 1
            continue
        # Near-duplicate check within one product (same photo saved twice or at two sizes).
        h = dhash(np.asarray(img.convert("L").resize((9, 8))))
        if any(hamming(h, o) <= args.dup_bits for o in seen_hashes[src.sku]):
            counts["duplicate"] += 1
            continue
        seen_hashes[src.sku].append(h)

        c = cache.get(src.rel, {})
        if c.get("skip"):
            counts["skipped (boxes.json)"] += 1
            continue
        box, method, score = None, "background_image", 1.0
        if cls is not None:
            # Reuse a cached box, except an uncertain plain-background one (DINO may be
            # installed by now) or when --rebox asks to redo them. Manual boxes always stay.
            unsure = c.get("method") == "background" and float(c.get("score", 1.0)) < 0.8
            if c.get("box") and (c.get("method") == "manual" or not (args.rebox or unsure)):
                box, method, score = c["box"], c.get("method", "manual"), float(c.get("score", 1.0))
            else:
                box, method, score = compute_box(img, args.boxer, get_dino,
                                                 args.dino_prompt.format(name=spoken[cls]), src.subset)
                if box is None:
                    counts["no box found"] += 1
                    cache[src.rel] = {"method": "none"}
                    continue
                box = [round(v, 5) for v in box]
                cache[src.rel] = {"box": box, "method": method, "score": score}
        split = sku_split[src.sku]
        # c/q = candidate or query photo, so equal file names in the two sets never collide.
        stem = f"p6k_{src.subset[0]}{src.sku:07d}_{safe_stem(Path(src.rel).stem)}"
        dst = out / "images" / split / (stem + ".jpg")
        save_image(img, src.path, dst, args.max_side)
        line = yolo_line(class_ids.index(cls), box) if cls is not None else ""
        write_text(out / "labels" / split / (stem + ".txt"), line + ("\n" if line else ""))
        counts[method] += 1
        p = tables.products[src.sku]
        entries.append({"source": src.rel, "subset": src.subset, "sku": src.sku, "class": cls,
                        "rule": mapping[src.sku][1], "split": split, "box": box, "method": method,
                        "score": score, "dataset_image": str(dst),
                        "specific": p.specific, "broad": p.broad, "description": p.description})
        if i % 200 == 0:
            print(f"  {i}/{len(use)} images")
            write_text(cache_path, json.dumps(cache, indent=1, ensure_ascii=False) + "\n")
    write_text(cache_path, json.dumps(cache, indent=1, ensure_ascii=False) + "\n")

    # --- data.yaml, manifest, review --------------------------------------------------------
    names_block = "".join(f"  - {c}\n" for c in class_ids)
    write_text(out / "data.yaml",
               "# Generated by training/get_products6k_data.py. Paths are relative to this file.\n"
               f"# Source: Products-6K {ZENODO_RECORD} (CC BY-NC-ND 4.0: do not share this copy)\n"
               "train: images/train\nval: images/val\ntest: images/test\n"
               f"nc: {len(class_ids)}\nnames:\n{names_block}")
    per_split = {s: collections.Counter(e["class"] or "_background" for e in entries if e["split"] == s)
                 for s in SPLITS}
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": ZENODO_RECORD, "roots": [str(r) for r in roots],
        "split": f"by product, stratified by class, val {args.val_frac} / test {args.test_frac}, seed {args.seed}",
        "boxer": args.boxer, "counts": dict(counts),
        "images_per_split": {s: sum(per_split[s].values()) for s in SPLITS},
        "images_per_class": {s: dict(sorted(per_split[s].items())) for s in SPLITS},
        "images": entries,
    }
    write_text(out / "manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    # split.json in the same shape as data/grocereye/split.json, so evaluate.py reports it (TR-2).
    write_text(out / "split.json", json.dumps({
        "created": manifest["created"], "source": f"Products-6K {ZENODO_RECORD}",
        "method": manifest["split"], "seed": args.seed, "val_frac": args.val_frac,
        "test_frac": args.test_frac, "classes": class_ids, "counts": manifest["images_per_split"],
        "instances_per_class": manifest["images_per_class"],
        "files": {s: sorted(Path(e["dataset_image"]).stem for e in entries if e["split"] == s)
                  for s in SPLITS},
    }, indent=1) + "\n")
    n_sheets = 0 if args.no_review else make_review_sheets(entries, out / "review")

    print(f"\nWrote {len(entries)} images: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("Per split: " + ", ".join(f"{s}={manifest['images_per_split'][s]}" for s in SPLITS))
    print("Per class (all splits): " + ", ".join(
        f"{c} {n}" for c, n in sorted(sum(per_split.values(), collections.Counter()).items())))
    print(f"Wrote {out / 'data.yaml'}\nWrote {out / 'manifest.json'}")
    unsure = sum(1 for e in entries if e["method"] == "background" and e["score"] < 0.8)
    if counts["no box found"] or (unsure and dino_state.get("boxer") is None):
        print(f"{counts['no box found']} photos got no box (left out) and {unsure} kept a less certain "
              "plain-background box. Install transformers + torch so Grounding DINO can box them, "
              "or fix them in boxes.json; they come first on the review sheets.")
    if n_sheets:
        print(f"Review the boxes: {out / 'review'} ({n_sheets} sheets, least certain first). "
              f"To fix one, edit its entry in {cache_path.name} (set \"skip\": true, or a \"box\" "
              "with \"method\": \"manual\") and rerun.")
    print("\nNext: python training/train.py --name grocereye_p6k --model models/grocereye_a/best.pt "
          f"--data data/grocereye/data.yaml --data {_rel(out / 'data.yaml')} --epochs 60")
    return 0


def repo(p: str | Path) -> Path:
    """Resolve a path relative to the repo root (absolute paths are kept)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(p: Path) -> str:
    """Show a path relative to the repo root when possible (nicer in printed messages)."""
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: find or fetch the data, then inspect or convert it."""
    # Greek category names must print even on consoles that are not UTF-8.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Dataset: " + ZENODO_RECORD)
    ap.add_argument("--from-dir", type=Path, action="append", metavar="DIR",
                    help="an extracted Candidate_Set / Query_Set folder (repeatable). Default: "
                         "data/Candidate_Set and data/Query_Set if they exist")
    ap.add_argument("--from-archive", type=Path, action="append", metavar="RAR",
                    help="a downloaded .rar to extract into data/ first (repeatable)")
    ap.add_argument("--download", choices=("candidate", "query", "all"),
                    help="download from Zenodo first: candidate (110 MB), query (2.7 GB) or all")
    ap.add_argument("--inspect", action="store_true",
                    help="print the files found and how every category maps; write nothing")
    ap.add_argument("--out", type=Path, default=Path("data/products6k"), help="output dir (default: %(default)s)")
    ap.add_argument("--map", type=Path, default=Path("training/products6k_map.yaml"),
                    help="category -> class rules (default: %(default)s)")
    ap.add_argument("--classes-yaml", type=Path, default=Path("config/classes.yaml"),
                    help="our class list, in model order (default: %(default)s)")
    ap.add_argument("--boxer", choices=("auto", "background", "dino", "full"), default="auto",
                    help="auto: plain-background box, else Grounding DINO (default); full: whole image")
    ap.add_argument("--no-dino", action="store_true", help="never load Grounding DINO (with --boxer auto)")
    ap.add_argument("--rebox", action="store_true", help="recompute boxes cached in boxes.json (manual boxes are kept)")
    ap.add_argument("--device", default="auto", help="Grounding DINO device: auto, cuda, cpu (default: %(default)s)")
    ap.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny",
                    help="Hugging Face model id (default: %(default)s; -base is slower, a bit better)")
    ap.add_argument("--dino-prompt", default="{name}. grocery product.",
                    help="text prompt; {name} = the class's spoken name (default: %(default)r)")
    ap.add_argument("--box-threshold", type=float, default=0.3, help="Grounding DINO box score floor")
    ap.add_argument("--text-threshold", type=float, default=0.25, help="Grounding DINO text score floor")
    ap.add_argument("--background-frac", type=float, default=0.0,
                    help="also add products that map to no class as background images (empty "
                         "labels), up to this fraction of the mapped images, e.g. 0.1 (default: off)")
    ap.add_argument("--val-frac", type=float, default=0.15, help="validation fraction of products (default: %(default)s)")
    ap.add_argument("--test-frac", type=float, default=0.15, help="test fraction of products (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="split seed (default: %(default)s)")
    ap.add_argument("--dup-bits", type=int, default=4,
                    help="photos of one product whose hashes differ in <= this many bits are duplicates")
    ap.add_argument("--max-side", type=int, default=1280, help="resize larger photos to this (default: %(default)s)")
    ap.add_argument("--no-review", action="store_true", help="skip the review contact sheets")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/downloads"),
                    help="where --download keeps the .rar files (default: %(default)s)")
    args = ap.parse_args(argv)

    roots = [repo(p) for p in (args.from_dir or [])]
    if args.download:
        which = ["candidate", "query"] if args.download == "all" else [args.download]
        roots += fetch_archives(which, repo(args.cache_dir), repo("data"))
    for a in args.from_archive or []:
        roots.append(extract_rar(repo(a), repo("data") / repo(a).stem))
    if not roots:
        roots = [repo("data") / n for n in ("Candidate_Set", "Query_Set") if (repo("data") / n).is_dir()]
    if not roots:
        print("No Products-6K data found. Either extract Candidate_Set.rar (and Query_Set.rar) into "
              f"data/, pass --from-dir, or use --download all. Files: {ZENODO_RECORD}")
        return 1
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        print(f"Not a folder: {missing[0]}")
        return 1
    print("Reading " + ", ".join(_rel(r) for r in roots))
    return convert(args, roots)


# Run main() only when this file is executed directly, not when tests import it.
if __name__ == "__main__":
    sys.exit(main())
