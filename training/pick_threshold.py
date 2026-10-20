"""Pick the find-mode confidence threshold from held-out labeled photos (TR-14, NFR-4).

WHAT THIS FILE IS FOR
---------------------
When the object detector looks at a camera frame it does not simply say "pasta"; it says
"pasta, and I am 83% sure". The server (see README.md) only announces an item to the user when
that confidence is above a cut-off number, ``guidance.conf_find`` in ``config/server.yaml``.
Choosing that number is a trade-off: set it too low and the device will sometimes say "found"
when it is wrong (a false alarm, which is very bad for a blind shopper); set it too high and it
will take longer to find items, or never find some at all. This script measures the trade-off
on photos the model has never seen, prints a table, and recommends the cut-off. With ``--write``
it updates the server config for you.

The requirement numbers in brackets (TR-14, NFR-4) refer to the product spec, PRD.pdf
("TR" = training requirement, "NFR" = non-functional requirement).

Runs the model once over held-out images (low confidence floor), then sweeps thresholds
(default 0.20..0.95 in 0.05 steps). At each threshold, predictions are greedily matched to
ground truth of the same class at IoU >= 0.5 (highest confidence first, each box used once) and
per-class precision/recall are computed. The recommendation is the LOWEST threshold at which
every evaluated class has precision >= --min-precision (default 0.90): precision over recall,
because a wrong "found" is worse than a slower find.

Evaluated classes = classes with ground truth in the held-out set, or --classes (the demo
categories). A class with no predictions at a threshold has undefined precision; it is treated
as passing (no false positives) and its recall shows 0. False positives on background-only
images count against the predicted class.

Examples::

    python training/pick_threshold.py --weights models/grocereye_b/best.pt \\
        --data data/ours/data.yaml --split test --classes pasta rice cereal coffee tea --write
    python training/pick_threshold.py --weights models/grocereye_b/best.pt --images data/ours/images/test

``--write`` edits ``guidance.conf_find`` in config/server.yaml in place (comments and every other
line are kept byte-for-byte). A markdown report goes to reports/threshold_<name>_<date>.md.

WORDS YOU WILL SEE
------------------
- **Confidence**: the model's own 0..1 score for how sure it is about a detection.
- **Confidence threshold**: the cut-off below which detections are ignored.
- **Ground truth (GT)**: the boxes a human drew on the photo, i.e. the correct answer.
- **Prediction**: a box the model drew.
- **IoU (Intersection over Union)**: how much two boxes overlap, 0 (not at all) to 1 (identical).
  A prediction "matches" a ground-truth box when their IoU is at least 0.5.
- **TP / FP / FN**: true positive (a correct detection), false positive (the model drew a box
  where there was nothing, or the wrong class), false negative (a real item the model missed).
- **Precision**: of all the boxes the model drew for a class, what fraction were right.
  TP / (TP + FP). High precision = few false alarms.
- **Recall**: of all the real items of a class, what fraction the model found. TP / (TP + FN).
- **Confusion matrix**: a table of "true class" rows versus "predicted class" columns. The
  diagonal holds correct answers; everything else shows which classes get mixed up.
- **Held-out / test split**: photos deliberately kept away from training so scores are honest.
- **NMS (non-maximum suppression)**: the step inside the detector that removes duplicate boxes
  around the same object.
- **Sweep**: trying every threshold in a range and recording the results for each.
- **@dataclass**: a Python shortcut for a small class that just holds a few fields.
- **Regular expression (re)**: a mini-language for matching text patterns; used here to find
  one specific line in a YAML file without disturbing the rest.
- **argparse**: the Python module that reads ``--options`` from the command line.
- **Path**: a Python object representing a file/folder path (from ``pathlib``).
"""

# Lets us write modern type hints on older Python versions. Type hints (`: float`, `-> str`) are
# notes about what kind of value a function expects/returns; Python does not enforce them.
from __future__ import annotations

import argparse     # command-line options
import re           # regular expressions, for editing one line of server.yaml
import sys          # exit codes and the import path
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Make the sibling scripts in training/ importable regardless of the current folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# `noqa: E402` tells the style checker these imports are deliberately after the sys.path line.
from get_grocereye_data import normalize_class_name, write_text  # noqa: E402
from train import REPO_ROOT, label_path_for, list_images, load_data_yaml, repo_path  # noqa: E402

# A box is (cls, x1, y1, x2, y2) for ground truth and (cls, conf, x1, y1, x2, y2) for predictions,
# in any consistent coordinate units (we use normalized 0..1).
#
# In other words, throughout this file a box is a plain tuple: class id first, then (for
# predictions only) the confidence, then the left, top, right and bottom edges.


# --------------------------------------------------------------------------------------------
# Pure math (unit tested without a model)
#
# "Pure" = these functions only compute from their inputs: no files, no model, no network.
# --------------------------------------------------------------------------------------------

def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU of two xyxy boxes.

    Overlap area divided by the area covered by both boxes together. 1.0 means identical boxes,
    0.0 means they do not touch.
    """
    # Width and height of the overlapping rectangle (0 if the boxes do not overlap).
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    # Union = area(a) + area(b) - overlap (so the overlap is not counted twice).
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# A @dataclass is a class that mainly holds fields; Python writes the constructor for us.
@dataclass
class Counts:
    """Tally of true positives, false positives and false negatives for one class."""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    # @property lets you read `counts.precision` like a field even though it is computed.
    @property
    def precision(self) -> float | None:
        """None when there are no predictions (undefined)."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        """None when there is no ground truth."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None


def match_image(preds: list[tuple], gts: list[tuple], conf_thr: float,
                iou_thr: float = 0.5) -> dict[int, Counts]:
    """Per-class TP/FP/FN for one image at one confidence threshold.

    ``preds``: (cls, conf, x1, y1, x2, y2); ``gts``: (cls, x1, y1, x2, y2). Predictions with
    conf >= conf_thr are matched greedily by descending confidence to the unmatched same-class
    ground-truth box with the highest IoU (must be >= iou_thr).

    Returns ``{class_id: Counts}`` for this one image. Called by ``sweep`` for every image and
    every threshold, and by evaluate.py at the chosen operating point.
    """
    out: dict[int, Counts] = {}
    used = [False] * len(gts)   # which ground-truth boxes have already been matched
    # Keep only confident-enough predictions, most confident first. The lambda returns the
    # negative confidence so that sorted() (ascending) puts the highest confidence first.
    kept = sorted((p for p in preds if p[1] >= conf_thr), key=lambda p: -p[1])
    for p in kept:
        c = int(p[0])
        best, best_iou = -1, iou_thr
        # Look for the still-unused ground-truth box of the same class with the biggest overlap.
        for j, g in enumerate(gts):
            if used[j] or int(g[0]) != c:
                continue
            v = iou(p[2:6], g[1:5])
            if v >= best_iou:
                best, best_iou = j, v
        # setdefault returns the existing Counts for this class or creates a fresh one.
        cnt = out.setdefault(c, Counts())
        if best >= 0:
            used[best] = True
            cnt.tp += 1   # correct detection
        else:
            cnt.fp += 1   # false alarm
    # Any ground-truth box nobody matched is a miss (false negative).
    for j, g in enumerate(gts):
        if not used[j]:
            out.setdefault(int(g[0]), Counts()).fn += 1
    return out


def confusion_matrix(images: list[tuple[list[tuple], list[tuple]]], nc: int, conf_thr: float,
                     iou_thr: float = 0.5) -> list[list[int]]:
    """Class-agnostic matching -> ``m[true][pred]`` counts, with index ``nc`` = background.

    Per image, prediction/GT pairs with IoU >= iou_thr are matched one-to-one, highest IoU first,
    regardless of class. Matched -> m[gt][pred]; unmatched GT -> m[gt][nc] (missed);
    unmatched prediction -> m[nc][pred] (false alarm on background).

    ``images`` is a list of (predictions, ground truths) pairs, one per image; ``nc`` is the
    number of classes. The result is a square table with ``nc + 1`` rows and columns.
    """
    # Build an (nc+1) x (nc+1) grid of zeros. The extra row/column stands for "background"
    # (nothing there).
    m = [[0] * (nc + 1) for _ in range(nc + 1)]
    for preds, gts in images:
        kept = [p for p in preds if p[1] >= conf_thr]
        # Every (prediction, ground truth) pair with its IoU, biggest overlap first.
        pairs = sorted(((iou(p[2:6], g[1:5]), i, j) for i, p in enumerate(kept)
                        for j, g in enumerate(gts)), reverse=True)
        used_p, used_g = set(), set()
        for v, i, j in pairs:
            if v < iou_thr:
                break   # the list is sorted, so nothing after this overlaps enough either
            if i in used_p or j in used_g:
                continue
            used_p.add(i)
            used_g.add(j)
            m[int(gts[j][0])][int(kept[i][0])] += 1
        # Unmatched ground truth: row = true class, column = background ("missed").
        for j, g in enumerate(gts):
            if j not in used_g:
                m[int(g[0])][nc] += 1
        # Unmatched prediction: row = background, column = predicted class ("false alarm").
        for i, p in enumerate(kept):
            if i not in used_p:
                m[nc][int(p[0])] += 1
    return m


def thresholds_range(start: float, stop: float, step: float) -> list[float]:
    """Inclusive float range rounded to avoid drift: (0.2, 0.95, 0.05) -> [0.2, 0.25, ..., 0.95].

    Adding 0.05 repeatedly in floating point gives values like 0.30000000000000004; computing
    each value as start + i * step and rounding keeps them clean.
    """
    n = int(round((stop - start) / step))
    return [round(start + i * step, 6) for i in range(n + 1)]


def sweep(images: list[tuple[list[tuple], list[tuple]]], thresholds: list[float],
          iou_thr: float = 0.5) -> dict[float, dict[int, Counts]]:
    """Accumulate per-class counts over ``(preds, gts)`` images for each threshold.

    Returns ``{threshold: {class_id: Counts}}`` summed over all images.
    """
    result: dict[float, dict[int, Counts]] = {}
    for t in thresholds:
        acc: dict[int, Counts] = {}
        for preds, gts in images:
            for c, cnt in match_image(preds, gts, t, iou_thr).items():
                a = acc.setdefault(c, Counts())
                a.tp += cnt.tp
                a.fp += cnt.fp
                a.fn += cnt.fn
        result[t] = acc
    return result


def failing_classes(counts: dict[int, Counts], classes: list[int], min_precision: float,
                    min_recall: float = 0.0) -> list[int]:
    """Classes from ``classes`` that FAIL the targets (empty list = all pass).

    A class with no predictions has undefined precision (None) and is NOT counted as failing on
    precision, because it produced no false alarms.
    """
    failing = []
    for c in classes:
        cnt = counts.get(c, Counts())
        p = cnt.precision
        r = cnt.recall
        # `r or 0.0` treats an undefined recall (None) as 0 when a recall target is set.
        if (p is not None and p < min_precision) or (min_recall > 0 and (r or 0.0) < min_recall):
            failing.append(c)
    return failing


def recommend_threshold(result: dict[float, dict[int, Counts]], classes: list[int],
                        min_precision: float = 0.90, min_recall: float = 0.0) -> float | None:
    """Lowest threshold where every class in ``classes`` meets the targets, else None.

    Lowest is preferred because a lower threshold finds items sooner; we only go higher when
    needed to keep false alarms down.
    """
    for t in sorted(result):
        if not failing_classes(result[t], classes, min_precision, min_recall):
            return t
    return None


def set_conf_find(text: str, value: float, section: str = "guidance", key: str = "conf_find") -> str:
    """Return ``text`` (YAML) with ``<section>.<key>`` set to ``value``, editing only that line.

    Keeps indentation, trailing comments, other keys and newline style. If the key is missing it
    is inserted as the first entry of the section. Raises ValueError if the section is missing.

    We edit the raw text instead of loading and re-saving the YAML because a YAML library would
    drop every comment in config/server.yaml.
    """
    lines = text.splitlines(keepends=True)
    # Pattern for the section header line, e.g. "guidance:" (optionally followed by a comment).
    sec_re = re.compile(rf"^{re.escape(section)}\s*:\s*(#.*)?$")
    # Pattern for the key line, split into named pieces (indent, separator, value, trailing
    # comment, newline) so they can be reassembled with only the value replaced.
    key_re = re.compile(rf"^(?P<ind>[ \t]+){re.escape(key)}(?P<sep>\s*:\s*)(?P<val>[^#\r\n]*?)"
                        rf"(?P<rest>\s*(#[^\r\n]*)?)(?P<nl>\r?\n?)$")
    # Find the section header. next(..., None) gives the first match or None.
    start = next((i for i, l in enumerate(lines) if sec_re.match(l.rstrip("\r\n"))), None)
    if start is None:
        raise ValueError(f"no top-level '{section}:' section found")
    # The section ends at the next line that starts in column 0 (another top-level key).
    end = len(lines)
    for i in range(start + 1, len(lines)):
        s = lines[i]
        if s.strip() and not s[0].isspace() and not s.lstrip().startswith("#"):
            end = i
            break
    # Format the number compactly but always with a decimal point.
    formatted = f"{value:.2f}".rstrip("0")
    formatted += "0" if formatted.endswith(".") else ""  # 0.5 -> "0.5", 0.55 -> "0.55", 1 -> "1.0"
    child_indent = None
    for i in range(start + 1, end):
        m = key_re.match(lines[i])
        if m:
            # Found the key: rebuild the line with the new value and return immediately.
            lines[i] = (m.group("ind") + key + m.group("sep") + formatted
                        + m.group("rest") + m.group("nl"))
            return "".join(lines)
        # Remember the indentation of the first real child line, to use if we must insert.
        if child_indent is None and lines[i].strip() and not lines[i].lstrip().startswith("#"):
            child_indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    # Key not present: insert it right under the section header, matching the file's newlines.
    nl = "\r\n" if lines[start].endswith("\r\n") else "\n"
    lines.insert(start + 1, f"{child_indent or '  '}{key}: {formatted}{nl}")
    return "".join(lines)


def format_table(result: dict[float, dict[int, Counts]], classes: list[int], names: list[str],
                 min_precision: float) -> str:
    """Markdown table: one row per threshold, P/R per class, and whether all pass."""
    # A lambda is a one-line function: None -> "-", otherwise two decimals.
    fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
    head = "| conf | " + " | ".join(f"{names[c]} P/R" for c in classes) + " | all P>=target |"
    sep = "|" + "---|" * (len(classes) + 2)
    rows = [head, sep]
    for t in sorted(result):
        cells = [f"{fmt(result[t].get(c, Counts()).precision)}/{fmt(result[t].get(c, Counts()).recall)}"
                 for c in classes]
        ok = not failing_classes(result[t], classes, min_precision)
        rows.append(f"| {t:.2f} | " + " | ".join(cells) + f" | {'yes' if ok else 'no'} |")
    return "\n".join(rows)


# --------------------------------------------------------------------------------------------
# Model / data I/O (lazy imports)
#
# These touch the disk and run the model. Heavy libraries are imported inside the functions
# ("lazily") so the pure math above can be tested without them installed.
# --------------------------------------------------------------------------------------------

def read_yolo_labels(path: Path) -> list[tuple]:
    """YOLO label file -> [(cls, x1, y1, x2, y2)] normalized. Missing file = background image.

    Label files store center x/y and width/height; this converts them to corner coordinates.
    """
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 5:
            # `*map(float, ...)` converts the four coordinate strings to numbers in one go.
            c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
            out.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def gather_images(data: str | None, split: str, images_dir: str | None,
                  labels_dir: str | None) -> tuple[list[Path], list[Path], list[str] | None]:
    """Resolve (images, label files, names-from-yaml-or-None).

    Two ways to point at the photos: a data yaml plus a split name, or a plain folder of images
    (with labels next to them or in ``labels_dir``). Returns parallel lists: image i goes with
    label file i.
    """
    names = None
    if images_dir:
        imgs = list_images(repo_path(images_dir))
        if labels_dir:
            lbls = [repo_path(labels_dir) / (p.stem + ".txt") for p in imgs]
        else:
            lbls = [label_path_for(p) for p in imgs]
    else:
        d = load_data_yaml(data)
        names = d["names"]
        if not d.get(split):
            raise SystemExit(f"{data} has no '{split}' split")
        imgs = [p for e in d[split] for p in list_images(e)]
        lbls = [label_path_for(p) for p in imgs]
    return imgs, lbls, names


def predict_all(weights: str, images: list[Path], imgsz: int, device: str | None,
                conf_floor: float, iou_nms: float, batch: int = 16):
    """Run the model; returns (list of [(cls, conf, x1, y1, x2, y2)] normalized, model names).

    ``conf_floor`` is a very low confidence cut-off so that the sweep later can try any
    threshold above it without re-running the model. Images are processed ``batch`` at a time.
    """
    from ultralytics import YOLO
    model = YOLO(weights)
    preds = []
    # range(0, n, batch) steps through the list in chunks of `batch` images.
    for i in range(0, len(images), batch):
        chunk = [str(p) for p in images[i:i + batch]]
        kw = dict(imgsz=imgsz, conf=conf_floor, iou=iou_nms, verbose=False)
        if device is not None:
            kw["device"] = device
        for r in model.predict(chunk, **kw):
            b = r.boxes
            # xyxyn = corner coordinates normalized to 0..1. `.cpu().tolist()` moves the numbers
            # off the GPU into plain Python lists.
            xyxyn = b.xyxyn.cpu().tolist()
            preds.append([(int(c), float(s), *box) for c, s, box in
                          zip(b.cls.cpu().tolist(), b.conf.cpu().tolist(), xyxyn)])
    names = model.names
    return preds, [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def read_server_conf_find(path: Path) -> float | None:
    """Current guidance.conf_find from server.yaml (None if missing)."""
    import yaml
    try:
        return float(((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("guidance") or {})
                     .get("conf_find"))
    except (OSError, TypeError, ValueError):
        # File unreadable, key missing (None -> TypeError) or not a number.
        return None


def resolve_classes(requested: list[str] | None, names: list[str], present: set[int]) -> list[int]:
    """Class ids to evaluate: --classes (by name, normalized) or those present in ground truth."""
    if not requested:
        return sorted(present)
    index = {normalize_class_name(n): i for i, n in enumerate(names)}
    out = []
    for r in requested:
        key = normalize_class_name(r)
        if key not in index:
            raise SystemExit(f"unknown class {r!r}; model classes: {', '.join(names)}")
        out.append(index[key])
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: predict, sweep thresholds, write the report, optionally write config."""
    # --- 1. Command-line options -----------------------------------------------------------
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--weights", default="models/grocereye_b/best.pt", help="model .pt (default: %(default)s)")
    # --data and --images are alternatives; argparse refuses both at once.
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--data", default=None, help="data yaml (use with --split)")
    src.add_argument("--images", default=None, help="folder of held-out images (YOLO labels alongside)")
    ap.add_argument("--labels", default=None,
                    help="labels folder for --images (default: images->labels path convention)")
    ap.add_argument("--split", default="test", help="split of --data to use (default: %(default)s)")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="restrict to these classes (e.g. the demo categories)")
    ap.add_argument("--min-precision", type=float, default=0.90, help="per-class precision target")
    ap.add_argument("--min-recall", type=float, default=0.0, help="optional per-class recall target")
    ap.add_argument("--start", type=float, default=0.20, help="first threshold (default: %(default)s)")
    ap.add_argument("--stop", type=float, default=0.95, help="last threshold (default: %(default)s)")
    ap.add_argument("--step", type=float, default=0.05, help="threshold step (default: %(default)s)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU for a match (default: %(default)s)")
    ap.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU at predict time")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--server-config", default="config/server.yaml")
    ap.add_argument("--write", action="store_true", help="write the recommendation to guidance.conf_find")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--name", default=None, help="report name (default: weights' folder name)")
    args = ap.parse_args(argv)

    # --- 2. Load the photos and run the model once ------------------------------------------
    if not args.data and not args.images:
        args.data = "data/grocereye/data.yaml"
    weights = repo_path(args.weights)
    if not weights.exists():
        print(f"weights not found: {weights}")
        return 1
    imgs, lbls, yaml_names = gather_images(args.data, args.split, args.images, args.labels)
    if not imgs:
        print("no images found")
        return 1
    thresholds = thresholds_range(args.start, args.stop, args.step)
    print(f"Predicting on {len(imgs)} images ...")
    # Predict with a floor at or below the lowest threshold so every threshold can be evaluated
    # from this one pass.
    preds, names = predict_all(str(weights), imgs, args.imgsz, args.device,
                               min(0.05, thresholds[0]), args.nms_iou, args.batch)
    if yaml_names and [normalize_class_name(n) for n in yaml_names] != [normalize_class_name(n) for n in names]:
        print("WARNING: data yaml class names differ from the model's; using the model's ids.")
    gts = [read_yolo_labels(p) for p in lbls]
    # Set of class ids that actually appear in the ground truth.
    present = {g[0] for gt in gts for g in gt}
    classes = resolve_classes(args.classes, names, present)
    n_bg = sum(1 for g in gts if not g)   # images with no labeled objects at all

    # --- 3. Sweep the thresholds and pick one -----------------------------------------------
    # zip pairs each image's predictions with its ground truth.
    result = sweep(list(zip(preds, gts)), thresholds, args.iou)
    rec = recommend_threshold(result, classes, args.min_precision, args.min_recall)
    table = format_table(result, classes, names, args.min_precision)
    print("\n" + table + "\n")

    # --- 4. Write the markdown report -------------------------------------------------------
    name = args.name or weights.parent.name
    srv = repo_path(args.server_config)
    current = read_server_conf_find(srv)
    lines = [f"# Confidence threshold sweep: {name} ({date.today().isoformat()})", "",
             f"- Weights: `{weights.relative_to(REPO_ROOT).as_posix() if weights.is_relative_to(REPO_ROOT) else weights}`",
             f"- Images: {len(imgs)} ({n_bg} background-only) from "
             + (f"`{args.images}`" if args.images else f"`{args.data}` split `{args.split}`"),
             f"- Classes evaluated: {', '.join(names[c] for c in classes)}",
             f"- Match: same class, IoU >= {args.iou}; targets: precision >= {args.min_precision}"
             + (f", recall >= {args.min_recall}" if args.min_recall else ""),
             f"- Current guidance.conf_find: {current}", ""]
    if rec is None:
        # Nothing passed everywhere: report the threshold with the fewest failing classes.
        best = max(result, key=lambda t: -len(failing_classes(result[t], classes, args.min_precision)))
        failing = failing_classes(result[best], classes, args.min_precision, args.min_recall)
        msg = (f"**No threshold meets the target for every class.** Fewest failures at {best:.2f}: "
               f"{', '.join(names[c] for c in failing)}. Collect more data for these or drop them from the demo.")
    else:
        msg = f"**Recommended conf_find: {rec:.2f}** (lowest threshold where every class meets the target)."
        # Warn about classes that "pass" only because the model never detects them at all.
        blind = [names[c] for c in classes if not result[rec].get(c, Counts()).tp]
        if blind:
            msg += (f" WARNING: no correct detections at all for {', '.join(blind)} at this threshold; "
                    "the model is not finding them.")
    lines += [msg, "", table, "",
              "P/R '-' = undefined (no predictions / no ground truth). Undefined precision counts as passing.", ""]
    # Also list false alarms of classes we were not asked to evaluate, as a heads-up.
    at = result[rec if rec is not None else thresholds[0]]
    other_fp = {c: cnt.fp for c, cnt in sorted(at.items()) if c not in classes and cnt.fp}
    if other_fp:
        lines += ["False positives of non-evaluated classes at that threshold: "
                  + ", ".join(f"{names[c]}={n}" for c, n in other_fp.items()), ""]
    report = repo_path(args.reports_dir) / f"threshold_{name}_{date.today().isoformat()}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    write_text(report, "\n".join(lines))
    print(msg.replace("**", ""))
    print(f"Report: {report}")

    # --- 5. Optionally update config/server.yaml --------------------------------------------
    if args.write:
        if rec is None:
            print("Not writing server config: no threshold met the target.")
            return 2
        write_text(srv, set_conf_find(srv.read_text(encoding="utf-8"), rec))
        print(f"Set guidance.conf_find: {rec:.2f} in {srv} (was {current})")
    return 0


# Standard Python idiom: run main() only when this file is executed directly, not when
# evaluate.py imports it for its helper functions.
if __name__ == "__main__":
    sys.exit(main())
