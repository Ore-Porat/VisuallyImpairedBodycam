"""Measure how well the list-only CLIP scoring separates items, and pick thresholds.

WHY THIS SCRIPT EXISTS
----------------------
The detector no longer asks CLIP "which of these labels is it?" (a softmax, which always
picks something). It asks, for each item on the shopping list, "does this crop match this
item?" and compares the answer with a threshold. Two numbers decide the behaviour:

    min_score  the lowest cosine similarity that counts as a match
    margin     when 2+ items are on the list, how far the winner must lead the runner-up

Guessing those numbers is how you get a detector that calls every box "pasta". This script
measures them on real labelled photos, before any of it ships.

TWO MODES
---------
1. Labelled folder (the GrocerEye validation split)::

       python training/calibrate_items.py \
           --data data/remapped/grocereye --split val \
           --out reports/clip_item_calibration.md

   Images and YOLO-format labels (``class cx cy w h``, all 0..1). Only the ground-truth
   boxes are cropped, so this measures CLIP alone, not YOLO-World's box quality.

2. One folder per item, no labels (the brand test)::

       python training/calibrate_items.py --images data/brand_test \
           --out reports/brand_calibration.md

   ``data/brand_test/coke/*.jpg`` and ``data/brand_test/sprite/*.jpg``. Each whole photo is
   treated as one crop. This is how you answer "can it tell Coke from Sprite?" with a number
   instead of an opinion.

The report measures the detector's ``any`` prompt matrix (every curated prompt, averaged).
At runtime a box whose YOLO label is "can" is scored against "a can of X" variants as well;
on the GrocerEye sample those scores differ by only a few thousandths, so the calibrated
threshold still applies, but a larger difference would mean re-calibrating that variant.

WHAT THE REPORT CONTAINS
------------------------
- **1-item list** (the shipped case): for every crop, is its own item's score above the
  threshold? This is the min_score that matters.
- **2-item list**: how often the right item wins a head-to-head, and by how much. This is
  the margin, and it is the Coke-vs-Sprite answer.
- **all items at once**: a reference number, not the shipped mode. It shows how much harder
  the problem is when everything competes.
- per item: its own score distribution, the threshold that reaches the precision target,
  and whether it is separable at all.
- the worst confusion pairs.

WORDS YOU WILL SEE
------------------
- cosine similarity: the angle between two CLIP vectors, from -1 to 1. Same meaning -> ~0.3,
  unrelated -> ~0.1. It is not a probability.
- precision: of the crops we accepted, how many were the right item. This matters for a blind
  shopper: a false "found" sends them to the wrong shelf.
- recall: of all the crops, how many we accepted correctly. Lower recall means more
  "I don't see it", which is the safe failure.
- Gt (ground truth): the box a human labelled.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Running from the repo root puts ``slr`` on the path; this also works from training/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slr.common.catalog import Catalog, CatalogItem, load_catalog, slugify  # noqa: E402
from slr.server.detector import pick_device  # noqa: E402
from slr.server.fast_detector import model_dtype_inputs, prepare_crop  # noqa: E402


@dataclass
class Sample:
    """One crop and the item it really is."""

    item_id: str
    crop: np.ndarray


# --------------------------------------------------------------------------- inputs


def load_dataset(data: Path, split: str, item_ids: list[str], limit: int | None) -> list[Sample]:
    """Crops from a YOLO-format folder (images/<split>, labels/<split>)."""
    import cv2

    images_dir = data / "images" / split
    labels_dir = data / "labels" / split
    if not images_dir.is_dir():
        raise SystemExit(f"no images folder: {images_dir}")

    samples: list[Sample] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            class_index = int(float(parts[0]))
            if not 0 <= class_index < len(item_ids):
                print(f"warning: {label_path}: class {class_index} is outside the catalog")
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            box = ((cx - bw / 2) * width, (cy - bh / 2) * height,
                   (cx + bw / 2) * width, (cy + bh / 2) * height)
            crop = prepare_crop(image, box, 0.10)
            if crop is not None:
                samples.append(Sample(item_ids[class_index], crop))
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_image_folders(images: Path, catalog: Catalog, limit: int | None) -> list[Sample]:
    """Crops from a folder of item subfolders, e.g. ``brand_test/coke/*.jpg``."""
    import cv2

    if not images.is_dir():
        raise SystemExit(f"no image folder: {images}")
    samples: list[Sample] = []
    for folder in sorted(images.iterdir()):
        if not folder.is_dir():
            continue
        resolved = catalog.resolve(folder.name)
        item_id = resolved.id if resolved is not None else slugify(folder.name)
        count = 0
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            samples.append(Sample(item_id, image))
            count += 1
            if limit is not None and len(samples) >= limit:
                return samples
        print(f"  {folder.name}: {count} photos -> {item_id}")
    return samples


# --------------------------------------------------------------------------- CLIP


def build_text_bank(catalog: Catalog, items: list[CatalogItem], processor, model, device: str):
    """One normalized CLIP vector per item, averaged over its prompts.

    Returns ``(ids, matrix)`` with matrix shape ``(n_items, dim)``.
    """
    import torch
    import torch.nn.functional as F

    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for item in items:
        prompts = catalog.prompts_for(item)
        start = len(texts)
        texts.extend(prompts)
        spans.append((start, len(texts)))
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
        features = model.get_text_features(**model_dtype_inputs(inputs, model))
        # Transformers versions differ: some return a pooled output object, others a tensor.
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = F.normalize(features, dim=-1)
    rows = [F.normalize(features[start:end].mean(dim=0), dim=-1) for start, end in spans]
    return [item.id for item in items], torch.stack(rows)


def embed_crops(processor, model, crops: list[np.ndarray], device: str,
                batch_size: int) -> np.ndarray:
    """Normalized CLIP image embeddings, shape ``(n_crops, dim)``."""
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image

    out: list[np.ndarray] = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start:start + batch_size]
        images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in batch]
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = model.get_image_features(**model_dtype_inputs(inputs, model))
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = F.normalize(features, dim=-1)
        out.append(features.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


def score_samples(samples: list[Sample], item_ids: list[str], text_bank: np.ndarray,
                  processor, model, device: str, batch_size: int):
    """Cosine score matrix ``(n_crops, n_items)`` plus the index of each crop's true item."""
    index = {item_id: i for i, item_id in enumerate(item_ids)}
    missing = sorted({s.item_id for s in samples if s.item_id not in index})
    if missing:
        raise SystemExit(f"items with no prompt: {missing}")
    true_idx = np.array([index[s.item_id] for s in samples], dtype=int)
    image_features = embed_crops(processor, model, [s.crop for s in samples], device, batch_size)
    scores = image_features @ text_bank.T          # (n_crops, n_items)
    return scores, true_idx


# --------------------------------------------------------------------------- metrics


def pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")


def fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def one_item_sweep(scores: np.ndarray, true_idx: np.ndarray, grid: np.ndarray) -> list[dict]:
    """Metrics for a list with ONE item: is this crop that item, yes or no?

    For every item and every crop we ask "does this crop score above the threshold for this
    item?". A crop of the right item that passes is a true positive; a crop of any other item
    that passes is a false positive. That is exactly the shipped 1-item-list decision.
    """
    n_items = scores.shape[1]
    belongs = true_idx[:, None] == np.arange(n_items)[None, :]     # (n_crops, n_items)
    rows: list[dict] = []
    for threshold in grid:
        accepted = scores >= threshold
        tp = int(np.sum(accepted & belongs))
        fp = int(np.sum(accepted & ~belongs))
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / len(true_idx) if len(true_idx) else float("nan")
        rows.append({"threshold": float(threshold), "precision": precision,
                     "recall": recall, "tp": tp, "fp": fp})
    return rows


def choose_threshold(rows: list[dict], min_precision: float) -> dict:
    """Lowest threshold that still reaches ``min_precision`` (more recall for free).

    Ties are broken towards the lower threshold, then the higher recall. When nothing reaches
    the target, the best-F1 row is returned so the report can show what the ceiling looks like.
    """
    passing = [r for r in rows if r["tp"] and r["precision"] >= min_precision]
    if passing:
        return passing[0]
    return best_f1_row(rows)


def best_f1_row(rows: list[dict]) -> dict:
    """The most balanced threshold (used when no threshold reaches the precision target)."""
    def f1(row):
        p, r = row["precision"], row["recall"]
        return 2 * p * r / (p + r) if p + r else 0.0
    return max(rows, key=f1)


def all_items_sweep(scores: np.ndarray, true_idx: np.ndarray) -> dict:
    """Reference: every item competes (the old behaviour). Not the shipped mode."""
    pred = scores.argmax(axis=1)
    correct = pred == true_idx
    return {"accuracy": float(correct.mean()) if len(true_idx) else 0.0,
            "n": len(true_idx)}


def per_item_rows(scores: np.ndarray, true_idx: np.ndarray, item_ids: list[str],
                  grid: np.ndarray, min_precision: float) -> list[dict]:
    """For each item: its own scores, and the threshold that reaches the precision target."""
    out: list[dict] = []
    for index, item_id in enumerate(item_ids):
        mine = true_idx == index
        if not mine.any():
            continue
        own = scores[mine, index]
        other = scores[~mine, index]
        best = None
        for threshold in grid:
            tp = int((own >= threshold).sum())
            fp = int((other >= threshold).sum())
            precision = tp / (tp + fp) if tp + fp else float("nan")
            if tp and precision >= min_precision:
                best = {"threshold": float(threshold), "precision": precision,
                        "recall": tp / len(own), "fp": fp}
                break
        out.append({
            "id": item_id,
            "n": int(mine.sum()),
            "own_p10": pct(own, 10),
            "own_p50": pct(own, 50),
            "own_p90": pct(own, 90),
            "other_p90": pct(other, 90),
            "threshold": best,
        })
    return out


def pairwise_rows(scores: np.ndarray, true_idx: np.ndarray, item_ids: list[str],
                  min_crops: int = 10) -> list[dict]:
    """Head-to-head accuracy for every pair of items (the 2-item list case).

    For crops of A and B, the pair is "which one is this?" only. Accuracy is how often the
    true item's score is the higher one. Lead is how far ahead it is.
    """
    out: list[dict] = []
    for i in range(len(item_ids)):
        for j in range(i + 1, len(item_ids)):
            mask = (true_idx == i) | (true_idx == j)
            if int(mask.sum()) < min_crops:
                continue
            t = true_idx[mask]
            s_i = scores[mask, i]
            s_j = scores[mask, j]
            true_score = np.where(t == i, s_i, s_j)
            other_score = np.where(t == i, s_j, s_i)
            correct = true_score > other_score
            out.append({
                "a": item_ids[i], "b": item_ids[j],
                "n": int(mask.sum()),
                "accuracy": float(correct.mean()),
                "lead_p10": pct(true_score - other_score, 10),
                "lead_p50": pct(true_score - other_score, 50),
                "wrong_winner": int((~correct).sum()),
            })
    return out


def confusion_pairs(scores: np.ndarray, true_idx: np.ndarray, item_ids: list[str],
                    threshold: float, top: int = 10) -> list[tuple[str, str, int]]:
    """The wrong labels CLIP gives above the threshold, most common first."""
    pred = scores.argmax(axis=1)
    best = scores.max(axis=1)
    wrong = (best >= threshold) & (pred != true_idx)
    counts: dict[tuple[int, int], int] = {}
    for t, p in zip(true_idx[wrong], pred[wrong]):
        counts[(int(t), int(p))] = counts.get((int(t), int(p)), 0) + 1
    pairs = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return [(item_ids[t], item_ids[p], n) for (t, p), n in pairs]


# --------------------------------------------------------------------------- report


def write_report(path: Path, *, title: str, source: str, n_crops: int, item_ids: list[str],
                 sweep: list[dict], recommended: dict, per_item: list[dict],
                 pairs: list[dict], confusion: list[tuple[str, str, int]],
                 reference: dict, min_precision: float) -> None:
    lines: list[str] = []
    add = lines.append
    add(f"# {title}")
    add("")
    add(f"Source: `{source}`  ")
    add(f"Crops: **{n_crops}**  Items: **{len(item_ids)}**  ")
    add(f"Precision target: **{min_precision:.2f}**")
    add("")
    add("Scores are raw CLIP cosine similarities, not probabilities.")
    add("")
    add("A crop is accepted when its best item score is at least `min_score`; with 2+ items")
    add("it must also lead the runner-up by `margin`.")
    add("")

    # ---- recommendation
    balanced = best_f1_row(sweep)
    add("## Recommended settings")
    add("")
    add("```yaml")
    add("model:")
    add(f"  fast_item_min_score: {recommended['threshold']:.3f}")
    add("```")
    add("")
    if recommended["precision"] != recommended["precision"] or \
            recommended["precision"] < min_precision:
        add(f"> **Warning:** no threshold reached precision {min_precision:.2f} for a 1-item")
        add("> list. The value above is the best F1 instead. Improve the prompts for the")
        add("> worst items below before shipping this item set.")
        add("")
    if recommended["recall"] < 0.10:
        add("> **Warning:** the precision-first threshold accepts almost nothing")
        add(f"> (recall {recommended['recall']:.3f}). That is safe but not usable: the shopper")
        add("> would mostly hear \"I don't see it\". The balanced alternative below is more")
        add("> honest about the trade-off; the real fix is better (more specific) prompts.")
        add("")
    add(f"**1-item list, precision-first:** at `min_score = {recommended['threshold']:.3f}`, "
        f"precision **{fmt(recommended['precision'])}**, recall **{fmt(recommended['recall'])}** "
        f"({recommended['tp']} true, {recommended['fp']} false accepts, {n_crops} crops).")
    add("")
    add(f"**1-item list, balanced (best F1):** at `min_score = {balanced['threshold']:.3f}`, "
        f"precision **{fmt(balanced['precision'])}**, recall **{fmt(balanced['recall'])}** "
        f"({balanced['tp']} true, {balanced['fp']} false accepts).")
    add("")

    # ---- 2-item lists
    add("## 2-item lists (the Coke vs Sprite question)")
    add("")
    if pairs:
        accuracies = np.array([p["accuracy"] for p in pairs])
        leads = np.array([p["lead_p10"] for p in pairs])
        good = int(np.sum(accuracies >= 0.90))
        add(f"- median head-to-head accuracy: **{np.median(accuracies):.3f}**")
        add(f"- pairs at or above 0.90: **{good} / {len(pairs)}**")
        add(f"- median 10th-percentile lead: **{np.median(leads):.3f}**")
        add("")
        add("Suggested `fast_item_margin`: the 10th percentile of the lead, floored at 0 and")
        add("capped at 0.10. Rejecting a true item when two are genuinely close is the safe")
        add("failure: the shopper hears nothing rather than the wrong name.")
        add("")
        add(f"```yaml\nmodel:\n  fast_item_margin: "
            f"{max(0.0, min(float(np.median(leads)), 0.10)):.3f}\n```")
        add("")
        add("| pair | crops | accuracy | median lead | wrong winner |")
        add("|---|---|---|---|---|")
        for row in sorted(pairs, key=lambda r: r["accuracy"])[:15]:
            add(f"| {row['a']} vs {row['b']} | {row['n']} | {row['accuracy']:.3f} | "
                f"{fmt(row['lead_p50'])} | {row['wrong_winner']} |")
        add("")
        add("The table shows the 15 hardest pairs. A pair at 1.000 is reliably separable;")
        add("a pair near 0.500 cannot be told apart by these prompts at all.")
    else:
        add("Not enough crops per item to compare pairs.")
    add("")

    # ---- reference
    add("## Reference: all 26 items competing")
    add("")
    add(f"Top-1 accuracy when every item is a candidate: **{reference['accuracy']:.3f}** "
        f"({reference['n']} crops).")
    add("")
    add("This is *not* the shipped mode. It is here to show how much the short list helps,")
    add("and to compare against the previous all-classes pipeline.")
    add("")

    # ---- 1-item sweep
    add("## Threshold sweep (1-item list)")
    add("")
    add("| min_score | precision | recall | true | false |")
    add("|---|---|---|---|---|")
    for row in sweep[::2]:                      # every other row keeps the table readable
        add(f"| {row['threshold']:.2f} | {fmt(row['precision'])} | {fmt(row['recall'])} | "
            f"{row['tp']} | {row['fp']} |")
    add("")

    # ---- per item
    add("## Per item")
    add("")
    add("`threshold` is the lowest score that reaches the precision target for that item on")
    add("its own (a 1-item list). `other p90` is how high other items' photos score against")
    add("this prompt: when it is close to the item's own median, the prompt is too generic.")
    add("")
    add("| item | crops | own p10 | own median | own p90 | other p90 | threshold | recall |")
    add("|---|---|---|---|---|---|---|---|")
    for row in per_item:
        thr = row["threshold"]
        add(f"| {row['id']} | {row['n']} | {fmt(row['own_p10'])} | {fmt(row['own_p50'])} | "
            f"{fmt(row['own_p90'])} | {fmt(row['other_p90'])} | "
            f"{fmt(thr['threshold']) if thr else 'none'} | "
            f"{fmt(thr['recall']) if thr else '0.000'} |")
    add("")

    # ---- confusion
    add(f"## Confusion pairs (all items competing, above {balanced['threshold']:.2f})")
    add("")
    if confusion:
        add("| true item | named as | count |")
        add("|---|---|---|")
        for true_id, pred_id, count in confusion:
            add(f"| {true_id} | {pred_id} | {count} |")
    else:
        add("None above the threshold.")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="dataset root with images/ and labels/")
    source.add_argument("--images", type=Path, help="folder of item subfolders, no labels")
    parser.add_argument("--split", default="val", help="dataset split for --data (default: val)")
    parser.add_argument("--catalog", type=Path, default=ROOT / "config" / "catalog.yaml")
    parser.add_argument("--user-catalog", type=Path,
                        default=ROOT / "data" / "catalog_user.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "clip_item_calibration.md")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="stop after N crops (quick test)")
    parser.add_argument("--min-precision", type=float, default=0.90)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog, args.user_catalog)

    if args.data is not None:
        item_ids = list(catalog.item_ids)
        items = [catalog.item(i) for i in item_ids]
        samples = load_dataset(args.data, args.split, item_ids, args.limit)
        source_text = f"{args.data} ({args.split})"
        title = "CLIP item calibration: GrocerEye"
    else:
        samples = load_image_folders(args.images, catalog, args.limit)
        # Only the items that actually have photos are scored.
        present = sorted({s.item_id for s in samples})
        items = []
        for item_id in present:
            item = catalog.item(item_id)
            if item is None:
                item = CatalogItem(id=item_id, spoken=item_id.replace("_", " "))
            items.append(item)
        item_ids = [item.id for item in items]
        source_text = str(args.images)
        title = "CLIP item calibration: brand test"

    if not samples:
        raise SystemExit("no crops found; check the path and the labels")

    # Heavy imports happen only after the arguments are understood.
    from transformers import CLIPModel, CLIPProcessor

    device = pick_device(args.device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    print(f"CLIP {args.clip_model} on {device}; {len(samples)} crops; {len(item_ids)} items")

    _ids, text_bank = build_text_bank(catalog, items, processor, model, device)
    text_bank = text_bank.float().cpu().numpy()
    scores, true_idx = score_samples(samples, item_ids, text_bank, processor, model,
                                     device, args.batch_size)

    grid = np.arange(0.05, 0.501, 0.01)
    sweep = one_item_sweep(scores, true_idx, grid)
    recommended = choose_threshold(sweep, args.min_precision)
    per_item = per_item_rows(scores, true_idx, item_ids, grid, args.min_precision)
    pairs = pairwise_rows(scores, true_idx, item_ids)
    # Confusions are most informative at the balanced threshold; the precision-first one can
    # be so high that it accepts almost nothing and hides every mistake.
    confusion = confusion_pairs(scores, true_idx, item_ids, best_f1_row(sweep)["threshold"])
    reference = all_items_sweep(scores, true_idx)

    write_report(args.out, title=title, source=source_text, n_crops=len(samples),
                 item_ids=item_ids, sweep=sweep, recommended=recommended,
                 per_item=per_item, pairs=pairs, confusion=confusion,
                 reference=reference, min_precision=args.min_precision)

    print(f"\n1-item list, precision-first: min_score={recommended['threshold']:.3f} "
          f"(precision {fmt(recommended['precision'])}, recall {fmt(recommended['recall'])})")
    balanced = best_f1_row(sweep)
    print(f"1-item list, balanced:        min_score={balanced['threshold']:.3f} "
          f"(precision {fmt(balanced['precision'])}, recall {fmt(balanced['recall'])})")
    if pairs:
        accuracies = np.array([p["accuracy"] for p in pairs])
        leads = np.array([p["lead_p10"] for p in pairs])
        print(f"2-item lists: median accuracy {np.median(accuracies):.3f}; "
              f"suggested margin {max(0.0, min(float(np.median(leads)), 0.10)):.3f}")
    print(f"all {len(item_ids)} competing: top-1 accuracy {reference['accuracy']:.3f}")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
