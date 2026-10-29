"""Add a ``hand`` class to the Products-6K data: box the shopper's hand in the in-store photos.

WHAT THIS FILE IS FOR
---------------------
The long-term goal is to guide the shopper's hand to the item, so the detector must see the
hand as well as the groceries. Nothing we have is labelled for hands, but every Products-6K
*query* photo (``data/Query_Set``) shows a product held in a hand inside the store, which is
exactly the view the bodycam gets. This script:

1. Runs Grounding DINO with the prompt ``"hand."`` over every query photo and keeps the
   confident hand boxes (at most ``--max-hands`` per photo, after removing overlaps).
2. Copies an existing converted Products-6K set (``--src``, default ``data/products6k_v2``)
   to ``--out`` and **appends** the hand boxes to the labels of the query photos already in it,
   so one image carries both its product box and its hand box(es).
3. Adds the query photos that the product conversion left out (products outside our classes,
   or a bad product box) as hand-only images, in the same product-wise split.
4. Writes ``data.yaml`` with the 25 grocery names plus ``hand`` as the last class, matching
   ``config/classes.yaml`` and ``data/grocereye/data.yaml``.

Output (relative to the repo root)::

    data/products6k_hands/images/{train,val,test}/*.jpg
    data/products6k_hands/labels/{train,val,test}/*.txt
    data/products6k_hands/data.yaml            # 26 names, hand last
    data/products6k_hands/hand_boxes.json      # cache; set "skip": true to drop a photo's hands
    data/products6k_hands/review/sheet_*.jpg   # thumbnails, least certain first

Then train on Freiburg + this set, starting from the best grocery weights::

    python training/train.py --name grocereye_hand --model models/grocereye_p6k2/best.pt \\
        --data data/grocereye/data.yaml --data data/products6k_hands/data.yaml --epochs 60

Freiburg photos contain no hands, so they double as "no hand here" examples.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_grocereye_data import SPLITS, write_text  # noqa: E402
from get_products6k_data import (IMAGE_EXTS, REPO_ROOT, DinoBoxer, find_sources,  # noqa: E402
                                 load_image, load_tables, make_review_sheets, repo, safe_stem,
                                 yolo_line)


def iou(a: list[float], b: list[float]) -> float:
    """Intersection over union of two ``[x0, y0, x1, y1]`` boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_hands(cands: list[tuple[float, float, float, float, float]], width: int, height: int,
               threshold: float, max_hands: int, min_area: float = 0.005, max_area: float = 0.6
               ) -> list[tuple[list[float], float]]:
    """Confident, plausible-size hand boxes, overlaps removed, best first (normalized boxes)."""
    keep: list[tuple[list[float], float]] = []
    for x0, y0, x1, y1, s in sorted(cands, key=lambda c: -c[4]):
        box = [x0 / width, y0 / height, x1 / width, y1 / height]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if s < threshold or not (min_area <= area <= max_area):
            continue
        if any(iou(box, k[0]) > 0.5 for k in keep):
            continue
        keep.append(([round(v, 5) for v in box], round(s, 3)))
        if len(keep) >= max_hands:
            break
    return keep


def hashed_split(key: str, seed: int, val_frac: float, test_frac: float) -> str:
    """Deterministic train/val/test choice for a product id that no manifest knows."""
    h = int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16) / 16 ** 32
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=Path("data/products6k_v2"), help="converted Products-6K set to extend")
    ap.add_argument("--query", type=Path, default=Path("data/Query_Set"), help="extracted Query_Set folder")
    ap.add_argument("--out", type=Path, default=Path("data/products6k_hands"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--threshold", type=float, default=0.35, help="hand box score floor")
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args(argv)

    import yaml
    src, out, query = repo(args.src), repo(args.out), repo(args.query)
    doc = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = [str(n) for n in doc["names"]]
    if "hand" in names:
        raise SystemExit(f"{src} already has a hand class")
    names.append("hand")
    hand_id = names.index("hand")
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    by_rel = {e["source"]: e for e in manifest["images"]}
    sku_split = {e["sku"]: e["split"] for e in manifest["images"]}
    split_cfg = manifest.get("split", "")

    # --- copy the product set ----------------------------------------------------------------
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        for s in SPLITS:
            (out / sub / s).mkdir(parents=True, exist_ok=True)
            for p in (src / sub / s).iterdir():
                shutil.copy2(p, out / sub / s / p.name)
    print(f"Copied {len(manifest['images'])} product images from {src}")

    # --- hand boxes on every query photo ---------------------------------------------------
    tables = load_tables([query])
    sources = [s for s in find_sources([query], tables) if s.subset == "query"]
    cache_path = out / "hand_boxes.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    dino = None
    entries, counts = [], collections.Counter()
    for i, s in enumerate(sources, 1):
        c = cache.get(s.rel)
        if c is None:
            if dino is None:
                dino = DinoBoxer(args.dino_model, args.device, 0.25, 0.25)
            img = load_image(s.path)
            w, h = img.size
            hands = pick_hands(dino.detect(img, "hand."), w, h, args.threshold, args.max_hands)
            c = cache[s.rel] = {"hands": [{"box": b, "score": sc} for b, sc in hands]}
        if c.get("skip"):
            counts["skipped"] += 1
            continue
        hands = [(hd["box"], float(hd["score"])) for hd in c.get("hands", [])]
        entry = by_rel.get(s.rel)
        if entry is not None:
            split, dst = entry["split"], Path(entry["dataset_image"])
            dst = out / "images" / split / dst.name
            label = out / "labels" / split / (dst.stem + ".txt")
            if not hands:
                counts["product photo, no hand found"] += 1
                continue
            text = label.read_text(encoding="utf-8") if label.exists() else ""
            if text and not text.endswith("\n"):
                text += "\n"
            text += "".join(yolo_line(hand_id, b) + "\n" for b, _ in hands)
            write_text(label, text)
            counts["hand added to product photo"] += 1
        else:
            if not hands:
                counts["extra photo, no hand found"] += 1
                continue
            key = s.sku if s.sku is not None else s.rel
            split = sku_split.get(s.sku) or hashed_split(str(key), args.seed, 0.15, 0.15)
            stem = f"p6k_hand_{safe_stem(Path(s.rel).stem)}"
            dst = out / "images" / split / (stem + ".jpg")
            img = load_image(s.path)
            from get_products6k_data import save_image
            save_image(img, s.path, dst, args.max_side)
            write_text(out / "labels" / split / (stem + ".txt"),
                       "".join(yolo_line(hand_id, b) + "\n" for b, _ in hands))
            counts["hand-only photo added"] += 1
        counts["hand boxes"] += len(hands)
        best = min(hands, key=lambda x: x[1])
        entries.append({"source": s.rel, "class": "hand", "method": "dino", "score": best[1],
                        "box": best[0], "split": split, "dataset_image": str(dst)})
        if i % 100 == 0:
            print(f"  {i}/{len(sources)} query photos")
            write_text(cache_path, json.dumps(cache, indent=1) + "\n")
    write_text(cache_path, json.dumps(cache, indent=1) + "\n")

    names_block = "".join(f"  - {c}\n" for c in names)
    write_text(out / "data.yaml",
               "# Generated by training/get_hands_data.py: Products-6K products + the shopper's hand.\n"
               f"# Built from {src.name}; hand boxes by Grounding DINO. CC BY-NC-ND 4.0: do not share.\n"
               "train: images/train\nval: images/val\ntest: images/test\n"
               f"nc: {len(names)}\nnames:\n{names_block}")
    per_split = {s: sum(1 for e in entries if e["split"] == s) for s in SPLITS}
    write_text(out / "hands_manifest.json", json.dumps({
        "src": str(src), "split": split_cfg, "counts": dict(counts), "hand_images_per_split": per_split,
        "images": entries}, indent=1) + "\n")
    n_sheets = 0 if args.no_review else make_review_sheets(entries, out / "review")
    print(f"\nHands: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("Photos with a hand box per split: " + ", ".join(f"{s}={per_split[s]}" for s in SPLITS))
    print(f"Wrote {out / 'data.yaml'} ({len(names)} classes, hand = id {hand_id})")
    if n_sheets:
        print(f"Review: {out / 'review'} ({n_sheets} sheets). To drop a photo's hands, set "
              f"\"skip\": true for it in {cache_path.name} and rerun.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
