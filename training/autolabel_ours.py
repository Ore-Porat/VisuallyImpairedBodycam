"""Turn capture.py sessions of OUR products into a labeled YOLO dataset, boxes drawn automatically.

WHAT THIS FILE IS FOR
---------------------
Stage B (training/README.md) needs photos of the actual products we demo with (Oreos, Cheerios,
Coca-Cola, Lay's, Cheetos, ...). ``capture.py`` saves them one session per product and setup
(``data/raw/<session>/``), but a detector also needs a box around the product in every photo.
Drawing hundreds of boxes by hand is slow, so this script lets Grounding DINO draw them: each
session shows ONE product, you say which class it is, and DINO finds the biggest confident box
for "cookies." / "soda can." / ... in each photo. Check the review sheet, fix or skip the odd
miss in ``boxes.json``, rerun, train.

Usage (each --session is ``<folder>=<class>[:<split>]``; split defaults to train)::

    python training/autolabel_ours.py --device cuda \\
        --session oreos_desk=cookies --session oreos_kitchen=cookies:test \\
        --session cheerios_desk=cereal --session cheerios_shelf=cereal:test \\
        --session cola_desk=soda:val --session empty_desk=background

Use a DIFFERENT session for the test split than for train (another room, light or background),
never the same photos: near-duplicate frames of one session must not leak (TR-12). A
``background`` session gets empty labels (shelves or tables with none of our products).

Output (relative to the repo root)::

    data/ours/images/{train,val,test}/<session>_NNNN.jpg
    data/ours/labels/{train,val,test}/<session>_NNNN.txt
    data/ours/data.yaml       # the full class list from config/classes.yaml, so no remap is needed
    data/ours/boxes.json      # cache; set "skip": true or a manual "box" per photo and rerun
    data/ours/review/sheet_*.jpg

Then::

    python training/train.py --name ours_b --model models/<best so far>/best.pt \\
        --data data/products6k_hands_v2/data.yaml --data data/ours/data.yaml --epochs 60
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_grocereye_data import SPLITS, write_text  # noqa: E402
from get_products6k_data import (IMAGE_EXTS, DinoBoxer, load_class_list, load_image,  # noqa: E402
                                 make_review_sheets, pick_box, repo, save_image, yolo_line)


def parse_session(spec: str) -> tuple[str, str, str]:
    """``'oreos_desk=cookies:test'`` -> ``('oreos_desk', 'cookies', 'test')`` (split defaults to train)."""
    if "=" not in spec:
        raise SystemExit(f"--session needs <folder>=<class>[:<split>], got {spec!r}")
    folder, rest = spec.split("=", 1)
    cls, _, split = rest.partition(":")
    split = split or "train"
    if split not in SPLITS:
        raise SystemExit(f"{spec!r}: split must be one of {SPLITS}")
    return folder.strip(), cls.strip(), split


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", action="append", required=True, metavar="FOLDER=CLASS[:SPLIT]",
                    help="a capture session folder under --raw and the class it shows (repeatable)")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"), help="capture.py output root")
    ap.add_argument("--out", type=Path, default=Path("data/ours"))
    ap.add_argument("--classes-yaml", type=Path, default=Path("config/classes.yaml"))
    ap.add_argument("--prompt", action="append", default=[], metavar="CLASS=TEXT",
                    help="DINO prompt for a class, e.g. cookies='blue cookie box.' (default: spoken name)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--box-threshold", type=float, default=0.3)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--rebox", action="store_true", help="recompute cached boxes (manual ones are kept)")
    ap.add_argument("--no-review", action="store_true")
    args = ap.parse_args(argv)

    import yaml
    out = repo(args.out)
    class_ids = load_class_list(repo(args.classes_yaml))
    spoken = yaml.safe_load(repo(args.classes_yaml).read_text(encoding="utf-8"))["classes"]
    prompts = {}
    for spec in args.prompt:
        k, _, v = spec.partition("=")
        prompts[k.strip()] = v.strip()
    sessions = [parse_session(s) for s in args.session]
    for folder, cls, _ in sessions:
        if cls != "background" and cls not in class_ids:
            raise SystemExit(f"{folder}: class {cls!r} is not in {args.classes_yaml}")
        if not (repo(args.raw) / folder).is_dir():
            raise SystemExit(f"session folder not found: {repo(args.raw) / folder}")

    cache_path = out / "boxes.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    dino: DinoBoxer | None = None
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        for s in SPLITS:
            (out / sub / s).mkdir(parents=True, exist_ok=True)

    entries, counts = [], collections.Counter()
    for folder, cls, split in sessions:
        photos = sorted(p for p in (repo(args.raw) / folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for p in photos:
            key = f"{folder}/{p.name}"
            c = cache.get(key, {})
            if c.get("skip"):
                counts["skipped"] += 1
                continue
            img = load_image(p)
            box, method, score = None, "background_image", 1.0
            if cls != "background":
                if c.get("box") and (c.get("method") == "manual" or not args.rebox):
                    box, method, score = c["box"], c.get("method", "manual"), float(c.get("score", 1.0))
                else:
                    if dino is None:
                        dino = DinoBoxer(args.dino_model, args.device, args.box_threshold, 0.25)
                    w, h = img.size
                    prompt = prompts.get(cls, f"{str(spoken.get(cls, cls)).replace('_', ' ')}.")
                    best = pick_box(dino.detect(img, prompt), w, h)
                    if best is None:
                        counts["no box found"] += 1
                        cache[key] = {"method": "none"}
                        continue
                    x0, y0, x1, y1, s = best
                    box = [round(v, 5) for v in (x0 / w, y0 / h, x1 / w, y1 / h)]
                    method, score = "dino", round(s, 3)
                    cache[key] = {"box": box, "method": method, "score": score}
            dst = out / "images" / split / f"{folder}_{p.stem}.jpg"
            save_image(img, p, dst, args.max_side)
            line = yolo_line(class_ids.index(cls), box) if box is not None else ""
            write_text(out / "labels" / split / f"{folder}_{p.stem}.txt", line + ("\n" if line else ""))
            counts[method] += 1
            entries.append({"source": key, "class": None if cls == "background" else cls, "split": split,
                            "box": box, "method": method, "score": score, "dataset_image": str(dst)})
    write_text(cache_path, json.dumps(cache, indent=1) + "\n")
    write_text(out / "data.yaml",
               "# Generated by training/autolabel_ours.py from capture.py sessions (our own photos).\n"
               "train: images/train\nval: images/val\ntest: images/test\n"
               f"nc: {len(class_ids)}\nnames:\n" + "".join(f"  - {c}\n" for c in class_ids))
    per_split = collections.Counter(e["split"] for e in entries)
    n_sheets = 0 if args.no_review else make_review_sheets(entries, out / "review")
    print(f"Wrote {len(entries)} images: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("Per split: " + ", ".join(f"{s}={per_split[s]}" for s in SPLITS))
    print("Per class: " + ", ".join(f"{c} {n}" for c, n in sorted(
        collections.Counter(e["class"] or "background" for e in entries).items())))
    print(f"Wrote {out / 'data.yaml'}")
    if n_sheets:
        print(f"Review the boxes: {out / 'review'} ({n_sheets} sheets, least certain first); fix or "
              f"skip a photo in {cache_path.name} and rerun.")
    if not per_split["val"]:
        print("NOTE: no val split; train.py needs one. Give some session the :val suffix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
