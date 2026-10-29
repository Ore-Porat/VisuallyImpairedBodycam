"""Convert the EgoHands dataset into YOLO data with one class: ``hand``.

WHAT THIS FILE IS FOR
---------------------
EgoHands (Bambach et al., ICCV 2015; vision.soic.indiana.edu/projects/egohands) is 48 Google
Glass videos of two people playing cards, chess, Jenga or puzzles, filmed from the wearer's eyes:
the same first-person view as our chest camera. 4,800 frames carry pixel-accurate outlines of
every visible hand (15,053 hands). We only need boxes, so each outline becomes the box around it.

Input: the ``egohands_data.zip`` from the project page (or its Roboflow/Kaggle re-hosts), unpacked
into ``data/egohands`` so that ``data/egohands/_LABELLED_SAMPLES/<VIDEO>/frame_XXXX.jpg`` and
``.../polygons.mat`` exist (``--from-dir`` for another folder).

Output (relative to the repo root)::

    data/egohands_yolo/images/{train,val,test}/*.jpg
    data/egohands_yolo/labels/{train,val,test}/*.txt   # "0 cx cy w h"
    data/egohands_yolo/data.yaml                       # names: [hand]

The split is BY VIDEO (seeded), so frames of one video never land in both train and test.
train.py remaps the single ``hand`` id onto the combined class list by name.

Run::

    python training/get_egohands_data.py                 # data/egohands -> data/egohands_yolo
    python training/get_egohands_data.py --max-side 960  # smaller copies (frames are 1280x720)

License: EgoHands is free for research use; cite the paper. Keep the converted copy in data/.
"""

from __future__ import annotations

import argparse
import collections
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_grocereye_data import SPLITS, write_text  # noqa: E402
from get_products6k_data import REPO_ROOT, repo, yolo_line  # noqa: E402

HAND_FIELDS = ("myleft", "myright", "yourleft", "yourright")


def polygon_box(poly: np.ndarray, width: int, height: int,
                min_side: float = 0.01) -> tuple[float, float, float, float] | None:
    """Normalized ``(x0, y0, x1, y1)`` around an N x 2 polygon in pixels, or None if empty/tiny."""
    pts = np.asarray(poly, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    box = (max(0.0, x0 / width), max(0.0, y0 / height), min(1.0, x1 / width), min(1.0, y1 / height))
    if box[2] - box[0] < min_side or box[3] - box[1] < min_side:
        return None
    return box


def split_videos(videos: list[str], val_frac: float, test_frac: float, seed: int) -> dict[str, str]:
    """Assign whole videos to train / val / test (seeded shuffle)."""
    order = sorted(videos)
    random.Random(seed).shuffle(order)
    n = len(order)
    n_test, n_val = round(n * test_frac), round(n * val_frac)
    out = {}
    for i, v in enumerate(order):
        out[v] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def read_polygons(mat_path: Path) -> list[list[np.ndarray]]:
    """Per frame, the list of hand polygons (each N x 2) from a video's polygons.mat."""
    from scipy.io import loadmat
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    frames = data["polygons"]
    frames = np.atleast_1d(frames)
    out = []
    for fr in frames:
        hands = []
        for name in HAND_FIELDS:
            poly = getattr(fr, name, None)
            if poly is None:
                continue
            poly = np.asarray(poly)
            if poly.size >= 6:
                hands.append(poly.reshape(-1, 2))
        out.append(hands)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-dir", type=Path, default=Path("data/egohands"),
                    help="unpacked EgoHands folder (contains _LABELLED_SAMPLES)")
    ap.add_argument("--out", type=Path, default=Path("data/egohands_yolo"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1280, help="resize frames larger than this")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame of a video (1 = all)")
    args = ap.parse_args(argv)

    src, out = repo(args.from_dir), repo(args.out)
    samples = next((p for p in [src / "_LABELLED_SAMPLES", src] if (p).is_dir()
                    and any(q.is_dir() for q in p.iterdir())), None)
    video_dirs = sorted(p for p in (samples or src).iterdir() if p.is_dir() and (p / "polygons.mat").exists()) \
        if samples else []
    if not video_dirs:
        print(f"No EgoHands videos found under {src} (expected <video>/polygons.mat folders). "
              "Unpack egohands_data.zip into data/egohands first.")
        return 1
    print(f"Found {len(video_dirs)} labelled videos under {samples}")
    split = split_videos([v.name for v in video_dirs], args.val_frac, args.test_frac, args.seed)

    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        for s in SPLITS:
            (out / sub / s).mkdir(parents=True, exist_ok=True)

    from PIL import Image
    counts = collections.Counter()
    for vd in video_dirs:
        frames = sorted(vd.glob("frame_*.jpg"))
        polys = read_polygons(vd / "polygons.mat")
        if len(polys) != len(frames):
            print(f"  {vd.name}: {len(frames)} frames but {len(polys)} label entries; using the shorter")
        s = split[vd.name]
        for i, (fp, hands) in enumerate(zip(frames, polys)):
            if i % args.every:
                continue
            with Image.open(fp) as im:
                im = im.convert("RGB")
                w, h = im.size
                boxes = [b for b in (polygon_box(p, w, h) for p in hands) if b is not None]
                stem = f"egohands_{vd.name}_{fp.stem}"
                dst = out / "images" / s / (stem + ".jpg")
                if max(w, h) > args.max_side:
                    im.thumbnail((args.max_side, args.max_side))
                    im.save(dst, "JPEG", quality=90)
                else:
                    shutil.copy2(fp, dst)
            write_text(out / "labels" / s / (stem + ".txt"),
                       "".join(yolo_line(0, b) + "\n" for b in boxes))
            counts[s] += 1
            counts["hands"] += len(boxes)
            counts["frames without a hand"] += not boxes
    write_text(out / "data.yaml",
               "# Generated by training/get_egohands_data.py from EgoHands (Bambach et al. 2015).\n"
               "train: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  - hand\n")
    print("Videos per split: " + ", ".join(f"{s}={sum(1 for v in split.values() if v == s)}" for s in SPLITS))
    print("Frames per split: " + ", ".join(f"{s}={counts[s]}" for s in SPLITS)
          + f"; hand boxes {counts['hands']}; frames without a hand {counts['frames without a hand']}")
    print(f"Wrote {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
