"""Fine-tune YOLO nano on GrocerEye data (Stage A, TR-3) or Freiburg + our photos (Stage B, TR-13).

Stage A: start from COCO-pretrained ``yolo11n.pt`` and train on ``data/grocereye/data.yaml``::

    python training/train.py --name grocereye_a

Stage B: start from the Stage A weights and train on Freiburg + our labeled photos together, so
the model does not forget the categories we did not photograph::

    python training/train.py --name grocereye_b --model models/grocereye_a/best.pt \\
        --data data/grocereye/data.yaml --data data/ours/data.yaml --epochs 60

Several ``--data`` files are merged into ``data/combined/<name>.yaml`` (see ``combine_data_yamls``).
Every source must use the GrocerEye class names (``tomato_sauce`` etc.; spaces/case are
normalized). If a source's class *order* differs, e.g. a Roboflow export that only contains the
demo categories, its labels are remapped into ``data/remapped/<source>/`` automatically.

Augmentation = Ultralytics defaults plus, to close GrocerEye's "natural setting" gap:

- ``--hsv-v`` brightness jitter (default 0.6, Ultralytics 0.4): dim/bright store lighting.
- ``--scale`` (default 0.7, Ultralytics 0.5): random zoom 0.3x..1.7x, so products also appear
  small in frame like at shelf distance, not only as 224px close-ups.
- Blur: Ultralytics applies Albumentations transforms only if ``albumentations`` is installed
  (``pip install albumentations``). Its defaults use Blur at p=0.01; when it is installed this
  script passes stronger Blur/MotionBlur (``--blur-p``) and JPEG compression (``--jpeg-p``, the
  device streams JPEG) via the ``augmentations`` train argument. Without it there is no blur.

At the end ``best.pt`` is copied to ``models/<name>/best.pt`` (TR-15) together with the run's
``args.yaml`` / ``results.csv``, and the ``config/server.yaml`` lines to use it are printed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_grocereye_data import normalize_class_name, write_text  # noqa: E402

SPLIT_KEYS = ("train", "val", "test")
IMAGE_EXTS = (".bmp", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")


def repo_path(p: str | Path) -> Path:
    """Resolve a path relative to the repo root (absolute paths are kept)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------------------------
# Data yaml helpers (shared with evaluate.py / pick_threshold.py)
# --------------------------------------------------------------------------------------------

def names_list(names) -> list[str]:
    """Ultralytics ``names`` (list or {id: name}) -> list ordered by class id."""
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def load_data_yaml(path: str | Path) -> dict:
    """Load a data yaml and resolve split entries to absolute paths (lists of str).

    Follows Ultralytics rules: ``path`` (if given) is the dataset root, relative to the cwd when
    it exists there, otherwise relative to the yaml's folder; without ``path`` the yaml's folder
    is the root.
    """
    import yaml
    path = repo_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = path.parent
    if data.get("path"):
        p = Path(data["path"])
        root = p if p.is_absolute() else (Path.cwd() / p if (Path.cwd() / p).exists() else path.parent / p)
    out = dict(data)
    out["yaml_file"] = str(path)
    out["root"] = str(root.resolve())
    out["names"] = names_list(data.get("names") or [f"class_{i}" for i in range(int(data.get("nc", 0)))])
    for k in SPLIT_KEYS:
        v = data.get(k)
        if not v:
            out[k] = []
            continue
        items = v if isinstance(v, list) else [v]
        out[k] = [str((root / x).resolve()) if not Path(x).is_absolute() else str(x) for x in items]
    return out


def list_images(entry: str | Path) -> list[Path]:
    """Images referenced by one split entry (a directory, a .txt list of paths, or one image)."""
    entry = Path(entry)
    if entry.is_dir():
        return sorted(p for p in entry.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if entry.suffix == ".txt" and entry.is_file():
        base = entry.parent
        out = []
        for line in entry.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                p = Path(line)
                out.append(p if p.is_absolute() else (base / p).resolve())
        return out
    if entry.suffix.lower() in IMAGE_EXTS and entry.is_file():
        return [entry]
    return []


def label_path_for(image: Path) -> Path:
    """Ultralytics convention: replace the last ``images`` dir with ``labels`` and the ext with .txt."""
    parts = list(image.parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def class_remap(src_names: list[str], dst_names: list[str]) -> dict[int, int]:
    """Map source class ids to destination ids by normalized name. Raises on unknown classes."""
    dst_index = {normalize_class_name(n): i for i, n in enumerate(dst_names)}
    remap, unknown = {}, []
    for i, n in enumerate(src_names):
        key = normalize_class_name(n)
        if key in dst_index:
            remap[i] = dst_index[key]
        else:
            unknown.append(n)
    if unknown:
        raise ValueError(f"classes {unknown} are not in the base class list {dst_names}; "
                         "rename them in your labeling tool (e.g. 'tomato sauce' -> tomato_sauce)")
    return remap


def remap_label_text(text: str, remap: dict[int, int]) -> str:
    """Rewrite the class id column of a YOLO label file."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if parts:
            parts[0] = str(remap[int(float(parts[0]))])
            out.append(" ".join(parts))
    return "\n".join(out) + ("\n" if out else "")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def remap_dataset(src: dict, dst_names: list[str], out_dir: Path) -> dict:
    """Copy (hard-link) a dataset's images and rewrite its labels to ``dst_names`` ids."""
    remap = class_remap(src["names"], dst_names)
    out = {"names": dst_names}
    for k in SPLIT_KEYS:
        imgs = [p for e in src[k] for p in list_images(e)]
        if not imgs:
            out[k] = []
            continue
        for img in imgs:
            new_img = out_dir / "images" / k / img.name
            _link_or_copy(img, new_img)
            lbl = label_path_for(img)
            text = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
            write_text(label_path_for(new_img), remap_label_text(text, remap))
        out[k] = [str((out_dir / "images" / k).resolve())]
    return out


def combine_data_yamls(paths: list[str | Path], out_path: str | Path,
                       remap_root: str | Path = "data/remapped") -> Path:
    """Merge several data yamls into one (TR-13). The first file's ``names`` define class ids.

    Each split becomes a list of absolute image dirs/lists. Sources whose names differ in order
    or are a subset are remapped (labels rewritten) under ``remap_root/<source dir name>/``.
    Returns the written yaml path.
    """
    import yaml
    sources = [load_data_yaml(p) for p in paths]
    base_names = sources[0]["names"]
    base_norm = [normalize_class_name(n) for n in base_names]
    merged = {k: [] for k in SPLIT_KEYS}
    for src in sources:
        if [normalize_class_name(n) for n in src["names"]] != base_norm:
            dest = repo_path(remap_root) / Path(src["root"]).name
            print(f"Remapping class ids of {src['yaml_file']} -> {dest}")
            src = {**src, **remap_dataset(src, base_names, dest)}
        for k in SPLIT_KEYS:
            merged[k] += [e for e in src[k] if e not in merged[k]]
    if not merged["train"] or not merged["val"]:
        raise ValueError("combined dataset needs at least one train and one val entry")
    out_path = repo_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {k: merged[k] for k in SPLIT_KEYS if merged[k]}
    doc["nc"] = len(base_names)
    doc["names"] = list(base_names)
    header = "# Generated by training/train.py from: " + ", ".join(str(p) for p in paths) + "\n"
    write_text(out_path, header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return out_path


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------

def blur_augmentations(blur_p: float, jpeg_p: float):
    """Albumentations list for the ``augmentations`` train arg, or None if not installed."""
    try:
        import albumentations as A
    except ImportError:
        return None
    return [
        A.Blur(blur_limit=(3, 7), p=blur_p),
        A.MotionBlur(blur_limit=(3, 9), p=blur_p),   # walking with a chest camera
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.ImageCompression(quality_range=(50, 95), p=jpeg_p),
    ]


def build_train_kwargs(args: argparse.Namespace, data: str) -> dict:
    """Keyword arguments for ``YOLO.train`` (pure; no Ultralytics import)."""
    kw = dict(
        data=data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
        name=args.name, project=str(repo_path(args.project)), seed=args.seed, workers=args.workers,
        hsv_v=args.hsv_v, scale=args.scale, fraction=args.fraction, exist_ok=args.exist_ok,
        plots=True, cache=args.cache or False,
    )
    if args.device is not None:
        kw["device"] = args.device
    for item in args.extra or []:
        k, _, v = item.partition("=")
        import yaml
        kw[k] = yaml.safe_load(v)  # "0.1" -> 0.1, "true" -> True
    return kw


def publish_weights(save_dir: Path, name: str, models_dir: Path) -> Path:
    """Copy best.pt (+ args.yaml, results.csv) from a run dir to models/<name>/ (TR-15)."""
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"{best} not found; did training finish?")
    dest = models_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest / "best.pt")
    for extra in ("args.yaml", "results.csv", "results.png"):
        if (save_dir / extra).exists():
            shutil.copy2(save_dir / extra, dest / extra)
    return dest / "best.pt"


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--data", action="append", metavar="YAML",
                    help="data yaml; repeat to combine sources (default: data/grocereye/data.yaml)")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="starting weights: COCO yolo11n.pt (Stage A) or a Stage A best.pt (Stage B)")
    ap.add_argument("--name", default="grocereye_a", help="run name; weights go to models/<name>/")
    ap.add_argument("--imgsz", type=int, default=320, help="train image size (images are 224x224)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20, help="early stopping patience (epochs)")
    ap.add_argument("--batch", type=float, default=32,
                    help="batch size; -1 = auto (60%% GPU memory), 0.0-1.0 = GPU memory fraction")
    ap.add_argument("--device", default=None, help="'0', 'cpu', 'mps', ... (default: Ultralytics picks)")
    ap.add_argument("--workers", type=int, default=4, help="dataloader workers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hsv-v", type=float, default=0.6, help="brightness augmentation (Ultralytics 0.4)")
    ap.add_argument("--scale", type=float, default=0.7, help="scale augmentation +/- gain (Ultralytics 0.5)")
    ap.add_argument("--blur-p", type=float, default=0.1,
                    help="Blur and MotionBlur probability (needs albumentations)")
    ap.add_argument("--jpeg-p", type=float, default=0.2,
                    help="JPEG compression probability (needs albumentations)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of the train set to use (e.g. 0.05 for a smoke test)")
    ap.add_argument("--cache", choices=("ram", "disk"), default=None, help="cache images")
    ap.add_argument("--project", default="runs/train", help="Ultralytics run dir (default: %(default)s)")
    ap.add_argument("--models-dir", default="models", help="where best.pt is published (default: %(default)s)")
    ap.add_argument("--combined-out", default=None,
                    help="combined yaml path when several --data are given (default: data/combined/<name>.yaml)")
    ap.add_argument("--exist-ok", action="store_true", help="reuse runs/train/<name> instead of <name>2, ...")
    ap.add_argument("--extra", action="append", metavar="KEY=VALUE",
                    help="any other Ultralytics train argument, e.g. --extra mosaic=0.5 (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="print the train arguments and exit")
    args = ap.parse_args(argv)
    if args.batch.is_integer() or args.batch < 0:
        args.batch = int(args.batch)

    data_files = args.data or ["data/grocereye/data.yaml"]
    if len(data_files) > 1:
        out = args.combined_out or f"data/combined/{args.name}.yaml"
        data = str(combine_data_yamls(data_files, out))
        print(f"Combined {len(data_files)} datasets -> {data}")
    else:
        data = str(repo_path(data_files[0]))
    if not Path(data).exists():
        print(f"{data} not found. Run: python training/get_grocereye_data.py")
        return 1

    kw = build_train_kwargs(args, data)
    augs = blur_augmentations(args.blur_p, args.jpeg_p)
    if augs is None:
        print("NOTE: albumentations not installed -> no blur augmentation "
              "(pip install albumentations to enable it).")
    else:
        kw["augmentations"] = augs
    model_path = args.model
    if Path(model_path).suffix == ".pt" and repo_path(model_path).exists():
        model_path = str(repo_path(model_path))
    print(f"Model: {model_path}\nTrain args: " + ", ".join(
        f"{k}={v}" for k, v in kw.items() if k != "augmentations"))
    if args.dry_run:
        return 0

    from ultralytics import YOLO  # lazy: heavy import
    model = YOLO(model_path)
    model.train(**kw)
    save_dir = Path(model.trainer.save_dir)
    best = publish_weights(save_dir, args.name, repo_path(args.models_dir))
    print(f"\nRun dir: {save_dir}\nPublished: {best}\n")
    print("Use it in config/server.yaml:")
    print(f"  model.weights: {_rel(best)}")
    print("  model.classes_file: config/classes.yaml")
    print(f"  model.imgsz: {args.imgsz}   # match the training size")
    print(f"\nNext: python training/evaluate.py --weights {_rel(best)}"
          + ("" if len(data_files) == 1 else f" --data {_rel(Path(data))}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
