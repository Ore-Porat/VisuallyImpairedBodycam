"""Fine-tune YOLO nano on GrocerEye data (Stage A, TR-3) or Freiburg + our photos (Stage B, TR-13).

WHAT THIS FILE IS FOR
---------------------
This is the script that actually teaches the object detector. The detector is the neural network
the laptop server runs on every camera frame to find grocery items (see README.md and
training/README.md). Training means: show the network thousands of labeled photos over and over,
and let it adjust its internal numbers ("weights") until its guesses match the labels. The
heavy lifting is done by the Ultralytics library; this script mostly prepares the inputs,
chooses sensible settings, and files away the result (``best.pt``) where the server can find it.

The requirement numbers in brackets (TR-3, TR-13, TR-15) refer to the product spec, PRD.pdf
("TR" = training requirement).

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

WORDS YOU WILL SEE
------------------
- **YOLO**: a family of fast object-detection networks. "YOLO11n" is the smallest ("nano")
  version of the current generation, chosen so it runs fast on a laptop.
- **Ultralytics**: the Python library that implements YOLO training and inference.
- **Weights / .pt file**: the learned numbers inside the network, saved to a file. ``yolo11n.pt``
  is a general-purpose starting point trained on the COCO dataset (80 everyday objects).
- **Fine-tuning**: starting from an already trained network and training it further on our own
  data, instead of starting from scratch. Much faster and needs far fewer photos.
- **Dataset / data.yaml**: a folder of images and label files, described by a small YAML file
  that says where the train/val/test images are and what the classes are called.
- **Label file**: a ``.txt`` next to each image listing its boxes: ``class_id cx cy w h``.
- **Class id**: the number for a category. The ORDER of names in ``data.yaml`` defines the ids,
  which is why two datasets with different orders must be "remapped" before being combined.
- **Epoch**: one full pass through all the training images. Training runs many epochs.
- **Batch**: how many images the GPU processes at once.
- **Augmentation**: randomly altering training images (brightness, zoom, blur, ...) so the model
  learns to cope with conditions that are not in the photos, e.g. a dim store or a shaky camera.
- **Early stopping / patience**: stop training when the validation score has not improved for
  N epochs, to save time and avoid over-fitting.
- **imgsz**: the square size images are resized to before entering the network.
- **Hard link**: a second directory entry for the same file, so a "copy" takes no extra space.
- **argparse**: the Python module that reads ``--options`` from the command line.
- **Path**: a Python object representing a file/folder path (from ``pathlib``).
"""

# Lets us write modern type hints (like `str | Path`) on older Python versions. Type hints are
# notes about what kind of value a function expects/returns; Python does not enforce them.
from __future__ import annotations

import argparse   # command-line options
import shutil     # copying files
import sys        # exit codes and the import path
from pathlib import Path

# The repository root (this file is training/train.py, so go up two folders).
REPO_ROOT = Path(__file__).resolve().parent.parent
# Make sibling scripts in training/ importable (e.g. `from get_grocereye_data import ...`),
# regardless of the folder the command is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# `noqa: E402` tells the style checker it is OK that this import is not at the very top (it has
# to come after the sys.path line above).
from get_grocereye_data import normalize_class_name, write_text  # noqa: E402

# The three piles of a dataset and the file extensions we treat as images.
SPLIT_KEYS = ("train", "val", "test")
IMAGE_EXTS = (".bmp", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")


def repo_path(p: str | Path) -> Path:
    """Resolve a path relative to the repo root (absolute paths are kept).

    ``expanduser`` turns a leading ``~`` into the user's home folder.
    """
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------------------------
# Data yaml helpers (shared with evaluate.py / pick_threshold.py)
#
# These read and merge the small YAML files that describe datasets.
# --------------------------------------------------------------------------------------------

def names_list(names) -> list[str]:
    """Ultralytics ``names`` (list or {id: name}) -> list ordered by class id.

    A data.yaml may write the class names as a list or as a dictionary ``{0: pasta, 1: rice}``.
    This returns a plain list either way, with position = class id.
    """
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def load_data_yaml(path: str | Path) -> dict:
    """Load a data yaml and resolve split entries to absolute paths (lists of str).

    Follows Ultralytics rules: ``path`` (if given) is the dataset root, relative to the cwd when
    it exists there, otherwise relative to the yaml's folder; without ``path`` the yaml's folder
    is the root.

    Returns a dictionary with ``names`` (list), ``root`` (absolute folder), ``yaml_file`` and,
    for each of train/val/test, a list of absolute image folders or list files.
    """
    import yaml
    path = repo_path(path)
    # `or {}` guards against a completely empty file (safe_load returns None for it).
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = path.parent
    if data.get("path"):
        p = Path(data["path"])
        # Conditional expression: a if cond else b. Absolute -> keep; else try the current
        # working directory first, then the yaml's own folder.
        root = p if p.is_absolute() else (Path.cwd() / p if (Path.cwd() / p).exists() else path.parent / p)
    out = dict(data)   # shallow copy so we do not modify what yaml gave us
    out["yaml_file"] = str(path)
    out["root"] = str(root.resolve())
    # If there are no names, invent "class_0", "class_1", ... based on the class count `nc`.
    out["names"] = names_list(data.get("names") or [f"class_{i}" for i in range(int(data.get("nc", 0)))])
    for k in SPLIT_KEYS:
        v = data.get(k)
        if not v:
            out[k] = []
            continue
        # A split can be one path or a list of paths; always work with a list.
        items = v if isinstance(v, list) else [v]
        out[k] = [str((root / x).resolve()) if not Path(x).is_absolute() else str(x) for x in items]
    return out


def list_images(entry: str | Path) -> list[Path]:
    """Images referenced by one split entry (a directory, a .txt list of paths, or one image).

    Ultralytics allows a split to be any of those three things; this flattens all of them to a
    sorted list of image paths.
    """
    entry = Path(entry)
    if entry.is_dir():
        # rglob("*") walks the folder recursively; keep only files with an image extension.
        return sorted(p for p in entry.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if entry.suffix == ".txt" and entry.is_file():
        # A text file with one image path per line, relative to the text file's folder.
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
    """Ultralytics convention: replace the last ``images`` dir with ``labels`` and the ext with .txt.

    ``data/x/images/train/a.png`` -> ``data/x/labels/train/a.txt``.
    """
    parts = list(image.parts)
    # Search the path components backwards (skipping the file name) for a folder called "images".
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def class_remap(src_names: list[str], dst_names: list[str]) -> dict[int, int]:
    """Map source class ids to destination ids by normalized name. Raises on unknown classes.

    Example: source names ``[rice, pasta]`` and destination ``[pasta, rice, tea]`` give
    ``{0: 1, 1: 0}``. Used when combining datasets whose class order differs.
    """
    # Dictionary comprehension: normalized name -> position in the destination list.
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
    """Rewrite the class id column of a YOLO label file.

    Each line is ``class_id cx cy w h``; only the first number changes.
    """
    out = []
    for line in text.splitlines():
        parts = line.split()
        if parts:
            # int(float(...)) tolerates ids written as "3.0".
            parts[0] = str(remap[int(float(parts[0]))])
            out.append(" ".join(parts))
    # Re-join with newlines; add a trailing newline only if there is any content.
    return "\n".join(out) + ("\n" if out else "")


def _link_or_copy(src: Path, dst: Path) -> None:
    """Make ``dst`` a hard link to ``src`` (no extra disk space); fall back to a real copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()   # delete the old file first
    try:
        dst.hardlink_to(src)
    except OSError:
        # Hard links fail across drives or on some file systems; copy2 also keeps timestamps.
        shutil.copy2(src, dst)


def remap_dataset(src: dict, dst_names: list[str], out_dir: Path) -> dict:
    """Copy (hard-link) a dataset's images and rewrite its labels to ``dst_names`` ids.

    ``src`` is a dictionary from ``load_data_yaml``. Returns a similar dictionary pointing at
    the new copy under ``out_dir``.
    """
    remap = class_remap(src["names"], dst_names)
    out = {"names": dst_names}
    for k in SPLIT_KEYS:
        # Nested list comprehension: every image of every entry of this split.
        imgs = [p for e in src[k] for p in list_images(e)]
        if not imgs:
            out[k] = []
            continue
        for img in imgs:
            new_img = out_dir / "images" / k / img.name
            _link_or_copy(img, new_img)
            lbl = label_path_for(img)
            # A missing label file means "no objects in this image" (a background image).
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
        # If this source's class list is not identical (after normalizing) to the first one,
        # rewrite its labels so the ids line up.
        if [normalize_class_name(n) for n in src["names"]] != base_norm:
            dest = repo_path(remap_root) / Path(src["root"]).name
            print(f"Remapping class ids of {src['yaml_file']} -> {dest}")
            # `{**a, **b}` builds a new dictionary with b's keys overriding a's.
            src = {**src, **remap_dataset(src, base_names, dest)}
        for k in SPLIT_KEYS:
            # Append entries not already present (avoid duplicates).
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
    """Albumentations list for the ``augmentations`` train arg, or None if not installed.

    Each entry is a random image transform applied with probability ``p`` to each training
    image, to make the model robust to blur and JPEG artefacts from the device's camera stream.
    """
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
    """Keyword arguments for ``YOLO.train`` (pure; no Ultralytics import).

    Translates our command-line options into the dictionary of settings Ultralytics expects.
    ``args`` is the parsed command line; ``data`` is the path of the data yaml to train on.
    """
    kw = dict(
        data=data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
        name=args.name, project=str(repo_path(args.project)), seed=args.seed, workers=args.workers,
        hsv_v=args.hsv_v, scale=args.scale, fraction=args.fraction, exist_ok=args.exist_ok,
        plots=True, cache=args.cache or False,
    )
    if args.device is not None:
        kw["device"] = args.device
    # --extra KEY=VALUE lets the user pass any other Ultralytics setting without us listing it.
    for item in args.extra or []:
        # partition splits at the first "=": "mosaic=0.5" -> ("mosaic", "=", "0.5").
        k, _, v = item.partition("=")
        import yaml
        kw[k] = yaml.safe_load(v)  # "0.1" -> 0.1, "true" -> True
    return kw


def publish_weights(save_dir: Path, name: str, models_dir: Path) -> Path:
    """Copy best.pt (+ args.yaml, results.csv) from a run dir to models/<name>/ (TR-15).

    Ultralytics writes its results under ``runs/train/<name>/``; the server looks in
    ``models/<name>/``. Returns the path of the published ``best.pt``.
    """
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"{best} not found; did training finish?")
    dest = models_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest / "best.pt")
    # Also keep the training settings and the per-epoch results table / plot for the record.
    for extra in ("args.yaml", "results.csv", "results.png"):
        if (save_dir / extra).exists():
            shutil.copy2(save_dir / extra, dest / extra)
    return dest / "best.pt"


def _rel(p: Path) -> str:
    """Show a path relative to the repo root when possible (nicer in printed messages)."""
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: parse options, prepare the data, train, publish the weights."""
    # --- 1. Command-line options -----------------------------------------------------------
    # The module docstring doubles as the --help text.
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    # action="append" means --data can be repeated; the values collect into a list.
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
    # --batch is read as a float so "0.5" (GPU memory fraction) works, but Ultralytics wants a
    # plain integer for whole numbers and -1, so convert those back.
    if args.batch.is_integer() or args.batch < 0:
        args.batch = int(args.batch)

    # --- 2. Work out which dataset to train on -------------------------------------------
    data_files = args.data or ["data/grocereye/data.yaml"]
    if len(data_files) > 1:
        # Several --data: merge them into one yaml (Stage B).
        out = args.combined_out or f"data/combined/{args.name}.yaml"
        data = str(combine_data_yamls(data_files, out))
        print(f"Combined {len(data_files)} datasets -> {data}")
    else:
        data = str(repo_path(data_files[0]))
    if not Path(data).exists():
        print(f"{data} not found. Run: python training/get_grocereye_data.py")
        return 1

    # --- 3. Assemble the training settings --------------------------------------------------
    kw = build_train_kwargs(args, data)
    augs = blur_augmentations(args.blur_p, args.jpeg_p)
    if augs is None:
        print("NOTE: albumentations not installed -> no blur augmentation "
              "(pip install albumentations to enable it).")
    else:
        kw["augmentations"] = augs
    model_path = args.model
    # A local .pt file is given relative to the repo root; a bare name like "yolo11n.pt" is
    # left alone so Ultralytics downloads it.
    if Path(model_path).suffix == ".pt" and repo_path(model_path).exists():
        model_path = str(repo_path(model_path))
    print(f"Model: {model_path}\nTrain args: " + ", ".join(
        f"{k}={v}" for k, v in kw.items() if k != "augmentations"))
    if args.dry_run:
        return 0

    # --- 4. Train and publish ---------------------------------------------------------------
    from ultralytics import YOLO  # lazy: heavy import
    model = YOLO(model_path)
    # `**kw` unpacks the dictionary into keyword arguments: train(data=..., epochs=..., ...).
    # This call can take from minutes to hours depending on the GPU.
    model.train(**kw)
    save_dir = Path(model.trainer.save_dir)
    best = publish_weights(save_dir, args.name, repo_path(args.models_dir))
    print(f"\nRun dir: {save_dir}\nPublished: {best}\n")
    # Tell the user exactly what to put in the server's config to use the new model.
    print("Use it in config/server.yaml:")
    print(f"  model.weights: {_rel(best)}")
    print("  model.classes_file: config/classes.yaml")
    print(f"  model.imgsz: {args.imgsz}   # match the training size")
    print(f"\nNext: python training/evaluate.py --weights {_rel(best)}"
          + ("" if len(data_files) == 1 else f" --data {_rel(Path(data))}"))
    return 0


# Standard Python idiom: run main() only when this file is executed directly, not when
# evaluate.py / pick_threshold.py import it for its helper functions.
if __name__ == "__main__":
    sys.exit(main())
