#!/usr/bin/env python3
"""Export the fast pipeline models to TensorRT FP16.

Exports:
  - YOLO-World detector with the generic product classes baked in
  - EgoHands hand detector

The engines are written to models/tensorrt/ and are specific to this GPU and
TensorRT version.

Example:
  python scripts/export_tensorrt.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DETECTOR_CLASSES = "product,package,can,bottle,box,bag,jar,container"


def export_model(model_path: Path, target: Path, imgsz: int, half: bool, workspace: float,
                 device: str, classes: list[str] | None = None):
    model = YOLO(str(model_path))
    if classes:
        model.set_classes(classes)
    print(f"Exporting {model_path} -> {target} (imgsz={imgsz}, half={half})", flush=True)
    engine = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        device=device,
        workspace=workspace,
        batch=1,
    )
    engine_path = Path(str(engine))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(engine_path, target)
    print(f"Wrote {target}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-model", default="yolov8l-worldv2.pt")
    parser.add_argument("--hand-model", default="models/hand_detector/best.pt")
    parser.add_argument("--detector-classes", default=DEFAULT_DETECTOR_CLASSES)
    parser.add_argument("--imgsz-world", type=int, default=640)
    parser.add_argument("--imgsz-hand", type=int, default=960)
    parser.add_argument("--workspace", type=float, default=4.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-dir", default="models/tensorrt")
    parser.add_argument("--world-out", default="shelf_world.engine")
    parser.add_argument("--hand-out", default="hand_egohands_960.engine")
    parser.add_argument("--no-half", action="store_true",
                        help="build FP32 engines when TensorRT 11 has no ModelOpt")
    parser.add_argument("--skip-world", action="store_true")
    parser.add_argument("--skip-hand", action="store_true")
    args = parser.parse_args()
    half = not args.no_half

    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out_dir
    world_path = root / args.world_model
    hand_path = root / args.hand_model
    classes = [name.strip() for name in args.detector_classes.split(",") if name.strip()]

    if not args.skip_world:
        if not world_path.exists():
            raise SystemExit(f"Missing YOLO-World model: {world_path}")
        export_model(world_path, out_dir / args.world_out, args.imgsz_world,
                     half, args.workspace, args.device, classes)

    if not args.skip_hand:
        if not hand_path.exists():
            raise SystemExit(f"Missing hand model: {hand_path}")
        export_model(hand_path, out_dir / args.hand_out, args.imgsz_hand,
                     half, args.workspace, args.device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
