"""Evaluate a trained detector and write a markdown report (TR-4, NFR-5, NFR-6).

Two views of the same split:

1. Ultralytics ``val`` (conf 0.001, the standard protocol): mAP50, mAP50-95, per-class AP50 and
   per-class precision/recall at the confidence that maximizes mean F1. mAP50 is the number to
   compare with GrocerEye's 84.59% mAP (Darknet/AlexeyAB mAP@0.5 on its own 20% test split).
2. Operating point: predictions at ``--op-conf`` (default: ``guidance.conf_find`` from
   config/server.yaml) matched to ground truth at IoU >= 0.5, giving the per-class precision/recall
   and confusion matrix the device will actually see. NFR-6 (Stage B) is checked on these.

Targets: ``--stage a`` (NFR-5) mAP50 >= 0.85. ``--stage b`` (NFR-6) per-class precision >= 0.90 and
recall >= 0.80 on our held-out photos (restrict to the demo categories with --classes); the
report lists exactly which classes fall short.

Writes ``reports/eval_<name>_<date>.md`` and a copy at ``models/<name>/eval_report.md`` (TR-15),
plus the confusion-matrix PNGs Ultralytics saves. Split info comes from the dataset's split.json.

Examples::

    python training/evaluate.py --weights models/grocereye_a/best.pt
    python training/evaluate.py --weights models/grocereye_b/best.pt --data data/ours/data.yaml \\
        --stage b --classes pasta rice cereal coffee tea beans soda chips jam tomato_sauce
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_grocereye_data import write_text  # noqa: E402
from pick_threshold import (Counts, confusion_matrix, gather_images, match_image,  # noqa: E402
                            predict_all, read_server_conf_find, read_yolo_labels, resolve_classes)
from train import REPO_ROOT, load_data_yaml, repo_path  # noqa: E402

GROCEREYE_MAP50 = 0.8459   # GrocerEye README: 84.59% mAP (@0.5), 70.10% average IoU


# --------------------------------------------------------------------------------------------
# Pure report helpers
# --------------------------------------------------------------------------------------------

def fmt(v: float | None, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def confusion_markdown(m: list[list[int]], names: list[str]) -> str:
    """Markdown table for ``m[true][pred]`` with a trailing background row/column."""
    labels = list(names) + ["background"]
    head = "| true \\ pred | " + " | ".join(labels) + " |"
    rows = [head, "|" + "---|" * (len(labels) + 1)]
    for i, row in enumerate(m):
        if i == len(names) and not any(row):
            continue
        cells = [f"**{v}**" if i == j and v else (str(v) if v else ".") for j, v in enumerate(row)]
        rows.append(f"| {labels[i]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def stage_b_shortfalls(op: dict[int, Counts], classes: list[int], names: list[str],
                       min_p: float, min_r: float) -> list[str]:
    """Human-readable list of classes below the NFR-6 targets at the operating point."""
    out = []
    for c in classes:
        cnt = op.get(c, Counts())
        p, r = cnt.precision, cnt.recall
        why = []
        if p is not None and p < min_p:
            why.append(f"precision {p:.2f} < {min_p:.2f}")
        if r is None:
            why.append("no ground truth in this split")
        elif r < min_r:
            why.append(f"recall {r:.2f} < {min_r:.2f}")
        if why:
            out.append(f"{names[c]}: " + ", ".join(why))
    return out


def split_section(split_json: Path | None) -> list[str]:
    """Markdown lines describing the dataset split (TR-2)."""
    if not split_json or not split_json.exists():
        return ["- split.json not found; split unknown."]
    info = json.loads(split_json.read_text(encoding="utf-8"))
    counts = info.get("counts", {})
    return [f"- Source: {info.get('source', '?')}",
            f"- Method: {info.get('method', '?')}",
            "- Images: " + ", ".join(f"{k} {v}" for k, v in counts.items()),
            f"- split.json: `{_rel(split_json)}` (created {info.get('created', '?')})"]


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


def render_report(ctx: dict) -> str:
    """Build the markdown report from a plain dict (see ``main`` for the keys)."""
    names, classes = ctx["names"], ctx["classes"]
    L = [f"# Evaluation: {ctx['name']} ({ctx['date']})", "",
         f"- Weights: `{ctx['weights']}`",
         f"- Data: `{ctx['data']}`, split `{ctx['split']}` ({ctx['n_images']} images, "
         f"{ctx['n_instances']} labeled boxes)",
         f"- imgsz {ctx['imgsz']}, stage {ctx['stage'].upper()}", "",
         "## Dataset split", "", *ctx["split_lines"], "",
         "## Summary", "",
         "| metric | value | target / baseline | status |", "|---|---|---|---|"]
    m50 = ctx["map50"]
    L.append(f"| mAP50 | {m50:.4f} | GrocerEye 0.8459 | "
             f"{'beats' if m50 >= GROCEREYE_MAP50 else 'below'} baseline ({(m50 - GROCEREYE_MAP50) * 100:+.2f} pts) |")
    if ctx["stage"] == "a":
        L.append(f"| mAP50 (NFR-5) | {m50:.4f} | >= {ctx['map_target']:.2f} | "
                 f"{'PASS' if m50 >= ctx['map_target'] else 'FAIL'} |")
    L.append(f"| mAP50-95 | {ctx['map']:.4f} | - | - |")
    if ctx["stage"] == "b":
        short = ctx["shortfalls"]
        L.append(f"| NFR-6 per-class P >= {ctx['p_target']:.2f}, R >= {ctx['r_target']:.2f} at conf "
                 f"{ctx['op_conf']:.2f} | {len(classes) - len(short)}/{len(classes)} classes | all | "
                 f"{'PASS' if not short else 'FAIL'} |")
    L += ["", "GrocerEye's 84.59% is Darknet mAP@0.5 on its own (unpublished) random 20% test split, so the "
          "comparison is approximate.", ""]
    if ctx["stage"] == "b":
        L += ["### Classes below NFR-6", ""]
        L += [f"- {s}" for s in ctx["shortfalls"]] or ["- none"]
        L.append("")
    L += ["## Per-class results", "",
          f"Ultralytics P/R are at the max-F1 confidence; operating-point P/R are at conf "
          f"{ctx['op_conf']:.2f}, IoU >= 0.5.", "",
          "| class | images | boxes | AP50 | AP50-95 | P (val) | R (val) | P @op | R @op | TP | FP | FN |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in classes:
        u = ctx["ul"].get(c, {})
        o = ctx["op"].get(c, Counts())
        L.append(f"| {names[c]} | {u.get('images', 0)} | {u.get('boxes', 0)} | {fmt(u.get('ap50'))} | "
                 f"{fmt(u.get('ap'))} | {fmt(u.get('p'))} | {fmt(u.get('r'))} | {fmt(o.precision)} | "
                 f"{fmt(o.recall)} | {o.tp} | {o.fp} | {o.fn} |")
    L += ["", f"## Confusion matrix (conf {ctx['op_conf']:.2f}, IoU >= 0.5, rows = true class)", "",
          confusion_markdown(ctx["cm"], names), ""]
    for png in ctx.get("pngs", []):
        L.append(f"![{Path(png).stem}]({png})")
    L += ["", "Ultralytics confusion-matrix PNGs are computed at its val conf (0.001), so they show more "
          "low-confidence detections than the table above.", ""]
    if ctx.get("speed"):
        L += ["## Speed (ms per image, this machine)", "",
              ", ".join(f"{k} {v:.1f}" for k, v in ctx["speed"].items()), ""]
    return "\n".join(L)


# --------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--weights", default="models/grocereye_a/best.pt", help="model .pt (default: %(default)s)")
    ap.add_argument("--data", default="data/grocereye/data.yaml", help="data yaml (default: %(default)s)")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--stage", choices=("a", "b"), default="a", help="which NFR targets to check")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="classes to report/check (default: all with ground truth in the split)")
    ap.add_argument("--op-conf", type=float, default=None,
                    help="operating confidence (default: guidance.conf_find in --server-config)")
    ap.add_argument("--map-target", type=float, default=0.85, help="NFR-5 mAP50 target")
    ap.add_argument("--precision-target", type=float, default=0.90, help="NFR-6 per-class precision")
    ap.add_argument("--recall-target", type=float, default=0.80, help="NFR-6 per-class recall")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default=None, help="report name (default: weights' folder name)")
    ap.add_argument("--server-config", default="config/server.yaml")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--project", default="runs/val", help="Ultralytics output dir")
    args = ap.parse_args(argv)

    weights = repo_path(args.weights)
    data_yaml = repo_path(args.data)
    if not weights.exists() or not data_yaml.exists():
        print(f"missing: {weights if not weights.exists() else data_yaml}")
        return 1
    name = args.name or weights.parent.name
    today = date.today().isoformat()
    op_conf = args.op_conf if args.op_conf is not None else (
        read_server_conf_find(repo_path(args.server_config)) or 0.5)

    from ultralytics import YOLO  # lazy
    model = YOLO(str(weights))
    kw = dict(data=str(data_yaml), split=args.split, imgsz=args.imgsz, batch=args.batch, plots=True,
              project=str(repo_path(args.project)), name=f"{name}_{args.split}", exist_ok=True)
    if args.device is not None:
        kw["device"] = args.device
    metrics = model.val(**kw)
    box = metrics.box
    names_map = metrics.names
    names = [names_map[k] for k in sorted(names_map)] if isinstance(names_map, dict) else list(names_map)

    imgs, lbls, _ = gather_images(str(data_yaml), args.split, None, None)
    gts = [read_yolo_labels(p) for p in lbls]
    present = {g[0] for gt in gts for g in gt}
    classes = resolve_classes(args.classes, names, present)

    ul = {}
    for i, c in enumerate(list(box.ap_class_index)):
        ul[int(c)] = {"p": float(box.p[i]), "r": float(box.r[i]), "ap50": float(box.ap50[i]),
                      "ap": float(box.ap[i])}
    for c in range(len(names)):
        d = ul.setdefault(c, {})
        d["images"] = sum(1 for gt in gts if any(g[0] == c for g in gt))
        d["boxes"] = sum(1 for gt in gts for g in gt if g[0] == c)

    preds, _ = predict_all(str(weights), imgs, args.imgsz, args.device, op_conf, 0.7, args.batch)
    pairs = list(zip(preds, gts))
    op: dict[int, Counts] = {}
    for p, g in pairs:
        for c, cnt in match_image(p, g, op_conf).items():
            a = op.setdefault(c, Counts())
            a.tp, a.fp, a.fn = a.tp + cnt.tp, a.fp + cnt.fp, a.fn + cnt.fn
    cm = confusion_matrix(pairs, len(names), op_conf)

    reports_dir = repo_path(args.reports_dir)
    model_dir = repo_path(args.models_dir) / name
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(metrics.save_dir)
    pngs = []
    for png in ("confusion_matrix_normalized.png", "confusion_matrix.png"):
        if (save_dir / png).exists():
            dst = f"eval_{name}_{today}_{png}"
            shutil.copy2(save_dir / png, reports_dir / dst)
            shutil.copy2(save_dir / png, model_dir / png)
            pngs.append(dst)

    split_json = Path(load_data_yaml(data_yaml)["root"]) / "split.json"
    ctx = dict(
        name=name, date=today, weights=_rel(weights), data=_rel(data_yaml), split=args.split,
        n_images=len(imgs), n_instances=sum(len(g) for g in gts), imgsz=args.imgsz, stage=args.stage,
        split_lines=split_section(split_json), map50=float(box.map50), map=float(box.map),
        map_target=args.map_target, p_target=args.precision_target, r_target=args.recall_target,
        op_conf=op_conf, names=names, classes=classes, ul=ul, op=op, cm=cm, pngs=pngs,
        shortfalls=stage_b_shortfalls(op, classes, names, args.precision_target, args.recall_target),
        speed=dict(metrics.speed) if getattr(metrics, "speed", None) else None,
    )
    report = render_report(ctx)
    out = reports_dir / f"eval_{name}_{today}.md"
    write_text(out, report)
    # the models/ copy references PNGs saved next to it
    model_report = report
    for png in pngs:
        model_report = model_report.replace(f"]({png})", f"]({png.split(f'{today}_', 1)[1]})")
    write_text(model_dir / "eval_report.md", model_report)

    print(f"\nmAP50 {box.map50:.4f} (GrocerEye 0.8459), mAP50-95 {box.map:.4f}")
    if args.stage == "a":
        print(f"NFR-5 mAP50 >= {args.map_target}: {'PASS' if box.map50 >= args.map_target else 'FAIL'}")
    else:
        short = ctx["shortfalls"]
        print(f"NFR-6 at conf {op_conf:.2f}: {'PASS' if not short else 'FAIL'}")
        for s in short:
            print(f"  {s}")
    print(f"Report: {out}\nCopy:   {model_dir / 'eval_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
