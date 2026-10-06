"""`slr` command line: sim, server, device, list, bench, replay, clips."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

from slr.common.config import (ConfigError, default_config_dir, load_classes, load_controls_config,
                               load_device_config, load_server_config)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("websockets", "asyncio", "urllib3", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _run(coro) -> None:
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        print("\nstopped")


# --------------------------------------------------------------------------- commands


def cmd_sim(args) -> None:
    from slr.sim import run_sim

    scfg = load_server_config(args.config_dir)
    dcfg = load_device_config(args.config_dir, sim=True)
    ccfg = load_controls_config(args.config_dir)
    if args.camera is not None:
        dcfg.camera.backend, dcfg.camera.index = "opencv", args.camera
    if args.file:
        dcfg.camera.backend, dcfg.camera.file = "file", args.file
    if args.no_audio:
        dcfg.audio.backend = "null"
    _run(run_sim(scfg, dcfg, ccfg, debug_view=False if args.no_debug else None))


def cmd_server(args) -> None:
    from slr.server.main import run_server

    cfg = load_server_config(args.config_dir)
    _run(run_server(cfg, debug_view=False if args.no_debug else None))


def cmd_device(args) -> None:
    dcfg = load_device_config(args.config_dir, sim=args.sim)
    ccfg = load_controls_config(args.config_dir)
    if args.action == "run":
        from slr.device.main import run_device

        _run(run_device(dcfg, ccfg, sim=args.sim))
    elif args.action == "calibrate-volume":
        calibrate_volume(dcfg, args.config_dir or default_config_dir())
    elif args.action == "test-controls":
        test_controls(dcfg, ccfg)


def cmd_list(args) -> None:
    from slr.server.session import ListError, ShoppingList

    cfg = load_server_config(args.config_dir)
    classes = load_classes(cfg.path(cfg.model.classes_file))
    lst = ShoppingList(cfg.path(cfg.paths.shopping_list), classes)
    name = " ".join(args.item) if getattr(args, "item", None) else ""
    try:
        if args.list_cmd == "add":
            cls = lst.add(name)
            print(f"Added {classes.get(cls, cls)}.")
        elif args.list_cmd == "remove":
            cls = lst.remove(name)
            print(f"Removed {classes.get(cls, cls)}.")
        elif args.list_cmd == "clear":
            lst.clear()
            print("List cleared.")
        elif args.list_cmd == "reset":
            lst.reset()
            print("All items marked pending.")
        elif args.list_cmd == "classes":
            print("Valid items: " + ", ".join(sorted(classes)))
            return
    except ListError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    print(lst.describe())


def cmd_replay(args) -> None:
    from slr.server.detector import ScriptedDetector, load_detector
    from slr.server.replay import format_timeline, load_script, run_replay

    cfg = load_server_config(args.config_dir)
    script = load_script(args.script)
    if args.fps:
        script["fps"] = args.fps
    detector = ScriptedDetector(args.detections) if args.detections else load_detector(cfg)
    timeline = run_replay(cfg, args.source, script, detector)
    print(format_timeline(timeline))
    if args.out:
        Path(args.out).write_text(json.dumps(timeline, indent=1))
        print(f"\ntimeline written to {args.out}")


def cmd_bench(args) -> None:
    from slr.bench import analyze_log, bench_once, format_report, parse_resolutions

    if args.log:
        print(format_report([(Path(args.log).name, analyze_log(Path(args.log)))]))
        return
    scfg = load_server_config(args.config_dir)
    dcfg = load_device_config(args.config_dir, sim=True)
    ccfg = load_controls_config(args.config_dir)
    from slr.device.camera import CameraError, FileCamera, OpenCVCamera
    from slr.server.detector import load_detector

    source = args.source
    if source == "webcam":
        try:
            OpenCVCamera(dcfg.camera.index, 320, 240).close()
        except CameraError:
            source = _sample_image()
            print(f"no webcam; benchmarking with {source}")
    elif source == "sample":
        source = _sample_image()
    detector = load_detector(scfg)
    rows = []
    for w, h in parse_resolutions(args.resolutions):
        cam = OpenCVCamera(dcfg.camera.index, w, h) if source == "webcam" else FileCamera(source, w, h)
        print(f"benchmarking {w}x{h} for {args.duration:.0f} s ...")
        r = asyncio.run(bench_once(scfg, dcfg, ccfg, w, h, args.duration, args.target, detector, cam))
        rows.append((f"{w}x{h}", r))
    report = format_report(rows)
    print("\n" + report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(f"# slr bench {time.strftime('%Y-%m-%d %H:%M')}\n\n"
                                  f"source: {source}, target: {args.target}, model: {scfg.model.weights}\n\n"
                                  + report + "\n")
        print(f"report written to {args.out}")


def _sample_image() -> str:
    try:
        import ultralytics

        p = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        if p.exists():
            return str(p)
    except ImportError:
        pass
    raise SystemExit("no webcam and no sample image; pass --source <video|image|folder>")


def cmd_clips(args) -> None:
    from slr.server.clips import (ClipStore, PhraseCatalog, build_system_clips, download_voice, make_tts,
                                  voice_tag)
    from slr.server.session import ShoppingList

    cfg = load_server_config(args.config_dir)
    if args.clips_cmd == "download-voice":
        print(f"voice saved to {download_voice(cfg)}")
        return
    tts = make_tts(cfg.tts, cfg)
    print(f"TTS engine: {tts.name}")
    classes = load_classes(cfg.path(cfg.model.classes_file))
    catalog = PhraseCatalog.load(cfg.root / "config" / "phrases.yaml", classes, voice_tag(cfg.tts, tts))
    if args.clips_cmd == "build-system":
        out = cfg.root / "device_assets" / "system_clips"
        paths = build_system_clips(catalog, tts, out)
        print(f"wrote {len(paths)} system clips to {out}")
        return
    store = ClipStore(cfg.path(cfg.paths.clips_dir), tts)
    lst = ShoppingList(cfg.path(cfg.paths.shopping_list), classes)
    n = store.prerender(catalog, catalog.prerender_set(lst.items, all_classes=True))
    print(f"rendered {n} new clips; {len(store.manifest())} clips in {store.dir}")


# --------------------------------------------------------------------------- hardware tools


def calibrate_volume(dcfg, config_dir: Path) -> None:
    """FR-44: record raw ADC readings at dial min / middle / max into config/device.yaml."""
    import statistics

    from slr.device.volume_dial import Ads1115Reader

    try:
        reader = Ads1115Reader(dcfg.volume.adc)
    except Exception as e:
        raise SystemExit(f"cannot read the ADS1115: {e}")
    points = {}
    for key, label in (("raw_min", "MINIMUM"), ("raw_mid", "the MIDDLE"), ("raw_max", "MAXIMUM")):
        input(f"Turn the volume dial to {label} and press Enter...")
        samples = []
        for _ in range(40):
            samples.append(reader.read())
            time.sleep(0.025)
        points[key] = int(statistics.median(samples))
        print(f"  {key} = {points[key]} (spread {max(samples) - min(samples)} counts)")
    vals = [points["raw_min"], points["raw_mid"], points["raw_max"]]
    if not (vals[0] < vals[1] < vals[2] or vals[0] > vals[1] > vals[2]):
        raise SystemExit("readings are not monotonic; check the wiring (FR-44) and try again")
    path = config_dir / "device.yaml"
    text = path.read_text(encoding="utf-8")
    for key, val in points.items():
        text, n = re.subn(rf"(^\s*{key}:\s*)[-\d.]+", rf"\g<1>{val}", text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"could not find {key} in {path}")
    path.write_text(text, encoding="utf-8")
    print(f"saved calibration to {path}")


def test_controls(dcfg, ccfg) -> None:
    """FR-76: live readings of every button, the mode switch and the volume dial."""
    from slr.device.controls import GpioControls
    from slr.device.volume_dial import Ads1115Reader, VolumeCurve, VolumeFilter

    events: list[str] = []
    try:
        ctl = GpioControls(ccfg, events.append, volume_buttons=False)
    except Exception as e:
        raise SystemExit(f"GPIO unavailable (run this on the Pi): {e}")
    ctl.start()
    try:
        reader = Ads1115Reader(dcfg.volume.adc)
        filt = VolumeFilter(VolumeCurve(dcfg.volume.calibration), dcfg.volume)
    except Exception as e:
        print(f"volume dial: not available ({e})")
        reader = filt = None
    print("Ctrl+C to stop.\n")
    try:
        while True:
            r = ctl.read()
            buttons = " ".join(f"{n}={'DOWN' if v else 'up  '}" for n, v in r["buttons"].items())
            line = f"{buttons} | switch a={r['a']} b={r['b']} -> {r['position'] or 'INVALID'}"
            if reader is not None:
                raw = reader.read()
                filt.add_sample(raw, time.monotonic())
                line += f" | dial raw={raw} pos={filt.last_position * 100:5.1f}% level={filt.level}"
            if events:
                line += " | events: " + ", ".join(events)
                events.clear()
            print("\r" + line.ljust(150), end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        ctl.stop()


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="slr", description="SLR SLR: Single Lens Reflex Sightless Label Reader")
    ap.add_argument("--config-dir", type=Path, default=None, help="config directory (default ./config)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sim", help="server + simulated device on this laptop (webcam, keyboard, speakers)")
    s.add_argument("--camera", type=int, default=None, help="webcam index")
    s.add_argument("--file", help="use a video / image / folder instead of the webcam")
    s.add_argument("--no-debug", action="store_true", help="no debug window")
    s.add_argument("--no-audio", action="store_true", help="no sound (logs only)")
    s.set_defaults(func=cmd_sim)

    s = sub.add_parser("server", help="run the laptop server for the real device")
    s.add_argument("--no-debug", action="store_true")
    s.set_defaults(func=cmd_server)

    s = sub.add_parser("device", help="run the device, or device tools")
    s.add_argument("action", nargs="?", default="run", choices=["run", "calibrate-volume", "test-controls"])
    s.add_argument("--sim", action="store_true", help="use the sim overrides from device.yaml")
    s.set_defaults(func=cmd_device)

    s = sub.add_parser("list", help="view and edit the shopping list")
    lsub = s.add_subparsers(dest="list_cmd", required=True)
    lsub.add_parser("show")
    lsub.add_parser("add").add_argument("item", nargs="+")
    lsub.add_parser("remove").add_argument("item", nargs="+")
    lsub.add_parser("clear")
    lsub.add_parser("reset", help="mark every item pending again")
    lsub.add_parser("classes", help="list valid item names")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("bench", help="measure latency, FPS and frame loss")
    s.add_argument("--duration", type=float, default=10.0, help="seconds per resolution")
    s.add_argument("--resolutions", default="640x480,416x312,320x240")
    s.add_argument("--source", default="webcam", help="webcam | sample | path to video/image/folder")
    s.add_argument("--target", default="pasta",
                   help="item to search for; hold one in view to measure capture->audio latency")
    s.add_argument("--log", help="analyze an existing session log instead")
    s.add_argument("--out", help="write a markdown report here")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("replay", help="replay a recording with a scripted control sequence")
    s.add_argument("source", help="video file or folder of frames")
    s.add_argument("--script", required=True, help="YAML control script")
    s.add_argument("--detections", help="scripted detections JSON instead of the model")
    s.add_argument("--fps", type=float)
    s.add_argument("--out", help="write the timeline as JSON")
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("clips", help="render speech clips")
    csub = s.add_subparsers(dest="clips_cmd", required=True)
    csub.add_parser("build", help="pre-render every phrase for every class")
    csub.add_parser("build-system", help="render the device's local system clips")
    csub.add_parser("download-voice", help="download the Piper voice (one time, needs internet)")
    s.set_defaults(func=cmd_clips)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
