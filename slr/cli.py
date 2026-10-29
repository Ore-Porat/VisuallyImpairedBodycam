"""`slr` command line: sim, server, device, list, bench, replay, clips.

How it fits in
--------------
This is the front door of the whole project. When you type ``slr sim`` or ``slr list add pasta``
in a terminal, Python starts here. The file does three things:

1. ``build_parser`` describes every command and option so Python can understand what you typed
   (and print ``--help`` text).
2. The ``cmd_*`` functions each carry out one command by loading the config files and calling
   into the real code in slr/server, slr/device, slr/sim.py, slr/bench.py.
3. Hardware helpers for the Raspberry Pi: ``calibrate_volume``, ``test_adc``,
   ``test_controls`` and ``test_camera``.

The README's "Command reference" table lists what each command is for.

Words you will see
------------------
- argparse: Python's standard library for reading command-line arguments. A "subparser" is a
  sub-command such as ``sim`` or ``list``; each gets its own options.
- ``args``: the object argparse hands back, with one attribute per option (``args.camera``).
- asyncio / coroutine: the server and device are written with ``async`` functions so they can
  wait for the network, camera and audio at the same time. ``asyncio.run(...)`` starts such a
  function and runs it to completion. A "coroutine" is the not-yet-started async call.
- logging: the standard way to print timestamped status messages; ``-v`` makes it chattier.
- lazy import: ``from x import y`` written inside a function instead of at the top. It makes
  ``slr --help`` and the light commands start fast, because heavy libraries (the neural
  network, camera, audio) are only loaded by the command that needs them.
- stderr: the terminal's separate channel for error messages (``file=sys.stderr``).
- sys.exit(2): stop the program with exit code 2, which scripts can read as "bad input".
- raise SystemExit("msg"): same as printing the message and exiting with an error.
- GPIO: the Raspberry Pi's input/output pins; ADC / MCP3002: the chip that reads the volume
  dial; FR-xx: requirement numbers from the product spec, PRD.pdf.
- regular expression (re): a pattern language for finding text, used here to find
  ``raw_min: 399`` lines in device.yaml and replace just the number.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

from slr.common.config import (ConfigError, default_config_dir, item_classes, load_classes, load_controls_config,
                               load_device_config, load_server_config)


def _setup_logging(verbose: bool) -> None:
    """Configure the log format once for the whole program.

    ``verbose`` (the -v flag) shows DEBUG messages too. A few chatty third-party libraries are
    turned down to WARNING so they do not flood the terminal.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("websockets", "asyncio", "urllib3", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _run(coro) -> None:
    """Run an async function to completion, and treat Ctrl+C as a normal, quiet exit."""
    try:
        asyncio.run(coro)
    except KeyboardInterrupt:
        print("\nstopped")


# --------------------------------------------------------------------------- commands
# One function per sub-command. argparse calls the right one via ``args.func`` (see main()).


def cmd_sim(args) -> None:
    """``slr sim``: run server + simulated device on this laptop (webcam, keyboard, speakers)."""
    from slr.sim import run_sim

    # Load all three config files; sim=True applies the ``sim:`` overrides from device.yaml.
    scfg = load_server_config(args.config_dir)
    dcfg = load_device_config(args.config_dir, sim=True)
    ccfg = load_controls_config(args.config_dir)
    # Command-line flags override the config.
    if args.camera is not None:
        dcfg.camera.backend, dcfg.camera.index = "opencv", args.camera
    if args.file:
        dcfg.camera.backend, dcfg.camera.file = "file", args.file
    if args.no_audio:
        dcfg.audio.backend = "null"
    # debug_view=None means "whatever the config says"; --no-debug forces it off.
    _run(run_sim(scfg, dcfg, ccfg, debug_view=False if args.no_debug else None))


def cmd_server(args) -> None:
    """``slr server``: run the laptop server and wait for the real device to connect."""
    from slr.server.main import run_server

    cfg = load_server_config(args.config_dir)
    _run(run_server(cfg, debug_view=False if args.no_debug else None))


def cmd_device(args) -> None:
    """``slr device [run|calibrate-volume|test-controls|test-camera|test-adc]``: run a device tool."""
    dcfg = load_device_config(args.config_dir, sim=args.sim)
    ccfg = load_controls_config(args.config_dir)
    if args.action == "run":
        from slr.device.main import run_device

        _run(run_device(dcfg, ccfg, sim=args.sim))
    elif args.action == "calibrate-volume":
        calibrate_volume(dcfg, args.config_dir or default_config_dir())
    elif args.action == "test-controls":
        test_controls(dcfg, ccfg)
    elif args.action == "test-camera":
        test_camera(dcfg)
    elif args.action == "test-adc":
        test_adc(dcfg)


def cmd_list(args) -> None:
    """``slr list show|add|remove|clear|reset|classes``: view and edit the shopping list.

    The list is a small YAML file (data/shopping_list.yaml) shared with the running server,
    which notices the change and reloads it.
    """
    from slr.server.session import ListError, ShoppingList

    cfg = load_server_config(args.config_dir)
    # Valid item names come from the model's class list.
    classes = item_classes(load_classes(cfg.path(cfg.model.classes_file)),
                           cfg.guidance.non_item_classes)   # no "hand" on the list
    lst = ShoppingList(cfg.path(cfg.paths.shopping_list), classes)
    # ``slr list add tomato sauce`` arrives as two words; join them back into one name.
    # getattr(..., None) is used because "show"/"clear" have no ``item`` argument at all.
    name = " ".join(args.item) if getattr(args, "item", None) else ""
    try:
        if args.list_cmd == "add":
            cls = lst.add(name)
            # classes.get(cls, cls) prints the spoken name if there is one, else the id.
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
        # e.g. an item name the model does not know. Print to stderr and exit with code 2.
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    # Every command except "classes" ends by showing the current list ("show" does only this).
    print(lst.describe())


def cmd_replay(args) -> None:
    """``slr replay <video> --script s.yaml``: re-run a recording and print the cue timeline.

    Useful for regression testing: the same video and button script should always produce
    the same spoken cues. ``--detections`` swaps the real model for pre-recorded boxes.
    """
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
        # indent=1 makes the JSON file readable by humans.
        Path(args.out).write_text(json.dumps(timeline, indent=1))
        print(f"\ntimeline written to {args.out}")


def cmd_bench(args) -> None:
    """``slr bench``: measure latency, FPS and UDP loss (see bench.py for details).

    With ``--log`` it just analyzes an existing session log. Otherwise it runs a live loopback
    test at each resolution in ``--resolutions`` using the webcam, a sample image, or a file.
    """
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
        # Quick probe: try to open the webcam at a tiny size and close it again. If that fails
        # (no camera on this machine), fall back to a bundled sample picture.
        try:
            OpenCVCamera(dcfg.camera.index, 320, 240).close()
        except CameraError:
            source = _sample_image()
            print(f"no webcam; benchmarking with {source}")
    elif source == "sample":
        source = _sample_image()
    # Load the neural network once and reuse it for every resolution.
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
        # Create the folder if needed, then write a small Markdown report with a header line.
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(f"# slr bench {time.strftime('%Y-%m-%d %H:%M')}\n\n"
                                  f"source: {source}, target: {args.target}, model: {scfg.model.weights}\n\n"
                                  + report + "\n")
        print(f"report written to {args.out}")


def _sample_image() -> str:
    """Find a picture to benchmark with when there is no webcam: the bus.jpg shipped with ultralytics."""
    try:
        import ultralytics

        p = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        if p.exists():
            return str(p)
    except ImportError:
        pass
    raise SystemExit("no webcam and no sample image; pass --source <video|image|folder>")


def cmd_clips(args) -> None:
    """``slr clips build|build-system|download-voice``: create the speech audio files.

    Speech is generated once with a text-to-speech (TTS) engine and saved as audio clips, so
    at run time the device just plays files (fast, works offline).
    - download-voice: fetch the Piper voice model (one time, needs internet).
    - build-system: render the clips the device needs with no server ("starting up", ...).
    - build: pre-render every phrase for every item class into the server's clip cache.
    """
    from slr.server.clips import (ClipStore, PhraseCatalog, build_system_clips, download_voice, make_tts,
                                  voice_tag)
    from slr.server.session import ShoppingList

    cfg = load_server_config(args.config_dir)
    if args.clips_cmd == "download-voice":
        print(f"voice saved to {download_voice(cfg)}")
        return
    # Pick the first TTS engine from the config that actually works on this machine.
    tts = make_tts(cfg.tts, cfg)
    print(f"TTS engine: {tts.name}")
    classes = load_classes(cfg.path(cfg.model.classes_file))
    # The catalog is every sentence the product can say, from config/phrases.yaml, with item
    # names filled in. The voice tag keeps clips from different voices apart.
    catalog = PhraseCatalog.load(cfg.root / "config" / "phrases.yaml", classes, voice_tag(cfg.tts, tts))
    if args.clips_cmd == "build-system":
        out = cfg.root / "device_assets" / "system_clips"
        paths = build_system_clips(catalog, tts, out)
        print(f"wrote {len(paths)} system clips to {out}")
        return
    store = ClipStore(cfg.path(cfg.paths.clips_dir), tts)
    lst = ShoppingList(cfg.path(cfg.paths.shopping_list), classes)
    # Render whatever is not cached yet; returns how many new clips were made.
    n = store.prerender(catalog, catalog.prerender_set(lst.items, all_classes=True))
    print(f"rendered {n} new clips; {len(store.manifest())} clips in {store.dir}")


# --------------------------------------------------------------------------- hardware tools
# These three only make sense on the Raspberry Pi with the real hardware attached.


def calibrate_volume(dcfg, config_dir: Path) -> None:
    """FR-44: record raw ADC readings at dial min / middle / max into config/device.yaml.

    The user is asked to turn the dial to each position and press Enter; 40 quick readings are
    taken and their median (middle value, immune to the odd glitch) is saved. The three numbers
    are then written back into device.yaml in place, leaving the rest of the file untouched.

    - ``dcfg``: device config (for the SPI/ADC wiring).
    - ``config_dir``: folder containing device.yaml to update.
    """
    import statistics

    from slr.device.volume_dial import Mcp3002Reader, check_calibration_points

    try:
        reader = Mcp3002Reader(dcfg.volume.adc)
    except Exception as e:
        raise SystemExit(f"cannot read the MCP3002: {e}")
    points = {}
    for key, label in (("raw_min", "MINIMUM"), ("raw_mid", "the MIDDLE"), ("raw_max", "MAXIMUM")):
        # input() pauses until the user presses Enter.
        input(f"Turn the volume dial to {label} and press Enter...")
        samples = []
        for _ in range(40):
            samples.append(reader.read())
            time.sleep(0.025)
        points[key] = int(statistics.median(samples))
        # "spread" = max - min of the samples; a big spread hints at a noisy or loose wire.
        print(f"  {key} = {points[key]} (spread {max(samples) - min(samples)} counts)")
    # Reject a point that sits on one ADC rail while the other end does not (a wiper that opened
    # at that end), and reject readings that do not climb or fall steadily. Never overwrite a
    # good calibration with a clearly broken one.
    try:
        check_calibration_points(points)
    except ValueError as e:
        raise SystemExit(f"{e}; calibration not saved")
    path = config_dir / "device.yaml"
    text = path.read_text(encoding="utf-8")
    for key, val in points.items():
        # Find the line "    raw_min: 464" and replace only the number. re.M makes ^ match the
        # start of each line; count=1 replaces the first match; n tells us how many matched.
        text, n = re.subn(rf"(^\s*{key}:\s*)[-\d.]+", rf"\g<1>{val}", text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"could not find {key} in {path}")
    path.write_text(text, encoding="utf-8")
    print(f"saved calibration to {path}")


def test_camera(dcfg) -> None:
    """Open the configured camera, grab one fresh frame and save it as a JPEG.

    This is the quickest way to check a USB webcam on the headless Pi: it prints the picture
    size and writes ``data/camera_test.jpg`` so you can copy it to the laptop and look at it.
    """
    from slr.device.camera import CameraError, make_jpeg_encoder, open_camera

    try:
        cam = open_camera(dcfg.camera, dcfg.root)
    except CameraError as e:
        raise SystemExit(f"camera unavailable: {e}")
    try:
        try:
            frame = cam.capture()
        except CameraError as e:
            raise SystemExit(f"camera opened but did not deliver a frame: {e}")
    finally:
        cam.close()
    out = dcfg.path("data/camera_test.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.write_bytes(make_jpeg_encoder()(frame, 85))
    except ImportError as e:
        raise SystemExit(f"cannot encode a JPEG: {e}")
    print(f"camera OK: {frame.shape[1]}x{frame.shape[0]} ({dcfg.camera.backend})")
    print(f"saved one frame to {out}")


def test_adc(dcfg) -> None:
    """Read the MCP3002 volume dial and show the raw SPI bytes, count and voltage.

    This needs only SPI, not GPIO or the camera, so it is the quickest way to check the dial
    divider and wiring. Turn the knob through its full range; the min/max values show whether
    the input is actually moving. Stop with Ctrl+C.
    """
    from slr.device.volume_dial import Mcp3002Reader

    try:
        reader = Mcp3002Reader(dcfg.volume.adc)
    except Exception as e:
        raise SystemExit(f"cannot read the MCP3002: {e}")
    low, high = 1023, 0
    print(f"MCP3002 on SPI{dcfg.volume.adc.spi_bus}.{dcfg.volume.adc.spi_device}, "
          f"CH{dcfg.volume.adc.channel}. Turn the knob; Ctrl+C to stop.")
    try:
        while True:
            raw = reader.read()
            low, high = min(low, raw), max(high, raw)
            volts = raw / 1023.0 * 3.3
            print(f"\rrx={reader.rx_text()}  raw={raw:4d}  V={volts:.3f}  "
                  f"range={low}-{high}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        reader.close()


def test_controls(dcfg, ccfg) -> None:
    """FR-76: live readings of every button, the mode switch and the volume dial.

    Prints one constantly-updating line in the terminal showing each button (DOWN/up), the two
    mode switch wires and the decoded position, the raw dial reading with its SPI bytes and
    voltage, and any events the button logic produced (short press, long press, ...). Stop with
    Ctrl+C.
    """
    from slr.device.controls import GpioControls
    from slr.device.volume_dial import Mcp3002Reader, VolumeCurve, VolumeFilter

    # The controls object reports events by calling a function we give it. ``events.append``
    # is that function here, so events simply pile up in this list until we print them.
    events: list[str] = []
    try:
        ctl = GpioControls(ccfg, events.append, volume_buttons=False)
    except Exception as e:
        raise SystemExit(f"GPIO unavailable (run this on the Pi): {e}")
    ctl.start()
    # The dial is optional: if the ADC chip cannot be read, show everything else anyway.
    try:
        reader = Mcp3002Reader(dcfg.volume.adc)
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
                # Feed the raw count through the same smoothing/levels logic the device uses.
                filt.add_sample(raw, time.monotonic())
                volts = raw / 1023.0 * 3.3
                line += (f" | dial raw={raw} rx={reader.rx_text()} V={volts:.3f} "
                         f"pos={filt.last_position * 100:5.1f}% level={filt.level}")
            if events:
                line += " | events: " + ", ".join(events)
                events.clear()
            # "\r" moves the cursor back to the start of the line so the same line is
            # overwritten each time; ljust pads it so leftovers from a longer line are erased.
            print("\r" + line.ljust(180), end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        ctl.stop()
        if reader is not None:
            reader.close()


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """Describe every ``slr`` command and option. Returns the argparse parser.

    Each ``sub.add_parser(...)`` is one sub-command; ``set_defaults(func=...)`` attaches the
    function that carries it out. Run ``slr --help`` or ``slr <command> --help`` to see the
    result of this description.
    """
    ap = argparse.ArgumentParser(prog="slr", description="SLR SLR: Single Lens Reflex Sightless Label Reader")
    # Options that apply to every command go on the top-level parser.
    ap.add_argument("--config-dir", type=Path, default=None, help="config directory (default ./config)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # slr sim
    s = sub.add_parser("sim", help="server + simulated device on this laptop (webcam, keyboard, speakers)")
    s.add_argument("--camera", type=int, default=None, help="webcam index")
    s.add_argument("--file", help="use a video / image / folder instead of the webcam")
    s.add_argument("--no-debug", action="store_true", help="no debug window")
    s.add_argument("--no-audio", action="store_true", help="no sound (logs only)")
    s.set_defaults(func=cmd_sim)

    # slr server
    s = sub.add_parser("server", help="run the laptop server for the real device")
    s.add_argument("--no-debug", action="store_true")
    s.set_defaults(func=cmd_server)

    # slr device [action]   (nargs="?" = the action is optional and defaults to "run")
    s = sub.add_parser("device", help="run the device, or device tools")
    s.add_argument("action", nargs="?", default="run",
                   choices=["run", "calibrate-volume", "test-controls", "test-camera", "test-adc"])
    s.add_argument("--sim", action="store_true", help="use the sim overrides from device.yaml")
    s.set_defaults(func=cmd_device)

    # slr list <sub-command>   (nargs="+" = one or more words, so "tomato sauce" works)
    s = sub.add_parser("list", help="view and edit the shopping list")
    lsub = s.add_subparsers(dest="list_cmd", required=True)
    lsub.add_parser("show")
    lsub.add_parser("add").add_argument("item", nargs="+")
    lsub.add_parser("remove").add_argument("item", nargs="+")
    lsub.add_parser("clear")
    lsub.add_parser("reset", help="mark every item pending again")
    lsub.add_parser("classes", help="list valid item names")
    s.set_defaults(func=cmd_list)

    # slr bench
    s = sub.add_parser("bench", help="measure latency, FPS and frame loss")
    s.add_argument("--duration", type=float, default=10.0, help="seconds per resolution")
    s.add_argument("--resolutions", default="640x480,416x312,320x240")
    s.add_argument("--source", default="webcam", help="webcam | sample | path to video/image/folder")
    s.add_argument("--target", default="pasta",
                   help="item to search for; hold one in view to measure capture->audio latency")
    s.add_argument("--log", help="analyze an existing session log instead")
    s.add_argument("--out", help="write a markdown report here")
    s.set_defaults(func=cmd_bench)

    # slr replay <source> --script <yaml>
    s = sub.add_parser("replay", help="replay a recording with a scripted control sequence")
    s.add_argument("source", help="video file or folder of frames")
    s.add_argument("--script", required=True, help="YAML control script")
    s.add_argument("--detections", help="scripted detections JSON instead of the model")
    s.add_argument("--fps", type=float)
    s.add_argument("--out", help="write the timeline as JSON")
    s.set_defaults(func=cmd_replay)

    # slr clips <sub-command>
    s = sub.add_parser("clips", help="render speech clips")
    csub = s.add_subparsers(dest="clips_cmd", required=True)
    csub.add_parser("build", help="pre-render every phrase for every class")
    csub.add_parser("build-system", help="render the device's local system clips")
    csub.add_parser("download-voice", help="download the Piper voice (one time, needs internet)")
    s.set_defaults(func=cmd_clips)
    return ap


def main(argv: list[str] | None = None) -> None:
    """Program entry point: parse the command line, set up logging, run the chosen command.

    ``argv`` lets tests pass a list of arguments; None means "use what was typed in the
    terminal". Config problems are printed as one friendly line and exit with code 2.
    """
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        # args.func is whichever cmd_* function the chosen sub-command registered.
        args.func(args)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        sys.exit(2)


# This is true only when the file is run directly (python slr/cli.py), not when it is imported
# by __main__.py or the ``slr`` command; in those cases they call main() themselves.
if __name__ == "__main__":
    main()
