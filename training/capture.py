"""Capture Stage B training photos from the laptop webcam or the Pi camera (TR-11, TR-12).

WHAT THIS FILE IS FOR
---------------------
The product (see README.md) uses an object detector, a small neural network that looks at a camera
frame and draws boxes around grocery items it recognizes. The detector we ship ("Stage A") was
trained on photos of German packaging taken close up. To make it work on US products seen from a
shelf, we need our own photos ("Stage B", see training/README.md). This script is the camera tool
that takes those photos. It does nothing clever: it opens a camera, shows a live preview, and saves
a JPEG every time you press a key (or every N seconds). The saved photos are later labeled by hand
in a labeling tool and fed to training/train.py.

The requirement numbers in brackets (TR-11, TR-12) refer to the product spec, PRD.pdf. "TR" means
training requirement.

Images go to ``data/raw/<session>/<session>_0001.jpg`` (session defaults to a timestamp). Use one
session per capture setup (place, lighting, camera) so the dataset can later be split by session
instead of randomly (TR-12): near-duplicate frames then never leak into the test set.

Controls:

- Preview window (default): SPACE saves a frame, Q or ESC quits.
- ``--headless`` (Pi over SSH, no display): press ENTER to save, type ``q`` + ENTER to quit.
- ``--interval N`` additionally saves a frame every N seconds (works with both).

Backends: ``opencv`` (laptop webcam, USB cameras) or ``picamera2`` (Pi camera through the SLR
lens, for domain match). ``auto`` uses picamera2 when it is importable, else OpenCV.

Examples::

    python training/capture.py --session store_a_dim --interval 2
    python training/capture.py --backend picamera2 --headless --interval 1.5 --resolution 1280x960

WORDS YOU WILL SEE
------------------
- **Backend**: the library that actually talks to the camera hardware. Two are supported here:
  OpenCV (``cv2``, works with any webcam) and picamera2 (only on a Raspberry Pi).
- **Frame**: one picture from the camera, stored as a grid of pixels (a NumPy array).
- **BGR**: the pixel color order OpenCV uses (blue, green, red) instead of the usual RGB.
- **Headless**: running without a screen/window, e.g. over an SSH terminal to the Pi.
- **Session**: a named folder of photos taken in one place with one setup.
- **argparse**: the Python module that reads ``--options`` typed on the command line.
- **Path**: a Python object that represents a file or folder path (from ``pathlib``).
- **Thread**: a second "worker" running at the same time as the main code; used here so the
  program can wait for keyboard input and read the camera at once.
- **Queue**: a thread-safe to-do list that one thread puts items into and another takes out.
- **Test set / data leak**: the photos kept aside to measure the model honestly. If a nearly
  identical photo is both in training and in the test set, the score is inflated (a "leak").
"""

# This line lets us write type hints like `tuple[int, int]` on older Python versions. Type hints
# (the `: int` and `-> str` bits on functions) are notes about what kind of value is expected.
# Python does not enforce them; they are documentation for people and tools.
from __future__ import annotations

import argparse      # reads command-line options such as --session and --interval
import queue         # a thread-safe list used to pass keyboard commands between threads
import sys           # access to stdin (keyboard input), the platform name, and exit codes
import threading     # lets us read the keyboard in the background while grabbing frames
import time          # clocks and sleeping
from datetime import datetime   # used to build the default session name from today's date/time
from pathlib import Path        # object-oriented file paths

# The folder that contains the whole repository. `__file__` is this script's own path;
# `.parent.parent` goes up two levels: training/ -> repo root. Output paths are built from this so
# the script works no matter which folder you run it from.
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_resolution(s: str) -> tuple[int, int]:
    """'1280x720' -> (1280, 720).

    Used by argparse to turn the text typed after ``--resolution`` into two numbers. If the text
    is not of the form WIDTHxHEIGHT, argparse shows a friendly error and stops.
    """
    # try/except: attempt the conversion; if it fails with a ValueError (e.g. "abc" is not a
    # number), jump to the `except` block instead of crashing.
    try:
        # Accept "1280x720", "1280X720" or "1280*720": lowercase it, swap * for x, split on x,
        # and convert both halves to integers.
        w, h = (int(x) for x in s.lower().replace("*", "x").split("x"))
    except ValueError as e:
        # ArgumentTypeError is the error type argparse knows how to print nicely.
        raise argparse.ArgumentTypeError(f"resolution must look like 1280x720, got {s!r}") from e
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return w, h


class OpenCVCamera:
    """Webcam via OpenCV. ``read()`` returns a BGR frame or None.

    A "class" bundles data (here the open camera handle) with the functions that use it. Both
    camera classes in this file offer the same three methods (``read``, ``close`` and the
    constructor), so the rest of the script does not care which one it got.
    """

    def __init__(self, index: int, resolution: tuple[int, int]):
        """Open webcam number ``index`` (0 = the first camera) at the requested size.

        ``__init__`` is the "constructor": it runs when you write ``OpenCVCamera(0, (1280, 720))``.
        """
        # OpenCV is imported inside the function so that `--help` works on machines without it.
        import cv2
        self.cv2 = cv2
        # On Windows, DirectShow (CAP_DSHOW) opens webcams much faster than the default driver.
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, api)
        # If the preferred driver failed, fall back to letting OpenCV pick any driver.
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        # Ask the camera for the requested size (it may silently pick the closest it supports).
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    def read(self):
        """Grab one frame. Returns the image, or None if the camera gave nothing this time."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        """Release the camera so other programs can use it."""
        self.cap.release()


class PiCamera:
    """Raspberry Pi camera via picamera2 (installed from apt on the Pi).

    Same three methods as ``OpenCVCamera`` so the two are interchangeable.
    """

    def __init__(self, resolution: tuple[int, int]):
        """Start the Pi camera at the requested size and wait a second for exposure to settle."""
        from picamera2 import Picamera2  # optional dependency
        self.cam = Picamera2()
        # "Still" configuration = optimized for single photos rather than video. RGB888 is the
        # pixel format; see the note in read() about its byte order.
        cfg = self.cam.create_still_configuration(main={"size": resolution, "format": "RGB888"})
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.0)  # let auto-exposure settle

    def read(self):
        """Grab one frame as a NumPy array."""
        return self.cam.capture_array()  # RGB888 in picamera2 is BGR byte order, as OpenCV expects

    def close(self):
        """Stop the camera."""
        self.cam.stop()


def open_camera(backend: str, index: int, resolution: tuple[int, int]):
    """Create the right camera object for ``backend`` ("auto", "opencv" or "picamera2").

    Returns a pair: (camera object, name of the backend actually used). With "auto" it tries
    the Pi camera first and silently falls back to OpenCV when picamera2 is not installed.
    """
    if backend in ("auto", "picamera2"):
        try:
            return PiCamera(resolution), "picamera2"
        except ImportError:
            # picamera2 is missing. That is only an error if the user explicitly asked for it.
            if backend == "picamera2":
                raise SystemExit("picamera2 is not installed (sudo apt install python3-picamera2)")
    return OpenCVCamera(index, resolution), "opencv"


def next_index(folder: Path, prefix: str) -> int:
    """First unused image number in ``folder`` (so re-running a session appends).

    Files are named ``<prefix>_0001.jpg``, ``<prefix>_0002.jpg``, ... This looks at what already
    exists and returns the next free number, so a second run of the same session never
    overwrites earlier photos.
    """
    nums = []
    # `glob` lists files matching a pattern; `*` means "anything".
    for p in folder.glob(f"{prefix}_*.jpg"):
        # p.stem is the file name without ".jpg"; slice off "<prefix>_" to get the number part.
        tail = p.stem[len(prefix) + 1:]
        if tail.isdigit():
            nums.append(int(tail))
    # max(..., default=0) handles the empty-folder case where there are no numbers yet.
    return max(nums, default=0) + 1


def stdin_commands(q: "queue.Queue[str]") -> None:
    """Headless keypresses: each ENTER is a capture, 'q' quits.

    Runs in a background thread. It blocks waiting for lines typed in the terminal and pushes
    a "capture" or "quit" command onto the queue, which the main loop checks each frame.
    """
    # `for line in sys.stdin` waits for the user to press ENTER and gives us what they typed.
    for line in sys.stdin:
        q.put("quit" if line.strip().lower() in ("q", "quit", "exit") else "capture")
    # If the terminal input closes (Ctrl-D / end of input), treat that as quit too.
    q.put("quit")


def main(argv: list[str] | None = None) -> int:
    """Run the capture tool. ``argv`` is the list of command-line words (None = use the real ones).

    Returns an exit code (0 = success) which ``sys.exit`` hands to the operating system.
    """
    # --- 1. Describe the command-line options --------------------------------------------
    # The module docstring at the top of the file doubles as the help text: its first paragraph
    # is the short description, the rest is printed after the option list.
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--session", default=None, help="session name (default: timestamp YYYYmmdd_HHMMSS)")
    ap.add_argument("--out-root", default="data/raw", help="parent folder (default: %(default)s)")
    ap.add_argument("--backend", choices=("auto", "opencv", "picamera2"), default="auto")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: %(default)s)")
    ap.add_argument("--resolution", type=parse_resolution, default=(1280, 720), metavar="WxH",
                    help="capture resolution (default: 1280x720)")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="also save every N seconds (0 = keypress only)")
    ap.add_argument("--max", type=int, default=0, help="stop after N images (0 = no limit)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default: %(default)s)")
    ap.add_argument("--headless", action="store_true", help="no preview window (Pi over SSH)")
    # parse_args reads the options; each becomes an attribute, e.g. args.session, args.interval.
    args = ap.parse_args(argv)

    import cv2  # lazy so --help works without OpenCV

    # --- 2. Decide where photos go and open the camera --------------------------------------
    # `a or b` picks `b` when `a` is empty/None: no --session given -> use the current timestamp.
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root)
    # Relative paths are taken from the repo root, not from wherever the terminal happens to be.
    folder = (out_root if out_root.is_absolute() else REPO_ROOT / out_root) / session
    # Create the folder (and any missing parents); do not complain if it already exists.
    folder.mkdir(parents=True, exist_ok=True)
    cam, backend = open_camera(args.backend, args.camera, args.resolution)
    n = next_index(folder, session)   # number for the next saved photo
    saved = 0                         # how many photos we saved in this run
    print(f"Camera: {backend} {args.resolution[0]}x{args.resolution[1]} -> {folder}")
    print("ENTER = save, q + ENTER = quit" if args.headless else "SPACE = save, Q/ESC = quit")

    # --- 3. Set up keyboard handling ----------------------------------------------------
    # In headless mode a background thread listens to the terminal and drops commands here.
    cmds: "queue.Queue[str]" = queue.Queue()
    if args.headless:
        # daemon=True means the thread will not keep the program alive when main() finishes.
        threading.Thread(target=stdin_commands, args=(cmds,), daemon=True).start()
    show = not args.headless          # whether to display a preview window
    last_auto = time.monotonic()      # when the last timed (--interval) photo was taken
    # try/finally: whatever happens inside (including Ctrl-C), the `finally` block runs so the
    # camera is always released.
    try:
        # --- 4. Main loop: grab a frame, show it, save it if asked ----------------------------
        while True:
            frame = cam.read()
            if frame is None:
                # Camera had nothing yet; wait a little and try again.
                time.sleep(0.05)
                continue
            want = False   # becomes True if anything this iteration asks to save the frame
            # Timed capture: --interval seconds have passed since the last auto-save.
            if args.interval > 0 and time.monotonic() - last_auto >= args.interval:
                want, last_auto = True, time.monotonic()
            if show:
                try:
                    # Draw the session name and the count on a COPY of the frame so the text
                    # does not end up in the saved photo.
                    preview = frame.copy()
                    cv2.putText(preview, f"{session}  saved {saved}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("capture", preview)
                    # waitKey(1) refreshes the window and returns the key pressed (or -1).
                    # `& 0xFF` keeps only the low 8 bits, the usual trick for a plain key code.
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    # No display (e.g. SSH without X). Switch to headless mode on the fly.
                    print("No display available; continuing headless (ENTER = save).")
                    show = False
                    threading.Thread(target=stdin_commands, args=(cmds,), daemon=True).start()
                    key = 255
                # ord("q") is the numeric code of the letter q; 27 is the ESC key.
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    want = True
            # Drain any commands the keyboard thread has queued up (headless mode).
            try:
                while True:
                    # get_nowait raises queue.Empty as soon as there is nothing left.
                    c = cmds.get_nowait()
                    if c == "quit":
                        # Raising KeyboardInterrupt reuses the same exit path as Ctrl-C below.
                        raise KeyboardInterrupt
                    want = True
            except queue.Empty:
                pass
            if want:
                # `{n:04d}` formats the number with leading zeros to 4 digits: 1 -> 0001.
                path = folder / f"{session}_{n:04d}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                print(f"saved {path.name}")
                n += 1
                saved += 1
                # --max N: stop after N photos (0 means unlimited).
                if args.max and saved >= args.max:
                    break
            if not show:
                # Without a window there is no waitKey pause, so sleep briefly to avoid
                # spinning the CPU at 100%.
                time.sleep(0.01)
    except KeyboardInterrupt:
        # Ctrl-C (or a "quit" command) lands here; nothing to do, just fall through to cleanup.
        pass
    finally:
        cam.close()
        if show:
            cv2.destroyAllWindows()
    print(f"{saved} images saved to {folder}")
    return 0


# This is the standard Python way of saying "only run main() when this file is executed as a
# script, not when another file imports it".
if __name__ == "__main__":
    sys.exit(main())
