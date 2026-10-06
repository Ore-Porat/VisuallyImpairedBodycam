"""Capture Stage B training photos from the laptop webcam or the Pi camera (TR-11, TR-12).

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
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_resolution(s: str) -> tuple[int, int]:
    """'1280x720' -> (1280, 720)."""
    try:
        w, h = (int(x) for x in s.lower().replace("*", "x").split("x"))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"resolution must look like 1280x720, got {s!r}") from e
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return w, h


class OpenCVCamera:
    """Webcam via OpenCV. ``read()`` returns a BGR frame or None."""

    def __init__(self, index: int, resolution: tuple[int, int]):
        import cv2
        self.cv2 = cv2
        api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, api)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


class PiCamera:
    """Raspberry Pi camera via picamera2 (installed from apt on the Pi)."""

    def __init__(self, resolution: tuple[int, int]):
        from picamera2 import Picamera2  # optional dependency
        self.cam = Picamera2()
        cfg = self.cam.create_still_configuration(main={"size": resolution, "format": "RGB888"})
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(1.0)  # let auto-exposure settle

    def read(self):
        return self.cam.capture_array()  # RGB888 in picamera2 is BGR byte order, as OpenCV expects

    def close(self):
        self.cam.stop()


def open_camera(backend: str, index: int, resolution: tuple[int, int]):
    if backend in ("auto", "picamera2"):
        try:
            return PiCamera(resolution), "picamera2"
        except ImportError:
            if backend == "picamera2":
                raise SystemExit("picamera2 is not installed (sudo apt install python3-picamera2)")
    return OpenCVCamera(index, resolution), "opencv"


def next_index(folder: Path, prefix: str) -> int:
    """First unused image number in ``folder`` (so re-running a session appends)."""
    nums = []
    for p in folder.glob(f"{prefix}_*.jpg"):
        tail = p.stem[len(prefix) + 1:]
        if tail.isdigit():
            nums.append(int(tail))
    return max(nums, default=0) + 1


def stdin_commands(q: "queue.Queue[str]") -> None:
    """Headless keypresses: each ENTER is a capture, 'q' quits."""
    for line in sys.stdin:
        q.put("quit" if line.strip().lower() in ("q", "quit", "exit") else "capture")
    q.put("quit")


def main(argv: list[str] | None = None) -> int:
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
    args = ap.parse_args(argv)

    import cv2  # lazy so --help works without OpenCV

    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root)
    folder = (out_root if out_root.is_absolute() else REPO_ROOT / out_root) / session
    folder.mkdir(parents=True, exist_ok=True)
    cam, backend = open_camera(args.backend, args.camera, args.resolution)
    n = next_index(folder, session)
    saved = 0
    print(f"Camera: {backend} {args.resolution[0]}x{args.resolution[1]} -> {folder}")
    print("ENTER = save, q + ENTER = quit" if args.headless else "SPACE = save, Q/ESC = quit")

    cmds: "queue.Queue[str]" = queue.Queue()
    if args.headless:
        threading.Thread(target=stdin_commands, args=(cmds,), daemon=True).start()
    show = not args.headless
    last_auto = time.monotonic()
    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
            want = False
            if args.interval > 0 and time.monotonic() - last_auto >= args.interval:
                want, last_auto = True, time.monotonic()
            if show:
                try:
                    preview = frame.copy()
                    cv2.putText(preview, f"{session}  saved {saved}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("capture", preview)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    print("No display available; continuing headless (ENTER = save).")
                    show = False
                    threading.Thread(target=stdin_commands, args=(cmds,), daemon=True).start()
                    key = 255
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    want = True
            try:
                while True:
                    c = cmds.get_nowait()
                    if c == "quit":
                        raise KeyboardInterrupt
                    want = True
            except queue.Empty:
                pass
            if want:
                path = folder / f"{session}_{n:04d}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                print(f"saved {path.name}")
                n += 1
                saved += 1
                if args.max and saved >= args.max:
                    break
            if not show:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if show:
            cv2.destroyAllWindows()
    print(f"{saved} images saved to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
