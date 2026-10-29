"""Camera backends. capture() always returns a fresh BGR frame (PRD 7.1: never send a queued one).

WHAT THIS FILE DOES
-------------------
Gives the rest of the device ONE simple way to grab a picture: ``camera.capture()`` returns the
newest image the camera has. Three interchangeable "backends" implement that:

- opencv: the Pi's USB webcam (default) and the laptop webcam in sim mode. A background
  thread keeps only the latest frame.
- picamera2: an optional CSI Camera Module (install python3-picamera2 via apt).
- file: a video file, an image, or a folder of images, looped. For bench and headless tests.

It also provides a JPEG encoder (turning an image into a small compressed byte string so it can
be sent over the network). JPEG encoding prefers simplejpeg (fast and small), then OpenCV, then
Pillow, so the Pi never needs an ML runtime.

HOW IT FITS IN
--------------
``slr/device/main.py`` calls ``open_camera`` once at start-up. ``slr/device/frame_sender.py``
then calls ``capture()`` and the JPEG encoder every time it is about to send a frame to the
laptop server. "Fresh frame" matters: if we sent a picture taken half a second ago, the
server's directions ("slightly left") would describe where the item WAS, not where it is.

WORDS YOU WILL SEE
------------------
- frame: one still image from the camera.
- BGR: the order of the colour channels in memory (blue, green, red). OpenCV's convention; the
  server expects it.
- numpy array (np.ndarray): a grid of numbers. An image is a height x width x 3 array of pixels.
- backend: see above, a swappable implementation behind a common interface.
- thread: a second line of execution running alongside the main program.
- Condition: a thread tool that lets one thread sleep until another says "something changed".
- JPEG: a compressed image format. "quality" 0..100 trades file size against sharpness.
- ImportError: the error Python raises when a library is not installed.
- ``raise ... from e``: raise our own clearer error while keeping the original one attached for
  debugging.
"""

# Allows modern type hints like ``np.ndarray | None`` on older Python versions.
from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

from slr.common.config import CameraConfig

log = logging.getLogger(__name__)

# File extensions the "file" backend treats as still images (a set, for fast "in" checks).
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class CameraError(RuntimeError):
    """Raised for anything camera related that goes wrong (not installed, not found, no frames).

    Having our own error type lets main.py catch camera problems specifically and keep the
    buttons and audio working (FR-55 in the product spec, PRD.pdf).
    """
    pass


class Picamera2Camera:
    """The real Pi camera, via the ``picamera2`` library that ships with Raspberry Pi OS."""

    def __init__(self, width: int, height: int):
        """Open and start the camera at the requested picture size (pixels)."""
        # Imported here so laptops without picamera2 can still import this module.
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise CameraError("picamera2 is not installed (sudo apt install python3-picamera2)") from e
        try:
            self.cam = Picamera2()
            # buffer_count=2 keeps the camera's internal queue tiny so frames are always recent.
            config = self.cam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}, buffer_count=2)
            self.cam.configure(config)
            self.cam.start()
        except Exception as e:
            raise CameraError(f"camera failed to start: {e}") from e

    def capture(self) -> np.ndarray:
        """Return the newest frame as a height x width x 3 BGR array."""
        try:
            # "RGB888" in picamera2 is BGR byte order in memory, which is what OpenCV expects
            return self.cam.capture_array("main")
        except Exception as e:
            raise CameraError(str(e)) from e

    def close(self) -> None:
        """Stop and release the camera. Errors are ignored: we are shutting down anyway."""
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass


class OpenCVCamera:
    """A USB UVC webcam on the Pi, or the laptop webcam in sim mode, through OpenCV.

    Webcams buffer several frames internally, so a naive ``read()`` can return an old picture.
    To avoid that, a background thread reads frames as fast as the webcam produces them and
    keeps only the latest; ``capture()`` then waits for a frame NEWER than the last one handed out.
    """

    def __init__(self, index: int, width: int, height: int):
        """``index`` is the webcam number (0 = /dev/video0); width/height the wanted size."""
        import cv2

        self.cv2 = cv2
        # Prefer the Linux V4L2 backend: it talks to /dev/videoN directly and skips OpenCV's
        # GStreamer/FFmpeg probes, which can be slow or pick the wrong device on a Pi.
        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY) if sys.platform.startswith("linux") else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened() and backend != cv2.CAP_ANY:
            self.cap.release()
            self.cap = cv2.VideoCapture(index)   # fall back to OpenCV's auto-detection
        if not self.cap.isOpened():
            raise CameraError(
                f"could not open USB camera {index} "
                f"(check /dev/video{index}; run `v4l2-ctl --list-devices`)")
        # Ask the driver to keep only a frame or two; the reader thread discards stale frames.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.size = (width, height)
        self._latest: np.ndarray | None = None   # the newest frame from the reader thread
        self._seq = 0                            # counts frames received, to detect "newer"
        self._cond = threading.Condition()       # lets capture() sleep until a new frame arrives
        self._running = True
        # daemon=True means this thread will not keep the program alive when the main part exits.
        self._thread = threading.Thread(target=self._reader, name="webcam", daemon=True)
        self._thread.start()
        # Make sure the webcam actually delivers pictures before declaring success.
        if self._wait_frame(after=0, timeout=5.0) is None:
            self.close()
            raise CameraError(
                f"USB camera {index} opened but gave no frames "
                f"(check /dev/video{index} with `v4l2-ctl --list-devices`)")

    def _reader(self) -> None:
        """Background loop: keep reading frames, store the latest, wake anyone waiting."""
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            # Some webcams ignore the requested size; resize so the rest of the code can rely on it.
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = self.cv2.resize(frame, self.size)
            with self._cond:
                self._latest = frame
                self._seq += 1
                self._cond.notify_all()

    def _wait_frame(self, after: int, timeout: float) -> np.ndarray | None:
        """Wait up to ``timeout`` seconds for a frame with sequence number > ``after``.

        Returns a COPY of it (so the reader thread can overwrite its own buffer safely), or
        None if nothing arrived in time. ``lambda:`` defines a tiny one-line function in place.
        """
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after, timeout):
                return None
            return self._latest.copy()

    def capture(self) -> np.ndarray:
        """Return a frame that is newer than the previous capture (never a stale one)."""
        # wait for a frame newer than the one we already have, so it is never stale
        frame = self._wait_frame(after=self._seq, timeout=2.0)
        if frame is None:
            raise CameraError("USB camera stopped delivering frames (check its USB cable)")
        return frame

    def close(self) -> None:
        """Stop the reader thread and release the webcam."""
        self._running = False
        self.cap.release()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


class FileCamera:
    """Loops over a video, an image, or a folder of images.

    Not a real camera: it replays recorded material so the pipeline can be tested or benchmarked
    on a machine with no camera at all (``slr sim --file ...``, ``slr bench``, ``slr replay``).
    """

    def __init__(self, path: str, width: int, height: int, resize: bool = True):
        """
        - path: a video file, a single image, or a folder containing images.
        - width/height: size to resize frames to (if ``resize`` is True).
        """
        import cv2

        self.cv2 = cv2
        self.path = Path(path)
        self.size = (width, height)
        self.resize = resize
        self.images: list[Path] = []   # used for image / folder sources
        self.cap = None                # used for video sources
        self.i = 0                     # which image comes next
        if self.path.is_dir():
            # All image files in the folder, in alphabetical order.
            self.images = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self.images:
                raise CameraError(f"no images in {self.path}")
        elif self.path.suffix.lower() in IMAGE_EXTS:
            self.images = [self.path]
        else:
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise CameraError(f"cannot open video {self.path}")
        self._cache: dict[Path, np.ndarray] = {}   # decoded images so we read each file once

    def capture(self) -> np.ndarray:
        """Return the next frame; wraps around to the start when the material runs out."""
        cv2 = self.cv2
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                # End of the video: rewind to frame 0 and read again.
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    raise CameraError("video has no frames")
        else:
            # ``%`` (modulo) wraps the counter around so we loop through the images forever.
            p = self.images[self.i % len(self.images)]
            self.i += 1
            if p not in self._cache:
                img = cv2.imread(str(p))
                if img is None:
                    raise CameraError(f"cannot read {p}")
                self._cache[p] = img
            frame = self._cache[p].copy()
        if self.resize and (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        return frame

    def close(self) -> None:
        """Release the video file, if one was open."""
        if self.cap is not None:
            self.cap.release()


def open_camera(cfg: CameraConfig, root: Path | None = None):
    """Build the camera backend named in config/device.yaml (``camera.backend``).

    ``root`` is the project folder, used to resolve a relative ``camera.file`` path.
    Returns one of the three camera classes above; raises CameraError if it cannot.
    """
    if cfg.backend == "picamera2":
        return Picamera2Camera(cfg.width, cfg.height)
    if cfg.backend == "opencv":
        return OpenCVCamera(cfg.index, cfg.width, cfg.height)
    if cfg.backend == "file":
        if not cfg.file:
            raise CameraError("camera.backend is file but camera.file is not set")
        path = Path(cfg.file)
        if root is not None and not path.is_absolute():
            path = root / path
        return FileCamera(str(path), cfg.width, cfg.height)
    raise CameraError(f"unknown camera backend {cfg.backend}")


# --------------------------------------------------------------------------- JPEG encoding


def make_jpeg_encoder():
    """Returns encode(bgr_image, quality) -> bytes using the lightest library available.

    Tries three libraries in order of preference. Each branch defines a small ``enc`` function
    and returns it; the caller just calls ``enc(img, quality)`` without caring which library is
    underneath. Functions in Python are values and can be returned like any other.
    """
    try:
        import simplejpeg

        def enc(img: np.ndarray, q: int) -> bytes:
            # ascontiguousarray makes sure the pixels are laid out in one unbroken block of memory.
            return simplejpeg.encode_jpeg(np.ascontiguousarray(img), quality=q, colorspace="BGR")

        return enc
    except ImportError:
        pass
    try:
        import cv2

        def enc(img: np.ndarray, q: int) -> bytes:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
            if not ok:
                raise RuntimeError("JPEG encode failed")
            return buf.tobytes()

        return enc
    except ImportError:
        pass
    # Last resort: Pillow (PIL). It wants RGB, so ``[:, :, ::-1]`` reverses the colour channels.
    import io

    from PIL import Image

    def enc(img: np.ndarray, q: int) -> bytes:
        bio = io.BytesIO()
        Image.fromarray(img[:, :, ::-1]).save(bio, format="JPEG", quality=int(q))
        return bio.getvalue()

    return enc
