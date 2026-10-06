"""Camera backends. capture() always returns a fresh BGR frame (PRD 7.1: never send a queued one).

- picamera2: Pi Camera Module 3 on the Pi Zero 2 W (install python3-picamera2 via apt).
- opencv: laptop webcam for sim mode. A background thread keeps only the latest frame.
- file: a video file, an image, or a folder of images, looped. For bench and headless tests.

JPEG encoding prefers simplejpeg (fast, small, ships with picamera2), then OpenCV, then Pillow,
so the Pi never needs an ML runtime.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from slr.common.config import CameraConfig

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class CameraError(RuntimeError):
    pass


class Picamera2Camera:
    def __init__(self, width: int, height: int):
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise CameraError("picamera2 is not installed (sudo apt install python3-picamera2)") from e
        try:
            self.cam = Picamera2()
            config = self.cam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}, buffer_count=2)
            self.cam.configure(config)
            self.cam.start()
        except Exception as e:
            raise CameraError(f"camera failed to start: {e}") from e

    def capture(self) -> np.ndarray:
        try:
            # "RGB888" in picamera2 is BGR byte order in memory, which is what OpenCV expects
            return self.cam.capture_array("main")
        except Exception as e:
            raise CameraError(str(e)) from e

    def close(self) -> None:
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass


class OpenCVCamera:
    def __init__(self, index: int, width: int, height: int):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise CameraError(f"could not open webcam {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.size = (width, height)
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._reader, name="webcam", daemon=True)
        self._thread.start()
        if self._wait_frame(after=0, timeout=5.0) is None:
            self.close()
            raise CameraError(f"webcam {index} opened but gave no frames")

    def _reader(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = self.cv2.resize(frame, self.size)
            with self._cond:
                self._latest = frame
                self._seq += 1
                self._cond.notify_all()

    def _wait_frame(self, after: int, timeout: float) -> np.ndarray | None:
        with self._cond:
            self._cond.wait_for(lambda: self._seq > after, timeout)
            return None if self._latest is None else self._latest.copy()

    def capture(self) -> np.ndarray:
        # wait for a frame newer than the one we already have, so it is never stale
        frame = self._wait_frame(after=self._seq, timeout=1.0)
        if frame is None:
            raise CameraError("webcam stopped delivering frames")
        return frame

    def close(self) -> None:
        self._running = False
        self.cap.release()


class FileCamera:
    """Loops over a video, an image, or a folder of images."""

    def __init__(self, path: str, width: int, height: int, resize: bool = True):
        import cv2

        self.cv2 = cv2
        self.path = Path(path)
        self.size = (width, height)
        self.resize = resize
        self.images: list[Path] = []
        self.cap = None
        self.i = 0
        if self.path.is_dir():
            self.images = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if not self.images:
                raise CameraError(f"no images in {self.path}")
        elif self.path.suffix.lower() in IMAGE_EXTS:
            self.images = [self.path]
        else:
            self.cap = cv2.VideoCapture(str(self.path))
            if not self.cap.isOpened():
                raise CameraError(f"cannot open video {self.path}")
        self._cache: dict[Path, np.ndarray] = {}

    def capture(self) -> np.ndarray:
        cv2 = self.cv2
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
                if not ok:
                    raise CameraError("video has no frames")
        else:
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
        if self.cap is not None:
            self.cap.release()


def open_camera(cfg: CameraConfig, root: Path | None = None):
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
    """Returns encode(bgr_image, quality) -> bytes using the lightest library available."""
    try:
        import simplejpeg

        def enc(img: np.ndarray, q: int) -> bytes:
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
    import io

    from PIL import Image

    def enc(img: np.ndarray, q: int) -> bytes:
        bio = io.BytesIO()
        Image.fromarray(img[:, :, ::-1]).save(bio, format="JPEG", quality=int(q))
        return bio.getvalue()

    return enc
