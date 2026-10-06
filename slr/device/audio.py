"""Device audio: clip playback with priority / interruption, software volume, panned tones.

``CueScheduler`` is the pure part (FR-63): a higher-priority cue interrupts a lower one, cues
with ``interrupt`` also cut off an equal-priority cue, and guidance cues never queue (a new one
replaces any waiting one, and stale ones are dropped).

Backends: ``sounddevice`` (one stereo stream; speech and tones are mixed in the callback, works
on the Pi through ALSA and on laptops), ``aplay`` (clips only) and ``null`` (no sound; for tests
and headless runs, clips still "finish" after their real duration).
"""

from __future__ import annotations

import logging
import math
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from slr.common.config import AudioConfig
from slr.common.protocol import PRIORITIES

log = logging.getLogger(__name__)

GUIDANCE = PRIORITIES["guidance"]
SYSTEM = PRIORITIES["system"]


@dataclass
class QueuedCue:
    clip: str                    # clip name (server cache) or system clip key
    priority: int
    interrupt: bool = False
    cue_id: int | None = None    # None for local system clips
    path: Path | None = None
    queued_at: float = 0.0


@dataclass
class CueScheduler:
    guidance_max_wait_s: float = 2.0
    current: QueuedCue | None = None
    queue: list[QueuedCue] = field(default_factory=list)

    def submit(self, cue: QueuedCue, now: float) -> str:
        """Returns 'play' (start now, interrupting whatever plays) or 'queued'."""
        cue.queued_at = now
        cur = self.current
        if cur is None:
            self.current = cue
            return "play"
        if cue.priority > cur.priority or (cue.interrupt and cue.priority >= cur.priority):
            self.current = cue
            # stale guidance and lower-priority chatter are dropped along with the interrupted cue
            self.queue = [q for q in self.queue if q.priority >= cue.priority and q.priority != GUIDANCE]
            return "play"
        if cue.priority == GUIDANCE:
            self.queue = [q for q in self.queue if q.priority != GUIDANCE]
        # insert after every queued cue of equal or higher priority (FIFO within a priority)
        idx = len(self.queue)
        for i, q in enumerate(self.queue):
            if q.priority < cue.priority:
                idx = i
                break
        self.queue.insert(idx, cue)
        return "queued"

    def finished(self, now: float) -> QueuedCue | None:
        """The current cue ended. Returns the next cue to play, if any."""
        self.current = None
        while self.queue:
            nxt = self.queue.pop(0)
            if nxt.priority == GUIDANCE and now - nxt.queued_at > self.guidance_max_wait_s:
                continue  # old directions are worse than none
            self.current = nxt
            return nxt
        return None

    def clear(self) -> None:
        self.current = None
        self.queue.clear()


# --------------------------------------------------------------------------- WAV loading


def load_wav(path: Path, target_rate: int) -> np.ndarray:
    """Load a 16-bit PCM WAV as float32 stereo at target_rate."""
    with wave.open(str(path), "rb") as wf:
        rate, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit WAV is supported")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    data = data.reshape(-1, ch)
    mono = data.mean(axis=1)
    if rate != target_rate and len(mono) > 1:
        n = int(round(len(mono) * target_rate / rate))
        mono = np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono).astype(np.float32)
    return np.repeat(mono[:, None], 2, axis=1)


def pan_gains(pan: float) -> tuple[float, float]:
    """Constant-power pan: -1 = full left, 0 = center, 1 = full right."""
    a = (max(-1.0, min(1.0, pan)) + 1) * math.pi / 4
    return math.cos(a), math.sin(a)


@dataclass
class ToneState:
    freq_hz: float = 880.0
    duration_ms: float = 60.0
    pan: float = 0.0
    interval_ms: float | None = None


def render_tone(state: ToneState, start: int, frames: int, rate: int, amp: float = 0.25) -> np.ndarray:
    """Beeps of duration_ms every interval_ms, panned, with 5 ms fades to avoid clicks."""
    out = np.zeros((frames, 2), np.float32)
    if state.interval_ms is None:
        return out
    n = np.arange(start, start + frames)
    period = max(1, int(rate * state.interval_ms / 1000))
    beep = max(1, int(rate * state.duration_ms / 1000))
    pos = n % period
    on = pos < beep
    if not on.any():
        return out
    fade = max(1, int(rate * 0.005))
    env = np.minimum(1.0, np.minimum(pos + 1, beep - pos) / fade) * on
    wave_ = amp * np.sin(2 * np.pi * state.freq_hz * n / rate) * env
    left, right = pan_gains(state.pan)
    out[:, 0] = wave_ * left
    out[:, 1] = wave_ * right
    return out


# --------------------------------------------------------------------------- backends


class SoundDeviceBackend:
    def __init__(self, cfg: AudioConfig, on_done: Callable[[int], None]):
        import sounddevice as sd

        self.rate = cfg.sample_rate
        self.on_done = on_done
        self.gain = 1.0
        self._lock = threading.Lock()
        self._clip: np.ndarray | None = None
        self._token: int = 0
        self._pos = 0
        self._tone = ToneState()
        self._tone_t = 0
        self.stream = sd.OutputStream(samplerate=self.rate, channels=2, dtype="float32",
                                      blocksize=cfg.blocksize, device=cfg.device, callback=self._callback)
        self.stream.start()

    def _callback(self, outdata, frames, time_info, status) -> None:
        done = False
        with self._lock:
            out = render_tone(self._tone, self._tone_t, frames, self.rate)
            self._tone_t += frames
            if self._clip is not None:
                out *= 0.5  # duck the tone under speech
                chunk = self._clip[self._pos:self._pos + frames]
                out[:len(chunk)] += chunk
                self._pos += frames
                if self._pos >= len(self._clip):
                    self._clip = None
                    done = self._token
            outdata[:] = np.clip(out * self.gain, -1, 1)
        if done is not False:
            threading.Thread(target=self.on_done, args=(done,), daemon=True).start()

    def play(self, samples: np.ndarray, token: int = 0) -> None:
        with self._lock:
            self._clip, self._pos, self._token = samples, 0, token

    def stop_clip(self) -> None:
        with self._lock:
            self._clip = None

    def set_tone(self, tone: ToneState) -> None:
        with self._lock:
            if self._tone.interval_ms is None and tone.interval_ms is not None:
                self._tone_t = 0  # start with a beep right away
            self._tone = tone

    def set_gain(self, gain: float) -> None:
        self.gain = gain

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()


class AplayBackend:
    """ALSA command-line playback. Clips only; tones need the sounddevice backend."""

    def __init__(self, cfg: AudioConfig, on_done: Callable[[int], None]):
        self.rate = cfg.sample_rate
        self.device = cfg.device
        self.on_done = on_done
        self.gain = 1.0
        self._proc: subprocess.Popen | None = None
        self._warned = False
        self._lock = threading.Lock()

    def play(self, samples: np.ndarray, token: int = 0) -> None:
        self.stop_clip()
        pcm = (np.clip(samples * self.gain, -1, 1) * 32767).astype("<i2").tobytes()
        cmd = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "2", "-r", str(self.rate)]
        if self.device is not None:
            cmd += ["-D", str(self.device)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with self._lock:
            self._proc = proc

        def feed():
            try:
                proc.stdin.write(pcm)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            proc.wait()
            with self._lock:
                mine = self._proc is proc
                if mine:
                    self._proc = None
            if mine and proc.returncode == 0:
                self.on_done(token)

        threading.Thread(target=feed, daemon=True).start()

    def stop_clip(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.terminate()

    def set_tone(self, tone: ToneState) -> None:
        if tone.interval_ms is not None and not self._warned:
            log.warning("aplay backend cannot play guidance tones; use audio.backend: sounddevice")
            self._warned = True

    def set_gain(self, gain: float) -> None:
        self.gain = gain

    def close(self) -> None:
        self.stop_clip()


class NullBackend:
    """No sound. Clips 'finish' after their real duration so timing still behaves."""

    def __init__(self, cfg: AudioConfig, on_done: Callable[[int], None]):
        self.rate = cfg.sample_rate
        self.on_done = on_done
        self.gain = 1.0
        self.tone = ToneState()
        self.played: list[int] = []
        self._timer: threading.Timer | None = None

    def play(self, samples: np.ndarray, token: int = 0) -> None:
        self.stop_clip()
        self.played.append(len(samples))
        self._timer = threading.Timer(len(samples) / self.rate, self.on_done, args=(token,))
        self._timer.daemon = True
        self._timer.start()

    def stop_clip(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def set_tone(self, tone: ToneState) -> None:
        self.tone = tone

    def set_gain(self, gain: float) -> None:
        self.gain = gain

    def close(self) -> None:
        self.stop_clip()


def make_backend(cfg: AudioConfig, on_done: Callable[[int], None]):
    if cfg.backend == "sounddevice":
        try:
            return SoundDeviceBackend(cfg, on_done)
        except Exception as e:
            log.error("sounddevice audio failed (%s); falling back to no audio", e)
            return NullBackend(cfg, on_done)
    if cfg.backend == "aplay":
        return AplayBackend(cfg, on_done)
    return NullBackend(cfg, on_done)


# --------------------------------------------------------------------------- player


class AudioPlayer:
    """Thread-safe front end: scheduler + backend + clip cache + volume."""

    def __init__(self, cfg: AudioConfig, on_started: Callable[[QueuedCue], None] | None = None,
                 backend=None):
        self.cfg = cfg
        self.on_started = on_started
        self.sched = CueScheduler()
        self._lock = threading.RLock()
        self.backend = backend or make_backend(cfg, self._on_done)
        if backend is not None:
            backend.on_done = self._on_done
        self._cache: dict[Path, np.ndarray] = {}
        self._token = 0
        self.history: list[str] = []

    def _samples(self, path: Path) -> np.ndarray:
        s = self._cache.get(path)
        if s is None:
            s = load_wav(path, self.backend.rate)
            if len(self._cache) > 300:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = s
        return s

    def play(self, cue: QueuedCue) -> None:
        if cue.path is None or not cue.path.exists():
            log.warning("missing clip %s", cue.clip)
            return
        with self._lock:
            if self.sched.submit(cue, time.monotonic()) == "play":
                self._start(cue)

    def _start(self, cue: QueuedCue) -> None:
        try:
            samples = self._samples(cue.path)
        except Exception as e:
            log.error("cannot load %s: %s", cue.path, e)
            nxt = self.sched.finished(time.monotonic())
            if nxt:
                self._start(nxt)
            return
        self._token += 1
        self.backend.play(samples, self._token)
        self.history.append(cue.clip)
        if self.on_started:
            self.on_started(cue)

    def _on_done(self, token: int = 0) -> None:
        with self._lock:
            if token != self._token:
                return  # a clip that was already interrupted
            nxt = self.sched.finished(time.monotonic())
            if nxt:
                self._start(nxt)

    def set_tone(self, freq_hz: float, duration_ms: float, pan: float, interval_ms: float | None) -> None:
        self.backend.set_tone(ToneState(freq_hz, duration_ms, pan, interval_ms))

    def stop_tone(self) -> None:
        self.backend.set_tone(ToneState())

    def set_gain(self, gain: float) -> None:
        self.backend.set_gain(gain)

    def idle(self) -> bool:
        with self._lock:
            return self.sched.current is None

    def wait_idle(self, timeout: float) -> None:
        end = time.monotonic() + timeout
        while not self.idle() and time.monotonic() < end:
            time.sleep(0.05)

    def close(self) -> None:
        self.backend.close()
