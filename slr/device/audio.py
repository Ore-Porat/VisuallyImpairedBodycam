"""Device audio: clip playback with priority / interruption, software volume, panned tones.

WHAT THIS FILE DOES
-------------------
This is the "speaker" side of the device. Everything the user hears in the headphones goes
through here:

- Speech clips ("Pasta found. Slightly left.") are short WAV files. The server decides WHAT to
  say and sends a "cue" message naming a clip; this file decides WHEN to play it, whether it
  should interrupt what is already playing, and how loud it should be.
- Guidance tones (the beeps that pan left/right while searching) are generated here from a
  frequency, duration and repeat interval sent by the server.

HOW IT FITS IN
--------------
The device (Raspberry Pi in the camera body, or a laptop in sim mode) only captures video, reads
buttons and plays audio. All the thinking happens on the laptop server (see README.md).
``slr/device/main.py`` creates one ``AudioPlayer`` and hands it cues as they arrive.

``CueScheduler`` is the pure part (FR-63; "FR-63" is a requirement number from the product spec,
PRD.pdf): a higher-priority cue interrupts a lower one, cues with ``interrupt`` also cut off an
equal-priority cue, and guidance cues never queue (a new one replaces any waiting one, and stale
ones are dropped). "Pure" means it does no I/O at all: feed it cues and times, it answers with
decisions. That makes it easy to unit test.

Backends: ``sounddevice`` (one stereo stream; speech and tones are mixed in the callback, works
on the Pi through ALSA and on laptops), ``aplay`` (clips only) and ``null`` (no sound; for tests
and headless runs, clips still "finish" after their real duration).

WORDS YOU WILL SEE
------------------
- WAV / PCM: an uncompressed audio file. "16-bit PCM" means each sample is a whole number from
  -32768 to 32767 describing the speaker position at that instant.
- sample rate: how many samples per second (22050 here). Higher = better quality, more data.
- stereo / mono: two channels (left, right) vs one.
- gain: a multiplier applied to the audio. 1.0 = unchanged, 0.5 = quieter, 0.0 = silence.
- pan: where a sound sits between the left (-1) and right (+1) ear.
- backend: an interchangeable "driver" that actually makes sound. Same interface, different
  underlying library, so the rest of the code does not care which is in use.
- callback: a function you hand to a library so it can call YOU back later (here: the sound
  card asks "give me the next 512 samples" over and over).
- thread: a second line of execution running at the same time as the main program. Audio is
  time critical, so it runs on its own thread and must not wait for anything else.
- lock: a "talking stick" that makes sure two threads do not touch the same data at the same
  moment. ``with self._lock:`` grabs it, and releases it automatically at the end of the block.
- dataclass: a Python shortcut for a class that is mostly a bundle of named fields.
- numpy (np): a library for fast math on big arrays of numbers. Audio samples are numpy arrays.
- priority: an integer from ``slr.common.protocol.PRIORITIES``; bigger numbers win.
"""

# Lets us write type hints like ``QueuedCue | None`` on older Python versions. Type hints are
# notes about what kind of value a variable holds; Python does not enforce them, they are
# documentation for people and tools.
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

# A logger writes diagnostic messages ("warning: clip missing") to the console / log file.
log = logging.getLogger(__name__)

# Shortcuts for the two priority numbers this file cares about specially.
GUIDANCE = PRIORITIES["guidance"]
SYSTEM = PRIORITIES["system"]


# ``@dataclass`` auto-generates the boring parts of a class (the __init__ that stores each
# field, a readable repr, equality). The lines below are the fields with their default values.
@dataclass
class QueuedCue:
    """One thing to say, waiting in line or currently playing.

    Fields:
    - clip: the clip's name, e.g. "pasta_found" (server-made clip) or "starting_up" (built in).
    - priority: how important it is. Bigger beats smaller (see protocol.PRIORITIES).
    - interrupt: True means "cut off a currently playing cue of the SAME priority too".
    - cue_id: the server's number for this cue so we can report "cue started" back; None for
      clips the device plays on its own (system clips).
    - path: where the WAV file is on disk.
    - queued_at: when it was submitted (filled in by the scheduler), used to drop stale directions.
    """

    clip: str                    # clip name (server cache) or system clip key
    priority: int
    interrupt: bool = False
    cue_id: int | None = None    # None for local system clips
    path: Path | None = None
    queued_at: float = 0.0


@dataclass
class CueScheduler:
    """Decides which cue plays now and which ones wait (pure logic, no sound).

    Think of a receptionist with one microphone: ``current`` is who is speaking, ``queue`` is the
    line of people waiting, sorted by importance. ``guidance_max_wait_s`` is how long a
    direction like "slightly left" may wait before it is considered out of date and thrown away.
    """

    guidance_max_wait_s: float = 2.0
    current: QueuedCue | None = None
    # ``field(default_factory=list)`` means "each scheduler gets its own fresh empty list"
    # (a plain ``= []`` default would be shared by every instance, a classic Python trap).
    queue: list[QueuedCue] = field(default_factory=list)

    def submit(self, cue: QueuedCue, now: float) -> str:
        """Returns 'play' (start now, interrupting whatever plays) or 'queued'.

        ``cue`` is the new request, ``now`` is the current time in seconds (passed in rather
        than read from the clock so tests can control time).
        """
        cue.queued_at = now
        cur = self.current
        # Nothing is playing: just start it.
        if cur is None:
            self.current = cue
            return "play"
        # More important than the current cue, or equally important and flagged "interrupt":
        # take over the microphone right now.
        if cue.priority > cur.priority or (cue.interrupt and cue.priority >= cur.priority):
            self.current = cue
            # stale guidance and lower-priority chatter are dropped along with the interrupted cue
            # (this is a "list comprehension": build a new list keeping only items that pass the test)
            self.queue = [q for q in self.queue if q.priority >= cue.priority and q.priority != GUIDANCE]
            return "play"
        # A new direction replaces any direction already waiting; only the newest one matters.
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
        """The current cue ended. Returns the next cue to play, if any.

        Called by the audio player when the sound card reports a clip is done. Guidance cues
        that waited too long are skipped: saying "slightly left" two seconds late could send the
        user the wrong way.
        """
        self.current = None
        while self.queue:
            nxt = self.queue.pop(0)
            if nxt.priority == GUIDANCE and now - nxt.queued_at > self.guidance_max_wait_s:
                continue  # old directions are worse than none
            self.current = nxt
            return nxt
        return None

    def clear(self) -> None:
        """Forget everything: nothing playing, nothing waiting."""
        self.current = None
        self.queue.clear()


# --------------------------------------------------------------------------- WAV loading


def load_wav(path: Path, target_rate: int) -> np.ndarray:
    """Load a 16-bit PCM WAV as float32 stereo at target_rate.

    Takes a file path and the sample rate the sound card is running at. Returns a numpy array of
    shape (number_of_samples, 2): one row per sample, columns are left and right. Values are
    floats between -1.0 and 1.0, which is what the audio libraries want.
    """
    # ``with`` opens the file and guarantees it is closed afterwards, even if an error happens.
    with wave.open(str(path), "rb") as wf:
        rate, ch, width = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit WAV is supported")
    # Reinterpret the raw bytes as little-endian 16-bit integers ("<i2"), then scale to -1..1.
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    # Reshape into one row per sample with ``ch`` columns (-1 = "work out this dimension").
    data = data.reshape(-1, ch)
    # Average the channels down to mono; we re-expand to stereo at the end.
    mono = data.mean(axis=1)
    # If the file's sample rate differs from the sound card's, resample by linear interpolation
    # (stretch or squash the waveform so it plays at the right speed and pitch).
    if rate != target_rate and len(mono) > 1:
        n = int(round(len(mono) * target_rate / rate))
        mono = np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono).astype(np.float32)
    # Duplicate the mono column into left and right.
    return np.repeat(mono[:, None], 2, axis=1)


def pan_gains(pan: float) -> tuple[float, float]:
    """Constant-power pan: -1 = full left, 0 = center, 1 = full right.

    Returns (left_gain, right_gain). Using cos/sin instead of a straight line keeps the sound
    equally loud as it moves across, which is what the ear expects.
    """
    a = (max(-1.0, min(1.0, pan)) + 1) * math.pi / 4
    return math.cos(a), math.sin(a)


@dataclass
class ToneState:
    """Describes the guidance beep: pitch, how long each beep lasts, where it sits left/right,
    and how often it repeats. ``interval_ms = None`` means "no beeping at all"."""

    freq_hz: float = 880.0
    duration_ms: float = 60.0
    pan: float = 0.0
    interval_ms: float | None = None


def render_tone(state: ToneState, start: int, frames: int, rate: int, amp: float = 0.25) -> np.ndarray:
    """Beeps of duration_ms every interval_ms, panned, with 5 ms fades to avoid clicks.

    Generates the next chunk of the beep pattern as audio samples.
    - state: the beep settings.
    - start: how many samples have been generated so far (so the pattern continues seamlessly).
    - frames: how many samples to produce now.
    - rate: samples per second.
    - amp: loudness of the beep (0..1).
    Returns a (frames, 2) stereo array; all zeros (silence) when beeping is off.
    """
    out = np.zeros((frames, 2), np.float32)
    if state.interval_ms is None:
        return out
    # Sample indices for this chunk, continuing from where the last chunk ended.
    n = np.arange(start, start + frames)
    # Length of one repeat (beep + silence) and of the beep itself, in samples.
    period = max(1, int(rate * state.interval_ms / 1000))
    beep = max(1, int(rate * state.duration_ms / 1000))
    # Position inside the current repeat; the beep is "on" for the first ``beep`` samples.
    pos = n % period
    on = pos < beep
    if not on.any():
        return out
    # A short ramp up and down at each beep edge so the speaker does not click.
    fade = max(1, int(rate * 0.005))
    env = np.minimum(1.0, np.minimum(pos + 1, beep - pos) / fade) * on
    # A pure sine wave at the requested pitch, shaped by the on/off envelope.
    wave_ = amp * np.sin(2 * np.pi * state.freq_hz * n / rate) * env
    left, right = pan_gains(state.pan)
    out[:, 0] = wave_ * left
    out[:, 1] = wave_ * right
    return out


# --------------------------------------------------------------------------- backends
#
# Each backend offers the same five methods: play, stop_clip, set_tone, set_gain, close, and
# calls ``on_done(token)`` when a clip finishes. The token lets the player ignore "finished"
# reports from clips it already interrupted.


class SoundDeviceBackend:
    """Real audio through the ``sounddevice`` library (PortAudio / ALSA).

    Opens one stereo output stream. The sound card repeatedly calls ``_callback`` asking for the
    next block of samples; we fill it with the current speech clip plus the guidance tone.
    """

    def __init__(self, cfg: AudioConfig, on_done: Callable[[int], None]):
        # Imported here (not at the top) so machines without sounddevice can still use the other
        # backends.
        import sounddevice as sd

        self.rate = cfg.sample_rate
        self.on_done = on_done
        self.gain = 1.0
        self._lock = threading.Lock()
        self._clip: np.ndarray | None = None   # samples of the clip now playing, or None
        self._token: int = 0                   # id of that clip, reported via on_done
        self._pos = 0                          # how far into the clip we are
        self._tone = ToneState()
        self._tone_t = 0                       # samples of tone rendered so far
        self.stream = sd.OutputStream(samplerate=self.rate, channels=2, dtype="float32",
                                      blocksize=cfg.blocksize, device=cfg.device, callback=self._callback)
        self.stream.start()

    def _callback(self, outdata, frames, time_info, status) -> None:
        """Called by the sound card thread every few milliseconds: fill ``outdata`` with the
        next ``frames`` samples (tone mixed with speech, scaled by the volume gain)."""
        done = False
        with self._lock:
            out = render_tone(self._tone, self._tone_t, frames, self.rate)
            self._tone_t += frames
            if self._clip is not None:
                out *= 0.5  # duck the tone under speech
                chunk = self._clip[self._pos:self._pos + frames]
                out[:len(chunk)] += chunk
                self._pos += frames
                # Reached the end of the clip: remember its token so we can report it below.
                if self._pos >= len(self._clip):
                    self._clip = None
                    done = self._token
            # Apply volume and clamp to the legal -1..1 range so loud mixes do not distort badly.
            outdata[:] = np.clip(out * self.gain, -1, 1)
        # Report "finished" from a separate thread so the audio callback itself never blocks.
        if done is not False:
            threading.Thread(target=self.on_done, args=(done,), daemon=True).start()

    def play(self, samples: np.ndarray, token: int = 0) -> None:
        """Start playing ``samples`` from the beginning, replacing any clip in progress."""
        with self._lock:
            self._clip, self._pos, self._token = samples, 0, token

    def stop_clip(self) -> None:
        """Silence the current speech clip (the tone keeps going)."""
        with self._lock:
            self._clip = None

    def set_tone(self, tone: ToneState) -> None:
        """Change the beep pattern (or turn it off with interval_ms=None)."""
        with self._lock:
            if self._tone.interval_ms is None and tone.interval_ms is not None:
                self._tone_t = 0  # start with a beep right away
            self._tone = tone

    def set_gain(self, gain: float) -> None:
        """Set the volume multiplier (0 = mute, 1 = full)."""
        self.gain = gain

    def close(self) -> None:
        """Stop the sound card stream and release it."""
        self.stream.stop()
        self.stream.close()


class AplayBackend:
    """ALSA command-line playback. Clips only; tones need the sounddevice backend.

    Each clip is played by launching the ``aplay`` program as a separate process and piping the
    raw samples into it. Simple and robust on Linux, but it cannot mix a tone underneath.
    """

    def __init__(self, cfg: AudioConfig, on_done: Callable[[int], None]):
        self.rate = cfg.sample_rate
        self.device = cfg.device
        self.on_done = on_done
        self.gain = 1.0
        self._proc: subprocess.Popen | None = None   # the running aplay process, if any
        self._warned = False
        self._lock = threading.Lock()

    def play(self, samples: np.ndarray, token: int = 0) -> None:
        """Stop whatever is playing, then launch ``aplay`` and feed it the clip."""
        self.stop_clip()
        # Convert floats back to 16-bit integers (the format aplay expects) and apply the gain.
        pcm = (np.clip(samples * self.gain, -1, 1) * 32767).astype("<i2").tobytes()
        cmd = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-c", "2", "-r", str(self.rate)]
        if self.device is not None:
            cmd += ["-D", str(self.device)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        with self._lock:
            self._proc = proc

        # A small function defined inside another function: it can see ``proc``, ``pcm`` and
        # ``token`` from the enclosing scope. It runs on a background thread so ``play`` returns
        # immediately instead of waiting for the whole clip to finish.
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
            # Only report "done" if this process was not replaced/stopped and it exited cleanly.
            if mine and proc.returncode == 0:
                self.on_done(token)

        threading.Thread(target=feed, daemon=True).start()

    def stop_clip(self) -> None:
        """Kill the aplay process if one is still running."""
        with self._lock:
            proc, self._proc = self._proc, None
        if proc and proc.poll() is None:
            proc.terminate()

    def set_tone(self, tone: ToneState) -> None:
        """Tones are not supported here; warn once so the user knows to switch backends."""
        if tone.interval_ms is not None and not self._warned:
            log.warning("aplay backend cannot play guidance tones; use audio.backend: sounddevice")
            self._warned = True

    def set_gain(self, gain: float) -> None:
        """Set the volume multiplier used for the NEXT clip (aplay cannot change mid-clip)."""
        self.gain = gain

    def close(self) -> None:
        """Stop playback."""
        self.stop_clip()


class NullBackend:
    """No sound. Clips 'finish' after their real duration so timing still behaves.

    Used by the automated tests and when no audio hardware is available. A timer fires
    ``on_done`` after exactly as long as the clip would have taken to play.
    """

    def __init__(self, cfg: AudioConfig, on_done: Callable[[int], None]):
        self.rate = cfg.sample_rate
        self.on_done = on_done
        self.gain = 1.0
        self.tone = ToneState()
        self.played: list[int] = []   # lengths of every clip "played", handy for tests
        self._timer: threading.Timer | None = None

    def play(self, samples: np.ndarray, token: int = 0) -> None:
        """Pretend to play: record the clip length and schedule the 'done' callback."""
        self.stop_clip()
        self.played.append(len(samples))
        # Duration in seconds = number of samples / samples per second.
        self._timer = threading.Timer(len(samples) / self.rate, self.on_done, args=(token,))
        self._timer.daemon = True
        self._timer.start()

    def stop_clip(self) -> None:
        """Cancel the pending 'done' timer, if any."""
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def set_tone(self, tone: ToneState) -> None:
        """Just remember the tone settings (tests can inspect them)."""
        self.tone = tone

    def set_gain(self, gain: float) -> None:
        """Just remember the gain."""
        self.gain = gain

    def close(self) -> None:
        """Stop any pending timer."""
        self.stop_clip()


def make_backend(cfg: AudioConfig, on_done: Callable[[int], None]):
    """Pick the backend named in config/device.yaml (``audio.backend``).

    If the real sound card cannot be opened the device keeps running silently rather than
    crashing, because the buttons and camera are still useful.
    """
    if cfg.backend == "sounddevice":
        # try/except: attempt the risky thing; if it raises an error, run the except block.
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
    """Thread-safe front end: scheduler + backend + clip cache + volume.

    This is the one object the rest of the device talks to. Give it a ``QueuedCue`` and it
    loads the WAV (caching it in memory), asks the scheduler whether to play now or wait, and
    drives the backend. It is safe to call from any thread.
    """

    def __init__(self, cfg: AudioConfig, on_started: Callable[[QueuedCue], None] | None = None,
                 backend=None):
        """
        - cfg: the ``audio:`` section of config/device.yaml.
        - on_started: optional function called whenever a cue actually begins playing (main.py
          uses it to tell the server "cue N started" for latency measurement).
        - backend: optional pre-made backend, used by tests; normally built from cfg.
        """
        self.cfg = cfg
        self.on_started = on_started
        self.sched = CueScheduler()
        # RLock = a lock the SAME thread may grab twice without deadlocking (needed because
        # _start can be called from inside _on_done which already holds it).
        self._lock = threading.RLock()
        self.backend = backend or make_backend(cfg, self._on_done)
        if backend is not None:
            backend.on_done = self._on_done
        self._cache: dict[Path, np.ndarray] = {}   # decoded WAVs, keyed by file path
        self._token = 0                             # increments per clip started
        self.history: list[str] = []                # names of every clip played (tests, debugging)

    def _samples(self, path: Path) -> np.ndarray:
        """Return the decoded samples for a WAV, loading it on first use and caching it."""
        s = self._cache.get(path)
        if s is None:
            s = load_wav(path, self.backend.rate)
            # Keep memory bounded: when the cache is big, drop the oldest entry.
            if len(self._cache) > 300:
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = s
        return s

    def play(self, cue: QueuedCue) -> None:
        """Ask to play a cue. It starts right away if the scheduler says so, otherwise it waits."""
        if cue.path is None or not cue.path.exists():
            log.warning("missing clip %s", cue.clip)
            return
        with self._lock:
            # time.monotonic() is a clock that only ever goes forward (unaffected by the user
            # changing the system time), ideal for measuring durations.
            if self.sched.submit(cue, time.monotonic()) == "play":
                self._start(cue)

    def _start(self, cue: QueuedCue) -> None:
        """Actually begin playing ``cue`` on the backend (called with the lock held)."""
        try:
            samples = self._samples(cue.path)
        except Exception as e:
            # Unreadable file: skip it and move on to whatever is next in the queue.
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
        """The backend reports a clip finished. Start the next queued cue, if any."""
        with self._lock:
            if token != self._token:
                return  # a clip that was already interrupted
            nxt = self.sched.finished(time.monotonic())
            if nxt:
                self._start(nxt)

    def set_tone(self, freq_hz: float, duration_ms: float, pan: float, interval_ms: float | None) -> None:
        """Start or change the guidance beep (see ToneState for what each number means)."""
        self.backend.set_tone(ToneState(freq_hz, duration_ms, pan, interval_ms))

    def stop_tone(self) -> None:
        """Turn the guidance beep off."""
        self.backend.set_tone(ToneState())

    def set_gain(self, gain: float) -> None:
        """Set the volume multiplier (computed from the 0..10 level by volume_dial.level_to_gain)."""
        self.backend.set_gain(gain)

    def idle(self) -> bool:
        """True when nothing is playing."""
        with self._lock:
            return self.sched.current is None

    def wait_idle(self, timeout: float) -> None:
        """Block until playback is finished or ``timeout`` seconds pass (used before shutdown so
        "Shutting down" is heard in full)."""
        end = time.monotonic() + timeout
        while not self.idle() and time.monotonic() < end:
            time.sleep(0.05)

    def close(self) -> None:
        """Release the audio hardware."""
        self.backend.close()
