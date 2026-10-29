"""Phrase catalog, offline TTS rendering and the clip cache + manifest (FR-53, FR-60..63).

Every phrase is rendered on the laptop into a WAV clip, one clip per full phrase (for natural
sound). Clip names are ``<key>-<hash of voice + text>`` so a clip regenerates automatically
when the phrase text, the spoken class name or the voice changes. The device syncs clips using
the manifest (name -> sha256) and fetches anything missing over HTTP.

WHAT THIS FILE DOES
-------------------
The product talks to the shopper, but the Raspberry Pi is too small to run a speech engine. So
the LAPTOP converts every sentence it might ever say into a small audio file (a "clip") ahead
of time, and the Pi just plays the files. This file has three parts:

1. ``PhraseCatalog``: reads ``config/phrases.yaml`` (templates like "{item} found. {h}.") and
   fills in the blanks to produce the final sentence and a unique clip file name.
2. TTS engines: several ways to turn text into speech, tried in order of quality (Piper, then
   espeak, then the operating system's voice, then a plain beep as a last resort).
3. ``ClipStore``: the folder of rendered WAV files plus a "manifest" (a list of file name ->
   checksum) that the device compares against its own copy to see what it is missing.

HOW IT FITS IN
--------------
guidance engine -> ``Say(key, params)`` -> ``PhraseCatalog.resolve`` -> text + clip name
-> ``ClipStore.ensure`` renders the WAV if needed -> the device downloads it over HTTP
(``list_web.py`` serves the files) and plays it.

WORDS YOU WILL SEE
------------------
- TTS: text-to-speech, software that turns written words into spoken audio.
- Piper: a good-quality offline TTS engine. Its "voice" is a downloadable model file (.onnx).
- WAV: a simple uncompressed audio file format.
- hash / sha1 / sha256: a short "fingerprint" computed from some data. Same data -> same
  fingerprint; any change -> a completely different one. Used to name clips and to let the
  device check whether its copy of a clip is current.
- manifest: a list of every clip name and its fingerprint.
- thread / lock: the server runs several things at once; a lock makes sure only one thread
  renders speech (or edits the manifest) at a time.
- Protocol (typing): "anything with these methods counts as a TTSEngine".
- FR-xx: requirement numbers in the product spec, ``PRD.pdf``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from slr.common.config import ConfigError, ServerConfig, TTSConfig, read_yaml
from slr.common.protocol import PRIORITIES, manifest_hash
from slr.common.types import ListItem
from slr.server.guidance import H_DIRS, HAND_DIRS, V_ZONES, Say

log = logging.getLogger(__name__)

# Phrase parameters whose value is a class id that must be turned into a spoken name.
ITEM_PARAMS = ("item", "next", "other")
# Phrase parameters whose value is a direction key ("slightly_left", "eye") to look up in
# the ``directions`` table of phrases.yaml.
DIRECTION_PARAMS = ("h", "v")


def join_spoken(names: list[str]) -> str:
    """Join names as a natural list: "pasta", "pasta and milk", "pasta, milk, and rice"."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


# --------------------------------------------------------------------------- phrase catalog


@dataclass(frozen=True)
class ResolvedCue:
    """A Say effect turned into concrete words: the final text, its priority and clip name."""

    key: str
    text: str
    priority: str
    clip: str


class PhraseCatalog:
    """All the sentences the product can say, loaded from ``config/phrases.yaml``."""

    def __init__(self, phrases: dict[str, dict], directions: dict[str, str],
                 system: dict[str, str], classes: dict[str, str], voice_tag: str = ""):
        """phrases:    key -> {"text": "...", "priority": "..."} from the YAML.
        directions: direction key -> spoken words ("slightly_left" -> "slightly left").
        system:     the device's built-in phrases ("Starting up", "Volume 5"...).
        classes:    class id -> spoken name.
        voice_tag:  a string describing the TTS voice; part of every clip name so changing
                    the voice re-renders everything.
        """
        self.phrases = phrases
        self.directions = directions
        self.system = system
        self.classes = classes
        self.voice_tag = voice_tag
        # Validate the YAML up front so a typo fails at startup, not mid-demo.
        for key, ph in phrases.items():
            if "text" not in ph:
                raise ConfigError(f"phrase {key!r} has no text")
            if ph.get("priority", "info") not in PRIORITIES:
                raise ConfigError(f"phrase {key!r} has unknown priority {ph.get('priority')!r}")

    # ``@classmethod`` means you call it on the class itself: PhraseCatalog.load(...).
    @classmethod
    def load(cls, path: Path, classes: dict[str, str], voice_tag: str = "") -> PhraseCatalog:
        """Read phrases.yaml and build a catalog."""
        data = read_yaml(path)
        return cls(data.get("phrases", {}), data.get("directions", {}),
                   data.get("system_phrases", {}), classes, voice_tag)

    def spoken(self, cls_id: str) -> str:
        """Spoken name for a class id; falls back to replacing underscores with spaces."""
        return self.classes.get(cls_id, cls_id.replace("_", " "))

    def text(self, key: str, params: dict) -> str:
        """Fill the phrase template for ``key`` with ``params`` and tidy the result."""
        ph = self.phrases.get(key)
        if ph is None:
            raise KeyError(f"no phrase {key!r} in phrases.yaml")
        # Convert each parameter into its spoken form before filling the template.
        values = {}
        for k, v in params.items():
            if k == "items" and isinstance(v, (list, tuple)):
                values[k] = join_spoken([self.spoken(str(x)) for x in v])
            elif k in ITEM_PARAMS:
                values[k] = self.spoken(str(v))
            elif k in DIRECTION_PARAMS:
                values[k] = self.directions.get(str(v), str(v))
            else:
                values[k] = v
        # ``.format(**values)`` replaces {item}, {h}, ... in the template.
        # ``" ".join(text.split())`` collapses any double spaces left by empty parts.
        text = " ".join(str(ph["text"]).format(**values).split())
        # Capitalise the first letter of every sentence (regex: start of string or after . ! ?).
        return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

    def clip_name(self, key: str, text: str) -> str:
        """File name (without .wav) for this phrase: key plus a fingerprint of voice + text."""
        digest = hashlib.sha1(f"{self.voice_tag}|{text}".encode()).hexdigest()[:10]
        return f"{key}-{digest}"

    def resolve(self, cue: Say) -> ResolvedCue:
        """Turn a Say effect into final text, priority and clip name."""
        text = self.text(cue.key, cue.params)
        return ResolvedCue(cue.key, text, self.phrases[cue.key].get("priority", "info"),
                           self.clip_name(cue.key, text))

    def prerender_set(self, items: Iterable[ListItem] = (), all_classes: bool = True) -> list[Say]:
        """Cues worth rendering ahead of time so the first use has no TTS delay.

        items:       the current shopping list (for "item 2 of 5" style phrases).
        all_classes: True = also render per-item phrases for every class the model knows.
        Returns a list of Say effects to feed into ``ClipStore.prerender``.
        """
        items = list(items)
        names = [it.name for it in items]
        cues: list[Say] = []
        # Phrases with no {placeholders} are always the same: render them all.
        for key, ph in self.phrases.items():
            if "{" not in ph["text"]:
                cues.append(Say(key))
        # Every direction combination ("slightly left", "left, low shelf", ...).
        for h in sorted(set(H_DIRS.values())):
            cues.append(Say("direction_h", {"h": h}))
            for v in V_ZONES:
                cues.append(Say("direction", {"h": h, "v": v}))
        # Hand cues ("Hand left.", "Reach forward.").
        for hd in HAND_DIRS:
            cues.append(Say("hand_dir", {"hd": hd}))
        # Per-item phrases for each class (or just the items on the list).
        targets = list(self.classes) if all_classes else names
        for c in targets:
            for key in ("found", "searching", "not_seeing", "within_reach", "identified",
                        "identified_on_list", "checked_off_done", "item_marked_found",
                        "item_marked_found_done", "item_marked_pending"):
                cues.append(Say(key, {"item": c}))
        # List-position phrases depend on the exact list, so only for the current items.
        total = len(items)
        for i, name in enumerate(names):
            cues.append(Say("item_announce", {"n": i + 1, "total": total, "item": name}))
            cues.append(Say("item_announce_found", {"n": i + 1, "total": total, "item": name}))
            cues.append(Say("status", {"found": 0, "total": total, "item": name}))
            for other in names:
                if other != name:
                    cues.append(Say("checked_off_next", {"item": name, "next": other}))
                    cues.append(Say("item_marked_found_next", {"item": name, "next": other}))
        return cues


# --------------------------------------------------------------------------- TTS engines
# Each engine has a ``name`` and a ``synth(text, path)`` method that writes a WAV file.


class TTSEngine(Protocol):
    """The shape every TTS engine must have."""

    name: str

    def synth(self, text: str, path: Path) -> None: ...


class PiperTTS:
    """Preferred engine: Piper, a neural offline voice. Needs the ``piper`` package + a voice file."""

    name = "piper"

    def __init__(self, voice_path: Path, length_scale: float = 1.0):
        """voice_path: the downloaded .onnx voice. length_scale: >1 slower speech, <1 faster."""
        from piper import PiperVoice  # imported lazily: the module is optional

        if not voice_path.exists():
            raise FileNotFoundError(f"Piper voice not found: {voice_path} (run `slr clips download-voice`)")
        self.voice = PiperVoice.load(str(voice_path))
        self.length_scale = length_scale
        # Piper is not thread-safe; only one render at a time.
        self._lock = threading.Lock()

    def synth(self, text: str, path: Path) -> None:
        """Render ``text`` to the WAV file at ``path``."""
        from piper import SynthesisConfig

        # Two context managers in one ``with``: hold the lock AND keep the WAV file open.
        with self._lock, wave.open(str(path), "wb") as wf:
            self.voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=self.length_scale))


class EspeakTTS:
    """Fallback: the classic ``espeak-ng`` command-line program (robotic but always works)."""

    name = "espeak"

    def __init__(self, speed_wpm: int = 165):
        """speed_wpm: words per minute."""
        # ``shutil.which`` finds a program on the PATH, like typing its name in a terminal.
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.exe:
            raise FileNotFoundError("espeak-ng not installed")
        self.speed = speed_wpm

    def synth(self, text: str, path: Path) -> None:
        """Run espeak as a separate program and have it write the WAV."""
        subprocess.run([self.exe, "-s", str(self.speed), "-w", str(path), text], check=True,
                       capture_output=True, timeout=30)


class Pyttsx3TTS:
    """OS voice (Windows SAPI / macOS) through pyttsx3, in a subprocess to keep COM happy."""

    name = "pyttsx3"
    # The tiny Python program run in the subprocess: init the OS voice, save text to a file.
    _SCRIPT = ("import sys, pyttsx3; e = pyttsx3.init(); "
               "e.save_to_file(sys.argv[1], sys.argv[2]); e.runAndWait()")

    def __init__(self):
        """Fails immediately (ImportError) if pyttsx3 is not installed."""
        import pyttsx3  # noqa: F401  (availability check)

    def synth(self, text: str, path: Path) -> None:
        """Run the helper script in a fresh Python process and check it produced audio."""
        subprocess.run([sys.executable, "-c", self._SCRIPT, text, str(path)], check=True,
                       capture_output=True, timeout=60)
        if not path.exists() or path.stat().st_size < 100:
            raise RuntimeError("pyttsx3 produced no audio")


class BeepTTS:
    """Last resort so the pipeline still runs with no TTS installed: a short double beep."""

    name = "beep"

    def __init__(self, sample_rate: int = 22050):
        """sample_rate: audio samples per second of the generated WAV."""
        self.sr = sample_rate

    def synth(self, text: str, path: Path) -> None:
        """Ignore the text; write beep-gap-beep as a sine wave."""
        # t = time of each sample over 80 ms; the beep is a 660 Hz sine at 40% volume.
        t = np.arange(int(self.sr * 0.08)) / self.sr
        beep = 0.4 * np.sin(2 * np.pi * 660 * t)
        gap = np.zeros(int(self.sr * 0.05))
        write_wav(path, np.concatenate([beep, gap, beep]), self.sr)


def make_tts(cfg: TTSConfig, server_cfg: ServerConfig | None = None) -> TTSEngine:
    """Try the engines listed in ``tts.engines`` (server.yaml) in order; return the first that works.

    "beep" is always appended as the final fallback, so this never fails to return something.
    """
    errors = []
    for name in cfg.engines + ["beep"]:
        try:
            if name == "piper":
                voice = server_cfg.path(cfg.piper_voice) if server_cfg else Path(cfg.piper_voice)
                return PiperTTS(voice, cfg.length_scale)
            if name == "espeak":
                # espeak takes words per minute; derive it from the same length_scale setting.
                return EspeakTTS(int(165 / max(cfg.length_scale, 0.1)))
            if name == "pyttsx3":
                return Pyttsx3TTS()
            if name == "beep":
                log.warning("no TTS engine available (%s); using beeps instead of speech", "; ".join(errors))
                return BeepTTS(cfg.sample_rate)
        except Exception as e:  # missing package, voice file or binary
            # Remember why this engine failed and try the next one.
            errors.append(f"{name}: {e}")
    raise RuntimeError("unreachable")


def voice_tag(cfg: TTSConfig, engine: TTSEngine) -> str:
    """A short string that identifies the voice. It is mixed into clip names (see clip_name)."""
    if engine.name == "piper":
        return f"piper:{Path(cfg.piper_voice).name}:{cfg.length_scale}"
    return f"{engine.name}:{cfg.length_scale}"


def download_voice(cfg: ServerConfig) -> Path:
    """Download the Piper voice (.onnx and its .json) once; used by ``slr clips download-voice``."""
    dest = cfg.path(cfg.tts.piper_voice)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url, target in ((cfg.tts.piper_voice_url, dest),
                        (cfg.tts.piper_voice_url + ".json", Path(str(dest) + ".json"))):
        if target.exists():
            continue
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, target)
    return dest


# --------------------------------------------------------------------------- WAV helpers


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Save float samples in -1..1 as a mono 16-bit WAV file."""
    # Clip to the valid range and scale to 16-bit integers ("<i2" = little-endian 2-byte int).
    data = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def sha256_file(path: Path) -> str:
    """Fingerprint of a file's contents (hex string). Same bytes -> same string."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- clip store


class ClipStore:
    """The folder of rendered clips (``data/clips``) and the manifest of what is in it."""

    def __init__(self, clips_dir: Path, tts: TTSEngine):
        """Scan ``clips_dir`` for existing .wav files and fingerprint them."""
        self.dir = Path(clips_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tts = tts
        self._lock = threading.Lock()       # serializes TTS rendering
        self._mlock = threading.Lock()      # guards the manifest dict (read by the HTTP thread)
        self._manifest: dict[str, str] = {}
        for p in self.dir.glob("*.wav"):
            # Skip half-written temp files from an interrupted render.
            if not p.name.endswith(".tmp.wav"):
                self._manifest[p.stem] = sha256_file(p)

    def path(self, name: str) -> Path:
        """Full path of the clip with this name."""
        return self.dir / f"{name}.wav"

    def has(self, name: str) -> bool:
        """True if the clip is already rendered."""
        with self._mlock:
            return name in self._manifest

    def ensure(self, name: str, text: str) -> Path:
        """Render the clip if it does not exist yet. Thread-safe; called from executor threads."""
        path = self.path(name)
        if self.has(name):
            return path
        with self._lock:
            # Check again inside the lock: another thread may have just rendered it.
            if self.has(name):
                return path
            # Render to a temp name, then rename, so a crash never leaves a broken .wav.
            tmp = path.with_suffix(".tmp.wav")
            self.tts.synth(text, tmp)
            tmp.replace(path)
            digest = sha256_file(path)
            with self._mlock:
                self._manifest[name] = digest
            log.debug("rendered clip %s: %r", name, text)
        return path

    def manifest(self) -> dict[str, str]:
        """A copy of name -> sha256 for every rendered clip."""
        with self._mlock:
            return dict(self._manifest)

    def manifest_hash(self) -> str:
        """One fingerprint of the whole manifest; the device compares it to spot changes."""
        return manifest_hash(self.manifest())

    def prerender(self, catalog: PhraseCatalog, cues: Iterable[Say]) -> int:
        """Render every cue in ``cues`` that is not cached yet. Returns how many were rendered."""
        n = 0
        for cue in cues:
            r = catalog.resolve(cue)
            if not self.has(r.clip):
                self.ensure(r.clip, r.text)
                n += 1
        return n


def build_system_clips(catalog: PhraseCatalog, tts: TTSEngine, out_dir: Path) -> list[Path]:
    """Render the device's local clips (FR-52): device_assets/system_clips/<key>.wav.

    These are the phrases the Pi must be able to say even before it has talked to the laptop
    ("Starting up", "Not connected", "Volume 3"...). They are committed to git.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, text in catalog.system.items():
        path = out_dir / f"{key}.wav"
        tts.synth(str(text), path)
        paths.append(path)
    return paths
