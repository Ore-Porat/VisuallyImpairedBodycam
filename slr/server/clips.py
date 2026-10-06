"""Phrase catalog, offline TTS rendering and the clip cache + manifest (FR-53, FR-60..63).

Every phrase is rendered on the laptop into a WAV clip, one clip per full phrase (for natural
sound). Clip names are ``<key>-<hash of voice + text>`` so a clip regenerates automatically
when the phrase text, the spoken class name or the voice changes. The device syncs clips using
the manifest (name -> sha256) and fetches anything missing over HTTP.
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
from slr.server.guidance import H_DIRS, V_ZONES, Say

log = logging.getLogger(__name__)

ITEM_PARAMS = ("item", "next", "other")
DIRECTION_PARAMS = ("h", "v")


# --------------------------------------------------------------------------- phrase catalog


@dataclass(frozen=True)
class ResolvedCue:
    key: str
    text: str
    priority: str
    clip: str


class PhraseCatalog:
    def __init__(self, phrases: dict[str, dict], directions: dict[str, str],
                 system: dict[str, str], classes: dict[str, str], voice_tag: str = ""):
        self.phrases = phrases
        self.directions = directions
        self.system = system
        self.classes = classes
        self.voice_tag = voice_tag
        for key, ph in phrases.items():
            if "text" not in ph:
                raise ConfigError(f"phrase {key!r} has no text")
            if ph.get("priority", "info") not in PRIORITIES:
                raise ConfigError(f"phrase {key!r} has unknown priority {ph.get('priority')!r}")

    @classmethod
    def load(cls, path: Path, classes: dict[str, str], voice_tag: str = "") -> PhraseCatalog:
        data = read_yaml(path)
        return cls(data.get("phrases", {}), data.get("directions", {}),
                   data.get("system_phrases", {}), classes, voice_tag)

    def spoken(self, cls_id: str) -> str:
        return self.classes.get(cls_id, cls_id.replace("_", " "))

    def text(self, key: str, params: dict) -> str:
        ph = self.phrases.get(key)
        if ph is None:
            raise KeyError(f"no phrase {key!r} in phrases.yaml")
        values = {}
        for k, v in params.items():
            if k in ITEM_PARAMS:
                values[k] = self.spoken(str(v))
            elif k in DIRECTION_PARAMS:
                values[k] = self.directions.get(str(v), str(v))
            else:
                values[k] = v
        text = " ".join(str(ph["text"]).format(**values).split())
        return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

    def clip_name(self, key: str, text: str) -> str:
        digest = hashlib.sha1(f"{self.voice_tag}|{text}".encode()).hexdigest()[:10]
        return f"{key}-{digest}"

    def resolve(self, cue: Say) -> ResolvedCue:
        text = self.text(cue.key, cue.params)
        return ResolvedCue(cue.key, text, self.phrases[cue.key].get("priority", "info"),
                           self.clip_name(cue.key, text))

    def prerender_set(self, items: Iterable[ListItem] = (), all_classes: bool = True) -> list[Say]:
        """Cues worth rendering ahead of time so the first use has no TTS delay."""
        items = list(items)
        names = [it.name for it in items]
        cues: list[Say] = []
        for key, ph in self.phrases.items():
            if "{" not in ph["text"]:
                cues.append(Say(key))
        for h in sorted(set(H_DIRS.values())):
            cues.append(Say("direction_h", {"h": h}))
            for v in V_ZONES:
                cues.append(Say("direction", {"h": h, "v": v}))
        targets = list(self.classes) if all_classes else names
        for c in targets:
            for key in ("found", "searching", "not_seeing", "within_reach", "identified",
                        "identified_on_list", "checked_off_done"):
                cues.append(Say(key, {"item": c}))
        total = len(items)
        for i, name in enumerate(names):
            cues.append(Say("item_announce", {"n": i + 1, "total": total, "item": name}))
            cues.append(Say("item_announce_found", {"n": i + 1, "total": total, "item": name}))
            cues.append(Say("status", {"found": 0, "total": total, "item": name}))
            for other in names:
                if other != name:
                    cues.append(Say("checked_off_next", {"item": name, "next": other}))
        return cues


# --------------------------------------------------------------------------- TTS engines


class TTSEngine(Protocol):
    name: str

    def synth(self, text: str, path: Path) -> None: ...


class PiperTTS:
    name = "piper"

    def __init__(self, voice_path: Path, length_scale: float = 1.0):
        from piper import PiperVoice  # imported lazily: the module is optional

        if not voice_path.exists():
            raise FileNotFoundError(f"Piper voice not found: {voice_path} (run `slr clips download-voice`)")
        self.voice = PiperVoice.load(str(voice_path))
        self.length_scale = length_scale
        self._lock = threading.Lock()

    def synth(self, text: str, path: Path) -> None:
        from piper import SynthesisConfig

        with self._lock, wave.open(str(path), "wb") as wf:
            self.voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=self.length_scale))


class EspeakTTS:
    name = "espeak"

    def __init__(self, speed_wpm: int = 165):
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.exe:
            raise FileNotFoundError("espeak-ng not installed")
        self.speed = speed_wpm

    def synth(self, text: str, path: Path) -> None:
        subprocess.run([self.exe, "-s", str(self.speed), "-w", str(path), text], check=True,
                       capture_output=True, timeout=30)


class Pyttsx3TTS:
    """OS voice (Windows SAPI / macOS) through pyttsx3, in a subprocess to keep COM happy."""

    name = "pyttsx3"
    _SCRIPT = ("import sys, pyttsx3; e = pyttsx3.init(); "
               "e.save_to_file(sys.argv[1], sys.argv[2]); e.runAndWait()")

    def __init__(self):
        import pyttsx3  # noqa: F401  (availability check)

    def synth(self, text: str, path: Path) -> None:
        subprocess.run([sys.executable, "-c", self._SCRIPT, text, str(path)], check=True,
                       capture_output=True, timeout=60)
        if not path.exists() or path.stat().st_size < 100:
            raise RuntimeError("pyttsx3 produced no audio")


class BeepTTS:
    """Last resort so the pipeline still runs with no TTS installed: a short double beep."""

    name = "beep"

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate

    def synth(self, text: str, path: Path) -> None:
        t = np.arange(int(self.sr * 0.08)) / self.sr
        beep = 0.4 * np.sin(2 * np.pi * 660 * t)
        gap = np.zeros(int(self.sr * 0.05))
        write_wav(path, np.concatenate([beep, gap, beep]), self.sr)


def make_tts(cfg: TTSConfig, server_cfg: ServerConfig | None = None) -> TTSEngine:
    errors = []
    for name in cfg.engines + ["beep"]:
        try:
            if name == "piper":
                voice = server_cfg.path(cfg.piper_voice) if server_cfg else Path(cfg.piper_voice)
                return PiperTTS(voice, cfg.length_scale)
            if name == "espeak":
                return EspeakTTS(int(165 / max(cfg.length_scale, 0.1)))
            if name == "pyttsx3":
                return Pyttsx3TTS()
            if name == "beep":
                log.warning("no TTS engine available (%s); using beeps instead of speech", "; ".join(errors))
                return BeepTTS(cfg.sample_rate)
        except Exception as e:  # missing package, voice file or binary
            errors.append(f"{name}: {e}")
    raise RuntimeError("unreachable")


def voice_tag(cfg: TTSConfig, engine: TTSEngine) -> str:
    if engine.name == "piper":
        return f"piper:{Path(cfg.piper_voice).name}:{cfg.length_scale}"
    return f"{engine.name}:{cfg.length_scale}"


def download_voice(cfg: ServerConfig) -> Path:
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
    data = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- clip store


class ClipStore:
    def __init__(self, clips_dir: Path, tts: TTSEngine):
        self.dir = Path(clips_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tts = tts
        self._lock = threading.Lock()       # serializes TTS rendering
        self._mlock = threading.Lock()      # guards the manifest dict (read by the HTTP thread)
        self._manifest: dict[str, str] = {}
        for p in self.dir.glob("*.wav"):
            if not p.name.endswith(".tmp.wav"):
                self._manifest[p.stem] = sha256_file(p)

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.wav"

    def has(self, name: str) -> bool:
        with self._mlock:
            return name in self._manifest

    def ensure(self, name: str, text: str) -> Path:
        """Render the clip if it does not exist yet. Thread-safe; called from executor threads."""
        path = self.path(name)
        if self.has(name):
            return path
        with self._lock:
            if self.has(name):
                return path
            tmp = path.with_suffix(".tmp.wav")
            self.tts.synth(text, tmp)
            tmp.replace(path)
            digest = sha256_file(path)
            with self._mlock:
                self._manifest[name] = digest
            log.debug("rendered clip %s: %r", name, text)
        return path

    def manifest(self) -> dict[str, str]:
        with self._mlock:
            return dict(self._manifest)

    def manifest_hash(self) -> str:
        return manifest_hash(self.manifest())

    def prerender(self, catalog: PhraseCatalog, cues: Iterable[Say]) -> int:
        n = 0
        for cue in cues:
            r = catalog.resolve(cue)
            if not self.has(r.clip):
                self.ensure(r.clip, r.text)
                n += 1
        return n


def build_system_clips(catalog: PhraseCatalog, tts: TTSEngine, out_dir: Path) -> list[Path]:
    """Render the device's local clips (FR-52): device_assets/system_clips/<key>.wav."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, text in catalog.system.items():
        path = out_dir / f"{key}.wav"
        tts.synth(str(text), path)
        paths.append(path)
    return paths
