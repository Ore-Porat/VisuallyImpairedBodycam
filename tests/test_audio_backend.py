"""Audio backend selection and the sounddevice -> aplay fallback."""

import slr.device.audio as audio
from slr.common.config import AudioConfig


def _fail_sounddevice(cfg, on_done):
    raise RuntimeError("PortAudio library not found")


def test_sounddevice_falls_back_to_aplay(monkeypatch):
    """A Pi with aplay but no PortAudio still gets speech playback."""
    monkeypatch.setattr(audio, "SoundDeviceBackend", _fail_sounddevice)
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/aplay")

    backend = audio.make_backend(AudioConfig(backend="sounddevice"), lambda token: None)

    assert isinstance(backend, audio.AplayBackend)


def test_sounddevice_falls_back_to_silence_without_aplay(monkeypatch):
    """If neither sounddevice nor aplay works, the device keeps running silently."""
    monkeypatch.setattr(audio, "SoundDeviceBackend", _fail_sounddevice)
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)

    backend = audio.make_backend(AudioConfig(backend="sounddevice"), lambda token: None)

    assert isinstance(backend, audio.NullBackend)


def test_explicit_aplay_backend(monkeypatch):
    """audio.backend: aplay bypasses sounddevice entirely."""
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/aplay")

    backend = audio.make_backend(AudioConfig(backend="aplay"), lambda token: None)

    assert isinstance(backend, audio.AplayBackend)
