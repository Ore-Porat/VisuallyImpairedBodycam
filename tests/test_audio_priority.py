"""Cue priority and interruption (FR-63) plus the audio helpers."""

import time
import wave
from pathlib import Path

import numpy as np
import pytest

from slr.common.config import AudioConfig
from slr.device.audio import (AudioPlayer, CueScheduler, NullBackend, QueuedCue, ToneState, load_wav,
                              pan_gains, render_tone)

INFO, GUIDANCE, CONFIRMATION, SYSTEM = 0, 1, 2, 3


def cue(name, prio, interrupt=False):
    return QueuedCue(clip=name, priority=prio, interrupt=interrupt)


def test_plays_immediately_when_idle():
    s = CueScheduler()
    assert s.submit(cue("a", INFO), 0) == "play"
    assert s.current.clip == "a"


def test_higher_priority_interrupts():
    s = CueScheduler()
    s.submit(cue("dir", GUIDANCE), 0)
    assert s.submit(cue("found", CONFIRMATION), 0.1) == "play"
    assert s.current.clip == "found"
    assert s.submit(cue("lost conn", SYSTEM), 0.2) == "play"


def test_lower_or_equal_priority_queues():
    s = CueScheduler()
    s.submit(cue("found", CONFIRMATION), 0)
    assert s.submit(cue("dir", GUIDANCE), 0.1) == "queued"
    assert s.submit(cue("searching", CONFIRMATION), 0.1) == "queued"
    # higher priority waits first, FIFO within a priority
    assert [q.clip for q in s.queue] == ["searching", "dir"]
    assert s.finished(0.2).clip == "searching"
    assert s.finished(0.3).clip == "dir"
    assert s.finished(0.4) is None


def test_interrupt_flag_cuts_off_equal_priority():
    s = CueScheduler()
    s.submit(cue("item 2", CONFIRMATION, True), 0)
    assert s.submit(cue("item 3", CONFIRMATION, True), 0.1) == "play"
    assert s.current.clip == "item 3"


def test_interrupt_flag_does_not_beat_higher_priority():
    s = CueScheduler()
    s.submit(cue("system", SYSTEM), 0)
    assert s.submit(cue("item", CONFIRMATION, True), 0.1) == "queued"


def test_guidance_never_queues_up():
    s = CueScheduler()
    s.submit(cue("found", CONFIRMATION), 0)
    s.submit(cue("left", GUIDANCE), 0.1)
    s.submit(cue("slightly left", GUIDANCE), 0.2)
    s.submit(cue("straight", GUIDANCE), 0.3)
    assert [q.clip for q in s.queue] == ["straight"]


def test_stale_guidance_dropped():
    s = CueScheduler(guidance_max_wait_s=2.0)
    s.submit(cue("help", CONFIRMATION), 0)
    s.submit(cue("left", GUIDANCE), 0.5)
    assert s.finished(5.0) is None


def test_interrupt_drops_stale_guidance_but_keeps_important():
    s = CueScheduler()
    s.submit(cue("found", CONFIRMATION), 0)
    s.submit(cue("left", GUIDANCE), 0.1)
    s.submit(cue("searching", CONFIRMATION), 0.1)
    s.submit(cue("lost", SYSTEM), 0.2)
    assert [q.clip for q in s.queue] == []  # the system cue interrupted; lower ones dropped
    s2 = CueScheduler()
    s2.submit(cue("dir", GUIDANCE), 0)
    s2.submit(cue("found", CONFIRMATION, True), 0.1)
    assert s2.current.clip == "found"


def test_pan_law():
    l, r = pan_gains(0)
    assert l == pytest.approx(r)
    assert pan_gains(-1) == pytest.approx((1, 0), abs=1e-9)
    assert pan_gains(1) == pytest.approx((0, 1), abs=1e-9)


def test_tone_render_rate_and_pan():
    rate = 22050
    tone = ToneState(freq_hz=880, duration_ms=60, pan=-1.0, interval_ms=200)
    out = render_tone(tone, 0, rate, rate)  # one second
    left, right = out[:, 0], out[:, 1]
    assert np.abs(right).max() < 1e-6       # fully left
    on = np.abs(left) > 1e-4
    starts = np.flatnonzero(on[1:] & ~on[:-1])
    assert 4 <= len(starts) <= 6            # ~5 beeps per second at 200 ms
    assert not render_tone(ToneState(), 0, 100, rate).any()


def write_test_wav(path: Path, rate=16000, secs=0.1, channels=1):
    n = int(rate * secs)
    data = (np.sin(np.arange(n * channels) / 5) * 10000).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())


def test_load_wav_resamples_to_stereo(tmp_path):
    p = tmp_path / "a.wav"
    write_test_wav(p, rate=16000, secs=0.5)
    s = load_wav(p, 22050)
    assert s.shape == (11025, 2)
    assert s.dtype == np.float32


def test_player_runs_queue_in_order(tmp_path):
    for name in ("a", "b", "c", "d"):
        write_test_wav(tmp_path / f"{name}.wav", secs=0.05)
    started = []
    player = AudioPlayer(AudioConfig(backend="null"), on_started=lambda c: started.append(c.clip))
    player.play(QueuedCue("a", CONFIRMATION, path=tmp_path / "a.wav", cue_id=1))
    player.play(QueuedCue("b", CONFIRMATION, path=tmp_path / "b.wav", cue_id=2))  # queued
    player.play(QueuedCue("c", SYSTEM, path=tmp_path / "c.wav", cue_id=3))  # interrupts a, drops stale b
    player.play(QueuedCue("d", GUIDANCE, path=tmp_path / "d.wav", cue_id=4))  # waits for c
    player.wait_idle(2.0)
    assert started == ["a", "c", "d"]
    assert player.idle()
