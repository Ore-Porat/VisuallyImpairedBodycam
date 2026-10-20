"""Cue priority and interruption (FR-63) plus the audio helpers.

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    ``slr/device/audio.py`` runs on the Pi and plays the spoken cues in the headphones.
    Cues can arrive faster than they can be spoken, so a ``CueScheduler`` decides what
    to do with each new cue: play it now (cutting off the current one), queue it, or
    drop it. Rules (FR-63 in the product spec, PRD.pdf):
      * priorities, low to high: info < guidance < confirmation < system
      * a higher-priority cue interrupts a lower one
      * equal or lower priority waits in a queue, unless the cue is flagged "interrupt"
      * guidance cues ("left", "straight") never pile up: the newest replaces the waiting
        one, and one that has waited too long is dropped because it is out of date
    The file also has small audio helpers: stereo panning math, rendering the guidance
    beep tone, loading WAV files, and the ``AudioPlayer`` that runs the queue for real.

Words you will see:
    priority numbers  INFO=0, GUIDANCE=1, CONFIRMATION=2, SYSTEM=3 (higher wins).
    submit(cue, t)    Offer a cue to the scheduler at time t; returns "play" or "queued".
    finished(t)       Tell the scheduler the current clip ended at time t; returns the
                      next cue to play, or None.
    pytest.approx     "Equal, allowing for tiny floating-point rounding differences."
    tmp_path          A temporary folder from pytest, deleted after the test.
    assert            "Check that this is true; if not, the test fails."
"""

import time
import wave
from pathlib import Path

import numpy as np
import pytest

from slr.common.config import AudioConfig
from slr.device.audio import (AudioPlayer, CueScheduler, NullBackend, QueuedCue, ToneState, load_wav,
                              pan_gains, render_tone)

# Priority levels, lowest to highest (same numbers the protocol uses).
INFO, GUIDANCE, CONFIRMATION, SYSTEM = 0, 1, 2, 3


def cue(name, prio, interrupt=False):
    """Shorthand for building a cue with just a clip name and a priority."""
    return QueuedCue(clip=name, priority=prio, interrupt=interrupt)


def test_plays_immediately_when_idle():
    """Nothing is playing, so the first cue starts right away."""
    s = CueScheduler()
    assert s.submit(cue("a", INFO), 0) == "play"
    assert s.current.clip == "a"


def test_higher_priority_interrupts():
    """A confirmation cuts off guidance, and a system cue cuts off a confirmation."""
    s = CueScheduler()
    s.submit(cue("dir", GUIDANCE), 0)
    assert s.submit(cue("found", CONFIRMATION), 0.1) == "play"
    assert s.current.clip == "found"
    assert s.submit(cue("lost conn", SYSTEM), 0.2) == "play"


def test_lower_or_equal_priority_queues():
    """Equal/lower cues wait; higher priority leaves the queue first, then first-come order."""
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
    """Item announcements are flagged interrupt=True so rapid 'next' presses do not queue up."""
    s = CueScheduler()
    s.submit(cue("item 2", CONFIRMATION, True), 0)
    assert s.submit(cue("item 3", CONFIRMATION, True), 0.1) == "play"
    assert s.current.clip == "item 3"


def test_interrupt_flag_does_not_beat_higher_priority():
    """The interrupt flag only works against equal priority, never against a system cue."""
    s = CueScheduler()
    s.submit(cue("system", SYSTEM), 0)
    assert s.submit(cue("item", CONFIRMATION, True), 0.1) == "queued"


def test_guidance_never_queues_up():
    """Three direction cues while busy: only the newest one is still waiting."""
    s = CueScheduler()
    s.submit(cue("found", CONFIRMATION), 0)
    s.submit(cue("left", GUIDANCE), 0.1)
    s.submit(cue("slightly left", GUIDANCE), 0.2)
    s.submit(cue("straight", GUIDANCE), 0.3)
    assert [q.clip for q in s.queue] == ["straight"]


def test_stale_guidance_dropped():
    """A direction that waited longer than guidance_max_wait_s is thrown away unspoken."""
    s = CueScheduler(guidance_max_wait_s=2.0)
    s.submit(cue("help", CONFIRMATION), 0)
    s.submit(cue("left", GUIDANCE), 0.5)
    assert s.finished(5.0) is None


def test_interrupt_drops_stale_guidance_but_keeps_important():
    """A system interrupt clears lower cues from the queue; confirmations can still interrupt."""
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
    """pan -1 is fully left, +1 fully right, 0 equal in both ears."""
    l, r = pan_gains(0)
    assert l == pytest.approx(r)
    assert pan_gains(-1) == pytest.approx((1, 0), abs=1e-9)
    assert pan_gains(1) == pytest.approx((0, 1), abs=1e-9)


def test_tone_render_rate_and_pan():
    """The guidance beep repeats at the requested interval and is panned to the left ear."""
    rate = 22050
    # 880 Hz beep, 60 ms long, fully left, repeating every 200 ms.
    tone = ToneState(freq_hz=880, duration_ms=60, pan=-1.0, interval_ms=200)
    out = render_tone(tone, 0, rate, rate)  # one second
    left, right = out[:, 0], out[:, 1]
    assert np.abs(right).max() < 1e-6       # fully left
    # Count the beeps: places where the left channel goes from silent to loud.
    on = np.abs(left) > 1e-4
    starts = np.flatnonzero(on[1:] & ~on[:-1])
    assert 4 <= len(starts) <= 6            # ~5 beeps per second at 200 ms
    # A default ToneState (no interval) is silence.
    assert not render_tone(ToneState(), 0, 100, rate).any()


def write_test_wav(path: Path, rate=16000, secs=0.1, channels=1):
    """Write a short sine-wave WAV file (16-bit samples) for the loading/playing tests."""
    n = int(rate * secs)
    data = (np.sin(np.arange(n * channels) / 5) * 10000).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())


def test_load_wav_resamples_to_stereo(tmp_path):
    """A mono 16 kHz file is converted to stereo float samples at the player's 22.05 kHz."""
    p = tmp_path / "a.wav"
    write_test_wav(p, rate=16000, secs=0.5)
    s = load_wav(p, 22050)
    # 0.5 s at 22050 samples/s = 11025 samples, 2 channels.
    assert s.shape == (11025, 2)
    assert s.dtype == np.float32


def test_player_runs_queue_in_order(tmp_path):
    """The real AudioPlayer (with the silent backend) honors the scheduler's decisions."""
    for name in ("a", "b", "c", "d"):
        write_test_wav(tmp_path / f"{name}.wav", secs=0.05)
    started = []
    # on_started is called each time a clip begins; we record the clip names.
    player = AudioPlayer(AudioConfig(backend="null"), on_started=lambda c: started.append(c.clip))
    player.play(QueuedCue("a", CONFIRMATION, path=tmp_path / "a.wav", cue_id=1))
    player.play(QueuedCue("b", CONFIRMATION, path=tmp_path / "b.wav", cue_id=2))  # queued
    player.play(QueuedCue("c", SYSTEM, path=tmp_path / "c.wav", cue_id=3))  # interrupts a, drops stale b
    player.play(QueuedCue("d", GUIDANCE, path=tmp_path / "d.wav", cue_id=4))  # waits for c
    player.wait_idle(2.0)
    assert started == ["a", "c", "d"]
    assert player.idle()
