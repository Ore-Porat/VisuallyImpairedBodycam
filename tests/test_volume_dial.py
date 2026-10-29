"""Volume dial: calibration interpolation, log curve, hysteresis, settle timing (FR-44, FR-45).

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    The volume knob is a potentiometer read through an MCP3002 analog-to-digital chip,
    which gives a "raw count" (an integer). ``slr/device/volume_dial.py`` turns that
    noisy count into a volume level 0..10 and an audio gain:
      * ``VolumeCurve``  maps raw count -> dial position 0.0..1.0 using three calibration
        points (min, middle, max) recorded with `slr device calibrate-volume`.
      * ``level_from_position`` turns the position into a whole level 0..10, with
        hysteresis so a knob sitting right on a boundary does not flicker 5 / 6 / 5 / 6.
      * ``VolumeFilter`` adds a median filter (rejects single bad readings) and a
        "settle" delay so "Volume 7" is only announced after the knob stops moving.
      * ``level_to_gain`` converts a level into a loudness multiplier with even dB steps.
      * ``StepVolume`` is the fallback for the up / down buttons or keyboard.
    FR-44 and FR-45 are requirement numbers from the product spec, PRD.pdf.

Words you will see:
    raw count     The integer the ADC chip returns (the test calibration uses 16-bit
                  style numbers around 24000..29000; the real MCP3002 gives 0..1023).
    hysteresis    Requiring a bit of extra movement before changing state, to avoid
                  flicker at a boundary.
    pytest.approx "Equal, allowing for tiny floating-point rounding differences."
    parametrize   Runs the same test once per listed input value.
    assert        "Check that this is true; if not, the test fails."
"""

import math

import pytest

from slr.common.config import VolumeCalibration, VolumeConfig
from slr.device.volume_dial import (StepVolume, VolumeCurve, VolumeFilter, level_from_position,
                                    level_to_gain)

# Calibration used by most tests: raw counts at dial minimum, middle and maximum.
CAL = VolumeCalibration(raw_min=23924, raw_mid=27397, raw_max=29332)


def test_piecewise_hits_calibration_points():
    """The three calibration counts map exactly to positions 0, 0.5 and 1; outside is clamped."""
    c = VolumeCurve(CAL)
    assert c.position(23924) == 0.0
    assert c.position(27397) == pytest.approx(0.5)
    assert c.position(29332) == 1.0
    assert c.position(10000) == 0.0 and c.position(40000) == 1.0


def test_piecewise_is_monotonic_and_linear_within_segments():
    """Between calibration points the mapping is a straight line and never goes backwards."""
    c = VolumeCurve(CAL)
    assert c.position((23924 + 27397) / 2) == pytest.approx(0.25)
    vals = [c.position(r) for r in range(23000, 30000, 37)]
    assert vals == sorted(vals)


def test_descending_wiring_works():
    """If the pot is wired backwards (counts go DOWN as you turn up) the curve still works."""
    c = VolumeCurve(VolumeCalibration(raw_min=29332, raw_mid=27397, raw_max=23924))
    assert c.position(29332) == 0.0
    assert c.position(27397) == pytest.approx(0.5)
    assert c.position(23924) == 1.0


@pytest.mark.parametrize("mid", [27397, 25000])  # bends either way
def test_log_curve_passes_through_points(mid):
    """curve: log fits a smooth logarithmic curve through the same three points."""
    cal = VolumeCalibration(raw_min=23924, raw_mid=mid, raw_max=29332, curve="log")
    c = VolumeCurve(cal)
    assert c._log is not None
    assert c.position(23924) == pytest.approx(0.0, abs=1e-6)
    assert c.position(mid) == pytest.approx(0.5, abs=1e-6)
    assert c.position(29332) == pytest.approx(1.0, abs=1e-6)
    vals = [c.position(r) for r in range(23924, 29332, 50)]
    assert vals == sorted(vals)


def test_log_curve_falls_back_when_points_are_linear():
    """If the middle point is exactly halfway, a log fit is impossible; use straight lines."""
    c = VolumeCurve(VolumeCalibration(raw_min=0, raw_mid=50, raw_max=100, curve="log"))
    assert c._log is None
    assert c.position(25) == pytest.approx(0.25)


def test_bad_calibration_rejected():
    """A middle value that is not between min and max is a config error."""
    with pytest.raises(ValueError):
        VolumeCalibration(raw_min=1, raw_mid=5, raw_max=3)


def test_quantize_with_hysteresis():
    """level_from_position(position*10, previous_level, hysteresis) holds near boundaries."""
    assert level_from_position(5.0, None, 0.5) == 5
    assert level_from_position(5.6, 5, 0.5) == 5      # just past the 5.5 boundary: hold
    assert level_from_position(5.8, 5, 0.5) == 6      # a quarter step past: move
    assert level_from_position(5.4, 6, 0.5) == 6
    assert level_from_position(5.2, 6, 0.5) == 5
    assert level_from_position(9.9, 9, 0.5) == 10
    # Out-of-range inputs are clamped to 0 and 10.
    assert level_from_position(-1, None, 0.5) == 0
    assert level_from_position(12, None, 0.5) == 10


def raw_for(position):
    """Raw count that maps to a dial position with the piecewise curve."""
    if position <= 0.5:
        return 23924 + (27397 - 23924) * position / 0.5
    return 27397 + (29332 - 27397) * (position - 0.5) / 0.5


def test_filter_no_flicker_with_noise_at_boundary():
    """A knob parked on the 5/6 boundary with electrical noise changes level at most once."""
    f = VolumeFilter(VolumeCurve(CAL), VolumeConfig())
    t = 0.0
    changes = []
    base = raw_for(0.55)  # right on the 5 / 6 boundary
    for i in range(400):  # 20 s at 20 Hz with +-15 counts of noise
        noise = 15 * math.sin(i * 1.7)
        # add_sample returns (new_level_or_None, level_to_announce_or_None).
        new, _ = f.add_sample(base + noise, t)
        if new is not None:
            changes.append(new)
        t += 0.05
    assert len(changes) <= 1


def test_median_rejects_spikes():
    """One wild reading (a spike to maximum) does not change the level."""
    f = VolumeFilter(VolumeCurve(CAL), VolumeConfig())
    t = 0.0
    for _ in range(5):
        f.add_sample(raw_for(0.3), t)
        t += 0.05
    new, _ = f.add_sample(raw_for(1.0), t)  # one spike
    assert new is None and f.level == 3


def test_announce_only_after_settle():
    """While the knob is turning nothing is spoken; 'Volume 7' comes once it rests 300 ms."""
    f = VolumeFilter(VolumeCurve(CAL), VolumeConfig())
    t = 0.0
    first, ann = f.add_sample(raw_for(0.3), t)
    assert first == 3 and ann is None          # boot: apply, don't announce
    anns = []
    for pos in (0.4, 0.5, 0.6, 0.7):           # turning the dial
        for _ in range(4):
            t += 0.05
            _, a = f.add_sample(raw_for(pos), t)
            anns.append(a)
    assert [a for a in anns if a is not None] == []
    for _ in range(8):                          # now held still for 400 ms
        t += 0.05
        _, a = f.add_sample(raw_for(0.7), t)
        anns.append(a)
    assert [a for a in anns if a is not None] == [7]


def test_gain_curve():
    """Level 0 is mute, level 10 is full volume, and each step up is the same number of dB."""
    assert level_to_gain(0, -30, 0) == 0.0
    assert level_to_gain(10, -30, 0) == pytest.approx(1.0)
    # Level 1 is -30 dB; decibels to gain multiplier: 10 ** (dB / 20).
    assert level_to_gain(1, -30, 0) == pytest.approx(10 ** (-30 / 20))
    gains = [level_to_gain(i, -30, 0) for i in range(1, 11)]
    ratios = [b / a for a, b in zip(gains, gains[1:])]
    assert max(ratios) == pytest.approx(min(ratios))  # even steps in dB


def test_step_volume_settles_before_announcing():
    """Button / keyboard volume: steps apply at once, the announcement waits for a pause."""
    v = StepVolume(VolumeConfig(initial_level=5))
    assert v.step(1, 0.0) == 6
    assert v.step(1, 0.1) == 7
    # poll(t) returns the level to announce once settle_ms have passed since the last step.
    assert v.poll(0.2) is None
    assert v.poll(0.41) == 7
    # Stepping past 10 stays at 10 and returns None (nothing changed).
    assert v.step(10, 1.0) == 10 and v.step(1, 1.1) is None
