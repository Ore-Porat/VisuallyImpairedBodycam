"""Mode switch decoding: truth table, debounce, invalid state (FR-42, FR-43)."""

import pytest

from slr.common.config import IdentifyTriggers, ModeSwitchConfig
from slr.device.controls import ModeSwitchDecoder


def make(**kw):
    trig = kw.pop("triggers", IdentifyTriggers())
    return ModeSwitchDecoder(ModeSwitchConfig(**kw), trig)


def settle(dec, a, b, t0, dt=0.01, dur=0.1):
    out, t = [], t0
    while t <= t0 + dur + 1e-9:
        out += dec.update(a, b, t)
        t += dt
    return out, t


@pytest.mark.parametrize("a,b,pos", [(0, 0, "off"), (1, 1, "on"), (0, 1, "toggle"), (1, 0, None)])
def test_truth_table(a, b, pos):
    assert make().decode(a, b) == pos


def test_invert_flag():
    dec = make(invert=True)
    assert dec.decode(1, 1) == "off"
    assert dec.decode(0, 0) == "on"
    assert dec.decode(1, 0) == "toggle"


def test_boot_acts_on_current_position():
    dec = make()
    out, _ = settle(dec, 1, 1, 0.0)
    assert out == ["mode_on"]
    dec = make()
    out, _ = settle(dec, 0, 0, 0.0)
    assert out == ["mode_off"]


def test_debounce_requires_stable_position():
    dec = make()
    settle(dec, 1, 1, 0.0)
    # a 50 ms blip to OFF is ignored (debounce 80 ms)
    assert dec.update(0, 0, 1.00) == []
    assert dec.update(0, 0, 1.05) == []
    assert dec.update(1, 1, 1.06) == []
    assert dec.update(1, 1, 1.20) == []
    assert dec.position == "on"
    # a real move counts once it has been stable for 80 ms
    assert dec.update(0, 0, 2.00) == []
    assert dec.update(0, 0, 2.079) == []
    assert dec.update(0, 0, 2.08) == ["mode_off"]
    assert dec.update(0, 0, 2.5) == []


def test_invalid_state_ignored():
    dec = make()
    settle(dec, 1, 1, 0.0)
    out, t = settle(dec, 1, 0, 1.0, dur=1.0)
    assert out == []
    assert dec.position == "on"
    assert dec.invalid_count > 0


def test_invalid_state_mid_movement_does_not_break_transition():
    dec = make()
    settle(dec, 1, 1, 0.0)
    events = dec.update(1, 0, 1.0) + dec.update(1, 0, 1.03)   # passing through the invalid state
    events += settle(dec, 0, 0, 1.04)[0]
    assert events == ["mode_off"]


def test_on_to_toggle_identifies_and_back_resumes():
    dec = make()
    settle(dec, 1, 1, 0.0)
    out, t = settle(dec, 0, 1, 1.0)
    assert out == ["identify"]
    out, _ = settle(dec, 1, 1, t)
    assert out == ["mode_on"]


def test_off_to_toggle_turns_on_first():
    dec = make()
    settle(dec, 0, 0, 0.0)
    out, _ = settle(dec, 0, 1, 1.0)
    assert out == ["mode_on", "identify"]


def test_toggle_trigger_can_be_disabled():
    dec = make(triggers=IdentifyTriggers(mode_toggle=False))
    settle(dec, 1, 1, 0.0)
    out, _ = settle(dec, 0, 1, 1.0)
    assert out == []


def test_current_events_for_reconnect():
    dec = make()
    assert dec.current_events() == []
    settle(dec, 0, 1, 0.0)
    assert dec.current_events() == ["mode_on"]


def test_custom_truth_table():
    dec = make(positions={"10": "off", "00": "on", "01": "toggle"})
    assert dec.decode(1, 0) == "off"
    assert dec.decode(1, 1) is None
