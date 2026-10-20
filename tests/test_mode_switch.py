"""Mode switch decoding: truth table, debounce, invalid state (FR-42, FR-43).

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    The camera's mode switch is a three-position lever wired to two GPIO pins, a and b.
    Each pin reads 0 or 1, so the pair forms a two-character code ("00", "11", "01").
    config/controls.yaml maps those codes to positions: OFF, ON and TOGGLE (a
    spring-loaded position used to trigger Identify). ``ModeSwitchDecoder`` (in
    ``slr/device/controls.py``) turns a stream of pin readings into events:
    "mode_on", "mode_off" and "identify". It must also ignore contact flicker
    (debounce, 80 ms) and the brief "10" state seen while the lever moves between
    positions. FR-42 and FR-43 are requirement numbers from the product spec, PRD.pdf.

    No hardware: tests call ``dec.update(a, b, time)`` by hand with made-up timestamps.

Words you will see:
    truth table   The list of pin combinations and the position each one means.
    parametrize   Runs the same test function once per listed row of inputs.
    assert        "Check that this is true; if not, the test fails."
"""

import pytest

from slr.common.config import IdentifyTriggers, ModeSwitchConfig
from slr.device.controls import ModeSwitchDecoder


def make(**kw):
    """Build a decoder with default settings, overriding any ModeSwitchConfig field via kw."""
    # "triggers" is pulled out of kw because it belongs to a different config object.
    trig = kw.pop("triggers", IdentifyTriggers())
    return ModeSwitchDecoder(ModeSwitchConfig(**kw), trig)


def settle(dec, a, b, t0, dt=0.01, dur=0.1):
    """Hold the pins at (a, b) from time t0 for dur seconds, sampling every dt.

    Returns (events, end_time). 0.1 s is longer than the 80 ms debounce, so the decoder
    has time to accept the new position.
    """
    out, t = [], t0
    while t <= t0 + dur + 1e-9:
        out += dec.update(a, b, t)
        t += dt
    return out, t


@pytest.mark.parametrize("a,b,pos", [(0, 0, "off"), (1, 1, "on"), (0, 1, "toggle"), (1, 0, None)])
def test_truth_table(a, b, pos):
    """Each pin pair decodes to the configured position; "10" is not a valid position."""
    assert make().decode(a, b) == pos


def test_invert_flag():
    """invert: true swaps 0 and 1 for wiring where 'closed' reads as 0."""
    dec = make(invert=True)
    assert dec.decode(1, 1) == "off"
    assert dec.decode(0, 0) == "on"
    assert dec.decode(1, 0) == "toggle"


def test_boot_acts_on_current_position():
    """At startup the decoder reports whatever position the switch is already in."""
    dec = make()
    out, _ = settle(dec, 1, 1, 0.0)
    assert out == ["mode_on"]
    dec = make()
    out, _ = settle(dec, 0, 0, 0.0)
    assert out == ["mode_off"]


def test_debounce_requires_stable_position():
    """A short blip is ignored; a real move counts only after 80 ms in the new position."""
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
    """The unused "10" combination never produces an event; it is only counted."""
    dec = make()
    settle(dec, 1, 1, 0.0)
    out, t = settle(dec, 1, 0, 1.0, dur=1.0)
    assert out == []
    assert dec.position == "on"
    assert dec.invalid_count > 0


def test_invalid_state_mid_movement_does_not_break_transition():
    """Passing through "10" on the way from ON to OFF still ends with one 'mode_off'."""
    dec = make()
    settle(dec, 1, 1, 0.0)
    events = dec.update(1, 0, 1.0) + dec.update(1, 0, 1.03)   # passing through the invalid state
    events += settle(dec, 0, 0, 1.04)[0]
    assert events == ["mode_off"]


def test_on_to_toggle_identifies_and_back_resumes():
    """ON -> TOGGLE fires 'identify'; springing back to ON fires 'mode_on' (resume)."""
    dec = make()
    settle(dec, 1, 1, 0.0)
    out, t = settle(dec, 0, 1, 1.0)
    assert out == ["identify"]
    out, _ = settle(dec, 1, 1, t)
    assert out == ["mode_on"]


def test_off_to_toggle_turns_on_first():
    """OFF -> TOGGLE (skipping ON) turns the device on and then identifies."""
    dec = make()
    settle(dec, 0, 0, 0.0)
    out, _ = settle(dec, 0, 1, 1.0)
    assert out == ["mode_on", "identify"]


def test_toggle_trigger_can_be_disabled():
    """With identify_triggers.mode_toggle off, TOGGLE does nothing."""
    dec = make(triggers=IdentifyTriggers(mode_toggle=False))
    settle(dec, 1, 1, 0.0)
    out, _ = settle(dec, 0, 1, 1.0)
    assert out == []


def test_current_events_for_reconnect():
    """After a reconnect the device re-sends its switch position so the laptop can catch up."""
    dec = make()
    assert dec.current_events() == []
    settle(dec, 0, 1, 0.0)
    assert dec.current_events() == ["mode_on"]


def test_custom_truth_table():
    """A different wiring can be described by changing the positions table in the config."""
    dec = make(positions={"10": "off", "00": "on", "01": "toggle"})
    assert dec.decode(1, 0) == "off"
    assert dec.decode(1, 1) is None
