"""Button debounce, short / long / hold presses and the shutdown combo (FR-40, PRD 6.4).

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    The camera body has three physical buttons: the shutter release, the film advance
    lever and the rewind knob. ``ButtonPanel`` (in ``slr/device/controls.py``) turns raw
    "pressed / not pressed" readings into named events such as "enter", "next_item",
    "identify" or "shutdown", depending on how long a button is held.
    FR-40 is a requirement number from the product spec, PRD.pdf; "PRD 6.4" is a section.

    The timings come from config/controls.yaml:
      * debounce 50 ms: ignore contact "bounce" (a real switch flickers for a few ms)
      * short press  < 0.8 s  -> shutter = enter, advance = next, rewind = prev
      * long press   >= 0.8 s -> shutter = identify (fires while still held)
      * hold         >= 2.0 s -> advance = help, rewind = status
      * shutter + advance together for 5 s -> shutdown

    No real hardware is used. The tests call ``panel.update(name, pressed, time)`` by hand
    with made-up timestamps, so they run instantly and deterministically.

Words you will see:
    assert        "Check that this is true; if not, the test fails."
    debounce      Ignoring very fast on/off flicker from a mechanical switch.
"""

from slr.common.config import load_controls_config
from slr.device.controls import ButtonPanel

# The real wiring/timing config shipped with the project.
CFG = load_controls_config()


def press(panel, name, t0, dur, dt=0.01, bounce=False):
    """Simulate holding a button for dur seconds; returns events from press to release + settle."""
    out = []
    t = t0
    if bounce:  # contact bounce: a few fast toggles that must be ignored
        for i in range(4):
            out += panel.update(name, i % 2 == 0, t)
            t += 0.005
    end = t0 + dur
    # Report "pressed" every dt seconds (10 ms, like the real 100 Hz polling loop) ...
    while t < end:
        out += panel.update(name, True, t)
        t += dt
    # ... then "released" for a while so the debounce logic can settle.
    for _ in range(10):
        out += panel.update(name, False, t)
        t += dt
    return out


def test_short_presses_map_to_events():
    """A quick tap of each button gives its basic event."""
    p = ButtonPanel(CFG)
    assert press(p, "shutter", 0, 0.2) == ["enter"]
    assert press(p, "advance", 1, 0.2) == ["next_item"]
    assert press(p, "rewind", 2, 0.2) == ["prev_item"]


def test_debounce_filters_bounce_and_glitches():
    """Contact bounce still gives one event; a 20 ms glitch gives none."""
    p = ButtonPanel(CFG)
    assert press(p, "shutter", 0, 0.2, bounce=True) == ["enter"]
    # a 20 ms glitch is shorter than the 50 ms debounce: nothing
    out = []
    for i in range(3):
        out += p.update("shutter", True, 1 + i * 0.01)
    for i in range(10):
        out += p.update("shutter", False, 1.03 + i * 0.01)
    assert out == []


def test_shutter_long_press_identifies_while_held():
    """Holding the shutter 0.8 s fires 'identify' immediately, not on release."""
    p = ButtonPanel(CFG)
    out = []
    t = 0.0
    while t < 0.9:
        out += p.update("shutter", True, t)
        t += 0.01
    assert out == ["identify"]          # fired while still held, at 0.8 s
    for _ in range(10):
        out += p.update("shutter", False, t)
        t += 0.01
    assert out == ["identify"]          # release adds nothing


def test_advance_hold_gives_help_and_medium_press_is_next():
    """A 1.2 s press of advance is still 'next'; 2.3 s is 'help'. Rewind held = 'status'."""
    p = ButtonPanel(CFG)
    assert press(p, "advance", 0, 1.2) == ["next_item"]
    assert press(p, "advance", 5, 2.3) == ["help"]
    assert press(p, "rewind", 10, 2.3) == ["status"]


def test_shutdown_combo_suppresses_individual_actions():
    """Shutter + advance held 5 s gives only 'shutdown', not enter/identify/help."""
    p = ButtonPanel(CFG)
    out = []
    t = 0.0
    while t < 5.3:
        out += p.update("shutter", True, t)
        out += p.update("advance", True, t)
        t += 0.01
    for _ in range(10):
        out += p.update("shutter", False, t)
        out += p.update("advance", False, t)
        t += 0.01
    assert out == ["shutdown"]


def test_combo_released_early_does_nothing():
    """Letting go of the combo after 1 s is harmless: no shutdown and no stray events."""
    p = ButtonPanel(CFG)
    out = []
    t = 0.0
    while t < 1.0:
        out += p.update("shutter", True, t) + p.update("advance", True, t)
        t += 0.01
    for _ in range(10):
        out += p.update("shutter", False, t) + p.update("advance", False, t)
        t += 0.01
    assert out == []


def test_shutter_long_press_trigger_can_be_disabled():
    """With identify_triggers.shutter_long_press off, a 1 s shutter press is just 'enter'."""
    # model_copy(deep=True) makes an independent copy of the config so we can change one
    # setting without affecting the other tests that share CFG.
    cfg = CFG.model_copy(deep=True)
    cfg.identify_triggers.shutter_long_press = False
    p = ButtonPanel(cfg)
    assert press(p, "shutter", 0, 1.0) == ["enter"]
