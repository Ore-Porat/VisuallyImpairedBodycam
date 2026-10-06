"""Button debounce, short / long / hold presses and the shutdown combo (FR-40, PRD 6.4)."""

from slr.common.config import load_controls_config
from slr.device.controls import ButtonPanel

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
    while t < end:
        out += panel.update(name, True, t)
        t += dt
    for _ in range(10):
        out += panel.update(name, False, t)
        t += dt
    return out


def test_short_presses_map_to_events():
    p = ButtonPanel(CFG)
    assert press(p, "shutter", 0, 0.2) == ["enter"]
    assert press(p, "advance", 1, 0.2) == ["next_item"]
    assert press(p, "rewind", 2, 0.2) == ["prev_item"]


def test_debounce_filters_bounce_and_glitches():
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
    p = ButtonPanel(CFG)
    assert press(p, "advance", 0, 1.2) == ["next_item"]
    assert press(p, "advance", 5, 2.3) == ["help"]
    assert press(p, "rewind", 10, 2.3) == ["status"]


def test_shutdown_combo_suppresses_individual_actions():
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
    cfg = CFG.model_copy(deep=True)
    cfg.identify_triggers.shutter_long_press = False
    p = ButtonPanel(cfg)
    assert press(p, "shutter", 0, 1.0) == ["enter"]
