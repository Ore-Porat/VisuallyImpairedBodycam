"""Physical controls -> abstract events (PRD 6.4, FR-40..43).

``ButtonPanel`` and ``ModeSwitchDecoder`` are pure: feed them raw readings with timestamps and
they return event names. The GPIO backend polls the pins and feeds them; the keyboard backend
(sim mode) maps keys straight to events.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

from slr.common.config import ControlsConfig, IdentifyTriggers, ModeSwitchConfig

log = logging.getLogger(__name__)

Emit = Callable[[str], None]


# --------------------------------------------------------------------------- buttons


@dataclass
class _Button:
    raw: bool = False
    raw_since: float = 0.0
    pressed: bool = False
    pressed_at: float = 0.0
    fired: bool = False
    suppressed: bool = False


class ButtonPanel:
    """Debounce + short / long / hold classification + the shutdown combo (FR-40).

    - A raw change must stay put for button_debounce_ms before it counts.
    - Released before long_press_s -> short action; released later -> long action.
    - Held for hold_s -> hold action fires while still held (so the user knows to let go).
      If a button's long and hold actions are the same, it fires at long_press_s instead.
    - All shutdown_combo buttons held for shutdown_hold_s -> "shutdown"; their own actions are
      suppressed as soon as the combo starts.
    """

    def __init__(self, cfg: ControlsConfig):
        self.cfg = cfg
        self.debounce = cfg.button_debounce_ms / 1000
        self.buttons = {name: _Button() for name in cfg.buttons}
        self.combo = list(cfg.shutdown_combo)
        self.combo_since: float | None = None
        self.combo_fired = False
        self.actions = {name: self._resolve(name) for name in cfg.buttons}

    def _resolve(self, name: str) -> tuple[str | None, str | None, str | None]:
        acts = self.cfg.actions.get(name)
        short = acts.short if acts else None
        long = (acts.long if acts else None) or short
        hold = (acts.hold if acts else None) or long
        if not self.cfg.identify_triggers.shutter_long_press:
            long = short if long == "identify" else long
            hold = short if hold == "identify" else hold
        return short, long, hold

    def update(self, name: str, raw: bool, t: float) -> list[str]:
        b = self.buttons[name]
        if raw != b.raw:
            b.raw, b.raw_since = raw, t
        return self.tick(t)

    def tick(self, t: float) -> list[str]:
        out: list[str] = []
        for name, b in self.buttons.items():
            if b.raw != b.pressed and t - b.raw_since >= self.debounce:
                out += self._press(name, b, t) if b.raw else self._release(name, b, t)
        out += self._holds(t)
        return out

    def _press(self, name: str, b: _Button, t: float) -> list[str]:
        b.pressed, b.pressed_at, b.fired, b.suppressed = True, t, False, False
        if self.combo and all(self.buttons[c].pressed for c in self.combo):
            self.combo_since, self.combo_fired = t, False
            for c in self.combo:
                self.buttons[c].suppressed = True
        return []

    def _release(self, name: str, b: _Button, t: float) -> list[str]:
        b.pressed = False
        if name in self.combo:
            self.combo_since = None
        if b.fired or b.suppressed:
            return []
        short, long, hold = self.actions[name]
        d = t - b.pressed_at
        ev = short if d < self.cfg.long_press_s else (long if d < self.cfg.hold_s else hold)
        return [ev] if ev else []

    def _holds(self, t: float) -> list[str]:
        out = []
        if self.combo_since is not None and not self.combo_fired \
                and t - self.combo_since >= self.cfg.shutdown_hold_s:
            self.combo_fired = True
            out.append("shutdown")
        for name, b in self.buttons.items():
            if not b.pressed or b.fired or b.suppressed:
                continue
            _, long, hold = self.actions[name]
            d = t - b.pressed_at
            if hold and hold != long and d >= self.cfg.hold_s:
                b.fired = True
                out.append(hold)
            elif long and long == hold and long != self.actions[name][0] and d >= self.cfg.long_press_s:
                b.fired = True
                out.append(long)
        return out


# --------------------------------------------------------------------------- mode switch


class ModeSwitchDecoder:
    """3-position switch on wires a and b (FR-42, FR-43).

    A position must be stable for debounce_ms. The invalid state (a=1, b=0 by default) is
    ignored and logged. The first stable position after boot produces the boot event.
    """

    def __init__(self, cfg: ModeSwitchConfig, triggers: IdentifyTriggers):
        self.cfg = cfg
        self.triggers = triggers
        self.debounce = cfg.debounce_ms / 1000
        self.position: str | None = None
        self.candidate: str | None = None
        self.candidate_since = 0.0
        self.invalid_count = 0
        self._last_invalid_logged = -1.0

    def decode(self, a: int, b: int) -> str | None:
        if self.cfg.invert:
            a, b = 1 - a, 1 - b
        return self.cfg.positions.get(f"{a}{b}")

    def update(self, a: int, b: int, t: float) -> list[str]:
        pos = self.decode(a, b)
        if pos is None:
            self.invalid_count += 1
            if t - self._last_invalid_logged > 1.0:
                log.info("mode switch invalid state a=%d b=%d ignored (mid-movement?)", a, b)
                self._last_invalid_logged = t
            return []
        if pos != self.candidate:
            self.candidate, self.candidate_since = pos, t
        if pos == self.position or t - self.candidate_since < self.debounce:
            return []
        prev, self.position = self.position, pos
        if prev is None:
            return ["mode_off"] if pos == "off" else ["mode_on"]
        return self._transition(prev, pos)

    def _transition(self, prev: str, pos: str) -> list[str]:
        if pos == "off":
            return ["mode_off"]
        if pos == "on":
            return ["mode_on"]  # the server ignores this if it is already active
        # toggle
        out = ["mode_on"] if prev == "off" else []
        if self.triggers.mode_toggle:
            out.append("identify")
        return out

    def current_events(self) -> list[str]:
        """Events that describe the current position (sent again after every reconnect)."""
        if self.position is None:
            return []
        return ["mode_off"] if self.position == "off" else ["mode_on"]


# --------------------------------------------------------------------------- backends


class GpioControls:
    """Polls buttons and mode switch pins with gpiozero and feeds the pure decoders."""

    def __init__(self, cfg: ControlsConfig, emit: Emit, volume_buttons: bool = False):
        from gpiozero import DigitalInputDevice

        self.cfg = cfg
        self.emit = emit
        pull = cfg.button_pull_up
        self.pins = {name: DigitalInputDevice(pin, pull_up=pull) for name, pin in cfg.buttons.items()}
        ms = cfg.mode_switch
        self.wire_a = DigitalInputDevice(ms.pin_a, pull_up=True)
        self.wire_b = DigitalInputDevice(ms.pin_b, pull_up=True)
        self.vol_pins = {}
        if volume_buttons:
            self.vol_pins = {"volume_up": DigitalInputDevice(cfg.volume_buttons.up, pull_up=pull),
                             "volume_down": DigitalInputDevice(cfg.volume_buttons.down, pull_up=pull)}
        self.panel = ButtonPanel(cfg)
        self.switch = ModeSwitchDecoder(ms, cfg.identify_triggers)
        self._vol_prev = {k: False for k in self.vol_pins}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, name="gpio-poll", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def read(self) -> dict:
        """Raw readings for `slr device test-controls`."""
        return {"buttons": {n: d.value for n, d in self.pins.items()},
                "a": self.wire_a.value, "b": self.wire_b.value,
                "position": self.switch.decode(self.wire_a.value, self.wire_b.value)}

    def _poll(self) -> None:
        period = 1 / self.cfg.poll_hz
        while not self._stop.is_set():
            t = time.monotonic()
            events = []
            for name, dev in self.pins.items():
                events += self.panel.update(name, bool(dev.value), t)
            events += self.switch.update(self.wire_a.value, self.wire_b.value, t)
            for name, dev in self.vol_pins.items():
                v = bool(dev.value)
                if v and not self._vol_prev[name]:
                    events.append(name)
                self._vol_prev[name] = v
            for ev in events:
                self.emit(ev)
            time.sleep(period)

    def current_events(self) -> list[str]:
        return self.switch.current_events()


class KeyboardControls:
    """Sim mode: keys from the terminal (and the debug window, via inject) -> events."""

    def __init__(self, cfg: ControlsConfig, emit: Emit):
        self.emit = emit
        self.keymap: dict[str, str] = {}
        for event, keys in cfg.keyboard.items():
            for k in keys:
                self.keymap[k.lower()] = event
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_terminal, name="keyboard", daemon=True)

    def start(self) -> None:
        if sys.stdin is not None and sys.stdin.isatty():
            self._thread.start()
        else:
            log.info("stdin is not a terminal; use the debug window for keys")

    def stop(self) -> None:
        self._stop.set()

    def inject(self, key: str) -> None:
        ev = self.keymap.get(key.lower())
        if ev:
            self.emit(ev)

    def current_events(self) -> list[str]:
        return ["mode_on"]  # no physical switch in sim: always "on"

    def help_text(self) -> str:
        by_event: dict[str, list[str]] = {}
        for k, ev in self.keymap.items():
            by_event.setdefault(ev, []).append(k)
        return "  ".join(f"{'/'.join(ks)}={ev}" for ev, ks in by_event.items())

    def _read_terminal(self) -> None:
        if sys.platform == "win32":
            self._read_windows()
        else:
            self._read_posix()

    def _read_windows(self) -> None:
        import msvcrt

        while not self._stop.is_set():
            if not msvcrt.kbhit():
                time.sleep(0.02)
                continue
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                code = msvcrt.getwch()
                key = {"K": "left", "M": "right"}.get(code)
            elif ch == " ":
                key = "space"
            elif ch == "\x03":
                import _thread

                _thread.interrupt_main()
                return
            else:
                key = ch
            if key:
                self.inject(key)

    def _read_posix(self) -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    seq = sys.stdin.read(2)
                    key = {"[D": "left", "[C": "right"}.get(seq)
                elif ch == " ":
                    key = "space"
                else:
                    key = ch
                if key:
                    self.inject(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model", encoding="utf-8", errors="ignore") as f:
            return "raspberry pi" in f.read().lower()
    except OSError:
        return False


def make_controls(cfg: ControlsConfig, emit: Emit, sim: bool, volume_buttons: bool = False):
    backend = cfg.backend
    if backend == "auto":
        backend = "keyboard" if sim or not is_raspberry_pi() else "gpio"
    if backend == "gpio":
        return GpioControls(cfg, emit, volume_buttons)
    return KeyboardControls(cfg, emit)
