"""Physical controls -> abstract events (PRD 6.4, FR-40..43).

WHAT THIS FILE DOES
-------------------
The camera body has three buttons (shutter release, film advance lever, rewind knob) and a
three-position mode switch. This file turns their raw electrical readings into named EVENTS
such as "enter", "next_item", "identify", "toggle_found", "help", "mode_on" or "shutdown".
The rest of the
device never sees pins or voltages, only event names. (FR-40..43 are requirement numbers from
the product spec, PRD.pdf.)

``ButtonPanel`` and ``ModeSwitchDecoder`` are pure: feed them raw readings with timestamps and
they return event names. "Pure" means no hardware, no clock, no side effects, which makes them
easy to unit test. The GPIO backend polls the pins and feeds them; the keyboard backend
(sim mode) maps keys straight to events.

HOW IT FITS IN
--------------
``slr/device/main.py`` calls ``make_controls`` with an ``emit`` function. Every time a control
produces an event, ``emit(event_name)`` is called, and main.py forwards the event to the laptop
server (which decides what it means in the current mode).

WORDS YOU WILL SEE
------------------
- GPIO: "general purpose input/output", the numbered pins on the Raspberry Pi that can read a
  button or drive an LED. Pin numbers come from config/controls.yaml.
- pull-up resistor: a button wired between a pin and ground reads 1 when released (the pin is
  "pulled up" to 3.3 V internally) and 0 when pressed. gpiozero hides that: ``value`` is True
  when pressed.
- debounce: a mechanical contact "bounces" (flickers on/off for a few milliseconds) when pressed.
  Debouncing means ignoring a change until it has stayed steady for a short time.
- poll / polling: checking the pins over and over at a fixed rate (100 times a second here).
- short / long / hold press: released quickly, released after 0.8 s, or held for 2 s.
- combo: holding two particular buttons together for 5 s shuts the device down safely.
- dataclass: a Python shortcut for a class that is mostly a bundle of named fields.
- thread: a second line of execution running alongside the main program.
- Callable: a type hint meaning "a function". ``Emit = Callable[[str], None]`` reads "a function
  that takes one string and returns nothing".
- sim mode: running on a laptop with the keyboard standing in for the buttons.
"""

# Allows modern type hints like ``str | None`` on older Python versions.
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

from slr.common.config import ControlsConfig, IdentifyTriggers, ModeSwitchConfig

log = logging.getLogger(__name__)

# The type of the "emit" function every backend is given: it takes an event name, returns nothing.
Emit = Callable[[str], None]


# --------------------------------------------------------------------------- buttons


# ``@dataclass`` auto-generates __init__ etc. from the field list. The leading underscore in the
# name is a Python convention for "internal to this file".
@dataclass
class _Button:
    """Bookkeeping for one physical button.

    - raw / raw_since: the latest electrical reading and when it last changed (for debouncing).
    - pressed / pressed_at: the debounced state and when the press began.
    - fired: True once a hold action has already been emitted for this press.
    - suppressed: True when this press is part of the shutdown combo (its own action is dropped).
    """

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

    Which event each button produces for short / long / hold comes from the ``actions`` table in
    config/controls.yaml.
    """

    def __init__(self, cfg: ControlsConfig):
        """``cfg`` is the parsed config/controls.yaml."""
        self.cfg = cfg
        self.debounce = cfg.button_debounce_ms / 1000
        # One _Button record per button name ("shutter", "advance", "rewind"). This is a
        # "dict comprehension": build a dictionary from a loop in one line.
        self.buttons = {name: _Button() for name in cfg.buttons}
        self.combo = list(cfg.shutdown_combo)
        self.combo_since: float | None = None   # when all combo buttons became pressed
        self.combo_fired = False
        # For each button: the (short, long, hold) event names.
        self.actions = {name: self._resolve(name) for name in cfg.buttons}

    def _resolve(self, name: str) -> tuple[str | None, str | None, str | None]:
        """Work out the (short, long, hold) events for one button, filling gaps sensibly.

        A missing long action falls back to short, a missing hold falls back to long. If the
        config says the shutter long press should NOT trigger identify, that mapping is
        replaced by the short action.
        """
        acts = self.cfg.actions.get(name)
        short = acts.short if acts else None
        long = (acts.long if acts else None) or short
        hold = (acts.hold if acts else None) or long
        if not self.cfg.identify_triggers.shutter_long_press:
            long = short if long == "identify" else long
            hold = short if hold == "identify" else hold
        return short, long, hold

    def update(self, name: str, raw: bool, t: float) -> list[str]:
        """Feed one raw reading (True = pressed) for button ``name`` at time ``t`` seconds.

        Returns the list of events produced right now (often empty).
        """
        b = self.buttons[name]
        # Note the moment the raw reading changed; debouncing is judged from that moment.
        if raw != b.raw:
            b.raw, b.raw_since = raw, t
        return self.tick(t)

    def tick(self, t: float) -> list[str]:
        """Advance time to ``t`` without a new reading: apply debounced changes and check holds.

        Returns the events produced.
        """
        out: list[str] = []
        for name, b in self.buttons.items():
            # The raw reading differs from the accepted state and has been steady long enough.
            if b.raw != b.pressed and t - b.raw_since >= self.debounce:
                out += self._press(name, b, t) if b.raw else self._release(name, b, t)
        out += self._holds(t)
        return out

    def _press(self, name: str, b: _Button, t: float) -> list[str]:
        """A debounced press began. Starts the shutdown combo timer if all its buttons are down."""
        b.pressed, b.pressed_at, b.fired, b.suppressed = True, t, False, False
        # ``all(...)`` is True only if every combo button is currently pressed.
        if self.combo and all(self.buttons[c].pressed for c in self.combo):
            self.combo_since, self.combo_fired = t, False
            for c in self.combo:
                self.buttons[c].suppressed = True
        return []

    def _release(self, name: str, b: _Button, t: float) -> list[str]:
        """A debounced release: classify the press by how long it lasted and emit its event."""
        b.pressed = False
        if name in self.combo:
            self.combo_since = None
        # A hold action already fired, or the press was part of the combo: nothing more to say.
        if b.fired or b.suppressed:
            return []
        short, long, hold = self.actions[name]
        d = t - b.pressed_at
        # Pick short / long / hold by duration (a nested conditional expression).
        ev = short if d < self.cfg.long_press_s else (long if d < self.cfg.hold_s else hold)
        return [ev] if ev else []

    def _holds(self, t: float) -> list[str]:
        """Emit events that fire WHILE a button is still held: the shutdown combo and hold actions."""
        out = []
        # Shutdown combo held long enough and not yet reported.
        if self.combo_since is not None and not self.combo_fired \
                and t - self.combo_since >= self.cfg.shutdown_hold_s:
            self.combo_fired = True
            out.append("shutdown")
        for name, b in self.buttons.items():
            if not b.pressed or b.fired or b.suppressed:
                continue
            _, long, hold = self.actions[name]
            d = t - b.pressed_at
            # Normal case: a distinct hold action fires once hold_s has elapsed.
            if hold and hold != long and d >= self.cfg.hold_s:
                b.fired = True
                out.append(hold)
            # Special case: long and hold are the same action (and differ from short), so fire it
            # as soon as long_press_s elapses rather than making the user wait for hold_s.
            elif long and long == hold and long != self.actions[name][0] and d >= self.cfg.long_press_s:
                b.fired = True
                out.append(long)
        return out


# --------------------------------------------------------------------------- mode switch


class ModeSwitchDecoder:
    """3-position switch on wires a and b (FR-42, FR-43).

    The switch has two contact wires. Reading both gives a two-digit code ("00", "11", "01")
    which config/controls.yaml maps to a position name: "off", "on" or "toggle".

    A position must be stable for debounce_ms. The invalid state (a=1, b=0 by default) is
    ignored and logged. The first stable position after boot produces the boot event.
    """

    def __init__(self, cfg: ModeSwitchConfig, triggers: IdentifyTriggers):
        """``cfg`` is the mode_switch section, ``triggers`` the identify_triggers section."""
        self.cfg = cfg
        self.triggers = triggers
        self.debounce = cfg.debounce_ms / 1000
        self.position: str | None = None     # the accepted (debounced) position
        self.candidate: str | None = None    # the position we are currently seeing
        self.candidate_since = 0.0           # since when
        self.invalid_count = 0
        self._last_invalid_logged = -1.0
        # TOGGLE is both a button and a momentary switch position:
        #   short flick -> toggle_found
        #   held >= toggle_hold_s -> identify
        # We remember when TOGGLE was entered so we can decide on release.
        self.toggle_since: float | None = None
        self.toggle_origin: str | None = None   # position we came from (on/off)
        self.toggle_found_sent = False
        self.toggle_identify_sent = False

    def decode(self, a: int, b: int) -> str | None:
        """Translate the two wire readings into a position name, or None if the code is unknown."""
        if self.cfg.invert:
            a, b = 1 - a, 1 - b
        return self.cfg.positions.get(f"{a}{b}")

    def update(self, a: int, b: int, t: float) -> list[str]:
        """Feed one reading of both wires at time ``t``. Returns the events produced (if any)."""
        pos = self.decode(a, b)
        if pos is None:
            # Usually the brief moment while the switch is moving between positions. Log at most
            # once a second so the log is not flooded.
            self.invalid_count += 1
            if t - self._last_invalid_logged > 1.0:
                log.info("mode switch invalid state a=%d b=%d ignored (mid-movement?)", a, b)
                self._last_invalid_logged = t
            return []
        # A new candidate position: start its stability timer.
        if pos != self.candidate:
            self.candidate, self.candidate_since = pos, t
        # If the position has not changed, the only possible event is "held TOGGLE".
        if pos == self.position:
            return self._toggle_hold(t, pos)
        # Not steady long enough yet.
        if t - self.candidate_since < self.debounce:
            return []
        prev, self.position = self.position, pos
        # First position after boot: report it plainly. A boot in TOGGLE waits for release so
        # it does not accidentally toggle a shopping-list item at power-on.
        if prev is None:
            if pos == "toggle":
                self._begin_toggle(None, t)
                # Do not turn a switch held at power-on into a list edit or Identify.
                self.toggle_found_sent = True
                self.toggle_identify_sent = True
                return []
            return ["mode_off"] if pos == "off" else ["mode_on"]
        # Entering TOGGLE: wait to see whether it is a short flick or a hold.
        if pos == "toggle":
            return self._enter_toggle(prev, t)
        # Leaving TOGGLE: a short flick means "toggle the current item".
        if prev == "toggle":
            return self._leave_toggle(pos)
        return self._transition(prev, pos)

    def _begin_toggle(self, origin: str | None, t: float) -> None:
        """Start measuring how long TOGGLE is held."""
        self.toggle_since = t
        self.toggle_origin = origin
        self.toggle_found_sent = False
        self.toggle_identify_sent = False

    def _enter_toggle(self, origin: str, t: float) -> list[str]:
        """TOGGLE was just accepted. A latching switch is a found-toggle immediately; a
        spring-loaded one waits for release so a hold can become Identify."""
        self._begin_toggle(origin, t)
        if self.cfg.toggle_latching:
            self.toggle_found_sent = True
            return ["toggle_found"]
        return []

    def _toggle_hold(self, t: float, pos: str) -> list[str]:
        """While TOGGLE is held, emit Identify once after ``toggle_hold_s``."""
        if (pos != "toggle" or self.toggle_since is None or self.cfg.toggle_latching
                or self.toggle_identify_sent or not self.triggers.mode_toggle):
            return []
        if t - self.toggle_since < self.cfg.toggle_hold_s:
            return []
        self.toggle_identify_sent = True
        self.toggle_found_sent = True   # a hold is Identify, not a found-toggle
        return ["identify"]

    def _leave_toggle(self, pos: str) -> list[str]:
        """TOGGLE returned to a normal position: short flick -> toggle_found."""
        out: list[str] = []
        if not self.toggle_found_sent:
            out.append("toggle_found")
            self.toggle_found_sent = True
        origin = self.toggle_origin
        # Springing back to where it came from is not a mode change. This also stops an
        # OFF -> TOGGLE -> OFF flick from briefly turning the device on and off.
        if origin is None or pos != origin:
            out += self._transition("toggle", pos)
        self.toggle_since = None
        self.toggle_origin = None
        self.toggle_identify_sent = False
        return out

    def _transition(self, prev: str, pos: str) -> list[str]:
        """Events for moving between normal positions ``prev`` and ``pos``."""
        if pos == "off":
            return ["mode_off"]
        if pos == "on":
            return ["mode_on"]  # the server ignores this if it is already active
        return []  # TOGGLE is handled in update()

    def current_events(self) -> list[str]:
        """Events that describe the current position (sent again after every reconnect)."""
        if self.position is None:
            return []
        return ["mode_off"] if self.position == "off" else ["mode_on"]


# --------------------------------------------------------------------------- backends


class GpioControls:
    """Polls buttons and mode switch pins with gpiozero and feeds the pure decoders.

    This is the real-hardware backend. A background thread reads every pin 100 times a second
    (``poll_hz``), hands the readings to ``ButtonPanel`` / ``ModeSwitchDecoder``, and calls
    ``emit`` for each event they return.
    """

    def __init__(self, cfg: ControlsConfig, emit: Emit, volume_buttons: bool = False):
        """
        - cfg: parsed config/controls.yaml (pin numbers, timings, actions).
        - emit: function to call with each event name.
        - volume_buttons: also watch the two fallback volume buttons (FR-46), used only when the
          volume dial could not be read.
        """
        # Imported here so laptops without gpiozero can still import this module.
        from gpiozero import DigitalInputDevice

        self.cfg = cfg
        self.emit = emit
        pull = cfg.button_pull_up
        # One input object per button, keyed by name. ``value`` is True when pressed.
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
        self._vol_prev = {k: False for k in self.vol_pins}   # last reading, to detect new presses
        # A threading.Event is a flag: ``stop()`` sets it and the poll loop sees it and exits.
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, name="gpio-poll", daemon=True)

    def start(self) -> None:
        """Begin polling on the background thread."""
        self._thread.start()

    def stop(self) -> None:
        """Ask the polling thread to exit."""
        self._stop.set()

    def read(self) -> dict:
        """Raw readings for `slr device test-controls`."""
        return {"buttons": {n: d.value for n, d in self.pins.items()},
                "a": self.wire_a.value, "b": self.wire_b.value,
                "position": self.switch.decode(self.wire_a.value, self.wire_b.value)}

    def _poll(self) -> None:
        """The polling loop: read all pins, decode, emit, sleep, repeat."""
        period = 1 / self.cfg.poll_hz
        while not self._stop.is_set():
            t = time.monotonic()
            events = []
            for name, dev in self.pins.items():
                events += self.panel.update(name, bool(dev.value), t)
            events += self.switch.update(self.wire_a.value, self.wire_b.value, t)
            # Fallback volume buttons: emit on the rising edge (not pressed -> pressed) only.
            for name, dev in self.vol_pins.items():
                v = bool(dev.value)
                if v and not self._vol_prev[name]:
                    events.append(name)
                self._vol_prev[name] = v
            for ev in events:
                self.emit(ev)
            time.sleep(period)

    def current_events(self) -> list[str]:
        """Events describing the mode switch position right now (re-sent on reconnect)."""
        return self.switch.current_events()


class KeyboardControls:
    """Sim mode: keys from the terminal (and the debug window, via inject) -> events.

    The ``keyboard:`` table in config/controls.yaml lists which keys stand for which event
    (e.g. space = enter, i = identify). A background thread reads single keystrokes from the
    terminal without waiting for Enter.
    """

    def __init__(self, cfg: ControlsConfig, emit: Emit):
        """Build the key -> event lookup table from the config."""
        self.emit = emit
        self.keymap: dict[str, str] = {}
        for event, keys in cfg.keyboard.items():
            for k in keys:
                self.keymap[k.lower()] = event
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_terminal, name="keyboard", daemon=True)

    def start(self) -> None:
        """Start reading the terminal, if there is a real one (not when output is redirected)."""
        if sys.stdin is not None and sys.stdin.isatty():
            self._thread.start()
        else:
            log.info("stdin is not a terminal; use the debug window for keys")

    def stop(self) -> None:
        """Ask the keyboard thread to exit."""
        self._stop.set()

    def inject(self, key: str) -> None:
        """Treat ``key`` as if it were pressed (used by the terminal reader and the debug window)."""
        ev = self.keymap.get(key.lower())
        if ev:
            self.emit(ev)

    def current_events(self) -> list[str]:
        """There is no physical switch in sim mode, so report the device as always ON."""
        return ["mode_on"]  # no physical switch in sim: always "on"

    def help_text(self) -> str:
        """A one-line summary like ``space=enter  i=identify ...`` printed at start-up."""
        by_event: dict[str, list[str]] = {}
        for k, ev in self.keymap.items():
            by_event.setdefault(ev, []).append(k)
        return "  ".join(f"{'/'.join(ks)}={ev}" for ev, ks in by_event.items())

    def _read_terminal(self) -> None:
        """Pick the keystroke reader for this operating system."""
        if sys.platform == "win32":
            self._read_windows()
        else:
            self._read_posix()

    def _read_windows(self) -> None:
        """Windows: use msvcrt to read keys one at a time. Arrow keys arrive as two characters."""
        import msvcrt

        while not self._stop.is_set():
            # kbhit() is True when a key is waiting; otherwise nap briefly and check again.
            if not msvcrt.kbhit():
                time.sleep(0.02)
                continue
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                # Special key prefix: the next character says which one (K = left, M = right).
                code = msvcrt.getwch()
                key = {"K": "left", "M": "right"}.get(code)
            elif ch == " ":
                key = "space"
            elif ch == "\x03":
                # Ctrl-C: make the main thread raise KeyboardInterrupt so the program quits normally.
                import _thread

                _thread.interrupt_main()
                return
            else:
                key = ch
            if key:
                self.inject(key)

    def _read_posix(self) -> None:
        """macOS / Linux: put the terminal in "cbreak" mode so keys arrive without Enter.

        Arrow keys arrive as an escape sequence (ESC [ D for left, ESC [ C for right). The
        terminal settings are restored in the ``finally`` block no matter how the loop ends.
        """
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                # Wait up to 0.1 s for input so the stop flag is checked regularly.
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
    """True when running on a Raspberry Pi (Linux exposes the board name in /proc)."""
    try:
        with open("/proc/device-tree/model", encoding="utf-8", errors="ignore") as f:
            return "raspberry pi" in f.read().lower()
    except OSError:
        return False


def make_controls(cfg: ControlsConfig, emit: Emit, sim: bool, volume_buttons: bool = False):
    """Build the right controls backend: GPIO on a real Pi, keyboard otherwise (or in sim mode).

    ``cfg.backend`` can force one ("gpio" / "keyboard"); "auto" picks based on the hardware.
    """
    backend = cfg.backend
    if backend == "auto":
        backend = "keyboard" if sim or not is_raspberry_pi() else "gpio"
    if backend == "gpio":
        return GpioControls(cfg, emit, volume_buttons)
    return KeyboardControls(cfg, emit)
