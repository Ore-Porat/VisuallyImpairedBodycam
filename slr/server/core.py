"""Server core: runs the pure guidance engine and turns its effects into protocol messages.

No sockets here. The live server (main.py) and replay (replay.py) share this class, so replay
regression tests exercise exactly the same logic as a real session.

WHAT THIS FILE DOES
-------------------
``ServerCore`` is a thin wrapper around the guidance engine in ``guidance.py``. The engine is
"pure": it only does math on its inputs and returns a new state plus a list of abstract
"effects" (say this phrase, play this tone, start/stop the camera stream). This file:

1. feeds each event into the engine (``handle``),
2. remembers the new state,
3. writes state changes to the session log and saves the shopping list if it changed,
4. converts each effect into a dictionary (a JSON message) that the device understands.

HOW IT FITS IN
--------------
Laptop server (``main.py``) -> ``ServerCore.handle(event)`` -> list of messages -> sent to the
Raspberry Pi over the WebSocket. ``replay.py`` uses the same class without any network.

WORDS YOU WILL SEE
------------------
- event: something that happened (a button press, a processed camera frame, or a clock tick).
- effect: something the engine wants done (speak, beep, stream on/off).
- cue: a spoken phrase sent to the device as a message of type "cue".
- dataclass: a Python class that is just a bundle of named fields (see ``guidance.py``).
- ``replace(obj, field=value)``: makes a COPY of a frozen dataclass with some fields changed.
- JSON: plain-text data format used for every message on the WebSocket.
"""

from __future__ import annotations

import logging
# ``replace`` copies a dataclass with a few fields changed (dataclasses are immutable here).
from dataclasses import replace

from slr.common.config import GuidanceConfig
from slr.common.protocol import PRIORITIES
from slr.server import guidance as g
from slr.server.clips import PhraseCatalog
from slr.server.session import ShoppingList
from slr.server.session_log import SessionLog

# A logger prints diagnostic messages to the terminal, tagged with this file's name.
log = logging.getLogger(__name__)


class ServerCore:
    """Holds the engine state and converts engine effects into messages for the device."""

    def __init__(self, gcfg: GuidanceConfig, catalog: PhraseCatalog, shopping: ShoppingList,
                 session_log: SessionLog | None = None):
        """Set up the core.

        gcfg:        the ``guidance:`` section of ``config/server.yaml`` (all the tunable numbers).
        catalog:     the phrase catalog that turns a phrase key + parameters into spoken text and
                     a clip file name.
        shopping:    the shopping list (loaded from ``data/shopping_list.yaml``).
        session_log: optional JSONL log; if ``None`` a disabled log is used.
        """
        self.gcfg = gcfg
        self.catalog = catalog
        self.shopping = shopping
        self.log = session_log or SessionLog(None)
        # The engine's state starts from the saved shopping list and current item index.
        self.state = g.initial_state(shopping.items, shopping.current)
        # Each spoken cue gets an increasing id number so the device can tell us when it started.
        self.cue_id = 0
        self.last_cue_text = ""
        self.cue_origin: dict[int, float] = {}  # cue_id -> capture time (server clock) of the trigger frame

    def handle(self, event: g.Event, now: float, origin_ts: float | None = None) -> list[dict]:
        """Feed one event into the engine and return the messages to send to the device.

        event:     a ``Control``, ``Frame`` or ``Tick`` from ``guidance.py``.
        now:       the current time in seconds (passed in so tests can fake the clock).
        origin_ts: for frame events, when that frame was captured; used to measure latency.
        Returns a list of message dicts (type "cue", "tone" or "stream").
        """
        prev = self.state
        # The engine returns a brand new state and a list of effects; the old state is untouched.
        self.state, effects = g.step(self.state, event, now, self.gcfg)
        # Log mode changes (standby -> idle -> searching -> ...) for later analysis.
        if self.state.mode != prev.mode:
            self.log.write("state", mode=self.state.mode.value, prev=prev.mode.value,
                           target=self.state.target)
        # If the engine checked an item off or moved to another item, save that to disk.
        if self.state.items != prev.items or self.state.index != prev.index:
            self.shopping.set_state(self.state.items, self.state.index)
        # A "list comprehension": build a new list by converting each effect into a message.
        return [self._message(e, origin_ts) for e in effects]

    def _message(self, effect: g.Effect, origin_ts: float | None) -> dict:
        """Convert one engine effect into the dictionary that becomes a JSON message."""
        # ``isinstance(x, Cls)`` asks "is x an object of this class?".
        if isinstance(effect, g.Say):
            # Resolve the phrase key + params into real words and a clip file name.
            r = self.catalog.resolve(effect)
            self.cue_id += 1
            if origin_ts is not None:
                # Remember when the frame that caused this cue was captured so that when the
                # device reports "cue started" we can compute camera-to-ear latency.
                self.cue_origin[self.cue_id] = origin_ts
                # Keep the dict from growing forever: drop the oldest entry past 500.
                if len(self.cue_origin) > 500:
                    self.cue_origin.pop(next(iter(self.cue_origin)))
            self.last_cue_text = r.text
            return {"type": "cue", "cue_id": self.cue_id, "clip": r.clip, "priority": PRIORITIES[r.priority],
                    "interrupt": effect.interrupt, "text": r.text, "key": r.key}
        if isinstance(effect, g.Tone):
            # The guidance beep: pitch, length, left/right position and how often to repeat.
            return {"type": "tone", "freq_hz": effect.freq_hz, "duration_ms": effect.duration_ms,
                    "pan": effect.pan, "interval_ms": effect.interval_ms}
        if isinstance(effect, g.Stream):
            # Tell the device to start or stop sending camera frames.
            msg = {"type": "stream", "on": effect.on, "max_fps": self.gcfg.stream_max_fps,
                   "purpose": effect.purpose}
            if effect.burst is not None:
                # Identify mode asks for a fixed number of frames (a "burst"), then stops.
                msg["burst"] = effect.burst
                msg["burst_id"] = effect.burst_id
            return msg
        # Should never happen; raising makes a programming error obvious instead of silent.
        raise TypeError(f"unknown effect {effect!r}")

    def reload_list(self) -> None:
        """Pick up list edits from the CLI or the web editor."""
        self.state = g.with_items(self.state, tuple(self.shopping.items), self.shopping.current)

    def device_disconnected(self) -> None:
        """Quietly stop everything; the device re-sends its switch position when it reconnects."""
        # Go back to standby and forget any search/identify in progress.
        self.state = replace(self.state, mode=g.Mode.STANDBY, target=None, track=g.Tracking(),
                             identify=None, other_history=(), search_ref=None)
