"""Server core: runs the pure guidance engine and turns its effects into protocol messages.

No sockets here. The live server (main.py) and replay (replay.py) share this class, so replay
regression tests exercise exactly the same logic as a real session.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from slr.common.config import GuidanceConfig
from slr.common.protocol import PRIORITIES
from slr.server import guidance as g
from slr.server.clips import PhraseCatalog
from slr.server.session import ShoppingList
from slr.server.session_log import SessionLog

log = logging.getLogger(__name__)


class ServerCore:
    def __init__(self, gcfg: GuidanceConfig, catalog: PhraseCatalog, shopping: ShoppingList,
                 session_log: SessionLog | None = None):
        self.gcfg = gcfg
        self.catalog = catalog
        self.shopping = shopping
        self.log = session_log or SessionLog(None)
        self.state = g.initial_state(shopping.items, shopping.current)
        self.cue_id = 0
        self.last_cue_text = ""
        self.cue_origin: dict[int, float] = {}  # cue_id -> capture time (server clock) of the trigger frame

    def handle(self, event: g.Event, now: float, origin_ts: float | None = None) -> list[dict]:
        prev = self.state
        self.state, effects = g.step(self.state, event, now, self.gcfg)
        if self.state.mode != prev.mode:
            self.log.write("state", mode=self.state.mode.value, prev=prev.mode.value,
                           target=self.state.target)
        if self.state.items != prev.items or self.state.index != prev.index:
            self.shopping.set_state(self.state.items, self.state.index)
        return [self._message(e, origin_ts) for e in effects]

    def _message(self, effect: g.Effect, origin_ts: float | None) -> dict:
        if isinstance(effect, g.Say):
            r = self.catalog.resolve(effect)
            self.cue_id += 1
            if origin_ts is not None:
                self.cue_origin[self.cue_id] = origin_ts
                if len(self.cue_origin) > 500:
                    self.cue_origin.pop(next(iter(self.cue_origin)))
            self.last_cue_text = r.text
            return {"type": "cue", "cue_id": self.cue_id, "clip": r.clip, "priority": PRIORITIES[r.priority],
                    "interrupt": effect.interrupt, "text": r.text, "key": r.key}
        if isinstance(effect, g.Tone):
            return {"type": "tone", "freq_hz": effect.freq_hz, "duration_ms": effect.duration_ms,
                    "pan": effect.pan, "interval_ms": effect.interval_ms}
        if isinstance(effect, g.Stream):
            msg = {"type": "stream", "on": effect.on, "max_fps": self.gcfg.stream_max_fps,
                   "purpose": effect.purpose}
            if effect.burst is not None:
                msg["burst"] = effect.burst
                msg["burst_id"] = effect.burst_id
            return msg
        raise TypeError(f"unknown effect {effect!r}")

    def reload_list(self) -> None:
        """Pick up list edits from the CLI or the web editor."""
        self.state = g.with_items(self.state, tuple(self.shopping.items), self.shopping.current)

    def device_disconnected(self) -> None:
        """Quietly stop everything; the device re-sends its switch position when it reconnects."""
        self.state = replace(self.state, mode=g.Mode.STANDBY, target=None, track=g.Tracking(),
                             identify=None, other_history=(), search_ref=None)
