"""Guidance engine: detections and controls in, audio cues out (PRD 8.2 to 8.4).

PURE: no I/O, no clock, no randomness. Every function takes the current time as an argument,
so the same inputs always give the same outputs and the tests are deterministic.

    state, effects = step(state, event, now, cfg)

Events are ``Control`` (a button / switch event), ``Frame`` (detections from one processed
camera frame) and ``Tick`` (time passed; drives timeouts). Effects are ``Say`` (speak a phrase
from config/phrases.yaml), ``Tone`` (start/update/stop the panned guidance beep) and ``Stream``
(start/stop the device's camera stream).

Say effects carry class ids, never spoken names. The phrase catalog turns them into words.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Union

from slr.common.config import GuidanceConfig
from slr.common.types import FOUND, PENDING, Detection, ListItem


class Mode(str, Enum):
    STANDBY = "standby"
    IDLE = "idle"
    SEARCHING = "searching"
    GUIDING = "guiding"
    WITHIN_REACH = "within_reach"
    IDENTIFYING = "identifying"


FIND_MODES = (Mode.SEARCHING, Mode.GUIDING, Mode.WITHIN_REACH)
TRACKING_MODES = (Mode.GUIDING, Mode.WITHIN_REACH)

H_DIRS = {(0, -1): "straight", (0, 1): "straight", (1, -1): "slightly_left",
          (1, 1): "slightly_right", (2, -1): "left", (2, 1): "right"}
H_LEVEL = {"straight": 0, "slightly_left": 1, "slightly_right": 1, "left": 2, "right": 2}
V_ZONES = ("high", "eye", "low")


# --------------------------------------------------------------------------- events


@dataclass(frozen=True)
class Control:
    event: str
    value: int | None = None


@dataclass(frozen=True)
class Frame:
    detections: tuple[Detection, ...] = ()
    purpose: str = "find"
    burst_id: int = 0


@dataclass(frozen=True)
class Tick:
    pass


Event = Union[Control, Frame, Tick]


# --------------------------------------------------------------------------- effects


@dataclass(frozen=True)
class Say:
    key: str
    params: dict = field(default_factory=dict)
    interrupt: bool = False


@dataclass(frozen=True)
class Tone:
    """Guidance beep. interval_ms None means stop."""

    pan: float = 0.0
    interval_ms: int | None = None
    freq_hz: int = 880
    duration_ms: int = 60


@dataclass(frozen=True)
class Stream:
    on: bool
    purpose: str = "find"
    burst: int | None = None
    burst_id: int = 0


Effect = Union[Say, Tone, Stream]


def say(key: str, interrupt: bool = False, **params) -> Say:
    return Say(key, params, interrupt)


# --------------------------------------------------------------------------- state


@dataclass(frozen=True)
class Tracking:
    """Everything about the current find-mode target. Reset on every new search / loss."""

    history: tuple[Detection | None, ...] = ()
    last_seen_at: float | None = None
    last_box: Detection | None = None
    h_dir: str | None = None
    v_zone: str | None = None
    said_h: str | None = None
    said_v: str | None = None
    said_at: float | None = None
    reach_since: float | None = None
    tone: tuple[float, int] | None = None
    closer_ref_h: float | None = None
    closer_said_at: float | None = None


@dataclass(frozen=True)
class Identify:
    burst_id: int
    started_at: float
    resume: Mode
    picks: tuple[tuple[str, float] | None, ...] = ()


@dataclass(frozen=True)
class State:
    mode: Mode = Mode.STANDBY
    items: tuple[ListItem, ...] = ()
    index: int = 0
    target: str | None = None
    search_ref: float | None = None
    not_found_said_at: float | None = None
    track: Tracking = Tracking()
    other_history: tuple[str | None, ...] = ()
    wrong_item_said_at: float | None = None
    identify: Identify | None = None
    burst_counter: int = 0
    volume: int | None = None

    @property
    def current_item(self) -> ListItem | None:
        return self.items[self.index] if 0 <= self.index < len(self.items) else None

    @property
    def streaming(self) -> bool:
        return self.mode in FIND_MODES or self.mode == Mode.IDENTIFYING


def initial_state(items: tuple[ListItem, ...] | list[ListItem] = (), index: int = 0) -> State:
    items = tuple(items)
    return State(items=items, index=index if 0 <= index < len(items) else 0)


def with_items(state: State, items: tuple[ListItem, ...] | list[ListItem], index: int | None = None) -> State:
    """Swap in an edited shopping list, keeping the current index valid."""
    items = tuple(items)
    idx = state.index if index is None else index
    if not items:
        idx = 0
    else:
        idx = min(max(idx, 0), len(items) - 1)
    return replace(state, items=items, index=idx)


# --------------------------------------------------------------------------- pure helpers


def _zone_with_hysteresis(zone_fn, value: float, prev: int | None, hyst: float) -> int:
    """Only leave the previous zone once the value is ``hyst`` past the boundary."""
    raw = zone_fn(value)
    if prev is None or raw == prev:
        return raw
    if raw > prev:
        return max(prev, zone_fn(value - hyst))
    return min(prev, zone_fn(value + hyst))


def adjusted_center(box: Detection, cfg: GuidanceConfig) -> tuple[float, float]:
    cx = 1.0 - box.cx if cfg.flip_horizontal else box.cx
    return cx, box.cy + cfg.vertical_offset


def classify_horizontal(cx: float, prev: str | None, cfg: GuidanceConfig) -> str:
    """Box center x (0..1) -> left / slightly_left / straight / slightly_right / right."""

    def zone(a: float) -> int:
        return 0 if a < cfg.straight_tol else (1 if a < cfg.slight_tol else 2)

    offset = cx - 0.5
    level = _zone_with_hysteresis(zone, abs(offset), H_LEVEL.get(prev) if prev else None,
                                  cfg.direction_hysteresis)
    return H_DIRS[(level, -1 if offset < 0 else 1)]


def classify_vertical(cy: float, prev: str | None, cfg: GuidanceConfig) -> str:
    """Box center y (0..1, after vertical_offset) -> high / eye / low."""

    def zone(y: float) -> int:
        return 0 if y < cfg.high_shelf_max else (2 if y > cfg.low_shelf_min else 1)

    prev_idx = V_ZONES.index(prev) if prev else None
    return V_ZONES[_zone_with_hysteresis(zone, cy, prev_idx, cfg.direction_hysteresis)]


def select_target(dets: tuple[Detection, ...], target: str | None, last_box: Detection | None,
                  cfg: GuidanceConfig) -> Detection | None:
    """Pick the target box in this frame: overlap with the tracked box first, else most confident."""
    cands = [d for d in dets if d.cls == target and d.conf >= cfg.conf_find]
    if not cands:
        return None
    if last_box is not None:
        best = max(cands, key=lambda d: d.iou(last_box))
        if best.iou(last_box) >= cfg.iou_track_min:
            return best
    return max(cands, key=lambda d: d.conf)


def is_stable(history: tuple[Detection | None, ...], cfg: GuidanceConfig) -> bool:
    return sum(1 for d in history if d is not None) >= cfg.stable_k


def pick_identify_candidate(dets: tuple[Detection, ...], cfg: GuidanceConfig) -> tuple[str, float] | None:
    """Most confident detection nearest the center (FR-31)."""
    if not dets:
        return None

    def score(d: Detection) -> float:
        dist = ((d.cx - 0.5) ** 2 + (d.cy - 0.5) ** 2) ** 0.5
        return d.conf - cfg.identify_center_weight * dist

    best = max(dets, key=score)
    return best.cls, best.conf


def tone_for(box: Detection, mode: Mode, cfg: GuidanceConfig) -> tuple[float, int]:
    """Parking-sensor style tone (FR-21): pan toward the target, faster when centered and close."""
    cx, _ = adjusted_center(box, cfg)
    pan = max(-1.0, min(1.0, (cx - 0.5) * 2))
    if mode == Mode.WITHIN_REACH:
        return round(pan, 2), cfg.tone_interval_min_ms
    centered = 1.0 - min(1.0, abs(cx - 0.5) / 0.5)
    size = min(1.0, box.h / cfg.reach_h) if cfg.reach_h > 0 else 1.0
    closeness = 0.5 * centered + 0.5 * size
    span = cfg.tone_interval_max_ms - cfg.tone_interval_min_ms
    return round(pan, 2), int(round(cfg.tone_interval_max_ms - span * closeness))


def vote_identify(picks: tuple[tuple[str, float] | None, ...], cfg: GuidanceConfig) -> str | None:
    """Majority vote over the burst (FR-32). Returns the winner or None if not confident."""
    votes = Counter(p[0] for p in picks if p is not None and p[1] >= cfg.conf_identify)
    if not votes:
        return None
    ranked = votes.most_common()
    winner, count = ranked[0]
    if count < cfg.identify_min_votes:
        return None
    if len(ranked) > 1 and ranked[1][1] == count:
        return None  # tie: never guess
    return winner


def next_pending_index(items: tuple[ListItem, ...], start: int) -> int | None:
    n = len(items)
    for k in range(1, n + 1):
        i = (start + k) % n
        if items[i].status == PENDING:
            return i
    return None


# --------------------------------------------------------------------------- transitions


def step(state: State, event: Event, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    if isinstance(event, Control):
        state, fx = _on_control(state, event, now, cfg)
    elif isinstance(event, Frame):
        state, fx = _on_frame(state, event, now, cfg)
    else:
        fx = []
    state, tfx = _timers(state, now, cfg)
    return state, fx + tfx


def _stop_tone(s: State) -> tuple[State, list[Effect]]:
    if s.track.tone is None:
        return s, []
    return replace(s, track=replace(s.track, tone=None)), [Tone(interval_ms=None)]


def _leave_find(s: State) -> tuple[State, list[Effect]]:
    """Stop any search / identify: stream off, tone off, tracking cleared."""
    if not s.streaming:
        return s, []
    s, fx = _stop_tone(s)
    s = replace(s, target=None, track=Tracking(), other_history=(), identify=None,
                search_ref=None, not_found_said_at=None)
    return s, fx + [Stream(on=False)]


def _start_search(s: State, target: str, now: float, cfg: GuidanceConfig) -> State:
    return replace(s, mode=Mode.SEARCHING, target=target, track=Tracking(), other_history=(),
                   search_ref=now, not_found_said_at=None, identify=None)


def _announce_item(s: State) -> Say:
    item = s.items[s.index]
    key = "item_announce_found" if item.status == FOUND else "item_announce"
    return say(key, interrupt=True, n=s.index + 1, total=len(s.items), item=item.name)


def _on_control(s: State, ev: Control, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    e = ev.event
    if e == "volume_set":
        # Volume is applied and announced on the device; the server only records it (FR-45).
        return replace(s, volume=ev.value), []
    if e == "shutdown":
        # The device says "Shutting down" from a local clip and powers off by itself.
        s, fx = _leave_find(s)
        return replace(s, mode=Mode.STANDBY), fx
    if e == "mode_off":
        s, fx = _leave_find(s)
        return replace(s, mode=Mode.STANDBY), fx + [say("paused", interrupt=True)]
    if e == "mode_on":
        if s.mode == Mode.STANDBY:
            return replace(s, mode=Mode.IDLE), [say("ready", interrupt=True)]
        return s, []  # already active (e.g. TOGGLE springing back to ON)

    if s.mode == Mode.STANDBY:
        return s, [say("standby_hint")]  # only the mode switch and shutdown work in standby
    if s.mode == Mode.IDENTIFYING:
        return s, []  # the burst takes about a second; ignore other controls meanwhile

    if e == "identify":
        return _start_identify(s, now, cfg)
    if e in ("next_item", "prev_item"):
        if not s.items:
            return s, [say("list_empty", interrupt=True)]
        s, fx = _leave_find(s)
        step_by = 1 if e == "next_item" else -1
        s = replace(s, mode=Mode.IDLE, index=(s.index + step_by) % len(s.items))
        return s, fx + [_announce_item(s)]
    if e == "enter":
        return _on_enter(s, now, cfg)
    if e == "help":
        return s, [say("help", interrupt=True)]
    if e == "status":
        return s, [_status(s)]
    return s, []


def _status(s: State) -> Say:
    if not s.items:
        return say("list_empty", interrupt=True)
    found = sum(1 for it in s.items if it.status == FOUND)
    if found == len(s.items):
        return say("status_done", interrupt=True, total=len(s.items))
    return say("status", interrupt=True, found=found, total=len(s.items), item=s.items[s.index].name)


def _on_enter(s: State, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    if s.mode == Mode.IDLE:
        if not s.items:
            return s, [say("list_empty", interrupt=True)]
        target = s.items[s.index].name
        s = _start_search(s, target, now, cfg)
        return s, [say("searching", interrupt=True, item=target),
                   Stream(on=True, purpose="find")]
    if s.mode in (Mode.SEARCHING, Mode.GUIDING):
        s, fx = _leave_find(s)
        return replace(s, mode=Mode.IDLE), fx + [say("search_stopped", interrupt=True)]
    if s.mode == Mode.WITHIN_REACH:
        target = s.target
        items = tuple(replace(it, status=FOUND) if i == s.index else it for i, it in enumerate(s.items))
        s, fx = _leave_find(s)
        nxt = next_pending_index(items, s.index)
        if nxt is None:
            s = replace(s, mode=Mode.IDLE, items=items)
            return s, fx + [say("checked_off_done", interrupt=True, item=target)]
        s = replace(s, mode=Mode.IDLE, items=items, index=nxt)
        return s, fx + [say("checked_off_next", interrupt=True, item=target, next=items[nxt].name)]
    return s, []


def _start_identify(s: State, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    resume = s.mode
    s, fx = _stop_tone(s)
    burst_id = (s.burst_counter % 0xFFFF) + 1
    s = replace(s, mode=Mode.IDENTIFYING, burst_counter=burst_id,
                identify=Identify(burst_id=burst_id, started_at=now, resume=resume))
    return s, fx + [say("identifying"),
                    Stream(on=True, purpose="identify", burst=cfg.identify_burst, burst_id=burst_id)]


def _finish_identify(s: State, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    ident = s.identify
    assert ident is not None
    if len(ident.picks) < cfg.identify_min_votes:
        cue = say("not_sure_loss", interrupt=True)  # too few frames arrived: never guess
    else:
        winner = vote_identify(ident.picks, cfg)
        if winner is None:
            cue = say("not_sure", interrupt=True)
        elif any(it.name == winner and it.status == PENDING for it in s.items):
            cue = say("identified_on_list", interrupt=True, item=winner)
        else:
            cue = say("identified", interrupt=True, item=winner)
    # FR-34: return to the previous state, resuming the search if one was active
    if ident.resume in FIND_MODES and s.target is not None:
        s = _start_search(s, s.target, now, cfg)
        return s, [cue, Stream(on=True, purpose="find")]
    s = replace(s, mode=Mode.IDLE if ident.resume != Mode.STANDBY else Mode.STANDBY,
                identify=None, target=None)
    return s, [cue, Stream(on=False)]


def _on_frame(s: State, f: Frame, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    if s.mode == Mode.IDENTIFYING:
        ident = s.identify
        if ident is None or f.purpose != "identify" or f.burst_id != ident.burst_id:
            return s, []
        ident = replace(ident, picks=ident.picks + (pick_identify_candidate(f.detections, cfg),))
        s = replace(s, identify=ident)
        if len(ident.picks) >= cfg.identify_burst:
            return _finish_identify(s, now, cfg)
        return s, []

    if s.mode not in FIND_MODES or f.purpose != "find" or s.target is None:
        return s, []

    det = select_target(f.detections, s.target, s.track.last_box, cfg)
    track = replace(s.track, history=(s.track.history + (det,))[-cfg.stable_n:])
    if det is not None:
        track = replace(track, last_seen_at=now, last_box=det)
    s = replace(s, track=track)

    if s.mode == Mode.SEARCHING:
        if det is not None:
            s = replace(s, search_ref=now)  # resets the "not seeing" timer (FR-12)
        if is_stable(track.history, cfg) and track.last_box is not None:
            s = replace(s, mode=Mode.GUIDING, other_history=())
            fx: list[Effect] = [say("found", interrupt=True, item=s.target)]
            s, gfx = _guide(s, track.last_box, now, cfg, acquired=True)
            return s, fx + gfx
        return _check_wrong_item(s, f, det, now, cfg)

    if det is None:
        return s, []
    return _guide(s, det, now, cfg, acquired=False)


def _check_wrong_item(s: State, f: Frame, det: Detection | None, now: float,
                      cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    """FR-20: a different known item, confidently detected and centered, said at most every few s."""
    if not cfg.wrong_item_enabled:
        return s, []
    other: str | None = None
    if det is None:
        cands = [d for d in f.detections
                 if d.cls != s.target and d.conf >= cfg.conf_wrong_item
                 and abs(adjusted_center(d, cfg)[0] - 0.5) < cfg.wrong_item_center_tol
                 and abs(d.cy - 0.5) < cfg.wrong_item_center_tol * 2]
        if cands:
            other = max(cands, key=lambda d: d.conf).cls
    hist = (s.other_history + (other,))[-cfg.stable_n:]
    s = replace(s, other_history=hist)
    if other is None or hist.count(other) < cfg.stable_k:
        return s, []
    if s.wrong_item_said_at is not None and now - s.wrong_item_said_at < cfg.wrong_item_repeat_s:
        return s, []
    return replace(s, wrong_item_said_at=now), [say("wrong_item", other=other, item=s.target)]


def _guide(s: State, box: Detection, now: float, cfg: GuidanceConfig,
           acquired: bool) -> tuple[State, list[Effect]]:
    """Direction, within-reach, getting-closer and tone cues for one sighting of the target."""
    t = s.track
    cx, cy = adjusted_center(box, cfg)
    h_dir = classify_horizontal(cx, t.h_dir, cfg)
    v_zone = classify_vertical(cy, t.v_zone, cfg)
    t = replace(t, h_dir=h_dir, v_zone=v_zone)
    fx: list[Effect] = []

    if s.mode == Mode.WITHIN_REACH:
        if h_dir == "straight" and box.h >= cfg.reach_h * cfg.reach_exit_ratio:
            s = replace(s, track=t)
            return _update_tone(s, box, cfg)
        # drifted away: back to guiding and say the new direction right away
        s = replace(s, mode=Mode.GUIDING)
        t = replace(t, reach_since=None, said_h=None, said_at=None)
    else:
        reach = h_dir == "straight" and box.h >= cfg.reach_h
        reach_since = (t.reach_since if t.reach_since is not None else now) if reach else None
        t = replace(t, reach_since=reach_since)
        if reach_since is not None and now - reach_since >= cfg.reach_hold_s:
            s = replace(s, mode=Mode.WITHIN_REACH, track=t)
            s, tfx = _update_tone(s, box, cfg)
            return s, [say("within_reach", interrupt=True, item=s.target)] + tfx

    speak_v = acquired or v_zone != t.said_v
    changed = h_dir != t.said_h
    due = t.said_at is None or now - t.said_at >= cfg.direction_repeat_s
    if changed or due or speak_v:
        if speak_v:
            fx.append(say("direction", h=h_dir, v=v_zone))
        else:
            fx.append(say("direction_h", h=h_dir))
        t = replace(t, said_h=h_dir, said_v=v_zone, said_at=now)
    if cfg.getting_closer_enabled:
        ref = t.closer_ref_h
        if ref is None or box.h < ref:
            t = replace(t, closer_ref_h=box.h)
        elif box.h >= ref * cfg.getting_closer_ratio and not fx and (
                t.closer_said_at is None or now - t.closer_said_at >= cfg.getting_closer_repeat_s):
            fx.append(say("getting_closer"))
            t = replace(t, closer_ref_h=box.h, closer_said_at=now)

    s = replace(s, track=t)
    s, tfx = _update_tone(s, box, cfg)
    return s, fx + tfx


def _update_tone(s: State, box: Detection, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    if not cfg.tones_enabled:
        return s, []
    pan, interval = tone_for(box, s.mode, cfg)
    prev = s.track.tone
    if prev is not None and abs(prev[0] - pan) < cfg.tone_min_change_pan \
            and abs(prev[1] - interval) < cfg.tone_min_change_ms:
        return s, []
    s = replace(s, track=replace(s.track, tone=(pan, interval)))
    return s, [Tone(pan=pan, interval_ms=interval, freq_hz=cfg.tone_freq_hz,
                    duration_ms=cfg.tone_duration_ms)]


def _timers(s: State, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    fx: list[Effect] = []
    if s.mode in TRACKING_MODES and s.track.last_seen_at is not None \
            and now - s.track.last_seen_at > cfg.lost_timeout_s:
        s, tfx = _stop_tone(s)
        s = replace(s, mode=Mode.SEARCHING, track=Tracking(), other_history=(), search_ref=now,
                    not_found_said_at=None)
        fx += tfx + [say("lost", interrupt=True)]

    if s.mode == Mode.SEARCHING and s.search_ref is not None \
            and now - s.search_ref >= cfg.not_found_prompt_s \
            and (s.not_found_said_at is None or now - s.not_found_said_at >= cfg.not_found_repeat_s):
        s = replace(s, not_found_said_at=now)
        fx.append(say("not_seeing", item=s.target))

    if s.mode == Mode.IDENTIFYING and s.identify is not None \
            and now - s.identify.started_at >= cfg.identify_timeout_s:
        s, ifx = _finish_identify(s, now, cfg)
        fx += ifx
    return s, fx
