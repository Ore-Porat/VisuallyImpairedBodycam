"""Guidance engine: detections and controls in, audio cues out (PRD 8.2 to 8.4).

PURE: no I/O, no clock, no randomness. Every function takes the current time as an argument,
so the same inputs always give the same outputs and the tests are deterministic.

    state, effects = step(state, event, now, cfg)

Events are ``Control`` (a button / switch event), ``Frame`` (detections from one processed
camera frame) and ``Tick`` (time passed; drives timeouts). Effects are ``Say`` (speak a phrase
from config/phrases.yaml), ``Tone`` (start/update/stop the panned guidance beep) and ``Stream``
(start/stop the device's camera stream).

Say effects carry class ids, never spoken names. The phrase catalog turns them into words.

=====================================================================================
THE BIG PICTURE (read this first if you are new to the code)
=====================================================================================

This file is the "brain" of the product. It decides WHAT TO SAY to the shopper based on what
the camera sees and which buttons were pressed. It does not talk to the camera, the network or
the speaker itself; other files do that and simply call ``step`` here.

It is written as a STATE MACHINE. Think of a board game token sitting on one of a few squares
(the "modes"). Every time something happens (an "event") the token may hop to another square and
some "effects" (speech, beeps, camera on/off) are produced along the way.

The modes (squares):

- STANDBY       The mode switch on the camera body is OFF. With ``stream_in_off`` the camera
                and detector keep running as a silent monitor, but no guidance is spoken.
- IDLE          Switched on, waiting. The shopper can browse the list, press Enter to search,
                or press Identify.
- SEARCHING     Camera is streaming, looking for the current list item. Nothing found yet.
- GUIDING       The item has been spotted. We say "left / right / eye level..." and beep.
- WITHIN_REACH  The item is big and centered in the frame: it is at arm's length. Pressing
                Enter now checks it off the list.
- IDENTIFYING   A short burst of frames is being analysed to answer "what am I holding?".
                Afterwards we go back to where we were.

The events (things that happen):

- Control(event, value)   A button/switch on the device: "enter", "next_item", "prev_item",
                          "identify", "mode_on", "mode_off", "help", "status", "volume_set",
                          "shutdown".
- Frame(detections, ...)  The object detector has looked at one camera picture and this is the
                          list of boxes it found (each box = class name + confidence + position).
- Tick()                  Nothing happened, but time passed. Used to notice time-outs ("I lost
                          the item", "I still don't see it").

The effects (things we ask other code to do):

- Say(key, params)        Speak the phrase named ``key`` from ``config/phrases.yaml``, filling
                          in ``params`` (like the item name).
- Tone(pan, interval_ms)  Play / update / stop the parking-sensor style beep. ``pan`` is -1
                          (left ear) to +1 (right ear). ``interval_ms=None`` means stop.
- Stream(on, purpose)     Switch the device camera stream on or off.

Everything is IMMUTABLE: we never modify a ``State`` in place. Instead ``replace(state, ...)``
returns a fresh copy with some fields changed. That is what makes the engine easy to test and
reason about: old states are never corrupted by accident.

WORDS YOU WILL SEE
------------------
- dataclass: a class that is just a bundle of named fields. ``@dataclass(frozen=True)`` makes it
  read-only (you cannot change a field after creation; use ``replace`` to get a changed copy).
- Enum: a fixed set of named constants (``Mode.IDLE`` etc.).
- type hints like ``x: float`` or ``-> State``: notes about what type a value is. Python does
  not enforce them; they are documentation for people and tools.
- ``str | None``: "a string, or nothing".
- tuple: an immutable list, written ``(a, b, c)``. ``tuple[Detection, ...]`` = "a tuple of any
  number of Detections".
- Detection: one box found by the detector (see ``slr/common/types.py``). ``.cls`` is the class
  name (e.g. "pasta"), ``.conf`` is the detector's confidence 0..1, ``.cx/.cy`` the box center,
  ``.h`` the box height. All positions are fractions of the frame (0 = left/top, 1 = right/bottom).
- confidence: how sure the detector is, from 0 (guess) to 1 (certain).
- IoU (intersection over union): how much two boxes overlap, 0 = none, 1 = identical.
- hysteresis: a deliberate "stickiness" so a value hovering around a boundary does not flip
  back and forth. We only change our mind once it is clearly past the line.
- cfg / GuidanceConfig: all the tunable numbers from the ``guidance:`` section of
  ``config/server.yaml`` (thresholds, time-outs, tone settings).
- FR-xx: requirement numbers from the product spec, ``PRD.pdf``.
"""

from __future__ import annotations

# Counter counts how many times each value appears (used for the identify vote).
from collections import Counter
# dataclass/field define simple record classes; replace makes a modified copy of one.
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Union

from slr.common.config import GuidanceConfig
from slr.common.types import FOUND, PENDING, Detection, ListItem


class Mode(str, Enum):
    """The six "squares" of the state machine. See the overview at the top of the file.

    Inheriting from ``str`` too means ``Mode.IDLE == "idle"`` is True, which makes logging and
    JSON output simple.
    """

    STANDBY = "standby"
    IDLE = "idle"
    SEARCHING = "searching"
    GUIDING = "guiding"
    WITHIN_REACH = "within_reach"
    IDENTIFYING = "identifying"


# Modes in which a "find" search is running (the camera streams and we look for the target).
FIND_MODES = (Mode.SEARCHING, Mode.GUIDING, Mode.WITHIN_REACH)
# Modes in which we have already seen the target and are following it frame by frame.
TRACKING_MODES = (Mode.GUIDING, Mode.WITHIN_REACH)

# Horizontal direction words. Key = (how far off center: 0 none / 1 a bit / 2 a lot,
# which side: -1 left / +1 right). Value = the direction key used in phrases.yaml.
H_DIRS = {(0, -1): "straight", (0, 1): "straight", (1, -1): "slightly_left",
          (1, 1): "slightly_right", (2, -1): "left", (2, 1): "right"}
# The reverse: direction word -> how far off center (needed for hysteresis).
H_LEVEL = {"straight": 0, "slightly_left": 1, "slightly_right": 1, "left": 2, "right": 2}
# Vertical zones from top of frame to bottom: high shelf, eye level, low shelf.
V_ZONES = ("high", "eye", "low")
# Hand cue words: the way the HAND must move to land on the item (the {hd} placeholder).
HAND_DIRS = ("hand_left", "hand_right", "hand_up", "hand_down", "hand_up_left", "hand_up_right",
             "hand_down_left", "hand_down_right", "hand_forward")


# --------------------------------------------------------------------------- events
# An "event" is the input to one step of the state machine.


@dataclass(frozen=True)
class Control:
    """A button, switch or dial event from the device.

    event: the name, e.g. "enter", "next_item", "identify", "mode_on", "volume_set".
    value: optional number that goes with it (only used for "volume_set").
    """

    event: str
    value: int | None = None


@dataclass(frozen=True)
class Frame:
    """The detector's results for one camera picture.

    detections: every box the detector found in this picture.
    purpose:    "find" (normal search stream) or "identify" (a burst for Identify mode).
    burst_id:   which identify burst this frame belongs to; 0 for find frames. Lets us ignore
                late frames from an older burst.
    """

    detections: tuple[Detection, ...] = ()
    purpose: str = "find"
    burst_id: int = 0


@dataclass(frozen=True)
class Tick:
    """Nothing happened except that time passed. The server sends one about 10 times a second."""

    pass


# "An Event is a Control OR a Frame OR a Tick."
Event = Union[Control, Frame, Tick]


# --------------------------------------------------------------------------- effects
# An "effect" is an instruction the engine hands back: it never performs it itself.


@dataclass(frozen=True)
class Say:
    """Speak a phrase.

    key:       the phrase name in ``config/phrases.yaml`` (e.g. "found", "direction").
    params:    values to fill into the phrase text, e.g. ``{"item": "pasta"}``. Item values are
               class ids; the phrase catalog turns them into spoken names.
    interrupt: True = cut off whatever is being spoken right now and say this instead.
    """

    key: str
    # ``field(default_factory=dict)`` gives each Say its own fresh empty dict as the default.
    params: dict = field(default_factory=dict)
    interrupt: bool = False


@dataclass(frozen=True)
class Tone:
    """Guidance beep. interval_ms None means stop.

    pan:         where the beep sits in the headphones: -1 = fully left, 0 = center, +1 = right.
    interval_ms: time between beeps. Shorter = closer / more centered. None = stop beeping.
    freq_hz:     pitch of the beep.
    duration_ms: length of each individual beep.
    """

    pan: float = 0.0
    interval_ms: int | None = None
    freq_hz: int = 880
    duration_ms: int = 60


@dataclass(frozen=True)
class Stream:
    """Turn the device's camera stream on or off.

    on:       True to start streaming frames to the laptop, False to stop.
    purpose:  "find" or "identify" (the device tags every frame with it).
    burst:    for identify: send exactly this many frames then stop. None = keep streaming.
    burst_id: number that identifies this particular burst.
    """

    on: bool
    purpose: str = "find"
    burst: int | None = None
    burst_id: int = 0


Effect = Union[Say, Tone, Stream]


def say(key: str, interrupt: bool = False, **params) -> Say:
    """Shorthand for building a ``Say``: ``say("found", interrupt=True, item="pasta")``.

    ``**params`` collects any extra keyword arguments into a dict.
    """
    return Say(key, params, interrupt)


# --------------------------------------------------------------------------- state
# The "state" is everything the engine needs to remember between steps.


@dataclass(frozen=True)
class Tracking:
    """Everything about the current find-mode target. Reset on every new search / loss.

    history:         the last few frames' sighting of the target (a Detection or None if the
                     target was not in that frame). Used to decide "stable" sightings.
    last_seen_at:    time the target was last seen (for the lost time-out).
    last_box:        the most recent box of the target (helps follow the same box next frame).
    h_dir / v_zone:  the current horizontal direction word and vertical zone.
    said_h / said_v: the direction / zone we last SPOKE (so we only repeat on change or timer).
    said_at:         when we last spoke a direction.
    reach_since:     when the box first looked "within reach" (it must hold for a moment).
    tone:            the (pan, interval) of the beep currently playing, so we only send updates
                     when it changes noticeably.
    closer_ref_h:    reference box height for the "getting closer" cue.
    closer_said_at:  when we last said "getting closer".
    hand_said / hand_said_at: the hand cue we last spoke and when (within reach only).
    hand_touch_since: when the hand first covered the item (must hold for a moment).
    """

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
    hand_said: str | None = None
    hand_said_at: float | None = None
    hand_touch_since: float | None = None


@dataclass(frozen=True)
class Identify:
    """Bookkeeping for one Identify burst ("what am I holding?").

    burst_id:   which burst we are waiting for (frames from other bursts are ignored).
    started_at: when the burst began (for the time-out).
    resume:     the mode to go back to afterwards (e.g. continue a search that was running).
    picks:      per frame, the best candidate (class, confidence) or None if nothing was seen.
    seen:       per frame, every confident item candidate (class, confidence). This is used for
                the multi-object answer; ``picks`` is kept for the single-object vote.
    """

    burst_id: int
    started_at: float
    resume: Mode
    picks: tuple[tuple[str, float] | None, ...] = ()
    seen: tuple[tuple[tuple[str, float], ...], ...] = ()


@dataclass(frozen=True)
class State:
    """The complete memory of the engine. One of these exists at a time; ``step`` returns a new one.

    mode:               which square the token is on (see ``Mode``).
    items:              the shopping list (name + pending/found).
    index:              which list item is current (0 = first).
    target:             class id being searched for, or None.
    search_ref:         when the search started / target last seen while SEARCHING, for the
                        "I don't see X" reminder.
    not_found_said_at:  when we last said "I don't see X".
    track:              see ``Tracking``.
    other_history:      last few frames' "some other list item is centered" sightings (FR-20).
    wrong_item_said_at: when we last said "that's Y, not X".
    identify:           see ``Identify``; None when not identifying.
    burst_counter:      increments per identify burst to make unique burst ids.
    volume:             last volume level reported by the device (just recorded).
    """

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

    # ``@property`` lets you write ``state.current_item`` (no parentheses) and run this code.
    @property
    def current_item(self) -> ListItem | None:
        """The list item at ``index``, or None if the list is empty / index is out of range."""
        return self.items[self.index] if 0 <= self.index < len(self.items) else None

    @property
    def streaming(self) -> bool:
        """True in every mode where the device camera should be sending frames."""
        return self.mode in FIND_MODES or self.mode == Mode.IDENTIFYING


def initial_state(items: tuple[ListItem, ...] | list[ListItem] = (), index: int = 0) -> State:
    """Build the starting state (STANDBY) from a shopping list and current item index."""
    items = tuple(items)
    # If the saved index is out of range, fall back to the first item.
    return State(items=items, index=index if 0 <= index < len(items) else 0)


def with_items(state: State, items: tuple[ListItem, ...] | list[ListItem], index: int | None = None) -> State:
    """Swap in an edited shopping list, keeping the current index valid."""
    items = tuple(items)
    idx = state.index if index is None else index
    if not items:
        idx = 0
    else:
        # Clamp the index into 0 .. len-1.
        idx = min(max(idx, 0), len(items) - 1)
    return replace(state, items=items, index=idx)


# --------------------------------------------------------------------------- pure helpers
# Small math helpers used by the transitions below. None of them change any state.


def _zone_with_hysteresis(zone_fn, value: float, prev: int | None, hyst: float) -> int:
    """Only leave the previous zone once the value is ``hyst`` past the boundary.

    zone_fn: a function that maps a number to a zone index (0, 1, 2...).
    value:   the number to classify.
    prev:    the zone we were in last time (None = first time, no stickiness).
    hyst:    how far past the boundary the value must be before we switch zones.
    """
    raw = zone_fn(value)
    if prev is None or raw == prev:
        return raw
    if raw > prev:
        # Moving up a zone: pretend the value is ``hyst`` smaller; only switch if it still
        # lands above the previous zone.
        return max(prev, zone_fn(value - hyst))
    # Moving down a zone: same idea in the other direction.
    return min(prev, zone_fn(value + hyst))


def adjusted_center(box: Detection, cfg: GuidanceConfig) -> tuple[float, float]:
    """The box center after applying the camera corrections from config.

    flip_horizontal swaps left/right for a mirrored camera; vertical_offset compensates for the
    camera tilting up or down on the chest.
    """
    cx = 1.0 - box.cx if cfg.flip_horizontal else box.cx
    return cx, box.cy + cfg.vertical_offset


def classify_horizontal(cx: float, prev: str | None, cfg: GuidanceConfig) -> str:
    """Box center x (0..1) -> left / slightly_left / straight / slightly_right / right.

    cx:   horizontal center of the box, 0 = left edge of the frame, 1 = right edge.
    prev: the direction word we used last time (for hysteresis), or None.
    """

    # Inner helper: how far off center (as a 0..1 fraction) maps to level 0 / 1 / 2.
    def zone(a: float) -> int:
        return 0 if a < cfg.straight_tol else (1 if a < cfg.slight_tol else 2)

    # Distance from the middle: negative = left of center, positive = right.
    offset = cx - 0.5
    level = _zone_with_hysteresis(zone, abs(offset), H_LEVEL.get(prev) if prev else None,
                                  cfg.direction_hysteresis)
    return H_DIRS[(level, -1 if offset < 0 else 1)]


def classify_vertical(cy: float, prev: str | None, cfg: GuidanceConfig) -> str:
    """Box center y (0..1, after vertical_offset) -> high / eye / low."""

    # 0 = high shelf (top of frame), 1 = eye level, 2 = low shelf (bottom of frame).
    def zone(y: float) -> int:
        return 0 if y < cfg.high_shelf_max else (2 if y > cfg.low_shelf_min else 1)

    prev_idx = V_ZONES.index(prev) if prev else None
    return V_ZONES[_zone_with_hysteresis(zone, cy, prev_idx, cfg.direction_hysteresis)]


def pick_hand(dets: tuple[Detection, ...], cfg: GuidanceConfig) -> Detection | None:
    """The most confident hand box in the frame, or None if no confident hand is visible."""
    hands = [d for d in dets if d.cls == cfg.hand_class and d.conf >= cfg.hand_conf]
    return max(hands, key=lambda d: d.conf) if hands else None


def hand_on_item(hand: Detection, item: Detection, cfg: GuidanceConfig) -> bool:
    """True when the hand covers the item: its center is inside the item box, or at least
    ``hand_touch_overlap`` of the hand box overlaps the item box."""
    ix = max(0.0, min(hand.x2, item.x2) - max(hand.x1, item.x1))
    iy = max(0.0, min(hand.y2, item.y2) - max(hand.y1, item.y1))
    area = hand.w * hand.h
    if area <= 0:
        return False
    inside = item.x1 <= hand.cx <= item.x2 and item.y1 <= hand.cy <= item.y2
    return inside or (ix * iy) / area >= cfg.hand_touch_overlap


def hand_direction(hand: Detection, item: Detection, cfg: GuidanceConfig) -> str:
    """Which way the hand must move to reach the item: one of ``HAND_DIRS``.

    Compares the two box centers (after the camera corrections). An axis closer than
    ``hand_cue_tol`` counts as aligned; aligned on both axes gives "hand_forward".
    """
    hx, hy = adjusted_center(hand, cfg)
    ix, iy = adjusted_center(item, cfg)
    dx, dy = ix - hx, iy - hy
    h = "" if abs(dx) < cfg.hand_cue_tol else ("right" if dx > 0 else "left")
    v = "" if abs(dy) < cfg.hand_cue_tol else ("down" if dy > 0 else "up")
    if not h and not v:
        return "hand_forward"
    return "hand_" + "_".join(p for p in (v, h) if p)


def select_target(dets: tuple[Detection, ...], target: str | None, last_box: Detection | None,
                  cfg: GuidanceConfig) -> Detection | None:
    """Pick the target box in this frame: overlap with the tracked box first, else most confident.

    dets:     all boxes in the frame.
    target:   the class id we are looking for.
    last_box: where the target was last frame (None if not yet seen).
    Returns the chosen box, or None if the target is not confidently in this frame.
    """
    # Keep only boxes of the right class with enough confidence.
    cands = [d for d in dets if d.cls == target and d.conf >= cfg.conf_find]
    if not cands:
        return None
    if last_box is not None:
        # Prefer the box that overlaps most with where we saw it last (follow the same item,
        # even if a second one of the same kind appears on the shelf).
        # ``key=lambda d: ...`` tells max() what number to compare for each candidate.
        best = max(cands, key=lambda d: d.iou(last_box))
        if best.iou(last_box) >= cfg.iou_track_min:
            return best
    # Otherwise take the one the detector is most sure about.
    return max(cands, key=lambda d: d.conf)


def is_stable(history: tuple[Detection | None, ...], cfg: GuidanceConfig) -> bool:
    """True if the target was seen in at least ``stable_k`` of the last few frames.

    This avoids announcing "found" on a single flickering detection.
    """
    return sum(1 for d in history if d is not None) >= cfg.stable_k


def identify_candidates(dets: tuple[Detection, ...], cfg: GuidanceConfig) -> tuple[tuple[str, float], ...]:
    """Every confident non-item class visible in one frame, as (class id, confidence).

    A class appears at most once per frame. The shopper's hand (and any other non-item class)
    can never be part of an Identify answer.
    """
    best: dict[str, float] = {}
    for d in dets:
        if d.cls in cfg.non_item_classes or d.conf < cfg.conf_identify:
            continue
        best[d.cls] = max(best.get(d.cls, 0.0), d.conf)
    return tuple(best.items())


def pick_identify_candidate(dets: tuple[Detection, ...], cfg: GuidanceConfig) -> tuple[str, float] | None:
    """Most confident detection nearest the center (FR-31).

    Returns (class id, confidence) of the best box, or None if the frame had no boxes.
    """
    # The shopper's hand (and any other non-item class) can never be the answer.
    dets = tuple(d for d in dets if d.cls not in cfg.non_item_classes)
    if not dets:
        return None

    # Score = confidence minus a penalty for being far from the center of the picture.
    def score(d: Detection) -> float:
        dist = ((d.cx - 0.5) ** 2 + (d.cy - 0.5) ** 2) ** 0.5
        return d.conf - cfg.identify_center_weight * dist

    best = max(dets, key=score)
    return best.cls, best.conf


def tone_for(box: Detection, mode: Mode, cfg: GuidanceConfig) -> tuple[float, int]:
    """Parking-sensor style tone (FR-21): pan toward the target, faster when centered and close.

    Returns (pan, interval_ms). pan: -1 left .. +1 right. interval: ms between beeps.
    """
    cx, _ = adjusted_center(box, cfg)
    # Map center position 0..1 to pan -1..+1 and clamp.
    pan = max(-1.0, min(1.0, (cx - 0.5) * 2))
    if mode == Mode.WITHIN_REACH:
        # At arm's length: fastest possible beeping.
        return round(pan, 2), cfg.tone_interval_min_ms
    # "closeness" blends two things, each 0..1: how centered the box is and how big it is
    # (a bigger box means the item is nearer the camera).
    centered = 1.0 - min(1.0, abs(cx - 0.5) / 0.5)
    size = min(1.0, box.h / cfg.reach_h) if cfg.reach_h > 0 else 1.0
    closeness = 0.5 * centered + 0.5 * size
    # Interpolate between the slowest and fastest beep intervals.
    span = cfg.tone_interval_max_ms - cfg.tone_interval_min_ms
    return round(pan, 2), int(round(cfg.tone_interval_max_ms - span * closeness))


def vote_identify(picks: tuple[tuple[str, float] | None, ...], cfg: GuidanceConfig) -> str | None:
    """Majority vote over the burst (FR-32). Returns the winner or None if not confident."""
    # Count one vote per frame for its candidate class, ignoring weak candidates.
    votes = Counter(p[0] for p in picks if p is not None and p[1] >= cfg.conf_identify)
    if not votes:
        return None
    # most_common() gives [(class, count), ...] sorted by count, highest first.
    ranked = votes.most_common()
    winner, count = ranked[0]
    if count < cfg.identify_min_votes:
        return None
    if len(ranked) > 1 and ranked[1][1] == count:
        return None  # tie: never guess
    return winner


def vote_identify_many(seen: tuple[tuple[tuple[str, float], ...], ...],
                       cfg: GuidanceConfig) -> list[str]:
    """All classes seen in at least ``identify_min_votes`` frames, best first.

    A class gets one vote per frame, not one per box, so a shelf full of the same product does
    not outvote everything else. The result is capped at ``identify_max_objects``. With the
    default of 1 this keeps the old tie-is-"Not sure" rule.
    """
    counts: Counter[str] = Counter()
    best_conf: dict[str, float] = {}
    for frame_objs in seen:
        for cls, conf in frame_objs:
            if conf < cfg.conf_identify:
                continue
            counts[cls] += 1
            best_conf[cls] = max(best_conf.get(cls, 0.0), conf)
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -best_conf[kv[0]], kv[0]))
    if ranked[0][1] < cfg.identify_min_votes:
        return []
    if cfg.identify_max_objects == 1:
        if len(ranked) > 1 and ranked[1][1] == ranked[0][1]:
            return []  # tie: never guess
        return [ranked[0][0]]
    return [cls for cls, count in ranked
            if count >= cfg.identify_min_votes][:cfg.identify_max_objects]


def next_pending_index(items: tuple[ListItem, ...], start: int) -> int | None:
    """Index of the next item after ``start`` that is still pending (wrapping around), or None."""
    n = len(items)
    for k in range(1, n + 1):
        # ``%`` wraps around to the top of the list after the last item.
        i = (start + k) % n
        if items[i].status == PENDING:
            return i
    return None


# --------------------------------------------------------------------------- transitions
# The functions below move the state machine. Each takes a State and returns
# (new State, list of Effects). ``step`` is the only public entry point.


def step(state: State, event: Event, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    """Advance the engine by one event. THE entry point; everything else is called from here.

    state: the current state.
    event: a Control, Frame or Tick.
    now:   current time in seconds (any clock, as long as it is consistent).
    cfg:   guidance settings.
    Returns (new state, effects to perform).

    Flow: 1) handle the event itself, 2) run the timers (time-outs that apply regardless of
    which event arrived), 3) return the combined list of effects.
    """
    if isinstance(event, Control):
        state, fx = _on_control(state, event, now, cfg)
    elif isinstance(event, Frame):
        state, fx = _on_frame(state, event, now, cfg)
    else:
        # A Tick: no direct effects, only the timers below.
        fx = []
    state, tfx = _timers(state, now, cfg)
    return state, fx + tfx


def _stop_tone(s: State) -> tuple[State, list[Effect]]:
    """If a beep is playing, emit a stop-tone effect and forget it."""
    if s.track.tone is None:
        return s, []
    return replace(s, track=replace(s.track, tone=None)), [Tone(interval_ms=None)]


def _streaming_now(s: State, cfg: GuidanceConfig) -> bool:
    """Whether the device has been asked to stream in this state.

    ``State.streaming`` covers the on-demand modes. ``stream_in_idle`` additionally keeps
    video flowing while the switch is ON and the engine is waiting in IDLE; ``stream_in_off``
    keeps it flowing while the switch is OFF (silent monitor mode).
    """
    return (s.streaming
            or (cfg.stream_in_idle and s.mode == Mode.IDLE)
            or (cfg.stream_in_off and s.mode == Mode.STANDBY))


def _leave_find(s: State, cfg: GuidanceConfig, *, keep_stream: bool = False) -> tuple[State, list[Effect]]:
    """Stop any search / identify: tone off, tracking cleared, and usually stream off.

    ``keep_stream`` says whether the caller wants the camera to stay live. The caller decides
    using ``stream_in_idle`` (ON) or ``stream_in_off`` (OFF).
    """
    if not _streaming_now(s, cfg):
        return s, []
    s, fx = _stop_tone(s)
    s = replace(s, target=None, track=Tracking(), other_history=(), identify=None,
                search_ref=None, not_found_said_at=None)
    return s, fx + [Stream(on=True, purpose="find") if keep_stream else Stream(on=False)]


def _start_search(s: State, target: str, now: float, cfg: GuidanceConfig) -> State:
    """Enter SEARCHING for ``target`` with fresh tracking data."""
    return replace(s, mode=Mode.SEARCHING, target=target, track=Tracking(), other_history=(),
                   search_ref=now, not_found_said_at=None, identify=None)


def _announce_item(s: State) -> Say:
    """The "Item 2 of 5: pasta" phrase for the current list item (different if already found)."""
    item = s.items[s.index]
    key = "item_announce_found" if item.status == FOUND else "item_announce"
    return say(key, interrupt=True, n=s.index + 1, total=len(s.items), item=item.name)


def _on_control(s: State, ev: Control, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    """Handle a button / switch event. Checked top to bottom; the first matching rule wins."""
    e = ev.event
    # --- events that work in every mode -------------------------------------------------
    if e == "volume_set":
        # Volume is applied and announced on the device; the server only records it (FR-45).
        return replace(s, volume=ev.value), []
    if e == "shutdown":
        # The device says "Shutting down" from a local clip and powers off by itself.
        s, fx = _leave_find(s, cfg)
        return replace(s, mode=Mode.STANDBY), fx
    if e == "mode_off":
        # Mode switch OFF: silent monitor. Keep the camera/inference running if configured,
        # but produce no guidance speech or tones.
        s, fx = _leave_find(s, cfg, keep_stream=cfg.stream_in_off)
        if cfg.stream_in_off and not any(isinstance(effect, Stream) and effect.on for effect in fx):
            fx.append(Stream(on=True, purpose="find"))
        return replace(s, mode=Mode.STANDBY), fx
    if e == "mode_on":
        if s.mode == Mode.STANDBY:
            if cfg.auto_find_on_mode_on:
                # ON means "guide me now": start Find for the current list item.
                if not s.items:
                    fx: list[Effect] = [say("list_empty", interrupt=True)]
                    if cfg.stream_in_idle:
                        fx.append(Stream(on=True, purpose="find"))
                    return replace(s, mode=Mode.IDLE), fx
                target = s.items[s.index].name
                s = _start_search(s, target, now, cfg)
                return s, [say("searching", interrupt=True, item=target),
                           Stream(on=True, purpose="find")]
            fx = [say("ready", interrupt=True)]
            if cfg.stream_in_idle:
                # Start the live video now; the detector runs, but no search starts until
                # the user presses Enter or triggers Identify.
                fx.append(Stream(on=True, purpose="find"))
            return replace(s, mode=Mode.IDLE), fx
        return s, []  # already active (e.g. TOGGLE springing back to ON)

    # --- everything below needs the device to be switched on ----------------------------
    if s.mode == Mode.STANDBY:
        return s, []  # silent monitor: only the mode switch, volume and shutdown work
    if s.mode == Mode.IDENTIFYING:
        return s, []  # the burst takes about a second; ignore other controls meanwhile

    if e == "identify":
        return _start_identify(s, now, cfg)
    if e in ("next_item", "prev_item"):
        # Browse the list. Cancels any running search first.
        if not s.items:
            return s, [say("list_empty", interrupt=True)]
        s, fx = _leave_find(s, cfg, keep_stream=cfg.stream_in_idle or cfg.auto_find_on_mode_on)
        step_by = 1 if e == "next_item" else -1
        # ``%`` wraps around: after the last item comes the first one again.
        s = replace(s, mode=Mode.IDLE, index=(s.index + step_by) % len(s.items))
        fx.append(_announce_item(s))
        if cfg.auto_find_on_mode_on:
            # ON keeps guiding: start searching for the newly selected item after announcing it.
            target = s.items[s.index].name
            s = _start_search(s, target, now, cfg)
            if not any(isinstance(effect, Stream) and effect.on for effect in fx):
                fx.append(Stream(on=True, purpose="find"))
        return s, fx
    if e == "enter":
        return _on_enter(s, now, cfg)
    if e == "help":
        return s, [say("help", interrupt=True)]
    if e == "status":
        return s, [_status(s)]
    # Unknown event name: ignore it.
    return s, []


def _status(s: State) -> Say:
    """The spoken progress report: "3 of 5 found, current item pasta"."""
    if not s.items:
        return say("list_empty", interrupt=True)
    found = sum(1 for it in s.items if it.status == FOUND)
    if found == len(s.items):
        return say("status_done", interrupt=True, total=len(s.items))
    return say("status", interrupt=True, found=found, total=len(s.items), item=s.items[s.index].name)


def _on_enter(s: State, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    """The Enter button (shutter release) means different things depending on the mode:

    IDLE                 -> start searching for the current item.
    SEARCHING / GUIDING  -> cancel the search.
    WITHIN_REACH         -> check the item off and move to the next pending one.
    """
    if s.mode == Mode.IDLE:
        if not s.items:
            return s, [say("list_empty", interrupt=True)]
        target = s.items[s.index].name
        s = _start_search(s, target, now, cfg)
        return s, [say("searching", interrupt=True, item=target),
                   Stream(on=True, purpose="find")]
    if s.mode in (Mode.SEARCHING, Mode.GUIDING):
        s, fx = _leave_find(s, cfg, keep_stream=cfg.stream_in_idle)
        return replace(s, mode=Mode.IDLE), fx + [say("search_stopped", interrupt=True)]
    if s.mode == Mode.WITHIN_REACH:
        return _check_off(s, cfg, now)
    return s, []


def _check_off(s: State, cfg: GuidanceConfig, now: float) -> tuple[State, list[Effect]]:
    """Mark the current item FOUND, stop the search and move on to the next pending item.

    Used by the shutter press within reach and by the hand guidance ("That's it."). With
    ``auto_find_on_mode_on`` the next item starts searching immediately; the "checked off"
    cue already names it, so no extra "Searching for..." cue is needed.
    """
    target = s.target
    # Build a new items tuple with the current item marked FOUND (all others unchanged).
    # ``enumerate`` gives (position, item) pairs.
    items = tuple(replace(it, status=FOUND) if i == s.index else it for i, it in enumerate(s.items))
    nxt = next_pending_index(items, s.index)
    keep_stream = cfg.stream_in_idle or (cfg.auto_find_on_mode_on and nxt is not None)
    s, fx = _leave_find(s, cfg, keep_stream=keep_stream)
    if nxt is None:
        # Nothing left to find: stay on this item and celebrate.
        s = replace(s, mode=Mode.IDLE, items=items)
        return s, fx + [say("checked_off_done", interrupt=True, item=target)]
    s = replace(s, items=items, index=nxt)
    if cfg.auto_find_on_mode_on:
        s = _start_search(s, items[nxt].name, now, cfg)
        if not any(isinstance(effect, Stream) and effect.on for effect in fx):
            fx.append(Stream(on=True, purpose="find"))
    else:
        s = replace(s, mode=Mode.IDLE)
    return s, fx + [say("checked_off_next", interrupt=True, item=target, next=items[nxt].name)]


def _start_identify(s: State, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    """Begin an Identify burst: remember where to return, ask the device for N frames."""
    resume = s.mode
    s, fx = _stop_tone(s)
    # Burst ids cycle through 1..65535 so they fit in the 16-bit field of the frame header.
    burst_id = (s.burst_counter % 0xFFFF) + 1
    s = replace(s, mode=Mode.IDENTIFYING, burst_counter=burst_id,
                identify=Identify(burst_id=burst_id, started_at=now, resume=resume))
    return s, fx + [say("identifying"),
                    Stream(on=True, purpose="identify", burst=cfg.identify_burst, burst_id=burst_id)]


def _finish_identify(s: State, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    """Count the votes, announce the result, and go back to the mode we came from."""
    ident = s.identify
    # ``assert`` crashes loudly if this is ever None; it documents an assumption.
    assert ident is not None
    if len(ident.picks) < cfg.identify_min_votes:
        cue = say("not_sure_loss", interrupt=True)  # too few frames arrived: never guess
    else:
        # Keep the original single-object vote exactly when only one answer is allowed.
        if cfg.identify_max_objects == 1:
            winner = vote_identify(ident.picks, cfg)
            winners = [winner] if winner is not None else []
        else:
            winners = vote_identify_many(ident.seen, cfg)
        if not winners:
            cue = say("not_sure", interrupt=True)
        elif len(winners) > 1:
            # Multi-object answer: "I see pasta, milk, and rice."
            cue = say("identified_many", interrupt=True, items=winners)
        else:
            winner = winners[0]
            # ``any(...)`` is True if at least one list item matches the winner and is still pending.
            if any(it.name == winner and it.status == PENDING for it in s.items):
                cue = say("identified_on_list", interrupt=True, item=winner)
            else:
                cue = say("identified", interrupt=True, item=winner)
    # FR-34: return to the previous state, resuming the search if one was active
    if ident.resume in FIND_MODES and s.target is not None:
        s = _start_search(s, s.target, now, cfg)
        return s, [cue, Stream(on=True, purpose="find")]
    s = replace(s, mode=Mode.IDLE if ident.resume != Mode.STANDBY else Mode.STANDBY,
                identify=None, target=None)
    keep_stream = cfg.stream_in_idle and s.mode == Mode.IDLE
    return s, [cue, Stream(on=True, purpose="find") if keep_stream else Stream(on=False)]


def _on_frame(s: State, f: Frame, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    """Handle the detector's output for one camera frame.

    Two very different jobs depending on mode:
    - IDENTIFYING: collect one candidate per frame; when the burst is complete, vote.
    - SEARCHING / GUIDING / WITHIN_REACH: find the target box, update tracking, then either
      announce "found" (first stable sighting) or give direction cues.
    """
    if s.mode == Mode.IDENTIFYING:
        ident = s.identify
        # Ignore frames that are not part of the burst we are waiting for.
        if ident is None or f.purpose != "identify" or f.burst_id != ident.burst_id:
            return s, []
        # Keep both the single best candidate and every confident class in this frame.
        ident = replace(
            ident,
            picks=ident.picks + (pick_identify_candidate(f.detections, cfg),),
            seen=ident.seen + (identify_candidates(f.detections, cfg),),
        )
        s = replace(s, identify=ident)
        if len(ident.picks) >= cfg.identify_burst:
            return _finish_identify(s, now, cfg)
        return s, []

    # Frames only matter while a find search is running.
    if s.mode not in FIND_MODES or f.purpose != "find" or s.target is None:
        return s, []

    # Which box (if any) is our target in this frame?
    det = select_target(f.detections, s.target, s.track.last_box, cfg)
    # Slide the sighting history window: append this result, keep only the last stable_n.
    track = replace(s.track, history=(s.track.history + (det,))[-cfg.stable_n:])
    if det is not None:
        track = replace(track, last_seen_at=now, last_box=det)
    s = replace(s, track=track)

    # A visible hand takes priority over camera-direction speech when hand guidance is allowed
    # early: compare the hand center with the item center and say which way to move the hand.
    hand = pick_hand(f.detections, cfg) if cfg.hand_guidance_enabled else None
    speak_directions = not (hand is not None and cfg.hand_guidance_always)

    if s.mode == Mode.SEARCHING:
        if det is not None:
            s = replace(s, search_ref=now)  # resets the "not seeing" timer (FR-12)
        if is_stable(track.history, cfg) and track.last_box is not None:
            # Seen often enough: announce "found" and give the first direction right away.
            s = replace(s, mode=Mode.GUIDING, other_history=())
            fx: list[Effect] = [say("found", interrupt=True, item=s.target)]
            s, gfx = _guide(s, track.last_box, now, cfg, acquired=True,
                            speak_directions=speak_directions)
            fx += gfx
            if hand is not None and (cfg.hand_guidance_always or s.mode == Mode.WITHIN_REACH):
                s, hfx = _guide_hand(s, f, track.last_box, now, cfg, hand=hand)
                fx += hfx
            return s, fx
        # Not found yet: maybe a different list item is right in front of the camera.
        return _check_wrong_item(s, f, det, now, cfg)

    # GUIDING or WITHIN_REACH: nothing to say if the target is not in this frame
    # (the lost time-out in _timers handles a long absence).
    if det is None:
        return s, []
    s, fx = _guide(s, det, now, cfg, acquired=False, speak_directions=speak_directions)
    # Hand cues start as soon as both boxes are visible when hand_guidance_always is on.
    if hand is not None and (cfg.hand_guidance_always or s.mode == Mode.WITHIN_REACH):
        s, hfx = _guide_hand(s, f, det, now, cfg, hand=hand)
        fx += hfx
    return s, fx


def _guide_hand(s: State, f: Frame, item: Detection, now: float,
                cfg: GuidanceConfig, hand: Detection | None = None) -> tuple[State, list[Effect]]:
    """Hand guidance while the target is visible: "Hand left." / "Reach forward." until the
    hand covers the item, then "That's it." (and the item is checked off if configured)."""
    if not cfg.hand_guidance_enabled:
        return s, []
    t = s.track
    if hand is None:
        hand = pick_hand(f.detections, cfg)
    if hand is None:
        return replace(s, track=replace(t, hand_touch_since=None)), []
    if hand_on_item(hand, item, cfg):
        since = t.hand_touch_since if t.hand_touch_since is not None else now
        t = replace(t, hand_touch_since=since)
        s = replace(s, track=t)
        # A brief overlap while the hand passes by is not a grab: it must hold for a moment.
        if now - since < cfg.hand_touch_hold_s or t.hand_said == "hand_on_item":
            return s, []
        s = replace(s, track=replace(t, hand_said="hand_on_item", hand_said_at=now))
        fx: list[Effect] = [say("hand_on_item", interrupt=True)]
        if cfg.hand_auto_check_off:
            s, cfx = _check_off(s, cfg, now)
            fx += cfx
        return s, fx
    word = hand_direction(hand, item, cfg)
    t = replace(t, hand_touch_since=None)
    due = t.hand_said_at is None or now - t.hand_said_at >= cfg.hand_repeat_s
    if word != t.hand_said or due:
        t = replace(t, hand_said=word, hand_said_at=now)
        return replace(s, track=t), [say("hand_dir", hd=word)]
    return replace(s, track=t), []


def _check_wrong_item(s: State, f: Frame, det: Detection | None, now: float,
                      cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    """FR-20: a different known item, confidently detected and centered, said at most every few s."""
    if not cfg.wrong_item_enabled:
        return s, []
    other: str | None = None
    if det is None:
        # Boxes of some other class, confident enough, and near the center of the frame.
        cands = [d for d in f.detections
                 if d.cls != s.target and d.cls not in cfg.non_item_classes
                 and d.conf >= cfg.conf_wrong_item
                 and abs(adjusted_center(d, cfg)[0] - 0.5) < cfg.wrong_item_center_tol
                 and abs(d.cy - 0.5) < cfg.wrong_item_center_tol * 2]
        if cands:
            other = max(cands, key=lambda d: d.conf).cls
    # Same sliding-window idea as the target history: require a few consistent frames.
    hist = (s.other_history + (other,))[-cfg.stable_n:]
    s = replace(s, other_history=hist)
    if other is None or hist.count(other) < cfg.stable_k:
        return s, []
    # Don't nag: repeat at most every wrong_item_repeat_s seconds.
    if s.wrong_item_said_at is not None and now - s.wrong_item_said_at < cfg.wrong_item_repeat_s:
        return s, []
    return replace(s, wrong_item_said_at=now), [say("wrong_item", other=other, item=s.target)]


def _guide(s: State, box: Detection, now: float, cfg: GuidanceConfig,
           acquired: bool, speak_directions: bool = True) -> tuple[State, list[Effect]]:
    """Direction, within-reach, getting-closer and tone cues for one sighting of the target.

    box:              where the target is in this frame.
    acquired:         True on the very first sighting (we always speak the full direction then).
    speak_directions: False while a hand is visible, so the hand cue is the only speech cue.
    """
    t = s.track
    cx, cy = adjusted_center(box, cfg)
    h_dir = classify_horizontal(cx, t.h_dir, cfg)
    v_zone = classify_vertical(cy, t.v_zone, cfg)
    t = replace(t, h_dir=h_dir, v_zone=v_zone)
    fx: list[Effect] = []

    if s.mode == Mode.WITHIN_REACH:
        # Still straight ahead and still big enough? Then just keep the beep updated.
        if h_dir == "straight" and box.h >= cfg.reach_h * cfg.reach_exit_ratio:
            s = replace(s, track=t)
            return _update_tone(s, box, cfg)
        # drifted away: back to guiding and say the new direction right away
        s = replace(s, mode=Mode.GUIDING)
        t = replace(t, reach_since=None, said_h=None, said_at=None,
                    hand_said=None, hand_said_at=None, hand_touch_since=None)
    else:
        # "Within reach" = centered AND the box is tall enough (close to the camera).
        reach = h_dir == "straight" and box.h >= cfg.reach_h
        # Remember when the reach condition first became true; clear it when it stops.
        reach_since = (t.reach_since if t.reach_since is not None else now) if reach else None
        t = replace(t, reach_since=reach_since)
        # It must hold for reach_hold_s seconds to avoid a false "within reach" on a blip.
        if reach_since is not None and now - reach_since >= cfg.reach_hold_s:
            s = replace(s, mode=Mode.WITHIN_REACH, track=t)
            s, tfx = _update_tone(s, box, cfg)
            reach_cue = [] if not speak_directions else [
                say("within_reach", interrupt=True, item=s.target)]
            return s, reach_cue + tfx

    # Decide whether to speak a direction now:
    # - speak_v: the vertical zone changed (or first sighting) -> full "left, eye level" phrase
    # - changed: the horizontal word changed -> short "slightly left" phrase
    # - due:     enough time passed since we last spoke -> repeat as a reminder
    speak_v = acquired or v_zone != t.said_v
    changed = h_dir != t.said_h
    due = t.said_at is None or now - t.said_at >= cfg.direction_repeat_s
    if speak_directions and (changed or due or speak_v):
        if speak_v:
            fx.append(say("direction", h=h_dir, v=v_zone))
        else:
            fx.append(say("direction_h", h=h_dir))
        t = replace(t, said_h=h_dir, said_v=v_zone, said_at=now)
    if cfg.getting_closer_enabled and speak_directions:
        # "Getting closer": the box has grown noticeably since the reference height.
        ref = t.closer_ref_h
        if ref is None or box.h < ref:
            # New (smaller) reference: the user moved back or this is the first measurement.
            t = replace(t, closer_ref_h=box.h)
        elif box.h >= ref * cfg.getting_closer_ratio and not fx and (
                t.closer_said_at is None or now - t.closer_said_at >= cfg.getting_closer_repeat_s):
            # Only say it if nothing else is being said this frame and not too soon after last time.
            fx.append(say("getting_closer"))
            t = replace(t, closer_ref_h=box.h, closer_said_at=now)

    s = replace(s, track=t)
    s, tfx = _update_tone(s, box, cfg)
    return s, fx + tfx


def _update_tone(s: State, box: Detection, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    """Send a new Tone effect only if the pan or interval changed enough to be worth it."""
    if not cfg.tones_enabled:
        return s, []
    pan, interval = tone_for(box, s.mode, cfg)
    prev = s.track.tone
    # The backslash continues a long condition on the next line.
    if prev is not None and abs(prev[0] - pan) < cfg.tone_min_change_pan \
            and abs(prev[1] - interval) < cfg.tone_min_change_ms:
        return s, []
    s = replace(s, track=replace(s.track, tone=(pan, interval)))
    return s, [Tone(pan=pan, interval_ms=interval, freq_hz=cfg.tone_freq_hz,
                    duration_ms=cfg.tone_duration_ms)]


def _timers(s: State, now: float, cfg: GuidanceConfig) -> tuple[State, list[Effect]]:
    """Time-based rules, checked after EVERY event (including plain Ticks).

    1. Lost: while guiding, the target has not been seen for lost_timeout_s -> back to SEARCHING.
    2. Not seeing: while searching, nothing for not_found_prompt_s -> remind "I don't see X".
    3. Identify time-out: the burst never completed -> finish with whatever votes we have.
    """
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
