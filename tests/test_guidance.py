"""Unit tests for the pure guidance engine (PRD 8.2 to 8.4). The most important test file.

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    ``slr/server/guidance.py`` is the brain of the product: it decides what the device
    should SAY and DO in response to button presses and camera detections. It is written
    as a "pure" function:

        new_state, effects = g.step(old_state, event, now, config)

    It never touches the network, the clock or the speakers. You give it the current
    state, one event (a button press, a camera frame, or a timer tick), the current time,
    and the config; it returns the new state and a list of "effects". That design makes
    it easy to test: the tests here feed in events with made-up timestamps and check the
    effects, so they run in milliseconds with no hardware.

    The modes the engine moves through (``Mode``):
      STANDBY -> IDLE (switch on) -> SEARCHING (press enter) -> GUIDING (target seen
      reliably) -> WITHIN_REACH (target big and centered) -> IDLE (press enter again to
      check the item off). IDENTIFYING is a short side-trip from IDLE or from a search.

    Kinds of effects:
      Say(key, params, interrupt)   speak the phrase named ``key`` from config/phrases.yaml
      Stream(on, purpose, ...)      tell the device to start / stop sending camera frames
      Tone(pan, interval_ms, ...)   the panned guidance beep (interval None = stop)

    "PRD 8.2" etc. are sections of the product spec, PRD.pdf; FR-xx are its requirement
    numbers.

Words you will see:
    Driver        A small helper class (below) that keeps the state and the clock between
                  calls so tests read like a story: press this, show that, wait so long.
    replace(...)  From dataclasses: make a copy of an object with some fields changed
                  (the state objects are immutable, so they are never edited in place).
    parametrize   Runs the same test once per listed row of inputs.
    assert        "Check that this is true; if not, the test fails."
"""

from dataclasses import replace

import pytest

from slr.common.config import GuidanceConfig
from slr.common.types import FOUND, PENDING, Detection, ListItem
from slr.server import guidance as g
from slr.server.guidance import Control, Frame, Mode, Say, Stream, Tick, Tone

# Two configs: the real defaults, and the same with the guidance beep switched off so
# most tests do not have to filter Tone effects out of the results.
CFG = GuidanceConfig()
NO_TONES = GuidanceConfig(tones_enabled=False)


# ---- small helpers used by every test --------------------------------------------------------

def items(*names):
    """Build a shopping list (tuple of ListItem) from item names, all still pending."""
    return tuple(ListItem(n) for n in names)


def det(cls="pasta", conf=0.9, cx=0.5, cy=0.5, w=0.1, h=0.2):
    """One detection box, given by its center and size (fractions 0..1 of the frame)."""
    return Detection.centered(cls, conf, cx, cy, w, h)


def frame(*dets, purpose="find", burst_id=0):
    """A camera-frame event carrying the given detections."""
    return Frame(tuple(dets), purpose, burst_id)


def says(fx):
    """Only the Say effects from a list of effects."""
    return [e for e in fx if isinstance(e, Say)]


def keys(fx):
    """Only the phrase names of the Say effects."""
    return [e.key for e in says(fx)]


def streams(fx):
    """Only the Stream effects."""
    return [e for e in fx if isinstance(e, Stream)]


def tones(fx):
    """Only the Tone effects."""
    return [e for e in fx if isinstance(e, Tone)]


class Driver:
    """Feeds events with an explicit clock and records everything said."""

    def __init__(self, state=None, cfg=NO_TONES, now=0.0):
        # Default starting point: a three-item list, switch already ON (mode IDLE).
        self.s = state if state is not None else replace(g.initial_state(items("pasta", "coffee", "rice")),
                                                          mode=Mode.IDLE)
        self.cfg = cfg
        self.now = now
        self.log = []

    def send(self, event, dt=0.0):
        """Advance the clock by dt seconds, feed one event, keep the new state, return effects."""
        self.now += dt
        self.s, fx = g.step(self.s, event, self.now, self.cfg)
        self.log.extend(fx)
        return fx

    def control(self, name, dt=0.0, value=None):
        """A button / switch event such as "enter", "next_item", "mode_off"."""
        return self.send(Control(name, value), dt)

    def frames(self, n, *dets, dt=0.1):
        """n camera frames, each showing the same detections, dt seconds apart."""
        out = []
        for _ in range(n):
            out += self.send(frame(*dets), dt)
        return out

    def tick(self, dt):
        """A timer tick dt seconds later with no new frame (used for timeouts)."""
        return self.send(Tick(), dt)


def searching_driver(**kw):
    """A driver that has already pressed enter and is searching for pasta."""
    d = Driver(**kw)
    d.control("enter")
    return d


def guiding_driver(**kw):
    """A driver that is searching and has seen pasta on the left for 3 frames (now GUIDING)."""
    d = searching_driver(**kw)
    d.frames(3, det(cx=0.2, h=0.1))
    assert d.s.mode == Mode.GUIDING
    return d


# --------------------------------------------------------------------------- direction math


@pytest.mark.parametrize("cx,expected", [
    (0.5, "straight"), (0.43, "straight"), (0.57, "straight"),
    (0.41, "slightly_left"), (0.30, "slightly_left"), (0.59, "slightly_right"),
    (0.25, "left"), (0.0, "left"), (0.75, "right"), (1.0, "right"),
])
def test_horizontal_zones(cx, expected):
    """Box center x -> straight (within 0.08 of center), slightly left/right, or left/right."""
    assert g.classify_horizontal(cx, None, CFG) == expected


@pytest.mark.parametrize("cy,expected", [
    (0.1, "high"), (0.329, "high"), (0.33, "eye"), (0.5, "eye"), (0.66, "eye"), (0.67, "low"), (0.95, "low"),
])
def test_vertical_zones(cy, expected):
    """Box center y -> high shelf (top third), eye level, or low shelf (bottom third)."""
    assert g.classify_vertical(cy, None, CFG) == expected


def test_horizontal_hysteresis_holds_previous_zone_near_boundary():
    """A box hovering on a zone boundary keeps its old zone until it is clearly past it."""
    # 0.415 is just past the 0.42 straight boundary: raw says slightly_left
    assert g.classify_horizontal(0.415, None, CFG) == "slightly_left"
    # but if we were straight, we stay straight until clearly past the boundary
    assert g.classify_horizontal(0.415, "straight", CFG) == "straight"
    assert g.classify_horizontal(0.39, "straight", CFG) == "slightly_left"
    # and coming back in we keep slightly_left until clearly inside
    assert g.classify_horizontal(0.425, "slightly_left", CFG) == "slightly_left"
    assert g.classify_horizontal(0.45, "slightly_left", CFG) == "straight"


def test_vertical_hysteresis():
    """The same boundary stickiness applies to high / eye / low."""
    assert g.classify_vertical(0.335, "high", CFG) == "high"
    assert g.classify_vertical(0.36, "high", CFG) == "eye"
    assert g.classify_vertical(0.65, "low", CFG) == "low"


def test_flip_horizontal_mirrors_direction():
    """flip_horizontal swaps left and right (for a mirrored camera)."""
    cfg = GuidanceConfig(flip_horizontal=True)
    cx, _ = g.adjusted_center(det(cx=0.1), cfg)
    assert g.classify_horizontal(cx, None, cfg) == "right"


def test_vertical_offset_shifts_zones():
    """vertical_offset moves the shelf zones up or down to compensate for camera tilt."""
    cfg = GuidanceConfig(vertical_offset=0.2)
    _, cy = g.adjusted_center(det(cy=0.25), cfg)
    assert g.classify_vertical(cy, None, cfg) == "eye"


# --------------------------------------------------------------------------- target selection


def test_select_target_ignores_low_confidence_and_other_classes():
    """Only boxes of the wanted class with confidence >= conf_find count as the target."""
    dets = (det("rice", 0.99), det("pasta", 0.3))
    assert g.select_target(dets, "pasta", None, CFG) is None


def test_select_target_prefers_highest_confidence():
    """With two pasta boxes and nothing tracked yet, the more confident one wins."""
    a, b = det(conf=0.6, cx=0.2), det(conf=0.8, cx=0.8)
    assert g.select_target((a, b), "pasta", None, CFG) == b


def test_select_target_prefers_tracked_box():
    """Once we are tracking a box, we stick with the one that overlaps it (no jumping)."""
    tracked = det(conf=0.9, cx=0.2)
    near = det(conf=0.6, cx=0.21)
    far = det(conf=0.95, cx=0.8)
    assert g.select_target((near, far), "pasta", tracked, CFG) == near


# --------------------------------------------------------------------------- list cycling (FR-10)


def test_next_and_prev_announce_items_and_wrap():
    """next / prev move through the list, announce "Item N of M", and wrap around."""
    d = Driver()
    fx = d.control("next_item")
    assert says(fx) == [Say("item_announce", {"n": 2, "total": 3, "item": "coffee"}, True)]
    d.control("next_item")
    fx = d.control("next_item")
    assert says(fx)[0].params["item"] == "pasta"
    fx = d.control("prev_item")
    assert says(fx)[0].params == {"n": 3, "total": 3, "item": "rice"}


def test_announce_found_item():
    """Moving onto an item that is already checked off says so."""
    s = replace(g.initial_state((ListItem("pasta", FOUND), ListItem("coffee"))), mode=Mode.IDLE, index=1)
    d = Driver(state=s)
    fx = d.control("next_item")
    assert keys(fx) == ["item_announce_found"]


def test_empty_list():
    """With no items, every list action says "list empty" and nothing else happens."""
    d = Driver(state=replace(g.initial_state(()), mode=Mode.IDLE))
    assert keys(d.control("next_item")) == ["list_empty"]
    assert keys(d.control("enter")) == ["list_empty"]
    assert keys(d.control("status")) == ["list_empty"]
    assert d.s.mode == Mode.IDLE


def test_next_item_during_search_stops_stream():
    """Changing item while searching cancels the search and stops the camera stream."""
    d = searching_driver()
    fx = d.control("next_item")
    assert Stream(on=False) in fx
    assert d.s.mode == Mode.IDLE and d.s.target is None


# --------------------------------------------------------------------------- search flow


def test_enter_starts_search_and_stream():
    """Pressing enter from IDLE says "Searching for pasta" and turns the camera stream on."""
    d = Driver()
    fx = d.control("enter")
    assert says(fx) == [Say("searching", {"item": "pasta"}, True)]
    assert streams(fx) == [Stream(on=True, purpose="find")]
    assert d.s.mode == Mode.SEARCHING and d.s.target == "pasta"


def test_enter_during_search_cancels():
    """Pressing enter again while searching stops the search."""
    d = searching_driver()
    fx = d.control("enter")
    assert keys(fx) == ["search_stopped"]
    assert Stream(on=False) in fx
    assert d.s.mode == Mode.IDLE


def test_enter_during_guiding_cancels_and_stops_tone():
    """Cancelling while guiding also silences the guidance beep."""
    d = guiding_driver(cfg=CFG)
    fx = d.control("enter")
    assert keys(fx) == ["search_stopped"]
    assert Tone(interval_ms=None) in fx
    assert d.s.mode == Mode.IDLE


def test_not_seeing_prompt_after_timeout_and_rate_limited():
    """After 4 s without the item: "Not seeing pasta. Turn slowly." Then only every 8 s."""
    d = searching_driver()
    assert keys(d.tick(3.9)) == []
    assert says(d.tick(0.1)) == [Say("not_seeing", {"item": "pasta"})]
    assert keys(d.tick(4.0)) == []          # 8 s repeat interval
    assert keys(d.tick(3.9)) == []
    assert keys(d.tick(0.1)) == ["not_seeing"]


def test_not_seeing_timer_resets_on_raw_detection():
    """Any glimpse of the item (even one frame) restarts the 4 s "not seeing" timer."""
    d = searching_driver()
    d.tick(3.0)
    d.frames(1, det())  # a single sighting resets the timer
    assert keys(d.tick(3.5)) == []
    assert keys(d.tick(0.6)) == ["not_seeing"]


def test_stable_detection_needs_k_of_n():
    """"Found" needs the item in 3 of the last 5 frames (stable_k of stable_n)."""
    d = searching_driver()
    d.frames(1, det())
    d.frames(1)  # miss
    d.frames(1)  # miss
    d.frames(1, det())
    assert d.s.mode == Mode.SEARCHING
    fx = d.frames(1, det())
    assert d.s.mode == Mode.GUIDING
    assert keys(fx)[0] == "found"


def test_single_frame_false_positive_is_filtered():
    """One hit followed by two misses, repeated, never reaches 3-of-5: no false "found"."""
    d = searching_driver()
    for _ in range(4):
        d.frames(1, det())
        d.frames(2)
    assert d.s.mode == Mode.SEARCHING
    assert "found" not in keys(d.log)


def test_low_confidence_target_never_found():
    """Detections just under conf_find (0.5) never count, however many there are."""
    d = searching_driver()
    d.frames(10, det(conf=0.49))
    assert d.s.mode == Mode.SEARCHING


def test_found_then_direction_with_vertical_zone():
    """The first cue after "found" gives both directions: "Left. High shelf.\""""
    d = searching_driver()
    fx = d.frames(3, det(cx=0.2, cy=0.2, h=0.1))
    assert says(fx) == [
        Say("found", {"item": "pasta"}, True),
        Say("direction", {"h": "left", "v": "high"}),
    ]


def test_direction_repeats_only_when_changed_or_after_interval():
    """Unchanged direction is repeated every 1.5 s; a changed one is spoken at once."""
    d = guiding_driver()
    d.log.clear()
    d.frames(5, det(cx=0.2, h=0.1), dt=0.2)          # 1.0 s, unchanged
    assert keys(d.log) == []
    d.frames(3, det(cx=0.2, h=0.1), dt=0.2)          # passes 1.5 s since last cue
    assert keys(d.log) == ["direction_h"]
    d.log.clear()
    fx = d.frames(1, det(cx=0.35, h=0.1))            # changed: spoken immediately
    assert says(fx) == [Say("direction_h", {"h": "slightly_left"})]


def test_vertical_zone_spoken_again_when_it_changes():
    """A change of shelf height brings back the two-part cue with the vertical zone."""
    d = guiding_driver()
    fx = d.frames(1, det(cx=0.2, cy=0.8, h=0.1))
    assert says(fx) == [Say("direction", {"h": "left", "v": "low"})]


def test_within_reach_requires_centered_and_big_for_hold_time():
    """"Within reach" needs a big (h >= 0.35), centered box held for a full 1.0 s."""
    d = guiding_driver()
    d.frames(1, det(cx=0.5, h=0.4), dt=0.1)
    assert d.s.mode == Mode.GUIDING
    d.frames(9, det(cx=0.5, h=0.4), dt=0.1)          # 0.9 s held
    assert d.s.mode == Mode.GUIDING
    fx = d.frames(1, det(cx=0.5, h=0.4), dt=0.1)     # 1.0 s held
    assert d.s.mode == Mode.WITHIN_REACH
    assert Say("within_reach", {"item": "pasta"}, True) in fx


def test_within_reach_timer_resets_if_condition_breaks():
    """One frame where the box is too small restarts the 1.0 s hold timer."""
    d = guiding_driver()
    d.frames(8, det(cx=0.5, h=0.4), dt=0.1)
    d.frames(1, det(cx=0.5, h=0.2), dt=0.1)          # too small: reset
    d.frames(8, det(cx=0.5, h=0.4), dt=0.1)
    assert d.s.mode == Mode.GUIDING


def test_off_center_big_box_is_not_within_reach():
    """A big box that is not straight ahead never counts as within reach."""
    d = guiding_driver()
    d.frames(20, det(cx=0.35, h=0.6), dt=0.1)
    assert d.s.mode == Mode.GUIDING


def within_reach_driver(**kw):
    """A driver that has reached WITHIN_REACH on pasta."""
    d = guiding_driver(**kw)
    d.frames(11, det(cx=0.5, h=0.4), dt=0.1)
    assert d.s.mode == Mode.WITHIN_REACH
    return d


def test_within_reach_exit_when_target_drifts():
    """Small shrinkage is tolerated (hysteresis); moving off-center drops back to GUIDING."""
    d = within_reach_driver()
    d.frames(3, det(cx=0.5, h=0.32), dt=0.1)         # above exit ratio: stays
    assert d.s.mode == Mode.WITHIN_REACH
    fx = d.frames(1, det(cx=0.25, h=0.4), dt=0.1)
    assert d.s.mode == Mode.GUIDING
    assert keys(fx) == ["direction_h"] or keys(fx) == ["direction"]


def test_enter_within_reach_checks_off_and_advances():
    """Enter while within reach: item marked found, next item announced, stream off."""
    d = within_reach_driver()
    fx = d.control("enter")
    assert says(fx) == [Say("checked_off_next", {"item": "pasta", "next": "coffee"}, True)]
    assert Stream(on=False) in fx
    assert d.s.mode == Mode.IDLE
    assert d.s.items[0].status == FOUND
    assert d.s.index == 1


def test_check_off_skips_found_items_and_wraps():
    """After checking off the last item, "next" wraps to the first still-pending item."""
    s = replace(g.initial_state((ListItem("pasta"), ListItem("coffee", FOUND), ListItem("rice"))),
                mode=Mode.IDLE, index=2)
    d = Driver(state=s)
    d.control("enter")
    d.frames(3, det("rice", h=0.4))
    d.frames(11, det("rice", h=0.4))
    fx = d.control("enter")
    assert says(fx)[0].params == {"item": "rice", "next": "pasta"}
    assert d.s.index == 0


def test_check_off_last_item():
    """Checking off the only remaining item says it was the last one."""
    s = replace(g.initial_state((ListItem("pasta"),)), mode=Mode.IDLE)
    d = Driver(state=s)
    d.control("enter")
    d.frames(14, det(h=0.4))
    assert d.s.mode == Mode.WITHIN_REACH
    assert keys(d.control("enter")) == ["checked_off_done"]
    assert d.s.items[0].status == FOUND


def test_lost_target_after_timeout_returns_to_searching():
    """1.5 s without the item while guiding: "Lost it.", beep off, back to SEARCHING."""
    d = guiding_driver(cfg=CFG)
    d.frames(5)                                       # 0.5 s of misses, not lost yet
    assert d.s.mode == Mode.GUIDING
    d.tick(0.9)
    assert d.s.mode == Mode.GUIDING                   # 1.4 s since last seen
    fx = d.tick(0.2)
    assert d.s.mode == Mode.SEARCHING
    assert keys(fx) == ["lost"]
    assert Tone(interval_ms=None) in fx
    # must re-acquire stably and say found again
    fx = d.frames(3, det())
    assert keys(fx)[0] == "found"


def test_lost_from_within_reach():
    """The same lost timeout applies in WITHIN_REACH."""
    d = within_reach_driver()
    fx = d.tick(1.6)
    assert keys(fx) == ["lost"] and d.s.mode == Mode.SEARCHING


def test_after_lost_not_seeing_prompt_restarts():
    """After "Lost it." the 4 s "not seeing" prompt timer starts over."""
    d = guiding_driver()
    d.tick(1.6)
    assert keys(d.tick(3.9)) == []
    assert keys(d.tick(0.2)) == ["not_seeing"]


# --------------------------------------------------------------------------- wrong item (FR-20)


def test_wrong_item_warning_requires_stable_centered_confident_other():
    """Looking straight at a different item for 3 frames: "That's rice, not pasta.\""""
    d = searching_driver()
    fx = d.frames(3, det("rice", 0.9))
    assert says(fx) == [Say("wrong_item", {"other": "rice", "item": "pasta"})]


def test_wrong_item_rate_limited():
    """The wrong-item warning repeats at most every 5 s."""
    d = searching_driver()
    d.frames(3, det("rice", 0.9))
    d.log.clear()
    d.frames(20, det("rice", 0.9), dt=0.2)            # 4 s later: still quiet
    assert "wrong_item" not in keys(d.log)
    d.frames(6, det("rice", 0.9), dt=0.2)
    assert keys(d.log).count("wrong_item") == 1


def test_wrong_item_ignores_off_center_or_low_confidence():
    """Other items at the edge of view, or seen with low confidence, are not mentioned."""
    d = searching_driver()
    d.frames(5, det("rice", 0.9, cx=0.85))
    d.frames(5, det("rice", 0.6))
    assert "wrong_item" not in keys(d.log)


def test_wrong_item_can_be_disabled():
    """wrong_item_enabled: false turns the feature off."""
    d = searching_driver(cfg=GuidanceConfig(wrong_item_enabled=False, tones_enabled=False))
    d.frames(5, det("rice", 0.95))
    assert "wrong_item" not in keys(d.log)


# --------------------------------------------------------------------------- identify (FR-30..34)


def test_identify_from_idle_votes_and_returns_to_idle():
    """Identify asks for a 5-frame burst, votes on the class, says "This is tea.", goes IDLE."""
    d = Driver()
    fx = d.control("identify")
    assert d.s.mode == Mode.IDENTIFYING
    burst = d.s.identify.burst_id
    assert Stream(on=True, purpose="identify", burst=5, burst_id=burst) in fx
    for _ in range(4):
        d.send(frame(det("tea", 0.8), purpose="identify", burst_id=burst), 0.1)
    assert d.s.mode == Mode.IDENTIFYING
    fx = d.send(frame(det("tea", 0.8), purpose="identify", burst_id=burst), 0.1)
    assert says(fx) == [Say("identified", {"item": "tea"}, True)]
    assert Stream(on=False) in fx
    assert d.s.mode == Mode.IDLE


def test_identify_on_list_adds_note():
    """If the identified item is on the list, the phrase adds "It's on your list.\""""
    d = Driver()
    d.control("identify")
    b = d.s.identify.burst_id
    for _ in range(5):
        fx = d.send(frame(det("rice", 0.8), purpose="identify", burst_id=b), 0.1)
    assert keys(fx) == ["identified_on_list"]


def test_identify_not_on_list_when_already_found():
    """An item already checked off is not announced as "on your list" again."""
    s = replace(g.initial_state((ListItem("rice", FOUND),)), mode=Mode.IDLE)
    d = Driver(state=s)
    d.control("identify")
    b = d.s.identify.burst_id
    for _ in range(5):
        fx = d.send(frame(det("rice", 0.8), purpose="identify", burst_id=b), 0.1)
    assert keys(fx) == ["identified"]


def test_identify_not_sure_when_votes_low_confidence():
    """Five consistent but low-confidence votes (0.5 < conf_identify) give "Not sure.\""""
    d = Driver()
    d.control("identify")
    b = d.s.identify.burst_id
    for _ in range(5):
        fx = d.send(frame(det("rice", 0.5), purpose="identify", burst_id=b), 0.1)
    assert keys(fx) == ["not_sure"]


def test_identify_not_sure_when_votes_split():
    """No class reaches 3 of 5 votes: "Not sure.\""""
    d = Driver()
    d.control("identify")
    b = d.s.identify.burst_id
    for cls in ("rice", "pasta", "rice", "pasta", "tea"):
        fx = d.send(frame(det(cls, 0.9), purpose="identify", burst_id=b), 0.1)
    assert keys(fx) == ["not_sure"]


def test_identify_too_few_frames_says_not_sure_without_guessing():
    """If frames were lost and only 2 arrived by the 1.5 s timeout, do not guess."""
    d = Driver()
    d.control("identify")
    b = d.s.identify.burst_id
    d.send(frame(det("rice", 0.99), purpose="identify", burst_id=b), 0.1)
    d.send(frame(det("rice", 0.99), purpose="identify", burst_id=b), 0.1)
    fx = d.tick(1.3)                                  # 1.5 s since start
    assert keys(fx) == ["not_sure_loss"]
    assert d.s.mode == Mode.IDLE


def test_identify_timeout_with_enough_frames_still_votes():
    """3 of 5 frames arrived before the timeout: that is enough to vote."""
    d = Driver()
    d.control("identify")
    b = d.s.identify.burst_id
    for _ in range(3):
        d.send(frame(det("rice", 0.9), purpose="identify", burst_id=b), 0.1)
    fx = d.tick(1.2)
    assert keys(fx) == ["identified_on_list"]


def test_identify_picks_center_detection_per_frame():
    """Within one frame, the box nearest the center wins over a more confident one at the edge."""
    dets = (det("rice", 0.95, cx=0.05, cy=0.05), det("tea", 0.85, cx=0.5, cy=0.5))
    assert g.pick_identify_candidate(dets, CFG) == ("tea", 0.85)


def test_identify_ignores_stale_or_find_frames():
    """Frames from an old burst, or Find-mode frames, do not count as identify votes."""
    d = Driver()
    d.control("identify")
    b = d.s.identify.burst_id
    d.send(frame(det("rice"), purpose="identify", burst_id=b + 7), 0.1)
    d.send(frame(det("rice"), purpose="find"), 0.1)
    assert d.s.identify.picks == ()


def test_identify_during_search_resumes_search():
    """Identify in the middle of a search answers, then goes back to searching for pasta."""
    d = guiding_driver(cfg=CFG)
    fx = d.control("identify")
    assert Tone(interval_ms=None) in fx
    b = d.s.identify.burst_id
    for _ in range(5):
        fx = d.send(frame(det("coffee", 0.9), purpose="identify", burst_id=b), 0.1)
    assert keys(fx) == ["identified_on_list"]
    assert Stream(on=True, purpose="find") in fx
    assert d.s.mode == Mode.SEARCHING and d.s.target == "pasta"


def test_controls_ignored_while_identifying():
    """Button presses during the short identify burst are ignored."""
    d = Driver()
    d.control("identify")
    assert d.control("next_item") == []
    assert d.control("identify") == []
    assert d.s.mode == Mode.IDENTIFYING


def test_burst_ids_increment():
    """Each identify gets a new burst id so late frames from the last one can be ignored."""
    d = Driver()
    d.control("identify")
    first = d.s.identify.burst_id
    d.tick(2.0)
    d.control("identify")
    assert d.s.identify.burst_id == first + 1


# --------------------------------------------------------------------------- standby / modes


def test_mode_off_from_any_state_goes_standby_and_stops_stream():
    """Switch OFF: say "Paused", stop the stream and the beep, enter STANDBY."""
    d = guiding_driver(cfg=CFG)
    fx = d.control("mode_off")
    assert d.s.mode == Mode.STANDBY
    assert keys(fx) == ["paused"]
    assert Stream(on=False) in fx and Tone(interval_ms=None) in fx


def test_mode_on_from_standby_says_ready():
    """Switch ON from STANDBY: "Ready." and IDLE."""
    d = Driver(state=g.initial_state(items("pasta")))
    assert d.s.mode == Mode.STANDBY
    fx = d.control("mode_on")
    assert keys(fx) == ["ready"] and d.s.mode == Mode.IDLE


def test_mode_on_while_active_is_ignored():
    """A repeated mode_on (e.g. after a reconnect) does not interrupt a running search."""
    d = searching_driver()
    assert d.control("mode_on") == []
    assert d.s.mode == Mode.SEARCHING


def test_only_mode_switch_works_in_standby():
    """In STANDBY every button just reminds you to turn the switch on."""
    d = Driver(state=g.initial_state(items("pasta")))
    for ev in ("enter", "next_item", "identify", "help"):
        fx = d.control(ev)
        assert keys(fx) == ["standby_hint"]
        assert d.s.mode == Mode.STANDBY


def test_frames_ignored_when_not_searching():
    """Camera frames arriving in IDLE have no effect."""
    d = Driver()
    assert d.frames(5, det()) == []
    assert d.s.mode == Mode.IDLE


def test_volume_is_recorded_silently():
    """The device reports its volume level; the engine just remembers it."""
    d = Driver()
    assert d.control("volume_set", value=7) == []
    assert d.s.volume == 7


def test_status_and_help():
    """status: "1 of 2 items found. Current item: milk."; help: the help text; all done."""
    s = replace(g.initial_state((ListItem("pasta", FOUND), ListItem("milk"))), mode=Mode.IDLE, index=1)
    d = Driver(state=s)
    assert says(d.control("status")) == [Say("status", {"found": 1, "total": 2, "item": "milk"}, True)]
    assert keys(d.control("help")) == ["help"]
    s = replace(s, items=(ListItem("pasta", FOUND),), index=0)
    d = Driver(state=s)
    assert keys(d.control("status")) == ["status_done"]


def test_shutdown_stops_streaming_silently():
    """The shutdown combo stops the stream without speaking (the device says "Shutting down")."""
    d = searching_driver()
    fx = d.control("shutdown")
    assert keys(fx) == [] and Stream(on=False) in fx
    assert d.s.mode == Mode.STANDBY


# --------------------------------------------------------------------------- tones (FR-21)


def test_tone_pans_toward_target_and_speeds_up_when_centered_and_close():
    """The beep comes from the target's side and repeats faster as you get close and centered."""
    # tone_for returns (pan, interval_ms).
    far_left = g.tone_for(det(cx=0.1, h=0.05), Mode.GUIDING, CFG)
    centered_big = g.tone_for(det(cx=0.5, h=0.35), Mode.GUIDING, CFG)
    assert far_left[0] < -0.5
    assert centered_big[0] == 0
    assert centered_big[1] < far_left[1]
    assert centered_big[1] == CFG.tone_interval_min_ms
    assert g.tone_for(det(cx=0.5, h=0.1), Mode.WITHIN_REACH, CFG)[1] == CFG.tone_interval_min_ms


def test_tone_emitted_on_change_only():
    """A new Tone effect is sent only when pan or interval changes noticeably."""
    d = searching_driver(cfg=CFG)
    fx = d.frames(3, det(cx=0.2, h=0.1))
    assert len(tones(fx)) == 1
    assert tones(d.frames(1, det(cx=0.21, h=0.1))) == []
    assert len(tones(d.frames(1, det(cx=0.6, h=0.1)))) == 1


def test_tones_disabled():
    """tones_enabled: false means no Tone effects at all."""
    d = searching_driver(cfg=NO_TONES)
    d.frames(10, det(cx=0.2))
    assert tones(d.log) == []


# --------------------------------------------------------------------------- getting closer (FR-22)


def test_getting_closer_when_box_grows():
    """Optional feature: when the box grows by 30% say "Getting closer.\""""
    cfg = GuidanceConfig(getting_closer_enabled=True, tones_enabled=False)
    d = searching_driver(cfg=cfg)
    d.frames(3, det(cx=0.2, h=0.1))
    d.log.clear()
    d.frames(1, det(cx=0.2, h=0.11))
    d.frames(1, det(cx=0.2, h=0.14))
    assert "getting_closer" in keys(d.log)


# --------------------------------------------------------------------------- list edits


def test_with_items_clamps_index():
    """When the list is edited externally, the current index stays within the new list."""
    s = replace(g.initial_state(items("a", "b", "c")), index=2)
    assert g.with_items(s, items("a")).index == 0
    assert g.with_items(s, ()).index == 0
    assert g.with_items(s, items("a", "b", "c", "d")).index == 2


def test_next_pending_index():
    """Find the next item that is still pending, starting from a position and wrapping."""
    its = (ListItem("a", FOUND), ListItem("b"), ListItem("c", FOUND))
    assert g.next_pending_index(its, 1) == 1
    assert g.next_pending_index(its, 2) == 1
    assert g.next_pending_index(tuple(ListItem(x, FOUND) for x in "ab"), 0) is None


def test_step_is_pure():
    """Calling step twice with the same inputs gives the same outputs and leaves the input alone."""
    s = searching_driver().s
    s1, fx1 = g.step(s, frame(det()), 5.0, CFG)
    s2, fx2 = g.step(s, frame(det()), 5.0, CFG)
    assert s1 == s2 and fx1 == fx2
    assert s.track.history == ()  # input state untouched
