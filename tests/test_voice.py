"""Voice commands (slr/server/voice.py): parsing, the assistant's replies, and the endpoints."""

import json
import urllib.request
import wave
from pathlib import Path

import pytest

from slr.common.catalog import load_catalog
from slr.common.types import FOUND
from slr.server.list_web import HttpServer, make_handler
from slr.server.session import ListError, ListItem
from slr.server.voice import VoiceAssistant, build_phrases, join_words, load_aliases, parse, words_of

ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "config/catalog.yaml")
KNOWN = CATALOG.item_map()
ALIASES = load_aliases(ROOT / "config/voice_aliases.yaml", KNOWN)
# Exactly what the server listens for: catalog names and aliases plus the voice aliases.
PHRASES = VoiceAssistant(CATALOG, ALIASES, lambda op, arg: "", lambda: ([], 0, KNOWN)).phrases
MODEL = ROOT / "models/vosk/vosk-model-small-en-us-0.15"


def test_words_fold_case_apostrophes_and_hyphens():
    assert words_of("Add Lay's Coca-Cola!") == ["add", "lays", "coca", "cola"]
    assert words_of("add [unk]") == ["add", "[unk]"]


def test_aliases_only_name_known_items():
    assert set(ALIASES) <= set(KNOWN)


@pytest.mark.parametrize("text, action, items", [
    ("add chips", "add", ["chips"]),
    ("Add Coca-Cola", "add", ["coca_cola_can"]),
    ("add a soda", "add", ["soda"]),               # generic soda is still an item
    ("add a Pepsi", "add", ["soda"]),
    ("add Oreos and cookies", "add", ["oreo", "cookies"]),
    ("add a can of Campbell's soup", "add", ["chicken_noodle_soup"]),
    ("add chicken noodle soup", "add", ["chicken_noodle_soup"]),
    ("add cereal", "add", ["cereal"]),
    ("add Cheerios", "add", ["cheerios"]),
    ("add a coke please", "add", ["coca_cola_can"]),
    ("add Lay's chips", "add", ["lays"]),        # the longer name wins over "chips"
    ("add potato chips", "add", ["chips"]),
    ("add chips and milk", "add", ["chips", "milk"]),
    ("tomato sauce", "add", ["tomato_sauce"]),  # an item alone means add
    ("I need spaghetti", "add", ["pasta"]),
    ("add milk chocolate", "add", ["chocolate"]),
    ("remove the milk", "remove", ["milk"]),
    ("take sugar off my list", "remove", ["sugar"]),
    ("I don't need sugar", "remove", ["sugar"]),
    ("what's on my list", "read", []),
    ("read my list", "read", []),
    ("what can I add", "catalog", []),
    ("clear the list", "clear", []),
    ("remove everything", "clear", []),
    ("undo", "undo", []),
    ("yes", "yes", []),
    ("no", "no", []),
    ("help", "help", []),
    ("banana", "unknown", []),
])
def test_parse(text, action, items):
    cmd = parse(text, PHRASES)
    assert (cmd.action, cmd.items) == (action, items)


@pytest.mark.parametrize("text, items, counts", [
    ("add 3 water", ["water"], [3]),
    ("add three bottles of water", ["water"], [3]),
    ("add two milk and chips", ["milk", "chips"], [2, 1]),
    ("add one more coke", ["coca_cola_can"], [1]),
    ("add chips two my list", ["chips"], [1]),     # "two" (heard for "to") before no item
    ("add 500 rice", ["rice"], [20]),               # capped
    ("remove 2 water", ["water"], [2]),
])
def test_parse_quantities(text, items, counts):
    cmd = parse(text, PHRASES)
    assert (cmd.items, cmd.counts) == (items, counts)


def test_recognizer_listens_for_numbers_not_their_homophones(va):
    a, _ = va
    vocab = a.vocabulary()
    assert {"two", "three", "four", "bottles"} <= vocab
    assert not vocab & {"to", "too", "for"}


def test_quantities_add_remove_read_and_undo_as_one_step(va):
    a, lst = va
    assert a.handle("add 3 water and chips") == (True, "Added 3 water and chips. 4 items on your list.")
    assert a.handle("what's on my list")[1] == "You have 4 items: 3 water and chips."
    assert a.handle("remove 2 water")[1] == "Removed 2 water. 2 items on your list."
    assert a.handle("remove 5 chips")[1].startswith("Removed chips.")   # only one was there
    a.handle("add 4 milk")
    assert a.handle("undo")[1].startswith("Took 4 milk off your list.")
    assert lst.names() == ["water"]


def test_items_typed_on_the_page_can_be_said_right_after(tmp_path):
    """A free-text item (data/catalog_user.yaml) is sayable without a restart."""
    cat = load_catalog(ROOT / "config/catalog.yaml", tmp_path / "catalog_user.yaml")
    added = []
    a = VoiceAssistant(cat, {}, lambda op, arg: added.append(arg) or "ok",
                       lambda: ([], 0, cat.item_map()))
    assert parse("add kombucha", a.phrases).items == []
    cat.resolve_or_create("kombucha")          # what the page's add box does
    ok, msg = a.handle("add kombucha")
    assert ok and msg.startswith("Added kombucha.") and added == ["kombucha"]
    assert "kombucha" in a.vocabulary()


def test_parse_keeps_unknown_words_for_a_suggestion():
    cmd = parse("add chipz", PHRASES)
    assert cmd.action == "add" and cmd.items == [] and cmd.leftover == ["chipz"]
    assert parse("add [unk]", PHRASES).unheard


def test_join_words():
    assert join_words(["a"]) == "a"
    assert join_words(["a", "b"]) == "a and b"
    assert join_words(["a", "b", "c"]) == "a, b, and c"


class FakeList:
    """Stands in for the server's list_op / page_data pair."""

    def __init__(self):
        self.items: list[ListItem] = []

    def op(self, name, arg):
        if name == "add":
            if arg not in KNOWN:
                raise ListError(f"'{arg}' is not an item the model knows")
            self.items.append(ListItem(arg))
        elif name == "remove":
            for i, it in enumerate(self.items):
                if it.name == arg:
                    del self.items[i]
                    break
            else:
                raise ListError(f"'{arg}' is not on the list")
        elif name == "clear":
            self.items = []
        return "ok"

    def data(self):
        return list(self.items), 0, KNOWN

    def names(self):
        return [it.name for it in self.items]


@pytest.fixture
def va():
    lst = FakeList()
    return VoiceAssistant(CATALOG, ALIASES, lst.op, lst.data), lst


def test_add_and_remove_reply_with_the_count(va):
    a, lst = va
    assert a.handle("add coke and chips") == (True, "Added Coca-Cola and chips. 2 items on your list.")
    assert a.handle("remove coke") == (True, "Removed Coca-Cola. 1 item on your list.")
    assert lst.names() == ["chips"]
    ok, msg = a.handle("remove milk")
    assert not ok and "Milk is not on your list." in msg


def test_unknown_item_suggests_a_close_name(va):
    a, _ = va
    ok, msg = a.handle("add chipz")
    assert not ok and "I don't know chipz." in msg and "Did you mean chip" in msg
    ok, msg = a.handle("add")
    assert not ok and msg.startswith("Add what?")
    assert a.handle("")[0] is False


def test_clear_needs_a_yes_and_can_be_undone(va):
    a, lst = va
    a.handle("add milk and rice")
    ok, msg = a.handle("clear my list")
    assert "Say yes or no" in msg and lst.names() == ["milk", "rice"]
    assert a.handle("no") == (True, "Okay, I kept your list.")
    assert lst.names() == ["milk", "rice"]
    a.handle("clear my list")
    assert a.handle("yes")[1].startswith("Removed every item.")
    assert lst.names() == []
    a.handle("undo")
    assert sorted(lst.names()) == ["milk", "rice"]
    # A yes with nothing asked does nothing.
    assert a.handle("yes")[0] is False


def test_undo_takes_back_adds_and_removes(va):
    a, lst = va
    a.handle("add tea")
    a.handle("add jam")
    assert a.handle("undo")[1].startswith("Took jam off your list.")
    a.handle("remove tea")
    assert a.handle("undo")[1].startswith("Put tea back on your list.")
    assert lst.names() == ["tea"]
    a.handle("undo")
    assert lst.names() == []
    assert a.handle("undo") == (False, "There is nothing to undo.")


def test_read_list_and_catalog(va):
    a, lst = va
    assert a.handle("what's on my list")[1].startswith("Your list is empty.")
    a.handle("add pasta and Lay's")
    lst.items[0] = ListItem("pasta", FOUND)
    assert a.handle("what's on my list")[1] == (
        "You have 2 items: pasta and Lay's chips. Found so far: pasta.")
    assert "tomato sauce" in a.handle("what can I add")[1]


def test_transcribe_without_a_model_explains_why(va):
    a, _ = va
    with pytest.raises(Exception, match="slr voice download-model"):
        a.transcribe(b"\0\0" * 1600)


# --------------------------------------------------------------------------- HTTP

@pytest.fixture
def server(va):
    a, lst = va
    srv = HttpServer("127.0.0.1", 0, make_handler(lst.data, lst.op, lambda: {}, lambda n: None, voice=a))
    srv.start()
    yield f"http://127.0.0.1:{srv.port}", lst
    srv.stop()


def _post(url, data, ctype="application/x-www-form-urlencoded"):
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Accept": "application/json", "Content-Type": ctype})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def test_typed_command_endpoint(server):
    base, lst = server
    reply = _post(base + "/list/say", b"text=add+coca-cola")
    assert reply["ok"] and reply["heard"] == "add coca-cola"
    assert reply["message"] == "Added Coca-Cola. 1 item on your list."
    assert reply["state"]["counts"] == {"coca_cola_can": 1}


def test_voice_endpoint_without_model_says_how_to_set_it_up(server):
    base, _ = server
    reply = _post(base + "/list/voice/start?rate=16000", b"", "application/octet-stream")
    assert reply["ok"] is False and "download-model" in reply["message"]
    reply = _post(base + "/list/voice/start?rate=5", b"", "application/octet-stream")
    assert reply["ok"] is False
    reply = _post(base + "/list/voice/chunk?session=nope", b"\0\0", "application/octet-stream")
    assert reply["ok"] is False and "Press the button" in reply["message"]


# --------------------------------------------------------------------------- live listening

def test_items_settle_once_another_word_follows():
    assert parse("add milk", PHRASES).settled == 0           # may still become milk chocolate
    cmd = parse("add milk chocolate and", PHRASES)
    assert cmd.items == ["chocolate"] and cmd.settled == 1
    assert parse("add chips and milk", PHRASES).settled == 1


class FakeRecognizer:
    """Plays back a script of Vosk results: each feed() pops the next step.

    A step is ("partial", text) for a running guess or ("final", text) for a finished sentence.
    A final step may add a third element: what Whisper hears for that sentence (see
    FakeChecker; pass ``free=True`` to attach one to the assistant with ``_session``).
    """

    def __init__(self, steps, free=False):
        self.steps = list(steps)
        self.words = []
        self.free = free
        self.step = ("partial", "")
        self.grammar = None

    def stream(self, rate=16000):
        return self

    def set_vocabulary(self, words):
        self.words = sorted(words)

    def AcceptWaveform(self, pcm):
        self.step = self.steps.pop(0) if self.steps else ("partial", "")
        return self.step[0] == "final"

    def Result(self):
        return json.dumps({"text": self.step[1]})

    def PartialResult(self):
        return json.dumps({"partial": self.step[1]})

    def FinalResult(self):
        return json.dumps({"text": ""})


class FakeChecker:
    """Stands in for Whisper: hears the third element of the recognizer's current step."""

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.calls = 0

    def transcribe(self, pcm, rate=16000):
        self.calls += 1
        step = self.recognizer.step
        return step[2] if len(step) > 2 else ""


def _listen(a, steps, free=True):
    """Open a session on scripted results; ``free`` attaches the fake Whisper."""
    a.recognizer = FakeRecognizer(steps)
    a.checker = FakeChecker(a.recognizer) if free else None
    return a.session(a.open_session())


def _session(va, steps):
    a, lst = va
    a.recognizer = FakeRecognizer(steps)
    return a, lst, a.session(a.open_session())


def test_items_are_added_while_the_shopper_is_still_talking(va):
    a, lst, s = _session(va, [
        ("partial", "add"),
        ("partial", "add chips"),              # "chips" might still be "chips ..." : wait
        ("partial", "add chips and"),          # now chips is certain
        ("partial", "add chips and milk"),
        ("partial", "add chips and milk chocolate"),
        ("final", "add chips and milk chocolate"),
    ])
    assert s.feed(b"") == [] and s.feed(b"") == []
    events = s.feed(b"")
    assert [(e["kind"], e["item"]) for e in events] == [("item", "chips")]
    assert lst.names() == ["chips"]
    assert s.feed(b"") == [] and s.feed(b"") == []
    events = s.feed(b"")                                   # the pause
    assert [e.get("item") for e in events] == ["chocolate", None]   # not milk
    assert events[1] == {"kind": "say", "message": "Added chips and chocolate."}
    assert lst.names() == ["chips", "chocolate"]
    events, summary = s.finish()
    assert events == []
    assert summary == "You have 2 items: chips and chocolate."


def _said(events):
    """The sentence the page speaks at a pause, or None."""
    return next((e["message"] for e in events if e["kind"] == "say"), None)


def test_live_session_speaks_at_each_pause_and_reads_the_list_at_the_end(va):
    a, lst, s = _session(va, [
        ("final", "add milk"),
        ("final", "[unk]"),                    # background noise: ignored, nothing said
        ("final", "add three water"),
        ("final", "remove milk"),
        ("final", "remove sugar"),
        ("final", "whats on my list"),
        ("final", "add [unk]"),
    ])
    assert _said(s.feed(b"")) == "Added milk."
    assert s.feed(b"") == []
    assert _said(s.feed(b"")) == "Added 3 water."
    assert _said(s.feed(b"")) == "Removed milk."
    assert _said(s.feed(b"")) == "Sugar is not on your list."
    assert _said(s.feed(b"")) == "You have 3 items: 3 water."
    assert "didn't recognise" in _said(s.feed(b""))
    _, summary = s.finish()
    assert summary == "You have 3 items: 3 water."


def test_live_quantity_waits_for_its_item(va):
    a, lst, s = _session(va, [
        ("partial", "add three"),
        ("partial", "add three water"),        # might still be "three water ..." : wait
        ("partial", "add three water and"),
        ("final", "add three water and two chips"),
    ])
    assert s.feed(b"") == [] and s.feed(b"") == []
    events = s.feed(b"")
    assert [(e["item"], e["count"]) for e in events] == [("water", 3)]
    assert lst.names() == ["water"] * 3
    assert _said(s.feed(b"")) == "Added 3 water and 2 chips."
    assert lst.names() == ["water"] * 3 + ["chips"] * 2


def test_live_clear_still_needs_yes_and_undo_takes_back_one_command(va):
    a, lst, s = _session(va, [
        ("final", "add tea and jam"),
        ("final", "undo"),
        ("final", "clear the list"),
        ("final", "add rice"),                 # a new command cancels the question
        ("final", "yes"),
        ("final", "clear the list"),
        ("final", "yes"),
    ])
    s.feed(b"")
    assert lst.names() == ["tea", "jam"]
    assert _said(s.feed(b"")).startswith("Took jam off")
    assert "Say yes or no" in _said(s.feed(b""))
    s.feed(b"")
    assert _said(s.feed(b"")).startswith("Yes to what?") and lst.names() == ["tea", "rice"]
    s.feed(b"")
    s.feed(b"")
    assert lst.names() == []


def test_a_session_that_heard_nothing_says_so(va):
    a, _, s = _session(va, [])
    s.feed(b"")
    assert s.finish()[1].startswith("I didn't hear a command.")


def test_live_endpoints(server, va):
    base, lst = server
    a, _ = va
    a.recognizer = FakeRecognizer([("partial", "add coke and"), ("final", "add coke and milk")])
    start = _post(base + "/list/voice/start?rate=16000", b"", "application/octet-stream")
    sid = start["session"]
    one = _post(base + f"/list/voice/chunk?session={sid}", b"\0\0" * 100, "application/octet-stream")
    assert [e["item"] for e in one["events"]] == ["coca_cola_can"] and one["state"]["counts"] == {"coca_cola_can": 1}
    two = _post(base + f"/list/voice/chunk?session={sid}", b"\0\0" * 100, "application/octet-stream")
    assert [e.get("item") for e in two["events"]] == ["milk", None]
    assert two["events"][1]["message"] == "Added Coca-Cola and milk."
    end = _post(base + f"/list/voice/end?session={sid}", b"", "application/octet-stream")
    assert end["message"] == "You have 2 items: Coca-Cola and milk."
    assert _post(base + f"/list/voice/end?session={sid}", b"", "application/octet-stream")["ok"] is False


@pytest.mark.skipif(not MODEL.exists(), reason="Vosk model not downloaded")
def test_real_recognition_of_silence_is_not_a_command(va):
    a, _ = va
    a.load_recognizer(MODEL)
    assert a.recognizer is not None
    text = a.transcribe(b"\0\0" * 16000)
    assert a.handle(text)[0] is False


@pytest.mark.skipif(not MODEL.exists(), reason="Vosk model not downloaded")
def test_real_recognition_of_synthesized_speech(va, tmp_path):
    piper = pytest.importorskip("piper")
    voice_file = ROOT / "models/voices/en_US-lessac-medium.onnx"
    if not voice_file.exists():
        pytest.skip("Piper voice not downloaded")
    wav = tmp_path / "s.wav"
    with wave.open(str(wav), "wb") as w:
        piper.PiperVoice.load(str(voice_file)).synthesize_wav("Add Coca-Cola.", w)
    a, lst = va
    a.load_recognizer(MODEL)
    with wave.open(str(wav), "rb") as w:
        text = a.transcribe(w.readframes(w.getnframes()), w.getframerate())
    assert a.handle(text)[0] and lst.names() == ["coca_cola_can"]


@pytest.mark.skipif(not MODEL.exists(), reason="Vosk model not downloaded")
def test_real_streaming_adds_the_first_item_before_the_sentence_ends(va, tmp_path):
    piper = pytest.importorskip("piper")
    voice_file = ROOT / "models/voices/en_US-lessac-medium.onnx"
    if not voice_file.exists():
        pytest.skip("Piper voice not downloaded")
    wav = tmp_path / "s.wav"
    with wave.open(str(wav), "wb") as w:
        piper.PiperVoice.load(str(voice_file)).synthesize_wav("Add chips and milk and Coca-Cola.", w)
    a, lst = va
    a.load_recognizer(MODEL)
    with wave.open(str(wav), "rb") as w:
        rate, pcm = w.getframerate(), w.readframes(w.getnframes())
    s = a.session(a.open_session(rate))
    piece = rate // 4 * 2                     # a quarter second of 16-bit audio
    seen_at = {}
    for i in range(0, len(pcm), piece):
        for ev in s.feed(pcm[i:i + piece]):
            if ev["kind"] == "item":
                seen_at[ev["item"]] = i / len(pcm)
    s.finish()
    assert lst.names() == ["chips", "milk", "coca_cola_can"]
    assert seen_at["chips"] < 0.6 and seen_at["milk"] < 0.9   # well before the end


# --------------------------------------------------------------------------- new items by voice


@pytest.mark.parametrize("text, items, new", [
    ("add kombucha", [], [("kombucha", 1)]),
    ("add two bottles of kombucha", [], [("kombucha", 2)]),
    ("add blue gatorade please", [], [("blue gatorade", 1)]),
    ("add almond milk", [], [("almond milk", 1)]),          # one new name, not milk
    ("add chips and kombucha", ["chips"], [("kombucha", 1)]),
    ("add kombucha and 3 tofu", [], [("kombucha", 1), ("tofu", 3)]),
    ("add whole milk", ["milk"], []),                        # an alias is not new
])
def test_parse_finds_new_item_names(text, items, new):
    cmd = parse(text, PHRASES)
    assert (cmd.action, cmd.items, cmd.new_names) == ("add", items, new)


@pytest.fixture
def real(tmp_path):
    """An assistant on a real catalog and shopping list, so new items can be created."""
    from slr.server.session import ShoppingList

    cat = load_catalog(ROOT / "config/catalog.yaml", tmp_path / "catalog_user.yaml")
    lst = ShoppingList(tmp_path / "list.yaml", cat)

    def op(name, arg):
        cls = lst.add(arg) if name == "add" else lst.remove(arg)
        lst.refresh_known()
        return cls

    a = VoiceAssistant(cat, ALIASES, op, lambda: (list(lst.items), lst.current, cat.item_map()))
    return a, lst, cat


def test_a_new_item_is_created_only_after_a_yes(real):
    a, lst, cat = real
    ok, msg = a.handle("add kombucha")
    assert ok and msg == "I don't have kombucha yet. Add it as a new item? Say yes or no."
    assert lst.items == [] and cat.resolve("kombucha") is None
    assert a.handle("no") == (True, "Okay, I didn't add it.")
    assert lst.items == []
    a.handle("add 2 kombucha")
    assert a.handle("yes") == (True, "Added 2 kombucha as a new item. 2 items on your list.")
    assert [it.name for it in lst.items] == ["kombucha", "kombucha"]
    assert cat.resolve("kombucha").preset is False      # saved to the user catalog
    # From now on it is a known item: no question, and its words are listened for.
    assert a.handle("add kombucha")[1] == "Added kombucha. 3 items on your list."
    assert "kombucha" in a.vocabulary()
    a.handle("undo")
    a.handle("undo")
    assert lst.items == []                              # undo takes back the whole 2


def test_typos_are_not_offered_as_new_items(real):
    a, lst, _ = real
    ok, msg = a.handle("add chipz")
    assert not ok and "Did you mean chips?" in msg and a.pending is None


def test_live_unknown_word_uses_dictation_and_asks(real):
    a, lst, cat = real
    s = _listen(a, [
        ("partial", "add chips and"),
        ("partial", "add chips and [unk]"),
        ("final", "add chips and [unk]", "add chips and kombucha"),
        ("final", "yes"),
    ])
    assert [e["item"] for e in s.feed(b"")] == ["chips"]
    assert s.feed(b"") == []
    events = s.feed(b"")
    assert [e.get("item") for e in events] == [None]      # chips is not added twice
    assert events[0]["message"] == ("Added chips. I don't have kombucha yet. "
                                    "Add it as a new item? Say yes or no.")
    assert _said(s.feed(b"")) == "Added kombucha as a new item. 2 items on your list."
    assert [it.name for it in lst.items] == ["chips", "kombucha"]
    assert s.finish()[1] == "You have 2 items: chips and kombucha."


def test_live_unknown_word_holds_items_until_the_pause(real):
    a, lst, _ = real
    s = _listen(a, [
        ("partial", "add [unk] milk and"),                # might be "almond milk"
        ("final", "add [unk] milk", "add almond milk"),
    ])
    assert s.feed(b"") == [] and lst.items == []
    assert "I don't have almond milk yet." in _said(s.feed(b""))
    assert lst.items == []


def test_whisper_corrects_a_confident_mishearing(real):
    """Vosk heard "almonds milk" (both known) and added them; Whisper heard "almond milk"."""
    a, lst, _ = real
    s = _listen(a, [
        ("partial", "add almonds milk and"),
        ("final", "add almonds milk", "Add almond milk."),
    ])
    assert [e["item"] for e in s.feed(b"")] == ["nuts", "milk"]  # streamed right away
    events = s.feed(b"")
    assert [(e.get("action"), e.get("item")) for e in events] == [
        ("remove", "nuts"), ("remove", "milk"), (None, None)]
    assert events[-1]["message"] == "I don't have almond milk yet. Add it as a new item? Say yes or no."
    assert lst.items == [] and a.history == []                   # nothing left behind


def test_whisper_does_not_override_known_items_it_agrees_with(real):
    a, lst, _ = real
    s = _listen(a, [("final", "add chips and milk", "Add chips and milk.")])
    assert _said(s.feed(b"")) == "Added chips and milk."
    assert a.checker.calls == 1
    s = _listen(a, [("final", "what's on my list", "What's on my list?")])
    s.feed(b"")
    assert a.checker.calls == 0           # questions do not need a second opinion


def test_fix_dictation_reads_a_leading_at_as_add():
    from slr.server.voice import fix_dictation
    assert fix_dictation("At a jar of Nutella.") == "add a jar of Nutella."
    assert fix_dictation("What's on my list?") == "What's on my list?"


def test_without_whisper_unknown_words_are_explained(real):
    a, lst, _ = real
    s = _listen(a, [("final", "add [unk]")], free=False)
    assert "didn't recognise" in _said(s.feed(b""))


@pytest.mark.skipif(not MODEL.exists(), reason="Vosk model not downloaded")
def test_real_speech_adds_a_brand_new_item(real, tmp_path):
    """Piper says 'Add kombucha.' then 'Yes.'; Whisper names the new item."""
    piper = pytest.importorskip("piper")
    voice_file = ROOT / "models/voices/en_US-lessac-medium.onnx"
    if not voice_file.exists():
        pytest.skip("Piper voice not downloaded")
    voice = piper.PiperVoice.load(str(voice_file))
    a, lst, cat = real
    a.load_recognizer(MODEL)
    from slr.common.config import load_server_config
    a.load_checker(load_server_config())
    if a.checker is None:
        pytest.skip("Whisper not downloaded (slr voice download-model)")
    said = []
    s = None
    for sentence in ("Add kombucha.", "Yes."):
        wav = tmp_path / "s.wav"
        with wave.open(str(wav), "wb") as w:
            voice.synthesize_wav(sentence, w)
        with wave.open(str(wav), "rb") as w:
            rate, pcm = w.getframerate(), w.readframes(w.getnframes())
        s = s or a.session(a.open_session(rate))
        pcm += b"\0\0" * rate          # a second of silence ends the sentence
        piece = rate // 4 * 2
        for i in range(0, len(pcm), piece):
            said += [e["message"] for e in s.feed(pcm[i:i + piece]) if e["kind"] == "say"]
    s.finish()
    assert said[0] == "I don't have kombucha yet. Add it as a new item? Say yes or no."
    assert said[1].startswith("Added kombucha as a new item.")
    assert [it.name for it in lst.items] == ["kombucha"]
