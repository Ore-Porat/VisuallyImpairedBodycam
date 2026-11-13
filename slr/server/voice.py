"""Voice commands for the shopping list page: "add chips", "remove milk", "what's on my list".

The shopper presses one large button on the list page and talks. The browser streams the
microphone to the laptop in quarter-second pieces; this module turns it into text with Vosk
(offline speech recognition), works out what the command means, and applies it to the list
through the same ``list_op`` the page buttons use. Items are added while the shopper is still
talking ("add chips and milk": chips goes on the list as soon as "and" is heard); when
listening stops, one sentence sums up the changes ("Added chips and milk. 3 items on your
list.").

WHAT THIS FILE DOES
-------------------
1. ``parse(text, phrases)``: text -> ``Command`` (an action and the items it names). Pure
   Python with no speech code, so it is easy to test with typed sentences.
2. ``VoiceAssistant``: carries out commands. It remembers recent changes for "undo" and asks
   "are you sure?" before clearing the whole list. ``VoiceSession`` is one press of the talk
   button: it is fed audio piece by piece and applies items as soon as they are certain.
3. ``SpeechRecognizer``: wraps a Vosk model that only listens for the words this module knows
   (command words plus every item name). That is fast and very accurate for known items, and
   it streams, so items are added while the shopper is still talking.
4. ``SentenceChecker``: Whisper, a stronger offline model, re-reads each finished sentence
   that adds or removes something or has a word Vosk did not know. Vosk cannot hear a word
   outside its list, and its free dictation turns product names into nonsense ("add
   kombucha" -> "outcome butcher"); Whisper gets them right. When Whisper hears a name that
   is not an item yet, the shopper is asked "Add it as a new item?" and a yes creates it,
   just like typing it on the page.
5. ``build_voice`` and ``download_model``: setup used by main.py, ``slr list web`` and
   ``slr voice download-model``.

WORDS YOU WILL SEE
------------------
- class id: the internal item name, e.g. ``tomato_sauce`` (see config/classes.yaml).
- alias: another way to say an item, e.g. "coke" for Coca-Cola (config/voice_aliases.yaml).
- phrase: an alias split into words, e.g. ("coca", "cola"). Matching works on words.
- grammar: the list of words Vosk is allowed to hear. Anything else comes out as "[unk]".
- PCM: raw audio samples. The page sends 16-bit mono samples, 16000 per second.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import secrets
import shutil
import tempfile
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import yaml

from slr.common.types import FOUND

log = logging.getLogger(__name__)

UNK = "[unk]"   # what Vosk outputs for a sound that is not one of its grammar words

# Command words. A sentence may mix them with item names in any order ("chips, add").
ADD_WORDS = {"add", "put", "buy", "get", "need", "want", "grab", "pick", "include"}
REMOVE_WORDS = {"remove", "delete", "drop", "take", "erase", "cross", "rid", "minus"}
NEGATIONS = {"dont", "not", "without"}            # "I don't need milk" -> remove milk
CLEAR_WORDS = {"clear", "empty", "wipe"}
ALL_WORDS = {"everything", "all"}
# No bare "list" here: with the microphone open, a stray "list" must not trigger a reading.
READ_WORDS = {"what", "whats", "read", "repeat", "tell", "items"}
CATALOG_WORDS = {"can", "available", "options", "sell"}   # "what can I add?"
UNDO_WORDS = {"undo", "oops", "back"}
YES_WORDS = {"yes", "yeah", "yep", "sure", "okay", "ok", "confirm", "correct"}
NO_WORDS = {"no", "nope", "cancel", "stop", "never", "keep"}
HELP_WORDS = {"help", "how", "commands"}
# Words that carry no meaning here; they are in the grammar so Vosk does not force them
# into an item name.
FILLER = {"please", "a", "an", "some", "the", "to", "my", "of", "you", "i", "me", "would",
          "like", "on", "onto", "off", "from", "for", "and", "also", "too", "with", "another",
          "more", "is", "are", "it", "that", "this",
          "in", "do", "does", "have", "there", "out", "now", "mind", "thanks", "thank", "up",
          "go", "again", "start", "over", "hey", "um", "uh", "be", "will", "just", "list"}

# Quantities: "add 3 water", "add three bottles of water". Typed digits work too.
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
           "twenty": 20}
MAX_QUANTITY = 20
# Words allowed between a number and its item: "three BOTTLES OF water".
UNIT_WORDS = {"bottle", "bottles", "bag", "bags", "box", "boxes", "can", "cans", "jar", "jars",
              "pack", "packs", "packet", "packets", "carton", "cartons", "of", "more", "extra"}
# Sound exactly like numbers ("two" / "to"). They are kept out of what the recognizer listens
# for, so it always writes the number; a stray "two" before a non-item is ignored anyway.
HOMOPHONES = {"to", "too", "for"}

COMMAND_WORDS = (ADD_WORDS | REMOVE_WORDS | NEGATIONS | CLEAR_WORDS | ALL_WORDS | READ_WORDS
                 | CATALOG_WORDS | UNDO_WORDS | YES_WORDS | NO_WORDS | HELP_WORDS)

HELP_TEXT = ("Say add, then an item, like: add chips, or add 3 water. Something new works "
             "too: I will ask before adding it. You can also say remove milk, what's on my "
             "list, what can I add, undo, or clear the list.")

PENDING_CLEAR_S = 20.0   # how long "Clear the whole list?" waits for a yes
HISTORY_LEN = 20         # how many changes "undo" can take back
SESSION_IDLE_S = 60.0    # a listening session with no audio for this long is dropped


def words_of(text: str) -> list[str]:
    """Lower-case words with apostrophes and hyphens folded: "Lay's Coca-Cola" -> lays coca cola."""
    text = text.lower().replace("'", "").replace("’", "").replace("-", " ")
    return re.findall(r"\[unk\]|[a-z0-9]+", text)


def build_phrases(known: dict[str, str], aliases: dict[str, list[str]]) -> dict[tuple[str, ...], str]:
    """Every way to say every item, as word tuples: ("tomato", "sauce") -> "tomato_sauce".

    known:   item id -> spoken name (the catalog's addable items, without the hand).
    aliases: class id -> extra names (from config/voice_aliases.yaml).
    The id and the spoken name always count; aliases are added on top.
    """
    phrases: dict[tuple[str, ...], str] = {}
    for cls, spoken in known.items():
        for name in [cls.replace("_", " "), spoken, *aliases.get(cls, [])]:
            words = tuple(words_of(name))
            if not words:
                continue
            if words in phrases and phrases[words] != cls:
                log.warning("voice alias %r is used for both %s and %s; keeping %s",
                            name, phrases[words], cls, phrases[words])
                continue
            phrases[words] = cls
    return phrases


def load_aliases(path: Path, known: dict[str, str]) -> dict[str, list[str]]:
    """Read config/voice_aliases.yaml. A missing file just means no extra names."""
    if not path.exists():
        log.warning("voice aliases file %s not found; only the item names will work", path)
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, list[str]] = {}
    for cls, names in (data.get("aliases") or {}).items():
        if cls not in known:
            log.warning("voice alias for unknown item %r in %s is ignored", cls, path)
            continue
        out[str(cls)] = [str(n) for n in (names or [])]
    return out


# --------------------------------------------------------------------------- parsing

@dataclass
class Command:
    """What one spoken sentence asks for.

    action:   add, remove, read, catalog, undo, clear, yes, no, help or unknown.
    items:    class ids named in the sentence, in order.
    counts:   how many of each item ("add 3 water" -> items ["water"], counts [3]).
    leftover: words that were neither command words nor item names (maybe a misheard item).
    unheard:  True if the recognizer heard a word outside its vocabulary ("[unk]").
    settled:  how many of ``items`` are followed by another word. While the shopper is still
              talking, only those are certain: "add milk" may yet become "add milk chocolate".
    new_names: names that are not items yet, with their counts: "add two kombucha" ->
              [("kombucha", 2)]. An unknown word right before an item joins it, so
              "almond milk" is one new name, not "almond" plus milk.
    """
    action: str
    items: list[str] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    leftover: list[str] = field(default_factory=list)
    unheard: bool = False
    settled: int = 0
    new_names: list[tuple[str, int]] = field(default_factory=list)


def number_of(word: str) -> int | None:
    """ "three" or "3" -> 3; anything else -> None."""
    if word.isdigit():
        return int(word)
    return NUMBERS.get(word)


def parse(text: str, phrases: dict[tuple[str, ...], str]) -> Command:
    """Turn a sentence into a ``Command``.

    Item names are found first, longest match wins, so "lays chips" is the Lay's item and
    not "chips". A number counts for the item right after it ("3 water", "three bottles of
    water"). The remaining words decide the action.
    """
    tokens = words_of(text)
    longest = max((len(p) for p in phrases), default=1)
    items: list[str] = []
    counts: list[int] = []
    rest: list[str] = []
    settled = 0
    quantity: int | None = None   # a number waiting for its item
    new_names: list[tuple[str, int]] = []
    run: list[str] = []           # unknown words in a row: maybe a new item's name
    run_qty: int | None = None

    def clamp(q: int | None) -> int:
        return max(1, min(q or 1, MAX_QUANTITY))

    def close_run() -> None:
        nonlocal run, run_qty
        if run:
            new_names.append((" ".join(run), clamp(run_qty)))
        run, run_qty = [], None

    i = 0
    while i < len(tokens):
        for n in range(min(longest, len(tokens) - i), 0, -1):
            cls = phrases.get(tuple(tokens[i:i + n]))
            if cls:
                if run:
                    # "almond milk": the unknown word describes this item, so the whole
                    # thing is one new name.
                    run.extend(tokens[i:i + n])
                    close_run()
                else:
                    items.append(cls)
                    counts.append(clamp(quantity))
                quantity = None
                i += n
                if i < len(tokens):
                    settled = len(items)
                break
        else:
            word = tokens[i]
            i += 1
            if number_of(word) is not None:
                close_run()
                quantity = number_of(word)
            elif quantity is not None and word in UNIT_WORDS:
                pass   # "three bottles of ...": keep waiting for the item
            elif word not in COMMAND_WORDS and word not in FILLER and word != UNK:
                if not run:
                    run_qty, quantity = quantity, None
                run.append(word)
                rest.append(word)
            else:
                close_run()
                quantity = None   # "two my list": the number belonged to nothing
                rest.append(word)
    close_run()

    words = set(rest)
    leftover = [w for w in rest if w not in COMMAND_WORDS and w not in FILLER and w != UNK]
    cmd = Command("unknown", items, counts, leftover, UNK in words, settled, new_names)

    if words & UNDO_WORDS and not items:
        cmd.action = "undo"
    elif not items and words & YES_WORDS:
        cmd.action = "yes"
    elif not items and words & NO_WORDS:
        cmd.action = "no"
    elif not items and (words & CLEAR_WORDS or (words & REMOVE_WORDS and words & ALL_WORDS)):
        cmd.action = "clear"
    elif not items and words & READ_WORDS and words & CATALOG_WORDS:
        cmd.action = "catalog"
    elif not items and words & READ_WORDS:
        cmd.action = "read"
    elif not items and words & HELP_WORDS:
        cmd.action = "help"
    elif words & (REMOVE_WORDS | NEGATIONS) or (items and words & CLEAR_WORDS):
        cmd.action = "remove"
    elif words & ADD_WORDS or items:
        cmd.action = "add"   # an item on its own ("chips") means add it
    return cmd


def join_words(names: list[str]) -> str:
    """["a"] -> "a";  ["a", "b"] -> "a and b";  ["a", "b", "c"] -> "a, b, and c"."""
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def amount(n: int, name: str) -> str:
    """ "water" for one, "3 water" for three."""
    return name if n == 1 else f"{n} {name}"


def count_phrase(n: int) -> str:
    """How many items, as a short sentence: "Your list is empty." / "3 items on your list." """
    if n == 0:
        return "Your list is empty."
    return f"{n} item{'s' if n != 1 else ''} on your list."


# --------------------------------------------------------------------------- the assistant

class VoiceUnavailable(RuntimeError):
    """Speech recognition cannot run (no model downloaded, still loading, or Vosk missing)."""


class VoiceAssistant:
    """Carries out spoken commands on the shopping list and phrases the spoken replies.

    items:     the item ``Catalog`` (slr/common/catalog.py), or a plain dict of id -> spoken
               name (tests). With a catalog the names are re-read before each command, so an
               item typed in on the web page can be said out loud right after.
    aliases:   item id -> extra spoken names (config/voice_aliases.yaml). A catalog item's own
               ``aliases`` count too.
    list_op:   the page's edit callback: list_op("add", "chips") -> message; raises on error.
    page_data: returns (items, current index, known[, uncurated]), like the page uses.
    """

    def __init__(self, items, aliases: dict[str, list[str]],
                 list_op: Callable[[str, str], str], page_data: Callable[[], tuple]):
        self.items = items
        self.extra_aliases = aliases
        self.known: dict[str, str] = {}
        self.phrases: dict[tuple[str, ...], str] = {}
        self._names_key = None
        self.list_op = list_op
        self.page_data = page_data
        self.recognizer: SpeechRecognizer | None = None
        self.checker: SentenceChecker | None = None   # Whisper, for new item names
        self.status = "Voice commands are not set up. Run: slr voice download-model."
        self.history: list[tuple[str, list[str]]] = []   # ("add" | "remove" | "clear", ids)
        # A yes/no question waiting for its answer: (kind, time asked, data), kind "clear"
        # ("Remove all items?") or "new" ("Add kombucha as a new item?").
        self.pending: tuple[str, float, object] | None = None
        self.sessions: dict[str, VoiceSession] = {}      # live listening, by session id
        self._lock = threading.Lock()                    # one command at a time

        self.refresh_names()

    def refresh_names(self) -> None:
        """Re-read item names (and the words the recognizer listens for) if they changed."""
        if isinstance(self.items, dict):
            known, aliases = dict(self.items), {}
        else:
            known = self.items.item_map()
            aliases = {it.id: list(it.aliases) for it in self.items.items if it.aliases}
        for cls, names in self.extra_aliases.items():
            aliases[cls] = aliases.get(cls, []) + list(names)
        key = (tuple(sorted(known.items())), tuple(sorted((k, tuple(v)) for k, v in aliases.items())))
        if key == self._names_key:
            return
        self._names_key = key
        self.known = known
        self.phrases = build_phrases(known, aliases)
        if self.recognizer is not None:
            self.recognizer.set_vocabulary(self.vocabulary())

    # ---------------------------------------------------------------- speech -> text

    def vocabulary(self) -> set[str]:
        """Every word a command can contain: the recognizer listens for exactly these."""
        words = set(COMMAND_WORDS | FILLER | UNIT_WORDS) | set(NUMBERS)
        for phrase in self.phrases:
            words.update(phrase)
        return words - HOMOPHONES

    def load_recognizer(self, model_dir: Path) -> None:
        """Load the Vosk model (about a second). Safe to run in a background thread."""
        if not model_dir.exists():
            self.status = ("Voice commands are not set up: the speech model is missing. "
                           "Run: slr voice download-model.")
            log.warning("voice: no speech model at %s (run `slr voice download-model`)", model_dir)
            return
        self.status = "Speech recognition is still starting. Try again in a moment."
        try:
            self.recognizer = SpeechRecognizer(model_dir, self.vocabulary())
        except Exception as e:   # vosk not installed, damaged model, ...
            self.status = "Voice commands are not working on this laptop. Please type a command."
            log.warning("voice: could not load the speech model: %s", e)
            return
        log.info("voice commands ready (%d words)", len(self.recognizer.words))

    def load_checker(self, cfg) -> None:
        """Load Whisper for new item names (a few seconds). Safe in a background thread.

        Without it, voice still handles every known item; a new one gets "I didn't recognise
        that item" and can be typed on the page instead.
        """
        try:
            self.checker = SentenceChecker.load(cfg)
        except Exception as e:   # not installed, not downloaded, no GPU libraries, ...
            log.warning("voice: new items by voice are off (Whisper did not load: %s). "
                        "Run `slr voice download-model` to enable them.", e)
            return
        log.info("voice: Whisper %s on %s ready for new item names",
                 self.checker.model_name, self.checker.device)

    def transcribe(self, pcm: bytes, rate: int = 16000) -> str:
        """Raw 16-bit mono audio -> text. Raises VoiceUnavailable with a speakable reason."""
        if self.recognizer is None:
            raise VoiceUnavailable(self.status)
        return self.recognizer.transcribe(pcm, rate)

    # ---------------------------------------------------------------- text -> list edits

    def _name(self, cls: str) -> str:
        return self.known.get(cls, cls.replace("_", " "))

    def _count(self) -> int:
        return len(self.page_data()[0])

    def _remember(self, action: str, ids: list[str]) -> None:
        self.history.append((action, ids))
        del self.history[:-HISTORY_LEN]

    def _suggest(self, leftover: list[str]) -> str:
        """ "Did you mean chips?" for a word that sounds close to an item name, else ""."""
        if not leftover:
            return ""
        spoken = {" ".join(p): cls for p, cls in self.phrases.items()}
        guess = difflib.get_close_matches(" ".join(leftover), list(spoken), n=1, cutoff=0.6)
        return f" Did you mean {self._name(spoken[guess[0]])}?" if guess else ""

    def can_create(self) -> bool:
        """True when unknown names can become new items (a real catalog is attached)."""
        return not isinstance(self.items, dict)

    def offer_new(self, names: list[tuple[str, int]]) -> str:
        """Ask before creating new items; a yes adds them (see ``_handle``).

        A name that is a near-miss of a known item ("chipz") gets "Did you mean chips?"
        instead: a typo should not quietly become a product.
        """
        spoken = {" ".join(p): cls for p, cls in self.phrases.items()}
        offer, notes = [], []
        for name, n in names:
            close = difflib.get_close_matches(name, list(spoken), n=1, cutoff=0.85)
            if close or not self.can_create():
                notes.append(f"I don't know {name}.{self._suggest(name.split())}")
            else:
                offer.append((name, n))
        if offer:
            self.pending = ("new", time.monotonic(), offer)
            listed = join_words([name for name, _ in offer]).replace(" and ", " or ")
            one = len(offer) == 1
            what = (("it" if offer[0][1] == 1 else str(offer[0][1])) if one else "them")
            notes.append(f"I don't have {listed} yet. Add {what} as a new "
                         f"item{'' if one else 's'}? Say yes or no.")
        return " ".join(notes)

    def _add_new(self, names: list[tuple[str, int]]) -> tuple[bool, str]:
        """The shopper said yes: create the items (exactly like typing them on the page)."""
        added, problems = [], []
        for name, n in names:
            ids: list[str] = []
            for _ in range(n):
                try:
                    self.list_op("add", name)
                except Exception as e:   # e.g. a name with no letters
                    problems.append(str(e))
                    break
                ids.append(self.page_data()[0][-1].name)
            if ids:
                self._remember("add", ids)
                added.append(amount(len(ids), name))
        self.refresh_names()   # the new name can be said from now on
        parts = [f"Added {join_words(added)} as a new item{'s' if len(added) > 1 else ''}."] if added else []
        parts += problems
        parts.append(count_phrase(self._count()))
        return bool(added), " ".join(parts)

    def _not_understood(self, cmd: Command, verb: str) -> str:
        """The reply when an add or remove names no item we know."""
        if cmd.leftover:
            return (f"I don't know {' '.join(cmd.leftover)}.{self._suggest(cmd.leftover)} "
                    "Say: what can I add, to hear every item.")
        if cmd.unheard:
            return ("I didn't recognise that item. Say: what can I add, to hear every item.")
        return f"{verb.capitalize()} what? Say {verb}, then the item, like: {verb} chips."

    def handle(self, text: str) -> tuple[bool, str]:
        """Carry out one sentence. Returns (ok, reply to speak and show)."""
        with self._lock:
            self.refresh_names()
            return self._handle(text)

    def _handle(self, text: str) -> tuple[bool, str]:
        if not words_of(text) or set(words_of(text)) == {UNK}:
            return False, "I didn't catch that. Please try again, or say help."
        cmd = parse(text, self.phrases)
        pending = self.pending
        if pending and time.monotonic() - pending[1] >= PENDING_CLEAR_S:
            pending = None   # asked too long ago
        self.pending = None

        if cmd.action == "yes":
            if pending and pending[0] == "clear":
                return self._clear()
            if pending and pending[0] == "new":
                return self._add_new(pending[2])
            return False, "Yes to what? Say help to hear what you can say."
        if cmd.action == "no":
            if pending and pending[0] == "new":
                return True, "Okay, I didn't add it."
            return True, "Okay, I kept your list." if pending else "Okay."
        if cmd.action == "clear":
            if not self._count():
                return True, "Your list is already empty."
            self.pending = ("clear", time.monotonic(), None)
            return True, f"Remove all {self._count()} items from your list? Say yes or no."
        if cmd.action == "add":
            return self._add(cmd)
        if cmd.action == "remove":
            return self._remove(cmd)
        if cmd.action == "read":
            return True, self._read()
        if cmd.action == "catalog":
            names = sorted(self._name(c) for c in self.known)
            return True, f"You can add: {join_words(names)}."
        if cmd.action == "undo":
            return self._undo()
        if cmd.action == "help":
            return True, HELP_TEXT
        if cmd.leftover:
            return False, f"Sorry, I didn't understand {text!r}.{self._suggest(cmd.leftover)} Say help."
        return False, "Sorry, I didn't understand. Say help to hear what you can say."

    def _add_one(self, cls: str, n: int = 1) -> str | None:
        """Add ``n`` of an item as one undo step. Returns an error to speak, or None."""
        done: list[str] = []
        error = None
        for _ in range(n):
            try:
                self.list_op("add", cls)
            except Exception as e:   # ListError: spoken as is
                error = str(e)
                break
            done.append(cls)
        if done:
            self._remember("add", done)
        return error

    def _remove_one(self, cls: str, n: int = 1) -> int:
        """Remove up to ``n`` of an item as one undo step. Returns how many were removed."""
        done: list[str] = []
        for _ in range(n):
            if cls not in [it.name for it in self.page_data()[0]]:
                break
            try:
                self.list_op("remove", cls)
            except Exception:
                break
            done.append(cls)
        if done:
            self._remember("remove", done)
        return len(done)

    def _take_back(self, cls: str, n: int) -> None:
        """Remove an add that turned out to be misheard, and forget it for undo."""
        for _ in range(n):
            try:
                self.list_op("remove", cls)
            except Exception:
                break
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i] == ("add", [cls] * n):
                del self.history[i]
                break

    def _not_on_list(self, missing: list[str]) -> str:
        names = join_words([self._name(c) for c in missing])
        return (f"{names[:1].upper() + names[1:]} {'is' if len(missing) == 1 else 'are'} "
                "not on your list.")

    def _grouped(self, ids: list[str]) -> list[str]:
        """["water", "water", "chips"] -> ["2 water", "chips"], in first-seen order."""
        counts: dict[str, int] = {}
        for cls in ids:
            counts[cls] = counts.get(cls, 0) + 1
        return [amount(n, self._name(cls)) for cls, n in counts.items()]

    def _add(self, cmd: Command) -> tuple[bool, str]:
        if not cmd.items and cmd.new_names:
            message = self.offer_new(cmd.new_names)
            return self.pending is not None, message   # ok = a question was asked
        if not cmd.items:
            return False, self._not_understood(cmd, "add")
        added, problems = [], []
        for cls, n in zip(cmd.items, cmd.counts):
            error = self._add_one(cls, n)
            if error:
                problems.append(error)
            else:
                added.append(amount(n, self._name(cls)))
        parts = []
        if added:
            parts.append(f"Added {join_words(added)}.")
        parts += problems
        parts.append(count_phrase(self._count()))
        if cmd.new_names:
            parts.append(self.offer_new(cmd.new_names))
        return bool(added), " ".join(parts)

    def _remove(self, cmd: Command) -> tuple[bool, str]:
        if not cmd.items:
            return False, self._not_understood(cmd, "remove")
        removed, missing = [], []
        for cls, n in zip(cmd.items, cmd.counts):
            got = self._remove_one(cls, n)
            if got:
                removed.append(amount(got, self._name(cls)))
            else:
                missing.append(cls)
        parts = []
        if removed:
            parts.append(f"Removed {join_words(removed)}.")
        if missing:
            parts.append(self._not_on_list(missing))
        parts.append(count_phrase(self._count()))
        return bool(removed), " ".join(parts)

    def _clear(self) -> tuple[bool, str]:
        names = [it.name for it in self.page_data()[0]]
        self.list_op("clear", "")
        self._remember("clear", names)
        return True, "Removed every item. Say undo to put them back."

    def _read(self) -> str:
        """The whole list, grouped: "You have 5 items: 3 water, chips, and milk." """
        items = self.page_data()[0]
        if not items:
            return "Your list is empty. Say add, then an item."
        names = self._grouped([it.name for it in items])
        found = self._grouped([it.name for it in items if it.status == FOUND])
        text = f"You have {len(items)} item{'s' if len(items) != 1 else ''}: {join_words(names)}."
        if found:
            text += f" Found so far: {join_words(found)}."
        return text

    def _undo(self) -> tuple[bool, str]:
        if not self.history:
            return False, "There is nothing to undo."
        action, ids = self.history.pop()
        names = join_words(self._grouped(ids))
        if action == "add":
            for cls in ids:
                try:
                    self.list_op("remove", cls)
                except Exception:
                    pass   # already removed some other way
            return True, f"Took {names} off your list. {count_phrase(self._count())}"
        for cls in ids:   # "remove" and "clear": add the items back
            self.list_op("add", cls)
        if action == "clear":
            return True, f"Put back all {len(ids)} items. {count_phrase(self._count())}"
        return True, f"Put {names} back on your list. {count_phrase(self._count())}"

    # ---------------------------------------------------------------- live listening

    def open_session(self, rate: int = 16000) -> str:
        """Start a live listening session; returns its id. Raises VoiceUnavailable."""
        if self.recognizer is None:
            raise VoiceUnavailable(self.status)
        with self._lock:
            self.refresh_names()   # the session listens for today's item names
            # Forget sessions whose page went away without saying goodbye.
            now = time.monotonic()
            for sid in [s for s, v in self.sessions.items() if now - v.last_used > SESSION_IDLE_S]:
                del self.sessions[sid]
            sid = secrets.token_hex(8)
            self.sessions[sid] = VoiceSession(self, rate)
            return sid

    def session(self, sid: str) -> VoiceSession:
        """The open session with this id. Raises VoiceUnavailable if it has ended."""
        with self._lock:
            s = self.sessions.get(sid)
        if s is None:
            raise VoiceUnavailable("Listening has stopped. Press the button to start again.")
        return s

    def close_session(self, sid: str) -> tuple[list[dict], str]:
        """Finish a session: (last events, the whole list read back)."""
        s = self.session(sid)
        with self._lock:
            self.sessions.pop(sid, None)
        return s.finish()


class VoiceSession:
    """One press of the talk button: audio arrives in small pieces while the shopper talks.

    Vosk gives a running guess of the current sentence (a "partial" result) after every
    piece, and a final result when the shopper pauses. Items are applied as soon as they are
    certain, so "add chips and milk and coke" puts chips on the list when "and" is heard,
    without waiting for the end. At each pause the page speaks what that sentence did
    ("Added 3 water and chips."); it cannot speak earlier, because the microphone is muted
    while it talks and the shopper's next words would be lost. Everything else ("what's on
    my list", "undo", "yes") is also answered at the pause. When listening stops, the whole
    list is read back.

    New items: the audio of each sentence is kept until the pause. If the sentence adds or
    removes something, or Vosk heard a word it does not know, Whisper re-reads it. When
    Whisper hears a name that is not an item, its reading wins: anything Vosk added for that
    sentence that Whisper did not hear is taken back ("almond milk" is not almonds plus
    milk), and the shopper is asked whether to create the new item.

    ``feed`` and ``finish`` return events for the page:
      {"kind": "item", "action": "add" | "remove", "item": id, "count": n, "ok": bool,
       "message": str}                       an item changed (shown on screen, not spoken)
      {"kind": "say", "message": str}        speak this now: the shopper has just paused
    """

    MAX_SENTENCE_BYTES = 16000 * 2 * 30   # keep at most 30 s of one sentence's audio

    def __init__(self, assistant: VoiceAssistant, rate: int):
        self.a = assistant
        self.rate = rate
        self.rec = assistant.recognizer.stream(rate)
        self.grammar = getattr(assistant.recognizer, "grammar", None)
        self.audio = bytearray()    # the current sentence, for Whisper
        self.done = 0               # items of the current sentence already applied
        self.commits: list[tuple[str, int]] = []   # (id, count) added during this sentence
        self.heard_any = False      # heard at least one command this session
        # What the current sentence did, spoken at the pause.
        self.added: list[str] = []
        self.removed: list[str] = []
        self.other: list[str] = []
        self.last_used = time.monotonic()
        self._lock = threading.Lock()

    def feed(self, pcm: bytes) -> list[dict]:
        """Take the next piece of audio; return what happened because of it."""
        with self._lock:
            self.last_used = time.monotonic()
            if len(self.audio) < self.MAX_SENTENCE_BYTES * self.rate // 16000:
                self.audio += pcm
            if self.rec.AcceptWaveform(pcm):   # True = the shopper paused: sentence is final
                return self._sentence(json.loads(self.rec.Result()).get("text", ""), final=True)
            return self._sentence(json.loads(self.rec.PartialResult()).get("partial", ""), final=False)

    def finish(self) -> tuple[list[dict], str]:
        """The button was pressed again: finish the last sentence and read the list back."""
        with self._lock:
            events = self._sentence(json.loads(self.rec.FinalResult()).get("text", ""), final=True)
            if not self.heard_any:
                return events, "I didn't hear a command. Press the button and try again, or say help."
            with self.a._lock:
                return events, self.a._read()

    def _second_opinion(self, cmd: Command) -> Command | None:
        """Whisper's reading of the sentence that just ended, when it could matter."""
        checker = self.a.checker
        if checker is None or not (cmd.action in ("add", "remove") or cmd.unheard):
            return None
        try:
            text = checker.transcribe(bytes(self.audio), self.rate)
        except Exception as e:
            log.warning("voice: Whisper failed on a sentence: %s", e)
            return None
        log.info("voice: Whisper heard %r (word list: %s)", text, cmd.action)
        return parse(fix_dictation(text), self.a.phrases) if text else None

    def _sentence(self, text: str, final: bool) -> list[dict]:
        """Act on the newest guess of the current sentence."""
        a = self.a
        cmd = parse(text, a.phrases)
        check = self._second_opinion(cmd) if final else None   # outside the lock: ~0.2 s
        with a._lock:
            cmd = parse(text, a.phrases)   # names may have changed meanwhile
            events: list[dict] = []
            if check is not None and check.action in ("add", "remove") and (
                    check.new_names or cmd.unheard):
                events += self._use_second_opinion(cmd, check)
            elif cmd.action in ("add", "remove"):
                ready = len(cmd.items) if final else cmd.settled
                if cmd.unheard and a.checker is not None and not final:
                    ready = self.done   # an unknown word: wait for the whole sentence
                for cls, n in zip(cmd.items[self.done:ready], cmd.counts[self.done:ready]):
                    event = self._item(cmd.action, cls, n)
                    events.append(event)
                    if cmd.action == "add" and event["ok"]:
                        self.commits.append((cls, n))
                self.done = max(self.done, ready)
                if final:
                    a.pending = None   # a "yes" after this must not answer an old question
                    if not cmd.items:
                        self.heard_any = True
                        self.other.append(a._not_understood(cmd, cmd.action))
            elif final and cmd.action != "unknown":
                self.heard_any = True
                self.other.append(a._handle(text)[1])
            # "unknown" sentences are usually background noise: ignore them while listening.
            if final:
                self.done = 0
                self.commits = []
                self.audio = bytearray()
                self._update_grammar()
                said = self._confirmation()
                if said:
                    events.append({"kind": "say", "message": said})
            return events

    def _use_second_opinion(self, cmd: Command, check: Command) -> list[dict]:
        """Apply Whisper's reading: its items, its new names, and take back Vosk's mistakes."""
        a = self.a
        events: list[dict] = []
        pending = list(self.commits)       # what Vosk already added this sentence
        for cls, n in zip(check.items, check.counts):
            same = next((c for c in pending if c[0] == cls), None)
            if same is not None and check.action == "add":
                pending.remove(same)       # already on the list
                continue
            events.append(self._item(check.action, cls, n))
        for cls, n in pending:             # Vosk added it, Whisper did not hear it
            a._take_back(cls, n)
            label = amount(n, a._name(cls))
            if label in self.added:
                self.added.remove(label)
            events.append({"kind": "item", "action": "remove", "item": cls, "count": n,
                           "ok": True, "message": f"Took back {label}."})
        a.pending = None
        self.heard_any = True
        if check.new_names and check.action == "add":
            self.other.append(a.offer_new(check.new_names))
        elif check.new_names:
            names = join_words([name for name, _ in check.new_names])
            self.other.append(f"{names[:1].upper()}{names[1:]} "
                              f"{'is' if len(check.new_names) == 1 else 'are'} not on your list.")
        elif not events:
            self.other.append(a._not_understood(cmd, check.action))
        return events

    def _update_grammar(self) -> None:
        """After a new item was created, listen for its name from the next sentence on."""
        grammar = getattr(self.a.recognizer, "grammar", None)
        if grammar and grammar != self.grammar and hasattr(self.rec, "SetGrammar"):
            self.rec.SetGrammar(grammar)
            self.grammar = grammar

    def _item(self, action: str, cls: str, n: int) -> dict:
        name = self.a._name(cls)
        self.heard_any = True
        if action == "add":
            error = self.a._add_one(cls, n)
            ok = error is None
            msg = f"Added {amount(n, name)}." if ok else error
            (self.added if ok else self.other).append(amount(n, name) if ok else msg)
        else:
            got = self.a._remove_one(cls, n)
            ok = got > 0
            msg = f"Removed {amount(got, name)}." if ok else self.a._not_on_list([cls])
            (self.removed if ok else self.other).append(amount(got, name) if ok else msg)
        log.info("voice: %s", msg)
        return {"kind": "item", "action": action, "item": cls, "count": n, "ok": ok, "message": msg}

    def _confirmation(self) -> str:
        """What the sentence that just ended did, in one short spoken line; then forget it."""
        parts = []
        if self.added:
            parts.append(f"Added {join_words(self.added)}.")
        if self.removed:
            parts.append(f"Removed {join_words(self.removed)}.")
        parts += self.other
        self.added, self.removed, self.other = [], [], []
        return " ".join(parts)


# --------------------------------------------------------------------------- Vosk

class SpeechRecognizer:
    """Offline speech to text with Vosk, limited to a fixed set of words.

    model_dir:  the unpacked Vosk model folder.
    vocabulary: the words to listen for; words the model has never heard of are dropped
                with a warning (Vosk cannot recognise a word that is not in its dictionary).
    """

    def __init__(self, model_dir: Path, vocabulary: Iterable[str]):
        import vosk   # imported here so the rest of the server works without it

        vosk.SetLogLevel(-1)   # Vosk is very chatty on stderr
        self._vosk = vosk
        self.model = vosk.Model(str(model_dir))
        self._warned: set[str] = set()
        self.set_vocabulary(vocabulary)

    def set_vocabulary(self, vocabulary: Iterable[str]) -> None:
        """Listen for these words from the next recognizer on (a new item was added)."""
        words = sorted(set(vocabulary))
        missing = [w for w in words if self.model.vosk_model_find_word(w) < 0]
        new = [w for w in missing if w not in self._warned]
        if new:
            log.warning("voice: the speech model does not know these words, so they cannot be "
                        "heard: %s (edit config/voice_aliases.yaml or the item's name)",
                        ", ".join(new))
            self._warned.update(new)
        self.words = [w for w in words if w not in missing]
        self.grammar = json.dumps(self.words + [UNK])

    def transcribe(self, pcm: bytes, rate: int = 16000) -> str:
        """16-bit little-endian mono audio at ``rate`` samples per second -> lower-case text."""
        # A fresh recognizer per request: they are cheap, and requests may run in parallel.
        rec = self.stream(rate)
        rec.AcceptWaveform(pcm)
        return json.loads(rec.FinalResult()).get("text", "")

    def stream(self, rate: int = 16000):
        """A Vosk recognizer to feed audio piece by piece (see ``VoiceSession``)."""
        return self._vosk.KaldiRecognizer(self.model, rate, self.grammar)


def fix_dictation(text: str) -> str:
    """Whisper sometimes writes "at" or "ad" for a leading "add" ("At a jar of Nutella")."""
    words = text.strip().split(maxsplit=1)
    if words and words[0].lower().strip(".,!?") in ("at", "ad", "adde"):
        return "add " + (words[1] if len(words) > 1 else "")
    return text


class SentenceChecker:
    """Whisper (faster-whisper) for one finished sentence at a time, fully offline.

    On the laptop GPU the small English model takes about 0.2 s per sentence; without a GPU
    the base model takes about 1.2 s. It only runs at a pause, never while streaming.
    """

    def __init__(self, model, model_name: str, device: str):
        self.model = model
        self.model_name = model_name
        self.device = device
        self._lock = threading.Lock()   # one sentence at a time

    @classmethod
    def load(cls, cfg) -> SentenceChecker:
        """Load the configured model from ``voice.whisper_dir`` (no network access).

        With ``whisper_device: auto`` the GPU is tried first and the CPU model is used if
        the GPU libraries are missing.
        """
        v = cfg.voice
        if v.whisper_device in ("auto", "cuda"):
            try:
                return cls._load(cfg, "cuda")
            except Exception as e:
                if v.whisper_device == "cuda":
                    raise
                log.info("voice: Whisper on the GPU failed (%s); using the CPU", e)
        return cls._load(cfg, "cpu")

    @classmethod
    def _load(cls, cfg, device: str) -> SentenceChecker:
        if device == "cuda":
            import ctranslate2
            if ctranslate2.get_cuda_device_count() == 0:
                raise RuntimeError("no CUDA device")
            # ctranslate2 needs cuBLAS/cuDNN. PyTorch ships them and loads them on import
            # (the server imports it anyway for the camera models).
            import torch  # noqa: F401
        from faster_whisper import WhisperModel

        v = cfg.voice
        name = v.whisper_model_gpu if device == "cuda" else v.whisper_model_cpu
        model = WhisperModel(name, device=device,
                             compute_type="float16" if device == "cuda" else "int8",
                             download_root=str(cfg.path(v.whisper_dir)), local_files_only=True)
        checker = cls(model, name, device)
        checker.transcribe(b"\0\0" * 16000, 16000)   # warm up (and fail here, not mid-sentence)
        return checker

    def transcribe(self, pcm: bytes, rate: int = 16000) -> str:
        """16-bit mono audio -> text ("" for under 0.2 s of audio)."""
        import numpy as np

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if rate != 16000 and len(audio):
            t_new = np.arange(0, len(audio) / rate, 1 / 16000)
            audio = np.interp(t_new, np.arange(len(audio)) / rate, audio).astype(np.float32)
        if len(audio) < 3200:
            return ""
        with self._lock:
            segments, _ = self.model.transcribe(audio, language="en", beam_size=1,
                                                condition_on_previous_text=False)
            return " ".join(s.text.strip() for s in segments).strip()


# --------------------------------------------------------------------------- setup

def build_voice(cfg, items, list_op: Callable[[str, str], str],
                page_data: Callable[[], tuple], background: bool = True) -> VoiceAssistant:
    """Create the assistant and start loading the speech model.

    cfg: the ServerConfig. items: the item ``Catalog`` (or an id -> spoken dict).
    background=True loads the model in a thread so startup is not delayed; until it is ready,
    voice requests get "still starting" as their reply.
    """
    known = items if isinstance(items, dict) else items.item_map()
    aliases = load_aliases(cfg.path(cfg.voice.aliases_file), known)
    assistant = VoiceAssistant(items, aliases, list_op, page_data)
    if not cfg.voice.enabled:
        assistant.status = "Voice commands are turned off in config/server.yaml."
        return assistant
    model_dir = cfg.path(cfg.voice.model_dir)

    def load() -> None:
        assistant.load_recognizer(model_dir)
        if cfg.voice.whisper_enabled and assistant.recognizer is not None:
            assistant.load_checker(cfg)

    if background:
        threading.Thread(target=load, name="voice-model", daemon=True).start()
    else:
        load()
    return assistant


def download_whisper(cfg) -> Path:
    """Download the Whisper models for new item names (both the GPU and the CPU one)."""
    from faster_whisper import WhisperModel

    root = cfg.path(cfg.voice.whisper_dir)
    for name in dict.fromkeys((cfg.voice.whisper_model_gpu, cfg.voice.whisper_model_cpu)):
        print(f"downloading Whisper {name} (once)")
        WhisperModel(name, device="cpu", compute_type="int8", download_root=str(root))
    return root


def download_model(cfg) -> Path:
    """Download and unpack the Vosk model once (``slr voice download-model``, needs internet)."""
    dest = cfg.path(cfg.voice.model_dir)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archive = Path(tmp) / "model.zip"
        print(f"downloading {cfg.voice.model_url}")
        urllib.request.urlretrieve(cfg.voice.model_url, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        # The zip holds one top-level folder; move it to the configured name.
        folders = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if len(folders) != 1:
            raise RuntimeError(f"unexpected layout in {cfg.voice.model_url}")
        shutil.move(str(folders[0]), dest)
    return dest
