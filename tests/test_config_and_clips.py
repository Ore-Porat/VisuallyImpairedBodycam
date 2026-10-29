"""Config validation and the phrase catalog / clip store.

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    Two things that live on the laptop side:
      1. Config loading (``slr/common/config.py``): the YAML files in config/ are read
         into checked Python objects. A typo in a key, or a value that makes no sense,
         must be an error at startup rather than a silent surprise later.
      2. Speech (``slr/server/clips.py``): ``PhraseCatalog`` holds the sentence
         templates from config/phrases.yaml ("{item} found.") and fills in the spoken
         item names. ``ClipStore`` turns each finished sentence into a WAV audio file
         ("clip") once, and remembers it so it is not rendered again.

Words you will see:
    fixture       A helper pytest builds before a test; ``catalog`` below loads the real
                  phrases file with a two-item pretend class list.
    tmp_path      A temporary folder from pytest, deleted after the test.
    pytest.raises A block that PASSES only if the code inside raises the named error.
    BeepTTS       A fake text-to-speech engine that produces a beep instead of a voice,
                  so tests do not need the Piper voice model downloaded.
    ROOT          The project folder (defined in tests/conftest.py).
"""

import pytest

from slr.common.config import (ConfigError, GuidanceConfig, load_classes, load_controls_config,
                               load_device_config, load_server_config)
from slr.common.protocol import PRIORITIES
from slr.common.types import ListItem
from slr.server.clips import BeepTTS, ClipStore, PhraseCatalog
from slr.server.guidance import Say

from conftest import ROOT


def test_shipped_configs_load():
    """The three config files committed to the repo load without errors and have sane values."""
    s = load_server_config()
    d = load_device_config()
    c = load_controls_config()
    assert s.network.udp_port == 8766 and s.network.ws_port == 8765
    assert s.guidance.stream_in_idle is True
    assert s.guidance.stream_in_off is True
    assert s.guidance.auto_find_on_mode_on is True
    assert s.guidance.hand_guidance_always is True
    assert s.guidance.identify_max_objects == 3
    assert d.camera.backend == "opencv"
    assert c.mode_switch.positions == {"00": "off", "11": "on", "01": "toggle"}
    assert c.mode_switch.toggle_hold_s == 0.8
    # s.path() turns a repo-relative path from the config into an absolute one.
    assert s.path("data/x") == ROOT / "data" / "x"


def test_sim_overlay():
    """The 'sim:' section of device.yaml overrides only the keys it lists."""
    d = load_device_config(sim=True)
    assert d.camera.backend == "opencv"
    assert d.audio.dac == "none"
    assert d.server.host == "127.0.0.1"
    assert d.volume.source == "keyboard"
    assert d.streaming.max_fps == 20  # untouched keys survive the merge


def test_classes_files_load():
    """Both class lists (grocery model and stock COCO model) load; COCO has its 80 classes."""
    grocer = load_classes(ROOT / "config" / "classes.yaml")
    assert grocer["tomato_sauce"] == "tomato sauce"
    assert len(load_classes(ROOT / "config" / "classes_coco.yaml")) == 80


def test_invalid_guidance_rejected():
    """Nonsense guidance settings are refused: k > n, or 'straight' wider than 'slight'."""
    with pytest.raises(ValueError):
        GuidanceConfig(stable_k=6, stable_n=5)
    with pytest.raises(ValueError):
        GuidanceConfig(straight_tol=0.3, slight_tol=0.2)


def test_typos_in_config_are_errors(tmp_path):
    """A misspelled key (conf_fnd instead of conf_find) is a ConfigError, not ignored."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "server.yaml").write_text("guidance:\n  conf_fnd: 0.4\n")
    with pytest.raises(ConfigError):
        load_server_config(tmp_path / "config")


@pytest.fixture
def catalog():
    """The real phrases.yaml, with a tiny class list and the voice name 'test'."""
    classes = {"pasta": "pasta", "tomato_sauce": "tomato sauce"}
    return PhraseCatalog.load(ROOT / "config" / "phrases.yaml", classes, "test")


def test_every_phrase_key_used_by_guidance_exists(catalog):
    """Every phrase name the guidance engine can say is defined in phrases.yaml."""
    import re

    # Scan the guidance source code for say("...") calls and collect the phrase names.
    src = (ROOT / "slr" / "server" / "guidance.py").read_text()
    used = set(re.findall(r'say\("([a-z_]+)"', src))
    # "<=" between sets means "is a subset of".
    assert used and used <= set(catalog.phrases)
    # Every phrase has a priority that the protocol knows about.
    assert all(ph["priority"] in PRIORITIES for ph in catalog.phrases.values())


def test_phrase_rendering_uses_spoken_names(catalog):
    """Templates are filled with spoken names and capitalized at the sentence start."""
    assert catalog.text("found", {"item": "tomato_sauce"}) == "Tomato sauce found."
    assert catalog.text("direction", {"h": "slightly_left", "v": "eye"}) == "Slightly left. Eye level."
    assert catalog.text("item_announce", {"n": 2, "total": 5, "item": "pasta"}) == "Item 2 of 5: pasta."
    assert catalog.text("item_marked_found", {"item": "tomato_sauce"}) == "Tomato sauce marked found."
    assert catalog.text("item_marked_found_next", {"item": "pasta", "next": "tomato_sauce"}) == \
        "Pasta marked found. Next: tomato sauce."
    assert catalog.text("item_marked_found_done", {"item": "pasta"}) == \
        "Pasta marked found. That was the last item on your list."
    assert catalog.text("identified_many", {"items": ["pasta", "milk", "rice"]}) == \
        "I see pasta, milk, and rice."
    assert catalog.text("wrong_item", {"other": "pasta", "item": "tomato_sauce"}) == \
        "That's pasta, not tomato sauce."


def test_clip_names_change_with_text(catalog):
    """Each distinct sentence (and each voice) gets its own clip file name."""
    a = catalog.resolve(Say("found", {"item": "pasta"}))
    b = catalog.resolve(Say("found", {"item": "tomato_sauce"}))
    assert a.clip != b.clip and a.clip.startswith("found-")
    assert a.priority == "confirmation"
    # Same sentence, different voice: different clip, so switching voices re-renders.
    other_voice = PhraseCatalog(catalog.phrases, catalog.directions, {}, catalog.classes, "other")
    assert other_voice.resolve(Say("found", {"item": "pasta"})).clip != a.clip


def test_system_phrases_cover_device_needs(catalog):
    """The device's built-in clips (played without the laptop) are all defined."""
    needed = {"starting_up", "connected", "connection_lost", "camera_error", "video_weak",
              "shutting_down", "not_connected"} | {f"volume_{i}" for i in range(11)}
    assert needed <= set(catalog.system)


def test_clip_store_renders_once_and_manifest(tmp_path, catalog):
    """Clips are rendered once, skipped the second time, and the manifest hash is stable."""
    store = ClipStore(tmp_path / "clips", BeepTTS())
    # The set of cues the server pre-renders for a list containing only "pasta".
    cues = catalog.prerender_set([ListItem("pasta")], all_classes=False)
    n = store.prerender(catalog, cues)
    # One file per distinct clip name.
    assert n == len({catalog.resolve(c).clip for c in cues})
    # Second call: everything is cached, nothing new rendered.
    assert store.prerender(catalog, cues) == 0
    # The manifest hash lets the device check whether it has the current set of clips.
    h = store.manifest_hash()
    reloaded = ClipStore(tmp_path / "clips", BeepTTS())
    assert reloaded.manifest_hash() == h


def test_item_classes_drops_the_hand():
    from slr.common.config import item_classes
    classes = {"pasta": "pasta", "hand": "hand", "rice": "rice"}
    assert item_classes(classes, ["hand"]) == {"pasta": "pasta", "rice": "rice"}
    assert item_classes(classes, []) == classes
