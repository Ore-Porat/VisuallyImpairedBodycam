"""Config validation and the phrase catalog / clip store."""

import pytest

from slr.common.config import (ConfigError, GuidanceConfig, load_classes, load_controls_config,
                               load_device_config, load_server_config)
from slr.common.protocol import PRIORITIES
from slr.common.types import ListItem
from slr.server.clips import BeepTTS, ClipStore, PhraseCatalog
from slr.server.guidance import Say

from conftest import ROOT


def test_shipped_configs_load():
    s = load_server_config()
    d = load_device_config()
    c = load_controls_config()
    assert s.network.udp_port == 8766 and s.network.ws_port == 8765
    assert d.camera.backend == "picamera2"
    assert c.mode_switch.positions == {"00": "off", "11": "on", "01": "toggle"}
    assert s.path("data/x") == ROOT / "data" / "x"


def test_sim_overlay():
    d = load_device_config(sim=True)
    assert d.camera.backend == "opencv"
    assert d.server.host == "127.0.0.1"
    assert d.volume.source == "keyboard"
    assert d.streaming.max_fps == 10  # untouched keys survive the merge


def test_classes_files_load():
    grocer = load_classes(ROOT / "config" / "classes.yaml")
    assert grocer["tomato_sauce"] == "tomato sauce"
    assert len(load_classes(ROOT / "config" / "classes_coco.yaml")) == 80


def test_invalid_guidance_rejected():
    with pytest.raises(ValueError):
        GuidanceConfig(stable_k=6, stable_n=5)
    with pytest.raises(ValueError):
        GuidanceConfig(straight_tol=0.3, slight_tol=0.2)


def test_typos_in_config_are_errors(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "server.yaml").write_text("guidance:\n  conf_fnd: 0.4\n")
    with pytest.raises(ConfigError):
        load_server_config(tmp_path / "config")


@pytest.fixture
def catalog():
    classes = {"pasta": "pasta", "tomato_sauce": "tomato sauce"}
    return PhraseCatalog.load(ROOT / "config" / "phrases.yaml", classes, "test")


def test_every_phrase_key_used_by_guidance_exists(catalog):
    import re

    src = (ROOT / "slr" / "server" / "guidance.py").read_text()
    used = set(re.findall(r'say\("([a-z_]+)"', src))
    assert used and used <= set(catalog.phrases)
    assert all(ph["priority"] in PRIORITIES for ph in catalog.phrases.values())


def test_phrase_rendering_uses_spoken_names(catalog):
    assert catalog.text("found", {"item": "tomato_sauce"}) == "Tomato sauce found."
    assert catalog.text("direction", {"h": "slightly_left", "v": "eye"}) == "Slightly left. Eye level."
    assert catalog.text("item_announce", {"n": 2, "total": 5, "item": "pasta"}) == "Item 2 of 5: pasta."
    assert catalog.text("wrong_item", {"other": "pasta", "item": "tomato_sauce"}) == \
        "That's pasta, not tomato sauce."


def test_clip_names_change_with_text(catalog):
    a = catalog.resolve(Say("found", {"item": "pasta"}))
    b = catalog.resolve(Say("found", {"item": "tomato_sauce"}))
    assert a.clip != b.clip and a.clip.startswith("found-")
    assert a.priority == "confirmation"
    other_voice = PhraseCatalog(catalog.phrases, catalog.directions, {}, catalog.classes, "other")
    assert other_voice.resolve(Say("found", {"item": "pasta"})).clip != a.clip


def test_system_phrases_cover_device_needs(catalog):
    needed = {"starting_up", "connected", "connection_lost", "camera_error", "video_weak",
              "shutting_down", "not_connected"} | {f"volume_{i}" for i in range(11)}
    assert needed <= set(catalog.system)


def test_clip_store_renders_once_and_manifest(tmp_path, catalog):
    store = ClipStore(tmp_path / "clips", BeepTTS())
    cues = catalog.prerender_set([ListItem("pasta")], all_classes=False)
    n = store.prerender(catalog, cues)
    assert n == len({catalog.resolve(c).clip for c in cues})
    assert store.prerender(catalog, cues) == 0
    h = store.manifest_hash()
    reloaded = ClipStore(tmp_path / "clips", BeepTTS())
    assert reloaded.manifest_hash() == h
