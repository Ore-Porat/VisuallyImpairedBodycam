"""Tests for the item catalog (``slr/common/catalog.py``).

What is being tested?
    The catalog is the join between a shopping-list id ("tomato_sauce") and the two things
    the rest of the program needs: the words to speak ("tomato sauce") and the CLIP prompts
    that find it ("a can of tomato sauce"). It also owns free-text items: typing a name the
    presets do not know must produce a usable id and survive a restart.

Words you will see
    preset   an item from config/catalog.yaml (shipped, read-only).
    user     an item from data/catalog_user.yaml (written by the program).
    slug     the id made from a name: lower case, spaces become underscores.
"""

from __future__ import annotations

import pytest
import yaml

from slr.common.catalog import (
    Catalog,
    CatalogError,
    CatalogItem,
    load_catalog,
    normalize_key,
    slugify,
)


def write(path, data):
    """Write a YAML file and return its path."""
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def presets_file(tmp_path):
    """A tiny preset catalog in the new format."""
    return write(tmp_path / "catalog.yaml", {
        "items": {
            "pasta": {"spoken": "pasta", "prompts": ["a package of pasta"]},
            "coke": {
                "spoken": "Coke",
                "aliases": ["coca cola", "coca-cola"],
                "prompts": ["a can of Coca-Cola", "a red can of Coke"],
            },
            "hand": {"spoken": "hand", "non_item": True},
        }
    })


# --------------------------------------------------------------------------- loading


def test_new_format_loads_prompts_and_aliases(tmp_path):
    cat = load_catalog(presets_file(tmp_path))
    assert cat.item_ids == ("pasta", "coke")          # hand is not addable
    pasta = cat.item("pasta")
    assert pasta.spoken == "pasta"
    assert pasta.prompts == ("a package of pasta",)
    assert pasta.preset is True
    assert cat.item("hand").non_item is True


def test_legacy_classes_format_still_loads(tmp_path):
    """config/classes.yaml and config/classes_coco.yaml use the old scalar shape."""
    path = write(tmp_path / "classes.yaml", {"classes": {"pasta": "pasta", "rice": "white rice"}})
    cat = load_catalog(path)
    assert cat.spoken_map() == {"pasta": "pasta", "rice": "white rice"}
    assert cat.item("rice").prompts == ()             # nothing curated: generated later


def test_empty_catalog_is_rejected(tmp_path):
    path = write(tmp_path / "empty.yaml", {"items": {}})
    with pytest.raises(CatalogError):
        load_catalog(path)


def test_unknown_entry_key_is_rejected(tmp_path):
    path = write(tmp_path / "bad.yaml", {"items": {"pasta": {"spoken": "pasta", "promt": ["x"]}}})
    with pytest.raises(CatalogError):
        load_catalog(path)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(CatalogError):
        load_catalog(tmp_path / "nope.yaml")


def test_user_catalog_overrides_a_preset(tmp_path):
    """A user entry with the same id replaces the shipped prompts (the fix-a-bad-prompt path)."""
    presets = presets_file(tmp_path)
    user = write(tmp_path / "user.yaml", {
        "items": {"pasta": {"spoken": "pasta", "prompts": ["a blue pasta box"]}}
    })
    cat = load_catalog(presets, user)
    # An edited preset is custom wording, so CLIP reads it as a photo (item_prompts).
    assert cat.prompts_for("pasta") == ("a photo of pasta", "a photo of a blue pasta box")
    assert cat.item("pasta").prompts == ("a blue pasta box",)   # stored as typed
    assert cat.is_preset("pasta") is False
    assert cat.is_preset("coke") is True


# --------------------------------------------------------------------------- lookups


def test_resolve_accepts_id_spoken_name_and_alias(tmp_path):
    cat = load_catalog(presets_file(tmp_path))
    for name in ("coke", "Coke", "COKE", "coca cola", "coca-cola", "  Coca   Cola  "):
        assert cat.resolve(name).id == "coke", name
    assert cat.resolve("sprite") is None


def test_spoken_falls_back_to_a_readable_id(tmp_path):
    cat = load_catalog(presets_file(tmp_path))
    assert cat.spoken("tomato_sauce") == "tomato sauce"


def test_prompts_are_generated_when_none_are_curated(tmp_path):
    cat = load_catalog(presets_file(tmp_path))
    item = CatalogItem(id="gatorade", spoken="blue Gatorade")
    assert cat.prompts_for(item) == ("a photo of blue Gatorade",)


# --------------------------------------------------------------------------- free text


def test_normalize_key_ignores_case_spacing_and_punctuation():
    assert normalize_key("  Coca-Cola ") == "coca cola"
    assert normalize_key("coca_cola") == "coca cola"
    assert normalize_key("Café") == "cafe"


def test_slugify_makes_safe_ids():
    assert slugify("Coca Cola") == "coca_cola"
    assert slugify("  Café au lait! ") == "cafe_au_lait"
    assert slugify("tomato_sauce") == "tomato_sauce"
    with pytest.raises(CatalogError):
        slugify("!!!")


def test_new_item_is_created_and_persisted(tmp_path):
    """Typing an unknown name must survive a restart: it is written to the user catalog."""
    presets = presets_file(tmp_path)
    user_path = tmp_path / "user.yaml"
    cat = load_catalog(presets, user_path)
    item = cat.resolve_or_create("Blue Gatorade")
    assert item.id == "blue_gatorade"
    assert item.spoken == "Blue Gatorade"
    assert item.preset is False
    assert user_path.exists()

    # A fresh load sees it, with the same spoken name.
    again = load_catalog(presets, user_path)
    assert again.item("blue_gatorade").spoken == "Blue Gatorade"
    assert again.resolve("blue gatorade").id == "blue_gatorade"


def test_resolve_or_create_is_idempotent(tmp_path):
    cat = load_catalog(presets_file(tmp_path), tmp_path / "user.yaml")
    first = cat.resolve_or_create("Sprite")
    second = cat.resolve_or_create("sprite")
    assert first.id == second.id == "sprite"
    assert list(cat.user) == ["sprite"]


def test_resolve_or_create_never_duplicates_a_preset(tmp_path):
    cat = load_catalog(presets_file(tmp_path), tmp_path / "user.yaml")
    item = cat.resolve_or_create("Coca Cola")          # an alias of the preset coke
    assert item.id == "coke"
    assert cat.user == {}


def test_user_prompts_can_be_saved(tmp_path):
    """The web editor will write custom prompts; they must round-trip."""
    presets = presets_file(tmp_path)
    user_path = tmp_path / "user.yaml"
    cat = load_catalog(presets, user_path)
    cat.user["coke"] = CatalogItem(
        id="coke", spoken="Coke", prompts=("a green Coke can",), preset=False
    )
    cat.save_user()
    again = load_catalog(presets, user_path)
    assert again.item("coke").prompts == ("a green Coke can",)


def test_empty_catalog_object_is_harmless():
    """No items at all is a valid state (an empty shopping list)."""
    cat = Catalog()
    assert cat.items == ()
    assert cat.item_ids == ()
    assert cat.resolve("anything") is None


# --------------------------------------------------------------------------- prompt editing


def test_user_prompts_override_a_preset_and_can_be_restored(tmp_path):
    """Saving prompts never edits the shipped file: it writes a user override."""
    presets = presets_file(tmp_path)
    user_path = tmp_path / "user.yaml"
    cat = load_catalog(presets, user_path)
    cat.set_prompts("pasta", ["a blue pasta box", "a box of penne"])
    assert cat.item("pasta").prompts == ("a blue pasta box", "a box of penne")
    assert cat.is_preset("pasta") is False         # the active entry is now the user's

    again = load_catalog(presets, user_path)
    assert again.item("pasta").prompts == ("a blue pasta box", "a box of penne")

    again.restore_prompts("pasta")
    assert again.prompts_for("pasta") == ("a package of pasta",)
    assert again.is_preset("pasta") is True


def test_set_prompts_for_an_unknown_item_creates_it(tmp_path):
    """A stale shopping-list id can still be given prompts."""
    cat = load_catalog(presets_file(tmp_path), tmp_path / "user.yaml")
    cat.set_prompts("blue_gatorade", ["a blue sports drink bottle"],
                    spoken="blue Gatorade")
    assert cat.item("blue_gatorade").spoken == "blue Gatorade"
    assert cat.prompts_for("blue_gatorade") == ("a photo of blue Gatorade",
                                                "a photo of a blue sports drink bottle")


def test_prompt_editing_rejects_empty_and_huge_input(tmp_path):
    cat = load_catalog(presets_file(tmp_path), tmp_path / "user.yaml")
    with pytest.raises(CatalogError):
        cat.set_prompts("pasta", ["   ", ""])
    with pytest.raises(CatalogError):
        cat.set_prompts("pasta", [f"prompt {i}" for i in range(13)])
    with pytest.raises(CatalogError):
        cat.set_prompts("pasta", ["x" * 301])


def test_items_for_synthesizes_a_missing_entry(tmp_path):
    """An old or hand-written list id must not crash the detector."""
    cat = load_catalog(presets_file(tmp_path))
    items = cat.items_for(["pasta", "mystery_item"])
    assert items[0].id == "pasta"
    assert items[1].id == "mystery_item"
    assert items[1].spoken == "mystery item"
    assert items[1].prompts == ()


def test_user_catalog_reloads_after_an_outside_edit(tmp_path):
    """The CLI writes data/catalog_user.yaml; a running server must pick the change up."""
    import os

    presets = presets_file(tmp_path)
    user_path = tmp_path / "user.yaml"
    cat = load_catalog(presets, user_path)
    assert cat.changed_on_disk() is False

    write(user_path, {"items": {"sprite": {"spoken": "Sprite",
                                           "prompts": ["a green can of Sprite"]}}})
    stat = os.stat(user_path)
    os.utime(user_path, (stat.st_atime, stat.st_mtime + 5))   # guarantee a new mtime
    assert cat.changed_on_disk() is True

    cat.reload_user()
    assert cat.resolve("Sprite").id == "sprite"
    assert cat.changed_on_disk() is False


def test_saving_prompts_does_not_look_like_an_outside_edit(tmp_path):
    cat = load_catalog(presets_file(tmp_path), tmp_path / "user.yaml")
    cat.set_prompts("pasta", ["a blue pasta box"])
    assert cat.changed_on_disk() is False


# --------------------------------------------------------------------------- shipped files


def test_shipped_catalog_matches_the_legacy_classes_file():
    """The migration must not change ids, spoken names or their order.

    GrocerEye's label files store class *indices*, so re-ordering the catalog would silently
    relabel the calibration data (soda would become spices, and so on).
    """
    from conftest import ROOT

    catalog = load_catalog(ROOT / "config" / "catalog.yaml")
    legacy = load_catalog(ROOT / "config" / "classes.yaml")
    assert list(catalog.spoken_map()) == list(legacy.spoken_map())
    assert catalog.spoken_map() == legacy.spoken_map()


def test_every_shipped_preset_has_prompt_text():
    """A preset without prompts would fall back to 'a photo of X': allowed, but not curated."""
    from conftest import ROOT

    catalog = load_catalog(ROOT / "config" / "catalog.yaml")
    for item in catalog.items:
        if item.non_item:
            assert item.prompts == ()          # the hand is never given to CLIP
        else:
            assert item.prompts, f"{item.id} has no curated prompts"


def test_shipped_server_config_points_at_the_catalog():
    from conftest import ROOT

    from slr.common.config import ModelConfig, load_server_config

    cfg = load_server_config(ROOT / "config")
    assert cfg.model.catalog_file == "config/catalog.yaml"
    assert cfg.paths.user_catalog == "data/catalog_user.yaml"
    # The old key is gone for good: a config that still uses it must fail loudly.
    with pytest.raises(Exception):
        ModelConfig(classes_file="config/classes.yaml")


# --------------------------------------------------------------------------- photo wording


def test_custom_prompts_always_read_as_photos():
    """Custom wording gets "a photo of" (better CLIP scores, DECISIONS.md); presets do not."""
    from slr.common.catalog import as_photo_prompt, box_prompt, item_prompts

    assert as_photo_prompt("a red can of Coke") == "a photo of a red can of Coke"
    assert as_photo_prompt("A photo of a can") == "A photo of a can"     # not doubled
    assert as_photo_prompt("  a   can ") == "a photo of a can"
    custom = CatalogItem(id="kombucha", spoken="kombucha", prompts=("a glass bottle",),
                         preset=False)
    assert item_prompts(custom) == ("a photo of kombucha", "a photo of a glass bottle")
    assert box_prompt(custom, "can") == "a photo of a can of kombucha"
    preset = CatalogItem(id="pasta", spoken="pasta", prompts=("a package of pasta",))
    assert item_prompts(preset) == ("a package of pasta",)
    assert box_prompt(preset, "can") == "a can of pasta"
