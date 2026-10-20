"""Tests for the shopping list (``slr/server/session.py``).

What is a test file?
    pytest runs every function whose name starts with ``test_``. Each one checks one
    behavior and fails loudly if it is wrong.

What is being tested here?
    ``ShoppingList`` keeps the user's list of items in a small YAML file on the laptop
    (normally data/shopping_list.yaml). It must:
      * accept only item names the detector knows (so you cannot search for "spaghetti"
        when the model only knows "pasta"),
      * understand both the internal id ("tomato_sauce") and the spoken name
        ("tomato sauce"), in any capitalization,
      * remember which items are already found, and which one is "current",
      * notice when the file was edited by someone else (the CLI or the web editor).

Words you will see:
    fixture       A helper that pytest builds before each test. ``lst`` below gives every
                  test a fresh, empty shopping list stored in a temporary folder.
    tmp_path      A temporary folder provided by pytest, deleted after the test.
    pytest.raises A block that PASSES only if the code inside raises the named error.
    assert        "Check that this is true; if not, the test fails."
"""

import pytest

from slr.common.types import FOUND, PENDING, ListItem
from slr.server.session import ListError, ShoppingList, normalize_name

# The set of items the (pretend) detector knows: internal id -> spoken name.
KNOWN = {"pasta": "pasta", "rice": "rice", "tomato_sauce": "tomato sauce", "coffee": "coffee"}


@pytest.fixture
def lst(tmp_path):
    """A fresh shopping list backed by a file in a temporary folder."""
    return ShoppingList(tmp_path / "shopping_list.yaml", KNOWN)


def test_add_known_items_and_persist(lst, tmp_path):
    """Added items are saved to disk and come back when the file is loaded again."""
    lst.add("pasta")
    lst.add("Tomato Sauce")
    # Open the same file with a brand-new ShoppingList object: it must see both items.
    again = ShoppingList(tmp_path / "shopping_list.yaml", KNOWN)
    assert [it.name for it in again.items] == ["pasta", "tomato_sauce"]
    assert all(it.status == PENDING for it in again.items)


def test_unknown_item_rejected_with_valid_list(lst):
    """Adding an item the model cannot detect fails, and the error names the valid choices."""
    with pytest.raises(ListError) as e:
        lst.add("spaghetti")
    # str(e.value) is the error message text.
    assert "pasta" in str(e.value) and "tomato_sauce" in str(e.value)
    assert lst.items == []


def test_normalize_accepts_id_spoken_and_spaces():
    """Different spellings of the same item all map to the one internal id."""
    assert normalize_name("tomato_sauce", KNOWN) == "tomato_sauce"
    assert normalize_name("  tomato   sauce ", KNOWN) == "tomato_sauce"
    assert normalize_name("PASTA", KNOWN) == "pasta"


def test_remove_and_clear(lst):
    """Removing an item keeps the 'current' pointer on the same remaining item."""
    for n in ("pasta", "rice", "coffee"):
        lst.add(n)
    lst.current = 2
    lst.remove("pasta")
    assert [it.name for it in lst.items] == ["rice", "coffee"]
    assert lst.current == 1  # still pointing at coffee
    # Removing something that is not on the list is an error.
    with pytest.raises(ListError):
        lst.remove("pasta")
    lst.clear()
    assert lst.items == [] and lst.current == 0


def test_status_persists_and_reset(lst, tmp_path):
    """Found / pending status and the current index survive a reload; reset clears them."""
    lst.add("pasta")
    lst.add("rice")
    # set_state is what the guidance engine calls after checking an item off.
    lst.set_state((ListItem("pasta", FOUND), ListItem("rice")), 1)
    again = ShoppingList(tmp_path / "shopping_list.yaml", KNOWN)
    assert again.items[0].status == FOUND and again.current == 1
    again.reset()
    assert all(it.status == PENDING for it in again.items) and again.current == 0


def test_changed_on_disk_detects_external_edit(lst, tmp_path):
    """If another program edits the file, the running server can notice and reload it."""
    lst.add("pasta")
    assert not lst.changed_on_disk()
    # Simulate "someone else" (the CLI or web editor) writing the same file.
    other = ShoppingList(tmp_path / "shopping_list.yaml", KNOWN)
    import os, time
    other.add("rice")
    # Change detection uses the file's modification time. Two writes within the same clock
    # tick could look identical, so push the time forward 5 seconds to make the test reliable.
    st = os.stat(other.path)
    os.utime(other.path, (st.st_atime, st.st_mtime + 5))  # guarantee a different mtime
    assert lst.changed_on_disk()
    lst.load()
    assert [it.name for it in lst.items] == ["pasta", "rice"]


def test_plain_string_items_accepted(tmp_path):
    """A hand-written file with plain names (no status) is accepted as all-pending."""
    p = tmp_path / "l.yaml"
    p.write_text("items: [pasta, rice]\n")
    lst = ShoppingList(p, KNOWN)
    assert [it.name for it in lst.items] == ["pasta", "rice"]


def test_describe(lst):
    """The spoken summary of the list uses the spoken names, not the internal ids."""
    assert "empty" in lst.describe()
    lst.add("tomato_sauce")
    assert "tomato sauce" in lst.describe()
