import pytest

from slr.common.types import FOUND, PENDING, ListItem
from slr.server.session import ListError, ShoppingList, normalize_name

KNOWN = {"pasta": "pasta", "rice": "rice", "tomato_sauce": "tomato sauce", "coffee": "coffee"}


@pytest.fixture
def lst(tmp_path):
    return ShoppingList(tmp_path / "shopping_list.yaml", KNOWN)


def test_add_known_items_and_persist(lst, tmp_path):
    lst.add("pasta")
    lst.add("Tomato Sauce")
    again = ShoppingList(tmp_path / "shopping_list.yaml", KNOWN)
    assert [it.name for it in again.items] == ["pasta", "tomato_sauce"]
    assert all(it.status == PENDING for it in again.items)


def test_unknown_item_rejected_with_valid_list(lst):
    with pytest.raises(ListError) as e:
        lst.add("spaghetti")
    assert "pasta" in str(e.value) and "tomato_sauce" in str(e.value)
    assert lst.items == []


def test_normalize_accepts_id_spoken_and_spaces():
    assert normalize_name("tomato_sauce", KNOWN) == "tomato_sauce"
    assert normalize_name("  tomato   sauce ", KNOWN) == "tomato_sauce"
    assert normalize_name("PASTA", KNOWN) == "pasta"


def test_remove_and_clear(lst):
    for n in ("pasta", "rice", "coffee"):
        lst.add(n)
    lst.current = 2
    lst.remove("pasta")
    assert [it.name for it in lst.items] == ["rice", "coffee"]
    assert lst.current == 1  # still pointing at coffee
    with pytest.raises(ListError):
        lst.remove("pasta")
    lst.clear()
    assert lst.items == [] and lst.current == 0


def test_status_persists_and_reset(lst, tmp_path):
    lst.add("pasta")
    lst.add("rice")
    lst.set_state((ListItem("pasta", FOUND), ListItem("rice")), 1)
    again = ShoppingList(tmp_path / "shopping_list.yaml", KNOWN)
    assert again.items[0].status == FOUND and again.current == 1
    again.reset()
    assert all(it.status == PENDING for it in again.items) and again.current == 0


def test_changed_on_disk_detects_external_edit(lst, tmp_path):
    lst.add("pasta")
    assert not lst.changed_on_disk()
    other = ShoppingList(tmp_path / "shopping_list.yaml", KNOWN)
    import os, time
    other.add("rice")
    st = os.stat(other.path)
    os.utime(other.path, (st.st_atime, st.st_mtime + 5))  # guarantee a different mtime
    assert lst.changed_on_disk()
    lst.load()
    assert [it.name for it in lst.items] == ["pasta", "rice"]


def test_plain_string_items_accepted(tmp_path):
    p = tmp_path / "l.yaml"
    p.write_text("items: [pasta, rice]\n")
    lst = ShoppingList(p, KNOWN)
    assert [it.name for it in lst.items] == ["pasta", "rice"]


def test_describe(lst):
    assert "empty" in lst.describe()
    lst.add("tomato_sauce")
    assert "tomato sauce" in lst.describe()
