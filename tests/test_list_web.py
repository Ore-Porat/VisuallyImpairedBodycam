"""The shopping list web editor (FR-4): page, static files and JSON posts."""

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from slr.common.catalog import load_catalog
from slr.common.types import FOUND, PENDING
from slr.server.list_web import STATIC, HttpServer, make_handler, render_page, state_payload
from slr.server.session import ListError, ListItem

ROOT = Path(__file__).resolve().parents[1]
KNOWN = load_catalog(ROOT / "config/catalog.yaml").item_map()


def test_page_works_without_javascript():
    items = [ListItem("pasta", FOUND), ListItem("tomato_sauce", PENDING)]
    page = render_page(items, 1, KNOWN, "Added pasta.")
    # Every action is a real form post.
    for action in ("add", "remove", "current", "reset", "clear"):
        assert f'action="/list/{action}"' in page
    # Items are added only by voice or the free-text box (with suggestions): no picture catalog.
    assert len(re.findall(r'action="/list/add"', page)) == 1
    assert '<option value="tomato sauce">' in page
    assert "<img" not in page and 'class="tile' not in page
    # Short visible button text; the item's name is in the accessible label.
    assert 'aria-label="Remove Pasta">Remove</button>' in page
    assert 'aria-label="Make Pasta current">Make current</button>' in page
    assert 'role="status"' in page and "Added pasta." in page
    assert '<th scope="row" class="name">Tomato sauce</th>' in page and '<span class="tag">Current</span>' in page
    assert '<noscript><meta http-equiv="refresh"' in page


def test_empty_list_disables_whole_list_buttons():
    page = render_page([], 0, KNOWN)
    assert "The list is empty." in page
    assert page.count("disabled") == 2


def test_state_payload_counts_duplicates():
    items = [ListItem("chips"), ListItem("chips"), ListItem("milk", FOUND)]
    state = state_payload(items, 0, KNOWN)
    assert state["counts"] == {"chips": 2, "milk": 1}
    assert state["empty"] is False
    assert "1 of 3 found" in state["html"]


@pytest.fixture
def server():
    items: list[ListItem] = []

    def op(name, arg):
        if name == "add":
            if arg not in KNOWN:
                raise ListError(f"'{arg}' is not an item the model knows")
            items.append(ListItem(arg))
            return f"Added {KNOWN[arg]}."
        raise ListError("unsupported in this test")

    srv = HttpServer("127.0.0.1", 0, make_handler(lambda: (list(items), 0, KNOWN), op,
                                                  lambda: {}, lambda name: None))
    srv.start()
    yield f"http://127.0.0.1:{srv.port}", items
    srv.stop()


def _get(url):
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), b""


def test_static_files_are_served(server):
    base, _ = server
    assert set(STATIC) == {"editor.css", "editor.js"}
    assert _get(base + "/static/editor.css")[:2] == (200, "text/css; charset=utf-8")
    # Nothing outside the bundled files, and no path tricks.
    assert _get(base + "/static/../list_web.py")[0] == 404
    assert _get(base + "/static/catalog.py")[0] == 404
    assert _get(base + "/catalog/milk.svg")[0] == 404   # the item pictures are gone


def test_post_answers_json_for_the_script_and_redirects_otherwise(server):
    base, items = server
    req = urllib.request.Request(base + "/list/add", data=b"item=coffee", method="POST",
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as r:
        reply = json.load(r)
    assert reply["ok"] is True and reply["message"] == "Added coffee."
    assert reply["state"]["counts"] == {"coffee": 1}

    req = urllib.request.Request(base + "/list/add", data=b"item=pizza", method="POST",
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as r:
        reply = json.load(r)
    assert reply["ok"] is False and "pizza" in reply["message"]

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    with pytest.raises(urllib.error.HTTPError) as e:
        opener.open(urllib.request.Request(base + "/list/add", data=b"item=milk", method="POST"))
    assert e.value.code == 303
    assert e.value.headers["Location"] == "/?msg=Added%20milk."
    assert [it.name for it in items] == ["coffee", "milk"]


# --------------------------------------------------------------------------- free text + prompts
# (merged from the partner's tests/test_list_web.py, 2026-11-12)

from urllib.parse import urlencode  # noqa: E402

from slr.server.list_web import (  # noqa: E402
    edit_prompts, list_version, prompt_entries, render_items, render_prompt_edit,
    render_prompts_page, static_url,
)

SMALL = {"cheerios": "Cheerios", "milk": "Milk"}

ENTRY = {
    "id": "pasta", "spoken": "pasta", "on_list": True, "preset": True,
    "override": False, "user_only": False, "prompts": ["a package of pasta"],
    "generated": False, "preset_prompts": ["a package of pasta"],
}


def test_render_items_marks_status_and_current_item():
    markup = render_items([ListItem("cheerios"), ListItem("milk", FOUND)], 1, SMALL)
    assert '<table class="rows">' in markup
    assert '<th scope="row" class="name">Cheerios</th><td class="state">Not found</td>' in markup
    assert ('<tr class="found current" data-name="milk"><td class="num">2</td>'
            '<th scope="row" class="name">Milk</th><td class="state">Found</td>'
            '<td class="pick"><span class="tag">Current</span></td>') in markup
    assert 'data-name="cheerios"' in markup


def test_render_page_includes_live_update_hooks():
    page = render_page([ListItem("cheerios")], 0, SMALL)
    assert 'id="items"' in page and 'data-version="' in page
    assert static_url("editor.js") in page        # the script polls /list/state.json
    assert '<noscript><meta http-equiv="refresh" content="3"></noscript>' in page


def test_list_version_changes_when_status_or_current_changes():
    items = [ListItem("cheerios"), ListItem("milk")]
    base = list_version(items, 0)
    assert base != list_version([ListItem("cheerios", FOUND), ListItem("milk")], 0)
    assert base != list_version(items, 1)


def test_state_payload_has_version_items_and_html():
    items = [ListItem("cheerios"), ListItem("milk", FOUND)]
    payload = state_payload(items, 1, SMALL)
    assert payload["version"] == list_version(items, 1)
    assert payload["items"] == [
        {"name": "cheerios", "found": False, "current": False},
        {"name": "milk", "found": True, "current": True},
    ]
    assert "Milk" in payload["html"]
    json.dumps(payload)


def test_render_page_offers_free_text_with_preset_suggestions():
    """The add box is a real text input with a datalist, not a closed drop-down."""
    page = render_page([ListItem("cheerios")], 0, SMALL)
    assert 'name="item" list="item-names"' in page
    assert '<datalist id="item-names">' in page
    assert '<option value="Milk"></option>' in page
    # The prompt editor stays at /prompts but is not linked from the list page.
    assert 'href="/prompts"' not in page


def test_render_items_does_not_mention_generated_prompts():
    # The list stays plain: prompt quality is only shown on the prompts page.
    markup = render_items([ListItem("cheerios"), ListItem("milk")], 0, SMALL, uncurated={"milk"})
    assert "generated prompts" not in markup


def test_prompt_index_and_edit_page_render():
    index = render_prompts_page([ENTRY])
    assert "Edit prompts for pasta" in index and "preset" in index
    edit = render_prompt_edit(ENTRY)
    assert 'name="prompts"' in edit and "a package of pasta" in edit
    assert "Save prompts" in edit
    assert "Restore the preset prompts" not in edit   # nothing to restore yet


def test_prompt_edit_page_escapes_user_text():
    page = render_prompt_edit(dict(ENTRY, spoken="<script>", prompts=["a <b> prompt"]))
    assert "<script>" not in page
    assert "&lt;script&gt;" in page and "&lt;b&gt;" in page


def test_prompts_endpoint_saves_and_shows_the_message():
    calls = []

    def prompt_op(item, text, action):
        calls.append((item, text, action))
        return f"Saved prompts for {item}."

    handler = make_handler(lambda: ([ListItem("pasta")], 0, {"pasta": "pasta"}),
                           lambda op, arg: "", lambda: {}, lambda name: None,
                           lambda: [ENTRY], prompt_op)
    srv = HttpServer("127.0.0.1", 0, handler)
    srv.start()
    try:
        base = f"http://127.0.0.1:{srv.port}"
        with urllib.request.urlopen(base + "/prompts", timeout=5) as r:
            assert "Edit prompts for pasta" in r.read().decode()
        body = urlencode({"item": "pasta", "prompts": "a blue pasta box", "action": "save"}).encode()
        with urllib.request.urlopen(urllib.request.Request(base + "/prompts", data=body,
                                                           method="POST"), timeout=5) as r:
            page = r.read().decode()                  # urllib follows the 303
    finally:
        srv.stop()
    assert calls == [("pasta", "a blue pasta box", "save")]
    assert "Saved prompts for pasta." in page and "Prompts for pasta" in page


def test_shared_prompt_helpers_edit_the_user_catalog(tmp_path):
    """The prompt editor logic used by both the server and `slr list web`."""
    cat = load_catalog(ROOT / "config/catalog.yaml", tmp_path / "catalog_user.yaml")
    entry = next(e for e in prompt_entries(cat, ["kombucha"]) if e["id"] == "kombucha")
    assert entry["on_list"] and entry["generated"] and not entry["preset"]
    assert edit_prompts(cat, "pasta", "a blue pasta box\n\n", "save") == "Saved 1 prompt(s) for pasta."
    assert cat.item("pasta").prompts == ("a blue pasta box",)
    edit_prompts(cat, "pasta", "", "restore")
    assert cat.prompts_for("pasta")[0] == "a package of pasta"
    with pytest.raises(ValueError):
        edit_prompts(cat, "pasta", "", "explode")
