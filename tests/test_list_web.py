"""Tests for the shopping-list page and its live-update endpoint (FR-4)."""

import json
from urllib.request import urlopen

from slr.common.types import FOUND, ListItem
from slr.server.list_web import (
    HttpServer,
    list_version,
    make_handler,
    render_items,
    render_page,
    render_prompt_edit,
    render_prompts_page,
    state_payload,
)

KNOWN = {"cheerios": "Cheerios", "milk": "Milk"}


def test_render_items_marks_status_and_current_item():
    items = [ListItem("cheerios"), ListItem("milk", FOUND)]
    markup = render_items(items, 1, KNOWN)
    assert "1. Cheerios: <span class=\"state\">not found yet</span>" in markup
    assert "2. Milk: <span class=\"state\">found</span>" in markup
    assert '<span class="current"> (current item)</span>' in markup
    assert 'data-name="cheerios"' in markup


def test_render_page_includes_live_update_hooks():
    page = render_page([ListItem("cheerios")], 0, KNOWN)
    assert 'id="items"' in page
    assert 'data-version="' in page
    assert "/list/state.json" in page
    assert "setInterval(refresh, 1000)" in page
    assert '<noscript><meta http-equiv="refresh" content="3"></noscript>' in page


def test_list_version_changes_when_status_or_current_changes():
    items = [ListItem("cheerios"), ListItem("milk")]
    base = list_version(items, 0)
    assert base != list_version([ListItem("cheerios", FOUND), ListItem("milk")], 0)
    assert base != list_version(items, 1)


def test_state_payload_has_version_items_and_html():
    items = [ListItem("cheerios"), ListItem("milk", FOUND)]
    payload = state_payload(items, 1, KNOWN)
    assert payload["version"] == list_version(items, 1)
    assert payload["items"] == [
        {"name": "cheerios", "found": False, "current": False},
        {"name": "milk", "found": True, "current": True},
    ]
    assert "Milk" in payload["html"]
    json.dumps(payload)  # must be serializable for the endpoint


def test_state_endpoint_serves_the_current_list():
    items = [ListItem("cheerios"), ListItem("milk", FOUND)]

    def page_data():
        return list(items), 1, KNOWN

    handler = make_handler(page_data, lambda op, arg: "", lambda: {}, lambda name: None)
    server = HttpServer("127.0.0.1", 0, handler)
    server.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.port}/list/state.json", timeout=5) as response:
            payload = json.load(response)
    finally:
        server.stop()

    assert payload["version"] == list_version(items, 1)
    assert payload["items"][1]["found"] is True
    assert payload["items"][1]["current"] is True
    assert "Cheerios" in payload["html"]


# --------------------------------------------------------------------------- free text + prompts


ENTRY = {
    "id": "pasta", "spoken": "pasta", "on_list": True, "preset": True,
    "override": False, "user_only": False, "prompts": ["a package of pasta"],
    "generated": False, "preset_prompts": ["a package of pasta"],
}


def test_render_page_offers_free_text_with_preset_suggestions():
    """The add box is a real text input with a datalist, not a closed drop-down."""
    page = render_page([ListItem("cheerios")], 0, KNOWN)
    assert 'name="item" list="item-names"' in page
    assert '<datalist id="item-names">' in page
    assert '<option value="Milk"></option>' in page
    assert 'href="/prompts"' in page


def test_render_items_marks_items_with_generated_prompts():
    items = [ListItem("cheerios"), ListItem("milk")]
    markup = render_items(items, 0, KNOWN, uncurated={"milk"})
    assert "generated prompts" in markup
    assert markup.count("generated prompts") == 1


def test_prompt_index_and_edit_page_render():
    index = render_prompts_page([ENTRY])
    assert "Edit prompts for pasta" in index
    assert "preset" in index

    edit = render_prompt_edit(ENTRY)
    assert 'name="prompts"' in edit
    assert "a package of pasta" in edit
    assert "Save prompts" in edit
    assert "Restore the preset prompts" not in edit   # nothing to restore yet


def test_prompt_edit_page_escapes_user_text():
    entry = dict(ENTRY, spoken="<script>", prompts=["a <b> prompt"])
    page = render_prompt_edit(entry)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;b&gt;" in page


def test_prompts_endpoint_saves_and_shows_the_message():
    from urllib.parse import urlencode
    from urllib.request import Request

    def page_data():
        return [ListItem("pasta")], 0, {"pasta": "pasta"}

    calls = []

    def prompt_op(item, text, action):
        calls.append((item, text, action))
        return f"Saved prompts for {item}."

    handler = make_handler(page_data, lambda op, arg: "", lambda: {}, lambda name: None,
                           lambda: [ENTRY], prompt_op)
    server = HttpServer("127.0.0.1", 0, handler)
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        with urlopen(base + "/prompts", timeout=5) as response:
            assert "Edit prompts for pasta" in response.read().decode()
        body = urlencode({"item": "pasta", "prompts": "a blue pasta box",
                          "action": "save"}).encode()
        request = Request(base + "/prompts", data=body, method="POST")
        with urlopen(request, timeout=5) as response:   # urllib follows the 303
            page = response.read().decode()
    finally:
        server.stop()

    assert calls == [("pasta", "a blue pasta box", "save")]
    assert "Saved prompts for pasta." in page
    assert "Prompts for pasta" in page
