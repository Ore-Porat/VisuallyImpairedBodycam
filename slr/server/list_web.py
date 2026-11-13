"""Small HTTP server on the laptop: clip downloads for the device (FR-53) and an accessible
shopping list editor for the helper (FR-4).

The editor is a picture catalog of every item the model knows plus the live list. Every
control is an ordinary HTML form with a real button and a real label, so the page works with a
keyboard, any screen reader and even without JavaScript. When scripts run, ``web/editor.js``
submits the same forms in the background, keeps focus where it was, and polls the list so
changes made on the bodycam appear within a second.

WHAT THIS FILE DOES
-------------------
A tiny web server built from Python's standard library. It answers these requests:

1. From the Raspberry Pi: "give me the clip manifest" and "give me clip X.wav". This is how
   the device downloads the speech clips the laptop rendered (see ``clips.py``).
2. From a person's browser (a helper, family member or the shopper with a screen reader):
   the shopping list page at ``http://<laptop>:8767/``, its stylesheet and script under
   ``/static/``, the item pictures under ``/catalog/<item>.svg`` (drawn in ``catalog.py``),
   and ``/list/state.json`` for live updates.
3. Form posts to ``/list/<op>`` (add, remove, current, reset, clear). A normal browser post is
   answered with a redirect back to the page; the script asks for JSON instead.
4. The item prompt editor at ``/prompts``: the CLIP sentences for each item, which is how a
   helper makes an unfamiliar product detectable. Any product name can be typed into the add
   box, not only the presets; a new name becomes a catalog item with generated prompts.
5. Voice commands: ``/list/voice/start``, ``/chunk`` and ``/end`` stream the microphone while
   the shopper talks, and ``/list/say`` takes a typed sentence; both go to the
   ``VoiceAssistant`` in ``voice.py`` ("add chips" -> list edit).

HOW IT FITS IN
--------------
``main.py`` creates the handler with ``make_handler(...)``, giving it callbacks to fetch the
current list and to apply edits (``_list_op_threadsafe``), then starts an ``HttpServer``.
The server runs in its own thread so it never blocks the asyncio event loop.

WORDS YOU WILL SEE
------------------
- HTTP: the protocol browsers use. A GET request asks for a page or file; a POST request
  submits a form. The server replies with a status code (200 OK, 404 not found, 303 "now go
  to this other address") plus headers and a body.
- handler: the class that receives one request and writes one response. ``do_GET`` and
  ``do_POST`` are the method names Python's ``BaseHTTPRequestHandler`` looks for.
- URL path / query: in ``/list?msg=Added`` the path is ``/list`` and the query is ``msg=Added``.
- HTML escaping (``html.escape``): turning characters like ``<`` into ``&lt;`` so user text can
  never be mistaken for HTML tags.
- live region (``role="status"``): a part of the page that screen readers announce
  automatically when its text changes.
- regular expression (regex): a pattern for matching text. ``CLIP_NAME`` allows only letters,
  digits, underscore and dash, which blocks tricks like ``../../secret``.
- progressive enhancement: the page works as plain HTML first; the script only makes it
  smoother.
- FR-xx: requirement numbers in the product spec, ``PRD.pdf``.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, urlparse

from slr.common.types import FOUND
from slr.server.catalog import aisles_for, art_for
from slr.server.voice import VoiceUnavailable

log = logging.getLogger(__name__)

# Only safe characters are allowed in a clip or item name requested over the network.
CLIP_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")

# Largest POST body accepted: about 15 s of 16-bit audio at 48000 samples per second (the page
# sends a piece every quarter second, so this is generous).
MAX_POST_BYTES = 48000 * 2 * 15

# op(name, arg) -> message. Runs on the server's event loop (thread-safe wrapper supplied by main).
ListOp = Callable[[str, str], str]

# The stylesheet and script served under /static/.
WEB_DIR = Path(__file__).with_name("web")
STATIC_TYPES = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}


def _load_static() -> dict[str, tuple[bytes, str, str]]:
    """Read every static file once: published path -> (bytes, content type, short hash)."""
    files = {}
    for path in sorted(WEB_DIR.rglob("*")):
        ctype = STATIC_TYPES.get(path.suffix)
        if path.is_file() and ctype:
            data = path.read_bytes()
            name = path.relative_to(WEB_DIR).as_posix()
            files[name] = (data, ctype, hashlib.sha1(data).hexdigest()[:10])
    return files


STATIC = _load_static()


def static_url(name: str) -> str:
    """URL of a static file with its content hash, so browsers may cache it safely."""
    entry = STATIC.get(name)
    return f"/static/{name}?v={entry[2]}" if entry else f"/static/{name}"


def list_version(items, current: int) -> str:
    """A short fingerprint of everything the page displays about the list.

    When the device toggles an item or changes the current index, this value changes, so the
    open page knows it needs to redraw. It is not a security token; it only avoids pointless
    DOM updates.
    """
    payload = json.dumps([[it.name, it.status] for it in items] + [current],
                         separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def display_name(item: str, known: dict[str, str]) -> str:
    """The spoken name in sentence case for the page: "tomato sauce" -> "Tomato sauce"."""
    spoken = known.get(item, item.replace("_", " "))
    return spoken[:1].upper() + spoken[1:]


def item_counts(items) -> dict[str, int]:
    """How many times each class id is on the list (the catalog shows "On list" / "On list x2")."""
    counts: dict[str, int] = {}
    for it in items:
        counts[it.name] = counts.get(it.name, 0) + 1
    return counts


# --------------------------------------------------------------------------- page pieces

def render_items(items, current: int, known: dict[str, str], uncurated=()) -> str:
    """The list part of the page: a line about the current item and the numbered list.

    The browser script swaps this whole fragment when the list changes. Buttons carry a
    ``data-focus`` key so the script can put keyboard focus back on the same control.
    uncurated: item ids that use a generated prompt. They are marked so the helper knows the
    detection will be weaker until the prompts are edited.
    """
    esc = html.escape
    uncurated = set(uncurated)
    if not items:
        return "<p>The list is empty. Add items from the catalog below.</p>"

    found = sum(1 for it in items if it.status == FOUND)
    cur_name = esc(display_name(items[current].name, known))
    rows = []
    for i, it in enumerate(items):
        name = esc(display_name(it.name, known))
        status = "found" if it.status == FOUND else "not found yet"
        cur = " (current item)" if i == current else ""
        hint = ('<span class="hint"> (generated prompts)</span>'
                if it.name in uncurated else "")
        make_current = "" if i == current else (
            '<form method="post" action="/list/current" class="inline" data-ajax>'
            f'<input type="hidden" name="index" value="{i}">'
            f'<button type="submit" data-focus="current-{i}">Make {name} current</button></form> '
        )
        rows.append(
            f'<li class="{"found" if it.status == FOUND else ""}" data-name="{esc(it.name)}">'
            f'<img src="/catalog/{esc(it.name)}.svg" alt="" width="32" height="32">'
            f'<span class="info">{name}: <span class="state">{status}</span>{cur}{hint}</span> '
            f'{make_current}'
            '<form method="post" action="/list/remove" class="inline" data-ajax>'
            f'<input type="hidden" name="item" value="{esc(it.name)}">'
            f'<button type="submit" data-focus="remove-{i}">Remove {name}</button></form></li>'
        )
    return (f'<p>Current item: <strong>{cur_name}</strong>. {found} of {len(items)} found.</p>'
            f'<ol class="rows">{"".join(rows)}</ol>')


def render_catalog(known: dict[str, str], counts: dict[str, int]) -> str:
    """Every aisle with a picture per item; each picture is a button that adds the item."""
    esc = html.escape
    sections = []
    for title, ids in aisles_for(known):
        tiles = []
        for item in ids:
            name = esc(display_name(item, known))
            n = counts.get(item, 0)
            tiles.append(
                f'<li class="tile{" on" if n else ""}" data-item="{esc(item)}" '
                f'data-search="{esc((known[item] + " " + item + " " + title).lower())}">'
                '<form method="post" action="/list/add" data-ajax>'
                f'<input type="hidden" name="item" value="{esc(item)}">'
                f'<button type="submit" data-focus="add-{esc(item)}">'
                f'<img src="/catalog/{esc(item)}.svg" alt="" width="96" height="96">'
                f'Add {name}</button></form>'
                f'<span class="count">{count_text(n)}</span></li>'
            )
        sections.append(f'<h3>{esc(title)}</h3><ul class="grid">{"".join(tiles)}</ul>')
    return "".join(sections)


def count_text(n: int) -> str:
    """The note under a catalog picture: nothing, "on list" or "on list (2)"."""
    return "" if not n else "on list" if n == 1 else f"on list ({n})"


def state_payload(items, current: int, known: dict[str, str], uncurated=()) -> dict:
    """JSON for the browser script: a version, per-item status, catalog counts, list HTML."""
    return {
        "version": list_version(items, current),
        "items": [
            {"name": it.name, "found": it.status == FOUND, "current": i == current}
            for i, it in enumerate(items)
        ],
        "counts": item_counts(items),
        "html": render_items(items, current, known, uncurated),
        "empty": not items,
    }


def render_page(items, current: int, known: dict[str, str], message: str = "",
                uncurated=()) -> str:
    """Build the HTML for the shopping list page.

    items:   the list entries (name + status).
    current: index of the current item.
    known:   class id -> spoken name: the catalog and the names shown on the page.
    message: optional confirmation text shown in the status live region.
    uncurated: item ids whose prompts are generated (marked in the list).
    """
    esc = html.escape
    version = list_version(items, current)
    disabled = " disabled" if not items else ""
    # Suggestions for the free-text add box. A <datalist>, not a <select>, so any product
    # name can be typed; picking a suggestion is just a shortcut.
    options = "".join(f'<option value="{esc(v)}"></option>'
                      for _, v in sorted(known.items(), key=lambda kv: kv[1].lower()))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- Without JavaScript, reload periodically so another client's edits still appear. -->
<noscript><meta http-equiv="refresh" content="3"></noscript>
<title>SLR SLR shopping list</title>
<link rel="stylesheet" href="{static_url("editor.css")}">
</head>
<body>
<main>
<h1>Shopping list</h1>
<!-- Live region: screen readers announce the confirmation message after each change. -->
<p role="status" id="status">{esc(message)}</p>

<h2>Speak a command</h2>
<!-- The talk button only works with the script (it records the microphone), so it starts hidden. -->
<p id="voice" hidden>
  <button type="button" id="talk" class="talk" aria-pressed="false" aria-describedby="voice-hint">
    Speak a command<span class="listening" aria-hidden="true"> (listening)</span></button>
</p>
<p id="heard" class="heard" aria-hidden="true"></p>
<p id="voice-hint" hidden>Press the button or Alt+V, wait for the beep, then talk: "add chips",
"add 3 water", "add milk and coke", "remove sugar", "what's on my list" or "help". Items go on
the list as soon as they are heard, and each time you pause you hear what was added. Press
again or Escape to stop; listening also stops after a few quiet seconds, and then your whole
list is read back.</p>
<p id="speak-option" hidden><label><input type="checkbox" id="speak-replies" checked>
  Read replies aloud (turn off if your screen reader already reads them)</label></p>
<form method="post" action="/list/say" data-ajax data-say>
  <label for="say">Or type a command: </label><input id="say" name="text" autocomplete="off">
  <button type="submit">Send</button>
</form>

<h2 id="list-heading" tabindex="-1">Items</h2>
<div id="items" data-version="{version}">{render_items(items, current, known, uncurated)}</div>
<div class="whole">
  <form method="post" action="/list/reset" class="inline" data-ajax>
    <button type="submit" data-focus="reset"{disabled}>Mark every item as not found</button>
  </form>
  <form method="post" action="/list/clear" class="inline" data-ajax data-confirm="Remove every item from the list?">
    <button type="submit" data-focus="clear"{disabled}>Remove every item</button>
  </form>
</div>

<p class="hint">Detection quality comes from the prompts. <a href="/prompts">Edit item prompts</a>.</p>

<h2>Add an item</h2>
<!-- Free text with preset suggestions. A new name becomes a catalog item with generated
     prompts, which can then be edited on the prompts page. -->
<form method="post" action="/list/add" data-ajax data-reset>
  <label for="item">Type any item: </label>
  <input id="item" name="item" list="item-names" autocomplete="off" required>
  <datalist id="item-names">{options}</datalist>
  <button type="submit" data-focus="add-typed">Add item</button>
  <p class="hint">Pick a suggestion or type any product name. New items use generated prompts;
  edit them on the <a href="/prompts">item prompts page</a> for better accuracy.</p>
</form>
<p>Or pick one of the preset items:</p>
<p id="search" hidden><label for="q">Search: </label><input id="q" type="search" autocomplete="off"></p>
<p id="no-match" hidden>No items match.</p>
{render_catalog(known, item_counts(items))}
</main>
<script src="{static_url("editor.js")}" defer></script>
</body>
</html>
"""

def prompt_entries(catalog, on_list: list[str]) -> list[dict]:
    """Everything the prompt editor needs: every catalog item plus anything on the list.

    catalog: the ``Catalog`` (slr/common/catalog.py). on_list: the ids on the shopping list.
    Shared by the full server and ``slr list web``.
    """
    ids = list(catalog.item_ids)
    for item_id in on_list:
        if item_id not in ids:
            ids.append(item_id)
    out = []
    for item_id in ids:
        item = catalog.items_for([item_id])[0]
        preset = catalog.preset.get(item_id)
        out.append({
            "id": item_id,
            "spoken": item.spoken,
            "on_list": item_id in on_list,
            "preset": preset is not None,
            "override": item_id in catalog.user and preset is not None,
            "user_only": item_id in catalog.user and preset is None,
            "prompts": list(catalog.prompts_for(item)),
            "generated": not item.prompts,
            "preset_prompts": list(preset.prompts) if preset else [],
        })
    return out


def edit_prompts(catalog, item_id: str, text: str, action: str) -> str:
    """Save ("save", one prompt per line) or restore ("restore") an item's CLIP prompts.

    Returns the confirmation message. Raises CatalogError / ValueError for bad input.
    """
    if action == "restore":
        catalog.restore_prompts(item_id)
        return f"Restored the default prompts for {catalog.spoken(item_id)}."
    if action == "save":
        prompts = [line.strip() for line in text.splitlines() if line.strip()]
        catalog.set_prompts(item_id, prompts)
        return f"Saved {len(prompts)} prompt(s) for {catalog.spoken(item_id)}."
    raise ValueError(f"unknown prompt action {action!r}")


def _badges(entry: dict) -> str:
    """Short human-readable labels for one prompt-editor row."""
    labels = ["on your list"] if entry["on_list"] else []
    labels.append("preset" if entry["preset"] else "custom item")
    if entry.get("override"):
        labels.append("your prompts")
    if entry.get("generated"):
        labels.append("generated prompts")
    return ", ".join(labels)


# Shared styles for the two prompt pages. Kept plain and high-contrast, like the list page.
_PROMPTS_STYLE = """
  body { font-family: system-ui, sans-serif; font-size: 1.25rem; line-height: 1.5;
         max-width: 40rem; margin: 1rem auto; padding: 0 1rem; color: #111; background: #fff; }
  button, input, textarea { font-size: 1.1rem; padding: .4rem .8rem; margin: .2rem 0; }
  textarea { width: 100%; box-sizing: border-box; font-family: inherit; }
  :focus { outline: 3px solid #1a5fb4; outline-offset: 2px; }
  .inline { display: inline; }
  li { margin-bottom: .6rem; }
  .hint { font-size: 1rem; color: #444; }
  #status { font-weight: bold; }
"""


def render_prompts_page(entries: list[dict], message: str = "") -> str:
    """Index of every item whose CLIP prompts can be edited."""
    esc = html.escape
    status = f'<p role="status" id="status">{esc(message)}</p>' if message else \
        '<p role="status" id="status"></p>'
    rows = []
    # Items on the shopping list come first: those are the ones that matter right now.
    for entry in sorted(entries, key=lambda e: (not e["on_list"], e["spoken"].lower())):
        rows.append(
            f'<li><a href="/prompts/edit?item={quote(entry["id"])}">'
            f'Edit prompts for {esc(entry["spoken"])}</a> '
            f'<span class="hint">({esc(_badges(entry))})</span></li>'
        )
    listing = "<ol>" + "".join(rows) + "</ol>" if rows else "<p>No items yet.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLR SLR item prompts</title>
<style>{_PROMPTS_STYLE}</style>
</head>
<body>
<main>
<h1>Item prompts</h1>
{status}
<p>CLIP finds an item by comparing the camera picture with these sentences. More specific
prompts (brand, colour, container) give better detection. Items marked
<strong>generated prompts</strong> still work, but using your own words usually helps.</p>
{listing}
<p><a href="/">Back to the shopping list</a></p>
</main>
</body>
</html>
"""


def render_prompt_edit(entry: dict, message: str = "") -> str:
    """Edit page for one item: one prompt per line, plus restore/remove."""
    esc = html.escape
    status = f'<p role="status" id="status">{esc(message)}</p>' if message else \
        '<p role="status" id="status"></p>'
    text = esc("\n".join(entry["prompts"]))
    generated = ('<p class="hint">These are generated prompts. Replacing them with '
                 'descriptions of the actual product usually improves detection.</p>'
                 if entry.get("generated") else "")
    # Custom wording (a user item, or a preset someone edited) is always read as a photo.
    if not entry["preset"] or entry.get("override"):
        generated += ('<p class="hint">Each prompt is read as a photo: "a red can" becomes '
                      f'"a photo of a red can", and "a photo of {esc(entry["spoken"])}" is always '
                      'included. Photo wording gives clearly better detection.</p>')
    # A preset only needs a restore button when the user has overridden it; a custom item
    # always gets one, because removing its prompts falls back to the generated sentence.
    restore = entry.get("override") or entry.get("user_only")
    restore_label = ("Restore the preset prompts" if entry["preset"]
                     else "Use generated prompts instead")
    restore_form = ""
    if restore:
        restore_form = f"""
<form method="post" action="/prompts" class="inline">
  <input type="hidden" name="item" value="{esc(entry["id"])}">
  <button type="submit" name="action" value="restore">{restore_label}</button>
</form>"""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLR SLR prompts for {esc(entry["spoken"])}</title>
<style>{_PROMPTS_STYLE}</style>
</head>
<body>
<main>
<h1>Prompts for {esc(entry["spoken"])}</h1>
{status}
<p class="hint">{esc(_badges(entry))}.</p>
{generated}
<form method="post" action="/prompts">
  <input type="hidden" name="item" value="{esc(entry["id"])}">
  <label for="prompts">Prompts for {esc(entry["spoken"])}, one per line</label>
  <textarea id="prompts" name="prompts" rows="8" spellcheck="true">{text}</textarea>
  <p class="hint">Describe what the camera sees. For example: "a red can of Coca-Cola",
  "a can of Coca-Cola", "a Coca-Cola can". Several short sentences work better than one
  long one. Up to 12 prompts.</p>
  <button type="submit" name="action" value="save">Save prompts</button>
</form>
{restore_form}
<p><a href="/prompts">Back to all item prompts</a></p>
</main>
</body>
</html>
"""


def make_handler(get_page_data: Callable[[], tuple], list_op: ListOp,
                 manifest: Callable[[], dict[str, str]], clip_path: Callable[[str], object],
                 prompts_data: Callable[[], list[dict]] | None = None,
                 prompt_op: Callable[[str, str, str], str] | None = None,
                 voice=None):
    """Build the request handler class, closing over the callbacks supplied by main.py.

    get_page_data: returns (items, current, known) or the same plus a set of uncurated ids.
    list_op:       applies a list edit: list_op("add", "pasta") -> confirmation message.
    manifest:      returns the clip manifest dict (name -> sha256).
    clip_path:     returns the Path of a clip by name.
    prompts_data:  returns the prompt-editor entries (optional; enables /prompts).
    prompt_op:     saves or restores prompts (optional; enables POST /prompts).
    voice:         a ``VoiceAssistant`` for spoken commands (voice.py), or None for none.

    Python's HTTP server wants a CLASS, not an object, so we define the class inside this
    function; the inner class can see the callbacks above.
    """
    class Handler(BaseHTTPRequestHandler):
        """Handles one HTTP request. Python creates a new instance per request."""

        server_version = "slr-slr"

        def log_message(self, fmt, *args):  # route through logging instead of stderr
            """Override the default request logging to use our logger at debug level."""
            log.debug("http: " + fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
            """Write a complete response: status code, headers, then the body bytes."""
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # The list and clips change, so they are never cached; static files are versioned.
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def _page(self):
            """get_page_data() may return 3 values (old callers) or 4 (with uncurated ids)."""
            data = get_page_data()
            items, current, known = data[0], data[1], data[2]
            uncurated = data[3] if len(data) > 3 else ()
            return items, current, known, uncurated

        def _state(self) -> dict:
            items, current, known, uncurated = self._page()
            return state_payload(items, current, known, uncurated)

        def do_GET(self):
            """Answer a GET: the manifest, a clip, a static file, a picture, the list page or
            the prompt editor."""
            url = urlparse(self.path)
            if url.path == "/clips/manifest.json":
                self._send(200, json.dumps(manifest()).encode(), "application/json")
                return
            if url.path == "/list/state.json":
                self._send(200, json.dumps(self._state()).encode(), "application/json")
                return
            if url.path.startswith("/clips/") and url.path.endswith(".wav"):
                # Strip "/clips/" from the front and ".wav" from the end to get the name.
                name = url.path[len("/clips/"):-len(".wav")]
                # Reject anything but a plain safe name (no path tricks).
                path = clip_path(name) if CLIP_NAME.match(name) else None
                if path is None or not path.exists():
                    self._send(404, b"no such clip", "text/plain")
                    return
                self._send(200, path.read_bytes(), "audio/wav")
                return
            if url.path.startswith("/static/"):
                # Only files read at startup can be served, so no path can escape WEB_DIR.
                entry = STATIC.get(url.path[len("/static/"):])
                if entry is None:
                    self._send(404, b"not found", "text/plain")
                    return
                self._send(200, entry[0], entry[1], cache="public, max-age=31536000, immutable")
                return
            if url.path.startswith("/catalog/") and url.path.endswith(".svg"):
                name = url.path[len("/catalog/"):-len(".svg")]
                if not CLIP_NAME.match(name):
                    self._send(404, b"not found", "text/plain")
                    return
                self._send(200, art_for(name).encode(), "image/svg+xml",
                           cache="public, max-age=86400")
                return
            if url.path in ("/", "/list"):
                # ``?msg=...`` carries the confirmation text after a redirect (see do_POST).
                msg = parse_qs(url.query).get("msg", [""])[0]
                items, current, known, uncurated = self._page()
                self._send(200, render_page(items, current, known, msg, uncurated).encode(),
                           "text/html; charset=utf-8")
                return
            if url.path in ("/prompts", "/prompts/edit") and prompts_data is not None:
                msg = parse_qs(url.query).get("msg", [""])[0]
                entries = prompts_data()
                if url.path == "/prompts":
                    self._send(200, render_prompts_page(entries, msg).encode(),
                               "text/html; charset=utf-8")
                    return
                item_id = parse_qs(url.query).get("item", [""])[0]
                entry = next((e for e in entries if e["id"] == item_id), None)
                if entry is None:
                    self._send(404, b"no such item", "text/plain")
                    return
                self._send(200, render_prompt_edit(entry, msg).encode(),
                           "text/html; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

        def _listen(self, step: str, body: bytes, query: str) -> dict:
            """Live listening: /list/voice/start, then /chunk with each piece of audio, then /end.

            Returns the JSON reply: ok, message, events (see ``VoiceSession``) and, for start,
            the session id the page sends back with every piece.
            """
            if voice is None:
                return {"ok": False, "message": "Voice commands are not available on this server."}
            q = parse_qs(query)
            try:
                if step == "start":
                    rate = q.get("rate", ["16000"])[0]
                    if not rate.isdigit() or not 8000 <= int(rate) <= 48000:
                        return {"ok": False, "message": "Bad audio sample rate."}
                    return {"ok": True, "message": "", "session": voice.open_session(int(rate))}
                sid = q.get("session", [""])[0]
                if step == "chunk":
                    return {"ok": True, "message": "", "events": voice.session(sid).feed(body)}
                if step == "end":
                    events, summary = voice.close_session(sid)
                    return {"ok": True, "message": summary, "events": events}
            except VoiceUnavailable as e:
                return {"ok": False, "message": str(e), "events": []}
            return {"ok": False, "message": f"unknown voice step {step!r}"}

        def do_POST(self):
            """Answer a form submission: apply the edit, then redirect (or reply with JSON)."""
            url = urlparse(self.path)
            if not (url.path.startswith("/list/") or url.path == "/prompts"):
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_POST_BYTES:
                self._send(413, b"too large", "text/plain")
                return
            raw = self.rfile.read(length)
            if url.path == "/prompts":
                self._prompts_post(parse_qs(raw.decode("utf-8", "replace")))
                return
            # The operation is the last part of the path: /list/add -> "add".
            op = url.path[len("/list/"):]
            if op.startswith("voice/"):
                # Live listening from the page script: raw 16-bit audio, always answered in JSON.
                body = self._listen(op[len("voice/"):], raw, url.query)
                if body.get("events") or op == "voice/end":
                    body["state"] = self._state()
                self._send(200, json.dumps(body).encode(), "application/json")
                return
            # Read the form body ("item=pasta" style) and parse it into a dict of lists.
            form = parse_qs(raw.decode("utf-8", "replace"))
            heard = None
            if op == "say":
                # A typed command: same understanding as a spoken one.
                heard = (form.get("text") or [""])[0]
                if voice is None:
                    ok, msg = False, "Voice commands are not available on this server."
                else:
                    ok, msg = voice.handle(heard)
            else:
                arg = (form.get("item") or form.get("index") or [""])[0]
                ok = True
                try:
                    msg = list_op(op, arg)
                except Exception as e:  # ListError and friends: show the message to the helper
                    msg, ok = str(e), False
            if "application/json" in (self.headers.get("Accept") or ""):
                # The page script: answer with the message and the new state in one go.
                body = {"ok": ok, "message": msg, "state": self._state()}
                if heard is not None:
                    body["heard"] = heard
                self._send(200, json.dumps(body).encode(), "application/json")
                return
            # 303 "See Other": tell the browser to GET the page again, carrying the message.
            # This "post-redirect-get" pattern stops a page refresh from repeating the edit.
            self.send_response(303)
            self.send_header("Location", "/?msg=" + quote(msg))
            self.end_headers()

        def _prompts_post(self, form: dict) -> None:
            """Save or restore an item's prompts, then go back to its edit page."""
            if prompt_op is None:
                self._send(404, b"not found", "text/plain")
                return
            item_id = (form.get("item") or [""])[0]
            text = (form.get("prompts") or [""])[0]
            action = (form.get("action") or ["save"])[0]
            try:
                msg = prompt_op(item_id, text, action)
            except Exception as e:  # CatalogError and friends: show the message
                msg = str(e)
            self.send_response(303)
            self.send_header("Location", f"/prompts/edit?item={quote(item_id)}&msg={quote(msg)}")
            self.end_headers()

    return Handler


class HttpServer:
    """Runs the handler on host:port in a background thread."""

    def __init__(self, host: str, port: int, handler):
        """Bind the socket now (so ``port`` is known); call ``start()`` to begin serving."""
        self.httpd = ThreadingHTTPServer((host, port), handler)
        # Per-request threads must not keep the program alive at exit.
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True)

    @property
    def port(self) -> int:
        """The actual port number (useful when the config asked for 0 = "any free port")."""
        return self.httpd.server_address[1]

    def start(self) -> None:
        """Begin serving requests in the background."""
        self.thread.start()

    def stop(self) -> None:
        """Stop serving and close the socket."""
        self.httpd.shutdown()
        self.httpd.server_close()


def serve_list_only(shopping, catalog, host: str, port: int, cfg=None) -> None:
    """Run just the list editor, without the camera, detector or speech (``slr list web``).

    Handy for working on the page. It edits the same ``data/shopping_list.yaml`` and
    ``data/catalog_user.yaml`` as the full server, and re-reads them when someone else (the
    CLI) changes them. Blocks until Ctrl+C.

    shopping: a ``ShoppingList`` built on ``catalog``.
    catalog:  the item ``Catalog`` (presets plus the user's items).
    cfg:      the ServerConfig; when given, voice commands are turned on as in the full server.
    """
    lock = threading.Lock()  # each request runs on its own thread

    def refresh() -> None:
        if catalog.changed_on_disk():
            catalog.reload_user()
            shopping.refresh_known()
        if shopping.changed_on_disk():
            shopping.load()

    def page_data():
        with lock:
            refresh()
            return (list(shopping.items), shopping.current, catalog.item_map(),
                    set(shopping.uncurated_items()))

    def list_op(op: str, arg: str) -> str:
        with lock:
            refresh()
            if op == "add":
                cls = shopping.add(arg)          # a new name becomes a catalog item
                shopping.refresh_known()
                msg = f"Added {catalog.spoken(cls)}."
                if cls in shopping.uncurated_items():
                    msg += (" It uses generated prompts; edit them on the Prompts page for"
                            " better accuracy.")
                return msg
            if op == "remove":
                cls = shopping.remove(arg)
                return f"Removed {catalog.spoken(cls)}."
            if op == "clear":
                shopping.clear()
                return "Removed every item."
            if op == "reset":
                shopping.reset()
                return "Every item is marked not found."
            if op == "current":
                if not arg.isdigit() or int(arg) >= len(shopping.items):
                    raise ValueError("no such item")
                shopping.set_state(tuple(shopping.items), int(arg))
                return f"Current item is now {catalog.spoken(shopping.items[int(arg)].name)}."
            raise ValueError(f"unknown list operation {op!r}")

    def prompts_data():
        with lock:
            refresh()
            return prompt_entries(catalog, [it.name for it in shopping.items])

    def prompt_op(item_id: str, text: str, action: str) -> str:
        with lock:
            msg = edit_prompts(catalog, item_id, text, action)
            shopping.refresh_known()
            return msg

    voice = None
    if cfg is not None:
        from slr.server.voice import build_voice
        voice = build_voice(cfg, catalog, list_op, page_data)
    server = HttpServer(host, port, make_handler(page_data, list_op, lambda: {}, lambda name: None,
                                                 prompts_data, prompt_op, voice))
    server.start()
    print(f"Shopping list editor on http://localhost:{server.port}/  (Ctrl+C to stop)", flush=True)
    try:
        server.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
