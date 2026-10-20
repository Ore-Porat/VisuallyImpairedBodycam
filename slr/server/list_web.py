"""Small HTTP server on the laptop: clip downloads for the device (FR-53) and an accessible
shopping list editor for the helper (FR-4).

The editor is plain HTML forms: real labels, real buttons, no scripts, no mouse-only controls,
and a status message in a live region after every change, so it works with any screen reader.

WHAT THIS FILE DOES
-------------------
A tiny web server built from Python's standard library. It answers two kinds of requests:

1. From the Raspberry Pi: "give me the clip manifest" and "give me clip X.wav". This is how
   the device downloads the speech clips the laptop rendered (see ``clips.py``).
2. From a person's browser (a helper, family member or the shopper with a screen reader):
   the shopping list page at ``http://<laptop>:8767/``. It shows the list and has buttons to
   add, remove, re-order and reset items. Every button is an ordinary HTML form, so the page
   needs no JavaScript and works with keyboard and screen readers.

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
- thread: the server runs on its own thread; ``ThreadingHTTPServer`` additionally handles each
  request on a thread so a slow download never blocks the list page.
- callback: a function handed in by ``main.py`` to be called later.
- FR-xx: requirement numbers in the product spec, ``PRD.pdf``.
"""

from __future__ import annotations

import html
import json
import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, quote, urlparse

from slr.common.types import FOUND

log = logging.getLogger(__name__)

# Only safe characters are allowed in a clip name requested over the network.
CLIP_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")

# op(name, arg) -> message. Runs on the server's event loop (thread-safe wrapper supplied by main).
ListOp = Callable[[str, str], str]


def render_page(items, current: int, known: dict[str, str], message: str = "") -> str:
    """Build the HTML for the shopping list page as one big string.

    items:   the list entries (name + status).
    current: index of the current item (gets the "(current item)" tag).
    known:   class id -> spoken name, used both for display and for the "add" drop-down.
    message: optional confirmation text shown in the status live region.
    """
    esc = html.escape
    rows = []
    # One <li> per item, each with a "Remove" form and a "Make current" form.
    for i, it in enumerate(items):
        spoken = esc(known.get(it.name, it.name))
        status = "found" if it.status == FOUND else "not found yet"
        cur = " (current item)" if i == current else ""
        rows.append(
            f"<li><span>{i + 1}. {spoken}: {status}{cur}</span> "
            f'<form method="post" action="/list/remove" class="inline">'
            f'<input type="hidden" name="item" value="{esc(it.name)}">'
            f'<button type="submit">Remove {spoken}</button></form> '
            f'<form method="post" action="/list/current" class="inline">'
            f'<input type="hidden" name="index" value="{i}">'
            f'<button type="submit">Make {spoken} the current item</button></form></li>'
        )
    items_html = "<ol>" + "".join(rows) + "</ol>" if rows else "<p>The list is empty.</p>"
    # The drop-down of every item the model knows, sorted by spoken name.
    options = "".join(f'<option value="{esc(k)}">{esc(v)}</option>'
                      for k, v in sorted(known.items(), key=lambda kv: kv[1]))
    # The status paragraph is always present (even empty) so screen readers can watch it.
    status = f'<p role="status" id="status">{esc(message)}</p>' if message else \
        '<p role="status" id="status"></p>'
    # This is an f-string, so single { } insert Python values and the CSS braces are doubled
    # ({{ }}) to mean literal braces.
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<!-- Makes the page scale properly on phones. -->
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLR SLR shopping list</title>
<style>
  /* Large, high-contrast text; narrow column that is easy to read. */
  body {{ font-family: system-ui, sans-serif; font-size: 1.25rem; line-height: 1.5;
         max-width: 40rem; margin: 1rem auto; padding: 0 1rem; color: #111; background: #fff; }}
  button, select {{ font-size: 1.1rem; padding: .4rem .8rem; margin: .2rem 0; }}
  /* A thick visible focus ring so keyboard users can see where they are. */
  :focus {{ outline: 3px solid #1a5fb4; outline-offset: 2px; }}
  .inline {{ display: inline; }}
  li {{ margin-bottom: .6rem; }}
  #status {{ font-weight: bold; }}
</style>
</head>
<body>
<main>
<h1>Shopping list</h1>
<!-- Live region: screen readers announce the confirmation message after each change. -->
{status}
<h2>Items</h2>
<!-- The numbered list of items, each with Remove / Make current buttons. -->
{items_html}
<h2>Add an item</h2>
<!-- Submitting this form sends a POST to /list/add with the chosen class id. -->
<form method="post" action="/list/add">
  <label for="item">Item to add</label>
  <select id="item" name="item">{options}</select>
  <button type="submit">Add item</button>
</form>
<h2>Whole list</h2>
<form method="post" action="/list/reset" class="inline">
  <button type="submit">Mark every item as not found</button>
</form>
<form method="post" action="/list/clear" class="inline">
  <button type="submit">Remove every item</button>
</form>
</main>
</body>
</html>
"""


def make_handler(get_page_data: Callable[[], tuple], list_op: ListOp,
                 manifest: Callable[[], dict[str, str]], clip_path: Callable[[str], object]):
    """Build the request handler class, closing over the callbacks supplied by main.py.

    get_page_data: returns (items, current, known) for the page.
    list_op:       applies an edit: list_op("add", "pasta") -> confirmation message.
    manifest:      returns the clip manifest dict (name -> sha256).
    clip_path:     returns the Path of a clip by name.

    Python's HTTP server wants a CLASS, not an object, so we define the class inside this
    function; the inner class can see the four callbacks above.
    """
    class Handler(BaseHTTPRequestHandler):
        """Handles one HTTP request. Python creates a new instance per request."""

        server_version = "slr-slr"

        def log_message(self, fmt, *args):  # route through logging instead of stderr
            """Override the default request logging to use our logger at debug level."""
            log.debug("http: " + fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            """Write a complete response: status code, headers, then the body bytes."""
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # Never cache: the list changes and clips can be re-rendered.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            """Answer a GET: the manifest, a clip file, or the list page."""
            url = urlparse(self.path)
            if url.path == "/clips/manifest.json":
                self._send(200, json.dumps(manifest()).encode(), "application/json")
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
            if url.path in ("/", "/list"):
                # ``?msg=...`` carries the confirmation text after a redirect (see do_POST).
                msg = parse_qs(url.query).get("msg", [""])[0]
                items, current, known = get_page_data()
                self._send(200, render_page(items, current, known, msg).encode(), "text/html; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            """Answer a form submission: apply the edit, then redirect back to the page."""
            url = urlparse(self.path)
            if not url.path.startswith("/list/"):
                self._send(404, b"not found", "text/plain")
                return
            # Read the form body ("item=pasta" style) and parse it into a dict of lists.
            length = int(self.headers.get("Content-Length") or 0)
            form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            # The operation is the last part of the path: /list/add -> "add".
            op = url.path[len("/list/"):]
            arg = (form.get("item") or form.get("index") or [""])[0]
            try:
                msg = list_op(op, arg)
            except Exception as e:  # ListError and friends: show the message to the helper
                msg = str(e)
            # 303 "See Other": tell the browser to GET the page again, carrying the message.
            # This "post-redirect-get" pattern stops a page refresh from repeating the edit.
            self.send_response(303)
            self.send_header("Location", "/?msg=" + quote(msg))
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
