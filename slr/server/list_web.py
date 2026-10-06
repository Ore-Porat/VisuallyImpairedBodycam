"""Small HTTP server on the laptop: clip downloads for the device (FR-53) and an accessible
shopping list editor for the helper (FR-4).

The editor is plain HTML forms: real labels, real buttons, no scripts, no mouse-only controls,
and a status message in a live region after every change, so it works with any screen reader.
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

CLIP_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")

# op(name, arg) -> message. Runs on the server's event loop (thread-safe wrapper supplied by main).
ListOp = Callable[[str, str], str]


def render_page(items, current: int, known: dict[str, str], message: str = "") -> str:
    esc = html.escape
    rows = []
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
    options = "".join(f'<option value="{esc(k)}">{esc(v)}</option>'
                      for k, v in sorted(known.items(), key=lambda kv: kv[1]))
    status = f'<p role="status" id="status">{esc(message)}</p>' if message else \
        '<p role="status" id="status"></p>'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLR SLR shopping list</title>
<style>
  body {{ font-family: system-ui, sans-serif; font-size: 1.25rem; line-height: 1.5;
         max-width: 40rem; margin: 1rem auto; padding: 0 1rem; color: #111; background: #fff; }}
  button, select {{ font-size: 1.1rem; padding: .4rem .8rem; margin: .2rem 0; }}
  :focus {{ outline: 3px solid #1a5fb4; outline-offset: 2px; }}
  .inline {{ display: inline; }}
  li {{ margin-bottom: .6rem; }}
  #status {{ font-weight: bold; }}
</style>
</head>
<body>
<main>
<h1>Shopping list</h1>
{status}
<h2>Items</h2>
{items_html}
<h2>Add an item</h2>
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
    class Handler(BaseHTTPRequestHandler):
        server_version = "slr-slr"

        def log_message(self, fmt, *args):  # route through logging instead of stderr
            log.debug("http: " + fmt, *args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/clips/manifest.json":
                self._send(200, json.dumps(manifest()).encode(), "application/json")
                return
            if url.path.startswith("/clips/") and url.path.endswith(".wav"):
                name = url.path[len("/clips/"):-len(".wav")]
                path = clip_path(name) if CLIP_NAME.match(name) else None
                if path is None or not path.exists():
                    self._send(404, b"no such clip", "text/plain")
                    return
                self._send(200, path.read_bytes(), "audio/wav")
                return
            if url.path in ("/", "/list"):
                msg = parse_qs(url.query).get("msg", [""])[0]
                items, current, known = get_page_data()
                self._send(200, render_page(items, current, known, msg).encode(), "text/html; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            url = urlparse(self.path)
            if not url.path.startswith("/list/"):
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            form = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            op = url.path[len("/list/"):]
            arg = (form.get("item") or form.get("index") or [""])[0]
            try:
                msg = list_op(op, arg)
            except Exception as e:  # ListError and friends: show the message to the helper
                msg = str(e)
            self.send_response(303)
            self.send_header("Location", "/?msg=" + quote(msg))
            self.end_headers()

    return Handler


class HttpServer:
    def __init__(self, host: str, port: int, handler):
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="http", daemon=True)

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
