"""JSONL session logs (FR-72): one line per frame, detection set, state change, cue, etc.

WHAT THIS FILE DOES
-------------------
While the server runs, it writes a diary of everything that happened to a text file in
``data/logs/``. Each line is one event: a camera frame arrived, the detector found something, the
guidance engine changed mode, a cue was sent to the device, and so on. Later, ``slr bench --log``
reads these files back to compute latency and frame-loss numbers, and people can read them to
understand what the device did during a test.

HOW IT FITS IN
--------------
The server's main loop (``slr/server/main.py``) and the engine wrapper (``slr/server/core.py``)
call ``SessionLog.write(...)`` whenever something notable happens. Nothing else depends on it; if
no log folder is given, every call quietly does nothing.

WORDS YOU WILL SEE
------------------
- FR-72: a requirement number from the product spec, ``PRD.pdf`` (FR = functional requirement).
- JSON: a plain-text way of writing data, e.g. ``{"kind": "cue", "text": "Pasta found"}``.
- JSONL: "JSON Lines". A file where every line is one complete JSON object. Easy to append to
  and easy to read back one line at a time.
- thread / lock: the server does several things at once on separate "threads". A lock makes
  sure two threads never write to the file at the same moment and jumble their lines together.
"""

# This import lets us write newer style type hints (like ``Path | None``) on older Pythons.
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path


class SessionLog:
    """Appends one JSON line per event to a file (or does nothing if no folder was given).

    Create it once at server start with the folder to write into. Every call to ``write`` adds
    a line. Call ``close`` at shutdown.
    """

    def __init__(self, logs_dir: Path | None, name: str | None = None):
        """Open (create) the log file.

        logs_dir: folder for the log files, e.g. ``data/logs``. ``None`` disables logging
                  entirely (used by tests and by replay).
        name:     file name without the ``.jsonl`` ending. Defaults to ``session_<date>_<time>``.
        """
        self.path: Path | None = None
        self._f = None
        # The lock guards the file: several threads call write(), one at a time please.
        self._lock = threading.Lock()
        if logs_dir is not None:
            # Make sure the folder exists (parents=True creates missing parent folders too).
            logs_dir.mkdir(parents=True, exist_ok=True)
            name = name or datetime.now().strftime("session_%Y%m%d_%H%M%S")
            self.path = logs_dir / f"{name}.jsonl"
            # "a" = append mode, buffering=1 = flush after every line so a crash loses nothing.
            self._f = open(self.path, "a", encoding="utf-8", buffering=1)

    def write(self, kind: str, **fields) -> None:
        """Append one event line.

        kind:     a short word saying what kind of event this is ("frame", "cue", "state", ...).
        **fields: any extra named values; they become keys in the JSON object. (``**fields``
                  means "accept any number of keyword arguments and collect them in a dict".)
        """
        if self._f is None:
            # Logging is disabled; silently ignore.
            return
        # Every record starts with the current time (seconds since 1970) and the kind.
        rec = {"t": round(time.time(), 4), "kind": kind, **fields}
        # separators=(",", ":") removes spaces to keep lines short; default=str turns anything
        # JSON doesn't know (like Path objects) into text instead of crashing.
        line = json.dumps(rec, separators=(",", ":"), default=str)
        # ``with self._lock:`` takes the lock, runs the block, and releases it even on error.
        with self._lock:
            self._f.write(line + "\n")

    def close(self) -> None:
        """Close the file. Safe to call more than once."""
        if self._f:
            self._f.close()
            self._f = None


def read_log(path: Path) -> list[dict]:
    """Read a whole JSONL log back into a list of dicts (one per line). Used by ``slr bench``."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip blank lines; parse every other line from JSON text back into a dict.
            if line:
                out.append(json.loads(line))
    return out
