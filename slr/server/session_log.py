"""JSONL session logs (FR-72): one line per frame, detection set, state change, cue, etc."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path


class SessionLog:
    def __init__(self, logs_dir: Path | None, name: str | None = None):
        self.path: Path | None = None
        self._f = None
        self._lock = threading.Lock()
        if logs_dir is not None:
            logs_dir.mkdir(parents=True, exist_ok=True)
            name = name or datetime.now().strftime("session_%Y%m%d_%H%M%S")
            self.path = logs_dir / f"{name}.jsonl"
            self._f = open(self.path, "a", encoding="utf-8", buffering=1)

    def write(self, kind: str, **fields) -> None:
        if self._f is None:
            return
        rec = {"t": round(time.time(), 4), "kind": kind, **fields}
        line = json.dumps(rec, separators=(",", ":"), default=str)
        with self._lock:
            self._f.write(line + "\n")

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


def read_log(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
