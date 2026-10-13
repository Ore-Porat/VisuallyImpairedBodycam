"""Volume dial: MCP3008 reading, calibration curve, filtering, hysteresis (FR-44, FR-45, FR-46).

The dial is a variable resistor (4.64k min, 6.04k middle, 7.0k max, roughly logarithmic) in a
divider read by an MCP3008 (SPI). ``VolumeCurve`` maps raw counts to a 0..1 dial position through the
three calibration points; ``VolumeFilter`` turns noisy samples into a steady 0..10 level.
Both are pure and unit tested.
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
import time
from collections import deque
from typing import Callable

from slr.common.config import AdcConfig, VolumeCalibration, VolumeConfig

log = logging.getLogger(__name__)

LEVELS = 10


class VolumeCurve:
    def __init__(self, cal: VolumeCalibration):
        self.cal = cal
        self.sign = 1.0 if cal.raw_max > cal.raw_min else -1.0  # dial wired backwards is fine
        x0, x1, x2 = (self.sign * v for v in (cal.raw_min, cal.raw_mid, cal.raw_max))
        self.x = (x0, x1, x2)
        self._log = None
        if cal.curve == "log":
            self._log = _fit_log(x0, x1, x2)
            if self._log is None:
                log.warning("log curve does not fit the calibration points; using piecewise")

    def position(self, raw: float) -> float:
        x = self.sign * raw
        x0, x1, x2 = self.x
        if self._log is not None:
            p = self._log(min(max(x, x0), x2))
        elif x <= x1:
            p = 0.5 * (x - x0) / (x1 - x0)
        else:
            p = 0.5 + 0.5 * (x - x1) / (x2 - x1)
        return min(1.0, max(0.0, p))


def _fit_log(x0: float, x1: float, x2: float) -> Callable[[float], float] | None:
    """Fit p = A + B*ln(x + C) through (x0, 0), (x1, 0.5), (x2, 1); mirrored if the curve bends
    the other way. Returns None if no such curve exists (the points are exactly linear)."""
    d1, d2 = x1 - x0, x2 - x0
    ratio = d1 / d2
    if abs(ratio - 0.5) < 1e-9:
        return None
    if ratio > 0.5:
        # reflect: x' = x0 + x2 - x, p' = 1 - p turns it into the concave case
        g = _fit_log(x0, x0 + x2 - x1, x2)
        return None if g is None else (lambda x: 1.0 - g(x0 + x2 - x))

    # concave case: find s = x0 + C > 0 with ln((d1+s)/s) / ln((d2+s)/s) = 0.5
    def f(s: float) -> float:
        return math.log((d1 + s) / s) / math.log((d2 + s) / s) - 0.5

    lo, hi = d2 * 1e-9, d2 * 1e9
    if f(lo) < 0 or f(hi) > 0:
        return None
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    s = math.sqrt(lo * hi)
    c = s - x0
    b = 1.0 / (math.log(x2 + c) - math.log(x0 + c))
    a = -b * math.log(x0 + c)
    return lambda x: a + b * math.log(x + c)


def level_from_position(x_levels: float, prev: int | None, hyst_steps: float) -> int:
    """Quantize 0..10 with hysteresis: stay at prev until the dial is hyst/2 past a boundary."""
    x_levels = min(max(x_levels, 0.0), LEVELS)
    raw = int(math.floor(x_levels + 0.5))
    raw = min(raw, LEVELS)
    if prev is None or raw == prev:
        return raw
    h = hyst_steps / 2
    if raw > prev:
        return max(prev, min(LEVELS, int(math.floor(x_levels - h + 0.5))))
    return min(prev, max(0, int(math.floor(x_levels + h + 0.5))))


def level_to_gain(level: int, min_db: float, max_db: float) -> float:
    """Perceptually even steps: level 1..10 spread evenly in dB, level 0 is mute."""
    if level <= 0:
        return 0.0
    db = min_db + (level - 1) * (max_db - min_db) / (LEVELS - 1)
    return 10 ** (db / 20)


class VolumeFilter:
    """Median filter + hysteresis + settle timer. add_sample returns (new_level, announce_level)."""

    def __init__(self, curve: VolumeCurve, cfg: VolumeConfig):
        self.curve = curve
        self.cfg = cfg
        self.samples: deque[float] = deque(maxlen=cfg.median_n)
        self.level: int | None = None
        self.announced: int | None = None
        self.changed_at = 0.0
        self.last_position = 0.0

    def add_sample(self, raw: float, t: float) -> tuple[int | None, int | None]:
        self.samples.append(raw)
        med = statistics.median(self.samples)
        self.last_position = self.curve.position(med)
        new_level = None
        lvl = level_from_position(self.last_position * LEVELS, self.level, self.cfg.hysteresis_steps)
        if self.level is None:
            # first reading after boot: apply it, don't announce it
            self.level = self.announced = lvl
            return lvl, None
        if lvl != self.level:
            self.level, self.changed_at = lvl, t
            new_level = lvl
        announce = None
        if self.level != self.announced and t - self.changed_at >= self.cfg.settle_ms / 1000:
            self.announced = announce = self.level
        return new_level, announce


class StepVolume:
    """Keyboard (+/-) or two-button (FR-46) volume with the same settle-then-announce rule."""

    def __init__(self, cfg: VolumeConfig):
        self.cfg = cfg
        self.level = cfg.initial_level
        self.announced = cfg.initial_level
        self.changed_at = 0.0

    def step(self, delta: int, t: float) -> int | None:
        new = min(LEVELS, max(0, self.level + delta))
        if new == self.level:
            return None
        self.level, self.changed_at = new, t
        return new

    def poll(self, t: float) -> int | None:
        if self.level != self.announced and t - self.changed_at >= self.cfg.settle_ms / 1000:
            self.announced = self.level
            return self.level
        return None


class Mcp3008Reader:
    """10-bit MCP3008 on SPI (0..1023). Needs SPI enabled (/dev/spidevB.D)."""

    def __init__(self, cfg: AdcConfig):
        if not 0 <= cfg.channel <= 7:
            raise ValueError(f"MCP3008 channel must be 0..7, got {cfg.channel}")
        import spidev

        self.spi = spidev.SpiDev()
        self.spi.open(cfg.spi_bus, cfg.spi_device)
        self.spi.max_speed_hz = cfg.max_speed_hz
        self.spi.mode = 0
        self.channel = cfg.channel
        self.read()  # raises if SPI is not available

    def read(self) -> int:
        # start bit, single-ended + channel, then clock out 10 result bits
        r = self.spi.xfer2([1, (8 + self.channel) << 4, 0])
        return ((r[1] & 3) << 8) | r[2]


class DialThread:
    """Samples the ADC at sample_hz on a background thread."""

    def __init__(self, reader: Mcp3008Reader, filt: VolumeFilter, sample_hz: float,
                 on_level: Callable[[int], None], on_announce: Callable[[int], None]):
        self.reader = reader
        self.filt = filt
        self.period = 1 / sample_hz
        self.on_level = on_level
        self.on_announce = on_announce
        self.last_raw: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="volume-dial", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.last_raw = raw = self.reader.read()
            except Exception as e:
                log.warning("ADC read failed: %s", e)
                time.sleep(1)
                continue
            new, announce = self.filt.add_sample(raw, time.monotonic())
            if new is not None:
                self.on_level(new)
            if announce is not None:
                self.on_announce(announce)
            time.sleep(self.period)
