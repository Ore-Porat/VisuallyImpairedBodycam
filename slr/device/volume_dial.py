"""Volume dial: MCP3002 reading, calibration curve, filtering, hysteresis (FR-44, FR-45, FR-46).

WHAT THIS FILE DOES
-------------------
Turns the position of a physical knob into a volume level from 0 (mute) to 10, and from that
level into a loudness multiplier for the audio player. (FR-44..46 are requirement numbers from
the product spec, PRD.pdf: calibration, steady readings, and the fallback buttons.)

The dial is a variable resistor in a divider read by an MCP3002 (SPI). ``VolumeCurve`` maps raw
counts to a 0..1 dial position through the three calibration points; ``VolumeFilter`` turns noisy
samples into a steady 0..10 level. Both are pure and unit tested.

HOW IT FITS IN
--------------
``slr/device/main.py`` builds ``Mcp3002Reader`` + ``VolumeCurve`` + ``VolumeFilter`` and runs them
on a ``DialThread``. The thread reports two things back: "apply this level now" (so the volume
changes instantly) and, once the knob has stopped moving for a moment, "announce this level"
(so the user hears "Volume 7" once, not ten times while turning). If there is no dial,
``StepVolume`` offers the same behaviour for +/- keys or two buttons.

WORDS YOU WILL SEE
------------------
- ADC: analogue-to-digital converter. The Pi cannot read a voltage directly; the MCP3002 chip
  measures one and reports it as a number from 0 to 1023 ("raw counts", 10 bits). The MCP3002
  has two input channels (CH0 and CH1).
- SPI: a simple wiring scheme (4 wires) the Pi uses to talk to chips like the MCP3002.
- voltage divider: a fixed resistor and the dial in series between 3.3 V and ground. The
  voltage at the midpoint moves as the knob turns; that is what the ADC measures.
- calibration: measuring the raw counts at the knob's minimum, middle and maximum once, so
  the code can convert any reading to a position between 0 and 1 regardless of the exact parts.
- median filter: take the last few readings and use the middle value. One wild reading
  (electrical noise) cannot move the median, unlike an average.
- hysteresis: a "sticky" boundary. Once at level 5, the reading must go clearly past the
  5/6 border to become 6, so a knob sitting right on a border does not flicker 5-6-5-6.
- settle time: how long the level must stay unchanged before it is announced.
- dB (decibel): a logarithmic loudness unit; equal dB steps sound like equal steps to the ear.
- gain: the multiplier applied to the audio samples (1.0 = full, 0.0 = silence).
- thread: a second line of execution. The dial is sampled 20 times a second on its own thread.
- Callable: type hint meaning "a function".
- deque: a list with a maximum length that automatically drops the oldest item.
"""

# Allows modern type hints like ``int | None`` on older Python versions.
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

# Top volume level. Levels run 0..10, where 0 is mute.
LEVELS = 10

# The MCP3002 is a 10-bit ADC, so a reading runs 0..1023. A reading this close to either end
# usually means the input is open or shorted, not that the knob is at a real position.
ADC_MAX_COUNTS = 1023
ADC_RAIL_MARGIN = 3


def at_rail(raw: int) -> bool:
    """True if ``raw`` is close enough to 0 or 1023 to look like a broken input."""
    return raw <= ADC_RAIL_MARGIN or raw >= ADC_MAX_COUNTS - ADC_RAIL_MARGIN


def check_calibration_points(points: dict[str, int]) -> None:
    """Validate measured min / mid / max readings before they are saved to device.yaml.

    A real potentiometer may legitimately touch both rails (0 and 1023). A divider should touch
    neither. But when only ONE end is at a rail and the other is not, the wiper is probably open
    or shorted at that end; saving that point would squash most of the dial into a tiny range.
    Raises ``ValueError`` with a user-facing message so ``calibrate_volume`` can abort safely.
    """
    keys = ("raw_min", "raw_mid", "raw_max")
    low_rail = at_rail(points["raw_min"])
    high_rail = at_rail(points["raw_max"])
    if low_rail != high_rail:
        bad = "raw_min" if low_rail else "raw_max"
        raise ValueError(
            f"{bad} = {points[bad]} is at the ADC rail while the other end is not; "
            "the dial wiper is probably open or shorted. Check the divider and wiring")
    vals = [points[key] for key in keys]
    if not (vals[0] < vals[1] < vals[2] or vals[0] > vals[1] > vals[2]):
        raise ValueError("readings are not monotonic; check the wiring")


class VolumeCurve:
    """Converts a raw ADC reading into a knob position from 0.0 (min) to 1.0 (max).

    Uses the three calibration points (raw_min, raw_mid, raw_max) from config/device.yaml.
    Default: two straight lines (min->mid and mid->max). Optional: a fitted logarithmic
    curve, which matches the dial's true behaviour more closely.
    """

    def __init__(self, cal: VolumeCalibration):
        """``cal`` holds raw_min, raw_mid, raw_max and the curve type ("piecewise" or "log")."""
        self.cal = cal
        self.sign = 1.0 if cal.raw_max > cal.raw_min else -1.0  # dial wired backwards is fine
        # Multiplying by ``sign`` makes the three points increase left to right either way.
        x0, x1, x2 = (self.sign * v for v in (cal.raw_min, cal.raw_mid, cal.raw_max))
        self.x = (x0, x1, x2)
        self._log = None
        if cal.curve == "log":
            self._log = _fit_log(x0, x1, x2)
            if self._log is None:
                log.warning("log curve does not fit the calibration points; using piecewise")

    def position(self, raw: float) -> float:
        """Map one raw reading to a position in 0..1 (clamped, so out-of-range readings are safe)."""
        x = self.sign * raw
        x0, x1, x2 = self.x
        if self._log is not None:
            p = self._log(min(max(x, x0), x2))
        elif x <= x1:
            # Lower half: straight line from (x0, 0.0) to (x1, 0.5).
            p = 0.5 * (x - x0) / (x1 - x0)
        else:
            # Upper half: straight line from (x1, 0.5) to (x2, 1.0).
            p = 0.5 + 0.5 * (x - x1) / (x2 - x1)
        return min(1.0, max(0.0, p))


def _fit_log(x0: float, x1: float, x2: float) -> Callable[[float], float] | None:
    """Fit p = A + B*ln(x + C) through (x0, 0), (x1, 0.5), (x2, 1); mirrored if the curve bends
    the other way. Returns None if no such curve exists (the points are exactly linear).

    Returns a FUNCTION (position = f(raw)) rather than numbers, so VolumeCurve can just call it.
    The maths: find the one horizontal shift C that makes a logarithm pass through all three
    points, by bisection (repeatedly halving a search range, 200 times, in log space).
    """
    d1, d2 = x1 - x0, x2 - x0
    ratio = d1 / d2
    # Middle point exactly halfway: the points are on a straight line, no log curve fits.
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
    # Bisection on a logarithmic scale: ``sqrt(lo*hi)`` is the geometric midpoint.
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    s = math.sqrt(lo * hi)
    c = s - x0
    # With C known, B and A follow directly from the end points.
    b = 1.0 / (math.log(x2 + c) - math.log(x0 + c))
    a = -b * math.log(x0 + c)
    return lambda x: a + b * math.log(x + c)


def level_from_position(x_levels: float, prev: int | None, hyst_steps: float) -> int:
    """Quantize 0..10 with hysteresis: stay at prev until the dial is hyst/2 past a boundary.

    - x_levels: the knob position scaled to 0..10 (a float, e.g. 6.4).
    - prev: the level we are currently at, or None on the very first reading.
    - hyst_steps: the width of the sticky zone around each boundary, in level units.
    Returns the new whole-number level.
    """
    x_levels = min(max(x_levels, 0.0), LEVELS)
    # Plain rounding to the nearest level (floor(x + 0.5) rounds half up).
    raw = int(math.floor(x_levels + 0.5))
    raw = min(raw, LEVELS)
    if prev is None or raw == prev:
        return raw
    h = hyst_steps / 2
    # Moving up: require the position to be ``h`` beyond the boundary before accepting it.
    if raw > prev:
        return max(prev, min(LEVELS, int(math.floor(x_levels - h + 0.5))))
    # Moving down: same idea in the other direction.
    return min(prev, max(0, int(math.floor(x_levels + h + 0.5))))


def level_to_gain(level: int, min_db: float, max_db: float) -> float:
    """Perceptually even steps: level 1..10 spread evenly in dB, level 0 is mute.

    Returns the multiplier for the audio samples. Example with min_db=-30, max_db=0: level 1 is
    30 dB quieter than level 10 (a multiplier of about 0.03), and every step is about 3.3 dB.
    """
    if level <= 0:
        return 0.0
    db = min_db + (level - 1) * (max_db - min_db) / (LEVELS - 1)
    # Standard conversion from decibels to an amplitude multiplier.
    return 10 ** (db / 20)


class VolumeFilter:
    """Median filter + hysteresis + settle timer. add_sample returns (new_level, announce_level).

    Feed it raw ADC readings as they arrive. It answers with two optional values:
    - new_level: the level just changed to this (apply it to the audio right away), or None.
    - announce_level: the knob has rested on this level for ``settle_ms`` and it has not been
      announced yet (say "Volume N"), or None.
    """

    def __init__(self, curve: VolumeCurve, cfg: VolumeConfig):
        """``curve`` converts raw to position; ``cfg`` is the volume section of device.yaml."""
        self.curve = curve
        self.cfg = cfg
        # A deque with maxlen keeps only the last ``median_n`` readings automatically.
        self.samples: deque[float] = deque(maxlen=cfg.median_n)
        self.level: int | None = None       # current level (None until the first reading)
        self.announced: int | None = None   # last level spoken aloud
        self.changed_at = 0.0               # when ``level`` last changed
        self.last_position = 0.0            # latest filtered knob position 0..1 (for display)

    def add_sample(self, raw: float, t: float) -> tuple[int | None, int | None]:
        """Process one raw reading taken at time ``t`` (seconds). See the class docstring."""
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
        # Announce only once the level has been steady for settle_ms and differs from the last
        # announcement.
        if self.level != self.announced and t - self.changed_at >= self.cfg.settle_ms / 1000:
            self.announced = announce = self.level
        return new_level, announce


class StepVolume:
    """Keyboard (+/-) or two-button (FR-46) volume with the same settle-then-announce rule."""

    def __init__(self, cfg: VolumeConfig):
        """Start at the configured initial level."""
        self.cfg = cfg
        self.level = cfg.initial_level
        self.announced = cfg.initial_level
        self.changed_at = 0.0

    def step(self, delta: int, t: float) -> int | None:
        """Move the level by ``delta`` (+1 or -1) at time ``t``. Returns the new level, or None
        if already at the end of the range."""
        new = min(LEVELS, max(0, self.level + delta))
        if new == self.level:
            return None
        self.level, self.changed_at = new, t
        return new

    def poll(self, t: float) -> int | None:
        """Called regularly: returns the level to announce once it has settled, else None."""
        if self.level != self.announced and t - self.changed_at >= self.cfg.settle_ms / 1000:
            self.announced = self.level
            return self.level
        return None


class Mcp3002Reader:
    """10-bit MCP3002 on SPI (0..1023). Needs SPI enabled (/dev/spidevB.D).

    Talks to the chip through the ``spidev`` library. The constructor does one test read so a
    missing or mis-wired chip fails loudly at start-up (main.py then falls back to buttons).
    The MCP3002 uses VDD as its voltage reference, so VDD must be 3.3 V on the Pi.
    """

    def __init__(self, cfg: AdcConfig):
        """``cfg`` gives the SPI bus/device numbers, the channel (0 or 1) and the clock speed."""
        if not 0 <= cfg.channel <= 1:
            raise ValueError(f"MCP3002 channel must be 0 or 1, got {cfg.channel}")
        # Imported here so machines without spidev can still import this module.
        import spidev

        self.spi = spidev.SpiDev()
        self.spi.open(cfg.spi_bus, cfg.spi_device)
        self.spi.max_speed_hz = cfg.max_speed_hz
        self.spi.mode = 0
        self.channel = cfg.channel
        # The most recent raw bytes, shown by `slr device test-adc` and `test-controls`.
        self.last_rx: tuple[int, int] | None = None
        # One test read: raises if SPI is not available. A first reading at a rail usually means
        # the divider is disconnected or VDD is wrong, so say so instead of silently using it.
        first = self.read()
        if at_rail(first):
            log.warning(
                "MCP3002 first reading is %d (bytes %s); the input may be open or shorted. "
                "Check the dial divider, CH%d wiring and that VDD is 3.3 V",
                first, self.rx_text(), self.channel)

    def rx_text(self) -> str:
        """Format the last SPI bytes for a log or terminal line."""
        if self.last_rx is None:
            return "-- --"
        return f"{self.last_rx[0]:02x} {self.last_rx[1]:02x}"

    def read(self) -> int:
        """Take one measurement and return it as 0..1023.

        The MCP3002 protocol: send two bytes (start + single-ended + channel + MSB-first, then
        a dummy byte) and receive two bytes back; the 10-bit answer is spread over the low two
        bits of the first byte and the whole second byte. ``<<`` and ``|`` are bit operations
        that reassemble it.
        """
        # 0x68 = single-ended CH0, 0x78 = single-ended CH1; both ask for MSB-first output.
        cmd = 0x68 | (self.channel << 4)
        r = self.spi.xfer2([cmd, 0x00])
        self.last_rx = (r[0], r[1])
        return ((r[0] & 0x03) << 8) | r[1]

    def close(self) -> None:
        """Release the SPI device."""
        self.spi.close()


class DialThread:
    """Samples the ADC at sample_hz on a background thread.

    Each sample goes through the ``VolumeFilter``; the two callbacks report level changes and
    settled announcements to main.py.
    """

    def __init__(self, reader: Mcp3002Reader, filt: VolumeFilter, sample_hz: float,
                 on_level: Callable[[int], None], on_announce: Callable[[int], None]):
        """
        - reader: the ADC to read.
        - filt: the VolumeFilter to push readings through.
        - sample_hz: readings per second.
        - on_level: called with the new level whenever it changes.
        - on_announce: called with a level once it has settled and should be spoken.
        """
        self.reader = reader
        self.filt = filt
        self.period = 1 / sample_hz
        self.on_level = on_level
        self.on_announce = on_announce
        self.last_raw: int | None = None   # most recent raw reading (shown by test-controls)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="volume-dial", daemon=True)

    def start(self) -> None:
        """Start sampling."""
        self._thread.start()

    def stop(self) -> None:
        """Ask the sampling loop to exit."""
        self._stop.set()

    def _run(self) -> None:
        """The sampling loop: read, filter, report, sleep, repeat until stopped."""
        while not self._stop.is_set():
            try:
                self.last_raw = raw = self.reader.read()
            except Exception as e:
                # A flaky SPI link should not crash the device; wait a second and try again.
                log.warning("ADC read failed: %s", e)
                time.sleep(1)
                continue
            new, announce = self.filt.add_sample(raw, time.monotonic())
            if new is not None:
                self.on_level(new)
            if announce is not None:
                self.on_announce(announce)
            time.sleep(self.period)
