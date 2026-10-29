"""Pydantic config models and YAML loaders (PRD rule 6: every tunable number lives in YAML).

Relative paths inside the configs are resolved against the repo root, which is the parent of
the config directory.

How it fits in
--------------
Nothing in the product is hard-coded: every port number, threshold, pin number and file path
lives in the three text files in the ``config/`` folder:

- ``server.yaml``   settings for the laptop (network, model, guidance rules, speech, paths)
- ``device.yaml``   settings for the Raspberry Pi (camera, streaming, audio, volume dial)
- ``controls.yaml`` which GPIO pin / keyboard key does what

This file describes the *shape* of those files (which keys exist, what type each value is,
what the default is, and which combinations are not allowed) and provides the ``load_*``
functions that read them. Every other part of the program receives one of these config
objects instead of opening the YAML files itself. If you add a new tunable, add it here AND in
the matching YAML file.

Words you will see
------------------
- YAML: a human-friendly text format for settings, using ``key: value`` lines and indentation
  for nesting. The ``yaml`` library turns it into Python dictionaries.
- Pydantic / BaseModel: a library that turns a class with typed fields into a validator. You
  hand it a dictionary; it checks every value has the right type, fills in defaults, and
  raises a clear error for anything wrong. Each ``class XConfig(_Model)`` below is one such
  validated section of a YAML file.
- ``extra="forbid"``: a typo in a YAML key (e.g. ``ws_prot``) is an error instead of being
  silently ignored.
- Literal["a", "b"]: the value must be exactly one of the listed words.
- ``x: int = 5``: a field named x that must be a whole number and is 5 if the YAML leaves it
  out. ``str | None`` means "text or nothing".
- Field(6, ge=0, le=10): default 6, must be >= 0 and <= 10.
- @model_validator(mode="after"): a check that runs after all fields are filled in, used for
  rules that involve several fields at once (e.g. "min must be below max").
- PrivateAttr: a field stored on the object but not part of the YAML (here: the repo root).
- @property: a method you read like an attribute (``cfg.root`` rather than ``cfg.root()``).
- Path: Python's object for file and folder paths; ``a / "b"`` joins path pieces.
- deep merge: combine two nested dictionaries so that inner keys are overridden one at a time
  rather than whole sections being replaced.
- GPIO: the general-purpose input/output pins on the Raspberry Pi, numbered "BCM" style.
- ADC: analog-to-digital converter (the MCP3002 chip) that reads the volume dial position.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class ConfigError(Exception):
    """Raised for any problem with a config file: missing, bad YAML, wrong value, etc.

    cli.py catches this and prints a friendly "config error: ..." message instead of a traceback.
    """
    pass


class _Model(BaseModel):
    """Base class for every config section. Forbids unknown keys so typos are caught early."""
    model_config = ConfigDict(extra="forbid")


class _Rooted(_Model):
    """A top-level config that knows the repo root so it can resolve relative paths.

    The YAML files say things like ``clips_dir: data/clips``. That path is meant relative to
    the project folder, not to wherever the user happened to run the command from, so each
    top-level config remembers where the project folder is.
    """

    # Set by ``_build`` right after loading. Defaults to the current folder until then.
    _root: Path = PrivateAttr(default_factory=Path.cwd)

    @property
    def root(self) -> Path:
        """The project folder (parent of the config/ directory)."""
        return self._root

    def path(self, p: str | Path) -> Path:
        """Turn a path from the config into an absolute one. Absolute paths are left alone."""
        p = Path(p)
        return p if p.is_absolute() else self._root / p


# --------------------------------------------------------------------------- server.yaml
# The classes in this section mirror the sections of config/server.yaml, one class per block.


class NetworkConfig(_Model):
    """``network:`` block of server.yaml: where the server listens."""
    # 0.0.0.0 = accept connections on every network interface of the laptop.
    host: str = "0.0.0.0"
    # Port numbers for the three channels (control, video frames, shopping list web page).
    ws_port: int = 8765
    udp_port: int = 8766
    http_port: int = 8767
    # Announce the server on the local network so the device can find it (see discovery.py).
    mdns: bool = True


class ModelConfig(_Model):
    """``model:`` block of server.yaml: which object detector to run and how."""
    # Which detection library: ultralytics (YOLO), OpenCV's darknet loader, or a scripted fake
    # detector used by tests and replays.
    backend: Literal["ultralytics", "darknet_opencv", "scripted"] = "ultralytics"
    # Model weights file. The trained grocery model lives in models/grocereye_a/best.pt.
    weights: str = "yolo11n.pt"
    # YAML file mapping class ids to spoken names.
    classes_file: str = "config/classes_coco.yaml"
    # "auto" picks the GPU if one is available, else the CPU.
    device: str = "auto"
    # Image size (pixels) the model is run at. Smaller = faster, less accurate.
    imgsz: int = 640
    # Detections below this confidence are dropped before guidance sees them.
    min_conf: float = 0.25
    # Use 16-bit floats on the GPU (faster) instead of 32-bit.
    half: bool = False
    # Only used by the darknet_opencv backend.
    darknet_cfg: str | None = None
    darknet_weights: str | None = None
    # Only used by the scripted backend: a JSON file of pre-recorded detections.
    scripted_file: str | None = None


class GuidanceConfig(_Model):
    """``guidance:`` block of server.yaml: every number the guidance engine uses.

    The guidance engine (slr/server/guidance.py) turns detections into spoken cues like
    "slightly left, eye level". These settings tune when it speaks and what it says.
    """
    # Minimum confidence to accept a detection in Find mode, Identify mode, and for the
    # "that is not the item you are looking for" warning.
    conf_find: float = 0.5
    conf_identify: float = 0.6
    conf_wrong_item: float = 0.7

    # An item counts as "seen" only if it appears in at least stable_k of the last stable_n
    # frames; this stops single-frame flickers from triggering speech.
    stable_k: int = 3
    stable_n: int = 5
    # Minimum box overlap (IoU) to treat a box as the same object as in the previous frame.
    iou_track_min: float = 0.3

    # Timing (seconds): when to say "not found yet, keep scanning", how often to repeat it,
    # how long an item may vanish before it counts as lost, and the gap between direction cues.
    not_found_prompt_s: float = 4.0
    not_found_repeat_s: float = 8.0
    lost_timeout_s: float = 1.5
    direction_repeat_s: float = 1.5

    # Left/right wording. Distance of the box center from the frame center (0..0.5):
    # within straight_tol = "straight ahead", within slight_tol = "slightly left/right",
    # beyond = "left/right".
    straight_tol: float = 0.08
    slight_tol: float = 0.25
    # Up/down wording by box center height (0 = top of frame): above high_shelf_max = "high
    # shelf", below low_shelf_min = "low shelf", between = "eye level".
    high_shelf_max: float = 0.33
    low_shelf_min: float = 0.66
    # Added to the center height to compensate for the camera tilting on the chest (README,
    # Calibration section).
    vertical_offset: float = 0.0
    # Small dead band so the wording does not flip back and forth at a boundary.
    direction_hysteresis: float = 0.02
    # Set true if the camera is mirrored and "left" / "right" come out swapped.
    flip_horizontal: bool = False

    # "Within reach": the box height (fraction of frame) at which the item is at arm's length,
    # how long it must stay that big, and how much smaller it may get before reach is lost.
    reach_h: float = 0.35
    reach_hold_s: float = 1.0
    reach_exit_ratio: float = 0.8

    # Identify mode: how many frames to grab, how many must agree on the same item, how long
    # to wait for them, and how much to favour boxes near the frame center.
    identify_burst: int = 5
    identify_min_votes: int = 3
    identify_timeout_s: float = 1.5
    identify_center_weight: float = 0.5

    # "Wrong item" warning when a confident non-target item sits in the center of view.
    wrong_item_enabled: bool = True
    wrong_item_center_tol: float = 0.15
    wrong_item_repeat_s: float = 5.0

    # Classes the detector knows that are not shopping items (the shopper's own hand). They are
    # never spoken as a found, wrong or identified item; later guidance will use them to steer.
    non_item_classes: list[str] = ["hand"]

    # Hand guidance: once the item is within reach, steer the shopper's hand onto it and say
    # "That's it." when the hand covers the item (needs a model with the hand class).
    hand_guidance_enabled: bool = True
    hand_class: str = "hand"
    hand_conf: float = 0.5          # hand box confidence floor
    hand_cue_tol: float = 0.08      # hand within this of the item center = aligned on that axis
    hand_touch_overlap: float = 0.3 # share of the hand box that must overlap the item = touching
    hand_touch_hold_s: float = 0.5  # ... held this long before "That's it."
    hand_repeat_s: float = 1.5      # repeat an unchanged hand cue this often
    hand_auto_check_off: bool = True  # "That's it." also checks the item off

    # Optional beeps that speed up as the item gets closer and pan left/right with it.
    tones_enabled: bool = True
    tone_freq_hz: int = 880
    tone_duration_ms: int = 60
    tone_interval_max_ms: int = 900
    tone_interval_min_ms: int = 120
    # Only send a new tone command if the interval or pan changed by at least this much.
    tone_min_change_ms: int = 60
    tone_min_change_pan: float = 0.1

    # Optional "getting closer" cue when the box grows by the given ratio.
    getting_closer_enabled: bool = False
    getting_closer_ratio: float = 1.3
    getting_closer_repeat_s: float = 3.0

    # Frame rate the server asks the device for while streaming.
    stream_max_fps: float = 10.0
    # Keep the camera stream on whenever the mode switch is ON, even while waiting in IDLE.
    # The detector still runs on every frame, but no search/guidance audio starts until the
    # user triggers Find or Identify. Useful for a live monitor and for camera setup.
    stream_in_idle: bool = False

    @model_validator(mode="after")
    def _check(self) -> GuidanceConfig:
        """Cross-field sanity checks; each raises a clear error for an impossible combination."""
        if not 1 <= self.stable_k <= self.stable_n:
            raise ValueError("need 1 <= stable_k <= stable_n")
        if not 0 <= self.straight_tol < self.slight_tol <= 0.5:
            raise ValueError("need 0 <= straight_tol < slight_tol <= 0.5")
        if not 0 < self.high_shelf_max < self.low_shelf_min < 1:
            raise ValueError("need 0 < high_shelf_max < low_shelf_min < 1")
        if self.identify_min_votes > self.identify_burst:
            raise ValueError("identify_min_votes cannot exceed identify_burst")
        if self.tone_interval_min_ms > self.tone_interval_max_ms:
            raise ValueError("tone_interval_min_ms must be <= tone_interval_max_ms")
        # getattr(self, name) reads a field by its name as a string, so one loop can check all
        # three confidence thresholds.
        for name in ("conf_find", "conf_identify", "conf_wrong_item", "hand_conf", "hand_touch_overlap"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.hand_cue_tol < 0.5:
            raise ValueError("need 0 <= hand_cue_tol < 0.5")
        return self


class TTSConfig(_Model):
    """``tts:`` block of server.yaml: text-to-speech, i.e. how the spoken clips are generated."""
    # Engines to try, in order, until one works. Piper is the good offline voice; "beep" is a
    # last-resort placeholder that just plays a tone.
    engines: list[Literal["piper", "espeak", "pyttsx3", "beep"]] = ["piper", "espeak", "pyttsx3"]
    # Where the Piper voice model file is (downloaded once by ``slr clips download-voice``).
    piper_voice: str = "models/voices/en_US-lessac-medium.onnx"
    piper_voice_url: str = ""
    # Speaking speed: >1 slower, <1 faster.
    length_scale: float = 1.0
    # Audio samples per second of the generated clips.
    sample_rate: int = 22050


class ServerPaths(_Model):
    """``paths:`` block of server.yaml: folders the server reads and writes (relative to repo)."""
    shopping_list: str = "data/shopping_list.yaml"
    clips_dir: str = "data/clips"
    logs_dir: str = "data/logs"
    captured_dir: str = "data/captured"


class DebugViewConfig(_Model):
    """``debug_view:`` block of server.yaml: the on-screen window showing the camera + boxes."""
    enabled: bool = True
    window_name: str = "SLR SLR debug"


class ServerConfig(_Rooted):
    """The whole of config/server.yaml, one attribute per block.

    ``XConfig()`` as a default means: if the YAML omits a whole block, use all its defaults.
    """
    network: NetworkConfig = NetworkConfig()
    model: ModelConfig = ModelConfig()
    guidance: GuidanceConfig = GuidanceConfig()
    tts: TTSConfig = TTSConfig()
    paths: ServerPaths = ServerPaths()
    debug_view: DebugViewConfig = DebugViewConfig()


# --------------------------------------------------------------------------- device.yaml
# These classes mirror config/device.yaml, the Raspberry Pi side.


class DeviceServerConfig(_Model):
    """``server:`` block of device.yaml: how the device reaches the laptop."""
    # Laptop address, or "auto" to find it with mDNS (discovery.py).
    host: str = "127.0.0.1"
    ws_port: int = 8765
    udp_port: int = 8766
    http_port: int = 8767
    # How long to search with mDNS before giving up.
    discovery_timeout_s: float = 5.0


class CameraConfig(_Model):
    """``camera:`` block of device.yaml."""
    # opencv = USB webcam (default on the Pi) or laptop webcam; picamera2 = optional CSI
    # Camera Module; file = a video or folder of images.
    backend: Literal["picamera2", "opencv", "file"] = "opencv"
    # Which webcam (opencv backend only): 0 = /dev/video0, and so on.
    index: int = 0
    # Video file / image folder (file backend only).
    file: str | None = None
    # Capture size in pixels.
    width: int = 640
    height: int = 480


class StreamingConfig(_Model):
    """``streaming:`` block of device.yaml: how frames are sent to the laptop."""
    max_fps: float = 10.0
    # How long to wait for the server's ack before counting a frame as lost.
    ack_timeout_ms: int = 300
    # Biggest UDP packet allowed. ``le=65000`` = must be at most 65000.
    max_datagram_bytes: int = Field(60000, le=65000)
    # JPEG compression quality (0..100) starts here and steps down whenever a frame is too big
    # for one packet, never below jpeg_quality_min.
    jpeg_quality_start: int = 75
    jpeg_quality_step: int = 10
    jpeg_quality_min: int = 40
    # Quality steps back up once jpeg_recover_frames frames in a row are under
    # jpeg_recover_ratio x the size limit.
    jpeg_recover_ratio: float = 0.6
    jpeg_recover_frames: int = 10
    # Say "video weak" after this many seconds without acks (FR-56 in PRD.pdf).
    video_weak_s: float = 3.0


class AudioConfig(_Model):
    """``audio:`` block of device.yaml: how clips are played."""
    # sounddevice = the Python audio library; aplay = the Linux command-line player; null = silent.
    backend: Literal["sounddevice", "aplay", "null"] = "sounddevice"
    # tlv320dac3100 = reset/configure the I2C codec at startup; none = a DAC that needs no setup.
    dac: Literal["tlv320dac3100", "none"] = "tlv320dac3100"
    # Output device name or number; None = system default.
    device: str | int | None = None
    sample_rate: int = 22050
    # Audio buffer size in samples; smaller = lower latency but more risk of crackle.
    blocksize: int = 512


class AdcConfig(_Model):
    """``volume.adc:`` block: how the MCP3002 analog-to-digital chip is wired to the Pi's SPI bus."""
    spi_bus: int = 0
    spi_device: int = 0         # chip select: 0 = CE0 (GPIO 8), 1 = CE1 (GPIO 7)
    # The MCP3002 has two input channels: CH0 (0) and CH1 (1).
    channel: int = Field(0, ge=0, le=1)
    # SPI clock speed.
    max_speed_hz: int = 1_000_000


class VolumeCalibration(_Model):
    """``volume.calibration:`` block: raw ADC readings at the dial's min / middle / max.

    Recorded by ``slr device calibrate-volume`` (FR-44). The three points let the software map
    any raw reading to a 0..100% dial position even if the dial is not linear. The defaults below
    are the values measured on the current dial; re-run calibration if the hardware changes.
    """
    raw_min: float = 399
    raw_mid: float = 450
    raw_max: float = 501
    # piecewise = straight lines between the three points; log = fit a logarithmic curve.
    curve: Literal["piecewise", "log"] = "piecewise"

    @model_validator(mode="after")
    def _monotonic(self) -> VolumeCalibration:
        """The three readings must go steadily up or steadily down, otherwise the wiring is wrong."""
        ascending = self.raw_min < self.raw_mid < self.raw_max
        descending = self.raw_min > self.raw_mid > self.raw_max
        if not (ascending or descending):
            raise ValueError("volume calibration points must be strictly monotonic (min, mid, max)")
        return self


class VolumeConfig(_Model):
    """``volume:`` block of device.yaml: the volume dial and how its readings become a level 0..10."""
    # adc = the real dial; buttons = fallback up/down buttons; keyboard = sim mode (+ / -).
    source: Literal["adc", "buttons", "keyboard"] = "adc"
    adc: AdcConfig = AdcConfig()
    calibration: VolumeCalibration = VolumeCalibration()
    # How often the dial is read, and how many readings are averaged (median) to remove noise.
    sample_hz: float = 20.0
    median_n: int = 5
    # The level only changes once the dial moves this far past the step boundary (prevents
    # flicker between two levels), and the new level is announced after settle_ms of no change.
    hysteresis_steps: float = 0.5
    settle_ms: int = 300
    # Loudness in decibels at level 1 and level 10 (level 0 is mute).
    min_db: float = -30.0
    max_db: float = 0.0
    # Level used before the dial has been read (0..10).
    initial_level: int = Field(6, ge=0, le=10)


class ConnectionConfig(_Model):
    """``connection:`` block of device.yaml: reconnect timing and heartbeats."""
    # After losing the server, wait this long before retrying, doubling each time up to the max.
    reconnect_initial_s: float = 0.5
    reconnect_max_s: float = 4.0
    # How often to send the "still alive" message.
    heartbeat_s: float = 2.0
    http_timeout_s: float = 5.0


class DevicePaths(_Model):
    """``paths:`` block of device.yaml: folders on the Pi (relative to repo)."""
    # Speech clips received from the server.
    clip_cache: str = "data/device_clips"
    # Clips the device can play even with no server ("starting up", "not connected", ...).
    system_clips: str = "device_assets/system_clips"
    # Small files that remember things across reboots (e.g. first run done).
    state_dir: str = "data/device_state"


class DeviceConfig(_Rooted):
    """The whole of config/device.yaml, one attribute per block."""
    device_id: str = "slr-01"
    server: DeviceServerConfig = DeviceServerConfig()
    camera: CameraConfig = CameraConfig()
    streaming: StreamingConfig = StreamingConfig()
    audio: AudioConfig = AudioConfig()
    volume: VolumeConfig = VolumeConfig()
    connection: ConnectionConfig = ConnectionConfig()
    # Command run to power off the Pi (e.g. ["sudo", "shutdown", "-h", "now"]); empty = do nothing.
    shutdown_command: list[str] = []
    paths: DevicePaths = DevicePaths()


# --------------------------------------------------------------------------- controls.yaml
# These classes mirror config/controls.yaml: which physical control triggers which event.

# Every event name the rest of the program understands. Buttons, the mode switch, the volume
# dial and sim-mode keyboard keys all produce one of these.
EVENTS = (
    "next_item", "prev_item", "enter", "identify", "mode_on", "mode_off",
    "volume_set", "help", "status", "shutdown",
)


class ButtonActions(_Model):
    """What one button does on a short press, a long press, and a long hold (each optional)."""
    short: str | None = None
    long: str | None = None
    hold: str | None = None


class ModeSwitchConfig(_Model):
    """``mode_switch:`` block: the three-position switch read from two GPIO pins.

    Each pin reads 0 or 1, giving a two-character key like "01"; ``positions`` says which key
    means off / on / toggle.
    """
    pin_a: int = 5
    pin_b: int = 6
    # Flip if the switch reads backwards.
    invert: bool = False
    # Ignore changes shorter than this (mechanical contacts bounce when they move).
    debounce_ms: int = 80
    positions: dict[str, Literal["off", "on", "toggle"]] = {"00": "off", "11": "on", "01": "toggle"}
    # True if the "toggle" position stays put; False if it springs back.
    toggle_latching: bool = False


class IdentifyTriggers(_Model):
    """Which actions start Identify mode ("what am I holding?")."""
    mode_toggle: bool = True
    shutter_long_press: bool = True


class VolumeButtons(_Model):
    """GPIO pins of the fallback volume up/down buttons (used if the dial cannot be read)."""
    up: int = 23
    down: int = 24


class ControlsConfig(_Rooted):
    """The whole of config/controls.yaml."""
    # gpio = real Pi pins; keyboard = sim mode; auto = gpio if available, else keyboard.
    backend: Literal["auto", "gpio", "keyboard"] = "auto"
    # Button name -> GPIO pin number (BCM numbering).
    buttons: dict[str, int] = {"shutter": 17, "advance": 27, "rewind": 22}
    # Buttons connect the pin to ground, so the pin is held high ("pulled up") when not pressed.
    button_pull_up: bool = True
    button_debounce_ms: int = 50
    # How long a press must last to count as "long" or "hold".
    long_press_s: float = 0.8
    hold_s: float = 2.0
    # Hold the shutdown_combo buttons together this long to power off safely.
    shutdown_hold_s: float = 5.0
    # How many times per second the pins are read.
    poll_hz: float = 100.0
    # Button name -> what its presses do.
    actions: dict[str, ButtonActions] = {}
    shutdown_combo: list[str] = ["shutter", "advance"]
    mode_switch: ModeSwitchConfig = ModeSwitchConfig()
    identify_triggers: IdentifyTriggers = IdentifyTriggers()
    volume_buttons: VolumeButtons = VolumeButtons()
    # Sim mode: event name -> list of keyboard keys that trigger it.
    keyboard: dict[str, list[str]] = {}

    @model_validator(mode="after")
    def _check(self) -> ControlsConfig:
        """Make sure every name mentioned in the file refers to something that exists."""
        # Set union (|): the known events plus the two volume button events.
        allowed = set(EVENTS) | {"volume_up", "volume_down"}
        for button, acts in self.actions.items():
            if button not in self.buttons:
                raise ValueError(f"actions refers to unknown button {button!r}")
            for ev in (acts.short, acts.long, acts.hold):
                if ev is not None and ev not in allowed:
                    raise ValueError(f"unknown event {ev!r} for button {button!r}")
        for ev in self.keyboard:
            if ev not in allowed:
                raise ValueError(f"unknown keyboard event {ev!r}")
        for b in self.shutdown_combo:
            if b not in self.buttons:
                raise ValueError(f"shutdown_combo refers to unknown button {b!r}")
        # Position keys must be exactly two characters, each "0" or "1".
        for key in self.mode_switch.positions:
            if len(key) != 2 or set(key) - {"0", "1"}:
                raise ValueError(f"mode_switch position key {key!r} must look like '01'")
        return self


# --------------------------------------------------------------------------- loaders
# The functions below read the YAML files from disk and build the objects above.


def default_config_dir() -> Path:
    """Where to look for the YAML files: the SLR_CONFIG_DIR environment variable, else ./config."""
    env = os.environ.get("SLR_CONFIG_DIR")
    return Path(env) if env else Path.cwd() / "config"


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file into a dictionary. Missing file or bad syntax becomes a ConfigError."""
    try:
        # ``with open(...)`` opens the file and closes it automatically at the end of the block.
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    # An empty file loads as None; treat it as an empty dictionary.
    return data or {}


def deep_merge(base: dict, overlay: dict) -> dict:
    """Return ``base`` with ``overlay`` applied on top, merging nested dictionaries key by key.

    Example: base ``{"camera": {"backend": "opencv", "width": 640}}`` plus overlay
    ``{"camera": {"backend": "file"}}`` gives ``{"camera": {"backend": "file", "width": 640}}``.
    Used for the ``sim:`` overrides in device.yaml. Neither input is modified.
    """
    out = dict(base)
    for k, v in overlay.items():
        # If both sides have a dictionary here, recurse; otherwise the overlay value wins.
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(model: type[_Rooted], data: dict, path: Path):
    """Validate ``data`` with the given config class and stamp the repo root onto the result.

    ``path`` is the YAML file, used in the error message and to find the repo root
    (two levels up: config/server.yaml -> config -> repo).
    """
    try:
        cfg = model.model_validate(data)
    except Exception as e:  # pydantic.ValidationError, kept generic for a clean message
        raise ConfigError(f"invalid config in {path}:\n{e}") from e
    cfg._root = path.parent.parent.resolve()
    return cfg


def load_server_config(config_dir: Path | None = None) -> ServerConfig:
    """Load config/server.yaml. ``config_dir`` overrides the folder; None = default_config_dir()."""
    path = (config_dir or default_config_dir()) / "server.yaml"
    return _build(ServerConfig, read_yaml(path), path)


def load_device_config(config_dir: Path | None = None, sim: bool = False) -> DeviceConfig:
    """Load config/device.yaml. With ``sim=True`` the file's ``sim:`` block overrides the rest.

    The ``sim:`` block is always removed from the data (pop) because DeviceConfig does not
    have a ``sim`` field and would reject it as an unknown key.
    """
    path = (config_dir or default_config_dir()) / "device.yaml"
    data = read_yaml(path)
    overlay = data.pop("sim", {}) or {}
    if sim:
        data = deep_merge(data, overlay)
    return _build(DeviceConfig, data, path)


def load_controls_config(config_dir: Path | None = None) -> ControlsConfig:
    """Load config/controls.yaml."""
    path = (config_dir or default_config_dir()) / "controls.yaml"
    return _build(ControlsConfig, read_yaml(path), path)


def item_classes(classes: dict[str, str], non_item_classes: list[str]) -> dict[str, str]:
    """The classes a shopper may put on the list: ``classes`` minus the non-item ones (the hand).

    The detector still reports non-item classes; they just cannot be shopping-list items.
    """
    skip = set(non_item_classes)
    return {k: v for k, v in classes.items() if k not in skip}


def load_classes(path: Path) -> dict[str, str]:
    """class_id -> spoken name (FR-62).

    Reads a classes file such as config/classes.yaml, which looks like::

        classes:
          pasta: pasta
          tomato_sauce: tomato sauce

    The keys are the ids the model outputs; the values are what the voice says. FR-62 is the
    requirement number in PRD.pdf.
    """
    data = read_yaml(path)
    classes = data.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ConfigError(f"{path} must contain a non-empty 'classes:' mapping")
    # Force both sides to strings (YAML could parse a bare number as an int).
    return {str(k): str(v) for k, v in classes.items()}
