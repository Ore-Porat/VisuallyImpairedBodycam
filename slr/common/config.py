"""Pydantic config models and YAML loaders (PRD rule 6: every tunable number lives in YAML).

Relative paths inside the configs are resolved against the repo root, which is the parent of
the config directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


class ConfigError(Exception):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Rooted(_Model):
    """A top-level config that knows the repo root so it can resolve relative paths."""

    _root: Path = PrivateAttr(default_factory=Path.cwd)

    @property
    def root(self) -> Path:
        return self._root

    def path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self._root / p


# --------------------------------------------------------------------------- server.yaml


class NetworkConfig(_Model):
    host: str = "0.0.0.0"
    ws_port: int = 8765
    udp_port: int = 8766
    http_port: int = 8767
    mdns: bool = True


class ModelConfig(_Model):
    backend: Literal["ultralytics", "darknet_opencv", "scripted"] = "ultralytics"
    weights: str = "yolo11n.pt"
    classes_file: str = "config/classes_coco.yaml"
    device: str = "auto"
    imgsz: int = 640
    min_conf: float = 0.25
    half: bool = False
    darknet_cfg: str | None = None
    darknet_weights: str | None = None
    scripted_file: str | None = None


class GuidanceConfig(_Model):
    conf_find: float = 0.5
    conf_identify: float = 0.6
    conf_wrong_item: float = 0.7

    stable_k: int = 3
    stable_n: int = 5
    iou_track_min: float = 0.3

    not_found_prompt_s: float = 4.0
    not_found_repeat_s: float = 8.0
    lost_timeout_s: float = 1.5
    direction_repeat_s: float = 1.5

    straight_tol: float = 0.08
    slight_tol: float = 0.25
    high_shelf_max: float = 0.33
    low_shelf_min: float = 0.66
    vertical_offset: float = 0.0
    direction_hysteresis: float = 0.02
    flip_horizontal: bool = False

    reach_h: float = 0.35
    reach_hold_s: float = 1.0
    reach_exit_ratio: float = 0.8

    identify_burst: int = 5
    identify_min_votes: int = 3
    identify_timeout_s: float = 1.5
    identify_center_weight: float = 0.5

    wrong_item_enabled: bool = True
    wrong_item_center_tol: float = 0.15
    wrong_item_repeat_s: float = 5.0

    tones_enabled: bool = True
    tone_freq_hz: int = 880
    tone_duration_ms: int = 60
    tone_interval_max_ms: int = 900
    tone_interval_min_ms: int = 120
    tone_min_change_ms: int = 60
    tone_min_change_pan: float = 0.1

    getting_closer_enabled: bool = False
    getting_closer_ratio: float = 1.3
    getting_closer_repeat_s: float = 3.0

    stream_max_fps: float = 10.0

    @model_validator(mode="after")
    def _check(self) -> GuidanceConfig:
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
        for name in ("conf_find", "conf_identify", "conf_wrong_item"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        return self


class TTSConfig(_Model):
    engines: list[Literal["piper", "espeak", "pyttsx3", "beep"]] = ["piper", "espeak", "pyttsx3"]
    piper_voice: str = "models/voices/en_US-lessac-medium.onnx"
    piper_voice_url: str = ""
    length_scale: float = 1.0
    sample_rate: int = 22050


class ServerPaths(_Model):
    shopping_list: str = "data/shopping_list.yaml"
    clips_dir: str = "data/clips"
    logs_dir: str = "data/logs"
    captured_dir: str = "data/captured"


class DebugViewConfig(_Model):
    enabled: bool = True
    window_name: str = "SLR SLR debug"


class ServerConfig(_Rooted):
    network: NetworkConfig = NetworkConfig()
    model: ModelConfig = ModelConfig()
    guidance: GuidanceConfig = GuidanceConfig()
    tts: TTSConfig = TTSConfig()
    paths: ServerPaths = ServerPaths()
    debug_view: DebugViewConfig = DebugViewConfig()


# --------------------------------------------------------------------------- device.yaml


class DeviceServerConfig(_Model):
    host: str = "127.0.0.1"
    ws_port: int = 8765
    udp_port: int = 8766
    http_port: int = 8767
    discovery_timeout_s: float = 5.0


class CameraConfig(_Model):
    backend: Literal["picamera2", "opencv", "file"] = "picamera2"
    index: int = 0
    file: str | None = None
    width: int = 640
    height: int = 480


class StreamingConfig(_Model):
    max_fps: float = 10.0
    ack_timeout_ms: int = 300
    max_datagram_bytes: int = Field(60000, le=65000)
    jpeg_quality_start: int = 75
    jpeg_quality_step: int = 10
    jpeg_quality_min: int = 40
    jpeg_recover_ratio: float = 0.6
    jpeg_recover_frames: int = 10
    video_weak_s: float = 3.0


class AudioConfig(_Model):
    backend: Literal["sounddevice", "aplay", "null"] = "sounddevice"
    device: str | int | None = None
    sample_rate: int = 22050
    blocksize: int = 512


class AdcConfig(_Model):
    i2c_address: int = 0x48
    channel: int = 0
    gain: float = 2


class VolumeCalibration(_Model):
    raw_min: float = 23924
    raw_mid: float = 27397
    raw_max: float = 29332
    curve: Literal["piecewise", "log"] = "piecewise"

    @model_validator(mode="after")
    def _monotonic(self) -> VolumeCalibration:
        ascending = self.raw_min < self.raw_mid < self.raw_max
        descending = self.raw_min > self.raw_mid > self.raw_max
        if not (ascending or descending):
            raise ValueError("volume calibration points must be strictly monotonic (min, mid, max)")
        return self


class VolumeConfig(_Model):
    source: Literal["adc", "buttons", "keyboard"] = "adc"
    adc: AdcConfig = AdcConfig()
    calibration: VolumeCalibration = VolumeCalibration()
    sample_hz: float = 20.0
    median_n: int = 5
    hysteresis_steps: float = 0.5
    settle_ms: int = 300
    min_db: float = -30.0
    max_db: float = 0.0
    initial_level: int = Field(6, ge=0, le=10)


class ConnectionConfig(_Model):
    reconnect_initial_s: float = 0.5
    reconnect_max_s: float = 4.0
    heartbeat_s: float = 2.0
    http_timeout_s: float = 5.0


class DevicePaths(_Model):
    clip_cache: str = "data/device_clips"
    system_clips: str = "device_assets/system_clips"
    state_dir: str = "data/device_state"


class DeviceConfig(_Rooted):
    device_id: str = "slr-01"
    server: DeviceServerConfig = DeviceServerConfig()
    camera: CameraConfig = CameraConfig()
    streaming: StreamingConfig = StreamingConfig()
    audio: AudioConfig = AudioConfig()
    volume: VolumeConfig = VolumeConfig()
    connection: ConnectionConfig = ConnectionConfig()
    shutdown_command: list[str] = []
    paths: DevicePaths = DevicePaths()


# --------------------------------------------------------------------------- controls.yaml

EVENTS = (
    "next_item", "prev_item", "enter", "identify", "mode_on", "mode_off",
    "volume_set", "help", "status", "shutdown",
)


class ButtonActions(_Model):
    short: str | None = None
    long: str | None = None
    hold: str | None = None


class ModeSwitchConfig(_Model):
    pin_a: int = 5
    pin_b: int = 6
    invert: bool = False
    debounce_ms: int = 80
    positions: dict[str, Literal["off", "on", "toggle"]] = {"00": "off", "11": "on", "01": "toggle"}
    toggle_latching: bool = False


class IdentifyTriggers(_Model):
    mode_toggle: bool = True
    shutter_long_press: bool = True


class VolumeButtons(_Model):
    up: int = 23
    down: int = 24


class ControlsConfig(_Rooted):
    backend: Literal["auto", "gpio", "keyboard"] = "auto"
    buttons: dict[str, int] = {"shutter": 17, "advance": 27, "rewind": 22}
    button_pull_up: bool = True
    button_debounce_ms: int = 50
    long_press_s: float = 0.8
    hold_s: float = 2.0
    shutdown_hold_s: float = 5.0
    poll_hz: float = 100.0
    actions: dict[str, ButtonActions] = {}
    shutdown_combo: list[str] = ["shutter", "advance"]
    mode_switch: ModeSwitchConfig = ModeSwitchConfig()
    identify_triggers: IdentifyTriggers = IdentifyTriggers()
    volume_buttons: VolumeButtons = VolumeButtons()
    keyboard: dict[str, list[str]] = {}

    @model_validator(mode="after")
    def _check(self) -> ControlsConfig:
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
        for key in self.mode_switch.positions:
            if len(key) != 2 or set(key) - {"0", "1"}:
                raise ValueError(f"mode_switch position key {key!r} must look like '01'")
        return self


# --------------------------------------------------------------------------- loaders


def default_config_dir() -> Path:
    env = os.environ.get("SLR_CONFIG_DIR")
    return Path(env) if env else Path.cwd() / "config"


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    return data or {}


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(model: type[_Rooted], data: dict, path: Path):
    try:
        cfg = model.model_validate(data)
    except Exception as e:  # pydantic.ValidationError, kept generic for a clean message
        raise ConfigError(f"invalid config in {path}:\n{e}") from e
    cfg._root = path.parent.parent.resolve()
    return cfg


def load_server_config(config_dir: Path | None = None) -> ServerConfig:
    path = (config_dir or default_config_dir()) / "server.yaml"
    return _build(ServerConfig, read_yaml(path), path)


def load_device_config(config_dir: Path | None = None, sim: bool = False) -> DeviceConfig:
    path = (config_dir or default_config_dir()) / "device.yaml"
    data = read_yaml(path)
    overlay = data.pop("sim", {}) or {}
    if sim:
        data = deep_merge(data, overlay)
    return _build(DeviceConfig, data, path)


def load_controls_config(config_dir: Path | None = None) -> ControlsConfig:
    path = (config_dir or default_config_dir()) / "controls.yaml"
    return _build(ControlsConfig, read_yaml(path), path)


def load_classes(path: Path) -> dict[str, str]:
    """class_id -> spoken name (FR-62)."""
    data = read_yaml(path)
    classes = data.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ConfigError(f"{path} must contain a non-empty 'classes:' mapping")
    return {str(k): str(v) for k, v in classes.items()}
