# SLR SLR: Single Lens Reflex Sightless Label Reader

A vintage SLR film camera, worn on a neck strap, retrofitted with a Raspberry Pi Zero 2 W and a
USB camera. It helps blind and low-vision shoppers find the right item on a shelf, hands free,
using only audio:

- **Find mode** guides you to an item on your shopping list: *"Pasta found. Slightly left. Eye level."* ... *"Hand up and left."* ... *"Reach forward."* ... *"That's it. Pasta checked off."* Hand cues start as soon as both your hand and the item are visible (the model must have the `hand` class; see `guidance.hand_guidance_enabled` and `hand_guidance_always`).
- **Identify mode** tells you what is in front of the camera: *"This is coffee. It's on your list."* or *"I see pasta, milk, and rice."* (up to `guidance.identify_max_objects`).

Team #4 Colorado Dreaming, GEEN 1400 Fall 2026. The product spec is [`PRD.pdf`](PRD.pdf);
implementation choices are in [`DECISIONS.md`](DECISIONS.md).

```
 SLR body (worn)                                     Laptop
 USB Camera -> Pi Zero 2 W  --UDP: JPEG frames------>  YOLO detector (GrocerEye-style model)
 buttons, mode switch       <-UDP: frame acks------   guidance engine (pure logic)
 volume dial (MCP3002)      --WebSocket: controls-->  session state + shopping list
 headphones (I2S DAC)       <-WebSocket: cues------   Piper TTS clips, debug view, logs
```

The Pi only captures, plays audio and reads controls. Everything else runs on the laptop, fully
offline.

## Quick start: sim mode (laptop only)

Sim mode runs the whole product on one laptop with the webcam, keyboard and speakers. It uses
the same code paths as the real device, including UDP and WebSocket over localhost.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[server,dev]"
# NVIDIA GPU (recommended): install the CUDA build of PyTorch first, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

slr clips download-voice        # one-time Piper voice download (needs internet once)
slr clips build-system          # render the device's local clips (already committed)
slr list add pasta              # see valid items with: slr list classes
slr sim
```

Sim keys (in the terminal or the debug window):

| Key | Event | What it does |
|---|---|---|
| `â†’` / `n` | next_item | next item on the list |
| `â†` / `p` | prev_item | previous item |
| `Space` | enter | stop/restart search / check off when within reach |
| `i` | identify | what is in front of the camera? |
| `o` / `x` | mode_on / mode_off | start guidance / silent monitor (the mode switch) |
| `+` / `-` | volume | volume 0 to 10 |
| `h` / `s` | help / status | spoken help, progress |
| `q` | shutdown | quit |
| `f` (debug window) | | save the current frame to `data/captured/` for training |

Options: `slr sim --camera 1`, `slr sim --file some_video.mp4`, `--no-debug`, `--no-audio`.

The shipped config sets `guidance.stream_in_idle: true` and `stream_in_off: true`: the camera
and detector run in both switch positions, so the debug view stays live. OFF is a silent
monitor. ON automatically searches for the current list item, gives shelf and hand cues, and
plays the guidance beeps. Identify is the TOGGLE position (or a long shutter press).

The first run renders a few hundred speech clips in the background (about a minute). After that
they are cached in `data/clips/`.

## The model

The server ships with the Stage A GrocerEye model (`models/grocereye_a/best.pt`, YOLO11n trained on
the Freiburg "5K Groceries" data: **test mAP50 0.930** vs GrocerEye's 0.846, see
[`models/grocereye_a/eval_report.md`](models/grocereye_a/eval_report.md)). It knows 25 categories
(pasta, rice, coffee, tea, ...), not brands. It was trained on close-up photos of German packaging, so
Identify mode (item held near the camera) works best; Stage B (our own photos) is needed for shelf
distance and US packaging. See [`training/README.md`](training/README.md).

To debug the pipeline with everyday objects instead, switch `config/server.yaml` to the stock COCO
model: `weights: yolo11n.pt`, `classes_file: config/classes_coco.yaml`, `imgsz: 640`.

Run `slr list classes` to see valid item names. Spoken names live in `config/classes.yaml`
(`tomato_sauce: tomato sauce`).

## Command reference

If the package is not installed, run the same commands from the repo root with
`python3 -m slr ...` or `python3 slr.py ...`.

| Command | Purpose |
|---|---|
| `slr sim` | server + simulated device on this laptop (FR-70) |
| `slr server` | laptop server for the real device |
| `slr device` | run the device (on the Pi; `--sim` for laptop backends) |
| `slr device test-adc` | live MCP3002 bytes/counts/voltage for the volume dial |
| `slr device test-controls` | live readings of buttons, mode switch and volume dial (FR-76) |
| `slr device test-camera` | open the camera and save one frame to `data/camera_test.jpg` |
| `slr device calibrate-volume` | record dial min / middle / max (FR-44) |
| `slr list show / add / remove / clear / reset / classes` | shopping list (FR-3, FR-6) |
| `slr bench` | latency, FPS and UDP loss; compares resolutions (FR-75) |
| `slr bench --log data/logs/session_*.jsonl` | same numbers from a real session log |
| `slr replay <video-or-folder> --script s.yaml` | cue timeline for a recording (FR-73) |
| `slr clips build / build-system / download-voice` | speech clips |

The shopping list can also be edited by a helper in any browser, with a screen reader if needed,
at `http://<laptop>:8767/` while the server runs (FR-4).

Session logs (frames, detections, state changes, cues, latency, loss) go to `data/logs/` as JSONL
(FR-72).

## Raspberry Pi setup

Hardware: Pi Zero 2 W, Raspberry Pi OS Lite 64-bit, USB UVC webcam (640x480 capable),
TLV320DAC3100 I2S DAC with 3.5 mm jack, MCP3002 ADC (SPI).

```bash
git clone https://github.com/Ore-Porat/VisuallyImpairedBodycam.git && cd VisuallyImpairedBodycam
sudo ./deploy/install_pi.sh     # apt deps, venv, SPI + I2C + I2S, sudoers, systemd service
sudo reboot
```

Then set `server.host` in `config/device.yaml` to the laptop's hotspot address (Windows mobile
hotspot: usually `192.168.137.1`; Linux NetworkManager hotspot: `10.42.0.1`), or `auto` to find
it via mDNS. The service `slr-device` starts on boot and says "Starting up", then "Ready" once
connected (FR-50).

After rebooting, `i2cdetect -y 1` should show the DAC at address `18`, and
`v4l2-ctl --list-devices` should list the USB camera. `aplay -l` should show the I2S sound card.

Troubleshooting:

- **`PortAudio library not found`**: `sudo apt-get install -y libportaudio2`, then check
  `.venv/bin/python3 -c "import sounddevice as sd; print(sd.query_devices())"`. If PortAudio is
  still unavailable, the device falls back to `aplay` for speech; guidance tones need
  `sounddevice`.
- **Camera not in `v4l2-ctl --list-devices`**: check `lsusb` and `dmesg | tail -30`. The camera
  must be a UVC device; a hub or cable that does not enumerate it will not appear as
  `/dev/video0`.

No ML runtime is ever installed on the Pi: OpenCV reads the USB camera, and YOLO detection
still runs on the laptop.

### Wiring (BCM numbering, set in `config/controls.yaml`)

| Part | Pins | Notes |
|---|---|---|
| Shutter release | GPIO 17 to GND | momentary; internal pull-up |
| Film advance lever | GPIO 27 to GND | momentary |
| Rewind knob | GPIO 22 to GND | momentary |
| Mode switch wire a / b | GPIO 5 / GPIO 6 | contact closed = 1. OFF=00, ON=11, TOGGLE=01. Flip `invert` if reversed |
| USB camera | any USB port (OTG adapter on the Zero 2 W) | appears as `/dev/video0`; set `camera.index` if it is a different number |
| MCP3002 pin 1 (CS/SHDN) | GPIO 8 / CE0, physical pin 24 | SPI0 chip select |
| MCP3002 pin 2 (CH0) | divider junction | volume input |
| MCP3002 pin 3 (CH1) | unused | second channel |
| MCP3002 pin 4 (VSS) | GND | ground |
| MCP3002 pin 5 (DIN) | GPIO 10 / MOSI, physical pin 19 | SPI data to chip |
| MCP3002 pin 6 (DOUT) | GPIO 9 / MISO, physical pin 21 | SPI data from chip |
| MCP3002 pin 7 (CLK) | GPIO 11 / SCLK, physical pin 23 | SPI clock |
| MCP3002 pin 8 (VDD) | 3.3 V, physical pin 1 or 17 | also the ADC reference; do not use 5 V |
| Volume dial | divider: 3.3 V, 5.6k fixed, dial, GND; midpoint to CH0 | any fixed resistor works, calibration handles it |
| I2S DAC (TLV320DAC3100) | BCK GPIO 18, WSEL GPIO 19, DIN GPIO 21, SDA GPIO 2, SCL GPIO 3, RST GPIO 4 (+ 3.3 V, GND) | `dtoverlay=i2s-dac`; no MCLK wire |
| Fallback volume buttons | GPIO 23 / 24 | used only if the MCP3002 can't be read (FR-46) |

Verify the hardware with `slr device test-adc` (dial only), `slr device test-controls`
(buttons/switch/dial) and `slr device test-camera` (saves one camera frame to
`data/camera_test.jpg`). If the USB camera is not `/dev/video0`, run `v4l2-ctl --list-devices`
and set `camera.index` in `config/device.yaml`.

### Calibration

- **Volume dial:** `slr device calibrate-volume`, turn to min / middle / max when asked. Values are
  saved to `config/device.yaml`. `curve: log` fits a log curve instead of piecewise lines.
- **Vertical zones (shelf height):** the zones depend on how the camera tilts on the chest. Run
  `slr server`, stand 1.5 m from a shelf, hold an item at eye level, and read the box center in the
  debug view (the gray lines mark 0.33 and 0.66). Set `guidance.vertical_offset` in
  `config/server.yaml` to `0.5 - <observed center y>`.
- **Within reach:** with the item at arm's length straight ahead, note the box height and set
  `guidance.reach_h` a little below it.
- **Hand cues:** if "Hand left" and "Hand right" come out swapped, `flip_horizontal` fixes both the
  shelf and the hand cues. Raise `guidance.hand_cue_tol` if the cues flip-flop while the hand is
  nearly on the item; lower `hand_touch_overlap` if "That's it" never comes.
- **Mirrored camera:** set `guidance.flip_horizontal: true` if "left" and "right" come out swapped.
- **Confidence:** `training/pick_threshold.py --write` sets `guidance.conf_find` from held-out photos (TR-14).

## Development

```bash
pytest                         # all tests; test_guidance.py is the most important
```

- `slr/server/guidance.py` is pure (no I/O, time passed in) and has thorough unit tests.
- `tests/test_end_to_end.py` runs a real server and device over localhost with a scripted
  detector, so the full pipeline is tested without a webcam or GPU.
- Every tunable number is in `config/*.yaml`.

Layout: `slr/common` (protocol, config), `slr/server` (detector, guidance, session, clips, debug
view, list editor), `slr/device` (camera, controls, volume dial, audio, frame sender, control
client), `training/` (dataset, train, evaluate, thresholds, capture), `deploy/` (Pi install).

## Demo checklist

1. Charge the device battery pack.
2. Start the laptop hotspot; confirm the Pi joins (`ping` it) and UDP isn't blocked between clients.
3. `slr server` on the laptop (debug view opens). Check the model line in the log.
4. Power on the device: it says "Starting up". With the switch OFF, video keeps running silently.
5. Mode switch to ON. It says "Searching for ..." and starts giving direction/hand cues. Turn the volume dial and check "Volume N" in the headphones.
6. `slr list reset` (or the web editor) and confirm the list with `slr list show`.
7. Quick check: Identify a known item, then switch ON for a Find of one item.
8. After the demo: hold shutter + advance for 5 s to shut down safely.

## Acceptance test

Protocol: PRD section 13. Copy `reports/acceptance_TEMPLATE.md` to
`reports/acceptance_<date>.md`, fill it in, and paste the numbers from
`slr bench --log data/logs/<session>.jsonl`.
