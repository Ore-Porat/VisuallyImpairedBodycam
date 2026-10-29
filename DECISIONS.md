# Implementation decisions

Choices made where the PRD (`PRD.pdf`) was ambiguous or left room (PRD rule 7). One line each, grouped by area.

## Repo and config
- The PRD lives in the repo as `PRD.pdf` (the original document); this file and the README reference its section numbers.
- `config/server.yaml` ships pointing at the trained Stage A model `models/grocereye_a/best.pt` (imgsz 320). Its weights (5.4 MB) are committed so a fresh clone works; other weights stay gitignored. Stock COCO (`yolo11n.pt` + `classes_coco.yaml`) remains available for pipeline debugging.
- `conf_identify` is 0.75. On one held-out Freiburg test photo per class (static image, so all 5 burst frames agree), Identify gave 21/25 correct with 3 wrong at 0.6, and 21 correct, 2 not sure and 2 wrong at 0.75. Higher thresholds cost correct answers without removing all errors. The remaining errors are confidently wrong close-ups (e.g. milk vs tea); Stage B photos plus `pick_threshold.py` on the demo categories are the fix, not a higher global threshold.
- `conf_find` stays at the PRD default 0.5: on the Freiburg test split no single threshold reaches 0.90 precision for all 25 classes (`reports/threshold_grocereye_a_*.md`). TR-14 should be run on our own held-out photos of the demo categories after Stage B.
- `classes_file` is the authoritative list of item names for the shopping list (FR-2), so `slr list add` works without loading the model. The server warns at startup if the model's class names don't match.
- Configs reject unknown keys (pydantic `extra="forbid"`) so typos fail loudly instead of being silently ignored.
- Sim overrides live in a `sim:` section of `device.yaml`, deep-merged by `slr sim` / `slr device --sim`.
- Paths in configs are relative to the repo root (the parent of `config/`).

## Protocol (PRD 7)
- Frame header is network byte order `!4sIIQBHHH` (27 bytes). The ack is binary too: `SLRA`, session_id, frame_id, infer_ms (float32), server_ts_ms. session_id is included so acks from a previous boot are ignored.
- Extension: device -> server `cue_started {cue_id, ts}` so the server can log true capture-to-audio latency (NFR-1).
- Extension: heartbeat carries optional `clock_offset_ms` (estimated by the device from ack round trips, NTP style) and `rtt_ms`, because the Pi has no internet time on the hotspot.
- Extension: `cue` messages also carry `text` and `key` (for logs and the debug view; the device ignores them).
- Clip URLs in `clips_sync` are server-relative (`/clips/<name>.wav`); the device prefixes its configured server host and `http_port`.
- HTTP (clips + list editor) is a third port (8767), served by a small threaded `http.server`, because the websockets HTTP hook can't take form POSTs.
- If the device gets a cue for a clip it doesn't have, it downloads it on demand before playing (few KB over the LAN). The server renders clips on demand too, so no phrase/item combination is ever missing.

## Guidance (PRD 8)
- Hysteresis (`direction_hysteresis`, 0.02) on the horizontal and vertical zones, so a box sitting on a boundary doesn't make the cue flip back and forth.
- Direction and vertical zone are spoken as one phrase ("Slightly left. Eye level.") because guidance cues replace each other in the audio queue; two separate guidance clips would cancel each other.
- `vertical_offset` is added to the box center y.
- "Not seeing X" timer counts from the search start or the last raw sighting of the target.
- Wrong-item warning (FR-20) also requires the other item to be stable (k of n frames), not just one frame: precision first. Off by default it would hide a real feature, so it's on, rate limited to every 5 s.
- Identify: per frame the pick maximizes `conf - 0.5 * distance_from_center`. A tie in the vote is "Not sure". With fewer than 3 frames in 1.5 s it says "Not sure." (FR-32).
- Controls other than the mode switch are ignored during the ~1 s identify burst.
- next/prev during a search stops the search and announces the new item.
- Enter on an already-found item still starts a search (the user chose it explicitly).
- In standby, any non-switch control says "Paused. Turn the switch on to continue." (FR-41 needs audio confirmation).
- Within-reach exits back to guiding only when the box drops below `reach_h * 0.8` or leaves the straight-ahead zone, to avoid flicker.
- "Getting closer" (FR-22, P2) is implemented but off by default.
- On device disconnect the server quietly goes to STANDBY; the device re-sends its switch position on reconnect, so the user hears "Connected" then "Ready". An active search is not resumed.
- `non_item_classes` (default `[hand]`): detector classes that are never spoken as a found, wrong (FR-20) or identified (FR-31) item. The hand is detected so later guidance can steer it to the item; today it is only filtered out.

## Audio (FR-60..66)
- Priorities: system 3 > confirmation 2 > guidance 1 > info 0. Each phrase's priority is in `phrases.yaml`.
- A cue with `interrupt: true` (item announcements, found, lost, results) also cuts off an equal-priority cue, so pressing "next" quickly feels responsive.
- When a cue interrupts, waiting cues of lower priority and all waiting guidance are dropped (they're stale). Waiting guidance older than 2 s is also dropped.
- Clip names are `<phrase key>-<sha1(voice | text)[:10]>`: clips regenerate automatically when text, spoken names or voice change (FR-61).
- TTS engines are tried in order: Piper, then espeak-ng, then the OS voice via pyttsx3, then (with a loud warning) beeps, so the pipeline never hard-fails.
- Volume level announcements ("Volume 6") are local system clips, so volume works with zero latency and even when disconnected.
- Tones are mixed with speech in one sounddevice stream and ducked 50% under speech. The `aplay` backend plays clips only (no tones).
- Volume gain: level 0 = mute, levels 1..10 evenly spaced in dB from -30 dB to 0 dB.

## Controls (PRD 6.4)
- Hold actions fire while the button is still held (at 2 s), so a blind user knows when to let go. The shutter long press (identify) fires at 0.8 s while held. When a button's long and hold actions differ (advance: next vs help), the long action is decided on release.
- The shutdown combo suppresses both buttons' own actions as soon as both are down.
- "Hysteresis of half a step" for the volume dial means a total band of half a step: the level changes once the dial is a quarter step past the boundary.
- Volume log curve (FR-44): `p = A + B ln(x + C)` fitted exactly through the three calibration points (mirrored when the curve bends the other way); if the points are exactly linear it falls back to piecewise.
- Mode switch: OFF -> TOGGLE directly sends `mode_on` then `identify`. TOGGLE -> ON sends `mode_on`, which the server ignores while active.
- FR-43 latching TOGGLE: with `toggle_latching: true` the device ignores buttons (except shutdown) while the switch sits in TOGGLE.
- First-run tutorial (FR-57): the device sends `help` on its first connection and stores a flag in `data/device_state/first_run_done`.
- Default pins (BCM): shutter 17, advance 27, rewind 22, switch a 5 / b 6, fallback volume buttons 23 / 24. These avoid SPI0 (7-11, MCP3008) and I2S (18-21).
- Default `device.yaml` volume calibration is computed from the measured resistances with a 5.6k divider read by a 10-bit MCP3008 at VREF 3.3 V: 464 / 531 / 569 counts.

## Device
- Camera is a USB UVC webcam read through OpenCV (the `opencv` backend is the default); the CSI `picamera2` backend remains available but is no longer installed by `install_pi.sh`.
- Camera stays started while idle (reopening a USB camera takes about 1 s); frames are only captured and sent while streaming (FR-54, NFR-12).
- JPEG encoding uses simplejpeg on the Pi, then OpenCV, then Pillow.
- The DAC is a TLV320DAC3100 configured over I2C at startup (`audio.dac: tlv320dac3100`). `dtoverlay=i2s-dac` supplies the generic I2S clocks; `audio.dac: none` is for a DAC that needs no setup.
- Reconnect backoff is 0.5 s doubling to 4 s, well inside the 10 s recovery target (NFR-10).
- Shutdown runs `sudo shutdown -h now` (allowed by a sudoers rule from `install_pi.sh`); in sim it just exits.

## Sim and tooling
- `slr sim` runs the server and the device in one process and one event loop, still talking over real UDP + WebSocket on localhost. Keys work in the terminal and in the debug window.
- `slr bench` runs a loopback server + sim device per resolution with a webcam or file source; `--log` analyzes a real session log. It waits for clip pre-rendering to finish first, because Piper rendering competes for CPU with detection.
- Replay runs the same `ServerCore` on a virtual clock and only consumes frames while the engine has streaming on, like the device.

## Training (PRD 10)
- Dataset source: the master-branch zip of `aleksandar-aleksandrov/groceries-object-detection-dataset` (the "5K Groceries" set linked from GrocerEye's README).
- Split: our own stratified-by-class 80/10/10 (seed 0), because the dataset's own split has only ~3% val and no test set, and GrocerEye's split was never published. Recorded in `data/grocereye/split.json`.
- Class ids are alphabetical by normalized snake_case name (matches the dataset's `classes.txt`).
- Augmentation beyond Ultralytics defaults: `hsv_v` 0.6, `scale` 0.7, and (with albumentations) Blur/MotionBlur p=0.1 and JPEG compression p=0.2 to match the device's JPEG stream.
- Stage B combining remaps class ids by name, so a Roboflow export with only the demo categories works.
- GrocerEye's 84.59% is compared against our mAP50 (it was Darknet mAP@0.5 on a different split).
- `pick_threshold.py` matches greedily at IoU >= 0.5; a class with no predictions counts as passing precision (with a warning); it refuses `--write` if no threshold qualifies.
- Products-6K (extra data) is mapped onto the existing 25 classes instead of adding classes or training per product (SKU): 1 to 5 photos per SKU is too few for detection, and the shopping list, `classes.yaml` and guidance stay unchanged.
- Products-6K class mapping uses only the specific and broad category columns. The free-text description was tried and dropped: flavor words pulled biscuits, soap and detergent into food classes.
- Products-6K boxes: studio photos use a plain-background box (exact); Grounding DINO (tiny) only for in-store photos and backdrops that are not clearly plain (edge-uniformity score < 0.8). Boxes are cached in `boxes.json` so manual fixes survive reruns.
- Products-6K split is by product, stratified by class, 70/15/15 with seed 0 (larger val/test shares than Freiburg because many classes have only a handful of products). Near-duplicate photos of one product (difference hash within 4 bits) are dropped.
- Products-6K in-store boxes (2026-10-27/28): the default DINO prompt `"{name}. grocery product."` boxed the whole shelf on a quarter of the hand-held photos; the plain `"{name}."` prompt fixed most of those but found nothing on 34. The kept set (`data/products6k_v2`) merges both runs and skips 36 photos. grocereye_p6k2 (trained on it) is the server model: Freiburg test mAP50 0.922, Products-6K test 0.796 (Stage A alone: 0.300).
- Hand class: a 26th class `hand`, last in the id order so the 25 grocery ids do not move. First labels are Grounding DINO `"hand."` boxes on the ~370 Products-6K in-store photos (`get_hands_data.py`), which match the bodycam view; an egocentric hand dataset is the next step if the in-store photos are not enough.
