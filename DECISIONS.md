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
- HTTP (clips + list editor) is a third port (8767), served by a small threaded `http.server`, because the websockets HTTP hook can't take form POSTs. The list page still uses ordinary forms; a small progressive-enhancement script polls `/list/state.json` once a second and patches statuses/current in place, so device edits appear without a manual reload. Without JavaScript a `noscript` meta refresh does the same every three seconds.
- If the device gets a cue for a clip it doesn't have, it downloads it on demand before playing (few KB over the LAN). The server renders clips on demand too, so no phrase/item combination is ever missing.

## Guidance (PRD 8)
- Hysteresis (`direction_hysteresis`, 0.02) on the horizontal and vertical zones, so a box sitting on a boundary doesn't make the cue flip back and forth.
- Direction and vertical zone are spoken as one phrase ("Slightly left. Eye level.") because guidance cues replace each other in the audio queue; two separate guidance clips would cancel each other.
- `vertical_offset` is added to the box center y.
- "Not seeing X" timer counts from the search start or the last raw sighting of the target.
- Wrong-item warning (FR-20) also requires the other item to be stable (k of n frames), not just one frame: precision first. Off by default it would hide a real feature, so it's on, rate limited to every 5 s.
- Identify: per frame the pick maximizes `conf - 0.5 * distance_from_center`. A class gets one vote per frame; classes above `identify_min_votes` are reported, up to `identify_max_objects` (shipped: 3). One object keeps the old "This is X" wording; several use "I see X, Y, and Z." A tie for the single-object answer is "Not sure". With fewer than 3 frames in 1.5 s it says "Not sure." (FR-32).
- All controls are ignored during the ~1 s identify burst, so a second flick cannot toggle an item mid-answer.
- next/prev during a search stops the search and announces the new item. With `auto_find_on_mode_on`, it then starts searching for that item.
- Enter on an already-found item still starts a search (the user chose it explicitly).
- A short flick to TOGGLE flips the current item between found and not found. Marking it found speaks "X marked found. Next: Y." (or "...That was the last item on your list."), moves the current index to the next pending item, and starts its search when ON. Marking a found item pending keeps it selected and restarts its search when ON. It works in OFF and ON. This means a second flick affects the newly current item; use previous-item or the web editor's "Make current" to go back and undo. If a search is running for the item marked found, that search stops before the next item starts.
- Holding TOGGLE for `mode_switch.toggle_hold_s` runs Identify instead, so no shutter button is required. The same hold works from OFF and returns to OFF afterwards.
- OFF is a silent monitor with `stream_in_off: true`: the camera and detector keep running (so the debug view shows boxes), but the engine stays in STANDBY and emits no guidance speech or tones. Normal controls are ignored silently. The explicit TOGGLE found/not-found command and held-TOGGLE Identify still speak; local volume/startup/connection clips still play.
- `mode_on` with `auto_find_on_mode_on: true` starts a Find for the current list item immediately ("Searching for X."). If the list is empty it says so and stays IDLE; if the current item is already found it announces that and waits, so OFF-toggled items are not guided. Enter still starts a search for a found item on purpose; next/prev and check-off continue guiding to pending items.
- Within-reach exits back to guiding only when the box drops below `reach_h * 0.8` or leaves the straight-ahead zone, to avoid flicker.
- "Getting closer" (FR-22, P2) is implemented but off by default.
- On device disconnect the server quietly goes to STANDBY; the device re-sends its switch position on reconnect. With the switch ON, the server restarts a search for the current item.
- `non_item_classes` (default `[hand]`): detector classes that are never spoken as a found, wrong (FR-20) or identified (FR-31) item, and never offered on the shopping list.
- Hand guidance compares the hand box center with the item box center: a short cue names the way the HAND must move ("Hand up and left.", "Reach forward."), spoken on change or every `hand_repeat_s`. With `hand_guidance_always: true` (shipped) it starts as soon as both boxes are visible and suppresses the camera-direction speech for that frame; the target tone keeps playing. When the hand covers the item (center inside the item box, or >= `hand_touch_overlap` of the hand box overlapping) for `hand_touch_hold_s`, it says "That's it." and checks the item off (`hand_auto_check_off`), even before the old WITHIN_REACH state. Only the most confident hand box counts, so a second shopper's hand in view is ignored.

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
- Mode switch: a short ON -> TOGGLE -> ON flick sends only `toggle_found`; springing back is not a mode change. Holding TOGGLE sends `identify` once after `toggle_hold_s` and suppresses the found-toggle. OFF -> TOGGLE is also only a command: it does not turn the device on.
- FR-43 latching TOGGLE: with `toggle_latching: true` the switch has no release gesture, so it sends `toggle_found` on entry and a held TOGGLE does not start Identify (use the shutter long press).
- First-run tutorial (FR-57): the device sends `help` on its first connection and stores a flag in `data/device_state/first_run_done`.
- Default pins (BCM): shutter 17, advance 27, rewind 22, switch a 5 / b 6, fallback volume buttons 23 / 24. These avoid SPI0 (7-11, MCP3002) and I2S (18-21).
- The volume ADC is an MCP3002 (10-bit, two channels), read with its two-byte SPI command (`0x68` for CH0, `0x78` for CH1). Its VDD is also the reference, so VDD must be 3.3 V.
- The shipped `device.yaml` volume calibration was measured on the current dial and divider: 399 / 450 / 501 counts (CH0). Re-run `slr device calibrate-volume` if the dial or wiring changes. Calibration rejects a reading at only one ADC rail (for example `raw_max` near 1022 while `raw_min` is around 399), because that means the wiper opened at that end; a real full-range potentiometer may legitimately touch both rails.

## Device
- Camera is a USB UVC webcam read through OpenCV (the `opencv` backend is the default); the CSI `picamera2` backend remains available but is no longer installed by `install_pi.sh`.
- Camera stays started while idle (reopening a USB camera takes about 1 s). The shipped `server.yaml` sets `guidance.stream_in_idle: true` and `stream_in_off: true`, so video and detection run in both switch positions. ON auto-starts Find (`auto_find_on_mode_on`); OFF is a silent monitor. With both flags off, the older on-demand stream (FR-54, NFR-12) is unchanged.
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
- `cookies` class (2026-10-28): the demo set includes Oreos and no class fit, so Products-6K sweet biscuits (plain, filled, coated: 19 products, 113 images) map to a new class `cookies`; salty crackers stay skipped. It sits after `water` and before `hand` in `config/classes.yaml`.
- Models for the demo (2026-10-28): trained on Products-6K (+ its hand labels) only, without Freiburg, at the team's request: Freiburg is 224 px shelf crops of German packaging and did not transfer to the five US demo products seen by a webcam. The real fix is our own photos: `autolabel_ours.py` boxes `capture.py` sessions with Grounding DINO so no hand labeling is needed. `get_egohands_data.py` converts EgoHands (needs a Roboflow/Kaggle download) for hand variety.
- Shipped model (2026-10-29): `models/snacks_yolo11n.pt`, a YOLO11n the team trained directly on photos of the six demo products (oreo, cheerios, coca_cola_can, campbells_chicken_noodle, lays, cheetos; dataset `../model/snacks_dataset`, 651/79/79 images, imgsz 640, run stopped at epoch 37/60 and kept as is). Test mAP50 0.888 (`reports/eval_snacks_yolo11n_test_2026-10-29.md`). It uses its own class list, `config/classes_snacks.yaml`, and `imgsz: 640`. It has no `hand` class, so hand cues stay silent until a hand class is trained in; the grocery models (`config/classes.yaml`, imgsz 320) remain in `models/` for that work. Tests that script grocery detections pin `classes_file` to `config/classes.yaml`.

