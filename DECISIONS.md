# Implementation decisions

Choices made where the PRD (`PRD.pdf`) was ambiguous or left room (PRD rule 7). One line each, grouped by area.

## Repo and config
- The PRD lives in the repo as `PRD.pdf` (the original document); this file and the README reference its section numbers.
- The shipped `config/server.yaml` uses the fast zero-shot pipeline (`backend: fast_grocery`): YOLO-World S TensorRT + CLIP ViT-B/16 + EgoHands 960. It has no trained grocery classifier and does not depend on the production camera or room. The old trained Stage A/snack weights remain in `models/` as historical artifacts but are no longer loaded.
- `conf_identify` is 0.75. On one held-out Freiburg test photo per class (static image, so all 5 burst frames agree), Identify gave 21/25 correct with 3 wrong at 0.6, and 21 correct, 2 not sure and 2 wrong at 0.75. Higher thresholds cost correct answers without removing all errors. The remaining errors are confidently wrong close-ups (e.g. milk vs tea); Stage B photos plus `pick_threshold.py` on the demo categories are the fix, not a higher global threshold.
- `conf_find` stays at the PRD default 0.5: on the Freiburg test split no single threshold reaches 0.90 precision for all 25 classes (`reports/threshold_grocereye_a_*.md`). TR-14 should be run on our own held-out photos of the demo categories after Stage B.
- `config/catalog.yaml` is the authoritative menu of item names for the shopping list (FR-2), so `slr list add` works without loading the model. Each entry carries the spoken name and the CLIP prompts; `config/classes.yaml` is kept only as a legacy fixture and is still readable by the loader. Items the user adds are written to `data/catalog_user.yaml` and merged over the presets, so an update never clobbers them. The server warns at startup about list items without curated prompts.
- The preset catalog ships generic groceries only, no brand entries. Brand-level accuracy depends on prompt specificity; the web page's custom-item editor (Phase 2) is where a user supplies it.
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
- Hand guidance compares the hand box center with the item box center: a short cue names the way the HAND must move ("Hand up and left.", "Reach forward."), spoken on change or every `hand_repeat_s`. With `hand_guidance_always: true` (shipped) it starts as soon as both boxes are visible and suppresses the camera-direction speech for that frame; the target tone keeps playing. When the hand covers the item (center inside the item box, or >= `hand_touch_overlap` of whichever box is smaller overlapping the other) for `hand_touch_hold_s`, it says "That's it." and checks the item off (`hand_auto_check_off`), even before the old WITHIN_REACH state. Only the most confident hand box counts, so a second shopper's hand in view is ignored.
- Object permanence (`hand_ghost_*`): a detector cannot see an item through the shopper's hand, so the item box disappears exactly when a grab succeeds. While both were visible the engine records `item center - hand center` and `item height / hand height`; when the item vanishes and the hand still covers the last known place it keeps a predicted "ghost" box at `hand + offset` and carries on guiding / checking off. Because the hand moves with the wearer, the ghost follows a head turn. A real detection always replaces the ghost. An anchor is only recorded while `hand_on_item` is true, so a hand that merely passes by cannot hide a missing item, and a ghost never outlives `hand_ghost_max_s`.
- Camera-motion gate (`camera_motion_*`): the camera is worn on the chest, so the picture moves whenever the shopper turns or walks. The server estimates the frame-to-frame global shift with phase correlation on the already-decoded frame (`slr/server/motion.py`, ~1-2 ms per frame) and passes it as `Frame.motion`. At or above `camera_motion_freeze` (0.008 = about 5 px per frame at 640x480, i.e. a slow scan) the "Lost it." time-out is frozen and a ghost's lifetime clock is restarted; tracking resumes after `camera_motion_settle_s` of stillness. The freeze is capped at `lost_timeout_s + camera_motion_max_freeze_s`, so a permanently shaky view still ends in "Lost it.". The Pi does no extra work.
- The "That's it." hold is deliberately *not* reset while the camera moves: the ghost is anchored to the hand, so the hand/item relationship stays valid while the background moves. Resetting it would make grabs fail during ordinary body sway. The hand's own movement is what makes a brief brush fail the hold.

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
- Shipped model (2026-10-29, superseded 2026-11-06): `models/snacks_yolo11n.pt`, a YOLO11n the team trained directly on photos of the six demo products (oreo, cheerios, coca_cola_can, campbells_chicken_noodle, lays, cheetos; dataset `../model/snacks_dataset`, 651/79/79 images, imgsz 640, run stopped at epoch 37/60 and kept as is). Test mAP50 0.888 (`reports/eval_snacks_yolo11n_test_2026-10-29.md`). It uses its own class list, `config/classes_snacks.yaml`, and `imgsz: 640`. It has no `hand` class, so hand cues stay silent until a hand class is trained in; the grocery models (`config/classes.yaml`, imgsz 320) remain in `models/` for that work. Tests that script grocery detections pin `catalog_file` to `config/catalog.yaml`.

## Fast grocery model (2026-11-06)
- The trained shelf/snack pipeline was strong in-domain but did not transfer to the real webcam/room. The fast zero-shot pipeline (YOLO-World + CLIP + EgoHands) handled the new scene better, so it is now the only shipped runtime. `backend` accepts `fast_grocery` and the test-only `scripted` replay backend; the old `ultralytics` and `darknet_opencv` backends were removed.
- The TensorRT engines are not committed (`models/**/*.engine` is gitignored) because they are specific to the GPU and TensorRT version. `models/tensorrt/README.md` documents copying/rebuilding them. The shipped S engine trades a little recall for ~31 FPS; switch to `shelf_world.engine` (L) if recall matters more.
- `fast_item_min_score` ships at 0.30 and `fast_item_margin` at 0.02, calibrated on the GrocerEye val split (`reports/clip_item_calibration.md`). These are **cosine similarities**, not probabilities: 2-item head-to-head accuracy is 0.957, while a 1-item list (no competitor) is the weak case and reaches only ~0.47 recall at ~0.50 precision. A close call is rejected. `fast_detector_conf` still ships at 0.01 for handheld/webcam recall; raise it to 0.1–0.25 in dense shelf scenes.
- Label filtering is list-only: YOLO-World proposes a box, CLIP scores it against the current shopping list's prompt vectors, and `choose_item` applies `fast_item_min_score` plus `fast_item_margin`. There are no reject classes and no YOLO-keyword compatibility map; a box that matches nothing is simply dropped. `hand` boxes come from EgoHands and remain a non-item class.
- `cookies` is in `config/catalog.yaml` with curated prompts like every other preset.
- `Detection.conf` is a raw CLIP cosine similarity for products and the EgoHands probability for hands. The four guidance thresholds are therefore on two scales; `conf_find`/`conf_identify`/`conf_wrong_item` were recalibrated to cosine values, `hand_conf` stays a probability.

## List-only CLIP verification (2026-11-12, Phases 0-1 shipped)
- The detector scores only the items on the shopping list. No generic class is ever emitted (`soda`, `package`, ...), and `DETECTOR_KEYWORDS` is removed: YOLO-World supplies the box, CLIP decides whether the crop matches a list item. The full design is `docs/clip-item-verification.md`.
- Consequence: Find, the "that's X, not Y" warning (FR-20) and Identify can only name list items. If the list is empty, Identify says "Your shopping list is empty." and does not start a burst. This is the accepted cost of list-only scoring (D2).
- Acceptance is "does this crop match item X?" per item, not a softmax over a label set. The detector's `PromptBank` holds one averaged CLIP vector per list item, rebuilt only when the list or a prompt changes. With one item, a calibrated absolute cosine floor decides; with two or more, the winner must also lead the runner-up by `fast_item_margin`. A close call returns nothing instead of guessing.
- Prompts are also generated from YOLO's own box label (`a can of Coke`, `a bottle of Coke`), which is what makes free-text items usable without hand-written prompts. The label is never spoken or emitted.
- Phase 0 measured this on the GrocerEye validation split (1148 crops): two-item head-to-head accuracy 0.957 (ViT-B/16) and 0.985 (ViT-L/14); all 26 items competing 0.611/0.745. One-item verification is the hard case: B/16 cannot reach 0.90 precision at a useful recall, L/14 reaches it only at 0.074 recall. Reports: `reports/clip_item_calibration*.md`. Consequence: the margin rule is the part that works; one-item lists are documented as the weakest mode and use per-item thresholds (a catalog entry may set `min_score`).
- The shipped backbone stays **ViT-B/16**. ViT-L/14 is more accurate but measured 182.6 ms for 20 crops versus 41.6 ms on the target laptop (4.4x), which does not fit the 20 fps budget. The model is a config value (`fast_clip_model`) for future hardware.
- Real-engine smoke test at 640x480: ~28 ms per frame for the full YOLO + CLIP + EgoHands path with a one-item list. A 25-image pasta sample gave 11/17 detections on pasta photos and 0/15 on rice photos (precision-safe; recall is the trade-off).
- The preset catalog is `config/catalog.yaml` (generic groceries only, curated prompts). User items and their prompts live in `data/catalog_user.yaml`, merged over the presets, so an update never overwrites them.

## Free-text items and the prompt editor (2026-11-12, Phase 2)
- Any product name can be added from the web page or `slr list add "Coca Cola"`. An unknown name is slugified (`coca_cola`), gets the typed words as its spoken name, and is saved to `data/catalog_user.yaml` with no curated prompts. The detector then uses the generated `a photo of X` and `a can of X`/`a bottle of X` prompts.
- The web page has an **item prompts** editor at `/prompts`: one prompt per line, up to 12 prompts of 300 characters. Saving a preset's prompts writes a user override; `config/catalog.yaml` is never modified. "Restore" removes the override (or, for a user item, falls back to the generated prompt).
- The editor refreshes the detector's prompt bank immediately (`set_items` on the next frame), so a prompt change needs no restart. It also updates the phrase catalog's id -> spoken map so new item names are spoken correctly.
- The add box is a text input with a `<datalist>` of preset names, not a closed drop-down: presets are a shortcut, not a limit. Items whose prompts are generated are marked in the list and warned about at startup.
- `slr list classes` prints presets and custom items; `slr list prompts <item>` prints the active prompts and whether they are preset, custom or generated.
- The running server watches `data/catalog_user.yaml` (same 5-tick check as the shopping list), so an item or prompt added by the CLI reaches it without a restart.
- `ShoppingList` accepts either a `Catalog` (normal server/CLI use) or the old `dict` (tests, old callers). The dict mode keeps the strict "unknown item" error.

