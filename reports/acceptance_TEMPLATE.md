# Acceptance test: <date>

Protocol: PRD section 13. Pass = NFR-1, NFR-4 and NFR-7 met.

- Model: `<model.weights>`  conf_find: `<value>`
- Laptop / hotspot:
- Tester(s):
- Session log(s): `data/logs/...`

## Setup
- [ ] Mock shelf, 3 heights, all demo categories among distractors, look-alikes next to each other
- [ ] Tester blindfolded, wearing the device, 1.5 m from the shelf facing it

## Find trials (NFR-7: >= 8 of 10 within 60 s; NFR-4: zero wrong-item announcements)

| # | Category | Lighting | Time (s) | Success (hand on correct item <= 60 s) | Wrong-item "found"? | Notes |
|---|---|---|---|---|---|---|
| 1 | | normal | | | | |
| 2 | | normal | | | | |
| 3 | | normal | | | | |
| 4 | | normal | | | | |
| 5 | | normal | | | | |
| 6 | | normal | | | | |
| 7 | | normal | | | | |
| 8 | | dim | | | | |
| 9 | | dim | | | | |
| 10 | | dim | | | | |

Successes: __ / 10. Wrong-item announcements: __.

## Identify trials (NFR-4: zero wrong "This is ...")

| # | Item handed over | Announcement | Correct / "Not sure" / Wrong |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

Correct: __  Not sure: __  Wrong: __

## Latency and loss (from `slr bench --log <session log>`)

Paste the table here. Targets: capture->audio p90 <= 500 ms (NFR-1), >= 5 FPS (NFR-2), UDP loss <= 10% (NFR-3).

## Result

- NFR-1:  pass / fail
- NFR-4:  pass / fail
- NFR-7:  pass / fail
- Notes and follow-ups:
