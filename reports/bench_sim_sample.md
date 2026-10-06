# slr bench 2026-10-06 10:12

source: C:\Users\orepo\Documents\boulder\GEEN1400 Fall 2026\VisuallyImpairedBodycam\.venv\Lib\site-packages\ultralytics\assets\bus.jpg, target: person, model: yolo11n.pt

| run | frames | FPS | infer p50/p90 ms | capture->decision p50/p90 ms | capture->audio p50/p90 ms | UDP loss | JPEG KB |
|---|---|---|---|---|---|---|---|
| 640x480 | 83 | 9.3 | 51/63 | 21/46 | 83/296 (7) | 0.0% | 55.2 |
| 416x312 | 95 | 9.6 | 50/64 | 28/49 | 68/87 (7) | 0.0% | 38.6 |
| 320x240 | 97 | 9.8 | 44/64 | 24/46 | 63/71 (7) | 0.0% | 25.0 |

Targets: capture->audio p90 <= 500 ms (NFR-1), FPS >= 5 (NFR-2), UDP loss <= 10% (NFR-3).
