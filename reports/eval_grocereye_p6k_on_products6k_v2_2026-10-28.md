# Evaluation: grocereye_p6k_on_products6k_v2 (2026-10-28)

- Weights: `models/grocereye_p6k/best.pt`
- Data: `data/products6k_v2/data.yaml`, split `test` (92 images, 92 labeled boxes)
- imgsz 320, stage A

## Dataset split

- Source: Products-6K https://zenodo.org/records/4428917
- Method: by product, stratified by class, val 0.15 / test 0.15, seed 0
- Images: train 428, val 101, test 92
- split.json: `data/products6k_v2/split.json` (created 2026-10-28T22:31:30+00:00)

## Summary

| metric | value | target / baseline | status |
|---|---|---|---|
| mAP50 | 0.7797 | GrocerEye 0.8459 | below baseline (-6.62 pts) |
| mAP50 (NFR-5) | 0.7797 | >= 0.85 | FAIL |
| mAP50-95 | 0.7526 | - | - |

GrocerEye's 84.59% is Darknet mAP@0.5 on its own (unpublished) random 20% test split, so the comparison is approximate.

## Per-class results

Ultralytics P/R are at the max-F1 confidence; operating-point P/R are at conf 0.50, IoU >= 0.5.

| class | images | boxes | AP50 | AP50-95 | P (val) | R (val) | P @op | R @op | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cake | 2 | 2 | 0.995 | 0.995 | 0.889 | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| candy | 4 | 4 | 0.888 | 0.888 | 1.000 | 0.000 | - | 0.000 | 0 | 0 | 4 |
| cereal | 9 | 9 | 0.793 | 0.779 | 0.866 | 0.723 | 0.875 | 0.778 | 7 | 1 | 2 |
| chips | 14 | 14 | 0.836 | 0.808 | 0.711 | 0.881 | 0.765 | 0.929 | 13 | 4 | 1 |
| chocolate | 12 | 12 | 0.598 | 0.590 | 0.854 | 0.488 | 0.750 | 0.500 | 6 | 2 | 6 |
| coffee | 8 | 8 | 0.955 | 0.955 | 0.639 | 0.875 | 0.667 | 1.000 | 8 | 4 | 0 |
| fish | 3 | 3 | 0.555 | 0.533 | 0.327 | 0.667 | 0.400 | 0.667 | 2 | 3 | 1 |
| flour | 5 | 5 | 0.544 | 0.544 | 1.000 | 0.000 | - | 0.000 | 0 | 0 | 5 |
| jam | 4 | 4 | 0.995 | 0.995 | 0.932 | 1.000 | 0.800 | 1.000 | 4 | 1 | 0 |
| juice | 3 | 3 | 0.504 | 0.489 | 0.356 | 0.667 | 0.200 | 0.333 | 1 | 4 | 2 |
| nuts | 2 | 2 | 0.662 | 0.662 | 0.622 | 0.500 | 0.250 | 0.500 | 1 | 3 | 1 |
| oil | 2 | 2 | 0.995 | 0.795 | 0.899 | 1.000 | 0.667 | 1.000 | 2 | 1 | 0 |
| pasta | 4 | 4 | 0.358 | 0.335 | 0.452 | 0.250 | 0.667 | 0.500 | 2 | 1 | 2 |
| rice | 5 | 5 | 0.995 | 0.995 | 0.914 | 1.000 | 0.833 | 1.000 | 5 | 1 | 0 |
| soda | 5 | 5 | 0.695 | 0.665 | 0.730 | 0.200 | 0.667 | 0.400 | 2 | 1 | 3 |
| spices | 2 | 2 | 0.995 | 0.846 | 1.000 | 0.781 | 1.000 | 1.000 | 2 | 0 | 0 |
| tomato_sauce | 5 | 5 | 0.678 | 0.678 | 0.945 | 0.400 | 0.667 | 0.400 | 2 | 1 | 3 |
| vinegar | 3 | 3 | 0.995 | 0.995 | 0.919 | 1.000 | 1.000 | 1.000 | 3 | 0 | 0 |

## Confusion matrix (conf 0.50, IoU >= 0.5, rows = true class)

| true \ pred | beans | cake | candy | cereal | chips | chocolate | coffee | corn | fish | flour | honey | jam | juice | milk | nuts | oil | pasta | rice | soda | spices | sugar | tea | tomato_sauce | vinegar | water | background |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beans | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| cake | . | **2** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| candy | . | . | . | . | 3 | . | 1 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| cereal | . | . | . | **7** | . | . | 1 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 1 |
| chips | . | . | . | . | **13** | . | . | . | . | . | . | . | . | . | 1 | . | . | . | . | . | . | . | . | . | . | . |
| chocolate | . | . | . | . | . | **6** | . | . | . | . | . | . | . | 1 | . | . | . | . | . | . | . | . | . | . | . | 5 |
| coffee | . | . | . | . | . | . | **8** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| corn | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| fish | . | . | . | . | . | . | . | . | **2** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 1 |
| flour | . | . | . | . | . | . | . | . | . | . | . | 1 | . | . | 1 | . | 1 | . | . | . | . | . | . | . | . | 2 |
| honey | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| jam | . | . | . | . | . | . | . | . | . | . | . | **4** | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| juice | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 1 | . | . | 1 | . | . | . | . | . | . | 1 |
| milk | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| nuts | . | . | . | . | 1 | . | . | . | . | . | . | . | . | . | **1** | . | . | . | . | . | . | . | . | . | . | . |
| oil | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | **2** | . | . | . | . | . | . | . | . | . | . |
| pasta | . | . | . | . | . | . | . | . | 2 | . | . | . | . | . | . | . | **2** | . | . | . | . | . | . | . | . | . |
| rice | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | **5** | . | . | . | . | . | . | . | . |
| soda | . | . | . | . | . | . | . | . | . | . | . | . | 1 | . | 1 | . | . | 1 | **2** | . | . | . | . | . | . | . |
| spices | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | **2** | . | . | . | . | . | . |
| sugar | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| tea | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| tomato_sauce | . | . | . | . | . | . | . | . | . | . | . | . | 1 | . | . | . | . | . | . | . | . | . | **2** | . | . | 2 |
| vinegar | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | **3** | . | . |
| water | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| background | . | . | . | 1 | . | 2 | 2 | . | 1 | . | . | . | 3 | . | . | . | . | . | . | . | . | . | 1 | . | . | . |

![eval_grocereye_p6k_on_products6k_v2_2026-10-28_confusion_matrix_normalized](eval_grocereye_p6k_on_products6k_v2_2026-10-28_confusion_matrix_normalized.png)
![eval_grocereye_p6k_on_products6k_v2_2026-10-28_confusion_matrix](eval_grocereye_p6k_on_products6k_v2_2026-10-28_confusion_matrix.png)

Ultralytics confusion-matrix PNGs are computed at its val conf (0.001), so they show more low-confidence detections than the table above.

## Speed (ms per image, this machine)

preprocess 0.9, inference 2.5, loss 0.0, postprocess 1.3
