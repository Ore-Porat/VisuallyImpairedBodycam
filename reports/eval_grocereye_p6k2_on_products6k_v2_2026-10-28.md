# Evaluation: grocereye_p6k2_on_products6k_v2 (2026-10-28)

- Weights: `models/grocereye_p6k2/best.pt`
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
| mAP50 | 0.7964 | GrocerEye 0.8459 | below baseline (-4.95 pts) |
| mAP50 (NFR-5) | 0.7964 | >= 0.85 | FAIL |
| mAP50-95 | 0.7580 | - | - |

GrocerEye's 84.59% is Darknet mAP@0.5 on its own (unpublished) random 20% test split, so the comparison is approximate.

## Per-class results

Ultralytics P/R are at the max-F1 confidence; operating-point P/R are at conf 0.50, IoU >= 0.5.

| class | images | boxes | AP50 | AP50-95 | P (val) | R (val) | P @op | R @op | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cake | 2 | 2 | 0.995 | 0.995 | 0.718 | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| candy | 4 | 4 | 0.845 | 0.845 | 0.764 | 0.822 | - | 0.000 | 0 | 0 | 4 |
| cereal | 9 | 9 | 0.973 | 0.907 | 0.876 | 0.786 | 0.857 | 0.667 | 6 | 1 | 3 |
| chips | 14 | 14 | 0.943 | 0.933 | 0.747 | 1.000 | 0.737 | 1.000 | 14 | 5 | 0 |
| chocolate | 12 | 12 | 0.557 | 0.554 | 0.591 | 0.667 | 0.625 | 0.417 | 5 | 3 | 7 |
| coffee | 8 | 8 | 0.995 | 0.995 | 0.481 | 1.000 | 0.667 | 1.000 | 8 | 4 | 0 |
| fish | 3 | 3 | 0.443 | 0.388 | 0.233 | 0.667 | 0.333 | 0.667 | 2 | 4 | 1 |
| flour | 5 | 5 | 0.866 | 0.866 | 0.777 | 0.800 | 1.000 | 0.400 | 2 | 0 | 3 |
| jam | 4 | 4 | 0.995 | 0.995 | 0.714 | 1.000 | 0.800 | 1.000 | 4 | 1 | 0 |
| juice | 3 | 3 | 0.218 | 0.169 | 0.011 | 0.030 | 0.333 | 0.333 | 1 | 2 | 2 |
| nuts | 2 | 2 | 0.497 | 0.472 | 0.478 | 1.000 | 0.000 | 0.000 | 0 | 1 | 2 |
| oil | 2 | 2 | 0.995 | 0.696 | 0.559 | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| pasta | 4 | 4 | 0.529 | 0.494 | 0.298 | 0.500 | 0.400 | 0.500 | 2 | 3 | 2 |
| rice | 5 | 5 | 0.995 | 0.995 | 0.728 | 1.000 | 0.714 | 1.000 | 5 | 2 | 0 |
| soda | 5 | 5 | 0.703 | 0.703 | 0.603 | 0.919 | 0.750 | 0.600 | 3 | 1 | 2 |
| spices | 2 | 2 | 0.995 | 0.846 | 0.750 | 1.000 | 1.000 | 1.000 | 2 | 0 | 0 |
| tomato_sauce | 5 | 5 | 0.795 | 0.795 | 0.952 | 0.800 | 1.000 | 0.400 | 2 | 0 | 3 |
| vinegar | 3 | 3 | 0.995 | 0.995 | 0.753 | 1.000 | 0.750 | 1.000 | 3 | 1 | 0 |

## Confusion matrix (conf 0.50, IoU >= 0.5, rows = true class)

| true \ pred | beans | cake | candy | cereal | chips | chocolate | coffee | corn | fish | flour | honey | jam | juice | milk | nuts | oil | pasta | rice | soda | spices | sugar | tea | tomato_sauce | vinegar | water | background |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beans | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| cake | . | **2** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| candy | . | . | . | . | 2 | 1 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 1 |
| cereal | . | . | . | **6** | . | 1 | 1 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 1 |
| chips | . | . | . | . | **14** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| chocolate | . | . | . | . | 1 | **4** | . | . | . | . | . | . | . | . | . | . | 1 | . | . | . | . | . | . | . | . | 6 |
| coffee | . | . | . | . | . | . | **7** | . | . | . | . | . | . | . | . | . | . | 1 | . | . | . | . | . | . | . | . |
| corn | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| fish | . | . | . | . | . | . | . | . | **2** | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 1 |
| flour | . | . | . | . | . | . | 1 | . | . | **1** | . | 1 | . | . | 1 | . | 1 | . | . | . | . | . | . | . | . | . |
| honey | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| jam | . | . | . | . | . | . | . | . | . | . | . | **4** | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| juice | . | . | . | . | . | . | . | . | . | . | . | . | **1** | . | . | . | . | . | 1 | . | . | . | . | 1 | . | . |
| milk | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| nuts | . | . | . | . | 2 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| oil | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | **2** | . | . | . | . | . | . | . | . | . | . |
| pasta | . | . | . | . | . | . | . | . | 2 | . | . | . | . | . | . | . | **2** | . | . | . | . | . | . | . | . | . |
| rice | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | **5** | . | . | . | . | . | . | . | . |
| soda | . | . | . | . | . | . | . | . | . | . | . | . | 1 | . | . | . | . | 1 | **2** | . | . | . | . | . | . | 1 |
| spices | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | **2** | . | . | . | . | . | . |
| sugar | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| tea | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| tomato_sauce | . | . | . | . | . | . | . | . | 1 | . | . | . | 1 | . | . | . | . | . | . | . | . | . | **2** | . | . | 1 |
| vinegar | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | **3** | . | . |
| water | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| background | . | . | . | 1 | . | 2 | 3 | . | 1 | 1 | . | . | . | . | . | . | 1 | . | 1 | . | . | . | . | . | . | . |

![eval_grocereye_p6k2_on_products6k_v2_2026-10-28_confusion_matrix_normalized](eval_grocereye_p6k2_on_products6k_v2_2026-10-28_confusion_matrix_normalized.png)
![eval_grocereye_p6k2_on_products6k_v2_2026-10-28_confusion_matrix](eval_grocereye_p6k2_on_products6k_v2_2026-10-28_confusion_matrix.png)

Ultralytics confusion-matrix PNGs are computed at its val conf (0.001), so they show more low-confidence detections than the table above.

## Speed (ms per image, this machine)

preprocess 0.8, inference 2.7, loss 0.0, postprocess 1.4
