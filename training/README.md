# Model training (PRD Section 10)

All commands run from the repo root with the venv active (`pip install -e .[training,dev]`;
`pip install albumentations` too, or blur augmentation is skipped). Every script has `--help`.

| Where | What |
|---|---|
| `data/grocereye/` | converted Freiburg / 5K Groceries dataset, `data.yaml`, `split.json` (gitignored) |
| `data/products6k/` | converted Products-6K, `data.yaml`, `manifest.json`, `boxes.json`, `review/` (gitignored) |
| `data/raw/<session>/` | photos from `capture.py` (gitignored) |
| `runs/` | Ultralytics run folders (gitignored) |
| `models/<name>/best.pt` | published weights (gitignored), next to `eval_report.md` (tracked) |
| `reports/` | `eval_<name>_<date>.md`, `threshold_<name>_<date>.md` |

## Stage A: reproduce GrocerEye (TR-1 to TR-4)

```
python training/get_grocereye_data.py                 # ~375 MB download + VOC -> YOLO
python training/train.py --name grocereye_a           # yolo11n.pt (COCO), imgsz 320, 100 epochs, patience 20
python training/evaluate.py --weights models/grocereye_a/best.pt
```

- Data comes from github.com/aleksandar-aleksandrov/groceries-object-detection-dataset, the set
  GrocerEye links to. If the download fails, use *Code -> Download ZIP* on that page and run
  `get_grocereye_data.py --from-archive <zip>` (or `--from-dir <folder>`).
- The split is our own stratified 80/10/10 (seed 0), recorded in `split.json` and in every eval
  report. GrocerEye's own 80/20 split was never published.
- `config/classes.yaml` is rewritten from the annotations. Any spoken names you edited by hand are kept.
- No local GPU? Use `training/train_colab.ipynb`.
- Smoke test first: `python training/train.py --name smoke --epochs 1 --fraction 0.05`.
- Target (NFR-5): test mAP50 >= 0.85, which matches or beats GrocerEye's 84.59%.

## Extra data: Products-6K (optional, between Stage A and Stage B)

[Products-6K](https://zenodo.org/records/4428917) is a grocery dataset from the Greek supermarket
Masoutis, made for visually impaired shoppers: studio photos of 500 products plus phone photos of
104 of them held in a hand inside the store. It has **no boxes** and its labels are Greek category
text, so `get_products6k_data.py` converts it:

- **Classes:** each product's category ("Καφές φίλτρου" = filter coffee) is mapped to one of our 25
  ids by the keyword rules in `training/products6k_map.yaml`. About 210 of the 512 public products
  map; the rest (detergent, biscuits, beer, ...) are left out. `--inspect` prints the full table.
- **Boxes:** studio photos have a plain white background, so the box is everything that is not
  background (exact, no AI). In-store photos and messy backdrops use Grounding DINO
  (`pip install transformers`; GPU recommended, so the Colab notebook is the easy route).
- **Split:** by product, stratified by class, 70/15/15 (seed 0), so photos of one product never
  land in both train and test. Near-duplicate photos of one product are dropped.
- **Check:** look at `data/products6k/review/sheet_*.jpg` (least certain boxes first). Fix or skip
  a photo in `data/products6k/boxes.json` and rerun.

```
python training/get_products6k_data.py --inspect        # what maps to what; writes nothing
python training/get_products6k_data.py                  # uses data/Candidate_Set (+ data/Query_Set)
python training/train.py --name grocereye_p6k --model models/grocereye_a/best.pt \
    --data data/grocereye/data.yaml --data data/products6k/data.yaml --epochs 60
python training/evaluate.py --weights models/grocereye_p6k/best.pt --data data/products6k/data.yaml
```

Extract `Candidate_Set.rar` (and `Query_Set.rar`, 2.7 GB) into `data/` first, or let the script
fetch them with `--download all`. On Colab use `training/train_products6k_colab.ipynb`. License:
CC BY-NC-ND 4.0, so the converted copy must not be shared (it stays in gitignored `data/`).

## Extra class: the shopper's hand (optional, after Products-6K)

The long-term goal is to guide the hand to the item, so the detector also learns a `hand` class.
It is the **last** class (id 25) in `config/classes.yaml` and `data/grocereye/data.yaml`, so the
25 grocery ids are unchanged. The guidance engine never speaks it as a found, wrong or identified
item (`guidance.non_item_classes` in `config/server.yaml`).

Labels come from `get_hands_data.py`: Grounding DINO boxes the hand in every Products-6K in-store
photo (the Query_Set, where a product is always held in a hand), appends those boxes to the
converted set's labels, and adds the in-store photos the product conversion left out as hand-only
images. Freiburg photos have no hands, so they are the "no hand here" examples.

```
python training/get_hands_data.py --device cuda      # data/products6k_v2 -> data/products6k_hands
python training/train.py --name grocereye_hand --model models/grocereye_p6k2/best.pt     --data data/grocereye/data.yaml --data data/products6k_hands/data.yaml --epochs 60
python training/evaluate.py --weights models/grocereye_hand/best.pt --data data/products6k_hands/data.yaml
```

Check `data/products6k_hands/review/sheet_*.jpg`; to drop a photo's hand boxes set `"skip": true`
for it in `hand_boxes.json` and rerun. A dedicated egocentric hand dataset (EgoHands, 100DOH) would
add far more variety than the ~370 in-store photos; the converter would only need a second source.

## Stage B: our own photos (TR-10 to TR-13)

1. **Pick 8 to 12 demo categories** (TR-10). Include look-alikes: pasta vs rice, coffee vs tea, juice vs soda.
2. **Capture** through the real camera if you can, so the photos match what the device sees:
   ```
   python training/capture.py --session kitchen_bright --interval 2          # laptop webcam
   python training/capture.py --backend picamera2 --headless --session pi_shelf_dim   # on the Pi
   ```
   Use a new `--session` for each place, lighting setup or day. Aim for **50 to 100 images per
   demo category**: shelf distance (0.5 to 2 m) plus close-ups, several US brands, bright/dim/mixed
   lighting, and cluttered shelves with other products. About **10% should be background-only**
   photos (shelves with none of our categories) to cut false positives.
3. **Label** in Roboflow or Label Studio. Use the GrocerEye class names (`tomato_sauce`, `beans`, ...)
   and **export in YOLO format**. Background images get empty label files. Your export may hold only
   the demo classes or use a different class order. That is fine: `train.py` remaps the class ids
   to the GrocerEye order by name.
4. **Split by capture session, not at random** (TR-12). Put whole sessions into
   `data/ours/images/{train,val,test}` (with matching `labels/`). Near-duplicate frames from one
   session must never land in both train and test. Write `data/ours/data.yaml`:
   ```yaml
   train: images/train
   val: images/val
   test: images/test
   names: [beans, cereal, coffee, ...]   # the names/order of YOUR export
   ```
5. **Fine-tune on Freiburg plus our photos together**, starting from the Stage A weights (TR-13).
   Training on both keeps the model from forgetting the other categories:
   ```
   python training/train.py --name grocereye_b --model models/grocereye_a/best.pt \
       --data data/grocereye/data.yaml --data data/ours/data.yaml --epochs 60
   ```
6. **Evaluate on our held-out photos only** (NFR-6: per-class precision >= 0.90, recall >= 0.80):
   ```
   python training/evaluate.py --weights models/grocereye_b/best.pt --data data/ours/data.yaml \
       --stage b --classes pasta rice cereal coffee tea beans soda chips jam tomato_sauce
   ```
   The report names every class that misses a target. Collect more photos of those classes.

## Picking the find threshold (TR-14)

```
python training/pick_threshold.py --weights models/grocereye_b/best.pt --data data/ours/data.yaml \
    --split test --classes pasta rice cereal coffee tea --write
```

The script sweeps confidence from 0.20 to 0.95 and picks the lowest value where every listed class
has precision >= 0.90. `--write` stores that value as `guidance.conf_find` in
`config/server.yaml`. Only that one line changes and comments are kept. Use a held-out split the
model never trained on. A split you also used for early stopping is optimistic.

## Using the weights (TR-15)

`train.py` copies `best.pt` to `models/<name>/` and prints the lines to put in `config/server.yaml`:

```yaml
model:
  weights: models/grocereye_a/best.pt
  classes_file: config/classes.yaml
  imgsz: 320
```

Commit `models/<name>/eval_report.md`. Weights are gitignored, so share them through the team
drive or download them from Colab.
