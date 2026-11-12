# Implementation plan: list-only CLIP verification

**Status:** Phases 0–3 complete · **Date:** 2026-11-12
**Scope:** laptop server only. No Raspberry Pi changes, no retraining, no photo enrolment.

---

## 1. Locked decisions

These came out of the design discussion and are treated as requirements.

| # | Decision |
|---|---|
| D1 | **No generic classes as labels.** The detector never emits `soda`, `product`, `package` as an item. |
| D2 | **The shopping list is the only candidate set.** Find and Identify both score list items only. |
| D3 | **No whole-catalog / all-classes runtime matrix.** Ever. |
| D4 | **`DETECTOR_KEYWORDS` is deleted.** YOLO-World's box goes straight to CLIP. |
| D5 | **No per-item negative prompts.** You cannot enumerate Sprite for every Coke. |
| D6 | **No photo enrolment.** Text prompts only. |
| D7 | Users may add arbitrary items. A preset catalog supplies *good prompts*, not a label universe. |
| D8 | `DETECTOR_CLASSES` (`can`, `bottle`, `box`, `bag`, `jar`, `container`, `package`, `product`) stays — it is the box vocabulary, not a label. |
| D9 | Accuracy over recall: a false "found" is worse than "I don't see it". Bias thresholds high. |
| D10 | The shipped catalog contains **generic groceries only** (no brand presets). The web page provides a **custom-item editor** where the user adds an item and its prompts. |
| D11 | Identify with an empty list says an explicit phrase ("Your list is empty."), not silence. |

---

## 2. Why the current pipeline cannot do this

`FastGroceryDetector` does closed-set classification:

```python
probs = ((features @ text_features.T) * 100).softmax(dim=-1)   # N labels
best  = top1(probs)
```

Softmax **always** picks something. Consequences:

- With a 1-item list, softmax over one logit is `1.0` for every crop. Useless.
- With `[coke, sprite]`, a Pepsi is forced to one of them.
- Saying "no" requires a competitor to win, i.e. an enumerate-everything reject set — exactly what D5 forbids.

Today the `DETECTOR_KEYWORDS` map papers over this by rejecting cross-type labels. That is a precision hack for a 26-class catalog and is a no-op for any class not in the dict.

**The fix is to change the question.** Not *"which of N is this?"* but *"does this crop match item X?"*, scored independently per item. Then "not on the list" is simply a low score — no competitor, no reject encyclopedia.

---

## 3. Target architecture

```
                        ┌──────────────────────────────┐
  frame ──► YOLO-World ─►│ boxes + detector label       │
                        │ (can / bottle / box / ...)   │
                        └──────────────┬───────────────┘
                                       │ crop (10% margin, square pad)
                                       ▼
                        ┌──────────────────────────────┐
   catalog ──► PromptBank│ CLIP image embedding         │
   (presets +           │ cosine vs each list item     │
    user items)         │ (box-label-aware vectors)    │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │ choose_item():               │
                        │  score >= min_score          │
                        │  and margin over runner-up   │
                        └──────────────┬───────────────┘
                                       ▼
                            Detection(cls=<list id>, conf=cosine)

  hand ──► EgoHands ──────────────────────────────────► Detection("hand", prob)
```

No softmax. No reject class. No keyword gate. No catalog-wide matrix.

---

## 4. Data model

### 4.1 `slr/common/catalog.py` (new)

```python
@dataclass(frozen=True)
class CatalogItem:
    id: str                      # detector class id, e.g. "coke"
    spoken: str                  # what the voice says, e.g. "Coke"
    prompts: tuple[str, ...]     # curated or generated text prompts
    aliases: tuple[str, ...] = ()
    preset: bool = True          # False = user-created

class Catalog:
    def __init__(self, preset: dict[str, CatalogItem], user: dict[str, CatalogItem]): ...
    def resolve(self, name: str) -> CatalogItem      # id | alias | spoken | slug; creates user item if absent
    def item(self, item_id: str) -> CatalogItem | None
    def spoken(self, item_id: str) -> str            # falls back to prettified id
    def spoken_map(self) -> dict[str, str]           # for PhraseCatalog / web page
    def is_preset(self, item_id: str) -> bool
    def save_user(self) -> None                      # atomic write of data/catalog_user.yaml
```

`resolve()` never raises for an unknown name (that is the point); it creates a user item and persists it. It raises only for empty/whitespace or >40 chars after slugifying.

### 4.2 `config/catalog.yaml` (new, presets)

```yaml
# The menu of addable items, with good CLIP prompts. Prompts are averaged.
items:
  pasta:
    spoken: pasta
    prompts:
      - "a photo of a pasta package"
      - "a package of dry pasta"
  coke:
    spoken: Coke
    aliases: [coca_cola, coca-cola, coca cola]
    prompts:
      - "a can of Coca-Cola"
      - "a Coca-Cola can"
      - "a red can of Coca-Cola"
```

`prompts` is optional. If omitted, prompts are generated (§6).

**Backward compatibility:** the loader also accepts the legacy shape

```yaml
classes:
  pasta: pasta
```

and produces `CatalogItem(id="pasta", spoken="pasta", prompts=())`. So `config/classes.yaml` keeps working until the repo is migrated.

### 4.3 `data/catalog_user.yaml` (runtime, gitignored)

Same `items:` shape, written on the first free-text add. Merged over presets; a user entry with a preset id overrides that preset. Never shipped, never overwritten by updates.

### 4.4 Config key

`model.catalog_file: config/catalog.yaml`.
`paths.user_catalog: data/catalog_user.yaml`.
`model.classes_file` was removed in Phase 3 after the migration; a config that still sets it
now fails loudly (pydantic forbids unknown keys). The loader still accepts the old
`classes:` *file shape* (`config/classes.yaml`, `config/classes_coco.yaml`), which is a data
format, not a config key.

---

## 5. Scoring and acceptance

### 5.1 Prompt bank (`slr/server/fast_detector.py`)

```python
class PromptBank:
    """Text features for the current shopping list. Built on the detector thread."""
    ids: tuple[str, ...]                     # list item ids, stable order
    any: Tensor                              # (N, D) averaged over all prompts per item
    by_label: dict[str, Tensor]              # detector label -> (N, D) vectors for that container
```

- Every prompt is encoded, L2-normalized, averaged per item, re-normalized.
- `by_label` holds prompts generated from YOLO's own label (`a can of Coke`, `a bottle of Coke`, …). Curated prompts go into `any`.
- A crop whose YOLO label is `can` uses `by_label["can"]` when present, else `any`.
- Cache invalidation key: `tuple((item.id, item.prompts) for item in items)`. Rebuild only when the list membership or prompts change. Check-off does not rebuild.

### 5.2 Decision rule

```python
def choose_item(scores, ids, min_score, margin):
    if not ids:
        return None
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    if scores[best] < min_score:
        return None
    if len(ids) >= 2 and (scores[best] - scores[order[1]]) < margin:
        return None          # ambiguous: say nothing rather than guess
    return ids[best], float(scores[best])
```

- `n == 1`: absolute threshold only (a margin is meaningless).
- `n >= 2`: threshold **and** margin. This is the Coke-vs-Sprite case and CLIP's best case.
- Emitted `Detection.conf` = raw cosine, clamped to `[0, 1]`.

### 5.3 Confidence semantics change (important)

`Detection.conf` changes from softmax probability to **cosine similarity** (typical range 0.15–0.45). `hand` detections keep the EgoHands probability. Therefore:

| Setting | Old meaning | New value |
|---|---|---|
| `fast_clip_conf: 0.80` | softmax floor | **replaced** by `fast_item_min_score: 0.30` |
| `guidance.conf_find: 0.5` | probability | `0.30` (cosine) |
| `guidance.conf_identify: 0.6` | probability | `0.33` (cosine) |
| `guidance.conf_wrong_item: 0.7` | probability | `0.33` (cosine) |
| `guidance.hand_conf: 0.5` | probability | **unchanged** (different model, different scale) |

All of these are outputs of the calibration step (§9), not guesses. `slr/server/debug_view.py` prints the raw number, which is the right tuning surface.

Audit list for `.conf` consumers: `guidance.py:443,530,560,609,634,1162`, `debug_view.py:120,128`, `common/types.py:109` (serialization), `bench.py`. Only the four guidance thresholds need value changes.

### 5.4 Empty list

`set_items(())` → `detect()` skips product CLIP entirely and returns hands only. No crash, no bogus detections. Identify with an empty list falls through to the existing `not_sure` phrase.

---

## 6. Prompt construction

Priority order per item:

1. **Curated prompts** from the catalog entry (if any).
2. **Box-label templates** for each `DETECTOR_CLASSES` value that is plausibly a container:
   `a can of {spoken}`, `a bottle of {spoken}`, `a box of {spoken}`, `a bag of {spoken}`,
   `a jar of {spoken}`, `a package of {spoken}`.
3. **Fallback:** `a photo of {spoken}`.

For presets, ship curated prompts for the 26 existing groceries (rewritten to be specific: brand, colour and container where known). The box-label templates are what make free-text items usable — CLIP sees *"a can of Coke"*, not *"a photo of coke"* — and they use YOLO's own vocabulary, so nothing is hardcoded per item.

Encoding cost at build time: `N × (curated + 7)` texts. For a 10-item list this is ~100 short strings, tens of milliseconds, once per list change. Per-frame cost actually **drops**: scoring is against N items instead of 29 labels.

---

## 7. Configuration changes

```yaml
model:
  catalog_file: config/catalog.yaml
  fast_item_min_score: 0.30   # calibrated; absolute cosine floor (see §9)
  fast_item_margin: 0.02      # calibrated; required lead over the runner-up when 2+ items
  fast_prompt_from_box: true  # generate "a can of X" prompts from YOLO's label

paths:
  user_catalog: data/catalog_user.yaml
```

Removed: `fast_clip_conf` (semantics changed; keeping the name would silently mistune).
Added validators: `0 < fast_item_min_score <= 1`, `0 <= fast_item_margin <= 1`.

---

## 8. Implementation phases

Each phase leaves `pytest` green and the server runnable.

### Phase 0 — Catalog + measurement harness (no runtime change) — **done**

| Task | File |
|---|---|
| Add `CatalogItem`, `Catalog`, `load_catalog()` (both YAML shapes), slugify, alias match | `slr/common/catalog.py` (new) |
| `config/catalog.yaml` with curated prompts for all 26 presets + `hand` | `config/catalog.yaml` (new) |
| `load_classes()` becomes a thin wrapper over the catalog (id → spoken) | `slr/common/config.py` |
| Add `catalog_file`, `paths.user_catalog`; keep `classes_file` as a deprecated alias | `slr/common/config.py`, `config/server.yaml` |
| Calibration script (see §9) | `training/calibrate_items.py` (new) |
| Catalog tests (16) | `tests/test_catalog.py` (new) |
| Measured reports | `reports/clip_item_calibration.md`, `reports/clip_item_calibration_large.md` |

**Delivered:** all 328 existing tests still pass, plus 16 catalog tests. The server reads the new catalog and behaves exactly as before (the detector still uses its old prompt table until Phase 1). The measurements in §9.4 are the output that Phase 1 depends on.

### Phase 1 — Independent scoring (the behavioural change) — **done**

| Task | File |
|---|---|
| `PromptBank`, `score_crops()`, `choose_item()` | `slr/server/fast_detector.py` |
| Delete `resolve_label()`, `DETECTOR_KEYWORD_BYPASS`, `DETECTOR_KEYWORDS`, `CLASS_DESCRIPTIONS`, `REJECT_CLASSES` (accept path) | `slr/server/fast_detector.py`, `slr/server/grocery_taxonomy.py` |
| `detect()`: lazy `PromptBank` rebuild on the detector thread; skip CLIP when the list is empty | `slr/server/fast_detector.py` |
| `set_items()`; called after every list change (`_list_op`, ticker reload, startup) | `slr/server/main.py` |
| Optional per-item `min_score` in the catalog entry, global fallback | `slr/common/catalog.py`, `fast_detector.py` |
| Recalibrate `conf_find` / `conf_identify` / `conf_wrong_item` to cosine values | `slr/common/config.py`, `config/server.yaml` |
| Backbone decision: keep ViT-B/16 (L/14 is 4.4x slower and misses the 20 fps budget) | `config/server.yaml` |
| Rewrite `test_resolve_label_rules` as `choose_item` / prompt-bank tests | `tests/test_fast_detector.py` |
| End-to-end wiring test: list edits reach `set_items` | `tests/test_end_to_end.py` |

**Delivered:** 360 tests pass. Real-engine smoke test at 640x480: ~28 ms/frame for
YOLO + CLIP + EgoHands with a one-item list. A 25-image pasta sample gave 11/17 detections
on pasta photos and 0/15 on rice photos, which is the intended precision-first trade-off.

**Acceptance:**
- 2-item list: a near-tie yields `[]`; a clear winner is accepted.
- 1-item list: a matching crop at high similarity is accepted; a crop below the floor yields `[]`.
- Empty list: no product detections, hands still reported.
- 2-item head-to-head accuracy on GrocerEye val is at least the Phase-0 baseline (0.957 for B/16, 0.985 for L/14).
- `pytest` green.

### Phase 2 — Free-text items and the custom-prompt editor — **done**

| Task | File |
|---|---|
| `resolve()` creates + persists a user item; `save_user()` atomic write | `slr/common/catalog.py` |
| `set_prompts()` / `restore_prompts()`; presets are overridden, never edited | `slr/common/catalog.py` |
| `normalize_name` / `ShoppingList` accept a `Catalog` (dict mode kept) | `slr/server/session.py` |
| Free-text input + `<datalist>` of presets in the add form | `slr/server/list_web.py` |
| **Custom-prompt editor** at `/prompts`: one prompt per line, save + restore; refreshes the detector immediately | `slr/server/list_web.py`, `slr/server/main.py` |
| Mark list items that use generated prompts | `list_web.py`, `main.py` |
| `slr list add "Coca Cola"`; `slr list classes`; `slr list prompts <item>` | `slr/cli.py` |
| Update `unknown_items()` → `uncurated_items()` | `slr/server/session.py`, `main.py` |

**Delivered:** 377 tests pass (with Phase 3). Add a free-text item from the page or CLI, edit
its prompts, save, and the next frame uses them; restart the server and the item is still there.

**Acceptance:** add `Coke` from the web page with no restart → appears in the list, persists in `data/catalog_user.yaml`, is scored on the next frame, and the phrase "Coke found." renders. Edit its prompts on the page → the next frame uses the new prompts. Removing the list item does not delete the catalog entry.

**Why the prompt editor matters:** the Phase 0 numbers (§9.4) show that absolute 1-item verification depends heavily on how specific the prompt is. "a can of soda" cannot separate Coke from Sprite, but "a red can of Coca-Cola" can be much better. The editor is how a user supplies that specificity without touching a YAML file.

### Phase 3 — Docs, Identify polish, housekeeping — **done**

| Task | File |
|---|---|
| Document "only list items are named" (Find, FR-20, Identify) | `DECISIONS.md`, `README.md` |
| Document catalog format, free-text limits, threshold tuning | `README.md` |
| Identify with an empty list: reuse the `list_empty` phrase, no burst | `slr/server/guidance.py`, `tests/test_guidance.py` |
| Remove the deprecated `classes_file` config key | `slr/common/config.py`, `config/classes.yaml` |
| Update the legacy class-file comments | `config/classes.yaml` |

**Delivered:** Identify on an empty list says "Your shopping list is empty." in ON and OFF,
without starting a burst. `model.classes_file` is gone; the legacy `classes:` *file shape*
still loads (data compatibility), but an old config key is a startup error.

### Phase 4 — Optional, measurement-gated

Only if Phase 1 calibration shows poor precision on generic-looking packages:

- **Coarse-label specificity test.** Compare the item score against a baseline prompt derived from YOLO's label (`a photo of a can`) and require a lead `δ_coarse`. Generalises to any user item because the baseline comes from `DETECTOR_CLASSES`, not from a fixed item list. Config: `fast_item_coarse_margin` (default `0` = off).
- **Threshold adaptation from TOGGLE.** When the shopper confirms a found item, log the similarity; adapt `min_score` slowly. Off by default.

---

## 9. Calibration procedure

The dataset supports this: `data/remapped/grocereye/` has 990 val images with YOLO-format labels (class index, cx, cy, w, h), and `config/catalog.yaml` order defines the index mapping.

### 9.1 Script

```
python training/calibrate_items.py \
  --data data/remapped/grocereye --split val \
  --out reports/clip_item_calibration.md
```

Per labelled box: `prepare_crop(..., 0.10)`, embed, score against every catalog item's averaged prompt vector. The report measures the three decision rules separately, because they behave very differently:

| Section | What it models |
|---|---|
| **1-item list** | one item on the list: "is this crop that item?" against an absolute threshold |
| **2-item list** | two items: head-to-head accuracy and the lead the winner needs (the margin) |
| **all items competing** | the old 26-way problem, kept as a reference only |
| per item | own score distribution, other items' scores against this prompt, per-item threshold |
| confusion pairs | which items CLIP mixes up |

### 9.2 Brand test (the Coke vs Sprite answer)

GrocerEye has only a generic `soda` class, so it cannot answer the brand question. Add a second mode:

```
python training/calibrate_items.py --images data/brand_test \
  --out reports/brand_calibration.md
```

`data/brand_test/coke/*.jpg`, `data/brand_test/sprite/*.jpg`, 10–20 photos each, taken with the bodycam at shelf distance. This produces a real number instead of an estimate, and the same script can be re-run when prompts change. This is the gate for D6: if text-only cannot separate Coke from Sprite, the honest outcome is "it says not found", not a wrong answer — and that becomes a documented limitation, not a silent failure.

### 9.3 End-to-end check

Run `training/evaluate.py` (existing) before Phase 1 and after, same split, and record whether removing the keyword map cost precision. If it did, Phase 4's coarse-label margin is the remedy — not bringing the map back.

### 9.4 Phase 0 results (measured, 1148 crops, 26 items)

Full reports: `reports/clip_item_calibration.md` (ViT-B/16, the shipped model) and `reports/clip_item_calibration_large.md` (ViT-L/14, for comparison).

| Metric | ViT-B/16 | ViT-L/14 |
|---|---|---|
| 2-item head-to-head accuracy, median | **0.957** | **0.985** |
| 2-item lead, median of per-pair p10 | 0.019 | 0.035 |
| 1-item, precision-first (target 0.90) | 1.000 precision at **0.002 recall** | 0.934 precision at **0.074 recall** |
| 1-item, balanced (best F1) | 0.499 precision / 0.470 recall | 0.574 precision / 0.578 recall |
| all 26 competing, top-1 | 0.611 | 0.745 |

What this means:

1. **Relative comparison works.** When two items are on the list, CLIP picks the right one ~96% (B/16) or ~98.5% (L/14) of the time. The Coke-vs-Sprite case is the *easy* case. A margin of ~0.02 (B/16) or ~0.035 (L/14) is the right order of magnitude.
2. **Absolute 1-item verification is the hard case.** With one item there is no competitor, so the decision rests entirely on the absolute cosine score. On this category-level data, B/16 cannot reach 90% precision at any useful recall; L/14 reaches it only by rejecting ~93% of correct crops. This is the measured cost of D1/D2 (no generics, no reject encyclopedia).
3. **It is partly a property of the benchmark.** GrocerEye classes are broad and visually near-identical (beans/corn/fish are all cans; the label text distinguishes them, and CLIP does not read text). Brand-level prompts ("a red can of Coca-Cola") should separate better, but that needs the brand test (§9.2) to confirm.
4. **L/14 is a real improvement but does not fit the budget.** 182.6 ms for 20 crops versus 41.6 ms for B/16 on the target laptop (4.4×). At 20 fps the whole frame has 50 ms, so Phase 1 keeps **B/16** as the shipped backbone and leaves `fast_clip_model` as a one-line switch for future hardware.

Consequences for Phase 1:

- Ship the **margin rule**; it is the part that works.
- For a 1-item list, do **not** pretend a single global threshold is calibrated. Use the per-item thresholds from the report as the starting point, expose them in the catalog entry, and document that a 1-item list is the weakest mode.
- Expect the failure mode "says not found" rather than a wrong "found"; that is the D9 trade-off working.
- Re-run the brand test before claiming Coke vs Sprite works.

---

## 10. Test plan

| File | Coverage |
|---|---|
| `tests/test_catalog.py` (new) | legacy + new YAML shapes; preset/user merge precedence; alias resolution; slugify; empty/too-long rejection; atomic save; prompt fallback chain |
| `tests/test_fast_detector.py` | `choose_item`: threshold, margin, 1-item, empty, tie → `None`; averaging + normalization of prompts; `by_label` selection; `set_items` rebuild invalidation; no softmax (scores are cosines); `detect` emits only list ids + `hand` |
| `tests/test_session.py` | free-text add; legacy dict still works; `uncurated_items()`; no regression in ordering/status |
| `tests/test_list_web.py` | text input round-trip; non-preset hint; state polling unaffected |
| `tests/test_guidance.py` | recalibrated defaults; no behaviour change at high `conf`; empty-list search |
| `tests/test_config_and_clips.py` | new keys validated; removed key rejected; catalog loads |
| `tests/test_end_to_end.py` | fake detector records `set_items`; adding/removing a list item pushes a new prompt set |
| `tests/test_replay.py` | replay uses the catalog; recorded session unchanged |

Roughly 30–40 new/rewritten tests. Existing guidance tests inject `conf=0.9`, so they remain valid after the cosine switch; only the four config defaults change.

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Cosine threshold drifts across prompt sets | mis-tuned accept | Calibrate per §9; keep `min_score` in config, not code; debug view shows the raw score |
| Generic-looking packages score high on everything | false "found" | Margin rule (n≥2) + high threshold + Phase 4 coarse margin |
| Small crops (40×80 px) defeat fine-grained CLIP | Coke/Sprite unreliable | Raise `fast_min_box` from 24 to ~40 and measure; document the limit |
| **1-item lists have no competitor** (measured: B/16 fails the 0.90 precision target, L/14 only at 7% recall) | false "found" or mostly "not found" | Per-item thresholds; prefer L/14; document the limitation; the custom-prompt editor is the user-side fix |
| Free-text prompt quality varies wildly | item never detected | Box-label templates (§6); hint in UI; report per-item recall |
| Removing softmax changes `conf` everywhere | quiet mis-tuning | Explicit audit list (§5.3); all four guidance thresholds recalibrated together |
| Prompt-set update races with detection | stale or mixed scoring | Rebuild lazily on the detector thread from an immutable requested tuple; no CLIP calls off-thread |
| Deleting the keyword map costs precision | more false labels | Measure Phase 0 vs Phase 1; Phase 4 margin is the sanctioned remedy |
| Identify can only name list items | feature reduction | Documented in `DECISIONS.md`; accepted under D2 |
| An item YOLO never proposes (odd shape) | undetectable regardless of CLIP | Documented; the box vocabulary is the outer bound |
| `classes_file` configs in the wild | startup error | Removed in Phase 3 after the migration; the error is intentional (the key no longer exists) |

---

## 12. Rollout and rollback

- **No Pi deployment.** Everything here is server-side, so the device copy is untouched.
- Phase 0 is a no-op at runtime and can land alone.
- Phase 1 changes behaviour; keep it as one commit so `git revert` restores the softmax pipeline.
- Phase 2 is additive (free text) and independently revertible.
- Run `slr sim --camera 1 --debug` after each phase and confirm: ON guides, OFF stays silent, Identify works, list edits update the prompt set without restart.
- Keep `config/classes.yaml` in the tree until Phase 3 so the legacy loader path stays tested.

---

## 13. Decisions taken (was: open questions)

1. **Catalog contents:** generic groceries only. No brand presets. The web page gets a custom-item prompt editor (Phase 2) so a user can add "Coca-Cola" and its prompts without editing files. — answered
2. **Identify with an empty list:** speak "Your list is empty." — answered
3. **Curated prompts:** drafted in `config/catalog.yaml` for all 26 presets. They still need a sanity check against real shelf photos, and the Phase 0 report identifies the weakest ones. — drafted

Remaining unknown: whether brand-specific prompts separate Coke from Sprite well enough. That needs the `data/brand_test` photos (§9.2); no amount of prompt drafting can answer it.

---

## 14. Effort

| Phase | Work | Estimate |
|---|---|---|
| 0 | catalog module, YAML, calibration script, tests, measured reports | **done** (0.5 day) |
| 1 | prompt bank, scoring, wiring, recalibration, tests, real-engine smoke test | **done** (1 day) |
| 2 | free text, custom-prompt editor, web/CLI, persistence, tests | **done** (1.5 days) |
| 3 | docs, Identify polish, cleanup | **done** (0.5 day) |
| 4 | optional, only if calibration demands | 0.5–1 day |

Total to a shippable state (Phases 0–3): **~3.5 days**, plus the time to collect the brand test photos.
