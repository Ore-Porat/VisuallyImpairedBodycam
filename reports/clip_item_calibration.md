# CLIP item calibration: GrocerEye

Source: `data/remapped/grocereye (val)`  
Crops: **1148**  Items: **26**  
Precision target: **0.90**

Scores are raw CLIP cosine similarities, not probabilities.

A crop is accepted when its best item score is at least `min_score`; with 2+ items
it must also lead the runner-up by `margin`.

## Recommended settings

```yaml
model:
  fast_item_min_score: 0.360
```

> **Warning:** the precision-first threshold accepts almost nothing
> (recall 0.002). That is safe but not usable: the shopper
> would mostly hear "I don't see it". The balanced alternative below is more
> honest about the trade-off; the real fix is better (more specific) prompts.

**1-item list, precision-first:** at `min_score = 0.360`, precision **1.000**, recall **0.002** (2 true, 0 false accepts, 1148 crops).

**1-item list, balanced (best F1):** at `min_score = 0.290`, precision **0.499**, recall **0.470** (539 true, 541 false accepts).

## 2-item lists (the Coke vs Sprite question)

- median head-to-head accuracy: **0.957**
- pairs at or above 0.90: **258 / 325**
- median 10th-percentile lead: **0.019**

Suggested `fast_item_margin`: the 10th percentile of the lead, floored at 0 and
capped at 0.10. Rejecting a true item when two are genuinely close is the safe
failure: the shopper hears nothing rather than the wrong name.

```yaml
model:
  fast_item_margin: 0.019
```

| pair | crops | accuracy | median lead | wrong winner |
|---|---|---|---|---|
| flour vs sugar | 54 | 0.500 | 0.000 | 27 |
| oil vs vinegar | 68 | 0.559 | 0.005 | 30 |
| milk vs sugar | 70 | 0.700 | 0.027 | 21 |
| cereal vs sugar | 88 | 0.727 | 0.026 | 24 |
| oil vs soda | 77 | 0.753 | 0.024 | 19 |
| chocolate vs cookies | 73 | 0.781 | 0.028 | 16 |
| flour vs spices | 89 | 0.787 | 0.027 | 19 |
| fish vs spices | 95 | 0.789 | 0.028 | 20 |
| candy vs flour | 87 | 0.793 | 0.046 | 18 |
| flour vs pasta | 59 | 0.797 | 0.030 | 12 |
| cake vs cereal | 84 | 0.798 | 0.045 | 17 |
| juice vs vinegar | 114 | 0.798 | 0.027 | 23 |
| cake vs cookies | 30 | 0.800 | 0.014 | 6 |
| oil vs water | 100 | 0.800 | 0.028 | 20 |
| coffee vs spices | 141 | 0.801 | 0.034 | 28 |

The table shows the 15 hardest pairs. A pair at 1.000 is reliably separable;
a pair near 0.500 cannot be told apart by these prompts at all.

## Reference: all 26 items competing

Top-1 accuracy when every item is a candidate: **0.611** (1148 crops).

This is *not* the shipped mode. It is here to show how much the short list helps,
and to compare against the previous all-classes pipeline.

## Threshold sweep (1-item list)

| min_score | precision | recall | true | false |
|---|---|---|---|---|
| 0.05 | 0.038 | 1.000 | 1148 | 28700 |
| 0.07 | 0.038 | 1.000 | 1148 | 28700 |
| 0.09 | 0.038 | 1.000 | 1148 | 28700 |
| 0.11 | 0.038 | 1.000 | 1148 | 28699 |
| 0.13 | 0.038 | 1.000 | 1148 | 28695 |
| 0.15 | 0.039 | 1.000 | 1148 | 28644 |
| 0.17 | 0.039 | 1.000 | 1148 | 28113 |
| 0.19 | 0.043 | 1.000 | 1148 | 25803 |
| 0.21 | 0.053 | 0.994 | 1141 | 20232 |
| 0.23 | 0.083 | 0.978 | 1123 | 12351 |
| 0.25 | 0.153 | 0.897 | 1030 | 5713 |
| 0.27 | 0.294 | 0.733 | 841 | 2022 |
| 0.29 | 0.499 | 0.470 | 539 | 541 |
| 0.31 | 0.727 | 0.199 | 229 | 86 |
| 0.33 | 0.827 | 0.037 | 43 | 9 |
| 0.35 | 0.833 | 0.004 | 5 | 1 |
| 0.37 | n/a | 0.000 | 0 | 0 |
| 0.39 | n/a | 0.000 | 0 | 0 |
| 0.41 | n/a | 0.000 | 0 | 0 |
| 0.43 | n/a | 0.000 | 0 | 0 |
| 0.45 | n/a | 0.000 | 0 | 0 |
| 0.47 | n/a | 0.000 | 0 | 0 |
| 0.49 | n/a | 0.000 | 0 | 0 |

## Per item

`threshold` is the lowest score that reaches the precision target for that item on
its own (a 1-item list). `other p90` is how high other items' photos score against
this prompt: when it is close to the item's own median, the prompt is too generic.

| item | crops | own p10 | own median | own p90 | other p90 | threshold | recall |
|---|---|---|---|---|---|---|---|
| beans | 29 | 0.279 | 0.319 | 0.334 | 0.271 | none | 0.000 |
| cake | 30 | 0.260 | 0.270 | 0.287 | 0.244 | 0.280 | 0.233 |
| candy | 67 | 0.240 | 0.271 | 0.297 | 0.252 | 0.300 | 0.030 |
| cereal | 54 | 0.266 | 0.285 | 0.330 | 0.275 | 0.320 | 0.185 |
| chips | 40 | 0.268 | 0.296 | 0.318 | 0.254 | 0.320 | 0.100 |
| chocolate | 73 | 0.236 | 0.274 | 0.306 | 0.239 | 0.280 | 0.425 |
| coffee | 72 | 0.261 | 0.292 | 0.311 | 0.260 | 0.300 | 0.292 |
| corn | 32 | 0.298 | 0.316 | 0.352 | 0.264 | 0.320 | 0.438 |
| fish | 26 | 0.246 | 0.268 | 0.319 | 0.272 | 0.330 | 0.115 |
| flour | 20 | 0.268 | 0.292 | 0.312 | 0.271 | none | 0.000 |
| honey | 38 | 0.254 | 0.278 | 0.300 | 0.258 | none | 0.000 |
| jam | 90 | 0.249 | 0.298 | 0.322 | 0.254 | 0.300 | 0.456 |
| juice | 79 | 0.279 | 0.302 | 0.322 | 0.285 | none | 0.000 |
| milk | 36 | 0.260 | 0.279 | 0.301 | 0.257 | none | 0.000 |
| nuts | 29 | 0.263 | 0.278 | 0.302 | 0.258 | 0.310 | 0.069 |
| oil | 33 | 0.263 | 0.298 | 0.332 | 0.287 | 0.340 | 0.091 |
| pasta | 39 | 0.250 | 0.284 | 0.319 | 0.259 | 0.310 | 0.282 |
| rice | 21 | 0.252 | 0.271 | 0.302 | 0.254 | 0.310 | 0.095 |
| soda | 44 | 0.260 | 0.290 | 0.307 | 0.265 | 0.310 | 0.023 |
| spices | 69 | 0.238 | 0.264 | 0.295 | 0.258 | none | 0.000 |
| sugar | 34 | 0.238 | 0.256 | 0.279 | 0.260 | none | 0.000 |
| tea | 51 | 0.273 | 0.295 | 0.316 | 0.267 | 0.300 | 0.431 |
| tomato_sauce | 40 | 0.301 | 0.321 | 0.336 | 0.277 | 0.330 | 0.250 |
| vinegar | 35 | 0.243 | 0.261 | 0.303 | 0.272 | none | 0.000 |
| water | 67 | 0.271 | 0.304 | 0.318 | 0.242 | 0.290 | 0.731 |

## Confusion pairs (all items competing, above 0.29)

| true item | named as | count |
|---|---|---|
| soda | juice | 15 |
| vinegar | oil | 13 |
| water | oil | 11 |
| soda | oil | 6 |
| juice | oil | 5 |
| rice | cereal | 5 |
| sugar | flour | 4 |
| coffee | spices | 3 |
| nuts | beans | 3 |
| nuts | chips | 3 |

