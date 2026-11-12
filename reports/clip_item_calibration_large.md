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
  fast_item_min_score: 0.250
```

> **Warning:** the precision-first threshold accepts almost nothing
> (recall 0.074). That is safe but not usable: the shopper
> would mostly hear "I don't see it". The balanced alternative below is more
> honest about the trade-off; the real fix is better (more specific) prompts.

**1-item list, precision-first:** at `min_score = 0.250`, precision **0.934**, recall **0.074** (85 true, 6 false accepts, 1148 crops).

**1-item list, balanced (best F1):** at `min_score = 0.210`, precision **0.574**, recall **0.578** (663 true, 492 false accepts).

## 2-item lists (the Coke vs Sprite question)

- median head-to-head accuracy: **0.985**
- pairs at or above 0.90: **306 / 325**
- median 10th-percentile lead: **0.035**

Suggested `fast_item_margin`: the 10th percentile of the lead, floored at 0 and
capped at 0.10. Rejecting a true item when two are genuinely close is the safe
failure: the shopper hears nothing rather than the wrong name.

```yaml
model:
  fast_item_margin: 0.035
```

| pair | crops | accuracy | median lead | wrong winner |
|---|---|---|---|---|
| oil vs vinegar | 68 | 0.735 | 0.016 | 18 |
| candy vs sugar | 101 | 0.743 | 0.017 | 26 |
| cereal vs rice | 75 | 0.787 | 0.029 | 16 |
| milk vs sugar | 70 | 0.814 | 0.057 | 13 |
| chocolate vs cookies | 73 | 0.822 | 0.040 | 13 |
| flour vs sugar | 54 | 0.833 | 0.016 | 9 |
| soda vs vinegar | 79 | 0.848 | 0.028 | 12 |
| juice vs vinegar | 114 | 0.860 | 0.034 | 16 |
| juice vs soda | 123 | 0.862 | 0.037 | 17 |
| candy vs cookies | 67 | 0.866 | 0.036 | 9 |
| oil vs soda | 77 | 0.870 | 0.037 | 10 |
| candy vs flour | 87 | 0.874 | 0.064 | 11 |
| flour vs spices | 89 | 0.876 | 0.054 | 11 |
| chocolate vs sugar | 107 | 0.879 | 0.044 | 13 |
| coffee vs spices | 141 | 0.887 | 0.050 | 16 |

The table shows the 15 hardest pairs. A pair at 1.000 is reliably separable;
a pair near 0.500 cannot be told apart by these prompts at all.

## Reference: all 26 items competing

Top-1 accuracy when every item is a candidate: **0.745** (1148 crops).

This is *not* the shipped mode. It is here to show how much the short list helps,
and to compare against the previous all-classes pipeline.

## Threshold sweep (1-item list)

| min_score | precision | recall | true | false |
|---|---|---|---|---|
| 0.05 | 0.039 | 1.000 | 1148 | 28661 |
| 0.07 | 0.039 | 1.000 | 1148 | 28329 |
| 0.09 | 0.041 | 1.000 | 1148 | 26878 |
| 0.11 | 0.048 | 1.000 | 1148 | 22967 |
| 0.13 | 0.065 | 0.997 | 1145 | 16455 |
| 0.15 | 0.106 | 0.974 | 1118 | 9458 |
| 0.17 | 0.197 | 0.929 | 1066 | 4355 |
| 0.19 | 0.362 | 0.809 | 929 | 1637 |
| 0.21 | 0.574 | 0.578 | 663 | 492 |
| 0.23 | 0.773 | 0.294 | 338 | 99 |
| 0.25 | 0.934 | 0.074 | 85 | 6 |
| 0.27 | 1.000 | 0.011 | 13 | 0 |
| 0.29 | 1.000 | 0.001 | 1 | 0 |
| 0.31 | n/a | 0.000 | 0 | 0 |
| 0.33 | n/a | 0.000 | 0 | 0 |
| 0.35 | n/a | 0.000 | 0 | 0 |
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
| beans | 29 | 0.188 | 0.236 | 0.252 | 0.178 | 0.240 | 0.483 |
| cake | 30 | 0.176 | 0.198 | 0.233 | 0.172 | 0.240 | 0.067 |
| candy | 67 | 0.150 | 0.189 | 0.214 | 0.169 | 0.230 | 0.015 |
| cereal | 54 | 0.169 | 0.201 | 0.247 | 0.173 | 0.240 | 0.185 |
| chips | 40 | 0.191 | 0.220 | 0.253 | 0.169 | 0.240 | 0.225 |
| chocolate | 73 | 0.157 | 0.203 | 0.234 | 0.149 | 0.200 | 0.521 |
| coffee | 72 | 0.202 | 0.224 | 0.241 | 0.178 | 0.230 | 0.250 |
| corn | 32 | 0.219 | 0.245 | 0.284 | 0.178 | 0.230 | 0.781 |
| fish | 26 | 0.186 | 0.225 | 0.261 | 0.187 | 0.250 | 0.154 |
| flour | 20 | 0.171 | 0.217 | 0.235 | 0.181 | none | 0.000 |
| honey | 38 | 0.190 | 0.219 | 0.253 | 0.182 | none | 0.000 |
| jam | 90 | 0.190 | 0.224 | 0.241 | 0.167 | 0.210 | 0.744 |
| juice | 79 | 0.206 | 0.231 | 0.256 | 0.200 | 0.250 | 0.190 |
| milk | 36 | 0.191 | 0.215 | 0.247 | 0.183 | 0.240 | 0.167 |
| nuts | 29 | 0.183 | 0.213 | 0.235 | 0.170 | 0.230 | 0.207 |
| oil | 33 | 0.181 | 0.221 | 0.249 | 0.197 | 0.250 | 0.121 |
| pasta | 39 | 0.177 | 0.223 | 0.244 | 0.177 | 0.220 | 0.538 |
| rice | 21 | 0.176 | 0.214 | 0.232 | 0.172 | none | 0.000 |
| soda | 44 | 0.186 | 0.219 | 0.240 | 0.182 | 0.240 | 0.136 |
| spices | 69 | 0.162 | 0.196 | 0.232 | 0.175 | 0.240 | 0.014 |
| sugar | 34 | 0.182 | 0.200 | 0.237 | 0.190 | 0.240 | 0.118 |
| tea | 51 | 0.183 | 0.208 | 0.232 | 0.166 | 0.220 | 0.353 |
| tomato_sauce | 40 | 0.210 | 0.228 | 0.260 | 0.183 | 0.260 | 0.100 |
| vinegar | 35 | 0.169 | 0.198 | 0.222 | 0.198 | none | 0.000 |
| water | 67 | 0.193 | 0.225 | 0.244 | 0.161 | 0.210 | 0.776 |

## Confusion pairs (all items competing, above 0.21)

| true item | named as | count |
|---|---|---|
| soda | juice | 13 |
| spices | coffee | 6 |
| jam | honey | 5 |
| vinegar | oil | 5 |
| water | vinegar | 5 |
| candy | chips | 4 |
| rice | cereal | 4 |
| sugar | milk | 4 |
| sugar | flour | 4 |
| water | oil | 4 |

