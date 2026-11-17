# Results

All scores use the challenge formula on our EAGLE-I truth: per task, pooled over counties, architectures, issues and
leads, 100 × (0.75 × normalised AUPRC + 0.25 × (1 − MAE)) on the risk scale. They are local scores, not results
certified by the organisers. Every table can be recomputed with the SQL warehouse ([SQL.md](SQL.md)):

```bash
make predict warehouse                                       # June–August 2025
python -m outage_forecast.warehouse query sql/analysis/model_vs_persistence.sql
```

Persistence holds the latest visible observation for every lead. It is the natural baseline for outages, which
usually last longer than a few hours.

## 1. Headline

| Window | Use of the window | Task A model / persistence | Task B model / persistence |
|---|---|---:|---:|
| June–August 2025 | **model selection** (the evaluation window of the challenge) | **54.52** / 38.82 | **75.13** / 65.88 |
| the other 9 months of 2025 | neither training nor selection | **55.00** / 46.92 | **72.75** / 66.59 |
| all of 2025 | | **54.88** / 43.92 | **73.44** / 66.33 |

The unseen months score about the same as the selection window, so the selection did not overfit that window. On
June–August the gain comes mostly from ranking (normalised AUPRC: A 0.44 against 0.24 for persistence, B 0.70
against 0.58). The risk MAE improves as well (A 0.134 against 0.160, B 0.083 against 0.100).

## 2. June–August 2025 broken down

These breakdowns are diagnostics. AUPRC does not average across groups, so the rows do not add up to the pooled score.

**By lead** (event rate 9–15 % in every band)

| Task A lead | 1–6 h | 7–12 h | 13–24 h | 25–48 h |
|---|---:|---:|---:|---:|
| score | 77.94 | 64.46 | 51.54 | 45.36 |

| Task B lead | 5–15 min | 20–60 min | 65–180 min | 185–360 min |
|---|---:|---:|---:|---:|
| score | 98.38 | 89.05 | 76.37 | 68.44 |

**By county** (model / persistence)

| County | Event rate (A) | Task A | Task B |
|---|---:|---:|---:|
| Wayne, MI (Detroit) | 16.4 % | 58.39 / 34.31 | 79.57 / 73.23 |
| Harris, TX (Houston) | 10.4 % | 39.08 / 24.12 | 53.62 / 42.09 |
| Rockingham, NH | 12.8 % | 37.29 / 22.82 | 51.13 / 36.25 |
| Los Angeles, CA | 3.5 % | 27.39 / 25.28 | 41.49 / 38.11 |
| Miami-Dade, FL | 21.6 % | 71.22 / 70.09 | 88.27 / 86.18 |

The gains are largest where outages come and go with storms (Wayne, Harris, Rockingham). They are smallest where
outage levels persist (Miami-Dade) or events are rare and small (Los Angeles).

**By architecture**: Task A ranges from 53.02 (ARC01, x₀ = 0.100 %) to 55.78 (ARC04, x₀ = 0.160 %), and Task B from
72.97 to 77.46. Lower thresholds turn more small outages into events, which makes them harder to rank.

## 3. 2025 month by month (frozen ensemble against persistence)

| Month | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task A gain | −0.2 | −4.3 | +17.9 | +5.3 | +5.5 | +14.1 | +12.7 | +11.6 | +10.4 | +22.4 | +12.1 | +18.3 |
| Task B gain | +1.9 | +2.4 | +2.2 | +2.0 | +1.8 | +8.7 | +8.5 | +8.6 | +10.3 | +11.7 | +11.7 | +9.7 |
| event rate (A) | 22 % | 14 % | 8 % | 5 % | 7 % | 21 % | 11 % | 7 % | 6 % | 7 % | 5 % | 6 % |

The model beats persistence in every month for Task B, and in 10 of 12 months for Task A. The two exceptions are
January and February, the months of the multi-day Los Angeles wildfire outages, when holding the last value is
nearly perfect.

## 4. Robustness (June–August 2025)

| Condition | Task A | Task B |
|---|---:|---:|
| all inputs | 54.52 | 75.13 |
| no radar (fallback models) | 53.79 | 74.65 |
| only the 5 basic weather fields, no radar | 42.27 | 72.86 |
| outage records 3 min late, weather runs 4 h late | 49.77 | 72.97 |
| no outage history at all | 33.04 | 24.72 |

See [MISSING_INPUTS.md](MISSING_INPUTS.md) for the fallback routes and [TEMPORAL_CORRECTNESS.md](TEMPORAL_CORRECTNESS.md)
for the delay experiment.

## 5. Reference model trained from scratch

| Evaluation | Reference | Persistence | Frozen ensemble |
|---|---:|---:|---:|
| 2024 (test year), A / B | 53.43 / 73.04 | 49.75 / 69.56 | in its training data |
| 2025 full year | 42.60 / 71.12 | 43.92 / 66.33 | 54.88 / 73.44 |

Protocol, selection table and the analysis of the gap are in [TRAINING.md](TRAINING.md).

## 6. Calibration and events

The predicted risk is calibrated below 0.5 and over-confident above it. In the top bin, Task A predicts 0.98 and
0.64 of those cases are events; Task B predicts 0.99 and 0.88 are events (`sql/analysis/reliability.sql`). All five
Miami-Dade peaks and four of the five Wayne peaks got risk ≥ 0.97 from the issue 0–24 h earlier. The small Los
Angeles peaks were missed (`sql/analysis/largest_events.sql`).
