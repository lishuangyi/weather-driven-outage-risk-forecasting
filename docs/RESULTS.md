# Results

All scores use the challenge formula on our EAGLE-I truth: per task, pooled over counties, architectures, issues and
leads, 100 × (0.75 × normalised AUPRC + 0.25 × (1 − MAE)) on the risk scale. They are local scores, not results
certified by the organisers. The benchmark tables are generated from a versioned run report (section 10), and every
table can be recomputed with the SQL warehouse ([SQL.md](SQL.md)):

```bash
make predict warehouse                                       # June–August 2025
python -m outage_forecast.warehouse query sql/analysis/model_vs_persistence.sql
```

Persistence holds the latest visible observation for every lead. It is the natural baseline for outages, which
usually last longer than a few hours.

## 1. Headline

<!-- run-report:headline -->
| Window (UTC issue time) | Role of the window in development | Task A model / persistence | Task B model / persistence |
|---|---|---:|---:|
| 2025-06-01 to 2025-08-31 | member and weight selection | **54.45** / 38.69 | **75.13** / 65.88 |
| the other 9 months of 2025 | retrospective; January–May informed one fallback decision | **55.00** / 46.92 | **72.75** / 66.59 |
| all of 2025 | | **54.88** / 43.92 | **73.44** / 66.33 |
<!-- /run-report:headline -->

The first three rows use the full-year 2025 inputs. The last row is the June–August package that `make predict`
uses: its truth ends on 31 August, so the last issues lose part of their targets. The breakdowns in section 2 use
that package. All rows are generated from the run report (section 10).

The gain over persistence is smaller on the retrospective months: Task A +8.1 against +15.8 in the selection window,
Task B +6.2 against +9.3. Part of the difference comes from the selection itself and part from the season: summer has
more storm-driven outages, and those are where the model gains most. With one year of data these two effects cannot
be separated. On June–August the gain comes mostly from ranking (normalised AUPRC: A 0.44 against 0.24 for
persistence, B 0.70 against 0.58). The risk MAE improves as well (A 0.134 against 0.160, B 0.083 against 0.100).

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

Rockingham's scores rest on far fewer cases than the others: only about a quarter (A) to a third (B) of its requests
have truth (section 8). The gains are largest where outages come and go with storms (Wayne, Harris, Rockingham). They are smallest where
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

<!-- run-report:reference -->
| Evaluation | Reference | Persistence | Frozen ensemble |
|---|---:|---:|---:|
| 2024 (chronological test), A / B | 55.01 / 72.54 | 49.75 / 69.56 | in its training data |
| 2025 full year (retrospective) | 44.30 / 68.87 | 43.92 / 66.33 | 54.88 / 73.44 |
<!-- /run-report:reference -->

These are v2 results, after a fix on 2026-11-18. The v1 results were withdrawn because missing labels had been read as
zeros. [TRAINING.md](TRAINING.md) has the protocol, the selection table, the monthly results and the revision
history.

## 6. Calibration and events

The predicted risk is calibrated below 0.5 and over-confident above it. In the top bin, Task A predicts 0.98 and
0.64 of those cases are events; Task B predicts 0.99 and 0.88 are events (`sql/analysis/reliability.sql`). All five
Miami-Dade peaks and four of the five Wayne peaks got risk ≥ 0.97 from the issue 0–24 h earlier. The small Los
Angeles peaks were missed (`sql/analysis/largest_events.sql`).

## 7. Example: Wayne County storm, June 2025

![48-hour forecasts against observed outages, Wayne County, 26–30 June 2025](assets/event_wayne_2025-06.png)

This is a selected case, not a typical one. Two Task A forecasts for Wayne County (Detroit) are drawn against the
observed outage share, on a log scale; the band marks the four architecture thresholds. The forecast issued on
27 June at 00:00 UTC rises into the risk band about a day before the storm peak, which reaches about 2.6 % of
customers. The forecast issued on 28 June follows the restoration but decays more slowly than the real recovery.
The figure is made by `scripts/make_figures.py`.

## 8. Coverage: which requests can be scored

The model and persistence are scored on one cohort: the requests whose truth is available and valid. Requests
without truth are excluded and counted (`sql/analysis/coverage.sql`; rules in [SQL.md](SQL.md#scoring-one-cohort-for-every-model)).

<!-- run-report:coverage -->
| 2025, full year | Requested | Scored | Share scored | Lowest month |
|---|---:|---:|---:|---|
| Task A, all counties | 350,400 | 294,092 | 83.9 % | |
| Task B, all counties | 2,102,400 | 1,809,572 | 86.1 % | |
| Task A, Rockingham NH | 70,080 | 17,356 | 24.8 % | 15.3 % (February) |
| Task B, Rockingham NH | 420,480 | 140,460 | 33.4 % | 22.0 % (January) |
| other four counties | | | 97.9–99.9 % | 90.1 % (Miami-Dade A, March) |

No case is excluded for invalid truth. For June–August 2025, 86.1 % (A) and 88.4 % (B) of the requests are scored.
<!-- /run-report:coverage -->

Coverage depends on the outage feed: Rockingham's EAGLE-I series has long gaps, which also explains its stale history
at issue time (`sql/analysis/data_freshness.sql`).

## 9. Uncertainty: paired block bootstrap

The question is whether the gain over persistence holds beyond the particular days of 2025. The analysis was fixed
in code before it was run (`outage_forecast.evaluation`, commit `5872df1`; the results are in a later commit):

- **Primary statistic.** Task A risk MAE of the model minus that of persistence, on the 2025 scoring cohort.
  Negative means the model is better.
- **Resampling.** Moving blocks of 7 consecutive UTC issue days. All counties, architectures and leads of a day stay
  together, and the model and persistence always see the same cases. 2,000 resamples with seed 0, and a 95 %
  percentile interval.
- **Sensitivity.** 3- and 14-day blocks.
- **Secondary statistic.** The full challenge score (model minus persistence), recomputed in every resample, because
  AUPRC cannot be split into per-case losses. 7-day blocks and 1,000 resamples.

<!-- run-report:bootstrap -->
| 2025, 365 issue days | Cases | Risk MAE, model / persistence | MAE difference, 95 % CI | Score difference, 95 % CI |
|---|---:|---|---|---|
| **Frozen ensemble, Task A** (primary) | 294,092 | 0.104 / 0.122 | **−0.018 [−0.028, −0.007]** | +11.0 [+5.6, +16.1] |
| Frozen ensemble, Task B | 1,809,572 | 0.067 / 0.078 | −0.012 [−0.015, −0.008] | +7.1 [+5.2, +9.0] |
| Reference model, Task A | 294,092 | 0.124 / 0.122 | +0.003 [−0.007, +0.013] | +0.4 [−3.4, +4.0] |
| Reference model, Task B | 1,809,572 | 0.074 / 0.078 | −0.004 [−0.006, −0.002] | +2.5 [+1.8, +3.5] |

With 3- and 14-day blocks, the MAE intervals of every row keep the same sign as with 7-day blocks.
<!-- /run-report:bootstrap -->

- **Frozen ensemble.** Both tasks beat persistence; neither interval comes close to zero. The Task A risk error is
  about 15 % lower.
- **Reference model, Task B.** It also beats persistence.
- **Reference model, Task A.** It cannot be told apart from persistence. Its interval includes zero for both
  statistics, so the small positive score difference in [TRAINING.md](TRAINING.md) is not evidence of skill.

**What the intervals do not cover.** They describe sampling variation across days within 2025. They do not cover
differences between years, counties or data sources. For the frozen ensemble, June–August 2025 was also its selection
window. The full results, including the bootstrap standard deviations, are in
`artifacts/evaluation/bootstrap_2025_*.json`.

## 10. Run report

The tables between `run-report` markers in this file, the README, [TRAINING.md](TRAINING.md) and
[MODEL_CARD.md](MODEL_CARD.md) are written by `python -m outage_forecast.run_report render` from one versioned report,
[`artifacts/runs/benchmark.json`](../artifacts/runs/benchmark.json). The evaluations are listed in
[`configs/benchmark.json`](../configs/benchmark.json), and `make report` runs them again; it needs the inputs and
prediction files named there. For each evaluation the report records:

- the predictions file, the input directory, the window of UTC issue times and its role in development;
- requested, scored and excluded cases, and coverage per county with the lowest month;
- the pooled cases, events, normalised AUPRC, risk MAE and score of the model and of persistence on one cohort.

It also records the provenance of the run:

- the git commit; the committed report must come from a clean checkout;
- the SHA-256 of every input file (checked against the `SHA256SUMS` of each input), prediction file and model file,
  the release manifest and the bootstrap results;
- the recipes and seeds: reference trainer seed 42, bootstrap seed 0; inference has no randomness;
- the machine: CPU, memory, GPU, Python and package versions.

`tests/test_run_report.py` runs in CI and fails when a generated table no longer matches the committed report, or
when a bootstrap file changed after the report was made.
