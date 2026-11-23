# Reference model: training from scratch

The frozen ensemble (`models/`, see [MODEL.md](MODEL.md)) was built with the original experiment code. That code
spans many scripts and caches and is not part of this repository. `outage_forecast.reference` is a clean
replacement that you can run end to end: one feature table, one fixed recipe and a chronological protocol.

```bash
python scripts/download_artifacts.py training full-2025     # 2022–2025 full-year inputs (≈1.1 GB)
make train                                                  # ≈3 min on a GPU; writes models/reference/
python -m outage_forecast.reference predict --input-dir data/inputs/input_2025 --output out/reference.csv
```

The saved models are in release v1.2.0 (`python scripts/download_artifacts.py reference`). The full training report,
with the selection table, scores and feature importance, is
[`artifacts/reference/reference_v2.json`](../artifacts/reference/reference_v2.json).

## Data

The training set is the warehouse table `gold.examples` ([SQL.md](SQL.md)). It has one row per (task, county, issue,
target), 79 features computed as of the issue time, and the label:

| Group | Features |
|---|---|
| set-up and calendar | lead, target hour and day of year (sine/cosine), county index, log customers served |
| outage history | the 30 history columns (latest value, lags, rolling mean/max/min, slopes, missing count) |
| weather | mean and max of 14 HRRR variables at the end of the target hour, from the latest run visible at T; age of that run |
| radar | maximum reflectivity exposure over the last 1 h / 6 h / 24 h, mean 35 dBZ coverage over 24 h |
| label | `truth_x` from `gold.labels` (Task A: mean of 4 quarter-hours; Task B: value at T or held 15-minute value) |
| label metadata | `truth_published_s`: when the revisions behind the label were published; used for purging, never a feature |

Each year has 87,600 Task A examples (48 leads) and 525,600 Task B examples (72 leads). Between 12 and 17 % have no
valid label, because one of the needed observations is missing or blank, and are left out: 73–75 thousand labelled Task A
examples and 450–463 thousand Task B examples per year. Only the five forecast counties have outage records in these
inputs; the frozen ensemble was also trained on 35 neighbouring counties.

## Recipe and protocol

XGBoost with a Tweedie objective (variance power 1.3), η = 0.05, depth 7, min child weight 10, row and column
subsampling 0.8 and seed 42. These values are fixed and were not tuned. Two variants are compared:

- **plain**: Tweedie regression of the outage share.
- **anchored**: the same, with log(persistence forecast) as base margin. The trees then learn a multiplicative change
  to "the latest observation stays", which is the natural null model for outages.

| Stage | Fit | Evaluate | Cut-off of the fit / selection set | Purpose |
|---|---|---|---|---|
| 1. selection | 2022 | 2023 | 2023-01-01 / 2024-01-01 | choose the variant and number of rounds per task, **by the challenge score** |
| 2. test | 2022–2023 | 2024 | 2024-01-01 | first look at 2024 |
| 3. final | 2022–2024 | 2025 | 2025-01-01 | the saved model; 2025 is a retrospective evaluation |

A set with cut-off C keeps only examples whose target **and** label publication time are before C. A target near the
end of a year, or a correction published later, therefore cannot reach the fit or the selection.
`tests/test_reference.py` checks this with a late correction, a blank revision and boundary-crossing targets. All
scores below come from the SQL metric (`sql/scoring`), with persistence scored on the same cases as each model.

### Selection on 2023 (challenge score; persistence: A 51.78, B 74.10)

<!-- run-report:selection -->
| Rounds | 25 | 50 | 100 | 200 | 400 | 800 | 1200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A plain | 56.48 | 57.46 | 57.64 | 57.59 | 57.21 | 56.86 | 56.62 |
| A anchored | 56.16 | **57.99** | 57.81 | 57.43 | 56.99 | 56.71 | 56.50 |
| B plain | 75.29 | 76.19 | 76.17 | 76.09 | 76.02 | 75.76 | 75.51 |
| B anchored | 76.13 | **76.76** | 76.49 | 76.50 | 76.50 | 76.34 | 76.15 |
<!-- /run-report:selection -->

The selection chose the anchored variant with 50 rounds for both tasks. Longer training hurts, so the best model is a
small correction to persistence.

## Results

<!-- run-report:reference -->
| Evaluation | Reference | Persistence | Frozen ensemble |
|---|---:|---:|---:|
| 2024 test, Task A / B | **55.01 / 72.54** | 49.75 / 69.56 | (trained on 2024) |
| 2025 full year | 44.30 / 68.87 | 43.92 / 66.33 | **54.88 / 73.44** |
| 2025 June–August | 42.62 / 69.19 | 38.69 / 65.88 | **54.45 / 75.13** |
<!-- /run-report:reference -->

- **2024 test.** The reference model beats persistence by 5.3 (A) and 3.0 (B).
- **2025, Task B.** +2.5 over persistence. It beats persistence in 10 of 12 months; January (−0.2) and May (−0.5) are
  roughly level. It is 4.6 points below the frozen ensemble.
- **2025, Task A.** +0.4 over persistence for the year. This is within noise: the paired block-bootstrap interval is
  [−3.4, +4.0] ([RESULTS.md](RESULTS.md#9-uncertainty-paired-block-bootstrap)), so on 2025 the model cannot be told
  apart from persistence. It wins 10 of 12 months by +0.7 to +9.4, and loses January
  (−8.8) and February (−15.6). Those are the months of the multi-day Los Angeles wildfire outages, which reached up to
  46 % of customers. During such an event a held observation is nearly perfect, while a model that forecasts each
  hour separately drifts back towards typical values.
- **The gap to the frozen ensemble** is 10.6 (A) and 4.6 (B) on 2025. It measures what the ensemble's extra parts add:
  event-duration features, a GRU over the 7-day sequence, a tail model for large outages, and eight times more training
  counties. The January–February failure is exactly the case the event features and the GRU state are built for.

Most important features (total gain): for Task A the last hour of outage history (rolling max and mean, latest value,
15–30 min lags), then snow, pressure and radar; for Task B the latest value and recent rolling means, then
precipitation, gusts and pressure.

## Revision history

- **v1 (2026-11-17), withdrawn.** It had two errors:
  - **NULL read as zero.** DuckDB returns a column with NULLs as a masked numpy array. Building the polars frame
    dropped the mask, so every missing label and missing feature became 0. In 2024, 76,656 of 614,880 labels were
    missing and were trained on as "no outage".
  - **No cut-off for selection.** The 2023 selection set had no cut-off at 2024-01-01.

  v1 reported 2024 A 53.43 / B 73.04 and 2025 A 42.60 / B 71.12.
- **v2 (2026-11-18), current.** NULLs are kept (`reference.fetch`). Every fit and selection set is cut off by target
  and label publication time. Persistence is scored separately for each predictions file. After the re-run:
  - Task A improved in both test years (2024 +1.6, 2025 +1.7).
  - Task B dropped: −0.5 on 2024 and −2.3 on 2025.

  The numbers are reported as they came out; nothing was re-tuned.
- **After seeing 2025.** The selection originally used early stopping on the Tweedie log-likelihood with the plain
  variant only (2025: A 41.97, B 70.48). After seeing 2025 it was changed to the challenge score with two variants. The
  choice is still made on 2023 alone, but 2025 is therefore not untouched for this recipe and is reported as a
  retrospective evaluation.

GPU training is reproducible with the fixed seed. CPU and GPU runs differ slightly.

## Possible next steps

- Event-state features in SQL (duration of the current outage, time since its peak) for multi-day events.
- Training on the 35 neighbouring counties as well. This needs their outage records in the inputs.
- Calibration of the risk (see `sql/analysis/reliability.sql`).
