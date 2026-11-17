# Reference model: training from scratch

The frozen ensemble (`models/`, see [MODEL.md](MODEL.md)) was built with the original experiment code. That code
spans many scripts and caches and is not part of this repository. `outage_forecast.reference` is a clean
replacement that a reader can run end to end: one feature table, one fixed recipe and a chronological protocol.

```bash
python scripts/download_artifacts.py training full-2025     # 2022–2025 full-year inputs (≈1.1 GB)
make train                                                  # ≈3 min on a GPU; writes models/reference/
python -m outage_forecast.reference predict --input-dir data/inputs/input_2025 --output out/reference.csv
```

The saved models are also in release v1.1.0 (`python scripts/download_artifacts.py reference`). The full training
report, including the selection table, scores and feature importance, is
[`artifacts/reference/reference_v1.json`](../artifacts/reference/reference_v1.json).

## Data

The training set is the warehouse table `gold.examples` ([SQL.md](SQL.md)). It has one row per (task, county, issue,
target) and 79 features, all computed as of the issue time, plus the label:

| Group | Features |
|---|---|
| set-up and calendar | lead, target hour and day of year (sine/cosine), county index, log customers served |
| outage history | the 30 history columns (latest value, lags, rolling mean/max/min, slopes, missing count) |
| weather | mean and max of 14 HRRR variables at the end of the target hour, from the latest run visible at T; age of that run |
| radar | maximum reflectivity exposure over the last 1 h / 6 h / 24 h, mean 35 dBZ coverage over 24 h |
| label | `truth_x` from `gold.labels` (Task A: mean of 4 quarter-hours; Task B: value at T or held 15-minute value) |

Per year there are 87,600 Task A examples (48 leads) and 525,600 Task B examples (72 leads). Only the five forecast
counties have outage records in these inputs. The frozen ensemble was also trained on 35 neighbouring counties.

## Recipe and protocol

XGBoost with a Tweedie objective (variance power 1.3), η = 0.05, depth 7, min child weight 10, row and column
subsampling 0.8 and seed 42. These values are fixed and were not tuned. Two variants are compared:

- **plain**: Tweedie regression of the outage share.
- **anchored**: the same, with log(persistence forecast) as base margin. The trees then learn a multiplicative change
  to "the latest observation stays", which is the natural null model for outages.

| Stage | Fit | Evaluate | Purpose |
|---|---|---|---|
| 1. selection | 2022 | 2023 | choose the variant and number of rounds per task, **by the challenge score** |
| 2. test | 2022–2023 | 2024 | first and only look at 2024 |
| 3. final | 2022–2024 | 2025 | the saved model; 2025 is never used for fitting or choosing |

At each boundary, training examples whose target lies in the evaluation period are dropped (purging). Every score
below comes from the SQL metric (`sql/scoring`), with persistence scored on the same cases.

### Selection on 2023 (challenge score; persistence: A 50.85, B 72.32)

| Rounds | 25 | 50 | 100 | 200 | 400 | 800 | 1200 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A plain | 56.29 | 57.31 | **57.62** | 57.14 | 56.63 | 55.97 | 55.58 |
| A anchored | 55.63 | 57.08 | 57.13 | 56.77 | 56.71 | 56.39 | 56.11 |
| B plain | 74.68 | 75.65 | 75.98 | 75.91 | 75.92 | 75.70 | 75.54 |
| B anchored | 75.28 | 76.31 | 76.72 | **76.75** | 76.74 | 76.50 | 76.34 |

Chosen: Task A plain, 100 rounds; Task B anchored, 200 rounds. Long training hurts both tasks, so the optimum
is a small model.

## Results

| Evaluation | Reference | Persistence | Frozen ensemble |
|---|---:|---:|---:|
| 2024 test, Task A / B | **53.43 / 73.04** | 49.75 / 69.56 | (trained on 2024) |
| 2025 full year | 42.60 / 71.12 | 43.92 / 66.33 | **54.88 / 73.44** |
| 2025 June–August | 41.29 / 71.39 | 38.69 / 65.88 | **54.45 / 75.13** |

- **Task B.** The simple model gets most of the way: +4.8 over persistence on 2025, and 2.3 points below the frozen
  ensemble. It beats persistence in 11 of 12 months of 2025 (May −0.1).
- **Task A.** It beats persistence in the 2024 test and in 9 of 12 months of 2025 (up to +10.8). It loses heavily
  in January (−12.5) and February (−26.0) 2025, when the Los Angeles wildfire outages lasted for days (up to 46 % of
  customers). A held observation is nearly perfect in such an event, while a tree that forecasts each hour
  separately pulls back towards typical values.
- **The gap to the frozen ensemble (Task A +12.3 on 2025)** measures what its extra parts add: event-duration
  features, a GRU over the 7-day sequence, a tail model for large outages, and 8 times more training counties. The
  January–February failure mode is exactly what the event features and the GRU state address.

Most important features (total gain): for Task A the rolling maxima of the outage share over 3–24 h, then radar exposure
and pressure/snow forecasts; for Task B the latest observation and the 15–30 min lags, then gusts, pressure and
precipitation.

## What was changed after looking at 2025

The first version of the trainer chose the number of rounds by early stopping on the Tweedie log-likelihood (59 / 95
rounds) and had only the plain variant. On 2025 it scored A 41.97 and B 70.48. I then changed the selection to the
challenge score and added the anchored variant. The choice is still made only on 2023 validation, but the change was
made after seeing 2025, so 2025 is not perfectly untouched for this recipe. The 2024 test result predates the
change. Training on a GPU is reproducible with the fixed seed; CPU and GPU runs differ slightly.

## Possible next steps

- Event-state features in SQL (duration of the current outage, time since peak) for the multi-day events.
- Training on the 35 neighbouring counties as well. This needs their outage records in the inputs.
- Calibration of the risk (see `sql/analysis/reliability.sql`).
