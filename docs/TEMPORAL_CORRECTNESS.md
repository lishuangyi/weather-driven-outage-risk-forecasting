# Temporal correctness

A forecast issued at time T may use only information that was available at T. Breaking this rule ("lookahead") is
the most common way a forecasting model looks better offline than it is. This project enforces the rule at three
levels: the data rule, the code, and tests that would fail if it were broken.

## The rule

A record counts at issue time T only if it was **observed and published** by T:

| Source | Observed by T | Published by T | Notes |
|---|---|---|---|
| EAGLE-I outages | `timestamp ≤ T` | `source_record_time ≤ T` | latest visible version per observation time; a blank latest version means missing, not the older value |
| HRRR weather | `model_run_time ≤ T` | `availability_time ≤ T` | latest visible run per target hour; Task A also averages the previous-day 18Z run |
| MRMS radar | frame time ≤ T | `availability_time ≤ T` | 24 h of frames before T |

The target is the outage share after T. The model never sees the target of the forecast it is making.

## Replay assumptions

The public archives do not record when each record was published. Each row therefore gets a publication time by a
fixed rule, and these are the only assumptions in the pipeline:

- HRRR: available at its run time (in reality a run is published some time after it starts).
- EAGLE-I: published at its observation time.
- MRMS: available 4 minutes after the frame time.

To measure how much the assumptions matter, we delayed every outage record by 3 minutes and every HRRR run by
4 hours, and rescored June–August 2025. The scores drop from 54.52 / 75.13 to 49.77 / 72.97. The code respects
any delays it is given and never uses data early. It does not try to compensate for delays.

## Where the code enforces it

| Place | What it does |
|---|---|
| `features/history.py` `OutageHistory.at` | per issue, keeps records with `timestamp ≤ T` and `source_record_time ≤ T` and resolves versions |
| `features/weather.py` `select_weather` | filters candidate runs on `availability ≤ T` and `model_run ≤ T` before choosing one |
| `features/radar.py` | uses only frames with `availability ≤ T` in the 24 h window |
| `sql/gold/01_visible_history.sql` | the same rule in SQL, followed by `QUALIFY row_number()` for versions |
| `sql/gold/04_context_features.sql` | weather and radar as of T in SQL (`run_s ≤ T AND available_s ≤ T`) |

## Tests that would catch a violation

| Test | How it works |
|---|---|
| `tests/test_no_lookahead.py` | Every record published after a cut-off (outages, weather runs, radar frames) is replaced by an extreme value. Forecasts issued at or before the cut-off must stay **bit-identical**, and later forecasts must change, which shows that the perturbation reaches the model. |
| `tests/test_sql_parity.py::test_sql_features_ignore_future_data` | the same perturbation for the 30 SQL history features |
| `tests/test_sql_parity.py::test_history_features_match_python` | the SQL and Python point-in-time features agree on every issue, so they would have to be wrong in the same way |

## Time in model development

- **Training years.** All frozen models were fitted on 2022–2024 labels. 2025 is never a training year.
- **Selection.** The ensemble weights and members were chosen on June–August 2025, the challenge window. This is
  disclosed with every score, and the other 9 months of 2025 serve as a retrospective check ([RESULTS.md](RESULTS.md)); January–May also informed one fallback decision.
- **Reference trainer.** The trainer uses a chronological split: 2022 fit, 2023 selection, 2024 test, 2025 final.
  Training examples whose target falls in the next period are dropped (purging). See [TRAINING.md](TRAINING.md).
