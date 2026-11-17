# Weather-Driven Power-Outage Risk Forecasting for AI Data Centers

Forecasts how many customers in a US county will be without power over the next **48 hours (hourly)** and the next
**6 hours (5-minute)**, and turns each forecast into a grid-supply risk for four AI data-center power architectures.
Inputs are public: EAGLE-I outage records, NOAA HRRR weather forecasts and NOAA MRMS radar.

![48-hour forecasts against observed outages, Wayne County, June 2025](docs/assets/event_wayne_2025-06.png)

*Wayne County (Detroit), 26–30 June 2025. The forecast issued on 27 June at 00:00 UTC rises above the risk band before
the storm, and the 28 June forecast follows the restoration, though it decays more slowly than the real recovery.*

**What this project shows**

- **Point-in-time correctness.** A forecast issued at time T only reads data that was published by T. This holds for
  the outage records, the weather runs and the radar frames, and a test checks it by changing every record published
  after T (`tests/test_no_lookahead.py`).
- **A hybrid model.** Gradient-boosted trees (XGBoost, Tweedie) on weather, radar and outage-history features are
  combined with a GRU over the last 7 days of outages. Fallback models take over when radar, weather or outage history
  is missing.
- **Reproducible inference.** 618,240 forecasts for June–August 2025 run in about 30 s on a GPU or 1 min on a CPU, and
  the output matches the frozen release byte for byte. Model files are checked against their SHA-256 before use.
- **A SQL warehouse (DuckDB).** It is a second implementation of the data side in plain SQL, organised as bronze,
  silver and gold layers. It covers point-in-time features (as-of joins on publication time), labels, data-quality
  checks and the challenge metric. Tests check that it reproduces the 30 Python history features and the official
  scores to within floating-point rounding (see [docs/SQL.md](docs/SQL.md)).
- **Training from scratch.** A reference trainer fits one XGBoost model per task on the SQL training table. It uses
  a chronological protocol: select on 2023, test on 2024, final model evaluated on 2025. The evaluation is
  candid: the trainer beats persistence on the 2024 test, and on 2025 it comes out below the frozen ensemble
  ([docs/TRAINING.md](docs/TRAINING.md)).
- **An end-to-end data pipeline**, from public cloud archives (`s3://hrrrzarr`, `s3://noaa-mrms-pds`, Figshare) to
  county-level Parquet tables. HRRR forecasts are aggregated over exact county polygons.

## Quick start (CPU, Python 3.12)

```bash
git clone https://github.com/lishuangyi/weather-driven-outage-risk-forecasting.git
cd weather-driven-outage-risk-forecasting
make setup          # pip install -r requirements/cpu.txt -r requirements/dev.txt && pip install -e .
make models data    # frozen models (39 MB) + June–August 2025 inputs (75 MB), SHA-256 verified
make predict        # out/predictions.csv: 618,240 rows
make train          # optional: reference model from scratch (needs the 2022–2025 inputs, ≈3 min on GPU)
make warehouse      # out/warehouse.duckdb: features, labels, quality checks, scores by county/lead/month (≈5 s)
make demo           # out/demo/summary.html: one week in Harris County with scores and charts
make test           # end-to-end, no-lookahead, SQL-vs-Python parity, input-validation and metric tests
```

The downloads come from this repository's [GitHub release](artifacts/release_v1.0.0.json) through the GitHub CLI if it
is installed, otherwise over plain HTTPS. Inference itself runs offline.

## Problem

| | Task A | Task B |
|---|---|---|
| Issue times (UTC) | daily 00:00 | 00:00, 06:00, 12:00, 18:00 |
| Horizon / step | 48 h / 1 h | 6 h / 5 min |
| Target | mean outage share of the 4 quarter-hours ending at T | outage share at T (15-min value held) |

Five counties are forecast: Los Angeles CA, Miami-Dade FL, Wayne MI, Rockingham NH and Harris TX. The outage share is
x = customers out / customers served. Architecture *a* turns it into a risk
P = 1 / (1 + exp(−4000 (x − x0ₐ))), with x0 from 0.100 % (ARC01) to 0.160 % (ARC04).

**Metric.** Each task pools all counties, architectures, issues and leads into one score,
100 × (0.75 × normalised AUPRC + 0.25 × (1 − MAE)), computed on the risk scale. AUPRC is normalised by the event rate,
and the event threshold is P ≥ 0.5.

## Results

These are local scores computed with the challenge formula on our EAGLE-I truth, not organiser-certified results. The
June–August 2025 window was **also used for model selection**, so treat it as a validation score. The models were
trained on 2022–2024.

| Window | Task A model / persistence | Task B model / persistence |
|---|---:|---:|
| June–August 2025 (selection window) | **54.52** / 38.82 | **75.13** / 65.88 |
| The other 9 months of 2025 (used for neither training nor selection) | **55.00** / 46.92 | **72.75** / 66.59 |

Persistence carries the latest observation forward to every target. The unseen months score about the same as the
selection window, so the selection did not overfit that window. Scored month by month, the model beats persistence in
all 12 months of 2025 for Task B and in 10 of 12 for Task A (January −0.2, February −4.3). Absolute scores depend
mainly on how many outage events a period contains, and Task A is weakest in quiet periods.

**Reference model trained from scratch** (`make train`, one XGBoost per task, selected on 2023 only):

| Evaluation | Reference | Persistence | Frozen ensemble |
|---|---:|---:|---:|
| 2024 (test year) | 53.43 / 73.04 | 49.75 / 69.56 | in training data |
| 2025 full year | 42.60 / 71.12 | 43.92 / 66.33 | 54.88 / 73.44 |

The simple model recovers most of the Task B skill. On Task A it loses to persistence during the multi-day Los
Angeles wildfire outages of January–February 2025. Those are the events that the frozen ensemble's event features
and GRU are built for ([docs/TRAINING.md](docs/TRAINING.md#results)).

**Robustness on June–August 2025:**

| Condition | Task A | Task B |
|---|---:|---:|
| All inputs | 54.52 | 75.13 |
| No radar (fallback models) | 53.79 | 74.65 |
| Only 5 basic weather fields, no radar | 42.27 | 72.86 |
| Outage records 3 min late, weather runs 4 h late | 49.77 | 72.97 |
| No outage history at all | 33.04 | 24.72 |

The SQL analysis queries break this down further. The model beats persistence in every county, with the largest
gain in Wayne County. Task A is over-confident in its top risk bin (predicted 0.98, observed 0.64), and the small
Los Angeles events are mostly missed ([docs/SQL.md](docs/SQL.md#analysis-queries-sqlanalysis)).

Recent outage history is the most important input. In a three-day test, seven days of history gave the same scores as
the full history.

## How it works

```
EAGLE-I outages ─┐                         ┌─ outage history: lags, rolling stats, event state ─┐
HRRR forecasts  ─┼─ as-of filter at T ─────┼─ weather: county mean/std/max/p95 of 14 variables ─┼─► XGBoost (Tweedie) ─┐
MRMS radar      ─┘   (published ≤ T only)  ├─ radar: reflectivity exposure in the last 24 h ────┘                     ├─► blend ─► x̂ ─► risk per architecture
                                           └─ 7-day 15-min sequence (240 × 6) ───────────────► GRU ────────────────┘
```

| Task | Members (weight) | Fallbacks |
|---|---|---|
| A | GRU (0.4), joint feature tree (0.4), Q95 tail model (0.2) | no radar → radar-free tree; no weather → weather-free tree; no history → history-free tree |
| B | joint Tweedie tree (0.5), GRU (0.5) | no radar → five-family tree system; no history → monthly climatology |

See [docs/MODEL.md](docs/MODEL.md) for the model details, [docs/SQL.md](docs/SQL.md) for the SQL warehouse,
[docs/TRAINING.md](docs/TRAINING.md) for training from scratch, [docs/DATA.md](docs/DATA.md) for sources, licences and
availability rules, and [docs/DEMO.md](docs/DEMO.md) for the demo tool.

## Repository layout

| Path | Content |
|---|---|
| `src/outage_forecast/` | inference package: input validation, point-in-time features, model members, blending, output; `warehouse.py` runs the SQL layers; `reference.py` trains and applies the reference model |
| `sql/` | DuckDB warehouse: `bronze/`, `silver/`, `gold/` layers and `analysis/` queries ([docs/SQL.md](docs/SQL.md)) |
| `configs/inference.json` | the single runtime configuration: feature schemas, member weights, fallback routes, model SHA-256 |
| `data_pipeline/` | builds the input tables from public archives (HRRR, MRMS, EAGLE-I, Census polygons) |
| `demo/` | demo runner, local scorer, request selector, charts; `notebooks/demo.ipynb` is the notebook front end |
| `tests/` | pytest suite on a 2-day input cut from the 2025 data (`scripts/make_fixture.py`) |
| `scripts/` | artifact download, fixture and figure builders |
| `artifacts/` | pinned release assets and their checksums |

## Limitations

- **Replay assumptions.** The archives do not record when each record was published, so an HRRR run counts as
  available at its run time, an outage record at its observation time and a radar frame 4 minutes after its frame time.
  Real publication delays lower the scores (see the robustness table above).
- Only the five counties and the issue grid above are supported. The trees use the weather of 35 neighbouring counties
  as well.
- Task B models the 15-minute series; 5-minute detail is not modelled.
- The frozen ensemble's weights come from the original experiment code, which is not included. The reference trainer
  (`outage_forecast.reference`) is the reproducible training path, and it is simpler and weaker on Task A.

## Background and credits

This started as my entry to Huawei TechArena 2026 (Topic 2, Challenge 2). This repository is an independent, cleaned-up
version, and no challenge materials are included. Code and models: [@lishuangyi](https://github.com/lishuangyi).

Data: EAGLE-I (ORNL, CC BY 4.0), NOAA HRRR and MRMS (public domain), US Census TIGER/Line (public domain). See
[DATA_LICENSES.csv](DATA_LICENSES.csv). Code licence: [MIT](LICENSE).
