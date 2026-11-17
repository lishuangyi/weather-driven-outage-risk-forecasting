# SQL warehouse (DuckDB)

`sql/` holds a second, independent implementation of the data side of the project in plain SQL: the
point-in-time outage history, the 30 history features of the models, the labels of both tasks, data-quality checks
and the challenge metric. It runs on DuckDB over the same input directory as the Python pipeline. Tests check that
the SQL and Python results agree.

```bash
make predict warehouse                      # out/predictions.csv, then out/warehouse.duckdb (≈5 s for June–August 2025)
python -m outage_forecast.warehouse report --db out/warehouse.duckdb
python -m outage_forecast.warehouse query sql/analysis/largest_events.sql
duckdb out/warehouse.duckdb                 # or explore it with the DuckDB CLI
```

## Layers

```
input directory (Parquet / CSV)
  └─ bronze   views over the delivered files, unchanged              bronze.requests, .customers, .outages, .weather, .radar
       └─ silver  typed tables with an explicit grain                 silver.requests, .customers, .outage_versions, .issues
       │          data-quality checks                                 silver.quality_checks
       └─ gold    point-in-time history per issue                     gold.visible_history
                  as-of lags, window aggregates, 30 features          gold.history_lags, .history_windows, .history_features
                  final observations and request labels               gold.final_outages, .labels
                  weather and radar as of the issue                   gold.weather_as_of, .radar_as_of
                  model-ready training/prediction table               gold.examples               (reference model)
       └─ scoring model and persistence cases next to their truth     gold.scored_cases           (needs --predictions)
                  metric by dimension, headline scores                gold.metrics, .scoreboard   (needs --predictions)
ref: architectures (x0, k) and request grids from configs/inference.json
```

| Table | Grain | Notes |
|---|---|---|
| `silver.outage_versions` | one delivered outage record | every revision kept; blank = NULL count; `on_grid` = 15-minute time |
| `silver.issues` | (task, county, issue time) | issues that have at least one request |
| `gold.visible_history` | (county, issue, observation time) | the version a forecaster could see at the issue (see below) |
| `gold.history_lags` | (county, issue, lag) | `ASOF LEFT JOIN`: last visible observation at or before T − lag |
| `gold.history_windows` | (county, issue, window) | count, mean, max, min and least-squares slope over [T − W, T] |
| `gold.history_features` | (task, county, issue) | the 30 model columns, named as in `features/history.py` |
| `gold.final_outages` | (county, observation time) | final value: latest revision of each observation |
| `gold.labels` | request | Task A: mean of the 4 quarter-hours ending at T; Task B: value at T or the 15-minute value held |
| `gold.examples` | (task, county, issue, target) | 79 features as of the issue + label: the training table of [TRAINING.md](TRAINING.md) |
| `gold.scored_cases` | (model, scorable request) | `model` = the predictions file, `persistence` = latest visible value held |
| `gold.metrics` | (model, task, dimension, group) | dimensions: pooled, architecture, county, lead band, month |

## The point-in-time rule

At issue time T a record counts only if it was observed **and published** by T:

```sql
JOIN silver.outage_versions o
  ON o.county = i.county AND o.on_grid
 AND o.ts_s  <= i.issue_s          -- observed by T
 AND o.src_s <= i.issue_s          -- published by T
...
QUALIFY row_number() OVER (PARTITION BY county, issue_s, ts_s ORDER BY src_s DESC, row_id DESC) = 1
```

Late corrections are therefore invisible to earlier issues. If the latest visible version is blank, the observation
counts as missing; an older value is not reused. The lags use an as-of join on this per-issue view:

```sql
FROM probes p
ASOF LEFT JOIN gold.visible_history v
  ON v.county = p.county AND v.issue_s = p.issue_s AND v.ts_s <= p.probe_s     -- probe = T - lag
```

The look-back is bounded to 3 days before T (`--lookback-minutes`). The longest lag and window are 24 h, so only a
series with a gap longer than 2 days could differ from the unbounded Python version. On the 2025 data both give the
same values.

## How the two implementations are kept equal (`tests/test_sql_parity.py`)

| Check | Result |
|---|---|
| 30 history features, every (task, county, issue) of the test input and of June–August 2025 (2,300 issues) | same missing values; values equal within 1e-12 relative |
| labels of every request (Task A and B) | identical, including unavailable and invalid truths |
| pooled, per-architecture and per-county scores, generalisation term, technical score | within 1e-9 of `demo/scoring.py`; June–August 2025: **54.5219 / 75.1347** |
| no lookahead | changing every record published after a cut-off leaves the SQL features of earlier issues unchanged |
| quality checks | pass on clean input; injected duplicate requests and early-published records are caught with the exact counts |
| determinism | repeated builds give bit-identical tables |

DuckDB aggregates in parallel, so a plain floating-point `sum` or `regr_slope` can change in the last bits from one
run to the next (about 1e-19 here). The slope therefore uses ordered aggregates (`sum(... ORDER BY t)`), and the
features are deterministic.

## The metric in SQL

Average precision with tied predictions treated as one threshold (as scikit-learn computes it):

```sql
thresholds AS (SELECT ..., pred, count(*) AS n, sum(y) AS tp FROM gold.case_groups GROUP BY ALL),
curve AS (SELECT *, sum(tp) OVER w / sum(n) OVER w AS precision_at,
                 sum(tp) OVER (PARTITION BY model, task, dimension, grp) AS positives
          FROM thresholds WINDOW w AS (PARTITION BY model, task, dimension, grp ORDER BY pred DESC ROWS UNBOUNDED PRECEDING))
SELECT ..., sum(tp / positives * precision_at) AS ap FROM curve ...
```

The same query scores every grouping (pooled, architecture, county, lead band, month) in one pass, for the model and
for the persistence baseline. Only the pooled number is the official one, because AUPRC does not average across
groups.

## Analysis queries (`sql/analysis/`)

| Query | Question | June–August 2025 finding |
|---|---|---|
| `model_vs_persistence.sql` | where does the model beat persistence? | gains in every county; largest in Wayne (A +24.1) and Rockingham (B +14.9); smallest in Miami-Dade, where persistence is already strong |
| `largest_events.sql` | did the model see the big outages coming? | all five Miami-Dade peaks and four of five Wayne peaks get risk ≥ 0.97 from the issue 0–24 h before them; the small Los Angeles peaks are missed (risk ≤ 0.21) |
| `reliability.sql` | is the predicted risk calibrated? | both tasks are close to calibrated below 0.5 and over-confident above it; top bin: Task A predicted 0.98 vs observed 0.64, Task B 0.99 vs 0.88. A calibration step is a natural next improvement |
| `data_freshness.sql` | how fresh was the history at forecast time? | Rockingham's feed lags (mean age 2 h for Task B, up to 23 h); the other counties are current at every issue |

Persistence computed in SQL scores 38.82 (A) and 65.88 (B) on June–August 2025. This is the same baseline as in the
README table.
