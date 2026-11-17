# Roadmap

| Stage | Content | Status |
|---|---|---|
| 1. Release | cleaned package, pinned artifacts with checksums, CPU quick start, tests (end-to-end, no-lookahead, input validation, metric), CI | done (v1.0.0) |
| 2. SQL / DuckDB | point-in-time feature and evaluation database: `bronze` (Parquet views) → `silver` (typed observations with revisions, forecast runs, quality checks) → `gold` (issue-level features, labels, scored cases). `ASOF JOIN` for the as-of lookups, SQL labels for both tasks, the challenge metric in SQL, and parity tests against the Python features and scorer | done ([SQL.md](SQL.md)) |
| 3. Reference training | `outage_forecast.reference`: training table built in SQL (`gold.examples`), one XGBoost model per task, chronological split (2022 fit / 2023 selection by the challenge score / 2024 test / 2025 final), purging at the boundaries, compared with persistence and the frozen ensemble | done ([TRAINING.md](TRAINING.md)) |
| 4. Write-up | [model card](MODEL_CARD.md), [results](RESULTS.md), [temporal correctness](TEMPORAL_CORRECTNESS.md), [missing inputs](MISSING_INPUTS.md), model diagrams (`docs/diagrams/`) | done |
| 5. Model work | event-state features in SQL for multi-day outages, risk calibration, training on neighbouring counties | ideas |
