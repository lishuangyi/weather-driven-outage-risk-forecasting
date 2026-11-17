# Roadmap

| Stage | Content | Status |
|---|---|---|
| 1. Release | cleaned package, pinned artifacts with checksums, CPU quick start, tests (end-to-end, no-lookahead, input validation, metric), CI | done (v1.0.0) |
| 2. SQL / DuckDB | point-in-time feature and evaluation database: `bronze` (Parquet views) → `silver` (typed observations with revisions, forecast runs, quality checks) → `gold` (issue-level features, labels, scored cases). `ASOF JOIN` for the as-of lookups, SQL labels for both tasks, the challenge metric in SQL, and parity tests against the Python features and scorer | done ([SQL.md](SQL.md)) |
| 3. Reference training | `train.py` that builds training sets from the released 2022–2024 inputs through the same feature code, fits one XGBoost model per task on a chronological split (2022 fit / 2023 validation / 2024 test, with purging at the boundaries), and reports it next to persistence and the frozen ensemble | planned |
| 4. Write-up | model card, results page, short notes on temporal correctness and missing-input behaviour | planned |
