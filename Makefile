PY ?= python
INPUT ?= data/inputs/input_2025
OUT ?= out/predictions.csv
FROZEN_2025 ?= out/frozen_2025_full.csv

.PHONY: setup models data predict warehouse sql-smoke train bootstrap calibration alerts report demo test lint
setup:            ## CPU environment (Python 3.12)
	$(PY) -m pip install -r requirements/cpu.txt -r requirements/dev.txt && $(PY) -m pip install -e . --no-deps
models:           ## download the frozen models (39 MB)
	$(PY) scripts/download_artifacts.py models
data:             ## download the 2025 evaluation-window inputs (75 MB)
	$(PY) scripts/download_artifacts.py test-window
predict:          ## full evaluation window: 618,240 forecasts in about a minute on CPU
	$(PY) infer.py --input-dir $(INPUT) --output $(OUT)
warehouse:        ## DuckDB warehouse of the inputs + scores of out/predictions.csv (out/warehouse.duckdb)
	PYTHONPATH=src $(PY) -m outage_forecast.warehouse build --input-dir $(INPUT) --predictions $(OUT) --db out/warehouse.duckdb
sql-smoke:        ## DuckDB warehouse on the 2-day fixture, scored with the recorded predictions; needs no model files
	PYTHONPATH=src $(PY) -m outage_forecast.warehouse build --input-dir tests/fixtures/mini_input --predictions tests/fixtures/expected_predictions.parquet --db out/smoke.duckdb
train:            ## reference model from scratch: needs `python scripts/download_artifacts.py training full-2025` (≈3 min on GPU)
	PYTHONPATH=src $(PY) -m outage_forecast.reference train
bootstrap:        ## paired block-bootstrap CI vs persistence on 2025 (needs full-2025 inputs and FROZEN_2025 predictions, ≈5 min)
	PYTHONPATH=src $(PY) -m outage_forecast.evaluation --input-dir data/inputs/input_2025_full --predictions $(FROZEN_2025) --name "frozen ensemble" --out artifacts/evaluation/bootstrap_2025_frozen.json
calibration:      ## event probability next to the challenge risk: fit June-August 2025, evaluate September-December
	PYTHONPATH=src $(PY) -m outage_forecast.calibration --input-dir data/inputs/input_2025_full --predictions $(FROZEN_2025) --out artifacts/evaluation/calibration_2025_frozen.json > /dev/null
alerts:           ## alert-budget replay of outage onsets (Task A, ARC01): fit June-August 2025, evaluate September-December
	PYTHONPATH=src $(PY) -m outage_forecast.alerts --input-dir data/inputs/input_2025_full --predictions $(FROZEN_2025) --out artifacts/evaluation/alerts_2025_frozen.json > /dev/null
report:           ## versioned run report of configs/benchmark.json (needs its inputs and prediction files), then the doc tables
	PYTHONPATH=src $(PY) -m outage_forecast.run_report run && PYTHONPATH=src $(PY) -m outage_forecast.run_report render
demo:             ## one-week demo with summary page (out/demo/summary.html)
	$(PY) demo/run_demo.py --data $(INPUT) --start 2025-07-08 --end 2025-07-15 --counties Harris --out out/demo
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests scripts docs/diagrams
