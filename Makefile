PY ?= python
INPUT ?= data/inputs/input_2025
OUT ?= out/predictions.csv

.PHONY: setup models data predict warehouse sql-smoke train demo test lint
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
demo:             ## one-week demo with summary page (out/demo/summary.html)
	$(PY) demo/run_demo.py --data $(INPUT) --start 2025-07-08 --end 2025-07-15 --counties Harris --out out/demo
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests scripts docs/diagrams
