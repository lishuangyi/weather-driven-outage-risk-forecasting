PY ?= python
INPUT ?= data/inputs/input_2025
OUT ?= out/predictions.csv

.PHONY: setup models data predict demo test lint
setup:            ## CPU environment (Python 3.12)
	$(PY) -m pip install -r requirements/cpu.txt -r requirements/dev.txt && $(PY) -m pip install -e . --no-deps
models:           ## download the frozen models (39 MB)
	$(PY) scripts/download_artifacts.py models
data:             ## download the 2025 evaluation-window inputs (75 MB)
	$(PY) scripts/download_artifacts.py test-window
predict:          ## full evaluation window: 618,240 forecasts in about a minute on CPU
	$(PY) infer.py --input-dir $(INPUT) --output $(OUT)
demo:             ## one-week demo with summary page (out/demo/summary.html)
	$(PY) demo/run_demo.py --data $(INPUT) --start 2025-07-08 --end 2025-07-15 --counties Harris --out out/demo
test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests scripts
