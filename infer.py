#!/usr/bin/env python3
"""Inference entry point (works from a plain checkout, no installation of the package needed).

    python infer.py --input-dir data/inputs/input_2025 --output out/predictions.csv

Task A (48 h, hourly)    = clip(0.4 GRU history network + 0.4 joint feature tree + 0.2 Q95 tail member)
Task B (6 h, 5-minute)   = clip(0.5 joint Tweedie tree + 0.5 GRU history network)
with per-row fallbacks for missing radar, weather or outage history (configs/inference.json). See README.md.
After `pip install -e .` the same command is available as `outage-forecast`.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from outage_forecast.cli import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main())
