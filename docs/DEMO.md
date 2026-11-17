# Demo tool

`demo/run_demo.py` runs the whole chain for a chosen period and writes a one-page HTML summary:

1. environment check (packages, device, SHA-256 of every model file),
2. input overview (what each table covers, outage history and weather available before each issue),
3. inference through `infer.py`, started as a separate process (the same command a user would run),
4. output check (one row per request, values in [0, 1], risk formula),
5. local score with the challenge formula, and forecast-against-observation charts.

```bash
python demo/run_demo.py --data data/inputs/input_2025 --start 2025-07-08 --end 2025-07-15 --counties Harris --out out/demo
python demo/run_demo.py --data data/inputs/input_2025 --start 2025-07-08 --tasks B          # one day, Task B only
python demo/run_demo.py --data data/inputs/input_2024_full --start 2024-07-08 --end 2024-07-12 --generate-requests
python demo/run_demo.py --data data/inputs/input_2025 --start 2025-07-08 --drop radar       # missing-input stress test
python demo/run_demo.py --check-only                                                        # environment only
```

| Option | Meaning |
|---|---|
| `--counties` | FIPS codes or names (Los Angeles, Miami-Dade, Wayne, Rockingham, Harris) |
| `--start`, `--end` | issues in [start, end) UTC; `--start` alone means one day |
| `--tasks` | `A`, `B` or both |
| `--generate-requests` | build the request grid instead of filtering the request table |
| `--a-leads`, `--b-leads`, `--b-issue-hours` | part of the lead grid (`MIN:MAX[:STEP]` minutes) or of the Task B issue hours |
| `--drop radar\|weather` | leave out an input to show the fallback models |
| `--device cpu\|cuda` | default: CUDA when available |

The tool never changes a prediction or an input file. `notebooks/demo.ipynb` runs the same tool from Jupyter
(`pip install -r requirements/notebook.txt`) and shows the scores and charts inline.

Two developer checks build on it: `demo/check_summary.py` renders the summary page for several selections, and
`demo/check_months.py` scores every month of a year.
