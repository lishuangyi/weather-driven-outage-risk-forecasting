# Missing inputs and fallbacks

Real feeds fail: a radar mosaic goes missing, a weather run is late, an outage feed stops. Instead of filling such
gaps with fake values, each case has its own model trained **without** that input. The route is chosen per county
and issue, and every run reports which route each forecast used.

## Routes

| Missing input (per county and issue) | Task A | Task B |
|---|---|---|
| radar: no usable frame in the 24 h before T | joint tree → radar-free event tree; weights unchanged | joint tree → five-family radar-free tree system |
| weather: no usable run, or none of TMP / GUST / CAPE | persistence-anchored weather-free tree replaces the ensemble | members kept, weather features missing |
| outage history: no usable record | history-free tree, trained with masked inputs, replaces everything | monthly median outage share of the county (climatology) |

Missing history takes priority over missing weather. The model files and the routing rules are in
`configs/inference.json`, and the code is in `task_a.py` and `task_b.py`.

## What it costs (June–August 2025)

| Condition | Task A | Task B |
|---|---:|---:|
| all inputs | 54.52 | 75.13 |
| no radar | 53.79 | 74.65 |
| only the 5 basic weather fields, no radar | 42.27 | 72.86 |
| no outage history | 33.04 | 24.72 |

- **Radar** adds little on average (+0.7 / +0.5), but it is cheap to lose.
- **Rich weather statistics** matter for Task A: county mean, std, max and p95 of 14 HRRR variables, against only 5
  basic fields. Routing the basic-fields case to the weather-free tree would have gained 4.9 points on June–August.
  It was not adopted, because it lost 10.1 points on January–May.
- **Outage history** is essential. In a 3-day test, seven days of history before the first issue gave the same
  scores as the full history, while one day or none lowered them by 2–3 points.

You can reproduce the missing-input cases with the demo tool. It removes an input and reruns the frozen system:

```bash
python demo/run_demo.py --data data/inputs/input_2025 --start 2025-07-08 --end 2025-07-15 --drop radar
```

## Invalid input is rejected, not guessed

Missing optional data leads to a fallback. Broken data stops the run with exit code 2 and writes nothing:
missing request or customer tables, unreadable files, timestamps without a UTC offset, customer totals that differ
from the official ones, and requests outside the issue/lead grid. `tests/test_inputs.py` covers these cases, and the
SQL warehouse reports the same checks as a table (`silver.quality_checks`).
