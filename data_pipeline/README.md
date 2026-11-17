# Data pipeline: building an input directory from public sources

`data/inputs/input_2025/` was built by this pipeline. It is **not used by inference**: `infer.py` reads only the package
and `--input-dir`.

```bash
python -m pip install -r requirements/data.txt      # on top of requirements/cpu.txt; HRRR decoding also needs libblosc
bash data_pipeline/build_inputs.sh --start 2025-06-01 --end 2025-09-01 --out data/inputs/input_2025     # all stages
bash data_pipeline/build_inputs.sh assemble                                                          # or named stages
```

Intermediate files go to `--work` (default `data_pipeline/work/`); downloads resume after an interruption and finished
outputs are skipped. The download stages need network access and tens of GB of raw downloads.

## Stages

| Stage | Script | Source (licence) | Output in `--work` |
|---|---|---|---|
| `sources` | `sources/download_public.py` (every calendar year from 7 days before the first issue) | EAGLE-I, Figshare article 24237376 v4 (CC BY 4.0, MD5-checked); US Census TIGER/Line 2025 counties (public domain) | `raw/eaglei/`, `raw/census/` |
| `eaglei` | `sources/eaglei_counts.py` | — | `eaglei_counts_<year>.parquet`: customers out per county and 15-minute time (40 counties) |
| `hrrr00z` | `hrrr/plan_window.py`, `hrrr/download.py`, `hrrr/county_statistics.py` | NOAA HRRR, `s3://hrrrzarr`, anonymous (public domain) | object plan of the window's 00Z runs; `hrrr00z/weather_<year>_full_polygon_robust.parquet`: exact county-polygon statistics, 14 variables × mean / std / max / p95, 40 counties, one file per calendar year |
| `hrrr18z` | `hrrr/runs_18z.py`, `hrrr/snow_axis.py` | same | `hrrr18z/daily/`: the previous-day 18Z runs, same statistics; `snow_axis.parquet`: verified ASNOW lead axis of the 00Z runs |
| `mrms` | `mrms/download.py`, `mrms/county_series.py` | NOAA MRMS `MergedReflectivityQCComposite_00.50`, `s3://noaa-mrms-pds`, anonymous (public domain) | `raw/mrms/`: regional crops every 10 min; `radar_series.parquet`: county frame statistics |
| `assemble` | `assemble_inputs.py` | — | `--out`: the five Parquet tables of the organizer layout |

`to_parquet.py` converts an input directory in the organizer CSV layout to Parquet (infer.py reads both).

Shipped reference data (`reference/`): the five scored counties and their 35 neighbours (`counties_40.txt`); the HRRR
variable levels, chunks and county-to-grid-cell plan (`hrrr_grid/`); the 2025 00Z object plan and the audited Zarr
variable metadata (`hrrr_plans/`); the 2025 ASNOW lead axis (`snow_axis_2025.parquet`, used when the `hrrr18z` stage has
not been run). Without any 18Z run the weather table simply has no 18Z rows (Task A then uses the 00Z run for its lag
columns).

## Availability rules (replay assumptions, declared in the report)

- HRRR: `availability_time = model_run_time` (the archive has no publication times).
- EAGLE-I: `source_record_time = timestamp` (observation-time replay).
- MRMS: `availability_time = grid_time + 4 min`; frames back-filled into the archive follow the same rule.

## Verification (2026-11-05)

- `assemble` on the stored intermediate files reproduces `data/inputs/input_2025` byte for byte (the shipped files are its
  output), and `infer.py` on it reproduces `output/predictions.csv` byte for byte.
- `eaglei` reproduces the stored 2025 counts (1,021,716 rows); `mrms/county_series.py` reproduces the stored frame
  statistics (3 days, 40 counties); `hrrr/runs_18z.py` reproduces a stored 18Z day (26,760 rows) and 00Z axis audit from
  cached objects; `mrms/download.py` decodes an archive object to the stored crops byte for byte.
- `hrrr/plan_window.py` reproduces the shipped 2025 plan row for row (11,844 objects); for a new date (2026-10-01) all
  126 planned data chunks and 28 metadata objects exist in the archive (HEAD check).
- End to end from public sources for a window outside the stored inputs (issues 2025-03-10 and 03-11:
  `build_inputs.sh --start 2025-03-10 --end 2025-03-12`, about 1 GB downloaded): the 00Z statistics computed from the
  freshly downloaded HRRR objects equal the stored statistics (3 runs, 80,280 rows); the downloaded MRMS frames equal the
  stored frames (3 days, 432 frames) and give the same county series; `infer.py` on the assembled tables (Parquet, and
  the same tables as CSV) gives output identical to the frozen package's on the CSV.

## Known limits

- EAGLE-I is published as annual files; a window needs the file of every calendar year it touches.
