# Data

## Input tables (one directory per run, Parquet or CSV)

| Table | Required | Columns |
|---|---|---|
| `requests/forecast_requests` | yes | `task_id, architecture_id, county_fips, issue_time, target_time` |
| `mcc/MCC` | yes | `county_fips, total_customers` |
| `eaglei/eaglei_outages` | yes¹ | `county_fips, timestamp, customers_out, source_record_time` |
| `weather/weather_forecasts` | yes¹ | `county_fips, model_run_time, availability_time, target_time`, 5 basic fields, optional HRRR statistics `<VAR>__<mean\|std\|max\|p95>` |
| `radar/mrms_county_series` | no | `county_fips, grid_time, availability_time, avail, f35, f50, max_dbz, ring35` |

¹ A missing outage or weather file counts as an empty table and the fallback models are used. An unreadable or
incomplete file stops the run with exit code 2.

## Released inputs (`python scripts/download_artifacts.py …`)

| Group | Content | Size |
|---|---|---:|
| `test-window` | `data/inputs/input_2025/`: issues 2025-06-01 … 08-31, 618,240 requests | 75 MB |
| `training` | `data/inputs/input_{2022,2023,2024}_full/`: every issue of the year | 3 × ~270 MB |
| `full-2025` | `data/inputs/input_2025_full/`: every issue of 2025 | 290 MB |

Weather covers 40 counties (5 forecast + 35 neighbours). Outages and radar cover the 5 forecast counties.

## Availability rules (replay assumptions)

The public archives do not store publication times. Each row therefore gets an availability time by a fixed rule, and
the runtime only reads rows whose availability time is ≤ the issue time:

- HRRR: `availability_time = model_run_time`
- EAGLE-I: `source_record_time = timestamp`
- MRMS: `availability_time = grid_time + 4 min`

Real feeds arrive later. With outage records 3 minutes late and weather runs 4 hours late, the June–August 2025
scores drop from 54.52 / 75.13 to 49.77 / 72.97. The runtime never uses data before it is available and does not
compensate for delays.

## Building the tables from public sources

`data_pipeline/` downloads and aggregates everything (`requirements/data.txt`; HRRR decoding also needs `libblosc`):

```bash
bash data_pipeline/build_inputs.sh --start 2025-06-01 --end 2025-09-01 --out data/inputs/input_2025
```

| Stage | Script | Source |
|---|---|---|
| `sources` | `sources/download_public.py` | EAGLE-I (Figshare 24237376 v4, MD5-checked), Census TIGER/Line 2025 counties |
| `eaglei` | `sources/eaglei_counts.py` | customers out per county and 15-minute time |
| `hrrr00z` | `hrrr/plan_window.py`, `hrrr/download.py`, `hrrr/county_statistics.py` | HRRR-Zarr: exact county-polygon statistics, 14 variables |
| `hrrr18z` | `hrrr/runs_18z.py`, `hrrr/snow_axis.py` | previous-day 18Z runs |
| `mrms` | `mrms/download.py`, `mrms/county_series.py` | MRMS composite reflectivity, regional crops every 10 min |
| `assemble` | `assemble_inputs.py` | the five tables above |

Running the pipeline on its stored intermediate files reproduces the released 2025 inputs byte for byte. A fresh
download for a new window (2025-03-10 … 03-12) gave statistics equal to the stored ones. More detail is in
[`data_pipeline/README.md`](../data_pipeline/README.md).

## Licences

See [`DATA_LICENSES.csv`](../DATA_LICENSES.csv). EAGLE-I is CC BY 4.0: © Oak Ridge National Laboratory. Our changes
are aggregation to county × 15-minute counts and Parquet conversion. NOAA and Census data are public domain.
