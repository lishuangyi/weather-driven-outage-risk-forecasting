#!/usr/bin/env bash
# Build an input directory (organizer layout, Parquet) from public sources. Network access is needed for the download
# stages; downloads resume and finished outputs are skipped. Not used by infer.py.
#
#   bash data_pipeline/build_inputs.sh [--start 2025-06-01] [--end 2025-09-01] [--work data_pipeline/work] \
#        [--out data/inputs/input_2025] [stage ...]
#
# Stages (default: all, in this order):
#   sources   EAGLE-I annual file (Figshare v4) and TIGER/Line 2025 county polygons
#   eaglei    EAGLE-I -> customers out per county and 15-minute time
#   hrrr00z   00Z HRRR runs of the window (object plan from plan_window.py) -> county-polygon statistics
#   hrrr18z   previous-day 18Z HRRR runs -> county-polygon statistics; 00Z ASNOW lead axis
#   mrms      MRMS composite reflectivity every 10 min -> county frame statistics
#   assemble  requests, MCC, weather, EAGLE-I and radar tables for the issue window
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
START=2025-06-01; END=2025-09-01; WORK="$HERE/work"; OUT="$HERE/../data/inputs/input_2025"; STAGES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --start) START=$2; shift 2;; --end) END=$2; shift 2;; --work) WORK=$2; shift 2;; --out) OUT=$2; shift 2;;
    *) STAGES+=("$1"); shift;;
  esac
done
[ ${#STAGES[@]} -eq 0 ] && STAGES=(sources eaglei hrrr00z hrrr18z mrms assemble)
day() { date -u -d "$1 $2 day" +%Y-%m-%d; }               # day 2025-06-01 -1 -> 2025-05-31
compact() { echo "${1//-/}"; }
FIRST=$(day "$START" -1); LAST=$(day "$END" -1); LAST18=$(day "$END" -2)
HISTORY=$(day "$START" -8)                                   # 7 days of outage history before the first issue
YEARS=$(seq "${HISTORY:0:4}" "${LAST:0:4}")                  # calendar years touched by the window
RUN_YEARS=$(seq "${FIRST:0:4}" "${LAST:0:4}")
COUNTIES=$(cat "$HERE/reference/counties_40.txt")
VARIABLES=APCP_1hr_acc_fcst,ASNOW_acc_fcst,CAPE,DPT,GUST,MSLMA,PRATE,PRES,RH,TCDC,TMP,UGRD,VGRD,VIS
CENSUS="$WORK/raw/census/tl_2025_us_county.zip"
P="python"

stage_sources()  { for Y in $YEARS; do $P "$HERE/sources/download_public.py" eaglei --year "$Y" --out "$WORK/raw/eaglei"; done
                   $P "$HERE/sources/download_public.py" census --out "$WORK/raw/census"; }
stage_eaglei()   { for Y in $YEARS; do $P "$HERE/sources/eaglei_counts.py" "$WORK/raw/eaglei/eaglei_outages_$Y.csv" "$WORK/eaglei_counts_$Y.parquet"; done; }
stage_hrrr00z()  { $P "$HERE/hrrr/plan_window.py" --start "$FIRST" --end "$LAST" --plan-out "$WORK/raw/hrrr00z/plan.parquet" \
                      --variables-out "$WORK/raw/hrrr00z/variables.parquet"
                   $P "$HERE/hrrr/download.py" --plan "$WORK/raw/hrrr00z/plan.parquet" --reference-variable-inventory "$WORK/raw/hrrr00z/variables.parquet" \
                      --raw-root "$WORK/raw/hrrr00z" --inventory-out "$WORK/raw/hrrr00z/object_inventory.parquet" --state-out "$WORK/raw/hrrr00z/download_state.json"
                   for Y in $RUN_YEARS; do
                     $P "$HERE/hrrr/county_statistics.py" --year "$Y" --cache-root "$WORK/raw/hrrr00z" --county-archive "$CENSUS" \
                        --counties "$COUNTIES" --variables "$VARIABLES" --output-dir "$WORK/hrrr00z" --overwrite
                   done; }
stage_hrrr18z()  { $P "$HERE/hrrr/runs_18z.py" full --root "$WORK/hrrr18z" --census "$CENSUS" --start "$(compact "$FIRST")" --end "$(compact "$LAST18")"
                   $P "$HERE/hrrr/runs_18z.py" coordinate00 --root "$WORK/hrrr18z" --start "$(compact "$FIRST")" --end "$(compact "$LAST")"
                   $P "$HERE/hrrr/snow_axis.py" "$WORK/hrrr18z/coordinates00" "$WORK/snow_axis.parquet"; }
stage_mrms()     { $P "$HERE/mrms/download.py" --frames-dir "$WORK/raw/mrms" --start "$FIRST" --end "$LAST"
                   $P "$HERE/mrms/county_series.py" --frames-dir "$WORK/raw/mrms" --start "$FIRST" --end "$LAST" --out "$WORK/radar_series.parquet"; }
stage_assemble() { SNOW="$WORK/snow_axis.parquet"; [ -f "$SNOW" ] || SNOW="$HERE/reference/snow_axis_2025.parquet"
                   $P "$HERE/assemble_inputs.py" --start "$START" --end "$END" --out "$OUT" \
                      --weather-00z $(for Y in $RUN_YEARS; do echo "$WORK/hrrr00z/weather_${Y}_full_polygon_robust.parquet"; done) --weather-18z-dir "$WORK/hrrr18z/daily" \
                      --snow-axis "$SNOW" --outages $(for Y in $YEARS; do echo "$WORK/eaglei_counts_$Y.parquet"; done) --radar "$WORK/radar_series.parquet"
                   (cd "$OUT" && find . -type f | sort | sed 's|^\./||' | xargs sha256sum); }

for s in "${STAGES[@]}"; do echo "== stage $s"; "stage_$s"; done
