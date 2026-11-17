-- Silver: typed, normalised tables with an explicit grain. Every revision of every record is kept;
-- which version counts is decided later, per issue time (gold) or finally (labels).
-- Times are UTC epoch seconds (DOUBLE, fractions kept) next to the TIMESTAMPTZ value.

-- grain: one row per delivered forecast request
CREATE OR REPLACE TABLE silver.requests AS
SELECT file_row_number                                         AS row_id,
       trim(task_id::VARCHAR)                                  AS task,
       trim(architecture_id::VARCHAR)                          AS arch,
       lpad(trim(county_fips::VARCHAR), 5, '0')                AS county,
       CAST(trim(issue_time::VARCHAR) AS TIMESTAMPTZ)          AS issue_time,
       CAST(trim(target_time::VARCHAR) AS TIMESTAMPTZ)         AS target_time,
       epoch(CAST(trim(issue_time::VARCHAR) AS TIMESTAMPTZ))   AS issue_s,
       epoch(CAST(trim(target_time::VARCHAR) AS TIMESTAMPTZ))  AS target_s
FROM bronze.requests;

-- grain: one row per county (customers served = denominator of the outage share)
CREATE OR REPLACE TABLE silver.customers AS
SELECT lpad(trim(county_fips::VARCHAR), 5, '0') AS county,
       CAST(total_customers AS BIGINT)          AS total_customers
FROM bronze.customers;

-- grain: one row per delivered outage record (county, observation time, publication time, delivery row);
-- several rows per observation time are revisions; a NULL count is a blank record
CREATE OR REPLACE TABLE silver.outage_versions AS
SELECT file_row_number                                                 AS row_id,
       lpad(trim(county_fips::VARCHAR), 5, '0')                        AS county,
       epoch(CAST(trim("timestamp"::VARCHAR) AS TIMESTAMPTZ))          AS ts_s,
       epoch(CAST(trim(source_record_time::VARCHAR) AS TIMESTAMPTZ))   AS src_s,
       TRY_CAST(customers_out AS DOUBLE)                               AS customers_out,
       epoch(CAST(trim("timestamp"::VARCHAR) AS TIMESTAMPTZ)) % 900 = 0 AS on_grid   -- 15-minute model grid
FROM bronze.outages;

-- grain: one row per (task, county, issue time) that has at least one request
CREATE OR REPLACE TABLE silver.issues AS
SELECT DISTINCT task, county, issue_s, issue_time
FROM silver.requests;
