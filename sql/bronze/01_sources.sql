-- Bronze: the delivered input tables, unchanged.
-- Views over the Parquet/CSV files of one input directory; nothing is copied, cast or filtered here.
-- {{requests}}, {{outages}}, ... are replaced by the runner with read_parquet(...) or read_csv(...) of the file found,
-- always with a 0-based file_row_number column (the delivery order, used to break ties between record versions).

CREATE OR REPLACE VIEW bronze.requests AS SELECT * FROM {{requests}};
CREATE OR REPLACE VIEW bronze.customers AS SELECT * FROM {{mcc}};
CREATE OR REPLACE VIEW bronze.outages AS SELECT * FROM {{outages}};
CREATE OR REPLACE VIEW bronze.weather AS SELECT * FROM {{weather}};
CREATE OR REPLACE VIEW bronze.radar AS SELECT * FROM {{radar}};
