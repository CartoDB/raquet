-- ============================================================================
-- Setup for 15GB slope benchmark on Snowflake and BigQuery
-- ============================================================================

-- ============================================================================
-- SNOWFLAKE
-- ============================================================================

-- 1. Create stage pointing to GCS partitioned files
CREATE OR REPLACE STAGE RAQUET_DB.RAQUET.RAQUET_15GB_STAGE
    URL = 'gcs://cartobq-raquet-libs/raquet-benchmark/15gb/slope_partitioned/'
    FILE_FORMAT = (TYPE = PARQUET BINARY_AS_TEXT = FALSE);

-- 2. Check what's in the stage
LIST @RAQUET_DB.RAQUET.RAQUET_15GB_STAGE;

-- 3. Load into table
CREATE OR REPLACE TABLE RAQUET_DB.RAQUET.SLOPE_15GB (
    block NUMBER,
    band_1 BINARY,
    metadata VARCHAR
);

COPY INTO RAQUET_DB.RAQUET.SLOPE_15GB
FROM @RAQUET_DB.RAQUET.RAQUET_15GB_STAGE
FILE_FORMAT = (TYPE = PARQUET BINARY_AS_TEXT = FALSE)
MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- 4. Verify
SELECT COUNT(*) as total_rows,
       COUNT_IF(block = 0) as metadata_rows,
       COUNT_IF(block != 0) as data_rows
FROM RAQUET_DB.RAQUET.SLOPE_15GB;

-- Expected: ~563K+ rows (563517 data + 156 metadata from partitioned files)

-- 5. Cluster on block for QUADBIN spatial pruning
ALTER TABLE RAQUET_DB.RAQUET.SLOPE_15GB CLUSTER BY (block);

-- ============================================================================
-- BIGQUERY
-- ============================================================================

-- Option A: Load from GCS into native table
-- bq load --source_format=PARQUET \
--     cartobq:raquet.slope_15gb \
--     'gs://cartobq-raquet-libs/raquet-benchmark/15gb/slope_partitioned/*.parquet'

-- Option B: External table (reads directly from GCS, slower but no import needed)
-- CREATE OR REPLACE EXTERNAL TABLE `cartobq.raquet.slope_15gb_external`
-- WITH PARTITION COLUMNS
-- OPTIONS (
--   format = 'PARQUET',
--   uris = ['gs://cartobq-raquet-libs/raquet-benchmark/15gb/slope_partitioned/*.parquet']
-- );

-- Verify
-- SELECT COUNT(*) as total_rows,
--        COUNTIF(block = 0) as metadata_rows,
--        COUNTIF(block != 0) as data_rows
-- FROM `cartobq.raquet.slope_15gb`;
