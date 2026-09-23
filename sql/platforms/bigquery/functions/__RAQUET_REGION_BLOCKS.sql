-- __RAQUET_REGION_BLOCKS
-- Table function that returns all block quadbins from a Raquet file that intersect a region
--
-- This handles multi-resolution Raquet files by polyfilling at each zoom level
-- from min_zoom to max_zoom and unioning the results.
--
-- Parameters:
--   geom: GEOGRAPHY - The region of interest
--   min_zoom: INT64 - Minimum zoom level in the Raquet file
--   max_zoom: INT64 - Maximum zoom level in the Raquet file
--   mode: STRING (optional) - 'intersects' (default), 'center', or 'contains'
--
-- Returns: TABLE<block INT64>
--   - block: QUADBIN identifiers that match the region per the selected mode
--
-- Usage:
--   -- Default (intersects, matches DuckDB behavior):
--   SELECT t.*
--   FROM `table` t
--   JOIN `cartobq.raquet.__RAQUET_REGION_BLOCKS`(
--       ST_GEOGFROMTEXT('POLYGON(...)'),
--       17, 17
--   ) rb ON t.block = rb.block
--
--   -- With explicit mode:
--   SELECT t.*
--   FROM `table` t
--   JOIN `cartobq.raquet.__RAQUET_REGION_BLOCKS`(
--       ST_GEOGFROMTEXT('POLYGON(...)'),
--       17, 17, 'center'
--   ) rb ON t.block = rb.block

-- 3-parameter version: intersects mode (default, matches DuckDB)
CREATE OR REPLACE TABLE FUNCTION `cartobq.raquet.__RAQUET_REGION_BLOCKS`(
    geom GEOGRAPHY,
    min_zoom INT64,
    max_zoom INT64
)
AS (
    SELECT DISTINCT block
    FROM UNNEST(GENERATE_ARRAY(min_zoom, max_zoom)) AS zoom,
    UNNEST(`carto-un`.carto.QUADBIN_POLYFILL_MODE(geom, zoom, 'intersects')) AS block
);

-- 4-parameter version: user-specified mode ('center', 'intersects', 'contains')
CREATE OR REPLACE TABLE FUNCTION `cartobq.raquet.__RAQUET_REGION_BLOCKS`(
    geom GEOGRAPHY,
    min_zoom INT64,
    max_zoom INT64,
    mode STRING
)
AS (
    SELECT DISTINCT block
    FROM UNNEST(GENERATE_ARRAY(min_zoom, max_zoom)) AS zoom,
    UNNEST(`carto-un`.carto.QUADBIN_POLYFILL_MODE(geom, zoom, mode)) AS block
);
