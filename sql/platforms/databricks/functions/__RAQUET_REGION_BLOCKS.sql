-- __RAQUET_REGION_BLOCKS for Databricks SQL Warehouse
-- Returns all block quadbins from a Raquet file that intersect a region
--
-- This handles multi-resolution Raquet files by polyfilling at each zoom level
-- from min_zoom to max_zoom and unioning the results.
--
-- Parameters:
--   geom_wkt: STRING - WKT geometry for the region of interest
--   min_zoom: INT - Minimum zoom level in the Raquet file
--   max_zoom: INT - Maximum zoom level in the Raquet file
--   mode: STRING (optional) - 'intersects' (default), 'center', or 'contains'
--
-- Returns: ARRAY<BIGINT> - Array of QUADBIN block identifiers
--
-- Note: BigQuery uses a TABLE FUNCTION for this. Databricks returns an ARRAY
-- which you can EXPLODE to get rows.
--
-- Requires: CARTO Analytics Toolbox (QUADBIN_POLYFILL_MODE)
--
-- Usage:
--   -- Default (intersects, matches DuckDB behavior):
--   SELECT EXPLODE(${catalog}.${schema}.__RAQUET_REGION_BLOCKS(
--       'POLYGON((...)))',
--       min_zoom, max_zoom
--   )) AS block;
--
--   -- With explicit mode:
--   SELECT EXPLODE(${catalog}.${schema}.__RAQUET_REGION_BLOCKS(
--       'POLYGON((...)))',
--       min_zoom, max_zoom, 'center'
--   )) AS block;

-- 3-parameter version: intersects mode (default, matches DuckDB)
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.__RAQUET_REGION_BLOCKS(
    geom_wkt STRING,
    min_zoom INT,
    max_zoom INT
)
RETURNS ARRAY<BIGINT>
LANGUAGE SQL
DETERMINISTIC
COMMENT 'Returns array of QUADBIN blocks intersecting a region across all zoom levels'
RETURN (
    SELECT COLLECT_SET(block)
    FROM (
        SELECT EXPLODE(QUADBIN_POLYFILL_MODE(
            ST_GEOMFROMTEXT(geom_wkt), zoom, 'intersects'
        )) AS block
        FROM (SELECT EXPLODE(SEQUENCE(min_zoom, max_zoom)) AS zoom)
    )
);

-- 4-parameter version: user-specified mode ('center', 'intersects', 'contains')
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.__RAQUET_REGION_BLOCKS(
    geom_wkt STRING,
    min_zoom INT,
    max_zoom INT,
    mode STRING
)
RETURNS ARRAY<BIGINT>
LANGUAGE SQL
DETERMINISTIC
COMMENT 'Returns array of QUADBIN blocks matching a region with specified mode'
RETURN (
    SELECT COLLECT_SET(block)
    FROM (
        SELECT EXPLODE(QUADBIN_POLYFILL_MODE(
            ST_GEOMFROMTEXT(geom_wkt), zoom, mode
        )) AS block
        FROM (SELECT EXPLODE(SEQUENCE(min_zoom, max_zoom)) AS zoom)
    )
);
