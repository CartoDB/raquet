-- __RAQUET_AUTO_ZOOM (Snowflake version)
-- Math-based auto zoom calculator for overview pyramid optimization
--
-- Estimates tile count at each zoom level using bounding box area math,
-- then picks the highest resolution where estimated tiles <= threshold.
--
-- Parameters:
--   GEOMETRY: GEOGRAPHY - The region to query
--   MIN_ZOOM: DOUBLE - Minimum zoom level available in the Raquet file
--   MAX_ZOOM: DOUBLE - Maximum zoom level (native resolution)
--   MAX_TILES: DOUBLE - Threshold for tile count (default 50)
--
-- Returns: NUMBER - Recommended zoom level

CREATE OR REPLACE FUNCTION RAQUET_DB.RAQUET.__RAQUET_AUTO_ZOOM(
    GEOMETRY GEOGRAPHY,
    MIN_ZOOM DOUBLE,
    MAX_ZOOM DOUBLE,
    MAX_TILES DOUBLE
)
RETURNS NUMBER
AS
$$
    SELECT COALESCE(
        MAX(f.VALUE::INT),
        MIN_ZOOM::INT
    )
    FROM (
        SELECT
            ST_XMAX(TO_GEOMETRY(GEOMETRY)) - ST_XMIN(TO_GEOMETRY(GEOMETRY)) AS width_deg,
            ST_YMAX(TO_GEOMETRY(GEOMETRY)) - ST_YMIN(TO_GEOMETRY(GEOMETRY)) AS height_deg
    ) dims,
    TABLE(FLATTEN(ARRAY_GENERATE_RANGE(MIN_ZOOM::INT, MAX_ZOOM::INT + 1))) f
    WHERE CEIL(dims.width_deg / (360.0 / POW(2, f.VALUE::INT))) *
          CEIL(dims.height_deg / (180.0 / POW(2, f.VALUE::INT))) <= MAX_TILES
$$;
