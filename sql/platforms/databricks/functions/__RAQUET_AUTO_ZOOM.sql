-- __RAQUET_AUTO_ZOOM for Databricks SQL Warehouse
-- Math-based auto zoom calculator for overview pyramid optimization
--
-- Estimates tile count at each zoom level using bounding box area math,
-- then picks the highest resolution where estimated tiles <= threshold.
--
-- Parameters:
--   geom_wkt: STRING - WKT geometry for the region to query
--   min_zoom: INT - Minimum zoom level available in the Raquet file
--   max_zoom: INT - Maximum zoom level (native resolution)
--   max_tiles: INT - Threshold for tile count (default 50)
--
-- Returns: INT - Recommended zoom level

CREATE OR REPLACE FUNCTION ${catalog}.${schema}.__RAQUET_AUTO_ZOOM(
    geom_wkt STRING,
    min_zoom INT,
    max_zoom INT,
    max_tiles INT
)
RETURNS INT
LANGUAGE SQL
DETERMINISTIC
COMMENT 'Auto-detect optimal zoom level based on query area size'
RETURN (
    SELECT COALESCE(
        MAX(z),
        min_zoom
    )
    FROM (
        SELECT
            z,
            CEIL(width_deg / (360.0 / POW(2, z))) *
            CEIL(height_deg / (180.0 / POW(2, z))) AS est_tiles
        FROM (
            SELECT
                ST_XMAX(ST_GEOMFROMTEXT(geom_wkt)) - ST_XMIN(ST_GEOMFROMTEXT(geom_wkt)) AS width_deg,
                ST_YMAX(ST_GEOMFROMTEXT(geom_wkt)) - ST_YMIN(ST_GEOMFROMTEXT(geom_wkt)) AS height_deg
        ) dims,
        (SELECT EXPLODE(SEQUENCE(min_zoom, max_zoom)) AS z)
    )
    WHERE est_tiles <= max_tiles
);
