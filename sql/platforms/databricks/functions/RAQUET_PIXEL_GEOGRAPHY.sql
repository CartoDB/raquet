-- RAQUET_PIXEL_GEOGRAPHY for Databricks SQL Warehouse
-- Returns the geographic center point (as WKT) of a pixel within a Raquet block
--
-- Each Raquet block is a 256x256 pixel grid. This function computes the
-- geographic coordinates (longitude, latitude) of a pixel's center point.
--
-- Parameters:
--   block: BIGINT - The QUADBIN identifier of the block
--   x: INT - Pixel x coordinate (0-255, left to right)
--   y: INT - Pixel y coordinate (0-255, top to bottom)
--
-- Returns: STRING - WKT POINT for the pixel center (use ST_GEOMFROMTEXT to convert)
--
-- Note: y=0 is the top (north) of the block, y=255 is the bottom (south)
-- Requires: CARTO Analytics Toolbox (QUADBIN_BOUNDARY)
--
-- Usage:
--   SELECT ${catalog}.${schema}.RAQUET_PIXEL_GEOGRAPHY(block, 128, 128) AS pixel_center
--   FROM raster_table
--   WHERE block != 0

CREATE OR REPLACE FUNCTION ${catalog}.${schema}.RAQUET_PIXEL_GEOGRAPHY(
    block BIGINT,
    x INT,
    y INT
)
RETURNS STRING
LANGUAGE SQL
DETERMINISTIC
COMMENT 'Returns WKT POINT for pixel center. Use ST_GEOMFROMTEXT() to convert to geometry.'
RETURN (
    SELECT ST_ASTEXT(ST_POINT(
        ST_XMIN(boundary) + (x + 0.5) * (ST_XMAX(boundary) - ST_XMIN(boundary)) / 256.0,
        ST_YMAX(boundary) - (y + 0.5) * (ST_YMAX(boundary) - ST_YMIN(boundary)) / 256.0
    ))
    FROM (SELECT QUADBIN_BOUNDARY(block) AS boundary)
);
