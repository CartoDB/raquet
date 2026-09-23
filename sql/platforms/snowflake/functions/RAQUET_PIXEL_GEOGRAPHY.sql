-- RAQUET_PIXEL_GEOGRAPHY (Snowflake version)
-- Returns the geographic center point of a pixel within a Raquet block
--
-- Each Raquet block is a 256x256 pixel grid. This function computes the
-- geographic coordinates (longitude, latitude) of a pixel's center point.
--
-- Parameters:
--   BLOCK: VARCHAR - The QUADBIN identifier of the block (VARCHAR for BigInt precision)
--   X: DOUBLE - Pixel x coordinate (0-255, left to right)
--   Y: DOUBLE - Pixel y coordinate (0-255, top to bottom)
--
-- Returns: GEOGRAPHY - The center point of the pixel
--
-- Note: y=0 is the top (north) of the block, y=255 is the bottom (south)
-- Requires: CARTO Analytics Toolbox (QUADBIN_BOUNDARY)
--
-- Usage:
--   SELECT RAQUET_DB.RAQUET.RAQUET_PIXEL_GEOGRAPHY(block::VARCHAR, 128, 128) AS pixel_center
--   FROM raster_table
--   WHERE block != 0

CREATE OR REPLACE FUNCTION RAQUET_DB.RAQUET.RAQUET_PIXEL_GEOGRAPHY(
    BLOCK VARCHAR,
    X DOUBLE,
    Y DOUBLE
)
RETURNS GEOGRAPHY
AS
$$
    SELECT ST_MAKEPOINT(
        ST_XMIN(geom) + (X + 0.5) * (ST_XMAX(geom) - ST_XMIN(geom)) / 256.0,
        ST_YMAX(geom) - (Y + 0.5) * (ST_YMAX(geom) - ST_YMIN(geom)) / 256.0
    )
    FROM (
        SELECT TO_GEOMETRY(CARTO_DEV_DATA.CARTO.QUADBIN_BOUNDARY(BLOCK::NUMBER)) AS geom
    )
$$;
