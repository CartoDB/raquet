-- RAQUET_PIXEL_GEOGRAPHY
-- Returns the geographic center point of a pixel within a Raquet block
--
-- Each Raquet block is a 256x256 pixel grid. This function computes the
-- geographic coordinates (longitude, latitude) of a pixel's center point.
--
-- Parameters:
--   block: INT64 - The QUADBIN identifier of the block
--   x: INT64 - Pixel x coordinate (0-255, left to right)
--   y: INT64 - Pixel y coordinate (0-255, top to bottom)
--
-- Returns: GEOGRAPHY - The center point of the pixel
--
-- Note: y=0 is the top (north) of the block, y=255 is the bottom (south)
--
-- Usage:
--   SELECT `cartobq.raquet.RAQUET_PIXEL_GEOGRAPHY`(block, 128, 128) AS pixel_center
--   FROM raster_table
--   WHERE block != 0

CREATE OR REPLACE FUNCTION `cartobq.raquet.RAQUET_PIXEL_GEOGRAPHY`(
    block INT64,
    x INT64,
    y INT64
)
RETURNS GEOGRAPHY
AS (
    (
        WITH bounds AS (
            SELECT
                ST_BOUNDINGBOX(`carto-un`.carto.QUADBIN_BOUNDARY(block)) AS bbox
        ),
        dims AS (
            SELECT
                bbox.xmin AS min_lon,
                bbox.xmax AS max_lon,
                bbox.ymin AS min_lat,
                bbox.ymax AS max_lat,
                (bbox.xmax - bbox.xmin) / 256.0 AS pixel_width,
                (bbox.ymax - bbox.ymin) / 256.0 AS pixel_height
            FROM bounds
        )
        SELECT ST_GEOGPOINT(
            min_lon + (x + 0.5) * pixel_width,
            max_lat - (y + 0.5) * pixel_height
        )
        FROM dims
    )
);
