-- ST_RASTERVALUE_GEOG
-- Gets the raster value at a geographic point (convenience wrapper that accepts GEOGRAPHY)
--
-- Parameters:
--   block: INT64 - The QUADBIN block identifier
--   band: BYTES - The compressed band data
--   point: GEOGRAPHY - The geographic point to query
--   metadata: STRING - JSON metadata from the Raquet file
--   band_index: INT64 - The band index (0-based)
--
-- Returns: FLOAT64 - The pixel value at the point

CREATE OR REPLACE FUNCTION `cartobq.raquet.ST_RASTERVALUE_GEOG`(
    block INT64,
    band BYTES,
    point GEOGRAPHY,
    metadata STRING,
    band_index INT64
)
RETURNS FLOAT64
AS (
    `cartobq.raquet.ST_RASTERVALUE`(
        block,
        band,
        ST_X(point),
        ST_Y(point),
        metadata,
        band_index
    )
);
