-- ST_RASTERVALUE_GEOG (Snowflake version)
-- Gets the raster value at a geographic point (convenience wrapper that accepts GEOGRAPHY)
--
-- Parameters:
--   BLOCK: VARCHAR - The QUADBIN block identifier (as string for precision)
--   BAND: BINARY - The compressed band data
--   POINT: GEOGRAPHY - The geographic point to query
--   METADATA: VARCHAR - JSON metadata from the Raquet file
--   BAND_INDEX: DOUBLE - The band index (0-based)
--
-- Returns: DOUBLE - The pixel value at the point

CREATE OR REPLACE FUNCTION RAQUET_DB.RAQUET.ST_RASTERVALUE_GEOG(
    BLOCK VARCHAR,
    BAND BINARY,
    POINT GEOGRAPHY,
    METADATA VARCHAR,
    BAND_INDEX DOUBLE
)
RETURNS DOUBLE
AS
$$
    RAQUET_DB.RAQUET.ST_RASTERVALUE(
        BLOCK,
        BAND,
        ST_X(POINT),
        ST_Y(POINT),
        METADATA,
        BAND_INDEX
    )
$$;
