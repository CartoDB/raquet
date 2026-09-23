-- __RAQUET_RESOLVE_ZOOM (Snowflake version)
-- Converts flexible resolution parameter to actual zoom level
--
-- Accepts these input types:
--   - Numeric string ('0'-'26'): Use exact zoom level (clamped to available range)
--   - 'auto': Pick based on query area size (uses __RAQUET_AUTO_ZOOM)
--   - 'min': Use minimum (coarsest) resolution available
--   - 'max': Use maximum (finest) resolution available
--   - NULL: Defaults to 'auto'
--
-- Parameters:
--   RESOLUTION: VARCHAR - Resolution specifier ('0'-'26', 'auto', 'min', 'max', or NULL)
--   GEOMETRY: GEOGRAPHY - The region to query (used for 'auto' mode)
--   METADATA: VARCHAR - JSON metadata from the Raquet file
--   MAX_TILES: DOUBLE - Threshold for auto mode (default 50)
--
-- Returns: NUMBER - The resolved zoom level

CREATE OR REPLACE FUNCTION RAQUET_DB.RAQUET.__RAQUET_RESOLVE_ZOOM(
    RESOLUTION VARCHAR,
    GEOMETRY GEOGRAPHY,
    METADATA VARCHAR,
    MAX_TILES DOUBLE
)
RETURNS NUMBER
AS
$$
    SELECT CASE
        -- Numeric value: use exact zoom (clamped to available range)
        WHEN TRY_CAST(RESOLUTION AS INT) IS NOT NULL
        THEN LEAST(
            GREATEST(
                RESOLUTION::INT,
                PARSE_JSON(METADATA):tiling:min_zoom::INT
            ),
            PARSE_JSON(METADATA):tiling:max_zoom::INT
        )

        -- 'min': use min_zoom (coarsest available)
        WHEN LOWER(RESOLUTION) = 'min'
        THEN PARSE_JSON(METADATA):tiling:min_zoom::INT

        -- 'max': use max_zoom (finest available)
        WHEN LOWER(RESOLUTION) = 'max'
        THEN PARSE_JSON(METADATA):tiling:max_zoom::INT

        -- 'auto' or NULL: compute optimal zoom based on query area size
        ELSE RAQUET_DB.RAQUET.__RAQUET_AUTO_ZOOM(
            GEOMETRY,
            PARSE_JSON(METADATA):tiling:min_zoom::INT,
            PARSE_JSON(METADATA):tiling:max_zoom::INT,
            COALESCE(MAX_TILES, 50)
        )
    END
$$;
