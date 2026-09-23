-- __RAQUET_RESOLVE_ZOOM
-- Converts flexible resolution parameter to actual zoom level
--
-- Accepts these input types:
--   - Numeric string ('0'-'26'): Use exact zoom level (clamped to available range)
--   - 'auto': Pick based on query area size (uses __RAQUET_AUTO_ZOOM)
--   - 'min': Use minimum (coarsest) resolution available
--   - 'max': Use maximum (finest) resolution available
--   - NULL: Defaults to 'auto'
--
-- Out-of-range zoom requests are clamped to [min_zoom, max_zoom]:
--   - zoom < min_zoom → returns min_zoom (graceful fallback)
--   - zoom > max_zoom → returns max_zoom (best available)
--
-- Parameters:
--   resolution: STRING - Resolution specifier ('0'-'26', 'auto', 'min', 'max', or NULL)
--   geometry: GEOGRAPHY - The region to query (used for 'auto' mode)
--   metadata: STRING - JSON metadata from the Raquet file
--   max_tiles: INT64 - Threshold for auto mode (default 50)
--
-- Returns: INT64 - The resolved zoom level
--
-- Usage:
--   SELECT `cartobq.raquet.__RAQUET_RESOLVE_ZOOM`('auto', geom, metadata, 50);
--   SELECT `cartobq.raquet.__RAQUET_RESOLVE_ZOOM`('5', geom, metadata, 50);
--   SELECT `cartobq.raquet.__RAQUET_RESOLVE_ZOOM`('min', geom, metadata, 50);
--   SELECT `cartobq.raquet.__RAQUET_RESOLVE_ZOOM`('max', geom, metadata, 50);

CREATE OR REPLACE FUNCTION `cartobq.raquet.__RAQUET_RESOLVE_ZOOM`(
    resolution STRING,
    geometry GEOGRAPHY,
    metadata STRING,
    max_tiles INT64
)
RETURNS INT64
AS (
    CASE
        -- Numeric value: use exact zoom (clamped to max available)
        WHEN SAFE_CAST(resolution AS INT64) IS NOT NULL
        THEN LEAST(
            GREATEST(
                SAFE_CAST(resolution AS INT64),
                CAST(JSON_VALUE(metadata, '$.tiling.min_zoom') AS INT64)
            ),
            CAST(JSON_VALUE(metadata, '$.tiling.max_zoom') AS INT64)
        )

        -- 'min': use min_zoom (coarsest available, useful for quick previews)
        WHEN LOWER(resolution) = 'min'
        THEN CAST(JSON_VALUE(metadata, '$.tiling.min_zoom') AS INT64)

        -- 'max': use max_zoom (finest available)
        WHEN LOWER(resolution) = 'max'
        THEN CAST(JSON_VALUE(metadata, '$.tiling.max_zoom') AS INT64)

        -- 'auto' or NULL: compute optimal zoom based on query area size
        ELSE `cartobq.raquet.__RAQUET_AUTO_ZOOM`(
            geometry,
            CAST(JSON_VALUE(metadata, '$.tiling.min_zoom') AS INT64),
            CAST(JSON_VALUE(metadata, '$.tiling.max_zoom') AS INT64),
            COALESCE(max_tiles, 50)
        )
    END
);
