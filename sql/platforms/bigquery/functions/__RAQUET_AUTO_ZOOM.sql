-- __RAQUET_AUTO_ZOOM
-- Math-based auto zoom calculator for overview pyramid optimization
--
-- Estimates tile count at each zoom level using bounding box area math,
-- then picks the highest resolution where estimated tiles ≤ threshold.
-- This is fast (no QUADBIN_POLYFILL calls) and accurate enough for most use cases.
--
-- Algorithm:
--   At zoom Z, each tile covers approximately:
--     - width:  360 / 2^Z degrees
--     - height: 180 / 2^Z degrees
--   Estimated tiles = ceil(bbox_width / tile_width) × ceil(bbox_height / tile_height)
--   Pick highest Z where estimated_tiles ≤ max_tiles
--
-- Parameters:
--   geometry: GEOGRAPHY - The region to query
--   min_zoom: INT64 - Minimum zoom level available in the Raquet file
--   max_zoom: INT64 - Maximum zoom level (native resolution)
--   max_tiles: INT64 - Threshold for tile count (default 50)
--
-- Returns: INT64 - Recommended zoom level

CREATE OR REPLACE FUNCTION `cartobq.raquet.__RAQUET_AUTO_ZOOM`(
    geometry GEOGRAPHY,
    min_zoom INT64,
    max_zoom INT64,
    max_tiles INT64
)
RETURNS INT64
AS (
    (
        WITH bbox AS (
            SELECT ST_BOUNDINGBOX(geometry) AS b
        ),
        dims AS (
            SELECT
                b.xmax - b.xmin AS width_deg,
                b.ymax - b.ymin AS height_deg
            FROM bbox
        ),
        -- Calculate estimated tile count at each zoom level
        zoom_estimates AS (
            SELECT
                z,
                -- Tile size at zoom z: 360/2^z degrees wide, ~180/2^z tall
                CEIL(width_deg / (360.0 / POW(2, z))) *
                CEIL(height_deg / (180.0 / POW(2, z))) AS est_tiles
            FROM dims,
            UNNEST(GENERATE_ARRAY(min_zoom, max_zoom)) AS z
        )
        -- Pick highest zoom where tiles ≤ threshold
        SELECT COALESCE(
            MAX(z),
            min_zoom  -- fallback if all exceed threshold
        )
        FROM zoom_estimates
        WHERE est_tiles <= max_tiles
    )
);
