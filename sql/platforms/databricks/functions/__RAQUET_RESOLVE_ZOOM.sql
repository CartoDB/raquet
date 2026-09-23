-- __RAQUET_RESOLVE_ZOOM for Databricks SQL Warehouse
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
--   resolution: STRING - Resolution specifier ('0'-'26', 'auto', 'min', 'max', or NULL)
--   geom_wkt: STRING - WKT geometry for the region (used for 'auto' mode)
--   metadata: STRING - JSON metadata from the Raquet file
--   max_tiles: INT - Threshold for auto mode (default 50)
--
-- Returns: INT - The resolved zoom level

CREATE OR REPLACE FUNCTION ${catalog}.${schema}.__RAQUET_RESOLVE_ZOOM(
    resolution STRING,
    geom_wkt STRING,
    metadata STRING,
    max_tiles INT
)
RETURNS INT
LANGUAGE SQL
DETERMINISTIC
COMMENT 'Resolve flexible resolution parameter to actual zoom level'
RETURN
    CASE
        -- Numeric value: use exact zoom (clamped to available range)
        WHEN CAST(resolution AS INT) IS NOT NULL
        THEN LEAST(
            GREATEST(
                CAST(resolution AS INT),
                CAST(GET_JSON_OBJECT(metadata, '$.tiling.min_zoom') AS INT)
            ),
            CAST(GET_JSON_OBJECT(metadata, '$.tiling.max_zoom') AS INT)
        )

        -- 'min': use min_zoom (coarsest available)
        WHEN LOWER(resolution) = 'min'
        THEN CAST(GET_JSON_OBJECT(metadata, '$.tiling.min_zoom') AS INT)

        -- 'max': use max_zoom (finest available)
        WHEN LOWER(resolution) = 'max'
        THEN CAST(GET_JSON_OBJECT(metadata, '$.tiling.max_zoom') AS INT)

        -- 'auto' or NULL: compute optimal zoom based on query area size
        ELSE ${catalog}.${schema}.__RAQUET_AUTO_ZOOM(
            geom_wkt,
            CAST(GET_JSON_OBJECT(metadata, '$.tiling.min_zoom') AS INT),
            CAST(GET_JSON_OBJECT(metadata, '$.tiling.max_zoom') AS INT),
            COALESCE(max_tiles, 50)
        )
    END;
