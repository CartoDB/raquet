-- RAQUET_AGGREGATE_STATS
-- Aggregates multiple tile statistics into a single result with correct stddev computation
--
-- This function properly combines statistics from multiple tiles using the parallel
-- variance algorithm, which correctly handles stddev aggregation (you can't just
-- average standard deviations).
--
-- Parameters:
--   stats_array: ARRAY<STRUCT<count INT64, sum FLOAT64, mean FLOAT64, min FLOAT64, max FLOAT64, stddev FLOAT64>>
--     Array of tile statistics from ST_RASTERSUMMARYSTATS
--
-- Returns: STRUCT<count INT64, sum FLOAT64, mean FLOAT64, min FLOAT64, max FLOAT64, stddev FLOAT64>
--
-- Usage with QUADBIN_POLYFILL for region statistics:
--
-- WITH metadata AS (
--     SELECT metadata FROM `project.dataset.raster` WHERE block = 0 LIMIT 1
-- ),
-- region_blocks AS (
--     SELECT block
--     FROM UNNEST(`carto-un`.carto.QUADBIN_POLYFILL_MODE(
--         ST_GEOGFROMTEXT('POLYGON((...))'),
--         CAST(JSON_VALUE(m.metadata, '$.block_resolution') AS INT64),
--         'intersects'
--     )) AS block,
--     metadata m
-- ),
-- tile_stats AS (
--     SELECT `cartobq.raquet.ST_RASTERSUMMARYSTATS`(r.band_1, m.metadata, 0) AS stats
--     FROM `project.dataset.raster` r
--     JOIN region_blocks rb ON r.block = rb.block
--     CROSS JOIN metadata m
-- )
-- SELECT `cartobq.raquet.RAQUET_AGGREGATE_STATS`(ARRAY_AGG(stats)) AS region_stats
-- FROM tile_stats;

CREATE OR REPLACE FUNCTION `cartobq.raquet.RAQUET_AGGREGATE_STATS`(
    stats_array ARRAY<STRUCT<count INT64, sum FLOAT64, mean FLOAT64, min FLOAT64, max FLOAT64, stddev FLOAT64>>
)
RETURNS STRUCT<count INT64, sum FLOAT64, mean FLOAT64, min FLOAT64, max FLOAT64, stddev FLOAT64>
AS (
    -- Filter out null stats and compute aggregates
    (
        WITH valid_stats AS (
            SELECT s.*
            FROM UNNEST(stats_array) AS s
            WHERE s.count > 0
        ),
        totals AS (
            SELECT
                SUM(count) AS total_count,
                SUM(sum) AS total_sum,
                MIN(min) AS global_min,
                MAX(max) AS global_max
            FROM valid_stats
        ),
        -- Compute combined variance using parallel algorithm:
        -- Total variance = Σ(nᵢ * (varᵢ + (μᵢ - μ)²)) / n
        -- where varᵢ = stddevᵢ², μᵢ = mean of tile i, μ = overall mean
        variance_components AS (
            SELECT
                t.total_count,
                t.total_sum,
                t.global_min,
                t.global_max,
                t.total_sum / t.total_count AS global_mean,
                SUM(
                    s.count * (
                        POW(s.stddev, 2) + POW(s.mean - (t.total_sum / t.total_count), 2)
                    )
                ) AS weighted_variance_sum
            FROM valid_stats s
            CROSS JOIN totals t
            WHERE t.total_count > 0
            GROUP BY t.total_count, t.total_sum, t.global_min, t.global_max
        )
        SELECT AS STRUCT
            CAST(total_count AS INT64) AS count,
            total_sum AS sum,
            global_mean AS mean,
            global_min AS min,
            global_max AS max,
            SQRT(weighted_variance_sum / total_count) AS stddev
        FROM variance_components
    )
);
