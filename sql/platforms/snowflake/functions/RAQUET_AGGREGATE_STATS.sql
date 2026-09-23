-- RAQUET_AGGREGATE_STATS (Snowflake version)
-- Aggregates multiple tile statistics into a single result with correct stddev computation
--
-- This function properly combines statistics from multiple tiles using the parallel
-- variance algorithm, which correctly handles stddev aggregation (you can't just
-- average standard deviations).
--
-- Parameters:
--   stats_array: ARRAY - Array of tile statistics objects from ST_RASTERSUMMARYSTATS
--     Each element should have: count, sum, mean, min, max, stddev
--
-- Returns: OBJECT with count, sum, mean, min, max, stddev
--
-- Usage:
-- WITH tile_stats AS (
--     SELECT RAQUET_DB.RAQUET.ST_RASTERSUMMARYSTATS(band_1, metadata, 0) AS stats
--     FROM raster_table
--     WHERE block != 0
-- )
-- SELECT RAQUET_DB.RAQUET.RAQUET_AGGREGATE_STATS(ARRAY_AGG(stats)) AS region_stats
-- FROM tile_stats;

CREATE OR REPLACE FUNCTION RAQUET_DB.RAQUET.RAQUET_AGGREGATE_STATS(
    STATS_ARRAY ARRAY
)
RETURNS OBJECT
LANGUAGE JAVASCRIPT
AS
$$
if (!STATS_ARRAY || STATS_ARRAY.length === 0) {
    return null;
}

// Filter valid stats (count > 0)
const valid = [];
for (let i = 0; i < STATS_ARRAY.length; i++) {
    const s = STATS_ARRAY[i];
    if (s && s.count > 0) {
        valid.push(s);
    }
}

if (valid.length === 0) {
    return { count: 0, sum: null, mean: null, min: null, max: null, stddev: null };
}

// Compute totals
let totalCount = 0;
let totalSum = 0;
let globalMin = Infinity;
let globalMax = -Infinity;

for (let i = 0; i < valid.length; i++) {
    const s = valid[i];
    totalCount += s.count;
    totalSum += s.sum;
    if (s.min < globalMin) globalMin = s.min;
    if (s.max > globalMax) globalMax = s.max;
}

const globalMean = totalSum / totalCount;

// Compute combined variance using parallel algorithm:
// Total variance = sum(n_i * (var_i + (mean_i - global_mean)^2)) / total_count
let weightedVarianceSum = 0;
for (let i = 0; i < valid.length; i++) {
    const s = valid[i];
    const variance = s.stddev * s.stddev;
    const meanDiff = s.mean - globalMean;
    weightedVarianceSum += s.count * (variance + meanDiff * meanDiff);
}

const globalStddev = Math.sqrt(weightedVarianceSum / totalCount);

return {
    count: totalCount,
    sum: totalSum,
    mean: globalMean,
    min: globalMin === Infinity ? null : globalMin,
    max: globalMax === -Infinity ? null : globalMax,
    stddev: globalStddev
};
$$;
