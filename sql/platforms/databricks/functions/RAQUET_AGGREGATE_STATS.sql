-- RAQUET_AGGREGATE_STATS for Databricks SQL Warehouse
-- Aggregates multiple tile statistics into a single result with correct stddev computation
--
-- This function properly combines statistics from multiple tiles using the parallel
-- variance algorithm, which correctly handles stddev aggregation (you can't just
-- average standard deviations).
--
-- Parameters:
--   stats_json_array: ARRAY<STRING> - Array of JSON strings from ST_RASTERSUMMARYSTATS
--     Each JSON should have: count, sum, mean, min, max, stddev
--
-- Returns: STRING (JSON) with count, sum, mean, min, max, stddev
--
-- Usage:
-- WITH tile_stats AS (
--     SELECT ${catalog}.${schema}.ST_RASTERSUMMARYSTATS(band_1, metadata, 0) AS stats
--     FROM raster_table
--     WHERE block != 0
-- )
-- SELECT ${catalog}.${schema}.RAQUET_AGGREGATE_STATS(COLLECT_LIST(stats)) AS region_stats
-- FROM tile_stats;

CREATE OR REPLACE FUNCTION ${catalog}.${schema}.RAQUET_AGGREGATE_STATS(
    stats_json_array ARRAY<STRING>
)
RETURNS STRING
LANGUAGE PYTHON
DETERMINISTIC
COMMENT 'Aggregates multiple tile statistics with correct parallel variance. Returns JSON: {count, sum, mean, min, max, stddev}'
AS $$
import json
import math

if stats_json_array is None or len(stats_json_array) == 0:
    return None

# Parse and filter valid stats
valid = []
for s_json in stats_json_array:
    if s_json is None:
        continue
    s = json.loads(s_json)
    if s.get('count', 0) > 0:
        valid.append(s)

if len(valid) == 0:
    return json.dumps({
        'count': 0,
        'sum': None,
        'mean': None,
        'min': None,
        'max': None,
        'stddev': None
    })

# Compute totals
total_count = 0
total_sum = 0.0
global_min = float('inf')
global_max = float('-inf')

for s in valid:
    total_count += s['count']
    total_sum += s['sum']
    if s['min'] < global_min:
        global_min = s['min']
    if s['max'] > global_max:
        global_max = s['max']

global_mean = total_sum / total_count

# Compute combined variance using parallel algorithm:
# Total variance = sum(n_i * (var_i + (mean_i - global_mean)^2)) / total_count
weighted_variance_sum = 0.0
for s in valid:
    variance = s['stddev'] ** 2
    mean_diff = s['mean'] - global_mean
    weighted_variance_sum += s['count'] * (variance + mean_diff ** 2)

global_stddev = math.sqrt(weighted_variance_sum / total_count)

return json.dumps({
    'count': total_count,
    'sum': total_sum,
    'mean': global_mean,
    'min': global_min if global_min != float('inf') else None,
    'max': global_max if global_max != float('-inf') else None,
    'stddev': global_stddev
})
$$;
