#!/usr/bin/env python3
"""
Benchmark the RaQuet SQL UDFs on 15GB slope raster (DC Metro + MD, 563K tiles).

Supports Snowflake and BigQuery.

Usage:
    python3 test/benchmark_15gb.py --engine snowflake
    python3 test/benchmark_15gb.py --engine bigquery
    python3 test/benchmark_15gb.py --engine snowflake --query A --runs 5

Setup (Snowflake):
    -- Create table from partitioned GCS files
    CREATE OR REPLACE STAGE raquet_15gb_stage
        URL = 'gcs://cartobq-raquet-libs/raquet-benchmark/15gb/slope_partitioned/'
        FILE_FORMAT = (TYPE = PARQUET BINARY_AS_TEXT = FALSE);

    COPY INTO SLOPE_15GB FROM @raquet_15gb_stage
        FILE_FORMAT = (TYPE = PARQUET BINARY_AS_TEXT = FALSE)
        MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

Setup (BigQuery):
    -- Load from GCS
    bq load --source_format=PARQUET \\
        yourproject:raquet.slope_15gb \\
        'gs://cartobq-raquet-libs/raquet-benchmark/15gb/slope_partitioned/*.parquet'
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from statistics import median

# ============================================================================
# Configuration
# ============================================================================

# Snowflake
SF_CONNECTION = os.environ.get("RAQUET_SF_CONNECTION", "raquet")
SF_DATABASE = os.environ.get("RAQUET_SF_DATABASE", "RAQUET_DB")
SF_SCHEMA = os.environ.get("RAQUET_SF_SCHEMA", "RAQUET")
SF_TABLE = f"{SF_DATABASE}.{SF_SCHEMA}.SLOPE_15GB"
SF_FN = f"{SF_DATABASE}.{SF_SCHEMA}"

# BigQuery
BQ_PROJECT = os.environ.get("RAQUET_BQ_PROJECT", "cartobq")
BQ_DATASET = os.environ.get("RAQUET_BQ_DATASET", "raquet")
BQ_TABLE = f"`{BQ_PROJECT}.{BQ_DATASET}.slope_15gb`"
BQ_FN = f"`{BQ_PROJECT}.{BQ_DATASET}`"

NUM_RUNS = 3

# Dataset covers DC metro + MD/DE: -77.54 to -75.76, 38.16 to 39.72
# 563,517 native-resolution tiles at zoom 17, ~73K with actual data (13% coverage)
# Dense data concentrated around -76.0, 39.2 (central MD)

# Test polygons — centered on dense data areas
SMALL_SITE = "POLYGON((-76.00 39.20, -75.99 39.20, -75.99 39.205, -76.00 39.205, -76.00 39.20))"  # ~0.5 km²
MEDIUM_SITE = "POLYGON((-76.05 39.15, -75.95 39.15, -75.95 39.25, -76.05 39.25, -76.05 39.15))"  # ~25 km²
LARGE_SITE = "POLYGON((-76.50 38.80, -76.00 38.80, -76.00 39.50, -76.50 39.50, -76.50 38.80))"   # ~3000 km²
FULL_AREA = "POLYGON((-77.54 38.16, -75.76 38.16, -75.76 39.72, -77.54 39.72, -77.54 38.16))"     # Full coverage


# ============================================================================
# Snowflake queries
# ============================================================================

def sf_query_a(polygon_wkt, mode='intersects'):
    mode_arg = f", '{mode}'" if mode != 'center' else ""
    return f"""
WITH metadata AS (
    SELECT metadata FROM {SF_TABLE} WHERE block = 0 LIMIT 1
),
site_blocks AS (
    SELECT f.VALUE::NUMBER AS block_id
    FROM metadata m,
    TABLE(FLATTEN({SF_FN}.__RAQUET_REGION_BLOCKS(
        ST_GEOGRAPHYFROMWKT('{polygon_wkt}'),
        17, 17{mode_arg}
    ))) f
),
tile_stats AS (
    SELECT
        {SF_FN}.ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats
    FROM {SF_TABLE} r
    JOIN site_blocks sb ON r.block = sb.block_id
    CROSS JOIN metadata m
    WHERE r.block != 0
)
SELECT
    SUM(stats:count::INT) as total_pixels,
    ROUND(SUM(stats:sum::FLOAT) / NULLIF(SUM(stats:count::INT), 0), 4) as mean_slope,
    MIN(stats:min::FLOAT) as min_slope,
    MAX(stats:max::FLOAT) as max_slope,
    COUNT(*) as num_tiles
FROM tile_stats;
"""


def sf_query_b():
    return f"""
WITH metadata AS (
    SELECT metadata FROM {SF_TABLE} WHERE block = 0 LIMIT 1
),
tile_stats AS (
    SELECT
        r.block,
        {SF_FN}.ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats
    FROM {SF_TABLE} r
    CROSS JOIN metadata m
    WHERE r.block != 0
    AND stats:count::INT > 0
)
SELECT
    COUNT(*) as total_cells,
    COUNT_IF(stats:mean::FLOAT < 3.0) as suitable_cells,
    ROUND(COUNT_IF(stats:mean::FLOAT < 3.0) * 100.0 / NULLIF(COUNT(*), 0), 1) as pct_suitable
FROM tile_stats;
"""


def sf_query_b_top20():
    return f"""
WITH metadata AS (
    SELECT metadata FROM {SF_TABLE} WHERE block = 0 LIMIT 1
),
tile_stats AS (
    SELECT
        r.block,
        {SF_FN}.ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats
    FROM {SF_TABLE} r
    CROSS JOIN metadata m
    WHERE r.block != 0
)
SELECT
    block,
    ROUND(stats:mean::FLOAT, 4) as mean_slope,
    ROUND(stats:max::FLOAT, 4) as max_slope,
    stats:count::INT as pixel_count
FROM tile_stats
WHERE stats:mean::FLOAT < 3.0 AND stats:count::INT > 0
ORDER BY mean_slope ASC
LIMIT 20;
"""


# ============================================================================
# BigQuery queries
# ============================================================================

def bq_query_a(polygon_wkt):
    return f"""
WITH metadata AS (
    SELECT ANY_VALUE(metadata) as metadata FROM {BQ_TABLE} WHERE block = 0
),
site_blocks AS (
    SELECT block FROM {BQ_FN}.__RAQUET_REGION_BLOCKS(
        ST_GEOGFROMTEXT('{polygon_wkt}'),
        17, 17
    )
),
tile_stats AS (
    SELECT
        {BQ_FN}.ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats
    FROM {BQ_TABLE} r
    JOIN site_blocks sb ON r.block = sb.block
    CROSS JOIN metadata m
    WHERE r.block != 0
)
SELECT
    SUM(CAST(stats.count AS INT64)) as total_pixels,
    ROUND(SUM(stats.sum) / NULLIF(SUM(CAST(stats.count AS INT64)), 0), 4) as mean_slope,
    MIN(stats.min) as min_slope,
    MAX(stats.max) as max_slope,
    COUNT(*) as num_tiles
FROM tile_stats;
"""


def bq_query_b():
    return f"""
WITH metadata AS (
    SELECT ANY_VALUE(metadata) as metadata FROM {BQ_TABLE} WHERE block = 0
),
tile_stats AS (
    SELECT
        r.block,
        {BQ_FN}.ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats
    FROM {BQ_TABLE} r
    CROSS JOIN metadata m
    WHERE r.block != 0
    AND CAST(stats.count AS INT64) > 0
)
SELECT
    COUNT(*) as total_cells,
    COUNTIF(stats.mean < 3.0) as suitable_cells,
    ROUND(COUNTIF(stats.mean < 3.0) * 100.0 / NULLIF(COUNT(*), 0), 1) as pct_suitable
FROM tile_stats;
"""


def bq_query_b_top20():
    return f"""
WITH metadata AS (
    SELECT ANY_VALUE(metadata) as metadata FROM {BQ_TABLE} WHERE block = 0
),
tile_stats AS (
    SELECT
        r.block,
        {BQ_FN}.ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats
    FROM {BQ_TABLE} r
    CROSS JOIN metadata m
    WHERE r.block != 0
)
SELECT
    block,
    ROUND(stats.mean, 4) as mean_slope,
    ROUND(stats.max, 4) as max_slope,
    CAST(stats.count AS INT64) as pixel_count
FROM tile_stats
WHERE stats.mean < 3.0 AND CAST(stats.count AS INT64) > 0
ORDER BY mean_slope ASC
LIMIT 20;
"""


# ============================================================================
# Engine runners
# ============================================================================

def run_snowflake(sql, timeout=600):
    """Execute SQL on Snowflake and return (result, elapsed_seconds)."""
    start = time.time()
    result = subprocess.run(
        ["snowsql", "-c", SF_CONNECTION, "-o", "output_format=json",
         "-o", "friendly=false", "-o", "timing=false", "-q", sql],
        capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"Snowflake error:\n{result.stderr.strip()}\n{result.stdout.strip()}")

    output = result.stdout.strip()
    if not output:
        return None, elapsed

    try:
        rows = json.loads(output)
        if isinstance(rows, list) and len(rows) > 0:
            return rows[0] if len(rows) == 1 else rows, elapsed
        return None, elapsed
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', output, re.DOTALL)
        if match:
            rows = json.loads(match.group())
            if isinstance(rows, list) and len(rows) > 0:
                return rows[0] if len(rows) == 1 else rows, elapsed
        return None, elapsed


def run_bigquery(sql, timeout=600):
    """Execute SQL on BigQuery and return (result, elapsed_seconds)."""
    start = time.time()
    result = subprocess.run(
        ["bq", "query", "--use_legacy_sql=false", "--format=json", "--max_rows=100", sql],
        capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"BigQuery error:\n{result.stderr.strip()}")

    output = result.stdout.strip()
    if not output:
        return None, elapsed

    try:
        rows = json.loads(output)
        if isinstance(rows, list) and len(rows) > 0:
            return rows[0] if len(rows) == 1 else rows, elapsed
        return None, elapsed
    except json.JSONDecodeError:
        return None, elapsed


# ============================================================================
# Benchmark runner
# ============================================================================

def run_benchmark(name, description, sql, engine_runner, num_runs=NUM_RUNS):
    """Run a query multiple times and return benchmark result."""
    print(f"\n  {description}")
    print(f"  Running {num_runs} times...")

    timings = []
    result = None

    for i in range(num_runs):
        try:
            r, elapsed = engine_runner(sql, timeout=1200)
            timings.append(round(elapsed, 2))
            if r is not None:
                result = r
            print(f"    Run {i+1}: {elapsed:.2f}s")
        except Exception as e:
            print(f"    Run {i+1}: ERROR - {e}")
            timings.append(None)

    valid_timings = [t for t in timings if t is not None]
    if not valid_timings:
        print(f"  FAILED - all runs errored")
        return None

    med = round(median(valid_timings), 2)
    print(f"  Median: {med}s | Result: {json.dumps(result, default=str)[:200]}")

    return {
        "name": name,
        "description": description,
        "median_seconds": med,
        "all_timings": timings,
        "result": result,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="15GB RaQuet SQL benchmark")
    parser.add_argument("--engine", choices=["snowflake", "bigquery"], required=True)
    parser.add_argument("--query", choices=["A", "B", "all"], default="all")
    parser.add_argument("--runs", type=int, default=NUM_RUNS)
    args = parser.parse_args()

    num_runs = args.runs

    # Select engine
    if args.engine == "snowflake":
        runner = run_snowflake
        gen_a = sf_query_a
        gen_b = sf_query_b
        gen_b20 = sf_query_b_top20
        table_info = SF_TABLE
    else:
        runner = run_bigquery
        gen_a = bq_query_a
        gen_b = bq_query_b
        gen_b20 = bq_query_b_top20
        table_info = BQ_TABLE

    print("=" * 60)
    print(f"  15GB Raquet-SQL Benchmark ({args.engine})")
    print(f"  Dataset: 15GB slope, 563K tiles, zoom 17")
    print(f"  Table: {table_info}")
    print(f"  Runs per query: {num_runs}")
    print("=" * 60)

    all_results = []

    if args.query in ("A", "all"):
        r = run_benchmark("a_small", "Query A: Small site (~0.5 km²)",
                         gen_a(SMALL_SITE), runner, num_runs)
        if r: all_results.append(r)

        r = run_benchmark("a_medium", "Query A: Medium site (~25 km²)",
                         gen_a(MEDIUM_SITE), runner, num_runs)
        if r: all_results.append(r)

        r = run_benchmark("a_large", "Query A: Large area (~900 km²)",
                         gen_a(LARGE_SITE), runner, num_runs)
        if r: all_results.append(r)

    if args.query in ("B", "all"):
        r = run_benchmark("b_full_scan", "Query B: Full area suitability scan (563K tiles)",
                         gen_b(), runner, num_runs)
        if r: all_results.append(r)

        r = run_benchmark("b_top20", "Query B: Top 20 flattest cells",
                         gen_b20(), runner, num_runs)
        if r: all_results.append(r)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", f"benchmark_15gb_{args.engine}_results.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*60}")
    print(f"\n  {'Query':<20} {'Median(s)':<12} {'Result Summary'}")
    print(f"  {'-'*20} {'-'*12} {'-'*30}")
    for r in all_results:
        res = r.get("result", {})
        summary = ""
        if isinstance(res, dict):
            for k in ["TOTAL_PIXELS", "total_pixels"]:
                if k in res:
                    summary = f"pixels={res[k]}, mean={res.get('MEAN_SLOPE', res.get('mean_slope', '?'))}"
                    break
            for k in ["TOTAL_CELLS", "total_cells"]:
                if k in res:
                    tc = res[k]
                    sc = res.get("SUITABLE_CELLS", res.get("suitable_cells", "?"))
                    pct = res.get("PCT_SUITABLE", res.get("pct_suitable", "?"))
                    summary = f"total={tc}, suitable={sc} ({pct}%)"
                    break
        elif isinstance(res, list):
            summary = f"{len(res)} rows"
        print(f"  {r['name']:<20} {r['median_seconds']:<12} {summary}")


if __name__ == "__main__":
    main()
