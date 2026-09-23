#!/usr/bin/env python3
"""
Snowflake benchmark for RaQuet SQL UDFs on 4.1GB slope raster.

Usage:
    python3 test/benchmark_snowflake.py                    # Run all benchmarks
    python3 test/benchmark_snowflake.py --table imported   # Only imported table
    python3 test/benchmark_snowflake.py --table external   # Only external table
    python3 test/benchmark_snowflake.py --query A          # Only Query A variants
    python3 test/benchmark_snowflake.py --runs 5           # 5 runs per query
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
SF_CONNECTION = os.environ.get("RAQUET_SF_CONNECTION", "raquet")
SF_DATABASE = os.environ.get("RAQUET_SF_DATABASE", "RAQUET_DB")
SF_SCHEMA = os.environ.get("RAQUET_SF_SCHEMA", "RAQUET")

IMPORTED_TABLE = f"{SF_DATABASE}.{SF_SCHEMA}.SLOPE"
EXTERNAL_TABLE = f"{SF_DATABASE}.{SF_SCHEMA}.SLOPE_EXTERNAL"

NUM_RUNS = 3

# Test polygons (Loudoun County, Virginia — "Data Center Alley")
SMALL_SITE = "POLYGON((-77.83 38.86, -77.82 38.86, -77.82 38.865, -77.83 38.865, -77.83 38.86))"
MEDIUM_SITE = "POLYGON((-77.85 38.85, -77.83 38.85, -77.83 38.87, -77.85 38.87, -77.85 38.85))"
LARGE_SITE = "POLYGON((-77.88 38.82, -77.78 38.82, -77.78 38.90, -77.88 38.90, -77.88 38.82))"
FULL_AREA = "POLYGON((-77.90 38.82, -77.19 38.82, -77.19 39.37, -77.90 39.37, -77.90 38.82))"

FN = f"{SF_DATABASE}.{SF_SCHEMA}"


def run_snowflake(sql, timeout=600):
    """Execute SQL on Snowflake and return (result_dict, elapsed_seconds)."""
    start = time.time()
    result = subprocess.run(
        ["snowsql", "-c", SF_CONNECTION, "-o", "output_format=json",
         "-o", "friendly=false", "-o", "timing=false", "-q", sql],
        capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        raise RuntimeError(f"Snowflake error:\n{stderr}\n{stdout}")

    # Parse JSON output from snowsql
    output = result.stdout.strip()
    if not output:
        return None, elapsed

    try:
        rows = json.loads(output)
        if isinstance(rows, list) and len(rows) > 0:
            return rows[0] if len(rows) == 1 else rows, elapsed
        return None, elapsed
    except json.JSONDecodeError:
        # Sometimes snowsql wraps output; try to extract JSON
        # Look for JSON array in output
        match = re.search(r'\[.*\]', output, re.DOTALL)
        if match:
            rows = json.loads(match.group())
            if isinstance(rows, list) and len(rows) > 0:
                return rows[0] if len(rows) == 1 else rows, elapsed
        print(f"  WARNING: Could not parse output: {output[:200]}")
        return None, elapsed


def check_table_exists(table_name):
    """Check if a table exists in Snowflake."""
    try:
        result = subprocess.run(
            ["snowsql", "-c", SF_CONNECTION, "-o", "output_format=json",
             "-o", "friendly=false", "-o", "timing=false",
             "-q", f"SELECT COUNT(*) as cnt FROM {table_name} LIMIT 1;"],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0 and "CNT" in result.stdout.upper()
    except Exception:
        return False


# ============================================================================
# Query generators
# ============================================================================

def query_a_site_stats(table, polygon_wkt):
    """Query A: Aggregate slope stats within a polygon."""
    return f"""
WITH metadata AS (
    SELECT metadata FROM {table} WHERE block = 0 LIMIT 1
),
site_blocks AS (
    SELECT f.VALUE::NUMBER AS block_id
    FROM metadata m,
    TABLE(FLATTEN({FN}.__RAQUET_REGION_BLOCKS(
        ST_GEOGRAPHYFROMWKT('{polygon_wkt}'),
        17, 17
    ))) f
),
tile_stats AS (
    SELECT
        {FN}.ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats
    FROM {table} r
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


def query_b_suitable_cells(table):
    """Query B: Find suitable cells (mean slope < 3°) in full area.

    Note: Uses full table scan + post-filter instead of JOIN with
    __RAQUET_REGION_BLOCKS, as the JOIN pattern causes internal errors
    on Snowflake Small warehouses with 67K+ tiles.
    """
    return f"""
WITH metadata AS (
    SELECT metadata FROM {table} WHERE block = 0 LIMIT 1
),
region_blocks AS (
    SELECT f.VALUE::NUMBER AS block_id
    FROM TABLE(FLATTEN({FN}.__RAQUET_REGION_BLOCKS(
        ST_GEOGRAPHYFROMWKT('{FULL_AREA}'),
        17, 17
    ))) f
),
tile_stats AS (
    SELECT
        r.block,
        {FN}.ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats
    FROM {table} r
    CROSS JOIN metadata m
    WHERE r.block != 0
),
filtered AS (
    SELECT block, stats
    FROM tile_stats
    WHERE block IN (SELECT block_id FROM region_blocks)
    AND stats:count::INT > 0
)
SELECT
    COUNT(*) as total_cells,
    COUNT_IF(stats:mean::FLOAT < 3.0) as suitable_cells,
    ROUND(COUNT_IF(stats:mean::FLOAT < 3.0) * 100.0 / NULLIF(COUNT(*), 0), 1) as pct_suitable
FROM filtered;
"""


def query_b_top20(table):
    """Query B Detail: Top 20 flattest cells.

    Uses full table scan + post-filter to avoid JOIN crash.
    """
    return f"""
WITH metadata AS (
    SELECT metadata FROM {table} WHERE block = 0 LIMIT 1
),
region_blocks AS (
    SELECT f.VALUE::NUMBER AS block_id
    FROM TABLE(FLATTEN({FN}.__RAQUET_REGION_BLOCKS(
        ST_GEOGRAPHYFROMWKT('{FULL_AREA}'),
        17, 17
    ))) f
),
tile_stats AS (
    SELECT
        r.block,
        {FN}.ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats
    FROM {table} r
    CROSS JOIN metadata m
    WHERE r.block != 0
)
SELECT
    block,
    ROUND(stats:mean::FLOAT, 4) as mean_slope,
    ROUND(stats:max::FLOAT, 4) as max_slope,
    stats:count::INT as pixel_count
FROM tile_stats
WHERE block IN (SELECT block_id FROM region_blocks)
AND stats:mean::FLOAT < 3.0 AND stats:count::INT > 0
ORDER BY mean_slope ASC
LIMIT 20;
"""


# ============================================================================
# Benchmark runner
# ============================================================================

def run_benchmark(name, description, sql, num_runs=NUM_RUNS, storage="imported"):
    """Run a query multiple times and return benchmark result."""
    print(f"\n  [{storage}] {description}")
    print(f"  Running {num_runs} times...")

    timings = []
    result = None

    for i in range(num_runs):
        try:
            r, elapsed = run_snowflake(sql, timeout=600)
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
        "engine": "snowflake",
        "storage": f"{'external_gcs' if storage == 'external' else storage}",
        "query": name.split("_")[0].upper(),
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
    parser = argparse.ArgumentParser(description="Snowflake RaQuet SQL benchmark")
    parser.add_argument("--table", choices=["imported", "external", "both"], default="both")
    parser.add_argument("--query", choices=["A", "B", "all"], default="all")
    parser.add_argument("--runs", type=int, default=NUM_RUNS)
    args = parser.parse_args()

    num_runs = args.runs
    all_results = []

    tables_to_test = []
    if args.table in ("imported", "both"):
        if check_table_exists(IMPORTED_TABLE):
            tables_to_test.append(("imported", IMPORTED_TABLE))
        else:
            print(f"WARNING: Imported table {IMPORTED_TABLE} not found, skipping")
    if args.table in ("external", "both"):
        if check_table_exists(EXTERNAL_TABLE):
            tables_to_test.append(("external", EXTERNAL_TABLE))
        else:
            print(f"WARNING: External table {EXTERNAL_TABLE} not found, skipping")

    if not tables_to_test:
        print("ERROR: No tables available for benchmarking")
        sys.exit(1)

    print("=" * 60)
    print("  Snowflake Raquet-SQL Benchmark")
    print("  Dataset: slope_masked.parquet (4.1 GB, 91,701 tiles)")
    print(f"  Tables: {', '.join(s for s, _ in tables_to_test)}")
    print(f"  Runs per query: {num_runs}")
    print("=" * 60)

    for storage, table in tables_to_test:
        print(f"\n{'='*60}")
        print(f"  Storage: {storage} ({table})")
        print(f"{'='*60}")

        if args.query in ("A", "all"):
            # Query A: Small site
            r = run_benchmark(
                "a_small_site",
                f"Small candidate site (~0.5 km²)",
                query_a_site_stats(table, SMALL_SITE),
                num_runs, storage,
            )
            if r: all_results.append(r)

            # Query A: Medium site
            r = run_benchmark(
                "a_medium_site",
                f"Medium candidate site (~4 km²)",
                query_a_site_stats(table, MEDIUM_SITE),
                num_runs, storage,
            )
            if r: all_results.append(r)

            # Query A: Large site
            r = run_benchmark(
                "a_large_site",
                f"Large candidate site (~50 km²)",
                query_a_site_stats(table, LARGE_SITE),
                num_runs, storage,
            )
            if r: all_results.append(r)

        if args.query in ("B", "all"):
            # Query B: Full area search
            r = run_benchmark(
                "b_full_area",
                f"Full area suitability scan ({FULL_AREA[:40]}...)",
                query_b_suitable_cells(table),
                num_runs, storage,
            )
            if r: all_results.append(r)

            # Query B Detail: Top 20
            r = run_benchmark(
                "b_top20",
                f"Top 20 flattest cells in full area",
                query_b_top20(table),
                num_runs, storage,
            )
            if r: all_results.append(r)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docs", "benchmark_snowflake_results.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to: {output_path}")
    print(f"  Total benchmarks: {len(all_results)}")
    print(f"{'='*60}")

    # Summary table
    print(f"\n  {'Query':<20} {'Storage':<12} {'Median(s)':<12} {'Result Summary'}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*30}")
    for r in all_results:
        result_summary = ""
        res = r.get("result", {})
        if isinstance(res, dict):
            if "TOTAL_PIXELS" in res or "total_pixels" in res:
                cnt = res.get("TOTAL_PIXELS", res.get("total_pixels", "?"))
                mean = res.get("MEAN_SLOPE", res.get("mean_slope", "?"))
                result_summary = f"pixels={cnt}, mean={mean}"
            elif "TOTAL_CELLS" in res or "total_cells" in res:
                tc = res.get("TOTAL_CELLS", res.get("total_cells", "?"))
                sc = res.get("SUITABLE_CELLS", res.get("suitable_cells", "?"))
                pct = res.get("PCT_SUITABLE", res.get("pct_suitable", "?"))
                result_summary = f"total={tc}, suitable={sc} ({pct}%)"
        elif isinstance(res, list):
            result_summary = f"{len(res)} rows"
        print(f"  {r['name']:<20} {r['storage']:<12} {r['median_seconds']:<12} {result_summary}")


if __name__ == "__main__":
    main()
