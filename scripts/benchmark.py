#!/usr/bin/env python3
"""
Benchmark raquet queries on DEM data using DuckDB.

Two query types for data center site suitability:
  Query A: Given a candidate site polygon, return slope statistics
  Query B: Given a larger area + slope threshold, find suitable grid cells

Usage:
    .venv/bin/python scripts/benchmark.py --slope-file data/slope.parquet
    .venv/bin/python scripts/benchmark.py --slope-file data/slope_single.parquet
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

# Two sets of polygons: one for Loudoun County (4GB), one for full MD/DE (15GB)
# Select with --polygons loudoun|maryland

POLYGONS_LOUDOUN = {
    "A": {
        "small_site": {
            "description": "Small candidate site (~0.5 km²)",
            "wkt": "POLYGON((-77.83 38.86, -77.82 38.86, -77.82 38.865, -77.83 38.865, -77.83 38.86))",
        },
        "medium_site": {
            "description": "Medium candidate site (~4 km²)",
            "wkt": "POLYGON((-77.85 38.85, -77.83 38.85, -77.83 38.87, -77.85 38.87, -77.85 38.85))",
        },
        "large_site": {
            "description": "Large candidate area (~50 km²)",
            "wkt": "POLYGON((-77.88 38.82, -77.78 38.82, -77.78 38.90, -77.88 38.90, -77.88 38.82))",
        },
    },
    "B": {
        "search_flat_3deg": {
            "description": "Find cells with mean slope < 3° in full area",
            "search_area": "POLYGON((-77.90 38.82, -77.19 38.82, -77.19 39.37, -77.90 39.37, -77.90 38.82))",
            "max_slope_degrees": 3.0,
        },
        "search_flat_5deg": {
            "description": "Find cells with mean slope < 5° in full area",
            "search_area": "POLYGON((-77.90 38.82, -77.19 38.82, -77.19 39.37, -77.90 39.37, -77.90 38.82))",
            "max_slope_degrees": 5.0,
        },
    },
}

# 15GB dataset: DC metro + MD/DE, -77.54 to -75.76, 38.16 to 39.72
# Dense data around -76.0, 39.2 (central MD)
POLYGONS_MARYLAND = {
    "A": {
        "small_site": {
            "description": "Small candidate site (~0.5 km²)",
            "wkt": "POLYGON((-76.00 39.20, -75.99 39.20, -75.99 39.205, -76.00 39.205, -76.00 39.20))",
        },
        "medium_site": {
            "description": "Medium candidate site (~25 km²)",
            "wkt": "POLYGON((-76.05 39.15, -75.95 39.15, -75.95 39.25, -76.05 39.25, -76.05 39.15))",
        },
        "large_site": {
            "description": "Large candidate area (~3000 km²)",
            "wkt": "POLYGON((-76.50 38.80, -76.00 38.80, -76.00 39.50, -76.50 39.50, -76.50 38.80))",
        },
    },
    "B": {
        "search_flat_3deg": {
            "description": "Find cells with mean slope < 3° in full area",
            "search_area": "POLYGON((-77.54 38.16, -75.76 38.16, -75.76 39.72, -77.54 39.72, -77.54 38.16))",
            "max_slope_degrees": 3.0,
        },
        "search_flat_5deg": {
            "description": "Find cells with mean slope < 5° in full area",
            "search_area": "POLYGON((-77.54 38.16, -75.76 38.16, -75.76 39.72, -77.54 39.72, -77.54 38.16))",
            "max_slope_degrees": 5.0,
        },
    },
}

# Default polygon sets (selected by --polygons flag)
POLYGON_SETS = {
    "loudoun": POLYGONS_LOUDOUN,
    "maryland": POLYGONS_MARYLAND,
}

# Active test cases (set in main())
TEST_CASES_A = POLYGONS_LOUDOUN["A"]
TEST_CASES_B = POLYGONS_LOUDOUN["B"]

RUNS_PER_QUERY = 3
DUCKDB_CMD = "/Users/jatorre/workspace/duckdb-raquet/build/release/duckdb"
RAQUET_EXT = "/Users/jatorre/workspace/duckdb-raquet/build/release/extension/raquet/raquet.duckdb_extension"
MAX_ZOOM = 17  # Native resolution zoom level


def run_duckdb(sql: str, timeout: int = 120) -> tuple[str, float]:
    """Run a DuckDB query and return (output, elapsed_seconds)."""
    full_sql = f"LOAD '{RAQUET_EXT}';\n{sql}"
    start = time.perf_counter()
    result = subprocess.run(
        [DUCKDB_CMD, "-unsigned", "-json"],
        input=full_sql,
        capture_output=True, text=True, timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"DuckDB error: {result.stderr}")
    return result.stdout.strip(), elapsed


def benchmark_query_a(slope_file: str) -> list[dict]:
    """Query A: Given a site polygon, return slope statistics."""
    print("\n" + "=" * 70)
    print("QUERY A: Score a candidate site (slope statistics within polygon)")
    print("=" * 70)

    results = []
    for name, case in TEST_CASES_A.items():
        wkt = case["wkt"]
        timings = []
        output = None

        for run in range(RUNS_PER_QUERY):
            sql = f"""
SELECT ST_RegionStats(band_1, block,
  '{wkt}'::GEOMETRY, metadata) AS stats
FROM read_raquet('{slope_file}', '{wkt}'::GEOMETRY, {MAX_ZOOM});
"""
            try:
                out, elapsed = run_duckdb(sql)
                timings.append(elapsed)
                if run == 0:
                    output = out
            except Exception as e:
                print(f"  ERROR on {name}: {e}")
                break

        if timings:
            median_time = sorted(timings)[len(timings) // 2]
            results.append({
                "query": "A",
                "name": name,
                "description": case["description"],
                "median_seconds": round(median_time, 3),
                "all_timings": [round(t, 3) for t in timings],
                "result": output,
            })
            print(f"\n  {name} ({case['description']})")
            print(f"  Result: {output}")
            print(f"  Timings: {[round(t,3) for t in timings]} -> median {median_time:.3f}s")

    return results


def benchmark_query_b(slope_file: str) -> list[dict]:
    """Query B: Given area + threshold, find suitable grid cells."""
    print("\n" + "=" * 70)
    print("QUERY B: Find suitable cells (slope below threshold in area)")
    print("=" * 70)

    results = []
    for name, case in TEST_CASES_B.items():
        wkt = case["search_area"]
        threshold = case["max_slope_degrees"]
        timings = []
        output = None

        for run in range(RUNS_PER_QUERY):
            sql = f"""
SELECT count(*) as total_cells,
       count(*) FILTER (WHERE (ST_RasterSummaryStats(band_1, metadata)).mean < {threshold}) as suitable_cells,
       round(count(*) FILTER (WHERE (ST_RasterSummaryStats(band_1, metadata)).mean < {threshold}) * 100.0 / count(*), 1) as pct_suitable
FROM read_raquet('{slope_file}', '{wkt}'::GEOMETRY, {MAX_ZOOM});
"""
            try:
                out, elapsed = run_duckdb(sql)
                timings.append(elapsed)
                if run == 0:
                    output = out
            except Exception as e:
                print(f"  ERROR on {name}: {e}")
                break

        if timings:
            median_time = sorted(timings)[len(timings) // 2]
            results.append({
                "query": "B",
                "name": name,
                "description": case["description"],
                "threshold_degrees": threshold,
                "median_seconds": round(median_time, 3),
                "all_timings": [round(t, 3) for t in timings],
                "result": output,
            })
            print(f"\n  {name} ({case['description']})")
            print(f"  Result: {output}")
            print(f"  Timings: {[round(t,3) for t in timings]} -> median {median_time:.3f}s")

    return results


def benchmark_query_b_detail(slope_file: str) -> list[dict]:
    """Query B detail: Return the actual suitable cells with their stats."""
    print("\n" + "=" * 70)
    print("QUERY B (detail): Return suitable cells with geometry")
    print("=" * 70)

    # Use the 3-degree threshold for the detail query
    case = TEST_CASES_B["search_flat_3deg"]
    wkt = case["search_area"]
    threshold = case["max_slope_degrees"]
    timings = []
    output = None

    for run in range(RUNS_PER_QUERY):
        sql = f"""
SELECT block,
       (ST_RasterSummaryStats(band_1, metadata)).mean as mean_slope,
       (ST_RasterSummaryStats(band_1, metadata)).max as max_slope,
       (ST_RasterSummaryStats(band_1, metadata)).count as pixel_count
FROM read_raquet('{slope_file}', '{wkt}'::GEOMETRY, {MAX_ZOOM})
WHERE (ST_RasterSummaryStats(band_1, metadata)).mean < {threshold}
  AND (ST_RasterSummaryStats(band_1, metadata)).count > 0
ORDER BY mean_slope ASC
LIMIT 20;
"""
        try:
            out, elapsed = run_duckdb(sql)
            timings.append(elapsed)
            if run == 0:
                output = out
        except Exception as e:
            print(f"  ERROR: {e}")
            break

    if timings:
        median_time = sorted(timings)[len(timings) // 2]
        print(f"\n  Top 20 flattest cells (mean slope < {threshold}°):")
        print(f"  {output}")
        print(f"  Timings: {[round(t,3) for t in timings]} -> median {median_time:.3f}s")
        return [{
            "query": "B_detail",
            "name": "top_20_flattest",
            "description": f"Top 20 flattest cells (< {threshold}°) with stats",
            "median_seconds": round(median_time, 3),
            "all_timings": [round(t, 3) for t in timings],
            "result": output,
        }]

    return []


def print_summary(all_results: list[dict]):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY: DEM Terrain Analysis for Data Center Suitability")
    print("=" * 80)
    print(f"\n{'Query':<10} {'Test':<25} {'Median (s)':<12} {'Timings'}")
    print("-" * 80)

    for r in all_results:
        query = r["query"]
        name = r["name"]
        print(f"{query:<10} {name:<25} {r['median_seconds']:<12} {r['all_timings']}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark raquet DEM queries")
    parser.add_argument("--slope-file", type=str, default="data/slope.parquet",
                        help="Slope parquet file (default: data/slope.parquet)")
    parser.add_argument("--elevation-file", type=str, default="data/elevation.parquet",
                        help="Elevation parquet file (default: data/elevation.parquet)")
    parser.add_argument("--output", type=str, default="data/benchmark_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per query (default: 3)")
    parser.add_argument("--polygons", type=str, default="loudoun",
                        choices=["loudoun", "maryland"],
                        help="Polygon set: loudoun (4GB) or maryland (15GB)")
    args = parser.parse_args()

    global RUNS_PER_QUERY, TEST_CASES_A, TEST_CASES_B
    RUNS_PER_QUERY = args.runs
    TEST_CASES_A = POLYGON_SETS[args.polygons]["A"]
    TEST_CASES_B = POLYGON_SETS[args.polygons]["B"]

    import glob as globmod
    slope_path = Path(args.slope_file)
    if not slope_path.exists() and not globmod.glob(args.slope_file):
        print(f"ERROR: {args.slope_file} not found")
        return

    all_results = []

    # Query A: Score a site
    all_results.extend(benchmark_query_a(args.slope_file))

    # Query B: Find suitable cells (summary)
    all_results.extend(benchmark_query_b(args.slope_file))

    # Query B: Find suitable cells (detail with top 20)
    all_results.extend(benchmark_query_b_detail(args.slope_file))

    # Summary
    print_summary(all_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
