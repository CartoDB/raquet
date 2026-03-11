#!/bin/bash
set -e

# Full pipeline: download, merge, slope, convert to raquet, benchmark
# Run from the project root: bash scripts/run_all.sh

VENV=".venv/bin"
DATA_DIR="data"
TILES_DIR="$DATA_DIR/dem_tiles"
MAX_GB="${1:-3}"  # Default 3GB, pass as argument to change

echo "============================================"
echo "DEM Benchmark Pipeline — Loudoun County, VA"
echo "Target download: ${MAX_GB} GB"
echo "============================================"

# Step 1: Download
echo ""
echo ">>> Step 1: Downloading 1m DEM tiles (${MAX_GB} GB)..."
$VENV/python scripts/download_dem.py --max-gb "$MAX_GB"

# Step 2: Merge tiles
echo ""
echo ">>> Step 2: Merging tiles..."
$VENV/python scripts/prepare_dem.py --tiles-dir "$TILES_DIR" --output-dir "$DATA_DIR"

# Step 3: Convert to raquet
echo ""
echo ">>> Step 3: Converting elevation to raquet..."
$VENV/raquet-io convert raster "$DATA_DIR/elevation.tif" "$DATA_DIR/elevation.parquet" --streaming -v

echo ""
echo ">>> Step 4: Converting slope to raquet..."
$VENV/raquet-io convert raster "$DATA_DIR/slope.tif" "$DATA_DIR/slope.parquet" --streaming -v

# Step 5: Inspect
echo ""
echo ">>> Step 5: Inspecting raquet files..."
$VENV/raquet-io inspect "$DATA_DIR/elevation.parquet"
echo ""
$VENV/raquet-io inspect "$DATA_DIR/slope.parquet"

# Step 6: Benchmark
echo ""
echo ">>> Step 6: Running benchmarks..."
$VENV/python scripts/benchmark.py \
  --slope-file "$DATA_DIR/slope.parquet" \
  --elevation-file "$DATA_DIR/elevation.parquet" \
  --output "$DATA_DIR/benchmark_results.json"

echo ""
echo ">>> File sizes:"
ls -lh "$DATA_DIR"/*.parquet "$DATA_DIR"/*.tif 2>/dev/null

echo ""
echo "============================================"
echo "Done! Results in $DATA_DIR/benchmark_results.json"
echo "============================================"
