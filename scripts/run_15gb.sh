#!/bin/bash
set -e

# Full pipeline for ~15GB DEM: download, merge, slope, mask, convert, partition, upload to GCS
# Run from the project root: bash scripts/run_15gb.sh
# Expected time: ~1-2 hours total

VENV=".venv/bin"
DATA_DIR="data/15gb"
TILES_DIR="$DATA_DIR/dem_tiles"
GCS_BUCKET="gs://cartobq-raquet-libs/raquet-benchmark/15gb"

# Expanded bbox: Northern Virginia + DC metro + parts of MD
# Covers Loudoun, Fairfax, Prince William, Arlington, parts of Montgomery County MD
# Expanded bbox covering DC metro: Loudoun, Fairfax, Prince William, Arlington, parts of MD
# If TNM API is slow/down, retry or reduce bbox
BBOX="-78.0,38.5,-76.8,39.5"
MAX_GB=15

# Set SKIP_DOWNLOAD=1 to skip download and use existing tiles
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

echo "============================================"
echo "DEM Benchmark Pipeline — DC Metro Area (15GB)"
echo "Bbox: $BBOX"
echo "Target download: ${MAX_GB} GB"
echo "GCS destination: $GCS_BUCKET"
echo "============================================"

mkdir -p "$DATA_DIR"

# Step 1: Download
if [ "$SKIP_DOWNLOAD" = "1" ]; then
    echo ""
    echo ">>> Step 1: SKIPPED (SKIP_DOWNLOAD=1)"
    echo "    Using existing tiles in $TILES_DIR"
    du -sh "$TILES_DIR" 2>/dev/null || echo "    WARNING: No tiles found!"
else
    echo ""
    echo ">>> Step 1: Downloading 1m DEM tiles (~${MAX_GB} GB) from S3..."
    echo "    This will take a while..."
    $VENV/python scripts/download_dem_s3.py \
        --max-gb "$MAX_GB" \
        --output-dir "$TILES_DIR"

    echo ""
    echo ">>> Download complete. Disk usage:"
    du -sh "$TILES_DIR"
fi

# Step 2: Merge tiles
echo ""
echo ">>> Step 2: Merging tiles into single elevation raster..."
$VENV/python scripts/prepare_dem.py \
    --tiles-dir "$TILES_DIR" \
    --output-dir "$DATA_DIR"

# Step 3: Convert to raquet (native resolution only, no overviews)
echo ""
echo ">>> Step 3: Converting slope to raquet (native resolution only)..."
$VENV/raquet-io convert raster \
    "$DATA_DIR/slope.tif" \
    "$DATA_DIR/slope.parquet" \
    --overviews none \
    --streaming \
    -v

echo ""
echo ">>> Step 3b: Converting elevation to raquet (native resolution only)..."
$VENV/raquet-io convert raster \
    "$DATA_DIR/elevation.tif" \
    "$DATA_DIR/elevation.parquet" \
    --overviews none \
    --streaming \
    -v

# Step 4: Inspect
echo ""
echo ">>> Step 4: Inspecting raquet files..."
$VENV/raquet-io inspect "$DATA_DIR/slope.parquet"
echo ""
$VENV/raquet-io inspect "$DATA_DIR/elevation.parquet"

# Step 5: Partition for cloud
echo ""
echo ">>> Step 5: Partitioning slope for cloud storage..."
$VENV/raquet-io partition \
    "$DATA_DIR/slope.parquet" \
    "$DATA_DIR/slope_partitioned/" \
    -v

echo ""
echo ">>> Step 5b: Partitioning elevation for cloud storage..."
$VENV/raquet-io partition \
    "$DATA_DIR/elevation.parquet" \
    "$DATA_DIR/elevation_partitioned/" \
    -v

# Step 6: Upload to GCS
echo ""
echo ">>> Step 6: Uploading to GCS..."

# Upload single files
echo "  Uploading single slope file..."
gsutil -m cp "$DATA_DIR/slope.parquet" "$GCS_BUCKET/slope.parquet"

echo "  Uploading single elevation file..."
gsutil -m cp "$DATA_DIR/elevation.parquet" "$GCS_BUCKET/elevation.parquet"

# Upload partitioned files
echo "  Uploading partitioned slope files..."
gsutil -m cp "$DATA_DIR/slope_partitioned/"*.parquet "$GCS_BUCKET/slope_partitioned/"

echo "  Uploading partitioned elevation files..."
gsutil -m cp "$DATA_DIR/elevation_partitioned/"*.parquet "$GCS_BUCKET/elevation_partitioned/"

# Summary
echo ""
echo "============================================"
echo "DONE! Summary:"
echo "============================================"
echo ""
echo "Local files:"
ls -lh "$DATA_DIR"/*.parquet 2>/dev/null
echo ""
echo "Partitioned slope:"
ls -lh "$DATA_DIR/slope_partitioned/"*.parquet 2>/dev/null | wc -l
echo " files"
du -sh "$DATA_DIR/slope_partitioned/"
echo ""
echo "GCS files:"
gsutil ls -l "$GCS_BUCKET/"
echo ""
echo "GCS partitioned slope:"
gsutil ls "$GCS_BUCKET/slope_partitioned/" | wc -l
echo " files"
echo ""
echo "Ready for benchmarking!"
echo "  Single file:      $GCS_BUCKET/slope.parquet"
echo "  Partitioned:      $GCS_BUCKET/slope_partitioned/"
echo "============================================"
