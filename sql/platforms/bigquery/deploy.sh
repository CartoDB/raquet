#!/bin/bash
#
# Deploy BigQuery Raquet UDFs
#
# Environment variables:
#   RAQUET_GCS_BUCKET  - GCS bucket for JS libraries (required)
#                        Example: gs://my-bucket-raquet-libs
#   RAQUET_DATASET     - BigQuery dataset (optional, default: cartobq.raquet)
#                        Example: my-project.raquet
#
# Usage:
#   export RAQUET_GCS_BUCKET=gs://my-bucket-raquet-libs
#   export RAQUET_DATASET=my-project.raquet
#   ./deploy.sh
#

set -e

# Default values
DEFAULT_GCS_BUCKET="gs://cartobq-raquet-libs"
DEFAULT_DATASET="cartobq.raquet"

# Check required environment variable
if [ -z "$RAQUET_GCS_BUCKET" ]; then
    echo "Error: RAQUET_GCS_BUCKET environment variable is required"
    echo ""
    echo "Usage:"
    echo "  export RAQUET_GCS_BUCKET=gs://your-bucket-raquet-libs"
    echo "  export RAQUET_DATASET=your-project.raquet  # optional"
    echo "  ./deploy.sh"
    exit 1
fi

# Use environment variable or default for dataset
DATASET="${RAQUET_DATASET:-$DEFAULT_DATASET}"

echo "Deploying BigQuery Raquet UDFs"
echo "  GCS Bucket: $RAQUET_GCS_BUCKET"
echo "  Dataset:    $DATASET"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_DIR="$SCRIPT_DIR/sql/raquet"

# Create temp directory for processed SQL files
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Define deployment order (dependencies first)
# __RAQUET_AUTO_ZOOM must be before __RAQUET_RESOLVE_ZOOM
# ST_RASTERVALUE must be before ST_RASTERVALUE_GEOG
DEPLOY_ORDER=(
    "RAQUET_DECODE_BAND.sql"
    "RAQUET_PIXEL.sql"
    "RAQUET_PIXEL_GEOGRAPHY.sql"
    "RAQUET_AGGREGATE_STATS.sql"
    "ST_RASTERSUMMARYSTATS.sql"
    "ST_RASTERVALUE.sql"
    "ST_RASTERVALUE_GEOG.sql"
    "ST_BANDMATH.sql"
    "ST_NORMALIZEDDIFFERENCE.sql"
    "ST_NORMALIZEDDIFFERENCESTATS.sql"
    "__RAQUET_AUTO_ZOOM.sql"
    "__RAQUET_RESOLVE_ZOOM.sql"
    "__RAQUET_PIXEL_POSITIONS.sql"
    "__RAQUET_REGION_BLOCKS.sql"
)

# Process and deploy each SQL file in order
for filename in "${DEPLOY_ORDER[@]}"; do
    sql_file="$SQL_DIR/$filename"
    temp_file="$TEMP_DIR/$filename"

    if [ ! -f "$sql_file" ]; then
        echo "Warning: $filename not found, skipping"
        continue
    fi

    # Replace bucket and dataset placeholders
    sed -e "s|gs://cartobq-raquet-libs|$RAQUET_GCS_BUCKET|g" \
        -e "s|cartobq\.raquet|$DATASET|g" \
        "$sql_file" > "$temp_file"

    echo "Deploying: $filename"
    bq query --use_legacy_sql=false < "$temp_file"
done

echo ""
echo "Deployment complete!"
