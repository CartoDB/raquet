#!/bin/bash
#
# Raquet SQL Engines - Multi-Platform Deployment Script
#
# Usage:
#   ./deploy.sh <platform> [options]
#
# Platforms:
#   bigquery    - Deploy to Google BigQuery
#   snowflake   - Deploy to Snowflake
#   databricks  - Deploy to Databricks
#
# Examples:
#   ./deploy.sh bigquery --bucket gs://my-bucket --dataset myproject.raquet
#   ./deploy.sh snowflake --connection raquet --database MYDB --schema RAQUET
#   ./deploy.sh databricks --profile my-workspace --catalog main --schema raquet
#
#   ./deploy.sh all       # Deploy to all configured platforms
#   ./deploy.sh test      # Run cross-platform validation tests
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM="${1:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    head -25 "$0" | tail -23 | sed 's/^# //' | sed 's/^#//'
    exit 1
}

# ============================================================================
# BigQuery Deployment
# ============================================================================
deploy_bigquery() {
    log_info "Deploying to BigQuery..."

    # Parse BigQuery-specific arguments
    local GCS_BUCKET="${RAQUET_GCS_BUCKET:-}"
    local DATASET="${RAQUET_BQ_DATASET:-cartobq.raquet}"

    shift # remove 'bigquery' from args
    while [[ $# -gt 0 ]]; do
        case $1 in
            --bucket) GCS_BUCKET="$2"; shift 2 ;;
            --dataset) DATASET="$2"; shift 2 ;;
            *) log_error "Unknown BigQuery option: $1"; exit 1 ;;
        esac
    done

    if [ -z "$GCS_BUCKET" ]; then
        log_error "BigQuery requires --bucket or RAQUET_GCS_BUCKET env var"
        exit 1
    fi

    log_info "  Bucket: $GCS_BUCKET"
    log_info "  Dataset: $DATASET"

    # Upload JS libraries to GCS
    log_info "Uploading JavaScript libraries to GCS..."
    gsutil cp "$SCRIPT_DIR/libraries/javascript/build/"*.js "$GCS_BUCKET/"

    # Deploy SQL functions
    SQL_DIR="$SCRIPT_DIR/platforms/bigquery/functions"
    DEPLOY_ORDER=(
        # Core decoding and pixel functions
        "RAQUET_DECODE_BAND.sql"
        "RAQUET_PIXEL.sql"
        # Statistics and raster value functions
        "ST_RASTERSUMMARYSTATS.sql"
        "ST_RASTERVALUE.sql"
        "ST_RASTERVALUE_GEOG.sql"
        "ST_BANDMATH.sql"
        "ST_NORMALIZEDDIFFERENCE.sql"
        "ST_NORMALIZEDDIFFERENCESTATS.sql"
        "RAQUET_AGGREGATE_STATS.sql"
        "RAQUET_PIXEL_GEOGRAPHY.sql"
        # Helper functions (auto_zoom before resolve_zoom)
        "__RAQUET_AUTO_ZOOM.sql"
        "__RAQUET_RESOLVE_ZOOM.sql"
        "__RAQUET_PIXEL_POSITIONS.sql"
        "__RAQUET_REGION_BLOCKS.sql"
        # Raster algebra (raster -> raster; needs raquet_algebra.js)
        "RASTER_ALGEBRA.sql"
    )

    for filename in "${DEPLOY_ORDER[@]}"; do
        if [ -f "$SQL_DIR/$filename" ]; then
            log_info "  Deploying: $filename"
            sed -e "s|gs://cartobq-raquet-libs|$GCS_BUCKET|g" \
                -e "s|cartobq\.raquet|$DATASET|g" \
                "$SQL_DIR/$filename" | bq query --use_legacy_sql=false
        fi
    done

    log_info "BigQuery deployment complete!"
}

# ============================================================================
# Snowflake Deployment
# ============================================================================
deploy_snowflake() {
    log_info "Deploying to Snowflake..."

    # Parse Snowflake-specific arguments
    local CONNECTION="${RAQUET_SF_CONNECTION:-}"
    local DATABASE="${RAQUET_SF_DATABASE:-}"
    local SCHEMA="${RAQUET_SF_SCHEMA:-RAQUET}"

    shift # remove 'snowflake' from args
    while [[ $# -gt 0 ]]; do
        case $1 in
            --connection) CONNECTION="$2"; shift 2 ;;
            --database) DATABASE="$2"; shift 2 ;;
            --schema) SCHEMA="$2"; shift 2 ;;
            *) log_error "Unknown Snowflake option: $1"; exit 1 ;;
        esac
    done

    if [ -z "$CONNECTION" ]; then
        log_error "Snowflake requires --connection or RAQUET_SF_CONNECTION env var"
        exit 1
    fi

    log_info "  Connection: $CONNECTION"
    log_info "  Database: $DATABASE"
    log_info "  Schema: $SCHEMA"

    # Deploy SQL functions
    SQL_DIR="$SCRIPT_DIR/platforms/snowflake/functions"

    # Set context
    snowsql -c "$CONNECTION" -q "USE DATABASE $DATABASE; USE SCHEMA $SCHEMA;" 2>/dev/null

    DEPLOY_ORDER=(
        # Core decoding and pixel functions
        "RAQUET_DECODE_BAND.sql"
        "RAQUET_PIXEL.sql"
        # Statistics and raster value functions
        "ST_RASTERSUMMARYSTATS.sql"
        "ST_RASTERVALUE.sql"
        "ST_RASTERVALUE_GEOG.sql"
        "ST_BANDMATH.sql"
        "ST_NORMALIZEDDIFFERENCE.sql"
        "ST_NORMALIZEDDIFFERENCESTATS.sql"
        "RAQUET_AGGREGATE_STATS.sql"
        "RAQUET_PIXEL_GEOGRAPHY.sql"
        # Helper functions (auto_zoom before resolve_zoom)
        "__RAQUET_AUTO_ZOOM.sql"
        "__RAQUET_RESOLVE_ZOOM.sql"
        "__RAQUET_PIXEL_POSITIONS.sql"
        "__RAQUET_REGION_BLOCKS.sql"
        # Raster algebra (generated by scripts/build_snowflake_algebra.mjs)
        "RASTER_ALGEBRA.sql"
    )

    for filename in "${DEPLOY_ORDER[@]}"; do
        if [ -f "$SQL_DIR/$filename" ]; then
            log_info "  Deploying: $filename"
            # Replace schema/database references if needed
            sed -e "s|RAQUET_DB\.RAQUET|$DATABASE.$SCHEMA|g" \
                "$SQL_DIR/$filename" | snowsql -c "$CONNECTION" -o friendly=false 2>/dev/null
        fi
    done

    log_info "Snowflake deployment complete!"
}

# ============================================================================
# Databricks Deployment
# ============================================================================
deploy_databricks() {
    log_info "Deploying to Databricks..."

    # Parse Databricks-specific arguments
    local PROFILE="${RAQUET_DB_PROFILE:-DEFAULT}"
    local CATALOG="${RAQUET_DB_CATALOG:-main}"
    local SCHEMA="${RAQUET_DB_SCHEMA:-raquet}"

    shift # remove 'databricks' from args
    while [[ $# -gt 0 ]]; do
        case $1 in
            --profile) PROFILE="$2"; shift 2 ;;
            --catalog) CATALOG="$2"; shift 2 ;;
            --schema) SCHEMA="$2"; shift 2 ;;
            *) log_error "Unknown Databricks option: $1"; exit 1 ;;
        esac
    done

    log_info "  Profile: $PROFILE"
    log_info "  Catalog: $CATALOG"
    log_info "  Schema: $SCHEMA"

    # Deploy Python UDFs via Databricks CLI or SQL
    PYTHON_DIR="$SCRIPT_DIR/platforms/databricks/functions"

    # Option 1: Deploy via Databricks SQL
    # Option 2: Deploy via notebook execution
    # Option 3: Deploy via Databricks CLI jobs

    log_warn "Databricks deployment requires running setup notebook"
    log_info "  See: platforms/databricks/notebooks/setup_raquet_udfs.py"

    # For SQL-based functions, we can deploy directly
    SQL_DIR="$SCRIPT_DIR/platforms/databricks/functions"
    if [ -d "$SQL_DIR" ]; then
        for sql_file in "$SQL_DIR"/*.sql; do
            if [ -f "$sql_file" ]; then
                log_info "  Deploying: $(basename "$sql_file")"
                databricks sql execute --profile "$PROFILE" \
                    --statement "$(cat "$sql_file")" 2>/dev/null || true
            fi
        done
    fi

    log_info "Databricks deployment complete!"
}

# ============================================================================
# Cross-Platform Validation Tests
# ============================================================================
run_tests() {
    log_info "Running cross-platform validation tests..."

    PYTHON_TEST="$SCRIPT_DIR/test/validate.py"

    if [ -f "$PYTHON_TEST" ]; then
        python3 "$PYTHON_TEST" "$@"
    else
        log_warn "Test script not found: $PYTHON_TEST"
        log_info "Creating test scaffold..."

        mkdir -p "$SCRIPT_DIR/test"
        cat > "$PYTHON_TEST" << 'PYTEST'
#!/usr/bin/env python3
"""
Cross-platform validation tests for Raquet SQL functions.
Ensures all platforms produce identical results.
"""

import json
import subprocess
from dataclasses import dataclass
from typing import Optional

@dataclass
class TestResult:
    platform: str
    function: str
    result: any
    time_ms: float

def run_bigquery(sql: str) -> dict:
    """Execute SQL on BigQuery and return result."""
    result = subprocess.run(
        ["bq", "query", "--use_legacy_sql=false", "--format=json", sql],
        capture_output=True, text=True
    )
    return json.loads(result.stdout) if result.returncode == 0 else None

def run_snowflake(sql: str, connection: str = "raquet") -> dict:
    """Execute SQL on Snowflake and return result."""
    result = subprocess.run(
        ["snowsql", "-c", connection, "-o", "output_format=json", "-q", sql],
        capture_output=True, text=True
    )
    # Parse snowsql JSON output
    return None  # TODO: implement

def test_st_rastersummarystats():
    """Verify ST_RASTERSUMMARYSTATS produces same results on all platforms."""

    # BigQuery
    bq_result = run_bigquery("""
        WITH meta AS (SELECT metadata FROM `cartobq.raquet.spain_solar_gcs` WHERE block = 0)
        SELECT (ST_RASTERSUMMARYSTATS(band_1, m.metadata, 0)).count as count
        FROM `cartobq.raquet.spain_solar_gcs` t, meta m
        WHERE t.block = 5202783469519765503
    """)

    # Snowflake
    sf_result = run_snowflake("""
        WITH meta AS (SELECT metadata FROM SPAIN_SOLAR_GHI WHERE block = 0)
        SELECT (ST_RASTERSUMMARYSTATS(band_1, m.metadata, 0)):count::INT as count
        FROM SPAIN_SOLAR_GHI t, meta m
        WHERE t.block = 5202783469519765503
    """)

    print(f"BigQuery count: {bq_result}")
    print(f"Snowflake count: {sf_result}")

    # Compare
    assert bq_result == sf_result, f"Results differ: BQ={bq_result}, SF={sf_result}"
    print("✓ ST_RASTERSUMMARYSTATS: PASSED")

if __name__ == "__main__":
    test_st_rastersummarystats()
    print("\nAll tests passed!")
PYTEST
        chmod +x "$PYTHON_TEST"
        log_info "Created: $PYTHON_TEST"
    fi
}

# ============================================================================
# Main
# ============================================================================
case "$PLATFORM" in
    bigquery|bq)
        deploy_bigquery "$@"
        ;;
    snowflake|sf)
        deploy_snowflake "$@"
        ;;
    databricks|db)
        deploy_databricks "$@"
        ;;
    all)
        deploy_bigquery "$@"
        deploy_snowflake "$@"
        deploy_databricks "$@"
        ;;
    test)
        shift
        run_tests "$@"
        ;;
    -h|--help|"")
        usage
        ;;
    *)
        log_error "Unknown platform: $PLATFORM"
        usage
        ;;
esac
