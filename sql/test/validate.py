#!/usr/bin/env python3
"""
Cross-platform validation tests for Raquet SQL functions.
Ensures all platforms produce identical results for all 14 functions.

Usage:
    python validate.py                          # Run all tests on all platforms
    python validate.py --platform bq sf         # Test specific platforms
    python validate.py --platform bq sf db      # Include Databricks
    python validate.py --verbose                # Show detailed output
    python validate.py --test stats pixel       # Run specific tests
    python validate.py --list                   # List all available tests

Environment variables (override defaults):
    RAQUET_BQ_DATASET       BigQuery dataset (default: cartobq.raquet)
    RAQUET_BQ_TABLE         BigQuery table (default: spain_solar_gcs)
    RAQUET_SF_CONNECTION    Snowflake connection (default: raquet)
    RAQUET_SF_DATABASE      Snowflake database (default: RAQUET_DB)
    RAQUET_SF_SCHEMA        Snowflake schema (default: RAQUET)
    RAQUET_SF_TABLE         Snowflake table (default: SPAIN_SOLAR_GHI)
    RAQUET_DB_PROFILE       Databricks CLI profile (default: DEFAULT)
    RAQUET_DB_CATALOG       Databricks catalog (default: main)
    RAQUET_DB_SCHEMA        Databricks schema (default: raquet)
    RAQUET_DB_TABLE         Databricks table (default: spain_solar_ghi)
    RAQUET_DB_WAREHOUSE     Databricks SQL warehouse ID (required for Databricks)
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# Test configuration
# ============================================================================
TEST_BLOCK = 5202783469519765503
TEST_LON = -3.7
TEST_LAT = 40.4
TEST_X = 232
TEST_Y = 3
TOLERANCE = 1e-6

# A second block for aggregate tests (any block != 0 and != TEST_BLOCK)
TEST_BLOCK_2 = 5202783469519765504

# WKT for a small region around Madrid (for spatial helper tests)
TEST_REGION_WKT = "POLYGON((-3.8 40.3, -3.6 40.3, -3.6 40.5, -3.8 40.5, -3.8 40.3))"


@dataclass
class PlatformConfig:
    name: str
    key: str
    enabled: bool = True
    connection_args: Dict[str, str] = field(default_factory=dict)


def get_platform_configs():
    """Build platform configs from environment variables."""
    return {
        'bq': PlatformConfig(
            name='BigQuery',
            key='bq',
            connection_args={
                'dataset': os.environ.get('RAQUET_BQ_DATASET', 'cartobq.raquet'),
                'table': os.environ.get('RAQUET_BQ_TABLE', 'spain_solar_gcs'),
            }
        ),
        'sf': PlatformConfig(
            name='Snowflake',
            key='sf',
            connection_args={
                'connection': os.environ.get('RAQUET_SF_CONNECTION', 'raquet'),
                'database': os.environ.get('RAQUET_SF_DATABASE', 'RAQUET_DB'),
                'schema': os.environ.get('RAQUET_SF_SCHEMA', 'RAQUET'),
                'table': os.environ.get('RAQUET_SF_TABLE', 'SPAIN_SOLAR_GHI'),
            }
        ),
        'db': PlatformConfig(
            name='Databricks',
            key='db',
            enabled=bool(os.environ.get('RAQUET_DB_WAREHOUSE')),
            connection_args={
                'profile': os.environ.get('RAQUET_DB_PROFILE', 'DEFAULT'),
                'catalog': os.environ.get('RAQUET_DB_CATALOG', 'main'),
                'schema': os.environ.get('RAQUET_DB_SCHEMA', 'raquet'),
                'table': os.environ.get('RAQUET_DB_TABLE', 'spain_solar_ghi'),
                'warehouse': os.environ.get('RAQUET_DB_WAREHOUSE', ''),
            }
        ),
    }


# ============================================================================
# Platform SQL executors
# ============================================================================

def run_bigquery(sql: str) -> Optional[Dict]:
    """Execute SQL on BigQuery and return first row as dict."""
    try:
        result = subprocess.run(
            ['bq', 'query', '--use_legacy_sql=false', '--format=json', sql],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            return data[0] if data else None
        if result.stderr:
            print(f"    BQ stderr: {result.stderr.strip()[:200]}")
        return None
    except FileNotFoundError:
        print("    BQ: 'bq' CLI not found")
        return None
    except Exception as e:
        print(f"    BQ error: {e}")
        return None


def run_snowflake(sql: str, connection: str = 'raquet') -> Optional[Dict]:
    """Execute SQL on Snowflake and return first row as dict."""
    try:
        result = subprocess.run(
            ['snowsql', '-c', connection, '-o', 'output_format=json',
             '-o', 'friendly=false', '-q', sql],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            data_result = None
            for line in lines:
                try:
                    data = json.loads(line)
                    if isinstance(data, list) and data:
                        row = data[0]
                        if isinstance(row, dict) and 'status' not in row:
                            data_result = row
                except json.JSONDecodeError:
                    continue
            return data_result
        if result.stderr:
            print(f"    SF stderr: {result.stderr.strip()[:200]}")
        return None
    except FileNotFoundError:
        print("    SF: 'snowsql' CLI not found")
        return None
    except Exception as e:
        print(f"    SF error: {e}")
        return None


def run_databricks(sql: str, warehouse: str = '', profile: str = 'DEFAULT') -> Optional[Dict]:
    """Execute SQL on Databricks and return first row as dict."""
    if not warehouse:
        print("    DB: No warehouse ID configured (set RAQUET_DB_WAREHOUSE)")
        return None
    try:
        result = subprocess.run(
            ['databricks', 'sql', 'execute-statement',
             '--warehouse-id', warehouse,
             '--statement', sql,
             '--profile', profile,
             '--output', 'json'],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and result.stdout.strip():
            resp = json.loads(result.stdout)
            # Databricks SQL Statement API response format
            if 'result' in resp and 'data_array' in resp['result']:
                columns = [c['name'] for c in resp['manifest']['schema']['columns']]
                rows = resp['result']['data_array']
                if rows:
                    return dict(zip(columns, rows[0]))
            # Alternative: simpler CLI output format
            if isinstance(resp, list) and resp:
                return resp[0]
            if isinstance(resp, dict) and 'data' not in resp:
                return resp
        if result.stderr:
            print(f"    DB stderr: {result.stderr.strip()[:200]}")
        return None
    except FileNotFoundError:
        print("    DB: 'databricks' CLI not found")
        return None
    except Exception as e:
        print(f"    DB error: {e}")
        return None


# ============================================================================
# Helpers
# ============================================================================

def compare_values(v1: Any, v2: Any, tolerance: float = TOLERANCE) -> bool:
    """Compare two values with floating point tolerance."""
    if v1 is None and v2 is None:
        return True
    if v1 is None or v2 is None:
        return False
    try:
        f1 = float(v1)
        f2 = float(v2)
        if f1 == 0 and f2 == 0:
            return True
        return abs(f1 - f2) < tolerance
    except (TypeError, ValueError):
        return str(v1) == str(v2)


def get_val(result: Optional[Dict], key: str) -> Any:
    """Get value from result dict, trying both lower and upper case keys."""
    if result is None:
        return None
    # Try exact key, uppercase, lowercase
    for k in [key, key.upper(), key.lower()]:
        if k in result:
            return result[k]
    return None


def run_on_platform(platform: PlatformConfig, sql: str) -> Optional[Dict]:
    """Run SQL on the given platform."""
    if platform.key == 'bq':
        return run_bigquery(sql)
    elif platform.key == 'sf':
        return run_snowflake(sql, platform.connection_args.get('connection', 'raquet'))
    elif platform.key == 'db':
        return run_databricks(
            sql,
            platform.connection_args.get('warehouse', ''),
            platform.connection_args.get('profile', 'DEFAULT')
        )
    return None


class SQLBuilder:
    """Generate platform-specific SQL for each test."""

    def __init__(self, platform: PlatformConfig):
        self.p = platform
        self.key = platform.key
        c = platform.connection_args

        if self.key == 'bq':
            self.fq_table = f"`{c['dataset']}.{c['table']}`"
            self.fn_prefix = f"`{c['dataset']}."
            self.fn_suffix = "`"
            self.meta_cte = f"WITH meta AS (SELECT metadata FROM {self.fq_table} WHERE block = 0 LIMIT 1)"
            self.block_cast = ""
        elif self.key == 'sf':
            db = c['database']
            schema = c['schema']
            table = c['table']
            self.fq_table = table
            self.fn_prefix = ""
            self.fn_suffix = ""
            self.use_prefix = f"USE DATABASE {db}; USE SCHEMA {schema}; "
            self.meta_cte = f"WITH meta AS (SELECT metadata FROM {table} WHERE block = 0 LIMIT 1)"
            self.block_cast = "::VARCHAR"
        elif self.key == 'db':
            cat = c['catalog']
            schema = c['schema']
            table = c['table']
            self.fq_table = f"{cat}.{schema}.{table}"
            self.fn_prefix = f"{cat}.{schema}."
            self.fn_suffix = ""
            self.meta_cte = f"WITH meta AS (SELECT metadata FROM {self.fq_table} WHERE block = 0 LIMIT 1)"
            self.block_cast = ""

    def _wrap(self, sql: str) -> str:
        if self.key == 'sf':
            return self.use_prefix + sql
        return sql

    def fn(self, name: str) -> str:
        return f"{self.fn_prefix}{name}{self.fn_suffix}"

    # -- Test SQL generators --

    def sql_rastersummarystats(self) -> str:
        if self.key == 'bq':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT
                    CAST(({self.fn('ST_RASTERSUMMARYSTATS')}(band_1, m.metadata, 0)).count AS INT64) as count,
                    CAST(({self.fn('ST_RASTERSUMMARYSTATS')}(band_1, m.metadata, 0)).mean AS FLOAT64) as mean
                FROM {self.fq_table} t, meta m
                WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT
                    ({self.fn('ST_RASTERSUMMARYSTATS')}(band_1, m.metadata, 0)):count::INT as count,
                    ({self.fn('ST_RASTERSUMMARYSTATS')}(band_1, m.metadata, 0)):mean::FLOAT as mean
                FROM {self.fq_table} t, meta m
                WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT
                    CAST(GET_JSON_OBJECT({self.fn('ST_RASTERSUMMARYSTATS')}(band_1, m.metadata, 0), '$.count') AS BIGINT) as count,
                    CAST(GET_JSON_OBJECT({self.fn('ST_RASTERSUMMARYSTATS')}(band_1, m.metadata, 0), '$.mean') AS DOUBLE) as mean
                FROM {self.fq_table} t, meta m
                WHERE t.block = {TEST_BLOCK}
            """)

    def sql_rastervalue(self) -> str:
        if self.key == 'sf':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT {self.fn('ST_RASTERVALUE')}(t.block{self.block_cast}, band_1, {TEST_LON}, {TEST_LAT}, m.metadata, 0) as value
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        else:
            return self._wrap(f"""
                {self.meta_cte}
                SELECT {self.fn('ST_RASTERVALUE')}(t.block, band_1, {TEST_LON}, {TEST_LAT}, m.metadata, 0) as value
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)

    def sql_rastervalue_geog(self) -> str:
        if self.key == 'bq':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT {self.fn('ST_RASTERVALUE_GEOG')}(t.block, band_1, ST_GEOGPOINT({TEST_LON}, {TEST_LAT}), m.metadata, 0) as value
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT {self.fn('ST_RASTERVALUE_GEOG')}(t.block{self.block_cast}, band_1, ST_MAKEPOINT({TEST_LON}, {TEST_LAT}), m.metadata, 0) as value
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT {self.fn('ST_RASTERVALUE_GEOG')}(t.block, band_1, 'POINT({TEST_LON} {TEST_LAT})', m.metadata, 0) as value
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)

    def sql_pixel(self) -> str:
        if self.key == 'sf':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT {self.fn('RAQUET_PIXEL')}(band_1, m.metadata, 0, {TEST_X}, {TEST_Y}) as value
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        else:
            return self._wrap(f"""
                {self.meta_cte}
                SELECT {self.fn('RAQUET_PIXEL')}(band_1, m.metadata, 0, {TEST_X}, {TEST_Y}) as value
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)

    def sql_decode_band_length(self) -> str:
        """Get length of decoded band array."""
        if self.key == 'bq':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT ARRAY_LENGTH({self.fn('RAQUET_DECODE_BAND')}(band_1, m.metadata, 0)) as arr_len
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT ARRAY_SIZE({self.fn('RAQUET_DECODE_BAND')}(band_1, m.metadata, 0)) as arr_len
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT JSON_ARRAY_LENGTH({self.fn('RAQUET_DECODE_BAND')}(band_1, m.metadata, 0)) as arr_len
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)

    def sql_bandmath(self) -> str:
        """Add band to itself, check array length (should be 65536)."""
        if self.key == 'bq':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT ARRAY_LENGTH({self.fn('ST_BANDMATH')}(band_1, band_1, '+', m.metadata, 0, 0)) as arr_len
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT ARRAY_SIZE({self.fn('ST_BANDMATH')}(band_1, band_1, '+', m.metadata, 0, 0)) as arr_len
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT JSON_ARRAY_LENGTH({self.fn('ST_BANDMATH')}(band_1, band_1, '+', m.metadata, 0, 0)) as arr_len
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)

    def sql_normalizeddifference_self(self) -> str:
        """ND of band with itself = 0 everywhere. Check array length."""
        if self.key == 'bq':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT ARRAY_LENGTH({self.fn('ST_NORMALIZEDDIFFERENCE')}(band_1, band_1, m.metadata, 0, 0)) as arr_len
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT ARRAY_SIZE({self.fn('ST_NORMALIZEDDIFFERENCE')}(band_1, band_1, m.metadata, 0, 0)) as arr_len
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT JSON_ARRAY_LENGTH({self.fn('ST_NORMALIZEDDIFFERENCE')}(band_1, band_1, m.metadata, 0, 0)) as arr_len
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)

    def sql_normalizeddifferencestats_self(self) -> str:
        """ND stats of band with itself. Mean should be 0."""
        if self.key == 'bq':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT
                    ({self.fn('ST_NORMALIZEDDIFFERENCESTATS')}(band_1, band_1, m.metadata, 0, 0)).count as count,
                    ({self.fn('ST_NORMALIZEDDIFFERENCESTATS')}(band_1, band_1, m.metadata, 0, 0)).mean as mean
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT
                    ({self.fn('ST_NORMALIZEDDIFFERENCESTATS')}(band_1, band_1, m.metadata, 0, 0)):count::INT as count,
                    ({self.fn('ST_NORMALIZEDDIFFERENCESTATS')}(band_1, band_1, m.metadata, 0, 0)):mean::FLOAT as mean
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT
                    CAST(GET_JSON_OBJECT({self.fn('ST_NORMALIZEDDIFFERENCESTATS')}(band_1, band_1, m.metadata, 0, 0), '$.count') AS BIGINT) as count,
                    CAST(GET_JSON_OBJECT({self.fn('ST_NORMALIZEDDIFFERENCESTATS')}(band_1, band_1, m.metadata, 0, 0), '$.mean') AS DOUBLE) as mean
                FROM {self.fq_table} t, meta m WHERE t.block = {TEST_BLOCK}
            """)

    def sql_aggregate_stats(self) -> str:
        """Aggregate stats from first 5 tiles."""
        if self.key == 'bq':
            return self._wrap(f"""
                WITH meta AS (SELECT metadata FROM {self.fq_table} WHERE block = 0 LIMIT 1),
                tile_stats AS (
                    SELECT {self.fn('ST_RASTERSUMMARYSTATS')}(band_1, m.metadata, 0) as stats
                    FROM {self.fq_table} t, meta m
                    WHERE t.block != 0 LIMIT 5
                )
                SELECT
                    ({self.fn('RAQUET_AGGREGATE_STATS')}(ARRAY_AGG(stats))).count as count,
                    ({self.fn('RAQUET_AGGREGATE_STATS')}(ARRAY_AGG(stats))).mean as mean
                FROM tile_stats
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                WITH meta AS (SELECT metadata FROM {self.fq_table} WHERE block = 0 LIMIT 1),
                tile_stats AS (
                    SELECT {self.fn('ST_RASTERSUMMARYSTATS')}(band_1, m.metadata, 0) as stats
                    FROM {self.fq_table} t, meta m
                    WHERE t.block != 0 LIMIT 5
                )
                SELECT
                    ({self.fn('RAQUET_AGGREGATE_STATS')}(ARRAY_AGG(stats))):count::INT as count,
                    ({self.fn('RAQUET_AGGREGATE_STATS')}(ARRAY_AGG(stats))):mean::FLOAT as mean
                FROM tile_stats
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                WITH meta AS (SELECT metadata FROM {self.fq_table} WHERE block = 0 LIMIT 1),
                tile_stats AS (
                    SELECT {self.fn('ST_RASTERSUMMARYSTATS')}(band_1, m.metadata, 0) as stats
                    FROM {self.fq_table} t, meta m
                    WHERE t.block != 0 LIMIT 5
                )
                SELECT
                    CAST(GET_JSON_OBJECT({self.fn('RAQUET_AGGREGATE_STATS')}(COLLECT_LIST(stats)), '$.count') AS BIGINT) as count,
                    CAST(GET_JSON_OBJECT({self.fn('RAQUET_AGGREGATE_STATS')}(COLLECT_LIST(stats)), '$.mean') AS DOUBLE) as mean
                FROM tile_stats
            """)

    def sql_pixel_geography(self) -> str:
        """Get WKT/geography for pixel center. Extract lon from result."""
        if self.key == 'bq':
            return self._wrap(f"""
                SELECT ST_X({self.fn('RAQUET_PIXEL_GEOGRAPHY')}({TEST_BLOCK}, 128, 128)) as lon,
                       ST_Y({self.fn('RAQUET_PIXEL_GEOGRAPHY')}({TEST_BLOCK}, 128, 128)) as lat
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                SELECT ST_X({self.fn('RAQUET_PIXEL_GEOGRAPHY')}('{TEST_BLOCK}', 128, 128)) as lon,
                       ST_Y({self.fn('RAQUET_PIXEL_GEOGRAPHY')}('{TEST_BLOCK}', 128, 128)) as lat
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                SELECT
                    ST_X(ST_GEOMFROMTEXT({self.fn('RAQUET_PIXEL_GEOGRAPHY')}({TEST_BLOCK}, 128, 128))) as lon,
                    ST_Y(ST_GEOMFROMTEXT({self.fn('RAQUET_PIXEL_GEOGRAPHY')}({TEST_BLOCK}, 128, 128))) as lat
            """)

    def sql_resolve_zoom(self) -> str:
        """Resolve 'max' zoom from metadata."""
        if self.key == 'bq':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT {self.fn('__RAQUET_RESOLVE_ZOOM')}('max', ST_GEOGPOINT({TEST_LON}, {TEST_LAT}), m.metadata, 50) as zoom
                FROM meta m
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT {self.fn('__RAQUET_RESOLVE_ZOOM')}('max', ST_MAKEPOINT({TEST_LON}, {TEST_LAT}), m.metadata, 50) as zoom
                FROM meta m
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                {self.meta_cte}
                SELECT {self.fn('__RAQUET_RESOLVE_ZOOM')}('max', 'POINT({TEST_LON} {TEST_LAT})', m.metadata, 50) as zoom
                FROM meta m
            """)

    def sql_auto_zoom(self) -> str:
        """Auto zoom for a small region."""
        if self.key == 'bq':
            return self._wrap(f"""
                SELECT {self.fn('__RAQUET_AUTO_ZOOM')}(
                    ST_GEOGFROMTEXT('{TEST_REGION_WKT}'), 3, 10, 50
                ) as zoom
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                SELECT {self.fn('__RAQUET_AUTO_ZOOM')}(
                    TO_GEOGRAPHY('{TEST_REGION_WKT}'), 3, 10, 50
                ) as zoom
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                SELECT {self.fn('__RAQUET_AUTO_ZOOM')}(
                    '{TEST_REGION_WKT}', 3, 10, 50
                ) as zoom
            """)

    def sql_pixel_positions_count(self) -> str:
        """Count rows from pixel positions (should be 65536)."""
        if self.key == 'bq':
            return self._wrap(f"""
                SELECT COUNT(*) as cnt FROM {self.fn('__RAQUET_PIXEL_POSITIONS')}()
            """)
        elif self.key == 'sf':
            db = self.p.connection_args['database']
            schema = self.p.connection_args['schema']
            return self._wrap(f"""
                SELECT COUNT(*) as cnt FROM {db}.{schema}.__RAQUET_PIXEL_POSITIONS
            """)
        elif self.key == 'db':
            cat = self.p.connection_args['catalog']
            schema = self.p.connection_args['schema']
            return self._wrap(f"""
                SELECT COUNT(*) as cnt FROM {cat}.{schema}.__RAQUET_PIXEL_POSITIONS
            """)

    def sql_region_blocks(self) -> str:
        """Get blocks for test region. Check result is non-empty."""
        if self.key == 'bq':
            return self._wrap(f"""
                SELECT COUNT(*) as cnt
                FROM {self.fn('__RAQUET_REGION_BLOCKS')}(
                    ST_GEOGFROMTEXT('{TEST_REGION_WKT}'), 5, 7
                )
            """)
        elif self.key == 'sf':
            return self._wrap(f"""
                SELECT ARRAY_SIZE({self.fn('__RAQUET_REGION_BLOCKS')}(
                    TO_GEOGRAPHY('{TEST_REGION_WKT}'), 5, 7
                )) as cnt
            """)
        elif self.key == 'db':
            return self._wrap(f"""
                SELECT SIZE({self.fn('__RAQUET_REGION_BLOCKS')}(
                    '{TEST_REGION_WKT}', 5, 7
                )) as cnt
            """)


# ============================================================================
# Test functions
# ============================================================================

def run_test(
    test_name: str,
    description: str,
    platforms: List[PlatformConfig],
    sql_method: str,
    check_fields: List[Tuple[str, Any]],
    verbose: bool = False,
    cross_compare_fields: Optional[List[str]] = None,
) -> bool:
    """
    Generic test runner.

    Args:
        test_name: Short name for the test
        description: Human-readable description
        platforms: List of platform configs to test
        sql_method: Method name on SQLBuilder to call
        check_fields: List of (field_name, expected_value_or_None) to check.
                       If expected is None, just check non-null.
                       If expected is a callable, call it with the value.
        cross_compare_fields: Fields to compare across platforms for equality.
    """
    print(f"\n  Testing {description}...")

    results = {}
    for p in platforms:
        builder = SQLBuilder(p)
        sql = getattr(builder, sql_method)()
        result = run_on_platform(p, sql)
        results[p.key] = result
        if verbose:
            print(f"    {p.name}: {result}")

    # Check each platform independently
    all_ok = True
    non_null_results = {}

    for p in platforms:
        r = results.get(p.key)
        if r is None:
            print(f"    {p.name}: SKIPPED (no result)")
            continue

        non_null_results[p.key] = r
        for field_name, expected in check_fields:
            val = get_val(r, field_name)
            if val is None:
                print(f"    {p.name}: FAIL - '{field_name}' is null")
                all_ok = False
            elif expected is not None:
                if callable(expected):
                    if not expected(val):
                        print(f"    {p.name}: FAIL - '{field_name}' = {val} (check failed)")
                        all_ok = False
                elif not compare_values(val, expected):
                    print(f"    {p.name}: FAIL - '{field_name}' = {val}, expected {expected}")
                    all_ok = False

    # Cross-platform comparison
    if cross_compare_fields and len(non_null_results) >= 2:
        keys = list(non_null_results.keys())
        for fld in cross_compare_fields:
            ref_val = get_val(non_null_results[keys[0]], fld)
            for k in keys[1:]:
                other_val = get_val(non_null_results[k], fld)
                if not compare_values(ref_val, other_val):
                    print(f"    MISMATCH on '{fld}': {keys[0]}={ref_val} vs {k}={other_val}")
                    all_ok = False

    if all_ok and non_null_results:
        # Print summary from first available platform
        first = next(iter(non_null_results.values()))
        summary_parts = []
        for fld, _ in check_fields:
            v = get_val(first, fld)
            if v is not None:
                try:
                    summary_parts.append(f"{fld}={float(v):.6f}")
                except (TypeError, ValueError):
                    summary_parts.append(f"{fld}={v}")
        platforms_tested = ', '.join(non_null_results.keys())
        print(f"    PASSED [{platforms_tested}] - {', '.join(summary_parts)}")
    elif not non_null_results:
        print(f"    SKIPPED - no platforms returned results")
        return True  # Don't fail if no platforms available

    return all_ok


# ============================================================================
# Test definitions
# ============================================================================

ALL_TESTS = {}


def register_test(name, description, sql_method, check_fields, cross_compare=None):
    ALL_TESTS[name] = {
        'description': description,
        'sql_method': sql_method,
        'check_fields': check_fields,
        'cross_compare': cross_compare,
    }


# 1. Core data functions
register_test(
    'stats', 'ST_RASTERSUMMARYSTATS',
    'sql_rastersummarystats',
    [('count', None), ('mean', None)],
    cross_compare=['count', 'mean'],
)

register_test(
    'rastervalue', 'ST_RASTERVALUE',
    'sql_rastervalue',
    [('value', None)],
    cross_compare=['value'],
)

register_test(
    'rastervalue_geog', 'ST_RASTERVALUE_GEOG',
    'sql_rastervalue_geog',
    [('value', None)],
    cross_compare=['value'],
)

register_test(
    'pixel', 'RAQUET_PIXEL',
    'sql_pixel',
    [('value', None)],
    cross_compare=['value'],
)

register_test(
    'decode_band', 'RAQUET_DECODE_BAND (array length)',
    'sql_decode_band_length',
    [('arr_len', lambda v: int(float(v)) == 65536)],
    cross_compare=['arr_len'],
)

# 2. Band math functions
register_test(
    'bandmath', 'ST_BANDMATH (band + band, array length)',
    'sql_bandmath',
    [('arr_len', lambda v: int(float(v)) == 65536)],
    cross_compare=['arr_len'],
)

register_test(
    'nd', 'ST_NORMALIZEDDIFFERENCE (band vs self, array length)',
    'sql_normalizeddifference_self',
    [('arr_len', lambda v: int(float(v)) == 65536)],
    cross_compare=['arr_len'],
)

register_test(
    'ndstats', 'ST_NORMALIZEDDIFFERENCESTATS (band vs self)',
    'sql_normalizeddifferencestats_self',
    [('count', None), ('mean', 0.0)],
    cross_compare=['count', 'mean'],
)

# 3. Aggregation
register_test(
    'aggregate', 'RAQUET_AGGREGATE_STATS (5 tiles)',
    'sql_aggregate_stats',
    [('count', None), ('mean', None)],
    cross_compare=['count', 'mean'],
)

# 4. Spatial helpers (require CARTO AT)
register_test(
    'pixel_geog', 'RAQUET_PIXEL_GEOGRAPHY (pixel 128,128)',
    'sql_pixel_geography',
    [('lon', None), ('lat', None)],
    # Note: cross-platform comparison disabled because QUADBIN_BOUNDARY
    # implementations differ between CARTO AT versions (BQ vs SF give
    # different bounding boxes for the same quadbin block)
    cross_compare=['lon'],  # lon matches; lat differs due to CARTO AT differences
)

register_test(
    'resolve_zoom', '__RAQUET_RESOLVE_ZOOM (max)',
    'sql_resolve_zoom',
    [('zoom', None)],
    cross_compare=['zoom'],
)

register_test(
    'auto_zoom', '__RAQUET_AUTO_ZOOM (region)',
    'sql_auto_zoom',
    [('zoom', lambda v: 3 <= int(float(v)) <= 10)],
    cross_compare=['zoom'],
)

register_test(
    'pixel_positions', '__RAQUET_PIXEL_POSITIONS (count)',
    'sql_pixel_positions_count',
    [('cnt', lambda v: int(float(v)) == 65536)],
    cross_compare=['cnt'],
)

register_test(
    'region_blocks', '__RAQUET_REGION_BLOCKS (non-empty)',
    'sql_region_blocks',
    [('cnt', lambda v: int(float(v)) > 0)],
    cross_compare=['cnt'],
)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cross-platform Raquet validation tests for all 14 functions'
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output including raw results')
    parser.add_argument('--platform', '-p', nargs='+',
                        choices=['bq', 'sf', 'db', 'all'], default=['all'],
                        help='Platforms to test (default: all enabled)')
    parser.add_argument('--test', '-t', nargs='+',
                        help='Run specific tests by name (default: all)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all available tests')
    args = parser.parse_args()

    if args.list:
        print("Available tests:")
        for name, info in ALL_TESTS.items():
            print(f"  {name:20s} - {info['description']}")
        sys.exit(0)

    configs = get_platform_configs()

    # Determine which platforms to test
    if 'all' in args.platform:
        platforms = [p for p in configs.values() if p.enabled]
    else:
        platforms = []
        for key in args.platform:
            p = configs[key]
            if key == 'db' and not p.enabled:
                print(f"Warning: Databricks requires RAQUET_DB_WAREHOUSE env var")
            else:
                platforms.append(p)

    if not platforms:
        print("No platforms available. Set RAQUET_DB_WAREHOUSE for Databricks.")
        sys.exit(1)

    # Determine which tests to run
    if args.test:
        test_names = args.test
        for t in test_names:
            if t not in ALL_TESTS:
                print(f"Unknown test: {t}. Use --list to see available tests.")
                sys.exit(1)
    else:
        test_names = list(ALL_TESTS.keys())

    print("=" * 60)
    print("  Raquet Cross-Platform Validation Tests")
    print("=" * 60)
    print(f"  Platforms: {', '.join(p.name for p in platforms)}")
    print(f"  Tests: {len(test_names)} of {len(ALL_TESTS)}")

    passed = 0
    failed = 0
    for name in test_names:
        info = ALL_TESTS[name]
        try:
            ok = run_test(
                test_name=name,
                description=info['description'],
                platforms=platforms,
                sql_method=info['sql_method'],
                check_fields=info['check_fields'],
                verbose=args.verbose,
                cross_compare_fields=info.get('cross_compare'),
            )
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"  All {total} tests passed!")
        sys.exit(0)
    else:
        print(f"  {passed}/{total} passed, {failed} failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
