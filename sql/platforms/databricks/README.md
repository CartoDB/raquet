# Databricks SQL Warehouse Platform

## Requirements

- Databricks workspace with **Pro or Serverless SQL Warehouse**
- Unity Catalog enabled
- CARTO Analytics Toolbox for Databricks (for QUADBIN functions)

## Deployment

```bash
# From repository root
./deploy.sh databricks --profile myworkspace --catalog main --schema raquet

# Or using environment variables
export RAQUET_DB_PROFILE=myworkspace
export RAQUET_DB_CATALOG=main
export RAQUET_DB_SCHEMA=raquet
./deploy.sh databricks
```

## Loading Data

```sql
-- Single file
CREATE TABLE catalog.schema.my_raster
  USING PARQUET LOCATION '/Volumes/catalog/schema/data/raster.parquet';

-- Partitioned dataset (directory of files from raquet-io partition)
CREATE TABLE catalog.schema.my_raster
  USING PARQUET LOCATION '/Volumes/catalog/schema/data/raster_partitioned/';
```

> **Note:** Partitioned datasets load multiple metadata rows (block=0). Use `LIMIT 1` in metadata CTEs to handle this correctly (shown in all examples below).

## Platform-Specific Notes

### Python UDFs in Unity Catalog

Unlike BigQuery and Snowflake (JavaScript), Databricks SQL Warehouse uses **Python UDFs**:

- Native `gzip` module (no Pako library needed)
- NumPy for fast array operations
- Python handles BigInt natively (no precision issues with QUADBIN blocks)
- Dependencies specified via `ENVIRONMENT` clause

### Key Limitation: 5 UDFs Per Query

**Databricks SQL Warehouse limits queries to 5 UDF calls maximum.** This affects complex multi-band queries:

```sql
-- This works (3 UDFs)
SELECT
    ST_RASTERVALUE(block, band_1, lon, lat, metadata, 0) as red,
    ST_RASTERVALUE(block, band_2, lon, lat, metadata, 1) as green,
    ST_RASTERVALUE(block, band_3, lon, lat, metadata, 2) as blue
FROM raster_data;

-- This FAILS (6 UDFs - exceeds limit)
SELECT
    ST_RASTERVALUE(block, band_1, lon, lat, metadata, 0) as red,
    ST_RASTERVALUE(block, band_2, lon, lat, metadata, 1) as green,
    ST_RASTERVALUE(block, band_3, lon, lat, metadata, 2) as blue,
    ST_RASTERVALUE(block, band_4, lon, lat, metadata, 3) as nir,
    ST_NORMALIZEDDIFFERENCE(band_4, band_1, metadata, 3, 0) as ndvi,
    ST_RASTERSUMMARYSTATS(band_1, metadata, 0) as stats
FROM raster_data;
```

**Workaround:** Split complex queries into multiple simpler queries.

### Return Type Differences

Due to Python UDF scalar-only limitation, array/struct returns are JSON strings:

| Function | BigQuery | Snowflake | Databricks |
|----------|----------|-----------|------------|
| `RAQUET_DECODE_BAND` | `ARRAY<FLOAT64>` | `ARRAY` | `STRING` (JSON array) |
| `ST_RASTERSUMMARYSTATS` | `STRUCT<...>` | `OBJECT` | `STRING` (JSON object) |
| `ST_BANDMATH` | `ARRAY<FLOAT64>` | `ARRAY` | `STRING` (JSON array) |
| `ST_NORMALIZEDDIFFERENCE` | `ARRAY<FLOAT64>` | `ARRAY` | `STRING` (JSON array) |

To parse JSON results in Databricks:

```sql
-- Parse stats JSON
SELECT
    FROM_JSON(
        ST_RASTERSUMMARYSTATS(band_1, metadata, 0),
        'STRUCT<count:BIGINT, sum:DOUBLE, mean:DOUBLE, min:DOUBLE, max:DOUBLE, stddev:DOUBLE>'
    ) as stats
FROM raster_data;

-- Or extract individual fields
SELECT
    GET_JSON_OBJECT(ST_RASTERSUMMARYSTATS(band_1, metadata, 0), '$.mean') as mean
FROM raster_data;
```

## Functions

| Function | Status | Returns | Description |
|----------|--------|---------|-------------|
| `RAQUET_DECODE_BAND` | ✅ | STRING (JSON) | Decode compressed band to array |
| `RAQUET_PIXEL` | ✅ | DOUBLE | Get pixel value at x,y |
| `ST_RASTERSUMMARYSTATS` | ✅ | STRING (JSON) | Compute band statistics |
| `ST_RASTERVALUE` | ✅ | DOUBLE | Get value at lon/lat |
| `ST_RASTERVALUE_GEOG` | ✅ | DOUBLE | Get value at GEOGRAPHY point |
| `ST_BANDMATH` | ✅ | STRING (JSON) | Pixel-wise arithmetic |
| `ST_NORMALIZEDDIFFERENCE` | ✅ | STRING (JSON) | NDVI/NDWI calculation |
| `ST_NORMALIZEDDIFFERENCESTATS` | ✅ | STRING (JSON) | Stats for normalized difference |

## Usage Examples

### Get Raster Value at Point

```sql
WITH metadata AS (
    SELECT metadata FROM catalog.schema.raster_table WHERE block = 0 LIMIT 1
)
SELECT
    ST_RASTERVALUE(
        r.block,
        r.band_1,
        -122.4, 37.8,  -- lon, lat
        m.metadata,
        0  -- band index
    ) as value
FROM catalog.schema.raster_table r, metadata m
WHERE r.block = QUADBIN_FROMGEOGPOINT(ST_POINT(-122.4, 37.8), 5)
```

### Get Tile Statistics

```sql
WITH metadata AS (
    SELECT metadata FROM catalog.schema.raster_table WHERE block = 0 LIMIT 1
)
SELECT
    r.block,
    FROM_JSON(
        ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0),
        'STRUCT<count:BIGINT, sum:DOUBLE, mean:DOUBLE, min:DOUBLE, max:DOUBLE, stddev:DOUBLE>'
    ) as stats
FROM catalog.schema.raster_table r, metadata m
WHERE r.block != 0
```

### Compute NDVI

```sql
WITH metadata AS (
    SELECT metadata FROM catalog.schema.satellite WHERE block = 0 LIMIT 1
)
SELECT
    r.block,
    FROM_JSON(
        ST_NORMALIZEDDIFFERENCESTATS(r.band_nir, r.band_red, m.metadata, 0, 1),
        'STRUCT<count:BIGINT, sum:DOUBLE, mean:DOUBLE, min:DOUBLE, max:DOUBLE, stddev:DOUBLE>'
    ) as ndvi_stats
FROM catalog.schema.satellite r, metadata m
WHERE r.block != 0
```

## Key Differences from BigQuery/Snowflake

| Aspect | BigQuery | Snowflake | Databricks |
|--------|----------|-----------|------------|
| UDF Language | JavaScript | JavaScript | Python |
| Binary input | base64 string | Uint8Array | bytes |
| Block type | INT64 | VARCHAR | BIGINT |
| Array returns | `ARRAY<FLOAT64>` | `ARRAY` | STRING (JSON) |
| Struct returns | `STRUCT` | `OBJECT` | STRING (JSON) |
| Max UDFs/query | Unlimited | Unlimited | **5** |
| Library loading | GCS external | Inlined | PyPI/Volumes |

## Dependencies

Functions use NumPy for array operations. This is specified in the `ENVIRONMENT` clause:

```sql
ENVIRONMENT (
    dependencies = '["numpy"]',
    environment_version = 'None'
)
```
