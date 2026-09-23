# RaQuet SQL engines

Multi-platform SQL UDFs for querying [RaQuet](../format-specs/raquet.md)-formatted raster data in BigQuery, Snowflake and Databricks: decode tiles, read pixel values, compute statistics and band math, and compute new rasters with `RASTER_ALGEBRA`.

These are standalone reference implementations, deployable into your own project. For managed, production-supported raster functions, see the [CARTO Analytics Toolbox](https://carto.com/analytics-toolbox).

## What is Raquet?

Raquet is a specification for storing raster data in Apache Parquet format using QUADBIN spatial indexing. It enables:
- SQL-based raster queries in data warehouses
- Efficient spatial filtering via QUADBIN
- Cloud-native storage with Parquet compression

## Supported Platforms

| Platform | Status | UDF Language | Notes |
|----------|--------|--------------|-------|
| BigQuery | ✅ Full support | JavaScript | External GCS libraries |
| Snowflake | ✅ Full support | JavaScript | Inlined libraries |
| Databricks | ✅ SQL Warehouse | Python | **Max 5 UDFs/query** |

## Functions

| Function | BigQuery | Snowflake | Databricks | Description |
|----------|:--------:|:---------:|:----------:|-------------|
| `RAQUET_DECODE_BAND` | ✅ | ✅ | ✅ | Decode compressed band to pixel array |
| `RAQUET_PIXEL` | ✅ | ✅ | ✅ | Get single pixel at tile coordinates |
| `ST_RASTERSUMMARYSTATS` | ✅ | ✅ | ✅ | Compute tile statistics |
| `ST_RASTERVALUE` | ✅ | ✅ | ✅ | Get value at geographic point |
| `ST_RASTERVALUE_GEOG` | ✅ | ✅ | ✅ | Same as above, accepts GEOGRAPHY |
| `ST_BANDMATH` | ✅ | ✅ | ✅ | Pixel-wise arithmetic (+, -, *, /) |
| `ST_NORMALIZEDDIFFERENCE` | ✅ | ✅ | ✅ | (b1-b2)/(b1+b2) for indices like NDVI |
| `ST_NORMALIZEDDIFFERENCESTATS` | ✅ | ✅ | ✅ | Stats for normalized difference |
| `RASTER_ALGEBRA` (procedure) | ✅ | 🧪 | — | Raster → raster: evaluate an expression over N rasters, write a RaQuet v0.5.0 table ([docs](docs/raster-algebra.md)) |
| `RAQUET_AGGREGATE_STATS` | ✅ | ✅ | ✅ | Aggregate stats across tiles |
| `RAQUET_BATCH_STATS` (UDTF) | — | ✅ | — | Per-tile stats for many tiles in one JS invocation |
| `RAQUET_PIXEL_GEOGRAPHY` | ✅ | ✅ | ✅ | Get GEOGRAPHY/WKT for pixel center |
| `__RAQUET_RESOLVE_ZOOM` | ✅ | ✅ | ✅ | Resolve flexible resolution parameter |
| `__RAQUET_AUTO_ZOOM` | ✅ | ✅ | ✅ | Auto-detect zoom level |
| `__RAQUET_PIXEL_POSITIONS` | ✅ | ✅ | ✅ | Generates 256x256 pixel grid (VIEW on SF/DB) |
| `__RAQUET_REGION_BLOCKS` | ✅ | ✅ | ✅ | Finds blocks intersecting region (ARRAY on SF/DB) |

## Cross-Platform Validation

All platforms produce identical results (validated with TCI satellite imagery, 3225 tiles):

| Test | BigQuery | Snowflake | Databricks |
|------|----------|-----------|------------|
| ST_RASTERSUMMARYSTATS (single tile) | count=3657, mean=246.452 | count=3657, mean=246.452 | count=3657, mean=246.452 |
| ST_RASTERSUMMARYSTATS (all tiles) | 193,146,254 pixels | 193,146,254 pixels | 193,146,254 pixels |
| RAQUET_PIXEL (233, 249) | 243.0 | 243.0 | 243.0 |
| ST_RASTERVALUE (33.5, 16.7) | 243.0 | 243.0 | 243.0 |

## Quick Start

### Deployment

BigQuery loads its UDF code from JavaScript bundles in GCS. The bundles are not committed, so build them before deploying (requires Node.js):

```bash
(cd libraries/javascript && npm ci && npm run build)
```

```bash
# BigQuery (uploads libraries/javascript/build/*.js to the bucket)
./deploy.sh bigquery --bucket gs://your-bucket --dataset yourproject.raquet

# Snowflake
./deploy.sh snowflake --connection myconn --database MYDB --schema RAQUET

# Databricks SQL Warehouse
./deploy.sh databricks --profile myworkspace --catalog main --schema raquet

# Run cross-platform validation tests
./deploy.sh test --verbose
```

### Load Raquet Data

```bash
# BigQuery - load from GCS (single file or partitioned directory)
bq load --source_format=PARQUET yourproject:raquet.my_raster gs://bucket/raster.parquet
bq load --source_format=PARQUET yourproject:raquet.my_raster 'gs://bucket/raster_partitioned/*.parquet'

# Snowflake - load from stage (supports partitioned datasets)
COPY INTO MY_RASTER FROM @my_stage/raster/
  FILE_FORMAT = (TYPE = PARQUET BINARY_AS_TEXT = FALSE)
  MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

# Databricks - create table from volume or cloud storage
CREATE TABLE catalog.schema.my_raster USING PARQUET LOCATION '/path/to/raster/';
```

> **Partitioned datasets:** When loading multiple Raquet partition files into a single table, the table will contain multiple metadata rows (block=0, one per partition file). All query examples below use `LIMIT 1` on the metadata CTE to handle this correctly. The metadata content is identical across partitions.

---

## SQL Examples by Platform

### BigQuery

#### Get raster value at a point

```sql
-- Get the red band value at coordinates (33.5, 16.7)
WITH metadata AS (
    SELECT metadata FROM `project.dataset.satellite` WHERE block = 0 LIMIT 1
)
SELECT
    `project.dataset.ST_RASTERVALUE`(
        r.block,
        r.band_1,
        33.5, 16.7,  -- lon, lat
        m.metadata,
        0  -- band index
    ) as red_value
FROM `project.dataset.satellite` r, metadata m
WHERE r.block = `carto-un`.carto.QUADBIN_FROMGEOGPOINT(ST_GEOGPOINT(33.5, 16.7), 7)
```

#### Compute tile statistics

```sql
-- Get statistics for a specific tile
WITH metadata AS (
    SELECT metadata FROM `project.dataset.raster` WHERE block = 0 LIMIT 1
)
SELECT
    r.block,
    `project.dataset.ST_RASTERSUMMARYSTATS`(r.band_1, m.metadata, 0) as stats
FROM `project.dataset.raster` r, metadata m
WHERE r.block = 5221556531052412927
```

#### Aggregate statistics across all tiles

```sql
-- Compute global statistics across entire raster
WITH metadata AS (
    SELECT metadata FROM `project.dataset.raster` WHERE block = 0 LIMIT 1
),
tile_stats AS (
    SELECT `project.dataset.ST_RASTERSUMMARYSTATS`(r.band_1, m.metadata, 0) as stats
    FROM `project.dataset.raster` r, metadata m
    WHERE r.block != 0
)
SELECT
    COUNT(*) as num_tiles,
    SUM((stats).count) as total_pixels,
    AVG((stats).mean) as avg_mean,
    MIN((stats).min) as global_min,
    MAX((stats).max) as global_max
FROM tile_stats
```

#### Compute NDVI

```sql
-- Calculate NDVI statistics: (NIR - Red) / (NIR + Red)
WITH metadata AS (
    SELECT metadata FROM `project.dataset.satellite` WHERE block = 0 LIMIT 1
)
SELECT
    r.block,
    `project.dataset.ST_NORMALIZEDDIFFERENCESTATS`(
        r.band_nir, r.band_red, m.metadata, 0, 1
    ) as ndvi_stats
FROM `project.dataset.satellite` r, metadata m
WHERE r.block != 0
```

#### Get RGB values at a location

```sql
-- Extract RGB values at a point
WITH metadata AS (
    SELECT metadata FROM `project.dataset.satellite` WHERE block = 0 LIMIT 1
),
point AS (SELECT 33.5 as lon, 16.7 as lat)
SELECT
    `project.dataset.ST_RASTERVALUE`(r.block, r.band_1, p.lon, p.lat, m.metadata, 0) as red,
    `project.dataset.ST_RASTERVALUE`(r.block, r.band_2, p.lon, p.lat, m.metadata, 1) as green,
    `project.dataset.ST_RASTERVALUE`(r.block, r.band_3, p.lon, p.lat, m.metadata, 2) as blue
FROM `project.dataset.satellite` r, metadata m, point p
WHERE r.block = `carto-un`.carto.QUADBIN_FROMGEOGPOINT(ST_GEOGPOINT(p.lon, p.lat), 7)
```

---

### Snowflake

#### Get raster value at a point

```sql
-- Get the red band value at coordinates (33.5, 16.7)
-- Note: block is cast to VARCHAR to preserve BigInt precision
WITH metadata AS (
    SELECT metadata FROM SATELLITE WHERE block = 0 LIMIT 1
)
SELECT
    ST_RASTERVALUE(
        r.block::VARCHAR,
        r.band_1,
        33.5, 16.7,  -- lon, lat
        m.metadata,
        0  -- band index
    ) as red_value
FROM SATELLITE r, metadata m
WHERE r.block = QUADBIN_FROMGEOGPOINT(ST_MAKEPOINT(33.5, 16.7), 7)
```

#### Compute tile statistics

```sql
-- Get statistics for a specific tile
-- Returns OBJECT with count, sum, mean, min, max, stddev
WITH metadata AS (
    SELECT metadata FROM RASTER_TABLE WHERE block = 0 LIMIT 1
)
SELECT
    r.block,
    ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats,
    (ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0)):mean::FLOAT as mean_value
FROM RASTER_TABLE r, metadata m
WHERE r.block = 5221556531052412927
```

#### Aggregate statistics across all tiles

```sql
-- Compute global statistics across entire raster
WITH metadata AS (
    SELECT metadata FROM RASTER_TABLE WHERE block = 0 LIMIT 1
),
tile_stats AS (
    SELECT ST_RASTERSUMMARYSTATS(r.band_1, m.metadata, 0) as stats
    FROM RASTER_TABLE r, metadata m
    WHERE r.block != 0
)
SELECT
    COUNT(*) as num_tiles,
    SUM(stats:count::INT) as total_pixels,
    AVG(stats:mean::FLOAT) as avg_mean,
    MIN(stats:min::FLOAT) as global_min,
    MAX(stats:max::FLOAT) as global_max
FROM tile_stats
```

#### Compute NDVI

```sql
-- Calculate NDVI: (NIR - Red) / (NIR + Red)
WITH metadata AS (
    SELECT metadata FROM SATELLITE WHERE block = 0 LIMIT 1
)
SELECT
    r.block,
    ST_NORMALIZEDDIFFERENCE(r.band_nir, r.band_red, m.metadata, 0, 1) as ndvi_array
FROM SATELLITE r, metadata m
WHERE r.block != 0
```

#### Get RGB values at a location

```sql
-- Extract RGB values at a point
WITH metadata AS (
    SELECT metadata FROM SATELLITE WHERE block = 0 LIMIT 1
)
SELECT
    ST_RASTERVALUE(r.block::VARCHAR, r.band_1, 33.5, 16.7, m.metadata, 0) as red,
    ST_RASTERVALUE(r.block::VARCHAR, r.band_2, 33.5, 16.7, m.metadata, 1) as green,
    ST_RASTERVALUE(r.block::VARCHAR, r.band_3, 33.5, 16.7, m.metadata, 2) as blue
FROM SATELLITE r, metadata m
WHERE r.block = QUADBIN_FROMGEOGPOINT(ST_MAKEPOINT(33.5, 16.7), 7)
```

---

### Databricks SQL Warehouse

> **Note:** Databricks has a **5 UDF per query limit**. Split complex queries if needed.
> Array/struct returns are JSON strings - use `GET_JSON_OBJECT()` or `FROM_JSON()` to parse.

#### Get raster value at a point

```sql
-- Get the red band value at coordinates (33.5, 16.7)
WITH metadata AS (
    SELECT metadata FROM catalog.schema.satellite WHERE block = 0 LIMIT 1
)
SELECT
    catalog.schema.st_rastervalue(
        r.block,
        r.band_1,
        33.5, 16.7,  -- lon, lat
        m.metadata,
        0  -- band index
    ) as red_value
FROM catalog.schema.satellite r, metadata m
WHERE r.block = QUADBIN_FROMGEOGPOINT(ST_POINT(33.5, 16.7), 7)
```

#### Compute tile statistics

```sql
-- Get statistics for a specific tile
-- Returns JSON string: {"count": N, "sum": X, "mean": X, "min": X, "max": X, "stddev": X}
WITH metadata AS (
    SELECT metadata FROM catalog.schema.raster WHERE block = 0 LIMIT 1
)
SELECT
    r.block,
    catalog.schema.st_rastersummarystats(r.band_1, m.metadata, 0) as stats_json,
    GET_JSON_OBJECT(
        catalog.schema.st_rastersummarystats(r.band_1, m.metadata, 0),
        '$.mean'
    ) as mean_value
FROM catalog.schema.raster r, metadata m
WHERE r.block = 5221556531052412927
```

#### Aggregate statistics across all tiles

```sql
-- Compute global statistics across entire raster
WITH metadata AS (
    SELECT metadata FROM catalog.schema.raster WHERE block = 0 LIMIT 1
),
tile_stats AS (
    SELECT catalog.schema.st_rastersummarystats(r.band_1, m.metadata, 0) as stats
    FROM catalog.schema.raster r, metadata m
    WHERE r.block != 0
)
SELECT
    COUNT(*) as num_tiles,
    SUM(CAST(GET_JSON_OBJECT(stats, '$.count') AS BIGINT)) as total_pixels,
    AVG(CAST(GET_JSON_OBJECT(stats, '$.mean') AS DOUBLE)) as avg_mean
FROM tile_stats
```

#### Compute NDVI statistics

```sql
-- Calculate NDVI statistics: (NIR - Red) / (NIR + Red)
WITH metadata AS (
    SELECT metadata FROM catalog.schema.satellite WHERE block = 0 LIMIT 1
)
SELECT
    r.block,
    catalog.schema.st_normalizeddifferencestats(
        r.band_nir, r.band_red, m.metadata, 0, 1
    ) as ndvi_stats_json
FROM catalog.schema.satellite r, metadata m
WHERE r.block != 0
```

#### Get RGB values at a location (uses 3 UDFs)

```sql
-- Extract RGB values at a point
-- Note: This uses 3 UDF calls (within the 5 UDF limit)
WITH metadata AS (
    SELECT metadata FROM catalog.schema.satellite WHERE block = 0 LIMIT 1
)
SELECT
    catalog.schema.st_rastervalue(r.block, r.band_1, 33.5, 16.7, m.metadata, 0) as red,
    catalog.schema.st_rastervalue(r.block, r.band_2, 33.5, 16.7, m.metadata, 1) as green,
    catalog.schema.st_rastervalue(r.block, r.band_3, 33.5, 16.7, m.metadata, 2) as blue
FROM catalog.schema.satellite r, metadata m
WHERE r.block = QUADBIN_FROMGEOGPOINT(ST_POINT(33.5, 16.7), 7)
```

#### Parse JSON stats with FROM_JSON

```sql
-- Use FROM_JSON for structured access to statistics
WITH metadata AS (
    SELECT metadata FROM catalog.schema.raster WHERE block = 0 LIMIT 1
),
stats AS (
    SELECT
        r.block,
        FROM_JSON(
            catalog.schema.st_rastersummarystats(r.band_1, m.metadata, 0),
            'STRUCT<count:BIGINT, sum:DOUBLE, mean:DOUBLE, min:DOUBLE, max:DOUBLE, stddev:DOUBLE>'
        ) as s
    FROM catalog.schema.raster r, metadata m
    WHERE r.block != 0
)
SELECT
    block,
    s.count,
    s.mean,
    s.stddev
FROM stats
```

---

## Platform-Specific Notes

### BigQuery
- JavaScript UDFs load external libraries from GCS bucket
- Binary data passed as base64-encoded strings
- Uses `STRUCT` for return types (access with `(stats).field`)
- Requires CARTO Analytics Toolbox for QUADBIN functions
- See [platforms/bigquery/README.md](platforms/bigquery/README.md)

### Snowflake
- JavaScript libraries inlined in SQL files (~25KB per function)
- Binary data passed directly as `Uint8Array`
- Block IDs use `VARCHAR` to preserve BigInt precision (QUADBIN blocks exceed JS `MAX_SAFE_INTEGER`)
- Uses `OBJECT` for return types (access with `stats:field::TYPE`)
- Requires CARTO Analytics Toolbox for QUADBIN functions
- See [platforms/snowflake/README.md](platforms/snowflake/README.md)

### Databricks SQL Warehouse
- Python UDFs with NumPy for fast array operations
- Native `gzip` module (no Pako library needed)
- **Limitation: Max 5 UDF calls per query** (split complex queries)
- Array/struct returns are JSON strings (parse with `GET_JSON_OBJECT` or `FROM_JSON`)
- Requires CARTO Analytics Toolbox for QUADBIN functions
- See [platforms/databricks/README.md](platforms/databricks/README.md)

## Supported Data Types

- `uint8`, `int8`
- `uint16`, `int16`
- `uint32`, `int32`
- `uint64`, `int64`
- `float16`, `float32`, `float64`

Note: `float16` (half-precision) is useful for ML/inference use cases where storage size matters.

## Project Structure

```
sql/
├── deploy.sh                      # Unified deployment script
├── platforms/
│   ├── bigquery/
│   │   ├── functions/             # BigQuery SQL functions (15)
│   │   └── README.md
│   ├── snowflake/
│   │   ├── functions/             # Snowflake SQL functions (16)
│   │   └── README.md
│   └── databricks/
│       ├── functions/             # Databricks SQL/Python UDFs (14)
│       └── README.md
├── test/
│   └── validate.py                # Cross-platform validation tests
├── docs/
│   └── feasibility/               # Platform feasibility studies
└── libraries/
    └── javascript/                # Shared JS libraries (pako, core logic)
```

## Dependencies

- [CARTO Analytics Toolbox](https://docs.carto.com/carto-user-manual/analytics-toolbox-for-bigquery/) for QUADBIN functions (all platforms)
- [pako](https://github.com/nodeca/pako) for gzip decompression (bundled in JS)
- [NumPy](https://numpy.org/) for Databricks Python UDFs

## License

Apache 2.0

## Related Projects

- [Raquet](https://github.com/cartodb/raquet) - Raquet specification and Python tools
- [DuckDB Raquet](https://github.com/cartodb/duckdb-raquet) - DuckDB extension for Raquet
