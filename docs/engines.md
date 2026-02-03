---
layout: default
title: Supported Engines
page_header: true
page_description: Query RaQuet files with DuckDB, BigQuery, Snowflake, Databricks, and more
---

## Overview

RaQuet files are standard Apache Parquet — they work with **any Parquet-compatible query engine**. For full raster analytics (point queries, zonal statistics, spatial filtering), use engines with spatial raster functions.

| Engine | Basic Parquet | Raster Functions | Notes |
|--------|--------------|------------------|-------|
| **DuckDB** | ✓ | ✓ (via extension) | Best for local analytics |
| **BigQuery** | ✓ | ✓ (Analytics Toolbox) | Fully managed, scalable |
| **Snowflake** | ✓ | ✓ (Analytics Toolbox) | Fully managed, scalable |
| **Databricks** | ✓ | ✓ (Analytics Toolbox) | Spark-based |
| **PostgreSQL** | ✓ | ✓ (Analytics Toolbox) | Self-hosted |
| **Spark** | ✓ | — | Native Parquet support |
| **Pandas** | ✓ | — | Via pyarrow |
| **Polars** | ✓ | — | Fast DataFrames |

---

## DuckDB

DuckDB provides the best local experience with the [DuckDB Raquet Extension](https://github.com/jatorre/duckdb-raquet).

### Installation

```sql
INSTALL raquet FROM community;
LOAD raquet;
```

### Key Functions

| Function | Description |
|----------|-------------|
| `read_raquet(file)` | Table function that propagates metadata |
| `ST_RasterValue(block, band, geom, metadata)` | Get pixel value at a point |
| `ST_RasterIntersects(block, geom)` | Spatial filter (EPSG:4326) |
| `ST_RasterSummaryStat(block, band, stat, metadata)` | Zonal statistics |

### Examples

**Point Query — Elevation at Madrid:**

```sql
LOAD raquet;

WITH point AS (
    SELECT 'POINT(-3.7038 40.4168)'::GEOMETRY AS geom
)
SELECT
    ST_RasterValue(block, band_1, point.geom, metadata) AS elevation_meters
FROM read_raquet('https://storage.googleapis.com/raquet_demo_data/world_elevation.parquet')
CROSS JOIN point
WHERE ST_RasterIntersects(block, point.geom);
```

**Zonal Statistics — Solar potential in a region:**

```sql
LOAD raquet;

WITH area AS (
    SELECT ST_GeomFromText('POLYGON((-4 40, -3 40, -3 41, -4 41, -4 40))') AS geom
)
SELECT
    SUM(ST_RasterSummaryStat(block, band_1, 'sum', metadata)) AS total_pvout
FROM read_raquet('https://storage.googleapis.com/raquet_demo_data/world_solar_pvout.parquet')
CROSS JOIN area
WHERE ST_RasterIntersects(block, area.geom);
```

**Time Series Analysis:**

```sql
LOAD raquet;

SELECT
    YEAR(time_ts) AS year,
    AVG(ST_RasterSummaryStat(block, band_1, 'mean', metadata)) AS avg_sst
FROM read_raquet('https://storage.googleapis.com/raquet_demo_data/cfsr_sst.parquet')
GROUP BY YEAR(time_ts)
ORDER BY year;
```

### Without Extension

RaQuet files work as standard Parquet even without the extension, but you need to manually filter the metadata row:

```sql
-- Basic query (no extension needed, manual metadata filtering)
SELECT *
FROM read_parquet('https://storage.googleapis.com/raquet_demo_data/world_elevation.parquet')
WHERE block != 0
LIMIT 10;

-- Read metadata
SELECT metadata
FROM read_parquet('file.parquet')
WHERE block = 0;
```

With the [DuckDB Raquet Extension](https://github.com/jatorre/duckdb-raquet), this becomes simpler:

```sql
INSTALL raquet FROM community;
LOAD raquet;

-- Cleaner API (metadata row excluded automatically)
SELECT * FROM read_raquet('file.parquet') LIMIT 10;

-- Read metadata
SELECT metadata FROM read_raquet_metadata('file.parquet');
```

---

## BigQuery

Query RaQuet files in BigQuery using [CARTO Analytics Toolbox](https://carto.com/analytics-toolbox).

### Loading Data

```sql
-- Create external table from GCS
CREATE EXTERNAL TABLE `project.dataset.elevation`
OPTIONS (
  format = 'PARQUET',
  uris = ['gs://raquet_demo_data/world_elevation.parquet']
);
```

### Raster Functions

With CARTO Analytics Toolbox installed:

```sql
-- Point query
SELECT
  `carto-un`.carto.RASTER_ST_VALUE(block, band_1, ST_GEOGPOINT(-3.7038, 40.4168)) AS elevation
FROM `project.dataset.elevation`
WHERE `carto-un`.carto.RASTER_ST_INTERSECTS(block, ST_GEOGPOINT(-3.7038, 40.4168));

-- Zonal statistics
SELECT
  SUM(`carto-un`.carto.RASTER_ST_SUMMARYSTATS(block, band_1).sum) AS total
FROM `project.dataset.elevation`
WHERE `carto-un`.carto.RASTER_ST_INTERSECTS(
  block,
  ST_GEOGFROMTEXT('POLYGON((-4 40, -3 40, -3 41, -4 41, -4 40))')
);
```

---

## Snowflake

Query RaQuet files in Snowflake using [CARTO Analytics Toolbox](https://carto.com/analytics-toolbox).

### Loading Data

```sql
-- Create stage for external data
CREATE STAGE raquet_stage
  URL = 'gcs://raquet_demo_data/'
  FILE_FORMAT = (TYPE = PARQUET);

-- Query directly from stage
SELECT *
FROM @raquet_stage/world_elevation.parquet
LIMIT 10;
```

### Raster Functions

With CARTO Analytics Toolbox:

```sql
SELECT
  carto.RASTER_ST_VALUE(block, band_1, ST_POINT(-3.7038, 40.4168)) AS elevation
FROM @raquet_stage/world_elevation.parquet
WHERE carto.RASTER_ST_INTERSECTS(block, ST_POINT(-3.7038, 40.4168));
```

---

## Databricks

Query RaQuet files in Databricks using Spark's native Parquet support plus [CARTO Analytics Toolbox](https://carto.com/analytics-toolbox).

### Loading Data

```python
# Read from cloud storage
df = spark.read.parquet("gs://raquet_demo_data/world_elevation.parquet")
df.createOrReplaceTempView("elevation")
```

### SQL Queries

```sql
-- Basic query
SELECT * FROM elevation WHERE block != 0 LIMIT 10;

-- With CARTO Analytics Toolbox
SELECT
  carto.RASTER_ST_VALUE(block, band_1, ST_POINT(-3.7038, 40.4168)) AS elevation
FROM elevation
WHERE carto.RASTER_ST_INTERSECTS(block, ST_POINT(-3.7038, 40.4168));
```

---

## PostgreSQL

Query RaQuet files in PostgreSQL using [CARTO Analytics Toolbox](https://carto.com/analytics-toolbox) or load data directly.

### Using parquet_fdw

```sql
-- Install parquet_fdw extension
CREATE EXTENSION parquet_fdw;

-- Create foreign table
CREATE FOREIGN TABLE elevation (
  block BIGINT,
  band_1 BYTEA,
  metadata JSONB
)
SERVER parquet_server
OPTIONS (filename '/path/to/world_elevation.parquet');

-- Query
SELECT * FROM elevation WHERE block != 0 LIMIT 10;
```

### With CARTO Analytics Toolbox

```sql
SELECT
  carto.RASTER_ST_VALUE(block, band_1, ST_SetSRID(ST_MakePoint(-3.7038, 40.4168), 4326)) AS elevation
FROM elevation
WHERE carto.RASTER_ST_INTERSECTS(block, ST_SetSRID(ST_MakePoint(-3.7038, 40.4168), 4326));
```

---

## Python (pandas/polars)

### pandas

```python
import pandas as pd

# Read RaQuet file
df = pd.read_parquet('https://storage.googleapis.com/raquet_demo_data/spain_solar_ghi.parquet')

# Get metadata
metadata_row = df[df['block'] == 0].iloc[0]
metadata = json.loads(metadata_row['metadata'])

# Filter data rows
data = df[df['block'] != 0]
```

### polars

```python
import polars as pl

# Read RaQuet file
df = pl.read_parquet('https://storage.googleapis.com/raquet_demo_data/spain_solar_ghi.parquet')

# Get metadata
metadata = df.filter(pl.col('block') == 0).select('metadata').item()

# Filter data rows
data = df.filter(pl.col('block') != 0)
```

---

## CARTO Analytics Toolbox

[CARTO Analytics Toolbox](https://carto.com/analytics-toolbox) provides consistent raster functions across BigQuery, Snowflake, Databricks, and PostgreSQL.

### Raster Functions

| Function | Description |
|----------|-------------|
| `RASTER_ST_VALUE` | Get pixel value at a point |
| `RASTER_ST_INTERSECTS` | Spatial filter |
| `RASTER_ST_SUMMARYSTATS` | Zonal statistics (sum, mean, min, max, etc.) |
| `RASTER_ST_CLIP` | Clip raster to geometry |

### Installation

See [CARTO Analytics Toolbox documentation](https://docs.carto.com/data-and-analysis/analytics-toolbox) for installation instructions for your platform.

---

## Performance Tips

1. **Use spatial filtering** — Always include `ST_RasterIntersects` or equivalent to enable row group pruning

2. **Query remote files directly** — Parquet's columnar format enables efficient range requests; no need to download first

3. **Split by zoom for large files** — Use `raquet-io split-zoom` to create per-zoom files for optimal remote queries

4. **Small row groups for remote access** — Use `--row-group-size 100-200` when converting files that will be queried remotely

