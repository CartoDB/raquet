---
layout: default
title: Overview
hero: true
hero_tagline: "Query rasters with SQL. Treat rasters as tables. Bring raster data into the lakehouse."
---

## What is RaQuet?

<div class="feature-grid">
  <div class="feature-card">
    <h3>Format</h3>
    <p>RaQuet defines an open specification for storing raster data in Apache Parquet. Each tile becomes a row, each band becomes a column. Standard format, no proprietary extensions.</p>
    <a href="https://github.com/CartoDB/raquet/blob/master/format-specs/raquet.md">Read the specification →</a>
  </div>
  <div class="feature-card">
    <h3>Tools</h3>
    <p>Convert any GDAL-supported raster (GeoTIFF, NetCDF, COG) to RaQuet. Query with DuckDB, visualize in the browser, or load into your data warehouse.</p>
    <a href="{{ site.baseurl }}/cli">View CLI reference →</a>
  </div>
  <div class="feature-card">
    <h3>Ecosystem</h3>
    <p>RaQuet is designed to work with the modern analytics stack. Full support for BigQuery, Snowflake, Databricks, DuckDB, and PostgreSQL through CARTO's Analytics Toolbox.</p>
    <a href="{{ site.baseurl }}/engines">See supported engines →</a>
  </div>
</div>

---

## Why Parquet for Raster Data?

**We believe people want to access their raster data like any other type of data: in SQL.**

You shouldn't have to export data and perform vector-raster intersections outside your analytics platform. But today, you can't just query a raster. Raster data remains locked in GIS- and HPC-oriented formats like GeoTIFF/COG and Zarr — powerful, but largely invisible to SQL engines.

RaQuet builds on the pioneering work of [PostGIS Raster](https://postgis.net/docs/RT_reference.html), which first demonstrated SQL-based raster analytics. But instead of being tied to PostgreSQL, RaQuet uses **Apache Parquet** — an open columnar format supported by virtually every modern analytics engine.

> **Key insight:** With RaQuet, **raster files become tables** in your data warehouse. Instead of treating rasters as opaque files, you can query them, join them with vector data, and govern them — all in the same system.

---

## RaQuet Principles

RaQuet's goal is to **align GIS with the rest of the analytics industry** — particularly Open Table Formats like Apache Iceberg and the separation of storage and compute.

- **SQL-First** — Query raster data with standard SQL in DuckDB, BigQuery, Snowflake, Spark, or any Parquet-compatible engine
- **Tables, Not Files** — Rasters become queryable tables, not opaque binary blobs
- **Interoperable** — Works with existing data warehouse infrastructure and governance
- **Cloud-Native** — QUADBIN spatial indexing enables efficient tile lookups with row group pruning
- **Open Format** — Standard Parquet with no proprietary extensions

---

## RaQuet vs COG vs Zarr

**RaQuet isn't competing with traditional raster formats** — it targets a different problem entirely: **interoperability in the analytics world**.

| | **COG (GeoTIFF)** | **Zarr** | **RaQuet** |
|---|---|---|---|
| **Best for** | GIS pipelines, visualization | Scientific computing (HPC) | Analytics / lakehouse / SQL |
| **Ecosystem** | GDAL, QGIS, rasterio | Xarray, Dask, Pangeo | DuckDB, BigQuery, Snowflake, Spark |
| **Strength** | Window reads, tiling, overviews | Chunked arrays, parallel compute | SQL queries, joins with vector data |
| **Limitation** | Not queryable in SQL | Requires specialized runtimes | Designed for tiles, not window reads |

**RaQuet works out of the box** in most analytics systems — and often provides comparable or better performance than pipelines involving export/import steps.

---

## Sample Data

Try these example RaQuet files — query them directly from cloud storage:

| Dataset | Source | Source Size | RaQuet Size | URL |
|---------|--------|-------------|-------------|-----|
| **World Elevation** | AAIGrid | 3.2 GB | 805 MB | [world_elevation.parquet](https://storage.googleapis.com/raquet_demo_data/world_elevation.parquet) |
| **World Solar PVOUT** | AAIGrid | 2.8 GB | 255 MB | [world_solar_pvout.parquet](https://storage.googleapis.com/raquet_demo_data/world_solar_pvout.parquet) |
| **CFSR SST** | NetCDF | 854 MB | 75 MB | [cfsr_sst.parquet](https://storage.googleapis.com/raquet_demo_data/cfsr_sst.parquet) |
| **TCI (Sentinel-2)** | GeoTIFF | 224 MB | 256 MB | [TCI.parquet](https://storage.googleapis.com/raquet_demo_data/TCI.parquet) |
| **Spain Solar GHI** | GeoTIFF | — | 15 MB | [spain_solar_ghi.parquet](https://storage.googleapis.com/raquet_demo_data/spain_solar_ghi.parquet) |

Data sources: [Global Solar Atlas](https://globalsolaratlas.info/), [Copernicus Sentinel-2](https://sentinel.esa.int/), [CFSR Reanalysis](https://rda.ucar.edu/datasets/ds093.0/)

---

## Example Queries

### Get Elevation at Madrid

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

### Sum Solar Potential in a Region

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

### Time-Series Analysis

```sql
LOAD raquet;

SELECT
    YEAR(time_ts) AS year,
    AVG(ST_RasterSummaryStat(block, band_1, 'mean', metadata)) AS avg_sst
FROM read_raquet('https://storage.googleapis.com/raquet_demo_data/cfsr_sst.parquet')
GROUP BY YEAR(time_ts)
ORDER BY year;
```

**Key functions:**
- `read_raquet(file)` — Table function that propagates metadata automatically
- `ST_RasterValue(block, band, geom, metadata)` — Returns pixel value at a POINT
- `ST_RasterIntersects(block, geom)` — Spatial filter (EPSG:4326)

---

## How It Works

Think of RaQuet as storing a raster where **each tile is a row** and **each band is a column**.

| block | band_1 | band_2 | band_3 | metadata |
|-------|--------|--------|--------|----------|
| 0 | NULL | NULL | NULL | `{"version": "0.3.0", ...}` |
| 5270498377487261695 | `<gzip>` | `<gzip>` | `<gzip>` | NULL |
| 5270498377487327231 | `<gzip>` | `<gzip>` | `<gzip>` | NULL |

- **block** — [QUADBIN](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery/key-concepts/spatial-indexes#quadbin) cell ID (tile location + zoom level)
- **band_N** — Gzip-compressed pixel data (row-major)
- **metadata** — JSON metadata in the `block=0` row

RaQuet requires **Web Mercator (EPSG:3857)** projection to leverage QUADBIN spatial indexing for efficient filtering.

---

## Getting Started

```bash
# Install
pip install raquet-io

# Convert a raster to RaQuet
raquet-io convert raster input.tif output.parquet

# Validate the output
raquet-io validate output.parquet

# Inspect metadata
raquet-io inspect output.parquet
```

<div style="margin-top: 2rem;">
<a href="{{ site.baseurl }}/viewer" class="btn btn-primary">Try the Viewer</a>
<a href="{{ site.baseurl }}/cli" class="btn btn-secondary" style="border-color: var(--color-accent); color: var(--color-accent);">CLI Reference</a>
</div>

---

## Roadmap: Apache Iceberg Integration

<div class="notice notice-info">
<strong>Status:</strong> Active development — not yet generally available.
</div>

We're working on registering RaQuet datasets as [Apache Iceberg](https://iceberg.apache.org/) tables — enabling rasters to be discovered and queried alongside vector data in your lakehouse.

GeoParquet brought vector data into the lakehouse. RaQuet does the same for raster. Iceberg unifies them under a single governance layer.

[Follow progress on GitHub →](https://github.com/CartoDB/raquet)
