---
layout: default
title: Home
---

<p align="center">
  <img src="{{ site.baseurl }}/assets/logo.svg" alt="RaQuet" width="400">
  <br>
  <em>An open source project by <a href="https://carto.com">CARTO</a></em>
</p>

# RaQuet: Raster + Parquet

<p style="font-size: 1.2em; color: #666; margin-top: -10px;">
Bring raster data into the lakehouse — query rasters with SQL, powered by production-grade analytics from CARTO.
</p>

RaQuet is a specification for storing and querying raster data using [Apache Parquet](https://parquet.apache.org/), enabling efficient cloud-native raster workflows. Developed by [CARTO](https://carto.com), RaQuet brings raster data into the modern data stack.

## Why RaQuet?

- **Cloud-Native**: Query raster data directly from cloud storage using DuckDB, BigQuery, or any Parquet-compatible tool
- **Efficient**: QUADBIN spatial indexing enables fast tile lookups with row group pruning
- **Simple**: Standard Parquet format works with existing data warehouse infrastructure
- **Interoperable**: Convert from GeoTIFF, COG, NetCDF, or ArcGIS ImageServer

---

## Why Parquet for Raster Data?

RaQuet exists to bring raster data into the same analytical ecosystem where vector data already lives. Over the last decade, vector geospatial data has successfully integrated into cloud data warehouses and lakehouses through open formats like Parquet, GeoParquet, and table formats like Iceberg. Raster data, however, remains locked in GIS- and HPC-oriented formats like GeoTIFF/COG and Zarr — powerful, but largely invisible to SQL engines and analytics platforms.

RaQuet bridges that gap by encoding rasters as Parquet. This makes raster tiles queryable with DuckDB, BigQuery, Snowflake, Spark, Trino/Presto — and makes rasters governable, versionable, and joinable inside the lakehouse.

> **Key idea:** COG/Zarr optimize storage & access. RaQuet optimizes integration & computation in the modern data stack.

<p align="center" style="margin: 30px 0;">
  <img src="{{ site.baseurl }}/assets/one-does-not-simply-query-a-raster.jpg" alt="Meme: One does not simply query a raster" style="max-width: 520px; width: 100%; border-radius: 8px;">
  <br>
  <em style="color: #666;">Because in most analytics stacks… you don't simply query a raster.</em>
  <br>
  <small style="color: #999; font-size: 0.75em;">Meme image used for illustrative purposes.</small>
</p>

### RaQuet vs COG vs Zarr

These formats serve different workflows and are **complementary**, not competing:

| | **COG (GeoTIFF)** | **Zarr** | **RaQuet** |
|---|---|---|---|
| **Best for** | GIS pipelines | Scientific / array computing | Analytics / lakehouse |
| **Strengths** | Optimized for GDAL-style window reads and visualization. Great for tiling + overviews. | Chunked multidimensional arrays (NumPy/Xarray/HPC). Parallel-friendly and cloud-native. | Parquet-native: works with warehouses and SQL engines. Joins with vector data in the same stack. |
| **Limitation** | Not natively queryable in SQL engines | Requires specialized runtimes/APIs (not warehouse-native) | Designed for tiles, not arbitrary window reads |

**RaQuet is complementary to COG and Zarr** — it's the representation designed for SQL + lakehouse workflows.

### Backed by Production Analytics Engines

RaQuet isn't just a specification — it's designed to plug directly into [CARTO's Analytics Toolbox](https://carto.com/analytics-toolbox), which already runs **natively inside major data warehouses**. This means you get production-grade spatial functions, not just file format support.

<div style="margin: 20px 0;">

**Supported platforms:**

- **BigQuery** — Full support via [Analytics Toolbox for BigQuery](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery)
- **Snowflake** — Full support via [Analytics Toolbox for Snowflake](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-snowflake)
- **Databricks** — Full support via [Analytics Toolbox for Databricks](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-databricks)
- **PostgreSQL** — Full support via [Analytics Toolbox for PostgreSQL](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-postgresql)
- **Redshift** — *Coming soon*
- **DuckDB** — *Coming soon*
- **Oracle** — *Coming soon*

</div>

With CARTO's toolboxes, you can perform spatial joins between raster tiles and vector geometries, run zonal statistics, and build ML pipelines — all in pure SQL, inside your warehouse.

### Roadmap: Apache Iceberg Integration

<p style="background: #f0f7ff; border-left: 4px solid #0066cc; padding: 12px 16px; margin: 20px 0;">
<strong>Status:</strong> Active development — not yet generally available.
</p>

We're actively working on support for registering RaQuet datasets as [Apache Iceberg](https://iceberg.apache.org/) tables. The goal: **publish rasters directly into Iceberg catalogs** so they can be discovered and queried like any other table in your lakehouse.

GeoParquet brought vector data into the lakehouse. RaQuet does the same for raster. Iceberg unifies them under a single governance and query layer — enabling true multimodal spatial analytics where vector and raster live side by side, versioned together, and queryable with the same SQL engine.

We're collaborating with the community to define best practices for spatial data in Iceberg. Follow progress on the [GitHub repository](https://github.com/CartoDB/raquet) or reach out if you're interested in early access.

---

## Quick Start

```bash
# Install
pip install raquet-io

# Convert a raster (GeoTIFF, COG, NetCDF) to RaQuet
raquet-io convert raster input.tif output.parquet
raquet-io convert raster climate.nc climate.parquet  # NetCDF with time dimension

# Inspect a RaQuet file
raquet-io inspect output.parquet

# Query with DuckDB
duckdb -c "SELECT * FROM read_parquet('output.parquet') WHERE block != 0 LIMIT 5"
```

## Try It Now

**[Open the RaQuet Viewer](viewer.html)** - A client-side viewer powered by [hyparquet](https://github.com/hyparam/hyparquet). Load any RaQuet file from a URL and explore it interactively using HTTP range requests, no server required!

---

## How It Works

Each row in a RaQuet file represents a single rectangular tile of raster data:

| block | band_1 | band_2 | band_3 | metadata |
|-------|--------|--------|--------|----------|
| 0 | NULL | NULL | NULL | `{"version": "0.1.0", ...}` |
| 5270201491262341119 | `<binary>` | `<binary>` | `<binary>` | NULL |
| 5270201491262406655 | `<binary>` | `<binary>` | `<binary>` | NULL |

- **block**: [QUADBIN](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery/key-concepts/spatial-indexes#quadbin) cell ID encoding tile location and zoom level
- **band_N**: Gzip-compressed binary pixel data (row-major order)
- **metadata**: JSON metadata in the special `block=0` row

## Resources

### Documentation

- **[Format Specification](https://github.com/CartoDB/raquet/blob/master/format-specs/raquet.md)** - Complete technical specification
- **[CLI Reference](#cli-reference)** - Command-line tool documentation
- **[Python API](https://github.com/CartoDB/raquet#python-api)** - Programmatic usage

### Tools

- **[RaQuet Viewer](viewer.html)** - Browser-based viewer using HTTP range requests ([how it works](viewer-how-it-works.html))
- **[raquet CLI](#cli-reference)** - Convert, inspect, and export RaQuet files

---

## CLI Reference

### Installation

```bash
# Basic installation
pip install raquet-io

# With all features
pip install "raquet-io[all]"
```

**Note:** GDAL must be installed separately. On macOS: `brew install gdal`

### Commands

#### `raquet-io inspect`

Display metadata and statistics for a RaQuet file.

```bash
raquet-io inspect landcover.parquet
raquet-io inspect landcover.parquet -v  # verbose
```

#### `raquet-io convert raster`

Convert any GDAL-supported raster (GeoTIFF, COG, NetCDF, etc.) to RaQuet format.

```bash
raquet-io convert raster input.tif output.parquet
raquet-io convert raster climate.nc climate.parquet  # NetCDF with time dimension

# With options
raquet-io convert raster input.tif output.parquet \
  --resampling bilinear \
  --block-size 512 \
  --row-group-size 200 \
  -v
```

**Note:** `raquet-io convert geotiff` is still supported as an alias.

| Option | Description |
|--------|-------------|
| `--zoom-strategy` | `auto`, `lower`, `upper` (default: `auto`) |
| `--resampling` | `near`, `bilinear`, `cubic`, etc. |
| `--block-size` | Block size in pixels (default: 256) |
| `--row-group-size` | Rows per Parquet row group (default: 200) |
| `-v, --verbose` | Enable verbose output |

#### `raquet-io convert imageserver`

Convert an ArcGIS ImageServer to RaQuet format.

```bash
raquet-io convert imageserver https://server/.../ImageServer output.parquet \
  --bbox "-122.5,37.5,-122.0,38.0" \
  --resolution 12
```

#### `raquet-io export geotiff`

Export a RaQuet file back to GeoTIFF.

```bash
raquet-io export geotiff input.parquet output.tif
```

#### `raquet-io split-zoom`

Split a RaQuet file by zoom level for optimized remote access.

```bash
raquet-io split-zoom input.parquet output_dir/
```

---

## FAQ

### What's the difference between RaQuet and COG (Cloud Optimized GeoTIFF)?

COG and RaQuet serve different layers of the data stack. COG is ideal for classic raster access patterns: window reads, visualization, GDAL pipelines, and serving tiles to web maps. RaQuet targets a different problem: making raster data **computable and governable** inside data warehouses and lakehouses using Parquet.

Think of it this way: COG is how you store and serve rasters; RaQuet is how you analyze and join them with the rest of your data.

| Feature | RaQuet | COG |
|---------|--------|-----|
| Format | Parquet | GeoTIFF |
| Query Tool | SQL (DuckDB, BigQuery) | GDAL, rasterio |
| Index Type | QUADBIN (discrete tiles) | Internal overviews |
| Data Warehouse | Native support | Requires conversion |
| Best for | Analytics, joins, SQL workflows | Visualization, window reads, GIS pipelines |

RaQuet is ideal when you need SQL-based queries, want to join raster with vector data, or need lakehouse governance (versioning, lineage, access control).

### Can I use RaQuet with BigQuery or Snowflake?

Yes! RaQuet files are standard Parquet files that can be loaded into any Parquet-compatible data warehouse. The QUADBIN indexing works natively with CARTO's Analytics Toolbox.

### How do I query specific tiles?

```sql
-- Query a specific tile by QUADBIN ID
SELECT block, band_1, band_2, band_3
FROM read_parquet('raster.parquet')
WHERE block = 5270201491262341119;

-- Query a range of tiles (efficient with sorted data)
SELECT block, band_1, band_2, band_3
FROM read_parquet('raster.parquet')
WHERE block BETWEEN 5270201491262341119 AND 5270201491263324159;
```

### What raster formats can I convert to RaQuet?

RaQuet supports any GDAL-readable raster format:
- GeoTIFF / Cloud Optimized GeoTIFF (COG)
- NetCDF (with CF time dimension support - adds `time_cf` and `time_ts` columns)
- ArcGIS ImageServer
- And [many more GDAL formats](https://gdal.org/en/stable/drivers/raster/index.html)

### Is there a size limit?

RaQuet can handle rasters of any size. For very large datasets, consider using `raquet-io split-zoom` to create separate files per zoom level for optimal query performance.

---

## Performance Considerations

RaQuet is optimized for efficient remote access. Here are key recommendations:

### Block Sorting (Critical)

RaQuet files **must have blocks sorted by QUADBIN ID** for optimal performance. This enables Parquet row group pruning, reducing data transfer by 90%+ for typical queries.

```bash
# The CLI automatically sorts blocks during conversion
raquet-io convert raster input.tif output.parquet
```

### Row Group Size

Row group size affects HTTP request efficiency when using client-side viewers. Our testing with hyparquet shows:

| Row Group Size | HTTP Requests/Tile | Reduction vs RG=1 | Best For |
|----------------|-------------------|-------------------|----------|
| 1 | ~11.3 | baseline | Maximum precision (rarely needed) |
| 4 | ~7.4 | 35% fewer | Good balance |
| 8-16 | ~5.1-5.3 | 54-55% fewer | **Recommended for web viewers** |
| 200+ | N/A | N/A | Server-side queries, full scans |

**Recommendation:** Use `--row-group-size 8` for files optimized for client-side web viewing:

```bash
raquet-io convert raster input.tif output.parquet --row-group-size 8
```

For server-side or SQL queries (DuckDB, BigQuery), larger row groups (200+) are more efficient.

### Query Patterns

**Fast (contiguous read):**
```sql
-- Range query - uses row group pruning effectively
SELECT * FROM read_parquet('file.parquet')
WHERE block BETWEEN 5270201491262341119 AND 5270201491263324159
```

**Slower (scattered reads):**
```sql
-- IN clause - may require multiple row group reads
SELECT * FROM read_parquet('file.parquet')
WHERE block IN (5270201491262341119, 5280000000000000000, ...)
```

### Client-Side vs Server-Side

| Environment | Performance | Best For |
|-------------|-------------|----------|
| DuckDB (server/native) | Fast (~200ms for 20 tiles) | Production APIs |
| hyparquet (browser) | Good (HTTP range requests) | Interactive demos, no backend needed |

The browser viewer uses hyparquet with HTTP range requests to fetch only the data needed for visible tiles.

### Zoom Level Splitting

For very large datasets, split by zoom level:

```bash
raquet-io split-zoom large_raster.parquet output_dir/
# Creates: zoom_11.parquet, zoom_12.parquet, etc.
```

This allows queries to target specific zoom levels without scanning the entire file.

---

## About

RaQuet is an open source project created and maintained by [**CARTO**](https://carto.com), the leading Location Intelligence platform. CARTO helps organizations unlock the power of spatial data through cloud-native analytics.

Learn more about CARTO's spatial data solutions at [carto.com](https://carto.com).

## License

RaQuet is open source under the [BSD-3-Clause License](https://github.com/CartoDB/raquet/blob/master/LICENSE).

## Contributing

Contributions are welcome! See the [GitHub repository](https://github.com/CartoDB/raquet) for issues and pull requests.
