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

RaQuet is a specification for storing and querying raster data using [Apache Parquet](https://parquet.apache.org/), enabling efficient cloud-native raster workflows. Developed by [CARTO](https://carto.com), RaQuet brings raster data into the modern data stack.

## Why RaQuet?

- **Cloud-Native**: Query raster data directly from cloud storage using DuckDB, BigQuery, or any Parquet-compatible tool
- **Efficient**: QUADBIN spatial indexing enables fast tile lookups with row group pruning
- **Simple**: Standard Parquet format works with existing data warehouse infrastructure
- **Interoperable**: Convert from GeoTIFF, COG, or ArcGIS ImageServer

## Quick Start

```bash
# Install
pip install raquet-io

# Convert a GeoTIFF to RaQuet
raquet-io convert geotiff input.tif output.parquet

# Inspect a RaQuet file
raquet-io inspect output.parquet

# Query with DuckDB
duckdb -c "SELECT * FROM read_parquet('output.parquet') WHERE block != 0 LIMIT 5"
```

## Try It Now

**[Open the RaQuet Viewer](viewer.html)** - A client-side viewer powered by DuckDB-WASM. Load any RaQuet file from a URL and explore it interactively, no server required!

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

- **[RaQuet Viewer](viewer.html)** - Browser-based viewer (DuckDB-WASM)
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

#### `raquet-io convert geotiff`

Convert a GeoTIFF to RaQuet format.

```bash
raquet-io convert geotiff input.tif output.parquet

# With options
raquet-io convert geotiff input.tif output.parquet \
  --resampling bilinear \
  --block-size 512 \
  --row-group-size 200 \
  -v
```

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

Both formats enable efficient cloud access to raster data. Key differences:

| Feature | RaQuet | COG |
|---------|--------|-----|
| Format | Parquet | GeoTIFF |
| Query Tool | SQL (DuckDB, BigQuery) | GDAL, rasterio |
| Index Type | QUADBIN (discrete tiles) | Internal overviews |
| Data Warehouse | Native support | Requires conversion |

RaQuet is ideal when you need SQL-based queries or integration with data warehouse workflows.

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

Currently supported:
- GeoTIFF / Cloud Optimized GeoTIFF (COG)
- ArcGIS ImageServer

### Is there a size limit?

RaQuet can handle rasters of any size. For very large datasets, consider using `raquet-io split-zoom` to create separate files per zoom level for optimal query performance.

---

## Performance Considerations

RaQuet is optimized for efficient remote access. Here are key recommendations:

### Block Sorting (Critical)

RaQuet files **must have blocks sorted by QUADBIN ID** for optimal performance. This enables Parquet row group pruning, reducing data transfer by 90%+ for typical queries.

```bash
# The CLI automatically sorts blocks during conversion
raquet-io convert geotiff input.tif output.parquet
```

### Row Group Size

Smaller row groups (default: 200 rows) enable finer-grained filtering:

```bash
raquet-io convert geotiff input.tif output.parquet --row-group-size 200
```

| Row Group Size | Best For |
|----------------|----------|
| 200 (default) | Remote access, cloud storage |
| 1000+ | Local queries, full scans |

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
| DuckDB-WASM (browser) | Slower (~5s for 20 tiles) | Interactive demos |

The browser viewer uses batched BETWEEN queries to mitigate WASM limitations.

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
