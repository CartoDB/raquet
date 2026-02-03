---
layout: default
title: FAQ
page_header: true
page_description: Frequently asked questions about RaQuet
---

## What's the difference between RaQuet and COG?

COG (Cloud Optimized GeoTIFF) and RaQuet serve different layers of the data stack.

**COG** is ideal for classic raster access patterns: window reads, visualization, GDAL pipelines, and serving tiles to web maps.

**RaQuet** targets a different problem: making raster data **queryable and governable** inside data warehouses and lakehouses using Parquet.

| Feature | RaQuet | COG |
|---------|--------|-----|
| Format | Apache Parquet | GeoTIFF |
| Query Tool | SQL (DuckDB, BigQuery, etc.) | GDAL, rasterio |
| Index Type | QUADBIN (discrete tiles) | Internal overviews |
| Data Warehouse | Native support | Requires conversion |
| Best for | Analytics, joins, SQL workflows | Visualization, GIS pipelines |

---

## What raster formats can I convert to RaQuet?

RaQuet supports any GDAL-readable raster format:

- **GeoTIFF** / Cloud Optimized GeoTIFF (COG)
- **NetCDF** (with CF time dimension support — adds `time_cf` and `time_ts` columns)
- **AAIGrid** (Esri ASCII Grid)
- **ArcGIS ImageServer** (via `raquet-io convert imageserver`)
- And [many more GDAL formats](https://gdal.org/en/stable/drivers/raster/index.html)

---

## Is there a size limit?

RaQuet can handle rasters of any size. For very large datasets, consider using `raquet-io split-zoom` to create separate files per zoom level for optimal query performance.

---

## How does compression compare?

RaQuet typically achieves significant compression compared to ASCII formats (AAIGrid) and often matches or beats NetCDF:

| Dataset | Source Format | Source Size | RaQuet Size | Compression |
|---------|---------------|-------------|-------------|-------------|
| World Elevation | AAIGrid | 3.2 GB | 805 MB | **75% smaller** |
| World Solar PVOUT | AAIGrid | 2.8 GB | 255 MB | **91% smaller** |
| CFSR SST | NetCDF | 854 MB | 75 MB | **91% smaller** |
| TCI (Sentinel-2) | GeoTIFF | 224 MB | 256 MB | Similar (includes pyramids) |

---

## Why Web Mercator?

RaQuet requires data to be in **Web Mercator (EPSG:3857)** projection. This is a deliberate design choice that enables:

1. **QUADBIN spatial indexing** — A Discrete Global Grid System (DGGS) that provides hierarchical, efficient spatial lookups
2. **Row group pruning** — Parquet readers can skip entire row groups based on spatial predicates
3. **Consistent tiling** — Same tile boundaries as web map standards (XYZ/TMS)

The converter automatically reprojects data to Web Mercator during conversion.

---

## Can I query RaQuet without the DuckDB extension?

Yes! RaQuet files are standard Parquet files. You can query them with any Parquet-compatible tool:

```sql
-- Basic Parquet query (no extension needed)
SELECT *
FROM read_parquet('https://storage.googleapis.com/raquet_demo_data/world_elevation.parquet')
WHERE block != 0
LIMIT 10;

-- Get metadata
SELECT metadata
FROM read_parquet('file.parquet')
WHERE block = 0;
```

However, the [DuckDB Raquet Extension](https://github.com/jatorre/duckdb-raquet) provides spatial functions like `ST_RasterValue`, `ST_RasterIntersects`, and `read_raquet()` for full raster analytics.

---

## How do I visualize RaQuet files?

Several options:

1. **[RaQuet Viewer]({{ site.baseurl }}/viewer)** — Browser-based viewer using HTTP range requests (no server needed)
2. **Export to GeoTIFF** — `raquet-io export geotiff input.parquet output.tif`
3. **Load in CARTO** — Upload to CARTO for visualization and analysis

---

## What's the metadata format?

RaQuet v0.4.0 stores metadata as JSON in the `block=0` row:

```json
{
  "file_format": "raquet",
  "version": "0.4.0",
  "width": 32768,
  "height": 14848,
  "crs": "EPSG:3857",
  "bounds": [-180, -60, 180, 85],
  "bounds_crs": "EPSG:4326",
  "tiling": {
    "scheme": "quadbin",
    "block_width": 256,
    "block_height": 256,
    "min_zoom": 0,
    "max_zoom": 7,
    "pixel_zoom": 15,
    "num_blocks": 7424
  },
  "compression": "gzip",
  "bands": [{
    "name": "band_1",
    "type": "float32",
    "nodata": null,
    "STATISTICS_MINIMUM": 0,
    "STATISTICS_MAXIMUM": 8848,
    "STATISTICS_MEAN": 234.5,
    "STATISTICS_STDDEV": 456.7
  }]
}
```

See the [format specification](https://github.com/CartoDB/raquet/blob/master/format-specs/raquet.md) for full details.

---

## Does RaQuet support time series?

Yes! When converting NetCDF files with CF time dimensions, RaQuet adds:

- `time_cf` — CF time value (numeric offset from reference date)
- `time_ts` — Standard timestamp for easy SQL filtering

```sql
SELECT
    YEAR(time_ts) AS year,
    AVG(ST_RasterSummaryStat(block, band_1, 'mean', metadata)) AS avg_value
FROM read_raquet('climate_data.parquet')
GROUP BY YEAR(time_ts);
```

---

## How does RaQuet compare to PostGIS Raster?

RaQuet builds on the concepts pioneered by PostGIS Raster, but with key differences:

| Aspect | RaQuet | PostGIS Raster |
|--------|--------|----------------|
| Storage | Parquet files (anywhere) | PostgreSQL tables |
| Query Engine | Any Parquet-compatible engine | PostgreSQL only |
| Scalability | Cloud-native, distributed | Limited by PostgreSQL |
| Performance | 10-100x faster for analytics | Good for smaller datasets |
| Governance | Lakehouse-native (Iceberg) | PostgreSQL-based |

RaQuet is ideal when you need to query rasters outside PostgreSQL, join with data in warehouses, or process at scale.

---

## Is RaQuet production-ready?

Yes. RaQuet is used in production at CARTO and is supported by the [Analytics Toolbox](https://carto.com/analytics-toolbox) across BigQuery, Snowflake, Databricks, and PostgreSQL.

The format specification is at v0.4.0. Version 0.3.0 is stable for production use; v0.4.0 adds experimental interleaved band layout and lossy compression support.

---

## How do I contribute?

RaQuet is open source under the BSD-3-Clause license. Contributions are welcome!

- [GitHub Repository](https://github.com/CartoDB/raquet)
- [Issue Tracker](https://github.com/CartoDB/raquet/issues)
- [Format Specification](https://github.com/CartoDB/raquet/blob/master/format-specs/raquet.md)
