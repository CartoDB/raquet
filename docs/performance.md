---
layout: default
title: Performance
page_header: true
page_description: Benchmarks, compression ratios, and optimization strategies
---

## Compression Benchmarks

RaQuet achieves significant compression compared to source formats, especially for ASCII-based rasters.

| Dataset | Source Format | Source Size | RaQuet Size | Reduction |
|---------|---------------|-------------|-------------|-----------|
| World Elevation | AAIGrid | 3.2 GB | 805 MB | **75%** |
| World Solar PVOUT | AAIGrid | 2.8 GB | 255 MB | **91%** |
| CFSR SST (time series) | NetCDF | 854 MB | 75 MB | **91%** |
| TCI (Sentinel-2 RGB) | GeoTIFF | 224 MB | 256 MB | Similar* |
| Spain Solar GHI | GeoTIFF | — | 15 MB | — |

*TCI includes full pyramid (zoom 0-12), adding overhead vs. single-resolution source.

### Why the Compression?

1. **Parquet columnar storage** — Efficient encoding for repetitive data patterns
2. **Gzip block compression** — Each tile's pixel data is gzip-compressed
3. **QUADBIN spatial indexing** — Compact tile addressing (single int64 per tile)
4. **No redundant headers** — Single metadata row vs. per-tile overhead

---

## DuckDB vs BigQuery

We benchmarked both RaQuet implementations using [TCI.parquet](https://storage.googleapis.com/raquet_demo_data/TCI.parquet) (Sentinel-2 True Color imagery, 261 MB, 3,225 tiles across zoom levels 7-14).

### Query Performance

| Query Type | DuckDB | BigQuery Native | BigQuery GCS |
|------------|--------|-----------------|--------------|
| Point Query (pixel at coordinate) | 4.0s | 3.2s | 4.7s |
| Single Tile Statistics | 1.3s | 2.4s | 2.5s |
| Region Statistics (545 tiles) | 2.7s | 6.0s | ~7s |
| Resolution Distribution | 0.8s | 2.0s | 2.4s |
| Full Table Aggregation | 0.7s | 2.0s | 2.6s |

*DuckDB tested on Apple M3 Max, querying directly from GCS via HTTPS. BigQuery tested with native tables and GCS external tables.*

### Key Findings

**DuckDB is 2-3x faster** for interactive queries due to native C++ implementation versus BigQuery's JavaScript UDFs. The performance gap is most noticeable on compute-heavy operations like region statistics where DuckDB processes 545 tiles in 2.7s compared to BigQuery's 6s.

**BigQuery GCS external tables** add only 20-30% overhead compared to native tables. This makes them a practical option for exploring RaQuet files without ETL — you can query parquet files directly in GCS and upgrade to native tables later if needed.

**All three approaches query the same parquet files.** A single RaQuet file in cloud storage can be accessed by DuckDB (via HTTPS), BigQuery external tables, or loaded into BigQuery native tables.

### When to Use Each

| Use Case | Recommended |
|----------|-------------|
| Interactive exploration | DuckDB — immediate response, no setup |
| Local file analysis | DuckDB — works offline, no cloud dependency |
| Small to medium datasets (<10GB) | DuckDB — simpler, faster |
| Large-scale batch processing (TB+) | BigQuery — distributed compute |
| Team data sharing | BigQuery — centralized access control |
| Quick cloud exploration | BigQuery GCS External — no data loading |

### Architecture Comparison

| Aspect | DuckDB | BigQuery |
|--------|--------|----------|
| Implementation | C++ Extension | JavaScript UDFs |
| QUADBIN Functions | Native C++ | CARTO Analytics Toolbox |
| Raster Decompression | zlib (C) | pako (JavaScript) |
| Data Access | Local, HTTPS, S3, GCS | Native tables, external tables |
| Cold Start | <1 second | 2-5 seconds |
| Scaling | Single machine | Distributed |

### Multi-Resolution Pyramid Queries

RaQuet files contain tiles at multiple zoom levels (overviews). When querying regions:

- **DuckDB:** `quadbin_intersects(block, geometry)` automatically finds tiles at all resolutions
- **BigQuery:** Use `__RAQUET_REGION_BLOCKS(geom, min_zoom, max_zoom)` to query across pyramid levels. Standard `QUADBIN_POLYFILL_MODE` only returns tiles at a single zoom level.

---

## How Queries Work

1. **Parquet footer read** — Single range request to get file metadata
2. **Row group pruning** — QUADBIN index enables skipping irrelevant row groups
3. **Column projection** — Only requested bands are read
4. **Block decompression** — Single gzip decompress for the matching tile

---

## Spatial Indexing with QUADBIN

RaQuet uses [QUADBIN](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery/key-concepts/spatial-indexes#quadbin) — a Discrete Global Grid System (DGGS) based on the Web Mercator projection.

### How QUADBIN Enables Fast Queries

```
QUADBIN cell ID: 5270498377487261695
                 ↓
Encodes: zoom level + tile X + tile Y in a single int64
```

When you query a point:

1. Point coordinates → QUADBIN cell ID (O(1) computation)
2. Parquet min/max statistics on `block` column → row group pruning
3. Only matching row groups are read from storage

### Row Group Pruning Example

For a 805 MB file with 7,424 tiles across 200 row groups:

- Without spatial filter: Read all 200 row groups
- With `ST_RasterIntersects`: Read 1-3 row groups (~4 MB)

**Result: 99%+ data skipped for point queries**

---

## Pyramid Structure

RaQuet stores multiple zoom levels in a single file, similar to COG overviews.

```
Zoom 0: 1 tile (global overview)
Zoom 1: 4 tiles
Zoom 2: 16 tiles
Zoom 3: 64 tiles
...
Zoom 7: 16,384 tiles (full resolution)
```

### Benefits

- **Automatic level selection** — Query engine picks appropriate zoom for viewport
- **Single file** — No separate overview files to manage
- **Efficient aggregation** — Lower zooms are pre-computed, not calculated on-the-fly

---

## Remote Query Optimization

### Recommended Settings for Remote Files

```bash
# Smaller row groups = better pruning for range requests
raquet-io convert raster input.tif output.parquet --row-group-size 100

# Split by zoom for very large files
raquet-io split-zoom large.parquet ./by_zoom/
```

### Row Group Size Trade-offs

| Row Group Size | File Size | Remote Query Speed | Local Query Speed |
|----------------|-----------|-------------------|-------------------|
| 50 | Larger | Fastest | Slower |
| 200 (default) | Medium | Fast | Fast |
| 1000 | Smaller | Slower | Fastest |

**Rule of thumb:** Use 100-200 for files that will be queried remotely.

---

## Split by Zoom

For very large datasets, splitting by zoom level optimizes remote access:

```bash
raquet-io split-zoom world_elevation.parquet ./split/
```

Output:
```
split/
├── zoom_0.parquet (1 tile, ~1 KB)
├── zoom_1.parquet (4 tiles, ~4 KB)
├── zoom_2.parquet (16 tiles, ~16 KB)
├── zoom_3.parquet (64 tiles, ~64 KB)
├── zoom_4.parquet (256 tiles, ~256 KB)
├── zoom_5.parquet (1,024 tiles, ~1 MB)
├── zoom_6.parquet (4,096 tiles, ~4 MB)
└── zoom_7.parquet (16,384 tiles, ~800 MB)
```

**Benefit:** A web viewer at zoom 5 only downloads the 1 MB file, not the full 805 MB.

---

## Comparison with COG

| Aspect | RaQuet | COG |
|--------|--------|-----|
| **Point query** | ~100ms (SQL) | ~50ms (GDAL range read) |
| **Window read** | Not optimized | Optimized (native) |
| **SQL joins** | Native | Requires export |
| **Zonal stats** | SQL aggregation | GDAL/rasterio |
| **Data warehouse** | Native Parquet | Requires ETL |

**Choose RaQuet when:** SQL access, data warehouse integration, and governance matter more than raw window-read performance.

**Choose COG when:** GIS pipelines, visualization, and GDAL ecosystem are primary use cases.

---

## Memory Efficiency

### DuckDB Streaming

DuckDB processes RaQuet files with streaming execution:

```sql
-- This doesn't load the entire file into memory
SELECT AVG(ST_RasterSummaryStat(block, band_1, 'mean', metadata))
FROM read_raquet('https://storage.googleapis.com/raquet_demo_data/world_elevation.parquet');
```

### Block-Level Processing

Each tile is independently compressed and can be processed without loading neighbors:

- **Block size:** 256×256 pixels (default)
- **Memory per block:** ~260 KB uncompressed (float32), ~50 KB compressed
- **Parallel processing:** Blocks can be processed independently

---

## Best Practices

### For Conversion

1. **Use appropriate resampling** — `bilinear` or `cubic` for continuous data (elevation, temperature), `near` for categorical data (land cover)

2. **Match block size to use case** — 256 (default) works well for most cases; 512 for very dense data

3. **Include statistics** — Always enabled by default; enables query optimization

### For Querying

1. **Always use spatial filters** — `ST_RasterIntersects` or equivalent enables massive data skipping

2. **Project only needed bands** — `SELECT band_1` not `SELECT *` for multi-band files

3. **Use `read_raquet()` in DuckDB** — Automatically propagates metadata for raster functions

### For Remote Access

1. **Use HTTPS URLs** — `https://storage.googleapis.com/...` works directly in DuckDB

2. **Enable CORS** — Required for browser-based access (viewer)

3. **Consider CDN** — CloudFlare or similar for high-traffic files

