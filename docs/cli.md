---
layout: default
title: CLI Reference
page_header: true
page_description: Command-line tools for converting, inspecting, and working with RaQuet files
---

## Installation

```bash
pip install raquet-io
```

Or with uv:

```bash
uv add raquet-io
```

---

## Commands Overview

| Command | Description |
|---------|-------------|
| `convert raster` | Convert any GDAL-readable raster to RaQuet |
| `convert imageserver` | Convert ArcGIS ImageServer to RaQuet |
| `inspect` | Display metadata and statistics |
| `validate` | Validate file structure and data integrity |
| `export geotiff` | Export RaQuet back to GeoTIFF |
| `split-zoom` | Split by zoom level for optimized remote access |

---

## convert raster

Convert any GDAL-readable raster format to RaQuet.

```bash
raquet-io convert raster INPUT_FILE OUTPUT_FILE [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT_FILE` | Path to source raster (GeoTIFF, COG, NetCDF, AAIGrid, etc.) |
| `OUTPUT_FILE` | Path for output `.parquet` file |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--zoom-strategy` | `auto` | Strategy for selecting zoom level: `auto`, `lower`, `upper` |
| `--resampling` | `near` | Resampling algorithm: `near`, `average`, `bilinear`, `cubic`, `cubicspline`, `lanczos`, `mode`, `max`, `min`, `med`, `q1`, `q3` |
| `--block-size` | `256` | Block size in pixels: `256`, `512`, or `1024` (see [Block Size](#block-size)) |
| `--target-size` | — | Target size for auto zoom calculation |
| `--row-group-size` | `200` | Rows per Parquet row group (smaller = better remote pruning) |
| `--band-layout` | `sequential` | Band storage: `sequential` or `interleaved` |
| `--compression` | `gzip` | Compression: `gzip`, `jpeg`, `webp`, or `none` |
| `--compression-quality` | `85` | Quality for lossy compression (1-100) |
| `-v, --verbose` | — | Enable verbose output |

### Examples

```bash
# Basic conversion
raquet-io convert raster elevation.tif elevation.parquet

# NetCDF with time dimension (automatically adds time columns)
raquet-io convert raster climate.nc climate.parquet

# High-quality resampling for continuous data
raquet-io convert raster dem.tif dem.parquet --resampling bilinear

# Larger blocks for dense data
raquet-io convert raster satellite.tif output.parquet --block-size 512

# Verbose output to monitor progress
raquet-io convert raster large.tif output.parquet -v

# Lossy compression for RGB satellite imagery (10-15x smaller files)
raquet-io convert raster satellite.tif output.parquet \
  --band-layout interleaved \
  --compression webp \
  --compression-quality 85
```

### Block Size

The `--block-size` option controls the pixel dimensions of each tile. The default is 256px (the web map standard), but 512px can be beneficial in certain scenarios.

| Block Size | Tiles (same area) | Best For |
|------------|-------------------|----------|
| **256px** (default) | More tiles | Standard web maps, maximum compatibility |
| **512px** | ~75% fewer tiles | High-latency networks, mobile, lossy compression |
| **1024px** | ~94% fewer tiles | Very large imagery, minimal HTTP overhead |

**When to use 512px:**
- Lossy compression (JPEG/WebP) — larger tiles compress more efficiently
- Mobile or high-latency networks — fewer HTTP round-trips
- Dense satellite imagery — reduces tile count significantly

**Example comparison (same source raster):**
```
256px WebP: 17 MB, 3,225 tiles
512px WebP: 16 MB,   877 tiles (73% fewer tiles, similar size)
```

```bash
# Optimized for mobile viewing with lossy compression
raquet-io convert raster satellite.tif output.parquet \
  --block-size 512 \
  --band-layout interleaved \
  --compression webp
```

### Supported Input Formats

Any [GDAL-readable raster format](https://gdal.org/en/stable/drivers/raster/index.html):

- **GeoTIFF** / Cloud Optimized GeoTIFF (COG)
- **NetCDF** (with CF time dimension support)
- **AAIGrid** (Esri ASCII Grid)
- **HDF5**, **GRIB**, **JPEG2000**
- And [many more](https://gdal.org/en/stable/drivers/raster/index.html)

---

## convert imageserver

Convert an ArcGIS ImageServer endpoint to RaQuet.

```bash
raquet-io convert imageserver URL OUTPUT_FILE [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `URL` | ArcGIS ImageServer REST endpoint |
| `OUTPUT_FILE` | Path for output `.parquet` file |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--token` | — | ArcGIS authentication token |
| `--bbox` | — | Bounding box filter: `xmin,ymin,xmax,ymax` (WGS84) |
| `--block-size` | `256` | Block size in pixels: `256`, `512`, or `1024` |
| `--resolution` | auto | Target QUADBIN pixel resolution |
| `--no-compression` | — | Disable gzip compression |
| `-v, --verbose` | — | Enable verbose output |

### Examples

```bash
# Basic conversion
raquet-io convert imageserver https://server/arcgis/rest/services/dem/ImageServer dem.parquet

# With bounding box filter
raquet-io convert imageserver https://server/arcgis/.../ImageServer output.parquet \
  --bbox "-122.5,37.5,-122.0,38.0"

# With authentication
raquet-io convert imageserver https://server/ImageServer output.parquet --token YOUR_TOKEN
```

---

## inspect

Display metadata and statistics for a RaQuet file.

```bash
raquet-io inspect FILE [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Show detailed output including band statistics |

### Examples

```bash
# Basic inspection
raquet-io inspect landcover.parquet

# Detailed output
raquet-io inspect landcover.parquet -v
```

### Sample Output

```
RaQuet File: spain_solar_ghi.parquet
Version: 0.4.0
Size: 15.2 MB

Dimensions: 9216 x 7936 pixels
CRS: EPSG:3857
Bounds (EPSG:4326): [-19.69, 26.43, 5.63, 44.09]

Tiling:
  Scheme: quadbin
  Block size: 256 x 256
  Zoom range: 3 - 9
  Pixel zoom: 17
  Total blocks: 1116

Bands: 1
  band_1: float32, nodata=null
    Min: 0.0, Max: 6.42, Mean: 0.67
```

---

## validate

Validate a RaQuet file for correctness and data integrity.

```bash
raquet-io validate FILE [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Show detailed validation output |
| `--json` | Output results as JSON |

### Validation Checks

- Schema validation (required columns: `block`, `metadata`, band columns)
- Metadata validation (version, structure, required fields)
- Pyramid validation (all zoom levels have data)
- Band statistics validation
- Data integrity checks

### Examples

```bash
# Basic validation
raquet-io validate raster.parquet

# Detailed output
raquet-io validate raster.parquet -v

# JSON output for automation
raquet-io validate raster.parquet --json
```

### Sample Output

```
Validating: spain_solar_ghi.parquet
✓ Schema valid
✓ Metadata valid (v0.4.0)
✓ Pyramid complete (zoom 3-9)
✓ Band statistics valid
✓ Data integrity OK

Validation passed
```

---

## export geotiff

Export a RaQuet file back to GeoTIFF format.

```bash
raquet-io export geotiff INPUT_FILE OUTPUT_FILE [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT_FILE` | Path to source RaQuet file |
| `OUTPUT_FILE` | Path for output GeoTIFF |

### Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |

### Examples

```bash
raquet-io export geotiff landcover.parquet landcover.tif
raquet-io export geotiff raster.parquet output.tif -v
```

---

## split-zoom

Split a RaQuet file by zoom level for optimized remote access.

When serving RaQuet files remotely, splitting by zoom level allows clients to query only the zoom level they need without downloading data for other zooms.

```bash
raquet-io split-zoom INPUT_FILE OUTPUT_DIR [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT_FILE` | Path to source RaQuet file |
| `OUTPUT_DIR` | Directory for output files (`zoom_N.parquet`) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--row-group-size` | `200` | Rows per Parquet row group |
| `-v, --verbose` | — | Enable verbose output |

### Examples

```bash
# Split by zoom level
raquet-io split-zoom raster.parquet ./split_output/

# Custom row group size
raquet-io split-zoom large.parquet ./by_zoom/ --row-group-size 100
```

### Output Structure

```
split_output/
├── zoom_3.parquet
├── zoom_4.parquet
├── zoom_5.parquet
├── zoom_6.parquet
└── zoom_7.parquet
```

Each file contains only the blocks for that zoom level, with the full metadata in the `block=0` row.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GDAL_DATA` | Path to GDAL data files |
| `PROJ_LIB` | Path to PROJ data files |

---

## Exit Codes

| Code | Description |
|------|-------------|
| `0` | Success |
| `1` | General error |
| `2` | Invalid arguments |

