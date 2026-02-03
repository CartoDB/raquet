# <img alt="RaQuet" src="logo.svg" width="400">

RaQuet is a specification for storing and querying raster data using [Apache Parquet](https://parquet.apache.org/), a column-oriented data file format. Users of data warehouse platforms rely on the simple interoperability of Parquet files to move data and perform queries.

**[Documentation](https://cartodb.github.io/raquet)** | **[Online Viewer](https://cartodb.github.io/raquet/viewer.html)** | **[Specification](format-specs/raquet.md)**

## Overview

Each row in a RaQuet file represents a single rectangular block of data. Location and zoom are given by a [Web Mercator tile z/x/y tile identifier](https://docs.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system) stored in the `block` column as a single 64-bit cell [Quadbin identifier](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-redshift/key-concepts/spatial-indexes#quadbin). Empty tiles can be omitted to reduce file size.

Raster data pixels are stored in row-major order binary packed blobs. By default, each band is stored in a separate column (`band_1`, `band_2`, etc.) with optional gzip compression. For RGB imagery, an **interleaved layout** stores all bands in a single `pixels` column, enabling **lossy compression** (JPEG/WebP) for 10-15x smaller files.

Pixel bands can be decoded via simple binary unpacking in any programming environment and converted to wire image formats like PNG or displayed directly in web visualization libraries like [MapLibre](https://maplibre.org).

Similar to [GeoParquet](https://geoparquet.org), RaQuet metadata is stored as a JSON object with details on coverage area, raster resolution, pixel data format, and other needed information. For compatibility with data warehouses the metadata is stored within a Parquet row at a special “0” row (`block=0x00`).

## Installation

```bash
# Basic installation (GeoTIFF conversion)
pip install raquet-io

# With rich output for CLI
pip install "raquet-io[rich]"

# With ImageServer support
pip install "raquet-io[imageserver]"

# All features
pip install "raquet-io[all]"
```

**Note:** GDAL must be installed separately. On macOS: `brew install gdal`. On Ubuntu: `apt install gdal-bin libgdal-dev`.

## CLI Usage

The `raquet` CLI provides commands for inspecting, converting, and exporting Raquet files.

### Inspect a Raquet file

```bash
# Display metadata and statistics
raquet-io inspect landcover.parquet

# With verbose output
raquet-io inspect landcover.parquet -v
```

### Convert to Raquet

#### From GeoTIFF

```bash
# Basic conversion
raquet-io convert geotiff input.tif output.parquet

# With custom options
raquet-io convert geotiff input.tif output.parquet \
  --resampling bilinear \
  --block-size 512 \
  -v
```

**Options:**
| Option | Description |
|--------|-------------|
| `--zoom-strategy` | Zoom level strategy: `auto`, `lower`, `upper` (default: `auto`) |
| `--resampling` | Resampling algorithm: `near`, `bilinear`, `cubic`, etc. (default: `near`) |
| `--block-size` | Block size: 256 (default), 512, or 1024. Use 512 for fewer HTTP requests. |
| `--target-size` | Target size for auto zoom calculation |
| `--overviews` | Overview generation: `auto` (full pyramid) or `none` (native resolution only) |
| `--min-zoom` | Minimum zoom level for overviews (overrides auto calculation) |
| `--streaming` | Memory-safe two-pass conversion for large files |
| `--band-layout` | Band storage: `sequential` (default) or `interleaved` |
| `--compression` | Compression: `gzip` (default), `jpeg`, `webp`, or `none` |
| `--compression-quality` | Quality for lossy compression (1-100, default: 85) |
| `-v, --verbose` | Enable verbose output |

**Lossy compression for satellite imagery:**
```bash
# WebP compression (best quality/size ratio for RGB imagery)
raquet-io convert geotiff satellite.tif output.parquet \
  --band-layout interleaved \
  --compression webp \
  --compression-quality 85

# JPEG compression (wider compatibility)
raquet-io convert geotiff satellite.tif output.parquet \
  --band-layout interleaved \
  --compression jpeg

# 512px blocks for fewer HTTP requests (recommended for mobile/high-latency)
raquet-io convert geotiff satellite.tif output.parquet \
  --block-size 512 \
  --band-layout interleaved \
  --compression webp
```

**Large file conversion:**
```bash
# Skip overviews for faster conversion (native resolution only)
raquet-io convert geotiff large.tif output.parquet --overviews none

# Memory-safe streaming mode for very large files
raquet-io convert geotiff huge.tif output.parquet --streaming -v

# Limit overview pyramid to zoom 5 and above
raquet-io convert geotiff input.tif output.parquet --min-zoom 5
```

#### From ArcGIS ImageServer

```bash
# Basic conversion
raquet-io convert imageserver https://server/arcgis/rest/services/dem/ImageServer dem.parquet

# With bounding box filter
raquet-io convert imageserver https://server/.../ImageServer output.parquet \
  --bbox "-122.5,37.5,-122.0,38.0"

# With specific resolution
raquet-io convert imageserver https://server/.../ImageServer output.parquet \
  --resolution 12 \
  -v
```

**Options:**
| Option | Description |
|--------|-------------|
| `--token` | ArcGIS authentication token |
| `--bbox` | Bounding box filter in WGS84: `xmin,ymin,xmax,ymax` |
| `--block-size` | Block size: 256 (default), 512, or 1024. Use 512 for fewer HTTP requests. |
| `--resolution` | Target QUADBIN pixel resolution (auto if not specified) |
| `--no-compression` | Disable gzip compression for block data |
| `-v, --verbose` | Enable verbose output |

### Export from Raquet

#### To GeoTIFF

```bash
# Export to GeoTIFF
raquet-io export geotiff input.parquet output.tif

# Include RaQuet overviews in GeoTIFF (uses pre-computed overview tiles)
raquet-io export geotiff input.parquet output.tif --overviews

# With verbose output
raquet-io export geotiff input.parquet output.tif -v
```

### Legacy Commands

For backwards compatibility, standalone commands are also available:

```bash
geotiff2raquet input.tif output.parquet
raquet2geotiff input.parquet output.tif
```

## Python API

```python
from raquet import geotiff2raquet, raquet2geotiff

# Convert GeoTIFF to Raquet
geotiff2raquet.main(
    "input.tif",
    "output.parquet",
    geotiff2raquet.ZoomStrategy.AUTO,
    geotiff2raquet.ResamplingAlgorithm.NEAR,
    block_zoom=8,  # 256px blocks
    target_size=None,
)

# Convert Raquet to GeoTIFF
raquet2geotiff.main("input.parquet", "output.tif")
```

### ImageServer Conversion

```python
from raquet.imageserver import imageserver_to_raquet

# Convert ImageServer to Raquet
result = imageserver_to_raquet(
    "https://server/arcgis/rest/services/dem/ImageServer",
    "output.parquet",
    bbox=(-122.5, 37.5, -122.0, 38.0),  # Optional WGS84 bounds
    block_size=256,
    target_resolution=12,  # Optional, auto-calculated if not specified
)

print(f"Created {result['num_blocks']} blocks with {result['num_bands']} bands")
```

## Querying with DuckDB

Raquet files can be queried directly with DuckDB:

```sql
-- Load Raquet file
SELECT * FROM read_parquet('raster.parquet') WHERE block != 0 LIMIT 10;

-- Get metadata
SELECT metadata FROM read_parquet('raster.parquet') WHERE block = 0;

-- Query specific tiles using QUADBIN
SELECT block, band_1
FROM read_parquet('raster.parquet')
WHERE block = quadbin_from_tile(x, y, z);
```

## Online Viewer

Try the **[RaQuet Viewer](https://cartodb.github.io/raquet/viewer.html)** - a client-side viewer powered by DuckDB-WASM that runs entirely in your browser. Load any publicly accessible RaQuet file and explore it interactively.

## Performance Tips

For optimal remote query performance:

1. **Block sorting**: Blocks are automatically sorted by QUADBIN ID during conversion, enabling Parquet row group pruning
2. **Row group size**: Use smaller row groups (default: 200) for cloud storage access
3. **Zoom splitting**: For large datasets, use `raquet-io split-zoom` to create per-zoom-level files

```bash
# Convert with optimized settings for remote access
raquet-io convert geotiff input.tif output.parquet --row-group-size 200
```

See the [full documentation](https://cartodb.github.io/raquet/#performance-considerations) for more details.

## Specification

See [format-specs/raquet.md](format-specs/raquet.md) for the full specification.

## Examples

See [examples/example_metadata.json](examples/example_metadata.json) for an example of the metadata.

See [examples/example_data.parquet](examples/example_data.parquet) for an example of the data.

## License

See [LICENSE](LICENSE) for the license.
