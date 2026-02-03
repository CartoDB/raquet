# RaQuet Specification v0.4.0

## Overview

RaQuet is a specification for storing and querying raster data using [Apache Parquet](https://parquet.apache.org/), a column-oriented data file format.

## Data Organization

The format organizes raster data in the following way:

1. The raster is divided into regular tiles (blocks). The block size MUST be divisible by 16. Common block sizes are 256x256 and 512x512.
2. [QUADBIN](#tiling-scheme) spatial indexing is used to identify and locate tiles. Each tile is assigned a QUADBIN cell ID based on its spatial location. Example: if QUADBIN level is 13 at pixel resolution and the block size is 256x256, the QUADBIN level of the block is 5.
3. Tile data is stored in a columnar structure for efficient access. Each tile is stored as a row in the file. The data within the bands in the tile is stored as separated columns, and pixels for the band is stored as binary encoded arrays, optionally zlib compressed.
4. Overview pyramids are **optional** and can be included for multi-resolution queries (similar to COGs). When present, overview factors MUST be consecutive powers of 2 (e.g., 2, 4, 8, 16, ...). Files without overviews contain only native resolution tiles.
5. Empty tiles (containing only NoData values) may be excluded to save space.
6. Rich metadata, containing statistics and other information, is included for data discovery and analysis.

This organization enables efficient spatial queries, band selection, and resolution-based filtering while maintaining compatibility with standard Parquet tools and workflows.

## File Structure

A RaQuet file must contain:

### Primary Table

Required columns:
- `block`: QUADBIN cell identifier (uint64) that represents the spatial location of the tile.
- One or more band columns (bytes) containing the actual raster data for each band. The name of the column MUST be the name of the band and band names are defined in the metadata. A common way to define band names is to use the following convention: `band_<band_index>`.
- `metadata`: String column containing the raster metadata in JSON format. This column MUST only have one row where `block = 0`. All other rows in this column MUST be NULL.

## Column Specifications

### Required Columns

#### block Column
- Type: int64
- Description: QUADBIN cell identifier that encodes the tile's spatial location and zoom level.
- Special value: `block = 0` is reserved for the metadata row.

#### Band Columns (Sequential Layout)
When `band_layout` is `"sequential"` (default):
- Type: bytes
- Naming convention: Configurable band names, optionally following the convention `band_<band_index>` (e.g., "band_1", "band_2", etc.).
- Content: Raw binary data of the raster tile for a single band.
- Format: Binary-encoded array of pixel values stored in row-major order (pixels are stored row by row from top to bottom, and within each row from left to right). The encoded array is represented as a flat binary array of the pixel values.
- **Endianness**: Little-endian byte order MUST be used for multi-byte data types (int16, int32, float32, etc.).
- Optional compression: Can be gzip compressed (independent of Parquet-level compression).

#### Pixels Column (Interleaved Layout)
When `band_layout` is `"interleaved"`:
- Type: bytes
- Column name: `pixels` (single column replaces individual band columns)
- Content: All bands interleaved at the pixel level (Band Interleaved by Pixel / BIP format).
- Format: For each pixel in row-major order, all band values are stored consecutively: `[R₀,G₀,B₀,R₁,G₁,B₁,...,Rₙ,Gₙ,Bₙ]` where n = width × height - 1.
- **Endianness**: Little-endian byte order for multi-byte data types.
- Compression: Supports gzip (lossless) or JPEG/WebP (lossy). Lossy compression **requires** this layout.
- **Lossy compression format**: When using JPEG or WebP, the `pixels` column contains a complete encoded image (JPEG/WebP binary data), not raw interleaved bytes. Clients decode using standard image decoders.

#### Metadata Column
- Type: string
- Content: JSON string containing raster metadata.
- Special handling: Only populated in the row where `block = 0`, NULL in all other rows.
- Format: See [Metadata Specification](#metadata-specification) for the JSON structure.

### Optional Columns

#### time_cf Column (Time Dimension)
- Type: int64 or float64
- Description: CF (Climate and Forecast) convention numeric time value. This is the authoritative time representation that preserves the exact values from the source data (e.g., NetCDF files).
- Content: Numeric offset from a reference date in specified units (e.g., "minutes since 1980-01-01 00:00").
- When present: The `time` metadata section MUST be included with `cf:units` specifying the time units.
- Example: For "minutes since 1980-01-01 00:00", a value of `44640` represents 1980-02-01 00:00.

#### time_ts Column (Derived Timestamp)
- Type: timestamp[us] (microsecond precision)
- Description: Convenience timestamp derived from `time_cf` for datasets using Gregorian-compatible calendars.
- Nullable: Yes. This column is NULL when the CF calendar cannot be converted to standard timestamps (e.g., 360_day, noleap calendars).
- Purpose: Enables standard timestamp-based queries without requiring CF time unit parsing.
- Derivation: Computed from `time_cf` using the `cf:units` and `cf:calendar` from metadata.

**Time Column Guidelines:**
- If `time_cf` is present, `time_ts` SHOULD also be present (may contain NULLs for non-Gregorian calendars).
- For single-timestep data, time columns MAY be omitted.
- Time values represent the **start** of each time period (e.g., monthly data for January 1980 has time = 1980-01-01T00:00:00).

**Primary Key with Time Dimension:**
When `time_cf` is present, the combination of (`block`, `time_cf`) forms the unique key for each row. Multiple rows may have the same `block` value but different timestamps. Without time columns, `block` alone is unique (excluding the metadata row at `block = 0`).

## Tiling Scheme

RaQuet uses the **QUADBIN** tiling scheme for spatial indexing. QUADBIN is a hierarchical geospatial index that encodes Web Mercator tile coordinates `(x, y, z)` into a single 64-bit integer. This encoding enables efficient spatial queries and Parquet row group pruning.

Key properties:
- **Single-column index**: Location and zoom level in one INT64 value
- **Morton order**: Spatially adjacent tiles have numerically similar IDs
- **Resolution range**: Zoom levels 0-26 (sub-meter precision at the equator)

For a complete explanation of the QUADBIN algorithm, bit layout, and encoding examples, see the **[QUADBIN Spatial Index documentation](https://cartodb.github.io/raquet/quadbin)**.

### Reference Implementations

- Python: [quadbin-py](https://github.com/CartoDB/quadbin-py) — `quadbin.tile_to_cell()`, `quadbin.cell_to_tile()`
- JavaScript: [@carto/quadbin](https://github.com/CartoDB/quadbin-js)
- SQL: [CARTO Analytics Toolbox](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery/sql-reference/quadbin)

### Row Ordering Recommendation

For optimal random-access performance when reading from cloud storage (S3, GCS, Azure Blob), producers SHOULD sort rows by QUADBIN cell ID. This enables Parquet row group pruning when filtering by spatial location, as QUADBIN's Morton-order encoding clusters spatially adjacent tiles together.

### Row Group Size Considerations

The optimal Parquet row group size involves trade-offs that depend on the primary use case:

- **Smaller row groups** (e.g., 50-200 rows): Better for web tiling and random tile access, as clients can fetch individual tiles with minimal data transfer overhead.
- **Larger row groups** (e.g., 1000+ rows): Better for analytics workloads that scan many tiles, as fewer row groups mean less metadata overhead and more efficient columnar reads.

Producers should consider their primary access pattern when choosing row group size. Further research is needed to establish optimal values for different use cases. The reference implementation currently uses 200 rows as a default.

The tiling scheme is defined by the following parameters:
- Each tile is identified by a QUADBIN cell ID (uint64)
- The resolution level is encoded in the QUADBIN cell ID
- Tiles are aligned to the QUADBIN grid at the specified resolution
- Default tile size is 256x256 pixels

## File Creation

The file creation process includes:
1. Reprojecting the raster to the target Coordinate Reference System, Web Mercator (EPSG:3857), at one of the [supported scales](https://learn.microsoft.com/en-us/bingmaps/articles/understanding-scale-and-resolution#calculating-resolution).
2. Dividing the raster into tiles.
3. Computing QUADBIN cell IDs for each tile.
4. Converting tile data (pixel values) to binary format.
5. Optional compression of tile data.
6. Writing metadata and tile data to Parquet format.

## Metadata Specification

The metadata is stored as a JSON string in the `metadata` column where `block = 0`. The JSON object has the following structure:

```json
{
    "file_format": "raquet",
    "version": "0.4.0",
    "width": 9216,
    "height": 7936,
    "crs": "EPSG:3857",
    "bounds": [-19.69, 26.43, 5.63, 44.09],
    "bounds_crs": "EPSG:4326",
    "compression": "gzip",
    "tiling": {
        "scheme": "quadbin",
        "block_width": 256,
        "block_height": 256,
        "min_zoom": 3,
        "max_zoom": 9,
        "pixel_zoom": 17,
        "num_blocks": 1116
    },
    "time": {
        "cf:units": "minutes since 1980-01-01 00:00:00",
        "cf:calendar": "standard",
        "resolution": "P1M",
        "interpretation": "period_start",
        "count": 432,
        "range": [0, 18889920]
    },
    "bands": [
        {
            "name": "band_1",
            "description": "Global Horizontal Irradiation",
            "type": "float32",
            "nodata": null,
            "unit": "kWh/m²/day",
            "scale": null,
            "offset": null,
            "colorinterp": "undefined",
            "colortable": null,
            "STATISTICS_MINIMUM": 0.0,
            "STATISTICS_MAXIMUM": 6.42,
            "STATISTICS_MEAN": 0.67,
            "STATISTICS_STDDEV": 1.63,
            "STATISTICS_VALID_PERCENT": 100.0,
            "histogram": {
                "min": -0.01,
                "max": 6.17,
                "buckets": 256,
                "counts": [55644410, 0, "..."]
            }
        }
    ]
}
```

### Metadata Fields Description

- **Format Identification**
  - `file_format`: String identifying this as a RaQuet file. MUST be `"raquet"`.
  - `version`: String indicating the RaQuet specification version. Current version is "0.4.0".

- **Raster Dimensions**
  - `width`, `height`: Integers specifying full resolution raster dimensions in pixels.

- **Coordinate Reference System**
  - `crs`: String indicating the CRS of the raster data. Always "EPSG:3857" (Web Mercator) for RaQuet.
  - `bounds`: Array [west, south, east, north] specifying geographic extent.
  - `bounds_crs`: String indicating the CRS of the bounds. Always "EPSG:4326" (WGS84) for RaQuet.

- **Band Layout** (optional, defaults to "sequential")
  - `band_layout`: String indicating how band data is organized.
    - `"sequential"` (default): Each band stored in a separate column (`band_1`, `band_2`, etc.). This is the traditional layout, optimal for single-band analysis.
    - `"interleaved"`: All bands stored in a single `pixels` column with pixel-interleaved format (R₀G₀B₀R₁G₁B₁...). This layout is required for lossy compression and may improve performance for RGB visualization.

- **Compression Information**
  - `compression`: String indicating the compression method used for band/pixel data.
    - Values: `"gzip"`, `"jpeg"`, `"webp"`, or `null`.
    - `"gzip"`: Lossless compression, works with any band layout and data type.
    - `"jpeg"`: Lossy compression, **requires** `band_layout: "interleaved"` and `uint8` data type. Best for photographic imagery. Supports 1 band (grayscale) or 3 bands (RGB).
    - `"webp"`: Lossy compression, **requires** `band_layout: "interleaved"` and `uint8` data type. Better compression than JPEG for web imagery. Supports 1-4 bands (grayscale, RGB, RGBA).
    - When `null`, band data is stored uncompressed.
  - `compression_quality` (optional): Integer 1-100 for lossy compression quality. Higher values mean better quality but larger files. Default: 85. Ignored for gzip/null compression.

- **Tiling Information**
  - `tiling`: Object containing tile/block configuration:
    - `scheme`: String identifying the tiling scheme. Always "quadbin" for RaQuet.
    - `block_width`, `block_height`: Integers specifying tile dimensions in pixels.
    - `min_zoom`: Integer indicating the minimum zoom level (most zoomed out overview available).
    - `max_zoom`: Integer indicating the maximum zoom level (native resolution).
    - `pixel_zoom`: Integer indicating the zoom level for individual pixels (max_zoom + log4(block_size)).
    - `num_blocks`: Integer count of non-empty blocks in the dataset.

  **Overview Availability:**
  - Overviews are **optional** in RaQuet files, similar to Cloud Optimized GeoTIFFs (COGs).
  - When `min_zoom == max_zoom`: No overviews exist; only native resolution tiles are available.
  - When `min_zoom < max_zoom`: Overviews exist for zoom levels from `min_zoom` to `max_zoom`.

  **Client Behavior for Out-of-Range Zoom Requests:**

  When a client requests tiles at a zoom level outside `[min_zoom, max_zoom]`:

  | Requested Zoom    | Behavior                          | Rationale                                |
  |-------------------|-----------------------------------|------------------------------------------|
  | `zoom < min_zoom` | Use `min_zoom` (coarsest available) | Graceful fallback for large-area queries |
  | `zoom > max_zoom` | Use `max_zoom` (finest available)   | Return best available resolution         |

  Graceful fallback is preferred over errors for analytics use cases. A continent-scale query benefits from receiving coarse overview data rather than nothing.

  **Function-Specific Behavior:**

  - **Point queries** (e.g., `ST_RasterValue`):
    - Return value from the tile at requested zoom if available
    - If no tile exists at that zoom (sparse pyramid), return NULL
    - Zoom clamping applies only when explicitly requesting a resolution

  - **Region analytics** (e.g., `ST_RegionStats`):
    - Resolution parameter accepts: integer, `'auto'`, `'min'`, or `'max'`
    - Integer values are clamped to `[min_zoom, max_zoom]`
    - `'auto'` selects optimal resolution based on query area size
    - `'min'` explicitly requests coarsest available (useful for quick previews)
    - `'max'` explicitly requests finest available (default)

- **Time Information** (optional, present when `time_cf` column exists)
  - `time`: Object containing CF convention time metadata:
    - `cf:units`: String specifying the CF time units (e.g., "minutes since 1980-01-01 00:00:00"). This follows the [CF Conventions](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#time-coordinate) time specification.
    - `cf:calendar`: String specifying the calendar system. Valid values:
      - `"standard"` or `"gregorian"`: Standard Gregorian calendar (default)
      - `"proleptic_gregorian"`: Gregorian calendar extended to dates before 1582-10-15
      - `"360_day"`: All months have 30 days
      - `"365_day"` or `"noleap"`: No leap years
      - `"366_day"` or `"all_leap"`: Every year is a leap year
      - `"julian"`: Julian calendar
    - `resolution`: Optional string in [ISO 8601 duration format](https://en.wikipedia.org/wiki/ISO_8601#Durations) indicating the time step (e.g., "P1M" for monthly, "P1D" for daily, "PT1H" for hourly).
    - `interpretation`: String indicating how time values should be interpreted. Value: `"period_start"` (time represents the beginning of each period).
    - `count`: Integer number of unique time steps in the dataset.
    - `range`: Array [min, max] of the first and last `time_cf` values.

- **Band Information**
  Each band entry in the `bands` array contains:
  - `name`: String identifier for the band, matching the column name in the Parquet file.
  - `description`: Optional string with human-readable band description (from GDAL GetDescription).
  - `type`: String indicating the data type. Valid values: `uint8, int8, uint16, int16, uint32, int32, uint64, int64, float16, float32, float64`.
  - `nodata`: The band-specific NoData value, or null if not set. For special floating-point values, use string encoding following [Zarr v3 conventions](https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html#fill-value-list):
    - `"NaN"` for Not-a-Number
    - `"Infinity"` for positive infinity
    - `"-Infinity"` for negative infinity
    - Numeric values should be encoded with sufficient precision (JSON number or string)
  - `unit`: Optional string indicating physical units (from GDAL GetUnitType). Examples: "meters", "kWh/m²/day".
  - `scale`: Optional float for converting DN to physical values: `physical = DN * scale + offset` (from GDAL GetScale).
  - `offset`: Optional float for converting DN to physical values (from GDAL GetOffset).
  - `colorinterp`: Optional string indicating color interpretation. Values follow [GDAL color interpretations](https://gdal.org/en/stable/user/raster_data_model.html) (lowercase):
    - Basic: `"undefined"`, `"gray"`, `"palette"`, `"red"`, `"green"`, `"blue"`, `"alpha"`
    - Extended (GDAL 3.5+): `"pan"`, `"coastal"`, `"rededge"`, `"nir"`, `"swir"`, `"mwir"`, `"lwir"`, `"tir"`, `"otherir"`
    - SAR (GDAL 3.5+): `"sar_ka"`, `"sar_k"`, `"sar_ku"`, `"sar_x"`, `"sar_c"`, `"sar_s"`, `"sar_l"`, `"sar_p"`
  - `colortable`: Optional object mapping pixel values to RGBA colors (for palette images):
    ```json
    {
      "0": [0, 0, 0, 255],
      "1": [255, 0, 0, 255],
      "255": [0, 0, 0, 0]
    }
    ```

- **Band Statistics** (GDAL-compatible keys)
  Each band entry also contains statistics using GDAL-compatible key names:
  - `STATISTICS_MINIMUM`: Numeric minimum value.
  - `STATISTICS_MAXIMUM`: Numeric maximum value.
  - `STATISTICS_MEAN`: Numeric mean value.
  - `STATISTICS_STDDEV`: Numeric standard deviation.
  - `STATISTICS_VALID_PERCENT`: Percentage of valid (non-nodata) pixels (0-100).

- **Histogram** (optional)
  - `histogram`: Object containing GDAL-style histogram:
    - `min`: Histogram minimum value.
    - `max`: Histogram maximum value.
    - `buckets`: Number of buckets (typically 256).
    - `counts`: Array of pixel counts per bucket.

### Examples of Common Use Cases

1. **Single-band Scientific Data (e.g., Solar Irradiation)**
```json
{
    "file_format": "raquet",
    "version": "0.4.0",
    "width": 9216,
    "height": 7936,
    "crs": "EPSG:3857",
    "bounds": [-19.69, 26.43, 5.63, 44.09],
    "bounds_crs": "EPSG:4326",
    "compression": "gzip",
    "tiling": {
        "scheme": "quadbin",
        "block_width": 256,
        "block_height": 256,
        "min_zoom": 3,
        "max_zoom": 9,
        "pixel_zoom": 17,
        "num_blocks": 1116
    },
    "bands": [{
        "name": "band_1",
        "description": "Global Horizontal Irradiation",
        "type": "float32",
        "nodata": null,
        "unit": "kWh/m²/day",
        "colorinterp": "undefined",
        "STATISTICS_MINIMUM": 0.0,
        "STATISTICS_MAXIMUM": 6.42,
        "STATISTICS_MEAN": 0.67,
        "STATISTICS_STDDEV": 1.63,
        "STATISTICS_VALID_PERCENT": 100.0,
        "histogram": {
            "min": -0.01,
            "max": 6.17,
            "buckets": 256,
            "counts": [55644410, 0, "..."]
        }
    }]
}
```

2. **RGB Satellite Image**
```json
{
    "file_format": "raquet",
    "version": "0.4.0",
    "width": 1024,
    "height": 1024,
    "crs": "EPSG:3857",
    "bounds": [0.0, 40.98, 45.0, 66.51],
    "bounds_crs": "EPSG:4326",
    "compression": "gzip",
    "tiling": {
        "scheme": "quadbin",
        "block_width": 256,
        "block_height": 256,
        "min_zoom": 4,
        "max_zoom": 6,
        "pixel_zoom": 14,
        "num_blocks": 16
    },
    "bands": [
        {
            "name": "band_1",
            "type": "uint8",
            "colorinterp": "red",
            "STATISTICS_MINIMUM": 0,
            "STATISTICS_MAXIMUM": 255,
            "STATISTICS_MEAN": 127.5,
            "STATISTICS_STDDEV": 45.2,
            "STATISTICS_VALID_PERCENT": 100.0
        },
        {
            "name": "band_2",
            "type": "uint8",
            "colorinterp": "green",
            "STATISTICS_MINIMUM": 0,
            "STATISTICS_MAXIMUM": 255,
            "STATISTICS_MEAN": 135.2,
            "STATISTICS_STDDEV": 42.1,
            "STATISTICS_VALID_PERCENT": 100.0
        },
        {
            "name": "band_3",
            "type": "uint8",
            "colorinterp": "blue",
            "STATISTICS_MINIMUM": 0,
            "STATISTICS_MAXIMUM": 255,
            "STATISTICS_MEAN": 98.7,
            "STATISTICS_STDDEV": 38.9,
            "STATISTICS_VALID_PERCENT": 100.0
        }
    ]
}
```

3. **RGB Satellite Image with Lossy Compression (Interleaved)**
```json
{
    "file_format": "raquet",
    "version": "0.4.0",
    "width": 10980,
    "height": 10980,
    "crs": "EPSG:3857",
    "bounds": [32.99, 16.19, 34.03, 17.18],
    "bounds_crs": "EPSG:4326",
    "band_layout": "interleaved",
    "compression": "webp",
    "compression_quality": 85,
    "tiling": {
        "scheme": "quadbin",
        "block_width": 256,
        "block_height": 256,
        "min_zoom": 6,
        "max_zoom": 12,
        "pixel_zoom": 20,
        "num_blocks": 1840
    },
    "bands": [
        {
            "name": "red",
            "type": "uint8",
            "colorinterp": "red",
            "STATISTICS_MINIMUM": 0,
            "STATISTICS_MAXIMUM": 255
        },
        {
            "name": "green",
            "type": "uint8",
            "colorinterp": "green",
            "STATISTICS_MINIMUM": 0,
            "STATISTICS_MAXIMUM": 255
        },
        {
            "name": "blue",
            "type": "uint8",
            "colorinterp": "blue",
            "STATISTICS_MINIMUM": 0,
            "STATISTICS_MAXIMUM": 255
        }
    ]
}
```

This example shows a Sentinel-2 TCI (True Color Image) stored with interleaved band layout and WebP lossy compression. The `pixels` column contains WebP-encoded tiles that browsers can decode natively. This format typically achieves 5-10x smaller file sizes compared to gzip for photographic imagery.

4. **Elevation Data with Units**
```json
{
    "file_format": "raquet",
    "version": "0.4.0",
    "width": 32768,
    "height": 14848,
    "crs": "EPSG:3857",
    "bounds": [-180.0, -60.24, 180.0, 65.37],
    "bounds_crs": "EPSG:4326",
    "compression": "gzip",
    "tiling": {
        "scheme": "quadbin",
        "block_width": 256,
        "block_height": 256,
        "min_zoom": 0,
        "max_zoom": 7,
        "pixel_zoom": 15,
        "num_blocks": 7424
    },
    "bands": [{
        "name": "band_1",
        "description": "Terrain elevation above sea level",
        "type": "int16",
        "nodata": -32768,
        "unit": "meters",
        "colorinterp": "undefined",
        "STATISTICS_MINIMUM": -413,
        "STATISTICS_MAXIMUM": 8848,
        "STATISTICS_MEAN": 339.2,
        "STATISTICS_STDDEV": 784.5,
        "STATISTICS_VALID_PERCENT": 74.5
    }]
}
```

5. **Time-Series Climate Data (NetCDF with CF conventions)**
```json
{
    "file_format": "raquet",
    "version": "0.4.0",
    "width": 1440,
    "height": 721,
    "crs": "EPSG:3857",
    "bounds": [-180.0, -90.0, 180.0, 90.0],
    "bounds_crs": "EPSG:4326",
    "compression": "gzip",
    "tiling": {
        "scheme": "quadbin",
        "block_width": 256,
        "block_height": 256,
        "min_zoom": 0,
        "max_zoom": 5,
        "pixel_zoom": 13,
        "num_blocks": 3
    },
    "time": {
        "cf:units": "minutes since 1980-01-01 00:00:00",
        "cf:calendar": "standard",
        "resolution": "P1M",
        "interpretation": "period_start",
        "count": 432,
        "range": [0, 18889920]
    },
    "bands": [{
        "name": "sst",
        "description": "Sea Surface Temperature",
        "type": "float64",
        "nodata": -9999,
        "unit": "K",
        "colorinterp": "undefined",
        "STATISTICS_MINIMUM": 271.3,
        "STATISTICS_MAXIMUM": 303.7,
        "STATISTICS_MEAN": 286.9,
        "STATISTICS_STDDEV": 11.4,
        "STATISTICS_VALID_PERCENT": 67.2
    }]
}
```

This example represents 36 years (1980-2015) of monthly sea surface temperature data. Each row in the Parquet file includes:
- `block`: QUADBIN tile ID
- `time_cf`: CF numeric time (minutes since 1980-01-01)
- `time_ts`: Derived timestamp (e.g., 1980-01-01T00:00:00)
- `sst`: Compressed raster tile data

6. **Analytics-Only File (No Overviews)**
```json
{
    "file_format": "raquet",
    "version": "0.4.0",
    "width": 400752,
    "height": 131072,
    "crs": "EPSG:3857",
    "bounds": [-180.0, -27.74, 180.0, 90.0],
    "bounds_crs": "EPSG:4326",
    "compression": "gzip",
    "tiling": {
        "scheme": "quadbin",
        "block_width": 256,
        "block_height": 256,
        "min_zoom": 10,
        "max_zoom": 10,
        "pixel_zoom": 18,
        "num_blocks": 45000
    },
    "bands": [{
        "name": "band_1",
        "description": "Deforestation Carbon",
        "type": "float32",
        "nodata": null,
        "unit": "tC/ha",
        "colorinterp": "undefined",
        "STATISTICS_MINIMUM": 0.0,
        "STATISTICS_MAXIMUM": 450.0,
        "STATISTICS_MEAN": 85.2,
        "STATISTICS_STDDEV": 62.4,
        "STATISTICS_VALID_PERCENT": 23.5
    }]
}
```

This example shows a large global raster converted without overviews (`min_zoom == max_zoom == 10`). This is suitable for analytics workloads that only query at native resolution and don't need visualization at lower zoom levels. Benefits include faster conversion time, smaller file size, and lower memory requirements during conversion.

## File Extension

RaQuet files MUST use `.parquet` as the file extension. This ensures compatibility with existing Parquet tools and maintains consistency with the underlying file format.

## Media Type

If a [media type](https://en.wikipedia.org/wiki/Media_type) (formerly: MIME type) is used, a RaQuet file MUST use [application/vnd.apache.parquet](https://www.iana.org/assignments/media-types/application/vnd.apache.parquet) as the media type.

## Raster to RaQuet Conversion

RaQuet supports conversion from any GDAL-readable raster format, including:

- **Cloud Optimized GeoTIFF (COG)**: Optimal for large imagery with existing overviews
- **GeoTIFF**: Standard georeferenced raster format
- **NetCDF**: Scientific data format with support for CF time dimensions
- **AAIGrid (Esri ASCII Grid)**: Common interchange format for elevation and scientific data
- **Other GDAL formats**: Any format supported by [GDAL raster drivers](https://gdal.org/en/stable/drivers/raster/index.html)

### COG-Specific Optimizations

When converting from COG files that meet these requirements, the converter can use existing overviews directly:

1. **Projection**: Source is in Web Mercator (EPSG:3857)
2. **Overview Structure**: Overview factors are consecutive powers of 2 (e.g., 2, 4, 8, 16, ...)
3. **Block Size**: All bands have the same block size

### NetCDF Time Dimension Support

When converting NetCDF files with CF (Climate and Forecast) convention time dimensions:

1. **Time columns are added**: `time_cf` (authoritative CF value) and `time_ts` (derived timestamp)
2. **One row per time step**: Each spatial tile × time combination becomes a separate row
3. **CF metadata preserved**: Time units, calendar, and reference date stored in metadata

For other data sources, the converter reprojects to Web Mercator and builds pyramids as needed.

## File Identification

To enable fast identification of RaQuet files without fully parsing the metadata row, producers SHOULD include a hint in the Parquet file-level key-value metadata:

- **Key**: `raquet:version`
- **Value**: The specification version (e.g., `"0.4.0"`)

This allows readers (e.g., a potential GDAL driver) to quickly distinguish RaQuet files from other Parquet files by reading only the Parquet footer.

**Fallback heuristic** for files without the key-value hint: A Parquet file is likely RaQuet if it contains:
- A `block` column of type INT64
- A `metadata` column of type STRING/UTF8
- One or more columns with BYTE_ARRAY type (band data)

## Processing Metadata (Optional)

Producers MAY include a `processing` object in the metadata to document how the file was created:

```json
{
    "processing": {
        "source_crs": "EPSG:4326",
        "resampling": "bilinear",
        "overview_resampling": "average",
        "created_by": "raquet-io 0.7.0",
        "created_at": "2024-01-15T10:30:00Z"
    }
}
```

Fields:
- `source_crs`: Original CRS before reprojection to EPSG:3857
- `resampling`: Algorithm used for reprojection (e.g., `"near"`, `"bilinear"`, `"cubic"`, `"average"`)
- `overview_resampling`: Algorithm used for overview generation
- `created_by`: Tool and version that created the file
- `created_at`: ISO 8601 timestamp of file creation

## Custom Metadata Extension

Producers MAY extend the metadata with custom fields. To avoid conflicts with future specification versions:

1. Custom fields SHOULD be placed under a `custom` object:
   ```json
   {
       "file_format": "raquet",
       "version": "0.4.0",
       "custom": {
           "organization": "ACME Corp",
           "project_id": "climate-2024",
           "license": "CC-BY-4.0"
       }
   }
   ```

2. Alternatively, custom fields at the root level SHOULD use a namespace prefix (e.g., `"acme:project_id"`).

Readers MUST ignore unrecognized fields to ensure forward compatibility.

## Design Rationale

This section documents key design decisions and their rationale.

### Why Web Mercator (EPSG:3857)?

RaQuet enforces Web Mercator projection for several reasons:

1. **Universal tiling compatibility**: Web Mercator is the de facto standard for web mapping tiles (Google Maps, OpenStreetMap, Bing Maps). This enables direct visualization without reprojection.
2. **QUADBIN efficiency**: The QUADBIN spatial index is designed around Web Mercator's power-of-2 tile subdivision.
3. **Interoperability**: Data can be served directly to web mapping libraries (MapLibre, Leaflet, etc.) without server-side reprojection.

**Trade-offs**: Web Mercator introduces area distortion, particularly at high latitudes, and is not suitable for precise geodetic measurements. For analytics requiring original projection fidelity, consider keeping source data alongside RaQuet exports, or use formats that preserve native projections.

### Why Metadata in a Row vs Parquet File Metadata?

RaQuet stores metadata as a JSON string in a special row (`block = 0`) rather than in Parquet's native key-value metadata:

1. **SQL accessibility**: Metadata can be queried with standard SQL without special Parquet metadata APIs.
2. **Data warehouse compatibility**: BigQuery, Snowflake, Redshift, and DuckDB can read row data easily; accessing file-level metadata varies by platform.
3. **Schema consistency**: The metadata row follows the same schema as data rows (block + metadata columns).
4. **Streaming writes**: Row-based metadata doesn't require rewriting file footers.

**Trade-off**: File identification requires reading the first row rather than just the footer. See [File Identification](#file-identification) for mitigation.

### Why Band-Level Compression (gzip) on Top of Parquet?

RaQuet allows optional gzip compression of band data independently of Parquet's built-in compression:

1. **Cell-level decompression**: Individual tiles can be decompressed without reading entire row groups.
2. **Network transfer**: Compressed blobs can be transferred and cached efficiently.
3. **Compatibility**: Works with Parquet readers that don't support all compression codecs.

**Recommendation**: For optimal file size, use RaQuet's band compression (`compression: "gzip"`) **or** Parquet compression (e.g., ZSTD), but not both. Using both adds overhead with minimal benefit. When using Parquet compression without band compression, set `compression: null` in metadata.

### Scope: 2D Rasters with Optional Time Dimension

RaQuet is designed for 2D spatial rasters with an optional time dimension (X, Y, T). It is not intended to replace general-purpose multi-dimensional array formats like Zarr, NetCDF, or HDF5 for:

- Arbitrary n-dimensional data (e.g., spectral × time × depth × lat × lon)
- Non-spatial array data
- Complex hierarchical data structures

For 3D+ scientific data, consider Zarr or NetCDF with RaQuet as an export target for spatial visualization layers.