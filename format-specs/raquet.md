# RaQuet Specification v0.3.0

## Overview

RaQuet is a specification for storing and querying raster data using [Apache Parquet](https://parquet.apache.org/), a column-oriented data file format.

## Data Organization

The format organizes raster data in the following way:

1. The raster is divided into regular tiles (blocks). The block size MUST be divisible by 16. Common block sizes are 256x256 and 512x512.
2. [QUADBIN](#tiling-scheme) spatial indexing is used to identify and locate tiles. Each tile is assigned a QUADBIN cell ID based on its spatial location. Example: if QUADBIN level is 13 at pixel resolution and the block size is 256x256, the QUADBIN level of the block is 5.
3. Tile data is stored in a columnar structure for efficient access. Each tile is stored as a row in the file. The data within the bands in the tile is stored as separated columns, and pixels for the band is stored as binary encoded arrays, optionally zlib compressed.
4. Overview pyramids can be optionally preserved for multi-resolution queries. Overview factors MUST be consecutive powers of 2 (e.g., 2, 4, 8, 16, ...).
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

#### Band Columns
- Type: bytes
- Naming convention: Configurable band names, optionally following the convention `band_<band_index>` (e.g., "band_1", "band_2", etc.).
- Content: Raw binary data of the raster tile.
- Format: Binary-encoded array of pixel values stored in row-major order (pixels are stored row by row from top to bottom, and within each row from left to right). The encoded array is represented as a flat binary array of the pixel values.
- Optional compression: Can be zlib compressed.

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

## Tiling Scheme

RaQuet uses the QUADBIN tiling scheme for spatial indexing. [QUADBIN](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery/key-concepts/spatial-indexes#quadbin) is a hierarchical geospatial index based on the Bing Maps Tile System (Quadkey). Designed to be cluster-efficient, it stores in a 64-bit number the information to uniquely identify any of the grid cells that result from uniformly subdividing a map in Mercator projection into four squares at different resolution levels, from 0 to 26 (less than 1m² at the equator). The bit layout is inspired in the H3 design, and provides different modes to store not only cells, but edges, corners or vertices.

The tilling scheme is defined by the following parameters:
- Each tile is identified by a QUADBIN cell ID (uint64).
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
    "version": "0.3.0",
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

- **Version Information**
  - `version`: String indicating the RaQuet specification version. Current version is "0.3.0".

- **Raster Dimensions**
  - `width`, `height`: Integers specifying full resolution raster dimensions in pixels.

- **Coordinate Reference System**
  - `crs`: String indicating the CRS of the raster data. Always "EPSG:3857" (Web Mercator) for RaQuet.
  - `bounds`: Array [west, south, east, north] specifying geographic extent.
  - `bounds_crs`: String indicating the CRS of the bounds. Always "EPSG:4326" (WGS84) for RaQuet.

- **Compression Information**
  - `compression`: String indicating the compression method used for band data.
    - Values: "gzip" or null.
    - When null, band data is stored uncompressed.

- **Tiling Information**
  - `tiling`: Object containing tile/block configuration:
    - `scheme`: String identifying the tiling scheme. Always "quadbin" for RaQuet.
    - `block_width`, `block_height`: Integers specifying tile dimensions in pixels.
    - `min_zoom`: Integer indicating the minimum zoom level (most zoomed out overview).
    - `max_zoom`: Integer indicating the maximum zoom level (native resolution).
    - `pixel_zoom`: Integer indicating the zoom level for individual pixels (max_zoom + log4(block_size)).
    - `num_blocks`: Integer count of non-empty blocks in the dataset.

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
  - `type`: String indicating the data type. Valid values: `uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float64`.
  - `nodata`: The band-specific NoData value, or null if not set.
  - `unit`: Optional string indicating physical units (from GDAL GetUnitType). Examples: "meters", "kWh/m²/day".
  - `scale`: Optional float for converting DN to physical values: `physical = DN * scale + offset` (from GDAL GetScale).
  - `offset`: Optional float for converting DN to physical values (from GDAL GetOffset).
  - `colorinterp`: Optional string indicating color interpretation. Values are [GDAL color interpretations](https://gdal.org/en/stable/user/raster_data_model.html): `"red"`, `"green"`, `"blue"`, `"alpha"`, `"gray"`, `"palette"`, `"undefined"`.
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
    "version": "0.3.0",
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
    "version": "0.3.0",
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

3. **Elevation Data with Units**
```json
{
    "version": "0.3.0",
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

4. **Time-Series Climate Data (NetCDF with CF conventions)**
```json
{
    "version": "0.3.0",
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