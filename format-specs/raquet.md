# Raquet Specification v0.1.0

## Overview

Raquet is a specification for storing and querying raster data using [Apache Parquet](https://parquet.apache.org/), a column-oriented data file format.

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

A Raquet file must contain:

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

## Tiling Scheme

Raquet uses the QUADBIN tiling scheme for spatial indexing. [QUADBIN](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-bigquery/key-concepts/spatial-indexes#quadbin) is a hierarchical geospatial index based on the Bing Maps Tile System (Quadkey). Designed to be cluster-efficient, it stores in a 64-bit number the information to uniquely identify any of the grid cells that result from uniformly subdividing a map in Mercator projection into four squares at different resolution levels, from 0 to 26 (less than 1m² at the equator). The bit layout is inspired in the H3 design, and provides different modes to store not only cells, but edges, corners or vertices.

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
    "version": "0.1.0",
    "compression": "gzip",
    "block_resolution": 5,
    "minresolution": 2,
    "maxresolution": 5,
    "nodata": 0,
    "bounds": [-180.0, -90.0, 180.0, 90.0],
    "center": [0.0, 0.0, 5],
    "width": 65536,
    "height": 32768,
    "block_width": 256,
    "block_height": 256,
    "num_blocks": 32768,
    "num_pixels": 2147483648,
    "pixel_resolution": 13,
    "bands": [
        {
            "type": "uint8",
            "name": "band_1",
            "stats": {
                "min": 0.0,
                "max": 255.0,
                "mean": 28.66,
                "stddev": 41.57,
                "sum": 2866073.99,
                "sum_squares": 1e15,
                "count": 100000,
                "quantiles": {
                    "3": [0.25, 0.5, 0.75],
                    "4": [0.2, 0.4, 0.6, 0.8]
                },
                "top_values": {
                    "0": 1000,
                    "1": 800,
                    "2": 600
                },
                "approximated_stats": true
            },
            "colorinterp": "red",
            "nodata": "0",
            "colortable": null
        }
    ],
}
```

### Metadata Fields Description

- **Compression Information**
  - `compression`: String indicating the compression method used for band data.
    - Values: "gzip" or null.
    - When null, band data is stored uncompressed.

- **Resolution Information**
  - `block_resolution`: Integer specifying the base resolution level for blocks (QUADBIN tiles). The range is 0-26.
  - `minresolution`: Integer indicating the minimum resolution level in the dataset, including overviews.
  - `maxresolution`: Integer indicating the maximum resolution level (same as block_resolution).
  - `pixel_resolution`: Integer computed as block_resolution + log4(block_size), representing the resolution level for individual pixels.

- **Band Information**
  Each band entry in the `bands` array contains:
  - `type`: String indicating the data type. Valid values: `uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32, float64`.
  - `name`: String identifier for the band, matching the column name in the Parquet file.
  - `stats`: Object containing statistical information:
    - `min`, `max`: Numeric values representing data range.
    - `mean`, `stddev`: Numeric values for distribution statistics.
    - `sum`: Total sum of all values.
    - `sum_squares`: Sum of squares, useful for variance calculations.
    - `count`: Number of valid pixels (excluding NoData).
    - `quantiles`: Optional object mapping quantiles to their values:
      - Keys: String representation of quantile.
      - Values: Array of quantile boundary values in ascending order
      - Example:
        ```json
         {
            "3": [10, 40],
            "4": [10, 20, 60]
         }
        ```
    - `top_values`: Optional object for discrete/categorical data:
      - Keys: String representation of pixel values
      - Values: Integer count of pixels with that value
      - Example:
        ```json
        {
          "0": 1000,   // 1000 pixels have value 0
          "1": 800,    // 800 pixels have value 1
          "2": 600     // 600 pixels have value 2
        }
        ```
    - `version`: String indicating the statistics computation version.
    - `approximated_stats`: Boolean indicating if statistics are approximated.
  - `colorinterp`: Optional string indicating color interpretation. Acepted values are valid [GDAL color interpretation vañues](https://gdal.org/en/stable/user/raster_data_model.html). Examples: `"red"`, `"green"`, `"blue"`, `"alpha"`, `"palette"`.
  - `nodata`: String representation of the band-specific NoData value.
  - `colortable`: Optional object mapping pixel values to RGBA colors. When present:
    - Keys: String representation of pixel values (0-255 for uint8)
    - Values: Arrays of 4 integers [red, green, blue, alpha], each in range 0-255
    - Used with `colorinterp: "palette"` to define color mapping
    - Example:
      ```json
      {
        "0": [0, 0, 0, 255],    // Black, fully opaque
        "1": [255, 0, 0, 255],  // Red, fully opaque
        "255": [0, 0, 0, 0]     // Transparent (typical NoData value)
      }
      ```

- **Spatial Information**
  - `bounds`: Array [west, south, east, north] specifying geographic extent in WGS84 coordinates.
  - `center`: Array [longitude, latitude, resolution] indicating the center point and resolution level.
  - `width`, `height`: Integers specifying full resolution raster dimensions in pixels.
  - `block_width`, `block_height`: Integers specifying tile dimensions in pixels.
  - `num_blocks`: Integer count of non-empty blocks in the dataset.
  - `num_pixels`: Integer total count of pixels in the full resolution raster.

### Examples of Common Use Cases

TODO: fix these incomplete examples

1. **Single-band Byte Raster (e.g., Land Classification)**
```json
{
    "compression": null,
    "block_resolution": 5,
    "bands": [{
        "type": "uint8",
        "name": "band_1",
        "stats": {
            "min": 0.0,
            "max": 10.0,
            "mean": 3.2,
            "stddev": 2.1,
            "top_values": {"0": 1000, "1": 800, "2": 600}
        }
    }]
}
```

2. **RGB Satellite Image**
```json
{
    "compression": "gzip",
    "block_resolution": 8,
    "bands": [
        {
            "type": "uint8",
            "name": "band_1",
            "colorinterp": "red",
            "stats": {"min": 0.0, "max": 255.0}
        },
        {
            "type": "uint8",
            "name": "band_2",
            "colorinterp": "green",
            "stats": {"min": 0.0, "max": 255.0}
        },
        {
            "type": "uint8",
            "name": "band_3",
            "colorinterp": "blue",
            "stats": {"min": 0.0, "max": 255.0}
        }
    ]
}
```

3. **Float32 Scientific Data (e.g., Elevation)**
```json
{
    "compression": "gzip",
    "block_resolution": 6,
    "nodata": null,
    "bands": [{
        "type": "float32",
        "name": "elevation",
        "stats": {
            "min": -413.0,
            "max": 8848.0,
            "mean": 339.2,
            "stddev": 784.5,
            "approximated_stats": false
        }
    }]
}
```

## File Extension

Raquet files MUST use `.parquet` as the file extension. This ensures compatibility with existing Parquet tools and maintains consistency with the underlying file format.

## Media Type

If a [media type](https://en.wikipedia.org/wiki/Media_type) (formerly: MIME type) is used, a Raquet file MUST use [application/vnd.apache.parquet](https://www.iana.org/assignments/media-types/application/vnd.apache.parquet) as the media type.

## COG to Raquet Conversion

While Raquet files can be created from scratch following this specification, the format was designed to efficiently store [Cloud Optimized GeoTIFF (COG)](https://www.cogeo.org/) data in a columnar format. When converting from COG to Raquet, the source file MUST meet these requirements:

1. **Tiling Scheme**: Must have `TILING_SCHEME=GoogleMapsCompatible` in the GeoTIFF tags
2. **Overview Structure**: Overview factors MUST be consecutive powers of 2 (e.g., 2, 4, 8, 16, ...)
3. **Block Size**: All bands MUST have the same block size

These requirements ensure optimal conversion to Raquet's QUADBIN tiling scheme and efficient overview level handling. For other data sources, implementers MUST ensure their data is organized according to the specifications in the following sections.