# <img alt="RaQuet" src="logo.svg" width="400">

RaQuet is a specification for storing and querying raster data using [Apache Parquet](https://parquet.apache.org/), a column-oriented data file format. Users of data warehouse platforms rely on the simple interoperability of Parquet files to move data and perform queries.

## Overview

Each row in a RaQuet file represents a single rectangular block of data. Location and zoom are given by a [Web Mercator tile z/x/y tile identifier](https://docs.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system) stored in the `block` column as a single 64-bit cell [Quadbin identifier](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-redshift/key-concepts/spatial-indexes#quadbin). Empty tiles can be omitted to reduce file size.

Raster data pixels are stored in row-major order binary packed blobs in per-band columns named `band_1`, `band_2`, etc. Valid pixel values include integers or floating point values. These blobs can be optionally compressed with `gzip` to further reduce file size.

Pixel bands can be decoded via simple binary unpacking in any programming environment and converted to wire image formats like PNG or displayed directly in web visualization libraries like [MapLibre](https://maplibre.org).

Similar to [GeoParquet](https://geoparquet.org), RaQuet metadata is stored as a JSON object with details on coverage area, raster resolution, pixel data format, and other needed information. For compatibility with data warehouses the metadata is stored within a Parquet row at a special “0” row (`block=0x00`).

## Specification

See [format-specs/raquet.md](format-specs/raquet.md) for the specification.

## License

See [LICENSE](LICENSE) for the license.

## Examples

See [examples/example_metadata.json](examples/example_metadata.json) for an example of the metadata.

See [examples/example_data.parquet](examples/example_data.parquet) for an example of the data.
