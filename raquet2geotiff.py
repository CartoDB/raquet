#!/usr/bin/env python3
"""Convert RaQuet file to GeoTIFF output

Usage:
    raquet2geotiff.py <raquet_filename> <geotiff_filename>

Required packages:
    - GDAL <https://pypi.org/project/GDAL/>
    - mercantile <https://pypi.org/project/mercantile/>
    - pyarrow <https://pypi.org/project/pyarrow/>
"""
import argparse
import gzip
import json
import logging

import mercantile
import pyarrow.compute
import pyarrow.parquet


def deinterleave_quadkey(quadkey: int) -> tuple[int, int, int]:
    """Convert a quadkey integer into z/x/y tile coordinates.

    Args:
        quadkey: 64-bit integer quadkey value

    Returns:
        Tuple of (z, x, y) tile coordinates
    """
    # Extract z from bits 52-58
    z = (quadkey >> 52) & 0x7F

    # Extract interleaved x,y bits from 0-51
    bits = (quadkey & ((1 << 52) - 1)) << 12

    # Deinterleave x and y
    x = y = 0
    for i in range(32):
        x |= ((bits >> (2 * i)) & 1) << i
        y |= ((bits >> (2 * i + 1)) & 1) << i

    # Right shift based on zoom level
    x = x >> (32 - z)
    y = y >> (32 - z)

    return z, x, y


def main1(raquet_filename):
    """Read RaQuet metadata and Table from a file

    Args:
        raquet_filename: RaQuet filename

    Returns:
        Tuple of (metadata, table)
    """
    table = pyarrow.parquet.read_table(raquet_filename)

    # Sort by block column
    table = table.sort_by("block")
    metadata = None

    # Get first row where block=0 to extract metadata
    block_zero = table.filter(pyarrow.compute.equal(table.column("block"), 0))
    if len(block_zero) > 0:
        metadata = json.loads(block_zero.column("metadata")[0].as_py())

    return metadata, table


def main2(metadata, table, geotiff_filename):
    """Create a GeoTIFF file from RaQuet metadata and PyArrow Table

    Args:
        metadata: RaQuet metadata
        table: PyArrow Table
        geotiff_filename: output GeoTIFF filename
    """
    # Create projection
    srs = osgeo.osr.SpatialReference()
    srs.ImportFromEPSG(3857)

    # Create transformer from geographic (lat/lon) to web mercator
    source_srs = osgeo.osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)  # WGS84
    source_srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osgeo.osr.CoordinateTransformation(source_srs, srs)

    # Transform coordinates
    xmin, ymin, _ = transform.TransformPoint(minlon, minlat)
    xmax, ymax, _ = transform.TransformPoint(maxlon, maxlat)

    # GDAL data types mapping
    gdal_types = {
        "uint8": osgeo.gdal.GDT_Byte,
        # "int8": osgeo.gdal.GDT_Int8, # supported by GDAL >= 3.7
        "uint16": osgeo.gdal.GDT_UInt16,
        "int16": osgeo.gdal.GDT_Int16,
        "uint32": osgeo.gdal.GDT_UInt32,
        "int32": osgeo.gdal.GDT_Int32,
        "uint64": osgeo.gdal.GDT_UInt64,
        "int64": osgeo.gdal.GDT_Int64,
        "float32": osgeo.gdal.GDT_Float32,
        "float64": osgeo.gdal.GDT_Float64,
    }

    # Create empty GeoTIFF with compression
    driver = osgeo.gdal.GetDriverByName("GTiff")
    output_width, output_height = metadata["width"], metadata["height"]
    raster = driver.Create(
        geotiff_filename,
        output_width,
        output_height,
        1,
        gdal_types[metadata["bands"][0]["type"]],
        options=[
            "COMPRESS=DEFLATE",
            "TILED=YES",
            f"BLOCKXSIZE={metadata['block_width']}",
            f"BLOCKYSIZE={metadata['block_height']}",
        ],
    )

    # Set projection
    raster.SetProjection(srs.ExportToWkt())

    # Set geotransform
    xres = (xmax - xmin) / output_width
    yres = (ymax - ymin) / output_height
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    raster.SetGeoTransform(geotransform)

    # Process table one row at a time
    for i in range(len(table)):
        if i > 0 and i % 1000 == 0:
            logging.info(f"Processed {i} of {len(table)} blocks...")
        block = table.column("block")[i].as_py()
        if block == 0:
            continue
        z, x, y = deinterleave_quadkey(block)
        if z != metadata["maxresolution"]:
            continue

        # Get mercator corner coordinates for this tile
        ulx, _, _, uly = mercantile.xy_bounds(x, y, z)

        # Convert mercator coordinates to pixel coordinates
        xoff = int(round((ulx - geotransform[0]) / geotransform[1]))
        yoff = int(round((uly - geotransform[3]) / geotransform[5]))

        # Get band data for this row and decompress if needed
        block_data = table.column("band_1")[i].as_py()
        if metadata.get("compression") == "gzip":
            block_data = gzip.decompress(block_data)

        # Write to raster
        band = raster.GetRasterBand(1)
        if metadata.get("nodata") is not None:
            band.SetNoDataValue(metadata["nodata"])
        band.WriteRaster(
            xoff, yoff, metadata["block_width"], metadata["block_height"], block_data
        )


parser = argparse.ArgumentParser(description="Convert RaQuet file to GeoTIFF output")
parser.add_argument("raquet_filename")
parser.add_argument("geotiff_filename")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output"
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    metadata, df = main1(args.raquet_filename)
    minlon, minlat, maxlon, maxlat = metadata["bounds"]

    # Import these after main1() to prevent binary conflicts on Mac
    import osgeo.gdal
    import osgeo.osr

    main2(metadata, df, args.geotiff_filename)
