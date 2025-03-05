#!/usr/bin/env python3
"""Convert RaQuet file to GeoTIFF output

Usage:
    raquet2geotiff.py <raquet_filename> <geotiff_filename>

Required packages:
    - GDAL <https://pypi.org/project/GDAL/>
    - mercantile <https://pypi.org/project/mercantile/>
    - pyarrow <https://pypi.org/project/pyarrow/>
    - quadbin <https://pypi.org/project/quadbin/>
"""
import argparse
import gzip
import json
import logging
import multiprocessing

import mercantile
import pyarrow.compute
import pyarrow.parquet
import quadbin


def write_geotiff(metadata: dict, geotiff_filename: str, pipe_in, pipe_out):
    """Worker process that writes a GeoTIFF through pipes.

    Args:
        metadata: dictionary of RaQuet metadata
        geotiff_filename: Name of GeoTIFF file to write
        pipe_in: Connection to receive data from parent
        pipe_out: Connection to send data to parent
    """
    # Import osgeo safely in this worker to avoid https://github.com/apache/arrow/issues/44696
    import osgeo.gdal
    import osgeo.osr

    # Create projection
    srs = osgeo.osr.SpatialReference()
    srs.ImportFromEPSG(3857)

    # Create transformer from geographic (lat/lon) to web mercator
    source_srs = osgeo.osr.SpatialReference()
    source_srs.ImportFromEPSG(4326)  # WGS84
    source_srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osgeo.osr.CoordinateTransformation(source_srs, srs)

    # Transform coordinates
    minlon, minlat, maxlon, maxlat = metadata["bounds"]
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

    while True:
        try:
            tile, block_data = pipe_in.recv()

            # Get mercator corner coordinates for this tile
            ulx, _, _, uly = mercantile.xy_bounds(tile)

            # Convert mercator coordinates to pixel coordinates
            xoff = int(round((ulx - geotransform[0]) / geotransform[1]))
            yoff = int(round((uly - geotransform[3]) / geotransform[5]))

            # Write to raster
            band = raster.GetRasterBand(1)
            if metadata.get("nodata") is not None:
                band.SetNoDataValue(metadata["nodata"])
            band.WriteRaster(
                xoff,
                yoff,
                metadata["block_width"],
                metadata["block_height"],
                block_data,
            )
        except EOFError:
            break

    pipe_in.close()
    pipe_out.close()


def open_geotiff_in_process(metadata: dict, geotiff_filename: str):
    """Opens a bidirectional connection to a GeoTIFF reader in another process.

    Returns:
        Tuple of (raster_geometry, send_pipe, receive_pipe) for bidirectional communication
    """
    # Create bidirectional pipes
    parent_recv, child_send = multiprocessing.Pipe(duplex=False)
    child_recv, parent_send = multiprocessing.Pipe(duplex=False)

    # Start worker process
    process = multiprocessing.Process(
        target=write_geotiff, args=(metadata, geotiff_filename, child_recv, child_send)
    )
    process.start()

    # Close child ends in parent process
    child_send.close()
    child_recv.close()

    return parent_send, parent_recv


def main(raquet_filename, geotiff_filename):
    """Read RaQuet file and write to a GeoTIFF datasource

    Args:
        raquet_filename: RaQuet filename
        geotiff_filename: GeoTIFF filename
    """
    table = pyarrow.parquet.read_table(raquet_filename)

    # Sort by block column
    table = table.sort_by("block")
    metadata = None

    # Get first row where block=0 to extract metadata
    block_zero = table.filter(pyarrow.compute.equal(table.column("block"), 0))
    if len(block_zero) > 0:
        metadata = json.loads(block_zero.column("metadata")[0].as_py())

    pipe_send, pipe_recv = open_geotiff_in_process(metadata, geotiff_filename)

    # Process table one row at a time
    for i in range(len(table)):
        if i > 0 and i % 1000 == 0:
            logging.info(f"Processed {i} of {len(table)} blocks...")
        block = table.column("block")[i].as_py()
        if block == 0:
            continue
        x, y, z = quadbin.cell_to_tile(block)
        if z != metadata["maxresolution"]:
            continue

        # Get band data for this row and decompress if needed
        block_data = table.column("band_1")[i].as_py()
        if metadata.get("compression") == "gzip":
            block_data = gzip.decompress(block_data)

        pipe_send.send((mercantile.Tile(x, y, z), block_data))


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
    main(args.raquet_filename, args.geotiff_filename)
