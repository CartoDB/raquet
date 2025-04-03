#!/usr/bin/env python3
"""Convert RaQuet file to GeoTIFF output

Usage:
    raquet2geotiff.py <raquet_filename> <geotiff_filename>

Required packages:
    - GDAL <https://pypi.org/project/GDAL/>
    - mercantile <https://pypi.org/project/mercantile/>
    - pyarrow <https://pypi.org/project/pyarrow/>
    - quadbin <https://pypi.org/project/quadbin/>

>>> import tempfile; _, geotiff_tempfile = tempfile.mkstemp(suffix=".tif")

Test case "europe.parquet"

>>> main("examples/europe.parquet", geotiff_tempfile)
>>> geotiff_info = read_geotiff_info(geotiff_tempfile)
>>> geotiff_info["size"]
[1024, 1024]

>>> [round(n, 8) for n in geotiff_info["geoTransform"]]
[0.0, 4891.96981025, 0.0, 10018754.17139462, 0.0, -4891.96981025]

>>> [(b["block"], b["type"]) for b in geotiff_info["bands"]]
[([256, 256], 'Byte'), ([256, 256], 'Byte'), ([256, 256], 'Byte'), ([256, 256], 'Byte')]

Test case "colored.parquet"

>>> main("tests/colored.parquet", geotiff_tempfile)
>>> geotiff_info = read_geotiff_info(geotiff_tempfile)

>>> band = geotiff_info["bands"][0]

>>> band["colorInterpretation"]
'Palette'

>>> [colorEntry for colorEntry in band["colorTable"]["entries"][:6]]
[[0, 0, 0, 0], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255], [255, 165, 0, 255], [255, 0, 0, 255]]


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

GDAL_COLOR_INTERP = {
    "undefined": 0,
    "gray": 1,
    "palette": 2,
    "red": 3,
    "green": 4,
    "blue": 5,
    "alpha": 6,
    "hue": 7,
    "saturation": 8,
    "lightness": 9,
    "cyan": 10,
    "magenta": 11,
    "yellow": 12,
    "black": 13,
    "pan": 14,
    "coastal": 15,
    "rededge": 16,
    "nir": 17,
    "swir": 18,
    "mwir": 19,
    "lwir": 20,
    "tir": 21,
    "otherir": 22,
    "sar_ka": 23,
    "sar_k": 24,
    "sar_ku": 25,
    "sar_x": 26,
    "sar_c": 27,
    "sar_s": 28,
    "sar_l": 29,
    "sar_p": 30,
}


def write_geotiff(metadata: dict, geotiff_filename: str, pipe: multiprocessing.Pipe):
    """Worker process that writes a GeoTIFF through pipes.

    Args:
        metadata: dictionary of RaQuet metadata
        geotiff_filename: Name of GeoTIFF file to write
        pipe: Connection to receive data from parent
    """
    # Import osgeo safely in this worker to avoid https://github.com/apache/arrow/issues/44696
    import osgeo.gdal
    import osgeo.osr

    osgeo.gdal.UseExceptions()

    try:
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

        assert len({b["type"] for b in metadata["bands"]}) == 1, (
            "Expect just one band type"
        )

        # Create empty GeoTIFF with compression
        driver = osgeo.gdal.GetDriverByName("GTiff")
        output_width, output_height = metadata["width"], metadata["height"]
        raster = driver.Create(
            geotiff_filename,
            output_width,
            output_height,
            len(metadata["bands"]),
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
            received = pipe.recv()
            if received is None:
                # Use a signal value to stop expecting further tiles
                break

            tile, *block_data = received

            # Get mercator corner coordinates for this tile
            ulx, _, _, uly = mercantile.xy_bounds(tile)

            # Convert mercator coordinates to pixel coordinates
            xoff = int(round((ulx - geotransform[0]) / geotransform[1]))
            yoff = int(round((uly - geotransform[3]) / geotransform[5]))

            # Write to raster
            for i, block_datum in enumerate(block_data):
                band = raster.GetRasterBand(i + 1)

                if metadata.get("nodata") is not None and band.GetNoDataValue() is None:
                    band.SetNoDataValue(metadata["nodata"])

                if (
                    metadata.get("bands")[i].get("colortable") is not None
                    and band.GetColorTable() is None
                ):
                    color_dict = metadata["bands"][i]["colortable"]
                    colorTable = osgeo.gdal.ColorTable()
                    for index, rgba in color_dict.items():
                        colorTable.SetColorEntry(int(index), tuple(rgba))
                    band.SetColorTable(colorTable)

                if metadata.get("bands")[i].get("colorinterp") is not None:
                    band.SetColorInterpretation(
                        GDAL_COLOR_INTERP[metadata["bands"][i]["colorinterp"]]
                    )

                band.WriteRaster(
                    xoff,
                    yoff,
                    metadata["block_width"],
                    metadata["block_height"],
                    block_datum,
                )

    finally:
        pipe.close()


def open_geotiff_in_process(metadata: dict, geotiff_filename: str):
    """Opens a bidirectional connection to a GeoTIFF reader in another process.

    Returns:
        Tuple of (raster_geometry, send_pipe, receive_pipe) for bidirectional communication
    """
    # Create a communication pipe
    child_recv, parent_send = multiprocessing.Pipe(duplex=False)

    # Start worker process
    process = multiprocessing.Process(
        target=write_geotiff, args=(metadata, geotiff_filename, child_recv)
    )
    process.start()

    # Close child end in parent process
    child_recv.close()

    return parent_send


def read_geotiff(geotiff_filename: str, pipe_out):
    """Worker process that reads a GeoTIFF through pipes.

    Args:
        geotiff_filename: Name of GeoTIFF file to read
        pipe_out: Connection to send data to parent
    """
    # Import osgeo safely in this worker to avoid https://github.com/apache/arrow/issues/44696
    import osgeo.gdal

    osgeo.gdal.UseExceptions()

    try:
        geotiff_info = osgeo.gdal.Info(geotiff_filename, format="json")
    except:  # noqa: E722 (fine with this bare except)
        geotiff_info = None

    pipe_out.send(geotiff_info)


def read_geotiff_info(geotiff_filename: str) -> dict:
    """Retrieve osgeo.gdal.Info() for a GeoTIFF file

    Returns:
        Dictionary response from https://gdal.org/en/stable/api/python/utilities.html#osgeo.gdal.Info
    """
    # Create bidirectional pipes
    parent_recv, child_send = multiprocessing.Pipe(duplex=False)

    # Start worker process
    process = multiprocessing.Process(
        target=read_geotiff, args=(geotiff_filename, child_send)
    )
    process.start()

    try:
        # Close child end in parent process
        gdal_info = parent_recv.recv()
        assert isinstance(gdal_info, dict)
    finally:
        child_send.close()
        parent_recv.close()

    return gdal_info


def read_metadata(table) -> dict:
    """Get first row where block=0 to extract metadata"""
    block_zero = table.filter(pyarrow.compute.equal(table.column("block"), 0))
    if len(block_zero) == 0:
        raise Exception("No block=0 in table")
    return json.loads(block_zero.column("metadata")[0].as_py())


def main(raquet_filename, geotiff_filename):
    """Read RaQuet file and write to a GeoTIFF datasource

    Args:
        raquet_filename: RaQuet filename
        geotiff_filename: GeoTIFF filename
    """
    table = pyarrow.parquet.read_table(raquet_filename)

    # Sort by block column
    table = table.sort_by("block")
    metadata = read_metadata(table)

    pipe = open_geotiff_in_process(metadata, geotiff_filename)

    try:
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
            block_data = [table.column(b["name"])[i].as_py() for b in metadata["bands"]]
            if metadata.get("compression") == "gzip":
                block_data = [gzip.decompress(d) for d in block_data]

            pipe.send((mercantile.Tile(x, y, z), *block_data))
    finally:
        # Send a None to signal end of tiles
        pipe.send(None)
        pipe.close()


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
