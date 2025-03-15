#!/usr/bin/env python3
"""Convert GeoTIFF file to RaQuet output

Usage:
    geotiff2raquet.py <geotiff_filename> <raquet_filename>

Required packages:
    - GDAL <https://pypi.org/project/GDAL/>
    - mercantile <https://pypi.org/project/mercantile/>
    - pyarrow <https://pypi.org/project/pyarrow/>
    - quadbin <https://pypi.org/project/quadbin/>

Test case "europe.tif"

    >>> import tempfile; _, raquet_tempfile = tempfile.mkstemp(suffix=".parquet")
    >>> main("examples/europe.tif", raquet_tempfile)
    >>> table1 = pyarrow.parquet.read_table(raquet_tempfile)
    >>> len(table1)
    17

    >>> table1.column_names
    ['block', 'metadata', 'band_1', 'band_2', 'band_3', 'band_4']

    >>> metadata1 = read_metadata(table1)
    >>> [metadata1[k] for k in ["compression", "width", "height", "num_blocks", "num_pixels", "nodata"]]
    ['gzip', 1024, 1024, 16, 1048576, None]

    >>> [metadata1[k] for k in ["block_resolution", "pixel_resolution", "minresolution", "maxresolution"]]
    [5, 13, 5, 5]

    >>> [round(b, 8) for b in metadata1["bounds"]]
    [0.0, 40.97989807, 45.0, 66.51326044]

    >>> [round(b, 8) for b in metadata1["center"]]
    [22.5, 53.74657926, 5]

    >>> {b["name"]: b["type"] for b in metadata1["bands"]}
    {'band_1': 'uint8', 'band_2': 'uint8', 'band_3': 'uint8', 'band_4': 'uint8'}

    >>> {k: round(v, 8) for k, v in sorted(metadata1["bands"][0]["stats"].items())}
    {'count': 1048576, 'max': 255, 'mean': 166.05272293, 'min': 13, 'stddev': 59.86040623, 'sum': 174118900, 'sum_squares': 34651971296}

    >>> {k: round(v, 8) for k, v in sorted(metadata1["bands"][1]["stats"].items())}
    {'count': 1048576, 'max': 255, 'mean': 152.49030876, 'min': 15, 'stddev': 73.9501475, 'sum': 159897678, 'sum_squares': 32764948700}

    >>> {k: round(v, 8) for k, v in sorted(metadata1["bands"][2]["stats"].items())}
    {'count': 1048576, 'max': 255, 'mean': 185.30587387, 'min': 15, 'stddev': 50.48477702, 'sum': 194307292, 'sum_squares': 39814764632}

    >>> {k: round(v, 8) for k, v in sorted(metadata1["bands"][3]["stats"].items())}
    {'count': 1048576, 'max': 255, 'mean': 189.74769783, 'min': 0, 'stddev': 83.36095331, 'sum': 198964882, 'sum_squares': 50531863662}

Test case "n37_w123_1arc_v2-cog.tif"

    >>> main("tests/n37_w123_1arc_v2-cog.tif", raquet_tempfile)
    >>> table2 = pyarrow.parquet.read_table(raquet_tempfile)
    >>> len(table2)
    5

    >>> table2.column_names
    ['block', 'metadata', 'band_1']

    >>> metadata2 = read_metadata(table2)
    >>> [metadata2[k] for k in ["compression", "width", "height", "num_blocks", "num_pixels", "nodata"]]
    ['gzip', 512, 512, 4, 262144, -32767.0]

    >>> [metadata2[k] for k in ["block_resolution", "pixel_resolution", "minresolution", "maxresolution"]]
    [11, 19, 11, 11]

    >>> [round(b, 8) for b in metadata2["bounds"]]
    [-122.6953125, 37.57941251, -122.34375, 37.85750716]

    >>> [round(b, 8) for b in metadata2["center"]]
    [-122.51953125, 37.71845983, 11]

    >>> {b["name"]: b["type"] for b in metadata2["bands"]}
    {'band_1': 'int16'}

    >>> {k: round(v, 8) for k, v in sorted(metadata2["bands"][0]["stats"].items())}
    {'count': 96292, 'max': 376, 'mean': 38.37027998, 'min': -8, 'stddev': 54.0568915, 'sum': 3694751, 'sum_squares': 452987447}

Test case "Annual_NLCD_LndCov_2023_CU_C1V0-cog.tif"

    >>> main("tests/Annual_NLCD_LndCov_2023_CU_C1V0-cog.tif", raquet_tempfile)
    >>> table3 = pyarrow.parquet.read_table(raquet_tempfile)
    >>> len(table3)
    43

    >>> table3.column_names
    ['block', 'metadata', 'band_1']

    >>> metadata3 = read_metadata(table3)
    >>> [metadata3[k] for k in ["compression", "width", "height", "num_blocks", "num_pixels", "nodata"]]
    ['gzip', 1536, 1792, 42, 2752512, 250.0]

    >>> [metadata3[k] for k in ["block_resolution", "pixel_resolution", "minresolution", "maxresolution"]]
    [13, 21, 13, 13]

    >>> {k: round(v, 8) for k, v in sorted(metadata3["bands"][0]["stats"].items())}
    {'count': 1216387, 'max': 95, 'mean': 75.84779926, 'min': 11, 'stddev': 14.05341831, 'sum': 92260277, 'sum_squares': 7326781745}

Test case "geotiff-discreteloss_2023-cog.tif"

    >>> main("tests/geotiff-discreteloss_2023-cog.tif", raquet_tempfile)
    >>> table4 = pyarrow.parquet.read_table(raquet_tempfile)
    >>> len(table4)
    26

    >>> table4.column_names
    ['block', 'metadata', 'band_1']

    >>> metadata4 = read_metadata(table4)
    >>> [metadata4[k] for k in ["compression", "width", "height", "num_blocks", "num_pixels", "nodata"]]
    ['gzip', 1280, 1280, 25, 1638400, 0.0]

    >>> [metadata4[k] for k in ["block_resolution", "pixel_resolution", "minresolution", "maxresolution"]]
    [13, 21, 13, 13]

    >>> {k: round(v, 8) for k, v in sorted(metadata4["bands"][0]["stats"].items())}
    {'count': 27325, 'max': 1, 'mean': 1.0, 'min': 1, 'stddev': 0.0, 'sum': 27325, 'sum_squares': 27325}

"""

import argparse
import dataclasses
import gzip
import itertools
import json
import logging
import math
import multiprocessing
import statistics
import struct

import mercantile
import pyarrow.compute
import pyarrow.parquet
import quadbin

# Zoom offset from tiles to pixels, e.g. 8 = 256px tiles
BLOCK_ZOOM = 8

# Decimal precision needed for resolution comparisons
DECM_PRECISION = 11

# List of acceptable ground resolutions for whole-number Web Mercator zooms
# See also https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system
VALID_RESOLUTIONS = [round(mercantile.CE / (2**i), DECM_PRECISION) for i in range(32)]


@dataclasses.dataclass
class RasterGeometry:
    """Convenience wrapper for details of raster geometry and transformation"""

    bandtypes: list[str]
    nodata: int | float | None
    zoom: int
    minlat: float
    minlon: float
    maxlat: float
    maxlon: float


@dataclasses.dataclass
class RasterStats:
    """Convenience wrapper for raster statistics"""

    count: int
    min: int | float
    max: int | float
    mean: int | float
    stddev: int | float
    sum: int | float
    sum_squares: int | float


@dataclasses.dataclass
class BandType:
    """Convenience wrapper for details of band data type"""

    fmt: str
    size: int
    typ: type
    name: str


def generate_tiles(rg: RasterGeometry):
    """Generate tiles for a given zoom level

    Args:
        rg: RasterGeometry instance

    Yields: tile
    """
    for tile in mercantile.tiles(rg.minlon, rg.minlat, rg.maxlon, rg.maxlat, rg.zoom):
        yield tile


def combine_stats(prev_stats: RasterStats | None, curr_stats: RasterStats):
    """Combine two RasterStats into one"""
    if prev_stats is None:
        return curr_stats

    if curr_stats is None:
        return prev_stats

    next_count = prev_stats.count + curr_stats.count
    prev_weight = prev_stats.count / next_count
    curr_weight = curr_stats.count / next_count

    next_stats = RasterStats(
        count=next_count,
        min=min(prev_stats.min, curr_stats.min),
        max=max(prev_stats.max, curr_stats.max),
        mean=prev_stats.mean * prev_weight + curr_stats.mean * curr_weight,
        stddev=prev_stats.stddev * prev_weight + curr_stats.stddev * curr_weight,
        sum=prev_stats.sum + curr_stats.sum,
        sum_squares=prev_stats.sum_squares + curr_stats.sum_squares,
    )

    logging.info("%s + %s = %s", prev_stats, curr_stats, next_stats)
    return next_stats


def read_statistics(
    values: list[int | float], nodata: int | float | None
) -> RasterStats:
    """Calculate statistics for list of raw band values and optional nodata value"""
    if nodata is not None:
        values = [val for val in values if val != nodata]

    if len(values) == 0:
        return None

    return RasterStats(
        count=len(values),
        min=min(values),
        max=max(values),
        mean=statistics.mean(values),
        stddev=statistics.stdev(values),
        sum=sum(val for val in values),
        sum_squares=sum(val**2 for val in values),
    )


def read_rasterband(
    band: "osgeo.gdal.Band",  # noqa: F821 (Band type safely imported in read_geotiff)
    bbox: tuple[int, int, int, int],
    xpads: tuple[int, int],
    ypads: tuple[int, int],
    band_type: BandType,
) -> tuple[bytes, RasterStats]:
    """Return uncompressed raster bytes padded to full tile size.

    Acts like numpy.pad() without requiring numpy dependency
    """
    data1 = band.ReadRaster(*bbox)
    pixel_values = struct.unpack(band_type.fmt * bbox[2] * bbox[3], data1)
    stats = read_statistics(pixel_values, band.GetNoDataValue())

    # Return early if no padding to be done
    if (xpads, ypads) == ((0, 0), (0, 0)):
        return data1, stats

    # Prepare nodata cell value in expected format
    if (_nodata := band.GetNoDataValue()) is not None:
        nodata = struct.pack(band_type.fmt, band_type.typ(_nodata))
    else:
        nodata = struct.pack(band_type.fmt, band_type.typ(0))

    # Pad start and end of each row if needed
    data2 = [data1]

    if xpads != (0, 0):
        rowsize = bbox[2] * band_type.size
        offsets = range(0, len(data1), rowsize)
        rowprefix, rowsuffix = nodata * xpads[0], nodata * xpads[1]
        data2 = [rowprefix + data1[off : off + rowsize] + rowsuffix for off in offsets]

    # Pad start and end of full table if needed
    data3 = data2

    if ypads != (0, 0):
        emptyrow = nodata * (xpads[0] + bbox[2] + xpads[1])
        data3 = [emptyrow] * ypads[0] + data2 + [emptyrow] * ypads[1]

    # Concatenate raw bytes and return
    return b"".join(data3), stats


def find_bounds(
    ds: "osgeo.gdal.Dataset", transform: "osgeo.osr.CoordinateTransformation"
) -> tuple[float, float, float, float]:
    """Return outer bounds for raster in a given transformation"""
    xoff, xres, _, yoff, _, yres = ds.GetGeoTransform()
    xdim, ydim = ds.RasterXSize, ds.RasterYSize

    x1, y1, _ = transform.TransformPoint(xoff, yoff)
    x2, y2, _ = transform.TransformPoint(xoff, yoff + ydim * yres)
    x3, y3, _ = transform.TransformPoint(xoff + xdim * xres, yoff)
    x4, y4, _ = transform.TransformPoint(xoff + xdim * xres, yoff + ydim * yres)

    x5, y5 = min(x1, x2, x3, x4), min(y1, y2, y3, y4)
    x6, y6 = max(x1, x2, x3, x4), max(y1, y2, y3, y4)

    return (x5, y5, x6, y6)


def find_resolution(
    ds: "osgeo.gdal.Dataset", transform: "osgeo.osr.CoordinateTransformation"
) -> float:
    """Return units per pixel for raster via a given transformation"""
    xoff, xres, _, yoff, _, yres = ds.GetGeoTransform()
    xdim, ydim = ds.RasterXSize, ds.RasterYSize

    x1, y1, _ = transform.TransformPoint(xoff, yoff)
    x2, y2, _ = transform.TransformPoint(xoff + xdim * xres, yoff + ydim * yres)

    return math.hypot(x2 - x1, y2 - y1) / math.hypot(xdim, ydim)


def read_geotiff(geotiff_filename: str, pipe_in, pipe_out):
    """Worker process that accesses a GeoTIFF through pipes.

    Args:
        geotiff_filename: Name of GeoTIFF file to open
        pipe_in: Connection to receive data from parent
        pipe_out: Connection to send data to parent
    """
    # Import osgeo safely in this worker to avoid https://github.com/apache/arrow/issues/44696
    import osgeo.gdal
    import osgeo.osr

    osgeo.gdal.UseExceptions()

    wgs84 = osgeo.osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    wgs84.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)

    web_mercator = osgeo.osr.SpatialReference()
    web_mercator.ImportFromEPSG(3857)

    gdaltype_bandtypes: dict[int, BandType] = {
        osgeo.gdal.GDT_Byte: BandType("B", 1, int, "uint8"),
        osgeo.gdal.GDT_CFloat32: BandType("f", 4, float, "float32"),
        osgeo.gdal.GDT_CFloat64: BandType("d", 8, float, "float64"),
        osgeo.gdal.GDT_CInt16: BandType("h", 2, int, "int16"),
        osgeo.gdal.GDT_CInt32: BandType("i", 4, int, "int32"),
        osgeo.gdal.GDT_Float32: BandType("f", 4, float, "float32"),
        osgeo.gdal.GDT_Float64: BandType("d", 8, float, "float64"),
        osgeo.gdal.GDT_Int16: BandType("h", 2, int, "int16"),
        osgeo.gdal.GDT_Int32: BandType("i", 4, int, "int32"),
        osgeo.gdal.GDT_Int64: BandType("q", 8, int, "int64"),
        osgeo.gdal.GDT_Int8: BandType("b", 1, int, "int8"),
        osgeo.gdal.GDT_UInt16: BandType("H", 2, int, "uint16"),
        osgeo.gdal.GDT_UInt32: BandType("I", 4, int, "uint32"),
        osgeo.gdal.GDT_UInt64: BandType("Q", 8, int, "uint64"),
    }

    try:
        ds = osgeo.gdal.Open(geotiff_filename)

        tx4326 = osgeo.osr.CoordinateTransformation(ds.GetSpatialRef(), wgs84)
        minlon, minlat, maxlon, maxlat = find_bounds(ds, tx4326)

        tx3857 = osgeo.osr.CoordinateTransformation(ds.GetSpatialRef(), web_mercator)
        resolution = find_resolution(ds, tx3857)
        zoom = round(math.log(mercantile.CE / 256 / resolution) / math.log(2))

        raster_geometry = RasterGeometry(
            [
                gdaltype_bandtypes[ds.GetRasterBand(band_num).DataType].name
                for band_num in range(1, 1 + ds.RasterCount)
            ],
            ds.GetRasterBand(1).GetNoDataValue(),
            zoom,
            minlat,
            minlon,
            maxlat,
            maxlon,
        )

        pipe_out.send(raster_geometry)

        tile_ds, prev_tile = None, None

        while True:
            try:
                received = pipe_in.recv()
                if received is None:
                    # Use this signal value because pipe.close() doesn't raise EOFError here on Linux
                    raise EOFError

                # Expand message to an intended raster area to retrieve
                band_num, tile = received

                # Overwrite tile_ds if needed
                if tile != prev_tile:
                    prev_tile = tile

                    # Initialize warped tile dataset and its bands
                    tile_ds = osgeo.gdal.GetDriverByName("GTiff").Create(
                        "/vsimem/tile.tif",
                        2**BLOCK_ZOOM,
                        2**BLOCK_ZOOM,
                        ds.RasterCount,
                        ds.GetRasterBand(1).DataType,
                    )
                    tile_ds.SetProjection(web_mercator.ExportToWkt())

                    for i in range(1, 1 + ds.RasterCount):
                        if (nodata := ds.GetRasterBand(i).GetNoDataValue()) is not None:
                            tile_ds.GetRasterBand(i).SetNoDataValue(nodata)

                    # Convert mercator coordinates to pixel coordinates
                    xmin, ymin, xmax, ymax = mercantile.xy_bounds(tile)
                    px_width = (xmax - xmin) / tile_ds.RasterXSize
                    px_height = (ymax - ymin) / tile_ds.RasterYSize
                    tile_ds.SetGeoTransform([xmin, px_width, 0, ymax, 0, -px_height])

                    osgeo.gdal.Warp(
                        destNameOrDestDS=tile_ds,
                        srcDSOrSrcDSTab=ds,
                        options=osgeo.gdal.WarpOptions(
                            resampleAlg=osgeo.gdal.GRA_CubicSpline,
                        ),
                    )

                band = tile_ds.GetRasterBand(band_num)

                # Pad to full tile size with NODATA fill if needed
                bbox = (0, 0, tile_ds.RasterXSize, tile_ds.RasterYSize)
                xpads, ypads = (0, 0), (0, 0)
                band_type = gdaltype_bandtypes[band.DataType]
                data, stats = read_rasterband(band, bbox, xpads, ypads, band_type)
                logging.info(
                    "Read %s bytes from band %s: %s...", len(data), band_num, data[:32]
                )

                pipe_out.send((gzip.compress(data), stats))

                # txoff = int(round((xmin - raster_geometry.xoff) / raster_geometry.xres))
                # tyoff = int(
                #     round((raster_geometry.yoff - ymax) / -raster_geometry.yres)
                # )
                # txsize = int(round((xmax - xmin) / raster_geometry.xres))
                # tysize = int(round((ymax - ymin) / -raster_geometry.yres))
                # expected_count = tysize * txsize
                #
                # # Respond with an empty block if requested area is too far east or north
                # if txoff >= ds.RasterXSize or tyoff >= ds.RasterYSize:
                #     pipe_out.send((None, None))
                #     continue
                #
                # # Adjust raster area to within valid values
                # xpad_before, xpad_after, ypad_before, ypad_after = 0, 0, 0, 0
                # if txoff < 0:
                #     txoff, txsize, xpad_before = 0, txsize + txoff, -txoff
                # if tyoff < 0:
                #     tyoff, tysize, ypad_before = 0, tysize + tyoff, -tyoff
                # if txoff + txsize > ds.RasterXSize:
                #     xpad_after = txsize - (ds.RasterXSize - txoff)
                #     txsize = ds.RasterXSize - txoff
                # if tyoff + tysize > ds.RasterYSize:
                #     ypad_after = tysize - (ds.RasterYSize - tyoff)
                #     tysize = ds.RasterYSize - tyoff
                #
                # # Respond with an empty block if requested area is zero width or height
                # if txsize == 0 or tysize == 0:
                #     pipe_out.send((None, None))
                #     continue
                #
                # logging.info(
                #     "Band %s nodata value: %s",
                #     band_num,
                #     ds.GetRasterBand(band_num).GetNoDataValue(),
                # )
                #
                # # Read raster data for this tile
                # band = ds.GetRasterBand(band_num)
                # logging.info(
                #     "txoff %s tyoff %s txsize %s tysize %s",
                #     txoff,
                #     tyoff,
                #     txsize,
                #     tysize,
                # )
                #
                # # Pad to full tile size with NODATA fill if needed
                # bbox = (txoff, tyoff, txsize, tysize)
                # xpads, ypads = (xpad_before, xpad_after), (ypad_before, ypad_after)
                # band_type = gdaltype_bandtypes[band.DataType]
                # data, stats = read_rasterband(band, bbox, xpads, ypads, band_type)
                # logging.info(
                #     "Read %s bytes from band %s: %s...", len(data), band_num, data[:32]
                # )
                # assert len(data) == expected_count * band_type.size
                #
                # pipe_out.send((gzip.compress(data), stats))
            except EOFError:
                break
    finally:
        pipe_in.close()
        pipe_out.close()


def open_geotiff_in_process(geotiff_filename: str):
    """Opens a bidirectional connection to a GeoTIFF reader in another process.

    Returns:
        Tuple of (raster_geometry, send_pipe, receive_pipe) for bidirectional communication
    """
    # Create bidirectional pipes
    parent_recv, child_send = multiprocessing.Pipe(duplex=False)
    child_recv, parent_send = multiprocessing.Pipe(duplex=False)

    # Start worker process
    process = multiprocessing.Process(
        target=read_geotiff, args=(geotiff_filename, child_recv, child_send)
    )
    process.start()

    # Close child ends in parent process
    child_send.close()
    child_recv.close()

    # The first message received is expected to be a RasterGeometry
    raster_geometry = parent_recv.recv()
    assert isinstance(raster_geometry, RasterGeometry)

    return raster_geometry, parent_send, parent_recv


def read_metadata(table) -> dict:
    """Get first row where block=0 to extract metadata"""
    block_zero = table.filter(pyarrow.compute.equal(table.column("block"), 0))
    if len(block_zero) == 0:
        raise Exception("No block=0 in table")
    return json.loads(block_zero.column("metadata")[0].as_py())


def get_raquet_dimensions(
    zoom: int, xmin: int, ymin: int, xmax: int, ymax: int
) -> dict[str, int]:
    """Get dictionary of basic dimensions for RaQuet metadata from tile bounds"""
    # First and last tile in rectangular coverage
    upper_left = mercantile.Tile(x=xmin, y=ymin, z=zoom)
    lower_right = mercantile.Tile(x=xmax, y=ymax, z=zoom)

    # Pixel dimensions
    block_width, block_height = 2**BLOCK_ZOOM, 2**BLOCK_ZOOM
    raster_width = (1 + lower_right.x - upper_left.x) * block_width
    raster_height = (1 + lower_right.y - upper_left.y) * block_height

    # Lat/lon corners
    nw, se = mercantile.bounds(upper_left), mercantile.bounds(lower_right)

    return {
        "bounds": [nw.west, se.south, se.east, nw.north],
        "center": [nw.west / 2 + se.east / 2, se.south / 2 + nw.north / 2, zoom],
        "width": raster_width,
        "height": raster_height,
        "block_width": block_width,
        "block_height": block_height,
        "pixel_resolution": zoom + BLOCK_ZOOM,
    }


def main(geotiff_filename, raquet_filename):
    """Read GeoTIFF datasource and write to a RaQuet file

    Args:
        geotiff_filename: GeoTIFF filename
        raquet_filename: RaQuet filename
    """
    raster_geometry, pipe_send, pipe_recv = open_geotiff_in_process(geotiff_filename)

    try:
        band_names = [f"band_{n}" for n in range(1, 1 + len(raster_geometry.bandtypes))]

        # Create table schema based on band count
        schema = pyarrow.schema(
            [
                ("block", pyarrow.uint64()),
                ("metadata", pyarrow.string()),
                *[(bname, pyarrow.binary()) for bname in band_names],
            ]
        )

        # Initialize empty lists to collect rows and stats
        rows, num_blocks, row_group_size = [], 0, 1000
        band_stats = [None for _ in raster_geometry.bandtypes]

        # Initialize the parquet writer
        writer = pyarrow.parquet.ParquetWriter(raquet_filename, schema)

        xmin, ymin, xmax, ymax = math.inf, math.inf, -math.inf, -math.inf

        for tile in generate_tiles(raster_geometry):
            logging.info(
                "Tile z=%s x=%s y=%s quadkey=%s bounds=%s",
                tile.z,
                tile.x,
                tile.y,
                hex(quadbin.tile_to_cell((tile.x, tile.y, tile.z))),
                mercantile.bounds(tile),
            )

            block_data, block_stats = [], []

            for band_num in range(1, 1 + len(raster_geometry.bandtypes)):
                pipe_send.send((band_num, tile))
                block_datum, block_stat = pipe_recv.recv()

                # Append data to list for this band
                block_data.append(block_datum)
                block_stats.append(block_stat)

            if all(block_datum is None for block_datum in block_data):
                continue

            # Append new row
            rows.append(
                {
                    "block": quadbin.tile_to_cell((tile.x, tile.y, tile.z)),
                    "metadata": None,
                    **{bname: block_data[i] for i, bname in enumerate(band_names)},
                }
            )

            # Accumulate band statistics and real bounds based on included tiles
            band_stats = [combine_stats(p, c) for p, c in zip(band_stats, block_stats)]
            xmin, ymin = min(xmin, tile.x), min(ymin, tile.y)
            xmax, ymax = max(xmax, tile.x), max(ymax, tile.y)

            # Write a row group when we hit the size limit
            if len(rows) >= row_group_size:
                rows_dict = {k: [row[k] for row in rows] for k in schema.names}
                rows, num_blocks = [], num_blocks + len(rows)
                writer.write_table(
                    pyarrow.Table.from_pydict(rows_dict, schema=schema),
                    row_group_size=row_group_size,
                )

        # Write remaining rows
        rows_dict = {k: [row[k] for row in rows] for k in schema.names}
        rows, num_blocks = [], num_blocks + len(rows)
        writer.write_table(
            pyarrow.Table.from_pydict(rows_dict, schema=schema),
            row_group_size=row_group_size,
        )

        # Define RaQuet metadata
        # See https://github.com/CartoDB/raquet/blob/master/format-specs/raquet.md#metadata-specification
        metadata_json = {
            "version": "0.1.0",
            "compression": "gzip",
            "block_resolution": raster_geometry.zoom,
            "minresolution": raster_geometry.zoom,
            "maxresolution": raster_geometry.zoom,
            "nodata": raster_geometry.nodata,
            "num_blocks": num_blocks,
            "num_pixels": num_blocks * (2**BLOCK_ZOOM) * (2**BLOCK_ZOOM),
            "bands": [
                {"type": btype, "name": bname, "stats": stats.__dict__}
                for btype, bname, stats in zip(
                    raster_geometry.bandtypes, band_names, band_stats
                )
            ],
            **get_raquet_dimensions(raster_geometry.zoom, xmin, ymin, xmax, ymax),
        }

        # Finish writing with metadata row
        rows_dict = {
            "block": [0],
            "metadata": [json.dumps(metadata_json)],
            **{bname: [None] for bname in band_names},
        }
        writer.write_table(pyarrow.Table.from_pydict(rows_dict, schema=schema))
        writer.close()

    finally:
        # Send a None because pipe.close() doesn't raise EOFError at the other end on Linux
        pipe_send.send(None)

        pipe_send.close()
        pipe_recv.close()


parser = argparse.ArgumentParser()
parser.add_argument("geotiff_filename")
parser.add_argument("raquet_filename")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output"
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    main(args.geotiff_filename, args.raquet_filename)
