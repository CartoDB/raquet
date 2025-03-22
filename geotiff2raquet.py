#!/usr/bin/env python3
"""Convert GeoTIFF file to RaQuet output

Usage:
    geotiff2raquet.py <geotiff_filename> <raquet_filename>

Help:
    geotiff2raquet.py --help

Required packages:
    - GDAL <https://pypi.org/project/GDAL/>
    - mercantile <https://pypi.org/project/mercantile/>
    - pyarrow <https://pypi.org/project/pyarrow/>
    - quadbin <https://pypi.org/project/quadbin/>

Tests

    >>> import tempfile, itertools
    >>> _, raquet_tempfile = tempfile.mkstemp(suffix=".parquet")

    >>> def print_stats(d):
    ...     print(*[f'{k}={v:.4g}' for k, v in sorted(d.items())])

Test case "europe.tif"

    >>> main("examples/europe.tif", raquet_tempfile, ZoomStrategy.AUTO, ResamplingAlgorithm.CubicSpline)
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

    >>> [round(b, 3) for b in metadata1["bounds"]]
    [0.0, 40.98, 45.0, 66.513]

    >>> [round(b, 3) for b in metadata1["center"]]
    [22.5, 53.747, 5]

    >>> {b["name"]: b["type"] for b in metadata1["bands"]}
    {'band_1': 'uint8', 'band_2': 'uint8', 'band_3': 'uint8', 'band_4': 'uint8'}

    >>> {b["name"]: b["colorinterp"] for b in metadata1["bands"]}
    {'band_1': 'Red', 'band_2': 'Green', 'band_3': 'Blue', 'band_4': 'Alpha'}

    >>> print_stats(metadata1["bands"][0]["stats"])
    count=1.049e+06 max=255 mean=104.7 min=0 stddev=63.24 sum=1.098e+08 sum_squares=1.827e+10

    >>> print_stats(metadata1["bands"][1]["stats"])
    count=1.049e+06 max=255 mean=91.15 min=0 stddev=58.76 sum=9.558e+07 sum_squares=1.642e+10

    >>> print_stats(metadata1["bands"][2]["stats"])
    count=1.049e+06 max=255 mean=124 min=0 stddev=68.08 sum=1.3e+08 sum_squares=2.342e+10

    >>> print_stats(metadata1["bands"][3]["stats"])
    count=1.049e+06 max=255 mean=189.7 min=0 stddev=83.36 sum=1.99e+08 sum_squares=5.053e+10

Test case "n37_w123_1arc_v2.tif"

    >>> main("tests/n37_w123_1arc_v2.tif", raquet_tempfile, ZoomStrategy.LOWER, ResamplingAlgorithm.CubicSpline)
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

    >>> [round(b, 3) for b in metadata2["bounds"]]
    [-122.695, 37.579, -122.344, 37.858]

    >>> [round(b, 3) for b in metadata2["center"]]
    [-122.52, 37.718, 11]

    >>> {b["name"]: b["type"] for b in metadata2["bands"]}
    {'band_1': 'int16'}

    >>> print_stats(metadata2["bands"][0]["stats"])
    count=9.692e+04 max=377 mean=38.22 min=-7 stddev=54.02 sum=3.704e+06 sum_squares=4.539e+08

Test case "Annual_NLCD_LndCov_2023_CU_C1V0.tif"

    >>> main("tests/Annual_NLCD_LndCov_2023_CU_C1V0.tif", raquet_tempfile, ZoomStrategy.UPPER, ResamplingAlgorithm.NearestNeighbour)
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

    >>> print_stats(metadata3["bands"][0]["stats"])
    count=1.216e+06 max=95 mean=75.85 min=11 stddev=16.47 sum=9.225e+07 sum_squares=7.415e+09

Test case "geotiff-discreteloss_2023.tif"

    >>> main("tests/geotiff-discreteloss_2023.tif", raquet_tempfile, ZoomStrategy.UPPER, ResamplingAlgorithm.NearestNeighbour)
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

    >>> print_stats(metadata4["bands"][0]["stats"])
    count=2.736e+04 max=1 mean=1 min=1 stddev=0 sum=2.736e+04 sum_squares=2.736e+04

Test case "colored.tif"

    >>> main("tests/colored.tif", raquet_tempfile, ZoomStrategy.AUTO, ResamplingAlgorithm.NearestNeighbour)
    >>> table5 = pyarrow.parquet.read_table(raquet_tempfile)
    >>> metadata5 = read_metadata(table5)

    >>> {b["name"]: b["colorinterp"] for b in metadata5["bands"]}
    {'band_1': 'Palette'}

    >>> color_dict= metadata5["bands"][0]["colortable"]
    >>> {k:list(v) for k, v in itertools.islice(color_dict.items(),6)}
    {'0': [0, 0, 0, 0], '1': [0, 255, 0, 255], '2': [0, 0, 255, 255], '3': [255, 255, 0, 255], '4': [255, 165, 0, 255], '5': [255, 0, 0, 255]}

"""

import argparse
import dataclasses
import enum
import gzip
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


class ZoomStrategy(enum.StrEnum):
    """Switch for web mercator zoom level selection

    See ZOOM_LEVEL_STRATEGY at https://gdal.org/en/stable/drivers/raster/cog.html#reprojection-related-creation-options
    """

    AUTO = "auto"
    LOWER = "lower"
    UPPER = "upper"


class ResamplingAlgorithm(enum.StrEnum):
    """Resampling method to use

    See -r option at https://gdal.org/en/stable/programs/gdalwarp.html#cmdoption-gdalwarp-r
    """

    NearestNeighbour = "near"
    Average = "average"
    Bilinear = "bilinear"
    Cubic = "cubic"
    CubicSpline = "cubicspline"
    Lanczos = "lanczos"
    Max = "max"
    Med = "med"
    Min = "min"
    Mode = "mode"
    Q1 = "q1"
    Q3 = "q3"
    RMS = "rms"
    Sum = "sum"


@dataclasses.dataclass
class RasterGeometry:
    """Convenience wrapper for details of raster geometry and transformation"""

    bandtypes: list[str]
    bandcolorinterp: list[str]
    bandcolortable: list[dict]
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


def get_colortable_dict(color_table: "osgeo.gdal.ColorTable"):  # noqa: F821 (Color table type safely imported in read_geotiff)
    color_dict = {
        str(i): list(color_table.GetColorEntry(i))
        for i in range(color_table.GetCount())
    }
    return color_dict


def combine_stats(
    prev_stats: RasterStats | None, curr_stats: RasterStats | None
) -> RasterStats | None:
    """Combine two RasterStats into one"""

    if prev_stats is None:
        return curr_stats

    if curr_stats is None:  # if there is any NODATA block after proper block, skip
        return prev_stats

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
    band: "osgeo.gdal.Band",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    band_type: BandType,
) -> tuple[bytes, RasterStats]:
    """Return uncompressed raster bytes and stats"""
    data = band.ReadRaster(0, 0, band.XSize, band.YSize)
    pixel_values = struct.unpack(band_type.fmt * band.XSize * band.YSize, data)
    stats = read_statistics(pixel_values, band.GetNoDataValue())

    return data, stats


def find_bounds(
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    transform: "osgeo.osr.CoordinateTransformation",  # noqa: F821 (osgeo types safely imported in read_geotiff)
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
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    transform: "osgeo.osr.CoordinateTransformation",  # noqa: F821 (osgeo types safely imported in read_geotiff)
) -> float:
    """Return units per pixel for raster via a given transformation"""
    xoff, xres, _, yoff, _, yres = ds.GetGeoTransform()
    xdim, ydim = ds.RasterXSize, ds.RasterYSize

    x1, y1, _ = transform.TransformPoint(xoff, yoff)
    x2, y2, _ = transform.TransformPoint(xoff + xdim * xres, yoff + ydim * yres)

    return math.hypot(x2 - x1, y2 - y1) / math.hypot(xdim, ydim)


def find_zoom(resolution: float, zoom_strategy: ZoomStrategy) -> int:
    """Calculate web mercator zoom from a raw meters/pixel resolution"""
    raw_zoom = math.log(mercantile.CE / 256 / resolution) / math.log(2)
    if zoom_strategy is ZoomStrategy.UPPER:
        zoom = math.ceil(raw_zoom)
    elif zoom_strategy is ZoomStrategy.LOWER:
        zoom = math.floor(raw_zoom)
    else:
        zoom = round(raw_zoom)
    return int(zoom)


def create_tile_ds(
    driver: "osgeo.gdal.Driver",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    web_mercator: "osgeo.osr.SpatialReference",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    tile: mercantile.Tile,
) -> "osgeo.gdal.Dataset":  # noqa: F821 (osgeo types safely imported in read_geotiff)
    # Initialize warped tile dataset and its bands
    tile_ds = driver.Create(
        "/vsimem/tile.tif",
        2**BLOCK_ZOOM,
        2**BLOCK_ZOOM,
        ds.RasterCount,
        ds.GetRasterBand(1).DataType,
    )
    tile_ds.SetProjection(web_mercator.ExportToWkt())

    for band_num in range(1, 1 + ds.RasterCount):
        if (nodata := ds.GetRasterBand(band_num).GetNoDataValue()) is not None:
            tile_ds.GetRasterBand(band_num).SetNoDataValue(nodata)

    # Convert mercator coordinates to pixel coordinates
    xmin, ymin, xmax, ymax = mercantile.xy_bounds(tile)
    px_width = (xmax - xmin) / tile_ds.RasterXSize
    px_height = (ymax - ymin) / tile_ds.RasterYSize
    tile_ds.SetGeoTransform([xmin, px_width, 0, ymax, 0, -px_height])

    return tile_ds


def read_geotiff(
    geotiff_filename: str,
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
    pipe: multiprocessing.Pipe,
):
    """Worker process that accesses a GeoTIFF through pipes.

    Send RasterGeometry via pipe first then follow with (tile, data, stats) tuples.

    Args:
        geotiff_filename: Name of GeoTIFF file to open
        zoom_strategy: Web mercator zoom level selection
        resampling_algorithm: Resampling method to use
        pipe: Connection to send data to parent
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

    resampling_algorithms: dict[str, int] = {
        ResamplingAlgorithm.Average: osgeo.gdal.GRA_Average,
        ResamplingAlgorithm.Bilinear: osgeo.gdal.GRA_Bilinear,
        ResamplingAlgorithm.Cubic: osgeo.gdal.GRA_Cubic,
        ResamplingAlgorithm.CubicSpline: osgeo.gdal.GRA_CubicSpline,
        ResamplingAlgorithm.Lanczos: osgeo.gdal.GRA_Lanczos,
        ResamplingAlgorithm.Max: osgeo.gdal.GRA_Max,
        ResamplingAlgorithm.Med: osgeo.gdal.GRA_Med,
        ResamplingAlgorithm.Min: osgeo.gdal.GRA_Min,
        ResamplingAlgorithm.Mode: osgeo.gdal.GRA_Mode,
        ResamplingAlgorithm.NearestNeighbour: osgeo.gdal.GRA_NearestNeighbour,
        ResamplingAlgorithm.Q1: osgeo.gdal.GRA_Q1,
        ResamplingAlgorithm.Q3: osgeo.gdal.GRA_Q3,
        ResamplingAlgorithm.RMS: osgeo.gdal.GRA_RMS,
        ResamplingAlgorithm.Sum: osgeo.gdal.GRA_Sum,
    }

    try:
        ds = osgeo.gdal.Open(geotiff_filename)
        band_nums = list(range(1, 1 + ds.RasterCount))
        bands = [ds.GetRasterBand(band_num) for band_num in band_nums]

        tx4326 = osgeo.osr.CoordinateTransformation(ds.GetSpatialRef(), wgs84)
        minlon, minlat, maxlon, maxlat = find_bounds(ds, tx4326)

        tx3857 = osgeo.osr.CoordinateTransformation(ds.GetSpatialRef(), web_mercator)
        resolution = find_resolution(ds, tx3857)
        zoom = find_zoom(resolution, zoom_strategy)

        raster_geometry = RasterGeometry(
            [gdaltype_bandtypes[band.DataType].name for band in bands],
            [
                osgeo.gdal.GetColorInterpretationName(band.GetColorInterpretation())
                for band in bands
            ],
            [
                get_colortable_dict(b.GetColorTable()) if b.GetColorTable() else None
                for b in bands
            ],
            ds.GetRasterBand(1).GetNoDataValue(),
            zoom,
            minlat,
            minlon,
            maxlat,
            maxlon,
        )

        pipe.send(raster_geometry)

        for tile in generate_tiles(raster_geometry):
            tile_ds = create_tile_ds(
                osgeo.gdal.GetDriverByName("GTiff"), web_mercator, ds, tile
            )

            osgeo.gdal.Warp(
                destNameOrDestDS=tile_ds,
                srcDSOrSrcDSTab=ds,
                options=osgeo.gdal.WarpOptions(
                    resampleAlg=resampling_algorithms[resampling_algorithm],
                ),
            )

            # Read data and stats from warped bands and send them to parent process
            block_data, block_stats = [], []

            for band_num in band_nums:
                band = tile_ds.GetRasterBand(band_num)
                data, stats = read_rasterband(band, gdaltype_bandtypes[band.DataType])
                logging.info(
                    "Read %s bytes from band %s: %s...", len(data), band_num, data[:32]
                )
                block_data.append(gzip.compress(data))
                block_stats.append(stats)

            pipe.send((tile, block_data, block_stats))
    finally:
        # Send a None to signal end of messages
        pipe.send(None)
        pipe.close()


def open_geotiff_in_process(
    geotiff_filename: str,
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
) -> tuple[RasterGeometry, multiprocessing.Pipe]:
    """Opens a bidirectional connection to a GeoTIFF reader in another process.

    Returns:
        Tuple of (raster_geometry, send_pipe, receive_pipe) for bidirectional communication
    """
    # Create communication pipe
    parent_recv, child_send = multiprocessing.Pipe(duplex=False)

    # Start worker process
    args = geotiff_filename, zoom_strategy, resampling_algorithm, child_send
    process = multiprocessing.Process(target=read_geotiff, args=args)
    process.start()

    # Close child end in parent process
    child_send.close()

    # The first message received is expected to be a RasterGeometry
    raster_geometry = parent_recv.recv()
    assert isinstance(raster_geometry, RasterGeometry)

    return raster_geometry, parent_recv


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


def main(
    geotiff_filename: str,
    raquet_filename: str,
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
):
    """Read GeoTIFF datasource and write to a RaQuet file

    Args:
        geotiff_filename: GeoTIFF filename
        raquet_filename: RaQuet filename
        zoom_strategy: ZoomStrategy member
    """
    raster_geometry, pipe = open_geotiff_in_process(
        geotiff_filename, zoom_strategy, resampling_algorithm
    )

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

        while True:
            received = pipe.recv()
            if received is None:
                # Use a signal value to stop expecting further tiles
                break

            # Expand message to block to retrieve
            tile, block_data, block_stats = received

            logging.info(
                "Tile z=%s x=%s y=%s quadkey=%s bounds=%s",
                tile.z,
                tile.x,
                tile.y,
                hex(quadbin.tile_to_cell((tile.x, tile.y, tile.z))),
                mercantile.bounds(tile),
            )

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
                {
                    "type": btype,
                    "name": bname,
                    "colorinterp": bcolorinterp,
                    "colortable": bcolortable,
                    "stats": stats.__dict__,
                }
                for btype, bname, bcolorinterp, bcolortable, stats in zip(
                    raster_geometry.bandtypes,
                    band_names,
                    raster_geometry.bandcolorinterp,
                    raster_geometry.bandcolortable,
                    band_stats,
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
        pipe.close()


parser = argparse.ArgumentParser()
parser.add_argument("geotiff_filename")
parser.add_argument("raquet_filename")
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output"
)
parser.add_argument(
    "--zoom-strategy",
    help="Strategy to determine zoom, see also https://gdal.org/en/stable/drivers/raster/cog.html#reprojection-related-creation-options",
    choices=list(ZoomStrategy),
    default=ZoomStrategy.AUTO,
)
parser.add_argument(
    "--resampling-algorithm",
    help="Resampling method to use, see also https://gdal.org/en/stable/programs/gdalwarp.html#cmdoption-gdalwarp-r",
    choices=list(ResamplingAlgorithm),
    default=ResamplingAlgorithm.NearestNeighbour,
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    main(
        args.geotiff_filename,
        args.raquet_filename,
        ZoomStrategy(args.zoom_strategy),
        ResamplingAlgorithm(args.resampling_algorithm),
    )
