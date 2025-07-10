#!/usr/bin/env python3
"""Convert GeoTIFF file to RaQuet output

Usage:
    geotiff2raquet.py <geotiff_filename> <raquet_destination>

Help:
    geotiff2raquet.py --help

Required packages:
    - GDAL <https://pypi.org/project/GDAL/>
    - mercantile <https://pypi.org/project/mercantile/>
    - pyarrow <https://pypi.org/project/pyarrow/>
    - quadbin <https://pypi.org/project/quadbin/>
"""

import argparse
import dataclasses
import enum
import gzip
import itertools
import json
import logging
import math
import multiprocessing
import os
import statistics
import struct
import sys
import typing

import mercantile
import pyarrow.compute
import pyarrow.parquet
import quadbin

try:
    import numpy
    import numpy.ma
except ImportError:
    logging.info("Optional Numpy package is unavailable, stats calculation may be slow")
    HAS_NUMPY = False
else:
    HAS_NUMPY = True

# Pixel dimensions of ideal minimum size
TARGET_MIN_SIZE = 128

# Calculate approximate stats from a lower zoom level
STATS_ZOOM_OFFSET = -2


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
class PixelWindow:
    """Convenience wrapper for details of valid raster pixel window"""

    xoff: int
    yoff: int
    xsize: int
    ysize: int


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

    def scale_by(self, zoom: int) -> "RasterStats":
        """Return approximate equivalent stats for a higher zoom"""
        return RasterStats(
            self.count * 4**zoom,
            self.min,
            self.max,
            self.mean,
            self.stddev,
            self.sum * 4**zoom,
            self.sum_squares * 4**zoom,
        )


@dataclasses.dataclass
class BandType:
    """Convenience wrapper for details of band data type"""

    fmt: str
    size: int
    typ: type
    name: str


@dataclasses.dataclass
class Frame:
    """Tile wrapper to track pixels and overviews when descending from 0/0/0"""

    tile: mercantile.Tile
    inputs: list[mercantile.Tile]
    outputs: list["osgeo.gdal.Dataset"]  # noqa: F821 (Color table type safely imported in read_geotiff)

    @staticmethod
    def create(parent: mercantile.Tile, raster_geometry: RasterGeometry) -> "Frame":
        """Generate a new frame with expected inputs and empty outputs"""
        parent_bbox = mercantile.bounds(parent)
        minlon, minlat, maxlon, maxlat = (
            max(parent_bbox.west, raster_geometry.minlon),
            max(parent_bbox.south, raster_geometry.minlat),
            min(parent_bbox.east, raster_geometry.maxlon),
            min(parent_bbox.north, raster_geometry.maxlat),
        )
        children = mercantile.tiles(minlon, minlat, maxlon, maxlat, parent.z + 1)
        return Frame(parent, list(children), [])


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

    # Return stats that might not be None
    if prev_stats is None:
        return curr_stats
    elif curr_stats is None:
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


def read_statistics_python(
    values: list[int | float], nodata: int | float | None
) -> RasterStats | None:
    """Calculate statistics for list of raw band values and optional nodata value"""
    if nodata is not None:
        values = [val for val in values if val != nodata and not math.isnan(val)]

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


def read_statistics_numpy(
    values: "numpy.array", nodata: int | float | None
) -> RasterStats | None:
    """Calculate statistics for array of numeric values and optional nodata value"""
    if nodata is not None:
        bad_values_mask = (values == nodata) | numpy.isnan(values)
        masked_values = numpy.ma.masked_array(values, bad_values_mask)
        value_count = int(masked_values.count())
    else:
        masked_values = values
        value_count = values.size

    if value_count == 0:
        return None

    if masked_values.dtype in (numpy.float16, numpy.float32, numpy.float64):
        ptype = float
    else:
        ptype = int

    return RasterStats(
        count=value_count,
        min=ptype(masked_values.min()),
        max=ptype(masked_values.max()),
        mean=float(masked_values.mean()),
        stddev=float(masked_values.std()),
        sum=ptype(masked_values.sum()),
        sum_squares=float((masked_values.astype(ptype) ** 2).sum()),
    )


def find_bounds(
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    transform: "osgeo.osr.CoordinateTransformation",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    pixel_window: PixelWindow,
) -> tuple[float, float, float, float]:
    """Return outer bounds for raster in a given transformation"""
    _xoff, xres, _, _yoff, _, yres = ds.GetGeoTransform()
    xoff = _xoff + pixel_window.xoff * xres
    yoff = _yoff + pixel_window.yoff * yres
    xspan, yspan = pixel_window.xsize * xres, pixel_window.ysize * yres

    x1, y1, _ = transform.TransformPoint(xoff, yoff)
    x2, y2, _ = transform.TransformPoint(xoff, yoff + yspan)
    x3, y3, _ = transform.TransformPoint(xoff + xspan, yoff)
    x4, y4, _ = transform.TransformPoint(xoff + xspan, yoff + yspan)

    x5, y5 = min(x1, x2, x3, x4), min(y1, y2, y3, y4)
    x6, y6 = max(x1, x2, x3, x4), max(y1, y2, y3, y4)

    return (x5, y5, x6, y6)


def find_pixel_window(
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    tx3857: "osgeo.osr.CoordinateTransformation",  # noqa: F821 (osgeo types safely imported in read_geotiff)
) -> PixelWindow:
    """Return valid pixel window for raster in a given transformation"""
    xoff, xres, _, yoff, _, yres = ds.GetGeoTransform()
    xspan, yspan = ds.RasterXSize * xres, ds.RasterYSize * yres

    # Skip a bunch of math if possible
    try:
        # Transform a selection of points to see if we're within web mercator bounds
        for dx, dy in itertools.permutations((0, 0.5, 1), 2):
            tx3857.TransformPoint(xoff + dx * xspan, yoff + dx * yspan)
        return PixelWindow(0, 0, ds.RasterXSize, ds.RasterYSize)
    except RuntimeError:
        pass

    # Calculate the source projection bounds for web mercator 0/0/0 world tile
    m_x1, m_y1, m_x2, m_y2 = mercantile.xy_bounds(mercantile.Tile(0, 0, 0))
    m_pts = (m_x1, m_y2), (m_x2, m_y2), (m_x2, m_y1), (m_x1, m_y1)
    bb_src = tx3857.GetInverse().TransformPoints(m_pts)

    # Calculate source projection envelope from 0/0/0 world tile bounds
    xmin, xmax = min(x for x, _, _ in bb_src), max(x for x, _, _ in bb_src)
    ymin, ymax = min(y for _, y, _ in bb_src), max(y for _, y, _ in bb_src)
    x1 = max(xmin, xoff) if xres > 0 else min(xmax, xoff)
    x2 = min(xmax, xoff + xspan) if xres > 0 else max(xmin, xoff + xspan)
    y1 = max(ymin, yoff) if yres > 0 else min(ymax, yoff)
    y2 = min(ymax, yoff + yspan) if yres > 0 else max(ymin, yoff + yspan)

    # Calculate source pixel envelope from projection envelope
    x3, x4 = math.ceil((x1 - xoff) / xres), math.floor((x2 - xoff) / xres)
    y3, y4 = math.ceil((y1 - yoff) / yres), math.floor((y2 - yoff) / yres)

    return PixelWindow(x3, y3, x4 - x3, y4 - y3)


def find_resolution(
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    transform: "osgeo.osr.CoordinateTransformation",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    pixel_window: PixelWindow,
) -> float:
    """Return units per pixel for raster via a given transformation"""
    _xoff, xres, _, _yoff, _, yres = ds.GetGeoTransform()
    xoff = _xoff + pixel_window.xoff * xres
    yoff = _yoff + pixel_window.yoff * yres
    xdim, ydim = pixel_window.xsize, pixel_window.ysize

    x1, y1, _ = transform.TransformPoint(xoff, yoff)
    x2, y2, _ = transform.TransformPoint(xoff + xdim * xres, yoff + ydim * yres)

    return math.hypot(x2 - x1, y2 - y1) / math.hypot(xdim, ydim)


def find_minzoom(rg: RasterGeometry, block_zoom: int) -> int:
    """Calculate minimum zoom for a reasonable image size 128px from raster geometry"""
    big_zoom = 32
    ul = mercantile.tile(lat=rg.maxlat, lng=rg.minlon, zoom=big_zoom)
    lr = mercantile.tile(lat=rg.minlat, lng=rg.maxlon, zoom=big_zoom)
    high_hypot = math.hypot(lr.x - ul.x, lr.y - ul.y)
    target_hypot = math.hypot(TARGET_MIN_SIZE, TARGET_MIN_SIZE)
    min_zoom = big_zoom - math.log(high_hypot / target_hypot) / math.log(2) - block_zoom
    return max(0, int(round(min_zoom)))


def find_zoom(resolution: float, zoom_strategy: ZoomStrategy, block_zoom: int) -> int:
    """Calculate web mercator zoom from a raw meters/pixel resolution"""
    tile_dim = 2**block_zoom
    raw_zoom = math.log(mercantile.CE / tile_dim / resolution) / math.log(2)
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
    block_zoom: int,
) -> "osgeo.gdal.Dataset":  # noqa: F821 (osgeo types safely imported in read_geotiff)
    # Initialize warped tile dataset and its bands
    tile_ds = driver.Create(
        f"/vsimem/tile-{tile.z}-{tile.x}-{tile.y}.tif",
        2**block_zoom,
        2**block_zoom,
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


def read_raster_data_stats(
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    gdaltype_bandtypes: dict[int, BandType] | None,
    include_stats: bool,
) -> tuple[list[bytes], list[RasterStats | None]]:
    """Read data and stats from warped bands"""
    block_data, block_stats = [], []

    for band_num in range(1, 1 + ds.RasterCount):
        band = ds.GetRasterBand(band_num)
        data = band.ReadRaster(0, 0, band.XSize, band.YSize)
        if not include_stats or gdaltype_bandtypes is None:
            # No stats wanted, or wanted but not available
            stats = None
        else:
            # Calculate stats
            band_type = gdaltype_bandtypes[band.DataType]
            if HAS_NUMPY:
                pixel_arr = numpy.frombuffer(data, dtype=getattr(numpy, band_type.name))
                stats = read_statistics_numpy(pixel_arr, band.GetNoDataValue())
            else:
                pixel_values = struct.unpack(band_type.fmt * band.XSize * band.YSize, data)
                stats = read_statistics_python(pixel_values, band.GetNoDataValue())
            logging.info(
                "Read %s bytes from band %s: %s...", len(data), band_num, data[:32]
            )
        block_data.append(gzip.compress(data))
        block_stats.append(stats)

    return block_data, block_stats


def read_geotiff(
    geotiff_filename: str,
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
    block_zoom: int,
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
    os.environ["PROJ_DEBUG"] = "0"

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
        src_ds = osgeo.gdal.Open(geotiff_filename)
        src_bands = [src_ds.GetRasterBand(n) for n in range(1, 1 + src_ds.RasterCount)]
        src_sref = src_ds.GetSpatialRef()

        tx3857 = osgeo.osr.CoordinateTransformation(src_sref, web_mercator)
        pixel_window = find_pixel_window(src_ds, tx3857)
        resolution = find_resolution(src_ds, tx3857, pixel_window)
        zoom = find_zoom(resolution, zoom_strategy, block_zoom)

        tx4326 = osgeo.osr.CoordinateTransformation(src_sref, wgs84)
        minlon, minlat, maxlon, maxlat = find_bounds(src_ds, tx4326, pixel_window)

        raster_geometry = RasterGeometry(
            [gdaltype_bandtypes[band.DataType].name for band in src_bands],
            [
                osgeo.gdal.GetColorInterpretationName(
                    band.GetColorInterpretation()
                ).lower()
                for band in src_bands
            ],
            [
                get_colortable_dict(b.GetColorTable()) if b.GetColorTable() else None
                for b in src_bands
            ],
            src_ds.GetRasterBand(1).GetNoDataValue(),
            zoom,
            minlat,
            minlon,
            maxlat,
            maxlon,
        )

        pipe.send(raster_geometry)

        # Driver and options for the warps to come
        gtiff_driver = osgeo.gdal.GetDriverByName("GTiff")
        opts = osgeo.gdal.WarpOptions(
            resampleAlg=resampling_algorithms[resampling_algorithm]
        )

        minzoom = find_minzoom(raster_geometry, block_zoom)
        stats_zoom = max(minzoom, raster_geometry.zoom + STATS_ZOOM_OFFSET)
        stats_zoom_diff = raster_geometry.zoom - stats_zoom

        # Start with a reasonable minimum zoom, then model a recursive descent into all
        # child tiles using a stack of frames. If we reach the maximum zoom read pixels
        # out of the original raster, otherwise stack child tiles and build overviews
        # from prior pixels.
        frames = [
            Frame.create(t, raster_geometry)
            for t in mercantile.tiles(
                raster_geometry.minlon,
                raster_geometry.minlat,
                raster_geometry.maxlon,
                raster_geometry.maxlat,
                minzoom,
            )
        ]

        while frames:
            frame = frames.pop()
            tile_ds: osgeo.gdal.Dataset | None = None
            create_args = gtiff_driver, web_mercator, src_ds, frame.tile, block_zoom
            do_stats = frame.tile.z == stats_zoom

            if frame.tile.z >= raster_geometry.zoom:
                # Read original source pixels at the highest requested zoom
                logging.info("Warp %s from original dataset", frame.tile)
                tile_ds = create_tile_ds(*create_args)
                osgeo.gdal.Warp(
                    destNameOrDestDS=tile_ds, srcDSOrSrcDSTab=src_ds, options=opts
                )
                d, s = read_raster_data_stats(tile_ds, gdaltype_bandtypes, do_stats)
                pipe.send((frame.tile, d, s))
            elif not frame.inputs:
                # Read overview pixels from earlier outputs
                logging.info("Overview %s from %s", frame.tile, frame.outputs)
                tile_ds = create_tile_ds(*create_args)
                for sub_ds in frame.outputs:
                    osgeo.gdal.Warp(
                        destNameOrDestDS=tile_ds, srcDSOrSrcDSTab=sub_ds, options=opts
                    )
                d, s1 = read_raster_data_stats(tile_ds, gdaltype_bandtypes, do_stats)
                s2 = [s.scale_by(stats_zoom_diff) if s else None for s in s1]
                pipe.send((frame.tile, d, s2))
            else:
                # Descend deeper into tile hierarchy
                next_tile = frame.inputs.pop()
                logging.info("Extend %s with %s", frame.tile, next_tile)
                frames.extend([frame, Frame.create(next_tile, raster_geometry)])

            if tile_ds is not None and frames:
                # Save current output for future overviews
                frames[-1].outputs.append(tile_ds)
    finally:
        # Send a None to signal end of messages
        pipe.send(None)
        pipe.close()


def open_geotiff_in_process(
    geotiff_filename: str,
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
    block_zoom: int,
) -> tuple[RasterGeometry, multiprocessing.Pipe]:
    """Opens a bidirectional connection to a GeoTIFF reader in another process.

    Returns:
        Tuple of (raster_geometry, send_pipe, receive_pipe) for bidirectional communication
    """
    # Create communication pipe
    parent_recv, child_send = multiprocessing.Pipe(duplex=False)

    # Start worker process
    args = geotiff_filename, zoom_strategy, resampling_algorithm, block_zoom, child_send
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
    zoom: int, xmin: int, ymin: int, xmax: int, ymax: int, block_zoom: int
) -> dict[str, int]:
    """Get dictionary of basic dimensions for RaQuet metadata from tile bounds"""
    # First and last tile in rectangular coverage
    upper_left = mercantile.Tile(x=xmin, y=ymin, z=zoom)
    lower_right = mercantile.Tile(x=xmax, y=ymax, z=zoom)

    # Pixel dimensions
    block_width, block_height = 2**block_zoom, 2**block_zoom
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
        "pixel_resolution": zoom + block_zoom,
    }


def create_schema(rg: RasterGeometry) -> tuple[pyarrow.lib.Schema, list[str]]:
    """Create table schema and band column names for RasterGeometry instance"""
    band_names = [f"band_{n}" for n in range(1, 1 + len(rg.bandtypes))]

    # Create table schema based on band count
    schema = pyarrow.schema(
        [
            ("block", pyarrow.uint64()),
            ("metadata", pyarrow.string()),
            *[(bname, pyarrow.binary()) for bname in band_names],
        ]
    )

    return schema, band_names


def create_metadata(
    rg: RasterGeometry,
    band_names: list[str],
    band_stats: list[RasterStats | None],
    block_count: int,
    minresolution: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    block_zoom: int,
) -> dict:
    """Create a dictionary of RaQuet metadata

    See https://github.com/CartoDB/raquet/blob/master/format-specs/raquet.md#metadata-specification
    """
    metadata_json = {
        "version": "0.1.0",
        "compression": "gzip",
        "block_resolution": rg.zoom,
        "minresolution": minresolution,
        "maxresolution": rg.zoom,
        "nodata": rg.nodata,
        "num_blocks": block_count,
        "num_pixels": block_count * (2**block_zoom) * (2**block_zoom),
        "bands": [
            {
                "type": btype,
                "name": bname,
                "colorinterp": bcolorinterp,
                "colortable": bcolortable,
                "stats": dict(approximated_stats=True, **stats.__dict__),
            }
            for btype, bname, bcolorinterp, bcolortable, stats in zip(
                rg.bandtypes,
                band_names,
                rg.bandcolorinterp,
                rg.bandcolortable,
                band_stats,
            )
        ],
        **get_raquet_dimensions(rg.zoom, xmin, ymin, xmax, ymax, block_zoom),
    }

    return metadata_json


def flush_rows_to_file(
    writer: pyarrow.parquet.ParquetWriter, schema: pyarrow.lib.Schema, rows: list[dict]
):
    """Write a list of rows then destructively clear it in-place"""
    if not rows:  # Skip writing if rows is empty
        return

    rows_dict = {key: [row[key] for row in rows] for key in schema.names}
    table = pyarrow.Table.from_pydict(rows_dict, schema=schema)
    writer.write_table(table, row_group_size=len(rows))

    # Destroy content of rows
    rows.clear()


def main(
    geotiff_filename: str,
    raquet_destination: str,
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
    block_zoom: int,
    target_size: int | None = None,
):
    """Read GeoTIFF datasource and write to RaQuet

    Args:
        geotiff_filename: GeoTIFF filename
        raquet_destination: RaQuet destination, file or dirname depending on target_size
        zoom_strategy: ZoomStrategy member
        resampling_algorithm: ResamplingAlgorithm member
        target_size: Integer number of bytes for individual parquet files in raquet_destination
    """
    if target_size is None:
        raquet_destinations = itertools.repeat((raquet_destination, math.inf))
    else:
        if os.path.exists(raquet_destination):
            raise ValueError(f"{raquet_destination} already exists")
        os.mkdir(raquet_destination)
        # Prepare a generator of sequential file names
        raquet_destinations = (
            (os.path.join(raquet_destination, f"part{i:03d}.parquet"), target_size)
            for i in itertools.count()
        )

    for raquet_filename in convert_to_raquet_files(
        geotiff_filename,
        raquet_destinations,
        zoom_strategy,
        resampling_algorithm,
        block_zoom,
    ):
        logging.info("Wrote %s", raquet_filename)


def convert_to_raquet_files(
    geotiff_filename: str,
    raquet_destinations: typing.Generator[tuple[str, float], None, None],
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
    block_zoom: int,
) -> typing.Generator[str, None, None]:
    """Read GeoTIFF datasource and write to RaQuet files

    Args:
        geotiff_filename: GeoTIFF filename
        raquet_destinations: tuples of (filename, target sizeof)
        zoom_strategy: ZoomStrategy member
        resampling_algorithm: ResamplingAlgorithm member

    Yields output filenames as they are written.
    """
    raster_geometry, pipe = open_geotiff_in_process(
        geotiff_filename, zoom_strategy, resampling_algorithm, block_zoom
    )

    try:
        schema, band_names = create_schema(raster_geometry)

        # Initialize empty lists to collect rows and stats
        max_rowcount, rows, band_stats = 1000, [], [None for _ in band_names]

        # Initialize a parquet writer
        rfname, target_sizeof = next(raquet_destinations)
        writer, sizeof_so_far = pyarrow.parquet.ParquetWriter(rfname, schema), 0

        xmin, ymin, xmax, ymax = math.inf, math.inf, -math.inf, -math.inf
        minresolution, block_count = math.inf, 0

        while True:
            received = pipe.recv()
            if received is None:
                # Use a signal value to stop expecting further tiles
                break

            # Expand message to block to retrieve
            tile, block_data, block_stats = received
            minresolution = min(minresolution, tile.z)

            logging.info(
                "Tile z=%s x=%s y=%s quadkey=%s bounds=%s",
                tile.z,
                tile.x,
                tile.y,
                hex(quadbin.tile_to_cell((tile.x, tile.y, tile.z))),
                mercantile.bounds(tile),
            )

            # No data in any band means we can skip this row
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
            sizeof_so_far += sum(sys.getsizeof(v) for v in rows[-1].values())

            # Write a row group and recalibrate sizeof_so_far when we hit the row count limit
            if len(rows) >= max_rowcount:
                flush_rows_to_file(writer, schema, rows)
                sizeof_so_far = os.stat(rfname).st_size

            # Write and yield a whole file when we hit the sizeof limit
            if sizeof_so_far > target_sizeof:
                if rows:
                    flush_rows_to_file(writer, schema, rows)
                writer.close()
                yield rfname

                # Reinitialize the parquet writer
                rfname, target_sizeof = next(raquet_destinations)
                writer, sizeof_so_far = pyarrow.parquet.ParquetWriter(rfname, schema), 0

            if tile.z == raster_geometry.zoom:
                # Calculate bounds and count blocks only at the highest zoom level
                xmin, ymin = min(xmin, tile.x), min(ymin, tile.y)
                xmax, ymax = max(xmax, tile.x), max(ymax, tile.y)
                block_count += 1

            if any(block_stats):
                # Accumulate band statistics and real bounds based on included tiles
                band_stats = [combine_stats(*bs) for bs in zip(band_stats, block_stats)]

        for i, stats in enumerate(band_stats):
            logging.info("Band %s %s", i + 1, stats)

        # Write remaining rows
        flush_rows_to_file(writer, schema, rows)

        # Finish writing with a single metadata row
        metadata_json = create_metadata(
            raster_geometry,
            band_names,
            band_stats,
            block_count,
            minresolution,
            xmin,
            ymin,
            xmax,
            ymax,
            block_zoom,
        )
        metadata_row = {
            "block": 0,
            "metadata": json.dumps(metadata_json),
            **{bname: None for bname in band_names},
        }
        flush_rows_to_file(writer, schema, [metadata_row])
        writer.close()
        yield rfname

    finally:
        pipe.close()


parser = argparse.ArgumentParser()
parser.add_argument("geotiff_filename")
parser.add_argument("raquet_destination")
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
parser.add_argument(
    "--target-size",
    help="Target byte size of all rows in each parquet file, actual file sizes may be larger",
    type=int,
)

parser.add_argument(
    "--block-size",
    help="Size of each square block in pixels, default=256px",
    choices=(256, 512, 1024),
    default=256,
    type=int,
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Zoom offset from tiles to pixels, e.g. 8 = 256px tiles
    block_zoom = int(math.log(args.block_size) / math.log(2))

    main(
        args.geotiff_filename,
        args.raquet_destination,
        ZoomStrategy(args.zoom_strategy),
        ResamplingAlgorithm(args.resampling_algorithm),
        block_zoom,
        args.target_size,
    )
