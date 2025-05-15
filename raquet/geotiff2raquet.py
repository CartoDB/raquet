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
"""

import argparse
import copy
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

# Pixel dimensions of ideal minimum size
TARGET_MIN_SIZE = 128


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

    # Special value for counting instances of block stats in combine_stats()
    blocks: int = 1


@dataclasses.dataclass
class NoDataStats:
    """Special case of raster statistics on an all-nodata raster"""

    blocks: int = 1


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
    prev_stats: RasterStats | NoDataStats | None, curr_stats: RasterStats | NoDataStats
) -> RasterStats | None:
    """Combine two RasterStats into one"""

    if prev_stats is None:
        # Likely to represent an initial None value, don't count blocks
        return curr_stats

    if isinstance(prev_stats, NoDataStats):
        # Count just the blocks on previous nodata stats
        next_stats = copy.deepcopy(curr_stats)
        next_stats.blocks += prev_stats.blocks
        return next_stats

    if isinstance(curr_stats, NoDataStats):
        # Count just the blocks on current nodata stats
        next_stats = copy.deepcopy(prev_stats)
        next_stats.blocks += curr_stats.blocks
        return next_stats

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
        blocks=prev_stats.blocks + curr_stats.blocks,
    )

    return next_stats


def read_statistics(
    values: list[int | float], nodata: int | float | None
) -> RasterStats | NoDataStats:
    """Calculate statistics for list of raw band values and optional nodata value"""
    if nodata is not None:
        values = [val for val in values if val != nodata]

    if len(values) == 0:
        return NoDataStats()

    return RasterStats(
        count=len(values),
        min=min(values),
        max=max(values),
        mean=statistics.mean(values),
        stddev=statistics.stdev(values),
        sum=sum(val for val in values),
        sum_squares=sum(val**2 for val in values),
    )


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


def find_minzoom(rg: RasterGeometry) -> int:
    """Calculate minimum zoom for a reasonable image size 128px from raster geometry"""
    big_zoom = 32
    ul = mercantile.tile(lat=rg.maxlat, lng=rg.minlon, zoom=big_zoom)
    lr = mercantile.tile(lat=rg.minlat, lng=rg.maxlon, zoom=big_zoom)
    high_hypot = math.hypot(lr.x - ul.x, lr.y - ul.y)
    target_hypot = math.hypot(TARGET_MIN_SIZE, TARGET_MIN_SIZE)
    min_zoom = big_zoom - math.log(high_hypot / target_hypot) / math.log(2) - BLOCK_ZOOM
    return int(round(min_zoom))


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
        f"/vsimem/tile-{tile.z}-{tile.x}-{tile.y}.tif",
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


def read_raster_data_stats(
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_geotiff)
    gdaltype_bandtypes: dict[int, BandType] | None,
    include_stats: bool = True,
) -> tuple[list[bytes], list[RasterStats | NoDataStats]]:
    """Read data and stats from warped bands"""
    block_data, block_stats = [], []

    for band_num in range(1, 1 + ds.RasterCount):
        band = ds.GetRasterBand(band_num)
        data = band.ReadRaster(0, 0, band.XSize, band.YSize)
        if gdaltype_bandtypes is not None and include_stats:
            band_type = gdaltype_bandtypes[band.DataType]
            pixel_values = struct.unpack(band_type.fmt * band.XSize * band.YSize, data)
            stats = read_statistics(pixel_values, band.GetNoDataValue())
            logging.info(
                "Read %s bytes from band %s: %s...", len(data), band_num, data[:32]
            )
        else:
            stats = NoDataStats()
        block_data.append(gzip.compress(data))
        block_stats.append(stats)

    return block_data, block_stats


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
        src_ds = osgeo.gdal.Open(geotiff_filename)
        src_bands = [src_ds.GetRasterBand(n) for n in range(1, 1 + src_ds.RasterCount)]
        src_sref = src_ds.GetSpatialRef()

        tx4326 = osgeo.osr.CoordinateTransformation(src_sref, wgs84)
        minlon, minlat, maxlon, maxlat = find_bounds(src_ds, tx4326)

        tx3857 = osgeo.osr.CoordinateTransformation(src_sref, web_mercator)
        resolution = find_resolution(src_ds, tx3857)
        zoom = find_zoom(resolution, zoom_strategy)

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
                find_minzoom(raster_geometry),
            )
        ]

        while frames:
            frame = frames.pop()
            tile_ds: osgeo.gdal.Dataset | None = None

            if frame.tile.z == raster_geometry.zoom:
                # Read original source pixels at the highest requested zoom
                logging.info("Warp %s from original dataset", frame.tile)
                tile_ds = create_tile_ds(gtiff_driver, web_mercator, src_ds, frame.tile)
                osgeo.gdal.Warp(
                    destNameOrDestDS=tile_ds, srcDSOrSrcDSTab=src_ds, options=opts
                )
                data, stats = read_raster_data_stats(tile_ds, gdaltype_bandtypes)
                pipe.send((frame.tile, data, stats))
            elif not frame.inputs:
                # Read overview pixels from earlier outputs
                logging.info("Overview %s from %s", frame.tile, frame.outputs)
                tile_ds = create_tile_ds(gtiff_driver, web_mercator, src_ds, frame.tile)
                for sub_ds in frame.outputs:
                    osgeo.gdal.Warp(
                        destNameOrDestDS=tile_ds, srcDSOrSrcDSTab=sub_ds, options=opts
                    )
                data, stats = read_raster_data_stats(tile_ds, None, include_stats=False)
                pipe.send((frame.tile, data, stats))
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
        schema, band_names = create_schema(raster_geometry)

        # Initialize empty lists to collect rows and stats
        row_group_size, rows, band_stats = 1000, [], [None for _ in band_names]

        # Initialize the parquet writer
        writer = pyarrow.parquet.ParquetWriter(raquet_filename, schema)

        xmin, ymin, xmax, ymax = math.inf, math.inf, -math.inf, -math.inf
        minresolution = math.inf

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

            # Write a row group when we hit the size limit
            if len(rows) >= row_group_size:
                rows_dict, rows = {k: [r[k] for r in rows] for k in schema.names}, []
                writer.write_table(
                    pyarrow.Table.from_pydict(rows_dict, schema=schema),
                    row_group_size=row_group_size,
                )

            # Skip stats from overview tiles?
            if tile.z < raster_geometry.zoom:
                continue

            # Accumulate band statistics and real bounds based on included tiles
            band_stats = [combine_stats(p, c) for p, c in zip(band_stats, block_stats)]
            xmin, ymin = min(xmin, tile.x), min(ymin, tile.y)
            xmax, ymax = max(xmax, tile.x), max(ymax, tile.y)

        for i, stats in enumerate(band_stats):
            logging.info("Band %s %s", i + 1, stats)

        # Write remaining rows
        rows_dict = {k: [row[k] for row in rows] for k in schema.names}
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
            "minresolution": minresolution,
            "maxresolution": raster_geometry.zoom,
            "nodata": raster_geometry.nodata,
            "num_blocks": band_stats[0].blocks,
            "num_pixels": band_stats[0].blocks * (2**BLOCK_ZOOM) * (2**BLOCK_ZOOM),
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
