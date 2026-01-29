#!/usr/bin/env python3
"""Convert raster files (GeoTIFF, NetCDF, etc.) to RaQuet output

Supports any GDAL-readable raster format including:
- GeoTIFF / Cloud Optimized GeoTIFF (COG)
- NetCDF (with CF time dimension support)
- And other formats supported by GDAL

Usage:
    raster2raquet.py <input_file> <raquet_destination>

Help:
    raster2raquet.py --help

Required packages:
    - GDAL <https://pypi.org/project/GDAL/>
    - mercantile <https://pypi.org/project/mercantile/>
    - pyarrow <https://pypi.org/project/pyarrow/>
    - quadbin <https://pypi.org/project/quadbin/>
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import enum
import gzip
import itertools
import json
import logging
import math
import multiprocessing
import os
import re
import statistics
import struct
import sys
import tempfile
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


class OverviewMode(enum.StrEnum):
    """Overview generation mode for RaQuet conversion

    Similar to COG overview options:
    - NONE: No overviews, only native resolution tiles (fastest, smallest files)
    - AUTO: Automatic overview generation (default, builds full pyramid)
    """

    NONE = "none"
    AUTO = "auto"


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
    cf_time: CFTimeInfo | None = None  # CF time dimension metadata
    band_time_values: list[float | None] | None = None  # CF time value for each band
    # GDAL band metadata (v0.3.0)
    band_scales: list[float | None] | None = None
    band_offsets: list[float | None] | None = None
    band_units: list[str] | None = None
    band_descriptions: list[str] | None = None
    band_nodatas: list[float | None] | None = None  # Per-band nodata values


@dataclasses.dataclass
class RasterStats:
    """Convenience wrapper for raster statistics"""

    count: int  # Valid (non-nodata) pixel count
    min: int | float
    max: int | float
    mean: int | float
    stddev: int | float
    sum: int | float
    sum_squares: int | float
    total_pixels: int = 0  # Total pixels including nodata (for valid_percent)

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
            self.total_pixels * 4**zoom,
        )

    @property
    def valid_percent(self) -> float:
        """Percentage of valid (non-nodata) pixels"""
        if self.total_pixels == 0:
            return 100.0
        return (self.count / self.total_pixels) * 100.0


@dataclasses.dataclass
class BandType:
    """Convenience wrapper for details of band data type"""

    fmt: str
    size: int
    typ: type
    name: str


@dataclasses.dataclass
class CFTimeInfo:
    """Container for CF convention time dimension metadata"""

    units: str  # e.g., "minutes", "hours", "days"
    reference_date: datetime.datetime  # Reference point for time values
    calendar: str  # e.g., "standard", "gregorian", "360_day"
    values: list[int | float]  # Time dimension values from source
    raw_units_string: str  # Original CF units string

    @property
    def is_gregorian_compatible(self) -> bool:
        """Check if calendar can be converted to standard timestamps"""
        return self.calendar.lower() in (
            "standard",
            "gregorian",
            "proleptic_gregorian",
        )

    def to_iso_duration(self) -> str | None:
        """Estimate ISO 8601 duration from time values if regular intervals"""
        if len(self.values) < 2:
            return None

        # Check if intervals are regular
        intervals = [self.values[i + 1] - self.values[i] for i in range(len(self.values) - 1)]
        if not intervals:
            return None

        avg_interval = sum(intervals) / len(intervals)
        is_regular = all(abs(i - avg_interval) / avg_interval < 0.01 for i in intervals) if avg_interval != 0 else True

        if not is_regular:
            return None

        # Map common intervals to ISO durations
        unit_map = {
            "minutes": {"1": "PT1M", "60": "PT1H", "1440": "P1D", "43200": "P1M", "44640": "P1M"},
            "hours": {"1": "PT1H", "24": "P1D", "720": "P1M", "744": "P1M"},
            "days": {"1": "P1D", "30": "P1M", "31": "P1M", "365": "P1Y", "366": "P1Y"},
            "months": {"1": "P1M", "12": "P1Y"},
            "years": {"1": "P1Y"},
        }

        interval_str = str(int(round(avg_interval)))
        return unit_map.get(self.units, {}).get(interval_str)


def parse_cf_time_units(units_string: str, calendar: str = "standard") -> CFTimeInfo | None:
    """Parse CF convention time units string into structured metadata.

    Args:
        units_string: CF units string like "minutes since 1980-01-01 00:00:00"
        calendar: CF calendar type (default "standard")

    Returns:
        CFTimeInfo instance or None if parsing fails
    """
    # Pattern: "<unit> since <reference_date>"
    # Examples: "minutes since 1980-01-01 00:00:00", "days since 1850-01-01"
    pattern = r"^(\w+)\s+since\s+(.+)$"
    match = re.match(pattern, units_string.strip(), re.IGNORECASE)

    if not match:
        logging.warning("Could not parse CF time units: %s", units_string)
        return None

    unit = match.group(1).lower()
    date_str = match.group(2).strip()

    # Normalize unit names
    unit_aliases = {
        "minute": "minutes",
        "hour": "hours",
        "day": "days",
        "month": "months",
        "year": "years",
        "second": "seconds",
    }
    unit = unit_aliases.get(unit, unit)

    # Parse reference date - try multiple formats
    ref_date = None
    date_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%Y%m%d",
    ]

    for fmt in date_formats:
        try:
            ref_date = datetime.datetime.strptime(date_str, fmt)
            break
        except ValueError:
            continue

    if ref_date is None:
        logging.warning("Could not parse reference date: %s", date_str)
        return None

    return CFTimeInfo(
        units=unit,
        reference_date=ref_date,
        calendar=calendar.lower() if calendar else "standard",
        values=[],  # Will be populated later
        raw_units_string=units_string,
    )


def cf_to_timestamp(cf_value: int | float, cf_info: CFTimeInfo) -> datetime.datetime | None:
    """Convert a CF time value to a Python datetime.

    Args:
        cf_value: Numeric time offset from reference date
        cf_info: CF time metadata

    Returns:
        datetime instance or None if conversion not possible (non-Gregorian calendar)
    """
    if not cf_info.is_gregorian_compatible:
        return None

    ref = cf_info.reference_date

    try:
        if cf_info.units == "seconds":
            return ref + datetime.timedelta(seconds=cf_value)
        elif cf_info.units == "minutes":
            return ref + datetime.timedelta(minutes=cf_value)
        elif cf_info.units == "hours":
            return ref + datetime.timedelta(hours=cf_value)
        elif cf_info.units == "days":
            return ref + datetime.timedelta(days=cf_value)
        elif cf_info.units == "months":
            # Approximate: add months by adjusting year/month
            total_months = ref.month + int(cf_value) - 1
            years_to_add = total_months // 12
            new_month = (total_months % 12) + 1
            return ref.replace(year=ref.year + years_to_add, month=new_month)
        elif cf_info.units == "years":
            return ref.replace(year=ref.year + int(cf_value))
        else:
            logging.warning("Unknown CF time unit: %s", cf_info.units)
            return None
    except (ValueError, OverflowError) as e:
        logging.warning("Failed to convert CF time %s: %s", cf_value, e)
        return None


def extract_cf_time_from_gdal(ds: "osgeo.gdal.Dataset") -> CFTimeInfo | None:  # noqa: F821
    """Extract CF time dimension metadata from a GDAL dataset.

    Looks for NetCDF time dimension metadata in GDAL's metadata domains.

    Args:
        ds: GDAL dataset (typically opened from NetCDF)

    Returns:
        CFTimeInfo instance or None if no time dimension found
    """
    metadata = ds.GetMetadata()

    # Look for time units (NetCDF convention)
    time_units = metadata.get("time#units")
    if not time_units:
        # Try alternate metadata keys
        for key in metadata:
            if key.endswith("#units") and "time" in key.lower():
                time_units = metadata[key]
                break

    if not time_units:
        return None

    # Get calendar (default to standard)
    calendar = metadata.get("time#calendar", "standard")

    # Parse units string
    cf_info = parse_cf_time_units(time_units, calendar)
    if cf_info is None:
        return None

    # Extract time values from NETCDF_DIM_time_VALUES
    time_values_str = metadata.get("NETCDF_DIM_time_VALUES")
    if time_values_str:
        # Format: "{val1,val2,val3,...}"
        try:
            values_str = time_values_str.strip("{}")
            cf_info.values = [float(v) for v in values_str.split(",")]
        except (ValueError, AttributeError) as e:
            logging.warning("Failed to parse time values: %s", e)

    logging.info(
        "Detected CF time dimension: %d values, units='%s', calendar='%s'",
        len(cf_info.values),
        cf_info.raw_units_string,
        cf_info.calendar,
    )

    return cf_info


def get_band_time_value(band: "osgeo.gdal.Band") -> float | None:  # noqa: F821
    """Get the CF time value for a raster band.

    Args:
        band: GDAL raster band

    Returns:
        CF time value (in the units specified by cf:units) or None if not a time-dimensioned band
    """
    metadata = band.GetMetadata()
    time_value_str = metadata.get("NETCDF_DIM_time")
    if time_value_str is not None:
        try:
            return float(time_value_str)
        except ValueError:
            pass
    return None


@dataclasses.dataclass
class Frame:
    """Tile wrapper to track pixels and overviews when descending from 0/0/0"""

    tile: mercantile.Tile
    inputs: list[mercantile.Tile]
    outputs: list["osgeo.gdal.Dataset"]  # noqa: F821 (Color table type safely imported in read_raster)

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


def get_colortable_dict(color_table: "osgeo.gdal.ColorTable"):  # noqa: F821 (Color table type safely imported in read_raster)
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
        total_pixels=prev_stats.total_pixels + curr_stats.total_pixels,
    )

    return next_stats


def read_statistics_python(
    values: list[int | float], nodata: int | float | None
) -> RasterStats | None:
    """Calculate statistics for list of raw band values and optional nodata value"""
    total_pixels = len(values)
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
        total_pixels=total_pixels,
    )


def read_statistics_numpy(
    values: "numpy.array", nodata: int | float | None
) -> RasterStats | None:
    """Calculate statistics for array of numeric values and optional nodata value"""
    total_pixels = values.size
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
        total_pixels=total_pixels,
    )


def is_web_mercator(
    sref: "osgeo.osr.SpatialReference | None",  # noqa: F821 (osgeo types safely imported in read_raster)
    web_mercator: "osgeo.osr.SpatialReference",  # noqa: F821 (osgeo types safely imported in read_raster)
) -> bool:
    """Check if spatial reference is EPSG:3857 web mercator"""
    if sref is None:
        return False
    return bool(sref.IsSame(web_mercator))


def find_matching_overview(
    band: "osgeo.gdal.Band",  # noqa: F821 (osgeo types safely imported in read_raster)
    target_reduction: float,
    tolerance: float = 0.1,
) -> int | None:
    """Find overview index that matches target reduction factor within tolerance.

    Args:
        band: GDAL raster band to check for overviews
        target_reduction: Desired reduction factor (e.g., 2.0 for half resolution)
        tolerance: Acceptable relative difference (default 10%)

    Returns:
        Overview index (0-based) or None if no matching overview found.
    """
    for i in range(band.GetOverviewCount()):
        ovr = band.GetOverview(i)
        ovr_reduction = band.XSize / ovr.XSize  # actual reduction factor
        if abs(ovr_reduction - target_reduction) / target_reduction < tolerance:
            return i
    return None


def find_bounds(
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_raster)
    transform: "osgeo.osr.CoordinateTransformation",  # noqa: F821 (osgeo types safely imported in read_raster)
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
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_raster)
    tx3857: "osgeo.osr.CoordinateTransformation",  # noqa: F821 (osgeo types safely imported in read_raster)
) -> PixelWindow:
    """Return valid pixel window for raster in a given transformation"""
    xoff, xres, _, yoff, _, yres = ds.GetGeoTransform()
    xspan, yspan = ds.RasterXSize * xres, ds.RasterYSize * yres

    # Skip a bunch of math if possible
    try:
        # Transform a selection of points to see if we're within web mercator bounds
        for dx, dy in itertools.permutations((0, 0.5, 1), 2):
            _, y, _ = tx3857.TransformPoint(xoff + dx * xspan, yoff + dx * yspan)
            if y < -mercantile.CE / 2 or mercantile.CE / 2 < y:
                # mercantile.CE is circumference of the earth in web mercator
                raise ValueError("Outside web mercator bounds")
        return PixelWindow(0, 0, ds.RasterXSize, ds.RasterYSize)
    except (RuntimeError, ValueError):
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
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_raster)
    transform: "osgeo.osr.CoordinateTransformation",  # noqa: F821 (osgeo types safely imported in read_raster)
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
    return max(0, min(rg.zoom, int(round(min_zoom))))


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
    driver: "osgeo.gdal.Driver",  # noqa: F821 (osgeo types safely imported in read_raster)
    web_mercator: "osgeo.osr.SpatialReference",  # noqa: F821 (osgeo types safely imported in read_raster)
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_raster)
    tile: mercantile.Tile,
    block_zoom: int,
    numpy_module: "module | None" = None,  # noqa: F821
) -> "osgeo.gdal.Dataset":  # noqa: F821 (osgeo types safely imported in read_raster)
    # Initialize warped tile dataset and its bands
    tile_size = 2**block_zoom
    tile_ds = driver.Create(
        f"/vsimem/tile-{tile.z}-{tile.x}-{tile.y}.tif",
        tile_size,
        tile_size,
        ds.RasterCount,
        ds.GetRasterBand(1).DataType,
    )
    tile_ds.SetProjection(web_mercator.ExportToWkt())

    for band_num in range(1, 1 + ds.RasterCount):
        src_band = ds.GetRasterBand(band_num)
        nodata = src_band.GetNoDataValue()
        dst_band = tile_ds.GetRasterBand(band_num)

        if nodata is not None:
            dst_band.SetNoDataValue(nodata)
            # Initialize tile with nodata values to avoid garbage in uncovered areas
            if numpy_module is not None:
                # Use numpy for efficient fill
                fill_value = nodata
                dtype = numpy_module.float64  # Safe default that works for all types
                fill_array = numpy_module.full((tile_size, tile_size), fill_value, dtype=dtype)
                dst_band.WriteArray(fill_array)
            else:
                # Fallback: fill with nodata using GDAL
                dst_band.Fill(nodata)

    # Convert mercator coordinates to pixel coordinates
    xmin, ymin, xmax, ymax = mercantile.xy_bounds(tile)
    px_width = (xmax - xmin) / tile_ds.RasterXSize
    px_height = (ymax - ymin) / tile_ds.RasterYSize
    tile_ds.SetGeoTransform([xmin, px_width, 0, ymax, 0, -px_height])

    return tile_ds


def read_raster_data_stats(
    ds: "osgeo.gdal.Dataset",  # noqa: F821 (osgeo types safely imported in read_raster)
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


def read_raster(
    input_filename: str,
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
    block_zoom: int,
    pipe: multiprocessing.Pipe,
    overview_mode: OverviewMode = OverviewMode.AUTO,
    min_zoom_override: int | None = None,
):
    """Worker process that accesses a raster file through pipes.

    Send RasterGeometry via pipe first then follow with (tile, data, stats) tuples.

    Args:
        input_filename: Path to raster file (GeoTIFF, NetCDF, or any GDAL-supported format)
        zoom_strategy: Web mercator zoom level selection
        resampling_algorithm: Resampling method to use
        block_zoom: Block zoom level (8 for 256px blocks)
        pipe: Connection to send data to parent
        overview_mode: Overview generation mode (NONE=no overviews, AUTO=full pyramid)
        min_zoom_override: Optional minimum zoom level override (overrides AUTO calculation)
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
        src_ds = osgeo.gdal.Open(input_filename)
        src_bands = [src_ds.GetRasterBand(n) for n in range(1, 1 + src_ds.RasterCount)]
        src_sref = src_ds.GetSpatialRef()

        # If no CRS defined, try reopening with ASSUME_LONGLAT=YES (useful for NetCDF)
        if src_sref is None:
            logging.info("No CRS defined, assuming WGS84 (lon/lat) coordinates")
            src_ds = osgeo.gdal.OpenEx(
                input_filename,
                open_options=["ASSUME_LONGLAT=YES"],
            )
            src_bands = [src_ds.GetRasterBand(n) for n in range(1, 1 + src_ds.RasterCount)]
            src_sref = src_ds.GetSpatialRef()

        # Check if we can use COG overviews directly (source is web mercator with overviews)
        overview_count = src_bands[0].GetOverviewCount()
        use_cog_overviews = is_web_mercator(src_sref, web_mercator) and overview_count > 0
        if use_cog_overviews:
            logging.info(
                "Source is web mercator with %d overviews - will use COG overviews directly",
                overview_count,
            )
        elif overview_count > 0:
            logging.info(
                "Source has %d overviews but is not web mercator - will rebuild pyramids",
                overview_count,
            )

        tx3857 = osgeo.osr.CoordinateTransformation(src_sref, web_mercator)
        pixel_window = find_pixel_window(src_ds, tx3857)
        resolution = find_resolution(src_ds, tx3857, pixel_window)
        zoom = find_zoom(resolution, zoom_strategy, block_zoom)

        tx4326 = osgeo.osr.CoordinateTransformation(src_sref, wgs84)
        minlon, minlat, maxlon, maxlat = find_bounds(src_ds, tx4326, pixel_window)

        # Extract CF time dimension metadata if available (e.g., from NetCDF)
        cf_time = extract_cf_time_from_gdal(src_ds)
        band_time_values = None
        if cf_time is not None:
            # Get CF time value for each band
            band_time_values = [get_band_time_value(band) for band in src_bands]
            logging.info("Band time values (first 10): %s", band_time_values[:10])

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
            cf_time=cf_time,
            band_time_values=band_time_values,
            # GDAL band metadata (v0.3.0)
            band_scales=[band.GetScale() for band in src_bands],
            band_offsets=[band.GetOffset() for band in src_bands],
            band_units=[band.GetUnitType() or "" for band in src_bands],
            band_descriptions=[band.GetDescription() or "" for band in src_bands],
            band_nodatas=[band.GetNoDataValue() for band in src_bands],
        )

        pipe.send(raster_geometry)

        # Driver and options for the warps to come
        gtiff_driver = osgeo.gdal.GetDriverByName("GTiff")

        # Get nodata value for warp options
        nodata_value = src_ds.GetRasterBand(1).GetNoDataValue()

        # WarpOptions for reading from source at max zoom
        opts_source = osgeo.gdal.WarpOptions(
            resampleAlg=resampling_algorithms[resampling_algorithm],
            srcNodata=nodata_value,
            dstNodata=nodata_value,
        )

        # WarpOptions for creating overviews from child tiles
        # Use Average resampling for better overview quality with nodata handling
        opts_overview = osgeo.gdal.WarpOptions(
            resampleAlg=osgeo.gdal.GRA_Average,
            srcNodata=nodata_value,
            dstNodata=nodata_value,
        )

        # Numpy module for tile initialization (if available)
        numpy_mod = numpy if HAS_NUMPY else None

        # Determine minimum zoom based on overview mode and optional override
        if overview_mode == OverviewMode.NONE:
            # No overviews: only generate tiles at native resolution
            minzoom = raster_geometry.zoom
            logging.info("Overview mode NONE: generating only native resolution tiles (zoom %d)", minzoom)
        elif min_zoom_override is not None:
            # User-specified minimum zoom (clamp between 0 and max zoom)
            minzoom = max(0, min(raster_geometry.zoom, min_zoom_override))
            logging.info("Min zoom override: generating zoom levels %d to %d", minzoom, raster_geometry.zoom)
        else:
            # Auto mode: calculate minimum zoom for reasonable overview
            minzoom = find_minzoom(raster_geometry, block_zoom)
            logging.info("Auto mode: generating zoom levels %d to %d", minzoom, raster_geometry.zoom)

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
            create_args = gtiff_driver, web_mercator, src_ds, frame.tile, block_zoom, numpy_mod
            do_stats = frame.tile.z == stats_zoom

            if frame.tile.z > raster_geometry.zoom:
                raise NotImplementedError("Zoom higher than expected by find_minzoom()")

            if frame.tile.z == raster_geometry.zoom:
                # Read original source pixels at the highest requested zoom
                logging.info("Warp %s from original dataset", frame.tile)
                tile_ds = create_tile_ds(*create_args)
                osgeo.gdal.Warp(
                    destNameOrDestDS=tile_ds, srcDSOrSrcDSTab=src_ds, options=opts_source
                )
                d, s = read_raster_data_stats(tile_ds, gdaltype_bandtypes, do_stats)
                pipe.send((frame.tile, d, s))
            elif not frame.inputs:
                # Build overview tile - either from COG overviews or from child tiles
                tile_ds = create_tile_ds(*create_args)

                # Try to use COG overview if available
                cog_overview_used = False
                if use_cog_overviews:
                    zoom_diff = raster_geometry.zoom - frame.tile.z
                    reduction_factor = 2 ** zoom_diff
                    ovr_idx = find_matching_overview(src_bands[0], reduction_factor)

                    if ovr_idx is not None:
                        # Read directly from COG overview - preserves user's original resampling
                        logging.info(
                            "Read %s from COG overview %d (reduction %.1fx)",
                            frame.tile,
                            ovr_idx,
                            reduction_factor,
                        )
                        osgeo.gdal.Warp(
                            destNameOrDestDS=tile_ds,
                            srcDSOrSrcDSTab=src_ds,
                            options=osgeo.gdal.WarpOptions(
                                overviewLevel=ovr_idx,
                                resampleAlg=osgeo.gdal.GRA_NearestNeighbour,
                                srcNodata=nodata_value,
                                dstNodata=nodata_value,
                            ),
                        )
                        cog_overview_used = True

                if not cog_overview_used:
                    # Build overview from child tiles using VRT mosaic
                    # We must combine all children into a VRT first, then warp once
                    # Sequential warps would overwrite each other's data
                    logging.info("Overview %s from %d child tiles", frame.tile, len(frame.outputs))

                    if len(frame.outputs) == 1:
                        # Single child - warp directly
                        osgeo.gdal.Warp(
                            destNameOrDestDS=tile_ds,
                            srcDSOrSrcDSTab=frame.outputs[0],
                            options=opts_overview
                        )
                    else:
                        # Multiple children - build VRT mosaic first
                        vrt_path = f"/vsimem/overview-vrt-{frame.tile.z}-{frame.tile.x}-{frame.tile.y}.vrt"
                        vrt_options = osgeo.gdal.BuildVRTOptions(
                            srcNodata=nodata_value,
                            VRTNodata=nodata_value,
                        )
                        vrt_ds = osgeo.gdal.BuildVRT(vrt_path, frame.outputs, options=vrt_options)
                        if vrt_ds is None:
                            logging.warning("Failed to build VRT for %s, falling back to sequential warp", frame.tile)
                            for sub_ds in frame.outputs:
                                osgeo.gdal.Warp(
                                    destNameOrDestDS=tile_ds,
                                    srcDSOrSrcDSTab=sub_ds,
                                    options=opts_overview
                                )
                        else:
                            vrt_ds.FlushCache()
                            osgeo.gdal.Warp(
                                destNameOrDestDS=tile_ds,
                                srcDSOrSrcDSTab=vrt_ds,
                                options=opts_overview
                            )
                            vrt_ds = None
                            osgeo.gdal.Unlink(vrt_path)

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


def open_raster_in_process(
    input_filename: str,
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
    block_zoom: int,
    overview_mode: OverviewMode = OverviewMode.AUTO,
    min_zoom_override: int | None = None,
) -> tuple[RasterGeometry, multiprocessing.Pipe]:
    """Opens a bidirectional connection to a raster reader in another process.

    Args:
        input_filename: Path to raster file
        zoom_strategy: Web mercator zoom level selection
        resampling_algorithm: Resampling method to use
        block_zoom: Block zoom level (8 for 256px blocks)
        overview_mode: Overview generation mode (NONE=no overviews, AUTO=full pyramid)
        min_zoom_override: Optional minimum zoom level override

    Returns:
        Tuple of (raster_geometry, receive_pipe) for communication
    """
    # Create communication pipe
    parent_recv, child_send = multiprocessing.Pipe(duplex=False)

    # Start worker process
    args = (
        input_filename,
        zoom_strategy,
        resampling_algorithm,
        block_zoom,
        child_send,
        overview_mode,
        min_zoom_override,
    )
    process = multiprocessing.Process(target=read_raster, args=args)
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
    zoom: int, xmin: int, ymin: int, xmax: int, ymax: int, block_zoom: int, minresolution: int, block_count: int
) -> dict:
    """Get dictionary of dimensions for RaQuet v0.3.0 metadata from tile bounds"""
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
        "width": raster_width,
        "height": raster_height,
        "bounds": [nw.west, se.south, se.east, nw.north],
        "tiling": {
            "scheme": "quadbin",
            "block_width": block_width,
            "block_height": block_height,
            "min_zoom": minresolution,
            "max_zoom": zoom,
            "pixel_zoom": zoom + block_zoom,
            "num_blocks": block_count,
        },
    }


def create_schema(rg: RasterGeometry) -> tuple[pyarrow.lib.Schema, list[str]]:
    """Create table schema and band column names for RasterGeometry instance

    For time-series data (CF time present), creates a single band column since
    all bands represent the same variable at different time steps.
    For regular multi-band data, creates one column per band.
    """
    # Build column list
    columns = [
        ("block", pyarrow.uint64()),
        ("metadata", pyarrow.string()),
    ]

    # Add time columns if CF time dimension is present
    if rg.cf_time is not None:
        columns.append(("time_cf", pyarrow.float64()))
        columns.append(("time_ts", pyarrow.timestamp("us")))
        # For time-series, all bands are the same variable - use single column
        band_names = ["band_1"]
    else:
        # Standard multi-band: one column per band
        band_names = [f"band_{n}" for n in range(1, 1 + len(rg.bandtypes))]

    # Add band columns
    columns.extend((bname, pyarrow.binary()) for bname in band_names)

    # Create schema with Parquet-level metadata for file identification
    # This allows readers (e.g., GDAL drivers) to identify RaQuet files
    # by reading only the Parquet footer
    schema = pyarrow.schema(columns, metadata={"raquet:version": "0.3.0"})

    return schema, band_names


def _sanitize_nodata(nodata: float | int | None) -> float | int | str | None:
    """Encode nodata value for JSON following Zarr v3 conventions.

    Special floating-point values are encoded as strings:
    - NaN -> "NaN"
    - +Infinity -> "Infinity"
    - -Infinity -> "-Infinity"

    See: https://zarr-specs.readthedocs.io/en/latest/v3/data-types/index.html#fill-value-list
    """
    if nodata is None:
        return None
    if isinstance(nodata, float):
        if math.isnan(nodata):
            return "NaN"
        if math.isinf(nodata):
            return "Infinity" if nodata > 0 else "-Infinity"
    return nodata


def _create_band_metadata(
    band_idx: int,
    btype: str,
    bname: str,
    bcolorinterp: str,
    bcolortable: dict | None,
    stats: RasterStats | None,
    rg: RasterGeometry,
) -> dict:
    """Create v0.3.0 band metadata with GDAL-compatible statistics keys"""
    # Get per-band GDAL metadata
    nodata = None
    if rg.band_nodatas and band_idx < len(rg.band_nodatas):
        nodata = _sanitize_nodata(rg.band_nodatas[band_idx])

    description = ""
    if rg.band_descriptions and band_idx < len(rg.band_descriptions):
        description = rg.band_descriptions[band_idx] or ""

    unit = ""
    if rg.band_units and band_idx < len(rg.band_units):
        unit = rg.band_units[band_idx] or ""

    scale = None
    if rg.band_scales and band_idx < len(rg.band_scales):
        scale = rg.band_scales[band_idx]

    offset = None
    if rg.band_offsets and band_idx < len(rg.band_offsets):
        offset = rg.band_offsets[band_idx]

    band_meta = {
        "name": bname,
        "description": description,
        "type": btype,
        "nodata": nodata,
        "unit": unit,
        "scale": scale,
        "offset": offset,
        "colorinterp": bcolorinterp,
        "colortable": bcolortable,
    }

    # Add GDAL-compatible statistics if available
    if stats is not None:
        band_meta["STATISTICS_MINIMUM"] = stats.min
        band_meta["STATISTICS_MAXIMUM"] = stats.max
        band_meta["STATISTICS_MEAN"] = stats.mean
        band_meta["STATISTICS_STDDEV"] = stats.stddev
        band_meta["STATISTICS_VALID_PERCENT"] = stats.valid_percent

    return band_meta


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
    """Create a dictionary of RaQuet v0.3.0 metadata

    See https://github.com/CartoDB/raquet/blob/master/format-specs/raquet.md#metadata-specification
    """
    # For time-series data, we have one band column but stats from the first band
    # For regular data, we have one band per column
    if rg.cf_time is not None:
        # Time-series: single band representing the variable
        # Use first band's type/colorinterp/colortable, combine stats from all
        combined_stats = None
        for stats in band_stats:
            combined_stats = combine_stats(combined_stats, stats)

        bands_metadata = [
            _create_band_metadata(
                0,
                rg.bandtypes[0],
                "band_1",
                rg.bandcolorinterp[0],
                rg.bandcolortable[0],
                combined_stats,
                rg,
            )
        ]
    else:
        # Standard multi-band
        bands_metadata = [
            _create_band_metadata(
                i,
                btype,
                bname,
                bcolorinterp,
                bcolortable,
                stats,
                rg,
            )
            for i, (btype, bname, bcolorinterp, bcolortable, stats) in enumerate(
                zip(
                    rg.bandtypes,
                    band_names,
                    rg.bandcolorinterp,
                    rg.bandcolortable,
                    band_stats,
                )
            )
        ]

    # Get dimensions and tiling info
    dimensions = get_raquet_dimensions(
        rg.zoom, xmin, ymin, xmax, ymax, block_zoom, minresolution, block_count
    )

    metadata_json = {
        "file_format": "raquet",
        "version": "0.3.0",
        "width": dimensions["width"],
        "height": dimensions["height"],
        "crs": "EPSG:3857",
        "bounds": dimensions["bounds"],
        "bounds_crs": "EPSG:4326",
        "tiling": dimensions["tiling"],
        "compression": "gzip",
        "bands": bands_metadata,
    }

    # Add time metadata if CF time dimension is present
    if rg.cf_time is not None:
        cf = rg.cf_time
        time_meta = {
            "cf:units": cf.raw_units_string,
            "cf:calendar": cf.calendar,
            "interpretation": "period_start",
            "count": len(cf.values),
        }

        # Add resolution if we can determine it
        iso_duration = cf.to_iso_duration()
        if iso_duration:
            time_meta["resolution"] = iso_duration

        # Add range if we have values
        if cf.values:
            time_meta["range"] = [cf.values[0], cf.values[-1]]

        metadata_json["time"] = time_meta

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
    input_filename: str,
    raquet_destination: str,
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
    block_zoom: int,
    target_size: int | None = None,
    row_group_size: int = 200,
    overview_mode: OverviewMode = OverviewMode.AUTO,
    min_zoom_override: int | None = None,
    streaming: bool = False,
):
    """Read raster datasource and write to RaQuet

    Args:
        input_filename: Raster file path (GeoTIFF, NetCDF, or any GDAL-supported format)
        raquet_destination: RaQuet destination, file or dirname depending on target_size
        zoom_strategy: ZoomStrategy member
        resampling_algorithm: ResamplingAlgorithm member
        target_size: Integer number of bytes for individual parquet files in raquet_destination
        row_group_size: Number of rows per Parquet row group (default 200 for efficient pruning)
        overview_mode: Overview generation mode (NONE=no overviews, AUTO=full pyramid)
        min_zoom_override: Optional minimum zoom level override
        streaming: Use two-pass streaming mode for memory-safe conversion of large files
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
        input_filename,
        raquet_destinations,
        zoom_strategy,
        resampling_algorithm,
        block_zoom,
        row_group_size,
        overview_mode,
        min_zoom_override,
        streaming,
    ):
        logging.info("Wrote %s", raquet_filename)


def convert_to_raquet_files(
    input_filename: str,
    raquet_destinations: typing.Generator[tuple[str, float], None, None],
    zoom_strategy: ZoomStrategy,
    resampling_algorithm: ResamplingAlgorithm,
    block_zoom: int,
    row_group_size: int = 200,
    overview_mode: OverviewMode = OverviewMode.AUTO,
    min_zoom_override: int | None = None,
    streaming: bool = False,
) -> typing.Generator[str, None, None]:
    """Read raster datasource and write to RaQuet files

    Args:
        input_filename: Raster file path (GeoTIFF, NetCDF, or any GDAL-supported format)
        raquet_destinations: tuples of (filename, target sizeof)
        zoom_strategy: ZoomStrategy member
        resampling_algorithm: ResamplingAlgorithm member
        block_zoom: Block zoom level (8 for 256px blocks)
        row_group_size: Number of rows per Parquet row group (default 200 for efficient pruning)
        overview_mode: Overview generation mode (NONE=no overviews, AUTO=full pyramid)
        min_zoom_override: Optional minimum zoom level override
        streaming: Use two-pass streaming mode for memory-safe conversion of large files

    Yields output filenames as they are written.

    Note: Rows are sorted by block ID before writing to enable efficient
    row group pruning when querying. This allows Parquet readers to skip
    entire row groups based on block statistics, significantly reducing
    data transfer for remote file access.

    When streaming=True, tiles are written to a temporary file first, then
    sorted and written to the final output. This uses O(row_group_size) memory
    instead of O(all_tiles), making it suitable for large rasters.
    """
    raster_geometry, pipe = open_raster_in_process(
        input_filename,
        zoom_strategy,
        resampling_algorithm,
        block_zoom,
        overview_mode,
        min_zoom_override,
    )

    try:
        schema, band_names = create_schema(raster_geometry)

        # Initialize stats tracking (small, always in memory)
        band_stats = [None for _ in band_names]
        xmin, ymin, xmax, ymax = math.inf, math.inf, -math.inf, -math.inf
        minresolution, block_count = math.inf, 0

        # Streaming mode: write tiles to temp file, then sort
        # Non-streaming mode: accumulate in memory (faster for small files)
        if streaming:
            logging.info("Streaming mode: writing tiles to temporary file")
            temp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
            temp_path = temp_file.name
            temp_file.close()

            # Create temp writer without sorting metadata (will sort later)
            temp_writer = pyarrow.parquet.ParquetWriter(temp_path, schema)
            temp_rows = []  # Small buffer for batching writes
        else:
            # Non-streaming: accumulate all rows in memory
            all_rows = []
            temp_path = None
            temp_writer = None
            temp_rows = None

        while True:
            received = pipe.recv()
            if received is None:
                # Use a signal value to stop expecting further tiles
                break

            # Expand message to block to retrieve
            tile, block_data, tile_stats = received
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

            # Build row(s) for this tile
            rows_to_add = []
            if raster_geometry.cf_time is not None and raster_geometry.band_time_values:
                # Time-series mode: each band is a different time step
                cf = raster_geometry.cf_time
                for band_idx, data in enumerate(block_data):
                    cf_value = raster_geometry.band_time_values[band_idx]
                    if cf_value is None:
                        continue
                    ts_value = cf_to_timestamp(cf_value, cf)
                    rows_to_add.append({
                        "block": quadbin.tile_to_cell((tile.x, tile.y, tile.z)),
                        "metadata": None,
                        "time_cf": cf_value,
                        "time_ts": ts_value,
                        "band_1": data,
                    })
            else:
                # Standard mode: all bands in single row
                rows_to_add.append({
                    "block": quadbin.tile_to_cell((tile.x, tile.y, tile.z)),
                    "metadata": None,
                    **{bname: block_data[i] for i, bname in enumerate(band_names)},
                })

            # Add rows to appropriate destination
            if streaming:
                temp_rows.extend(rows_to_add)
                # Flush to temp file periodically to keep memory low
                if len(temp_rows) >= row_group_size:
                    flush_rows_to_file(temp_writer, schema, temp_rows)
                    temp_rows = []
            else:
                all_rows.extend(rows_to_add)

            if tile.z > raster_geometry.zoom:
                raise NotImplementedError("Zoom higher than expected by find_minzoom()")

            if tile.z == raster_geometry.zoom:
                # Calculate bounds and count blocks only at the highest zoom level
                xmin, ymin = min(xmin, tile.x), min(ymin, tile.y)
                xmax, ymax = max(xmax, tile.x), max(ymax, tile.y)
                block_count += 1

            if any(tile_stats):
                # Accumulate band statistics and real bounds based on included tiles
                band_stats = [combine_stats(*bs) for bs in zip(band_stats, tile_stats)]

        for i, stats in enumerate(band_stats):
            logging.info("Band %s %s", i + 1, stats)

        # Handle streaming vs non-streaming finalization
        if streaming:
            # Flush remaining rows to temp file
            if temp_rows:
                flush_rows_to_file(temp_writer, schema, temp_rows)
            temp_writer.close()

            # Read temp file and sort
            logging.info("Streaming: reading and sorting temporary file...")
            temp_table = pyarrow.parquet.read_table(temp_path)

            # Sort by block ID (and time for time-series)
            if raster_geometry.cf_time is not None:
                logging.info("Sorting %d rows by (block, time_cf)...", len(temp_table))
                sort_keys = [("block", "ascending"), ("time_cf", "ascending")]
            else:
                logging.info("Sorting %d rows by block ID...", len(temp_table))
                sort_keys = [("block", "ascending")]

            sorted_indices = pyarrow.compute.sort_indices(temp_table, sort_keys=sort_keys)
            sorted_table = temp_table.take(sorted_indices)

            # Clean up temp file
            os.unlink(temp_path)
            temp_path = None

            # Convert sorted table back to row iterator for writing
            all_rows = sorted_table.to_pylist()
            del sorted_table, temp_table  # Free memory
        else:
            # Non-streaming: sort in-memory rows
            if raster_geometry.cf_time is not None:
                logging.info("Sorting %d rows by (block, time_cf) for optimized row group pruning...", len(all_rows))
                all_rows.sort(key=lambda row: (row["block"], row.get("time_cf", 0)))
            else:
                logging.info("Sorting %d rows by block ID for optimized row group pruning...", len(all_rows))
                all_rows.sort(key=lambda row: row["block"])

        # Build metadata now (we have all the info we need)
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
        # Add time columns to metadata row if CF time is present
        if raster_geometry.cf_time is not None:
            metadata_row["time_cf"] = None
            metadata_row["time_ts"] = None

        # Now write sorted rows, splitting into multiple files if target_sizeof is set
        # Use page index for finer-grained filtering and sorting metadata
        from pyarrow.parquet import SortingColumn
        rfname, target_sizeof = next(raquet_destinations)
        writer = pyarrow.parquet.ParquetWriter(
            rfname,
            schema,
            write_page_index=True,  # Enable page-level column indexes
            sorting_columns=[SortingColumn(0)],  # Column 0 (block) is sorted
        )
        sizeof_so_far = 0
        rows_in_current_file = []

        for row in all_rows:
            rows_in_current_file.append(row)
            # Estimate size using memory footprint (like original code)
            sizeof_so_far += sum(sys.getsizeof(v) for v in row.values())

            # Write a row group when we hit the batch size limit
            if len(rows_in_current_file) >= row_group_size:
                flush_rows_to_file(writer, schema, rows_in_current_file)
                rows_in_current_file = []
                sizeof_so_far = os.stat(rfname).st_size

            # Split to new file when we hit the sizeof limit
            if sizeof_so_far > target_sizeof:
                if rows_in_current_file:
                    flush_rows_to_file(writer, schema, rows_in_current_file)
                    rows_in_current_file = []
                writer.close()
                yield rfname

                # Start new file
                rfname, target_sizeof = next(raquet_destinations)
                writer = pyarrow.parquet.ParquetWriter(
                    rfname,
                    schema,
                    write_page_index=True,
                    sorting_columns=[SortingColumn(0)],
                )
                sizeof_so_far = 0

        # Flush remaining rows
        if rows_in_current_file:
            flush_rows_to_file(writer, schema, rows_in_current_file)

        # Add metadata row and close final file
        flush_rows_to_file(writer, schema, [metadata_row])
        writer.close()
        yield rfname

    finally:
        pipe.close()


parser = argparse.ArgumentParser()
parser.add_argument("input_filename")
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
        args.input_filename,
        args.raquet_destination,
        ZoomStrategy(args.zoom_strategy),
        ResamplingAlgorithm(args.resampling_algorithm),
        block_zoom,
        args.target_size,
    )
