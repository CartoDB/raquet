#!/usr/bin/env python3
"""Convert GeoTIFF file to RaQuet output

Usage:
    geotiff2raquet.py <geotiff_filename> <raquet_filename>

Required packages:
    - GDAL <https://pypi.org/project/GDAL/>
    - mercantile <https://pypi.org/project/mercantile/>
    - pyarrow <https://pypi.org/project/pyarrow/>
    - quadbin <https://pypi.org/project/quadbin/>

>>> import tempfile; _, raquet_tempfile = tempfile.mkstemp(suffix=".parquet")

Test case "europe.tif"

>>> main("examples/europe.tif", raquet_tempfile)
>>> table1 = pyarrow.parquet.read_table(raquet_tempfile)
>>> len(table1)
17

>>> table1.column_names
['block', 'metadata', 'band_1', 'band_2', 'band_3', 'band_4']

>>> metadata1 = read_metadata(table1)
>>> [metadata1[k] for k in ["compression", "width", "height", "minresolution", "maxresolution"]]
['gzip', 1024, 1024, 5, 5]

>>> [round(b, 8) for b in metadata1["bounds"]]
[0.0, 40.97989807, 45.0, 66.51326044]

>>> [b["name"] for b in metadata1["bands"]]
['band_1', 'band_2', 'band_3', 'band_4']

Test case "san-francisco.tif"

>>> main("examples/san-francisco.tif", raquet_tempfile)
>>> table2 = pyarrow.parquet.read_table(raquet_tempfile)
>>> len(table2)
5

>>> table2.column_names
['block', 'metadata', 'band_1']

>>> metadata2 = read_metadata(table2)
>>> [metadata2[k] for k in ["compression", "width", "height", "minresolution", "maxresolution"]]
['gzip', 266, 362, 11, 11]

>>> [round(b, 8) for b in metadata2["bounds"]]
[-122.6953125, 37.57941251, -122.34375, 37.85750716]

>>> [b["name"] for b in metadata2["bands"]]
['band_1']

"""

import argparse
import dataclasses
import gzip
import itertools
import json
import logging
import math
import multiprocessing
import struct

import mercantile
import pyarrow.compute
import pyarrow.parquet
import quadbin

EARTH_DIAMETER = mercantile.CE
SCALE_PRECISION = 11


@dataclasses.dataclass
class RasterGeometry:
    """Convenience wrapper for details of raster geometry and transformation"""

    bands: int
    width: int
    height: int
    zoom: int
    minlat: float
    minlon: float
    maxlat: float
    maxlon: float
    xoff: float
    xres: float
    gt2: float
    yoff: float
    gt4: float
    yres: float


@dataclasses.dataclass
class BandType:
    """Convenience wrapper for details of band data type"""

    fmt: str
    size: int
    typ: type


def generate_tiles(rg: RasterGeometry):
    """Generate tiles for a given zoom level

    Args:
        rg: RasterGeometry instance

    Returns:
        Generator of tiles
    """
    logging.info("xoff %s yoff %s zoom %s", rg.xoff, rg.yoff, rg.zoom)

    tileres = EARTH_DIAMETER / 2**rg.zoom
    ulx = int((rg.xoff + EARTH_DIAMETER / 2) / tileres)
    uly = int((EARTH_DIAMETER / 2 - rg.yoff) / tileres)
    logging.info("tileres %s ulx %s uly %s", tileres, ulx, uly)

    lrx = int((rg.xoff + rg.width * rg.xres + EARTH_DIAMETER / 2) / tileres)
    lry = int((EARTH_DIAMETER / 2 - (rg.yoff + rg.height * rg.yres)) / tileres)
    logging.info("lrx %s lry %s", lrx, lry)

    for x, y in itertools.product(range(ulx, lrx + 1), range(uly, lry + 1)):
        yield mercantile.Tile(x, y, rg.zoom)


def read_rasterband(
    band: "osgeo.gdal.Band",  # noqa: F821 (Band type safely imported in read_geotiff)
    bbox: tuple[int, int, int, int],
    xpads: tuple[int, int],
    ypads: tuple[int, int],
    band_type: BandType,
) -> bytes:
    """Return uncompressed raster bytes padded to full tile size.

    Acts like numpy.pad() without requiring numpy dependency
    """
    data1 = band.ReadRaster(*bbox)

    # Return early if no padding to be done
    if (xpads, ypads) == ((0, 0), (0, 0)):
        return data1

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
    return b"".join(data3)


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

    gdaltype_bandtypes: dict[int, BandType] = {
        osgeo.gdal.GDT_Byte: BandType("B", 1, int),
        osgeo.gdal.GDT_CFloat32: BandType("f", 4, float),
        osgeo.gdal.GDT_CFloat64: BandType("d", 8, float),
        osgeo.gdal.GDT_CInt16: BandType("h", 2, int),
        osgeo.gdal.GDT_CInt32: BandType("i", 4, int),
        osgeo.gdal.GDT_Float32: BandType("f", 4, float),
        osgeo.gdal.GDT_Float64: BandType("d", 8, float),
        osgeo.gdal.GDT_Int16: BandType("h", 2, int),
        osgeo.gdal.GDT_Int32: BandType("i", 4, int),
        osgeo.gdal.GDT_Int64: BandType("q", 8, int),
        osgeo.gdal.GDT_Int8: BandType("b", 1, int),
        osgeo.gdal.GDT_UInt16: BandType("H", 2, int),
        osgeo.gdal.GDT_UInt32: BandType("I", 4, int),
        osgeo.gdal.GDT_UInt64: BandType("Q", 8, int),
    }

    try:
        ds = osgeo.gdal.Open(geotiff_filename)
        sref = ds.GetSpatialRef()
        xmin, xres, _, ymax, _, yres = ds.GetGeoTransform()

        web_mercator = osgeo.osr.SpatialReference()
        web_mercator.ImportFromEPSG(3857)
        if sref.ExportToProj4() != web_mercator.ExportToProj4():
            raise ValueError("Source SRS is not EPSG:3857")

        valid_scales = [
            round(EARTH_DIAMETER / (2**i), SCALE_PRECISION) for i in range(32)
        ]
        if round(-yres, SCALE_PRECISION) not in valid_scales:
            raise ValueError(f"Vertical pixel size {-yres} is not a valid scale")
        if round(xres, SCALE_PRECISION) not in valid_scales:
            raise ValueError(f"Horizontal pixel size {xres} is not a valid scale")

        zoom = valid_scales.index(round(xres, SCALE_PRECISION)) - 8
        xmax = xmin + ds.RasterXSize * xres
        ymin = ymax + ds.RasterYSize * yres

        wgs84 = osgeo.osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)  # WGS84
        wgs84.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        transform = osgeo.osr.CoordinateTransformation(web_mercator, wgs84)

        minlon, minlat, _ = transform.TransformPoint(xmin, ymin)
        maxlon, maxlat, _ = transform.TransformPoint(xmax, ymax)

        raster_geometry = RasterGeometry(
            ds.RasterCount,
            ds.RasterXSize,
            ds.RasterYSize,
            zoom,
            minlat,
            minlon,
            maxlat,
            maxlon,
            *ds.GetGeoTransform(),
        )

        pipe_out.send(raster_geometry)

        while True:
            try:
                received = pipe_in.recv()
                if received is None:
                    # Use this signal value because pipe.close() doesn't raise EOFError here on Linux
                    raise EOFError

                # Expand message to an intended raster area to retrieve
                i, tile = received

                # Convert mercator coordinates to pixel coordinates
                xmin, ymin, xmax, ymax = mercantile.xy_bounds(tile)
                txoff = int(round((xmin - raster_geometry.xoff) / raster_geometry.xres))
                tyoff = int(
                    round((raster_geometry.yoff - ymax) / -raster_geometry.yres)
                )
                txsize = int(round((xmax - xmin) / raster_geometry.xres))
                tysize = int(round((ymax - ymin) / -raster_geometry.yres))
                expected_count = tysize * txsize

                # Respond with an empty block if requested area is too far east or north
                if txoff >= ds.RasterXSize or tyoff >= ds.RasterYSize:
                    pipe_out.send((None,))
                    continue

                # Adjust raster area to within valid values
                xpad_before, xpad_after, ypad_before, ypad_after = 0, 0, 0, 0
                if txoff < 0:
                    txoff, txsize, xpad_before = 0, txsize + txoff, -txoff
                if tyoff < 0:
                    tyoff, tysize, ypad_before = 0, tysize + tyoff, -tyoff
                if txoff + txsize > ds.RasterXSize:
                    xpad_after = txsize - (ds.RasterXSize - txoff)
                    txsize = ds.RasterXSize - txoff
                if tyoff + tysize > ds.RasterYSize:
                    ypad_after = tysize - (ds.RasterYSize - tyoff)
                    tysize = ds.RasterYSize - tyoff

                # Respond with an empty block if requested area is zero width or height
                if txsize == 0 or tysize == 0:
                    pipe_out.send((None,))
                    continue

                logging.info(
                    "Band %s nodata value: %s", i, ds.GetRasterBand(i).GetNoDataValue()
                )

                # Read raster data for this tile
                band = ds.GetRasterBand(i)
                logging.info(
                    "txoff %s tyoff %s txsize %s tysize %s",
                    txoff,
                    tyoff,
                    txsize,
                    tysize,
                )

                # Pad to full tile size with NODATA fill if needed
                bbox = (txoff, tyoff, txsize, tysize)
                xpads, ypads = (xpad_before, xpad_after), (ypad_before, ypad_after)
                band_type = gdaltype_bandtypes[band.DataType]
                data = read_rasterband(band, bbox, xpads, ypads, band_type)
                logging.info(
                    "Read %s bytes from band %s: %s...", len(data), i, data[:12]
                )
                assert len(data) == expected_count * band_type.size

                pipe_out.send((gzip.compress(data),))
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


def main(geotiff_filename, raquet_filename):
    """Read GeoTIFF datasource and write to a RaQuet file

    Args:
        geotiff_filename: GeoTIFF filename
        raquet_filename: RaQuet filename
    """
    raster_geometry, pipe_send, pipe_recv = open_geotiff_in_process(geotiff_filename)

    try:
        assert raster_geometry.gt2 == 0 and raster_geometry.gt4 == 0, "Expect no skew"
        band_names = [f"band_{i}" for i in range(1, 1 + raster_geometry.bands)]

        # Create table schema based on band count
        schema = pyarrow.schema(
            [
                ("block", pyarrow.uint64()),
                ("metadata", pyarrow.string()),
                *[(bname, pyarrow.binary()) for bname in band_names],
            ]
        )

        # Initialize table with no rows
        table = pyarrow.Table.from_pydict(
            {fname: [] for fname in schema.names}, schema=schema
        )

        minlat, minlon, maxlat, maxlon = math.inf, math.inf, -math.inf, -math.inf

        for tile in generate_tiles(raster_geometry):
            logging.info(
                "Tile z=%s x=%s y=%s quadkey=%s bounds=%s",
                tile.z,
                tile.x,
                tile.y,
                hex(quadbin.tile_to_cell((tile.x, tile.y, tile.z))),
                mercantile.bounds(tile),
            )

            block_data = []

            for i in range(1, 1 + raster_geometry.bands):
                pipe_send.send((i, tile))
                (block_datum,) = pipe_recv.recv()

                # Append data to list for this band
                block_data.append(block_datum)

            if all(block_datum is None for block_datum in block_data):
                continue

            # Create new row in table after processing all bands
            tile_row = pyarrow.Table.from_pydict(
                {
                    "block": [quadbin.tile_to_cell((tile.x, tile.y, tile.z))],
                    "metadata": [None],
                    **{bname: [block_data[i]] for i, bname in enumerate(band_names)},
                },
                schema=schema,
            )
            table = pyarrow.concat_tables([table, tile_row])

            # Accumulate real bounds based on included tiles
            ll_bounds = mercantile.bounds(tile)
            minlat, minlon = min(minlat, ll_bounds.south), min(minlon, ll_bounds.west)
            maxlat, maxlon = max(maxlat, ll_bounds.north), max(maxlon, ll_bounds.east)

        # Prepend metadata row to the complete table
        metadata_json = json.dumps(
            {
                "bounds": [minlon, minlat, maxlon, maxlat],
                "compression": "gzip",
                "width": raster_geometry.width,
                "height": raster_geometry.height,
                "minresolution": raster_geometry.zoom,
                "maxresolution": raster_geometry.zoom,
                "block_width": None,
                "block_height": None,
                "bands": [{"type": None, "name": bname} for bname in band_names],
            }
        )
        metadata_row = pyarrow.Table.from_pydict(
            {
                "block": [0],
                "metadata": [metadata_json],
                **{bname: [None] for bname in band_names},
            },
            schema=schema,
        )
        table = pyarrow.concat_tables([metadata_row, table])

        # Write table to parquet file
        pyarrow.parquet.write_table(table, raquet_filename)

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
