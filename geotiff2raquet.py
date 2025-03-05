#!/usr/bin/env python3
"""Convert GeoTIFF file to RaQuet output

Usage:
    geotiff2raquet.py <geotiff_filename> <raquet_filename>

Required packages:
    - GDAL <https://pypi.org/project/GDAL/>
    - mercantile <https://pypi.org/project/mercantile/>
    - pyarrow <https://pypi.org/project/pyarrow/>
"""
import argparse
import dataclasses
import gzip
import itertools
import json
import logging
import multiprocessing

import mercantile
import pyarrow.compute
import pyarrow.parquet

EARTH_DIAMETER = mercantile.CE
SCALE_PRECISION = 11


@dataclasses.dataclass
class RasterGeometry:
    bands: int
    width: int
    height: int
    zoom: int
    xoff: float
    xres: float
    gt2: float
    yoff: float
    gt4: float
    yres: float


def interleave_quadkey(z: int, x: int, y: int) -> int:
    """Convert z/x/y tile coordinates into a quadkey integer.

    Args:
        z: Zoom level
        x: Tile X coordinate
        y: Tile Y coordinate

    Returns:
        64-bit integer quadkey value
    """
    # Left shift x,y based on zoom level
    x = x << (32 - z)
    y = y << (32 - z)

    # Interleave bits
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    y = (y | (y << 16)) & 0x0000FFFF0000FFFF

    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    y = (y | (y << 8)) & 0x00FF00FF00FF00FF

    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F

    x = (x | (x << 2)) & 0x3333333333333333
    y = (y | (y << 2)) & 0x3333333333333333

    x = (x | (x << 1)) & 0x5555555555555555
    y = (y | (y << 1)) & 0x5555555555555555

    # Combine components into final quadkey
    quadkey = (
        0x4000000000000000  # Base value
        | (1 << 59)  # Mode
        | (z << 52)  # Zoom level
        | ((x | (y << 1)) >> 12)  # Interleaved coordinates
        | (0xFFFFFFFFFFFFF >> (z * 2))  # Mask
    )

    return quadkey


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

    for x, y in itertools.product(range(ulx, lrx), range(uly, lry - 1)):
        yield mercantile.Tile(x, y, rg.zoom)


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

    ds = osgeo.gdal.Open(geotiff_filename)
    sref = ds.GetSpatialRef()
    _, xres, *_, yres = ds.GetGeoTransform()

    web_mercator = osgeo.osr.SpatialReference()
    web_mercator.ImportFromEPSG(3857)
    if sref.ExportToProj4() != web_mercator.ExportToProj4():
        raise ValueError("Source SRS is not EPSG:3857")

    valid_scales = [round(EARTH_DIAMETER / (2**i), SCALE_PRECISION) for i in range(32)]
    if round(-yres, SCALE_PRECISION) not in valid_scales:
        raise ValueError(f"Vertical pixel size {-yres} is not a valid scale")
    if round(xres, SCALE_PRECISION) not in valid_scales:
        raise ValueError(f"Horizontal pixel size {xres} is not a valid scale")

    zoom = valid_scales.index(round(xres, SCALE_PRECISION)) - 8

    raster_geometry = RasterGeometry(
        ds.RasterCount,
        ds.RasterXSize,
        ds.RasterYSize,
        zoom,
        *ds.GetGeoTransform(),
    )

    pipe_out.send(raster_geometry)

    while True:
        try:
            i, txoff, tyoff, txsize, tysize = pipe_in.recv()

            logging.info(
                "Band %s nodata value: %s", i, ds.GetRasterBand(i).GetNoDataValue()
            )

            # Read raster data for this tile
            band = ds.GetRasterBand(i)
            logging.info(
                "txoff %s tyoff %s txsize %s tysize %s", txoff, tyoff, txsize, tysize
            )
            data = band.ReadRaster(txoff, tyoff, txsize, tysize)
            logging.info("Read %s bytes from band %s: %s...", len(data), i, data[:12])

            pipe_out.send((data,))
        except EOFError:
            break

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


def main(geotiff_filename, raquet_filename):
    """Read GeoTIFF datasource and write to a RaQuet file

    Args:
        geotiff_filename: GeoTIFF filename
        raquet_filename: RaQuet filename
    """
    raster_geometry, pipe_send, pipe_recv = open_geotiff_in_process(geotiff_filename)
    assert raster_geometry.gt2 == 0.0 and raster_geometry.gt4 == 0.0, "Expect no skew"
    band_names = [f"band_{i}" for i in range(1, 1 + raster_geometry.bands)]

    # Create table schema based on band count
    fields = [
        ("block", pyarrow.uint64()),
        ("metadata", pyarrow.string()),
        *[(band_name, pyarrow.binary()) for band_name in band_names],
    ]

    schema = pyarrow.schema(fields)

    # Create empty table
    table = pyarrow.Table.from_pydict({name: [] for name, _ in fields}, schema=schema)

    for tile in generate_tiles(raster_geometry):
        logging.info(
            "Tile z=%s x=%s y=%s quadkey=%s bounds=%s",
            tile.z,
            tile.x,
            tile.y,
            hex(interleave_quadkey(tile.z, tile.x, tile.y)),
            mercantile.xy_bounds(tile),
        )
        xmin, ymin, xmax, ymax = mercantile.xy_bounds(tile)

        # Convert mercator coordinates to pixel coordinates
        txoff = int(round((xmin - raster_geometry.xoff) / raster_geometry.xres))
        tyoff = int(round((raster_geometry.yoff - ymin) / -raster_geometry.yres))
        txsize = int(round((xmax - xmin) / raster_geometry.xres))
        tysize = int(round((ymax - ymin) / -raster_geometry.yres))

        band_data = []

        for i in range(1, 1 + raster_geometry.bands):
            pipe_send.send((i, txoff, tyoff, txsize, tysize))
            (data,) = pipe_recv.recv()

            # Append data to list for this band
            band_data.append(data)

        # Create new row in table after processing all bands
        quadkey = interleave_quadkey(tile.z, tile.x, tile.y)
        tile_row = pyarrow.Table.from_pydict(
            {
                "block": [quadkey],
                "metadata": [None],
                **{band_name: [band_data[i]] for i, band_name in enumerate(band_names)},
            },
            schema=schema,
        )

        table = pyarrow.concat_tables([table, tile_row])

    metadata_json = json.dumps(
        {
            "bounds": [None, None, None, None],
            "compression": "gzip",
            "width": raster_geometry.width,
            "height": raster_geometry.height,
            "minresolution": None,
            "maxresolution": raster_geometry.zoom,
            "block_width": None,
            "block_height": None,
            "bands": [{"type": None, "name": band_name} for band_name in band_names],
        }
    )

    metadata_row = pyarrow.Table.from_pydict(
        {
            "block": [0],
            "metadata": [json.dumps(metadata_json)],
            **{band_name: [None] for band_name in band_names},
        },
        schema=schema,
    )

    table = pyarrow.concat_tables([table, metadata_row])

    # Write table to parquet file
    # print(table)
    pyarrow.parquet.write_table(table, raquet_filename)


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
