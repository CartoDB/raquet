#!/usr/bin/env python3
"""Convert GeoTIFF file to RaQuet output

Usage:
    geotiff2raquet.py <geotiff_filename> <raquet_filename>

Required packages:
    - GDAL <https://pypi.org/project/GDAL/>
    - mercantile <https://pypi.org/project/mercantile/>
    - pyarrow <https://pypi.org/project/pyarrow/>
    - quadbin <https://pypi.org/project/quadbin/>

>>> import tempfile; _, raquet_filename = tempfile.mkstemp(suffix=".raquet")
>>> main("examples/europe.tif", raquet_filename)
>>> table = pyarrow.parquet.read_table(raquet_filename)
>>> metadata = read_metadata(table)
>>> [metadata[k] for k in ["compression", "width", "height", "minresolution", "maxresolution"]]
['gzip', 1024, 1024, 5, 5]

>>> [b["name"] for b in metadata["bands"]]
['band_1', 'band_2', 'band_3', 'band_4']

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
import quadbin

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

    osgeo.gdal.UseExceptions()

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

        # Initialize table with just metadata at block=0
        metadata_json = json.dumps(
            {
                "bounds": [None, None, None, None],
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
        table = pyarrow.Table.from_pydict(
            {
                "block": [0],
                "metadata": [metadata_json],
                **{bname: [None] for bname in band_names},
            },
            schema=schema,
        )

        for tile in generate_tiles(raster_geometry):
            logging.info(
                "Tile z=%s x=%s y=%s quadkey=%s bounds=%s",
                tile.z,
                tile.x,
                tile.y,
                hex(quadbin.tile_to_cell((tile.x, tile.y, tile.z))),
                mercantile.xy_bounds(tile),
            )
            xmin, ymin, xmax, ymax = mercantile.xy_bounds(tile)

            # Convert mercator coordinates to pixel coordinates
            txoff = int(round((xmin - raster_geometry.xoff) / raster_geometry.xres))
            tyoff = int(round((raster_geometry.yoff - ymin) / -raster_geometry.yres))
            txsize = int(round((xmax - xmin) / raster_geometry.xres))
            tysize = int(round((ymax - ymin) / -raster_geometry.yres))

            block_data = []

            for i in range(1, 1 + raster_geometry.bands):
                pipe_send.send((i, txoff, tyoff, txsize, tysize))
                (block_datum,) = pipe_recv.recv()

                # Append data to list for this band
                block_data.append(block_datum)

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

        # Write table to parquet file
        pyarrow.parquet.write_table(table, raquet_filename)

    except:
        pipe_send.close()
        pipe_recv.close()
        raise


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
