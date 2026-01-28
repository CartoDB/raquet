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


def get_metadata_compat(metadata: dict, key: str, default=None):
    """Get metadata value with v0.2.0/v0.3.0 compatibility.

    v0.3.0 moved some fields into nested 'tiling' object.
    """
    # v0.3.0 tiling fields
    tiling_keys = {
        "block_width": "block_width",
        "block_height": "block_height",
        "minresolution": "min_zoom",
        "maxresolution": "max_zoom",
    }

    # Check v0.3.0 tiling object first
    if "tiling" in metadata and key in tiling_keys:
        tiling = metadata["tiling"]
        v3_key = tiling_keys[key]
        if v3_key in tiling:
            return tiling[v3_key]

    # Fall back to v0.2.0 direct access
    return metadata.get(key, default)


def write_geotiff(
    metadata: dict,
    geotiff_filename: str,
    pipe: multiprocessing.Pipe,
    include_overviews: bool = False,
):
    """Worker process that writes a GeoTIFF through pipes.

    Args:
        metadata: dictionary of RaQuet metadata
        geotiff_filename: Name of GeoTIFF file to write
        pipe: Connection to receive data from parent
        include_overviews: If True, write RaQuet overview tiles to GeoTIFF overviews
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

        # Get block dimensions (compatible with v0.2.0 and v0.3.0)
        block_width = get_metadata_compat(metadata, "block_width")
        block_height = get_metadata_compat(metadata, "block_height")
        max_zoom = get_metadata_compat(metadata, "maxresolution")
        min_zoom = get_metadata_compat(metadata, "minresolution")

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
                f"BLOCKXSIZE={block_width}",
                f"BLOCKYSIZE={block_height}",
            ],
        )

        # Set projection
        raster.SetProjection(srs.ExportToWkt())

        # Set geotransform for base resolution
        xres = (xmax - xmin) / output_width
        yres = (ymax - ymin) / output_height
        geotransform = (xmin, xres, 0, ymax, 0, -yres)
        raster.SetGeoTransform(geotransform)

        # If including overviews, calculate overview levels from zoom range
        overview_factors = []
        if include_overviews and min_zoom is not None and max_zoom is not None:
            for z in range(max_zoom - 1, min_zoom - 1, -1):
                factor = 2 ** (max_zoom - z)
                overview_factors.append(factor)

            if overview_factors:
                logging.info(f"Creating overview levels: {overview_factors}")
                # Create empty overview structure using NONE resampling
                # We'll overwrite with actual RaQuet tiles
                raster.BuildOverviews("NEAREST", overview_factors)

        # Collect overview tiles to write after base tiles
        overview_tiles = []

        while True:
            received = pipe.recv()
            if received is None:
                # Use a signal value to stop expecting further tiles
                break

            tile, *block_data = received

            # Check if this is a base tile or overview tile
            if tile.z == max_zoom:
                # Base resolution tile - write directly to raster
                ulx, _, _, uly = mercantile.xy_bounds(tile)
                xoff = int(round((ulx - geotransform[0]) / geotransform[1]))
                yoff = int(round((uly - geotransform[3]) / geotransform[5]))

                for i, block_datum in enumerate(block_data):
                    band = raster.GetRasterBand(i + 1)

                    if metadata.get("nodata") is not None and band.GetNoDataValue() is None:
                        band.SetNoDataValue(metadata["nodata"])

                    # Handle per-band nodata (v0.3.0)
                    band_meta = metadata.get("bands", [{}])[i]
                    if band_meta.get("nodata") is not None and band.GetNoDataValue() is None:
                        band.SetNoDataValue(band_meta["nodata"])

                    if band_meta.get("colortable") is not None and band.GetColorTable() is None:
                        color_dict = band_meta["colortable"]
                        colorTable = osgeo.gdal.ColorTable()
                        for index, rgba in color_dict.items():
                            colorTable.SetColorEntry(int(index), tuple(rgba))
                        band.SetColorTable(colorTable)

                    if band_meta.get("colorinterp") is not None:
                        band.SetColorInterpretation(
                            GDAL_COLOR_INTERP[band_meta["colorinterp"]]
                        )

                    band.WriteRaster(xoff, yoff, block_width, block_height, block_datum)

            elif include_overviews and tile.z < max_zoom:
                # Overview tile - save for later
                overview_tiles.append((tile, block_data))

        # Write overview tiles
        if overview_tiles:
            logging.info(f"Writing {len(overview_tiles)} overview tiles...")
            for tile, block_data in overview_tiles:
                # Find which overview level this zoom corresponds to
                factor = 2 ** (max_zoom - tile.z)
                ovr_index = overview_factors.index(factor) if factor in overview_factors else -1

                if ovr_index < 0:
                    continue

                # Calculate pixel coordinates at this overview level
                ovr_width = output_width // factor
                ovr_height = output_height // factor

                # Skip if overview is smaller than block size (can't fit full tiles)
                if ovr_width < block_width or ovr_height < block_height:
                    logging.debug(
                        f"Skipping overview z={tile.z} - image {ovr_width}x{ovr_height} "
                        f"smaller than block {block_width}x{block_height}"
                    )
                    continue

                ovr_xres = (xmax - xmin) / ovr_width
                ovr_yres = (ymax - ymin) / ovr_height

                ulx, _, _, uly = mercantile.xy_bounds(tile)
                xoff = int(round((ulx - xmin) / ovr_xres))
                yoff = int(round((ymax - uly) / ovr_yres))

                # Clamp write dimensions to overview bounds
                write_width = min(block_width, ovr_width - xoff)
                write_height = min(block_height, ovr_height - yoff)

                if write_width <= 0 or write_height <= 0:
                    continue

                for i, block_datum in enumerate(block_data):
                    band = raster.GetRasterBand(i + 1)
                    ovr_band = band.GetOverview(ovr_index)
                    if ovr_band is not None:
                        # If we need partial write, we'd need to decode/re-encode
                        # For now, only write full blocks that fit
                        if write_width == block_width and write_height == block_height:
                            ovr_band.WriteRaster(xoff, yoff, block_width, block_height, block_datum)

    finally:
        pipe.close()


def open_geotiff_in_process(
    metadata: dict, geotiff_filename: str, include_overviews: bool = False
):
    """Opens a bidirectional connection to a GeoTIFF writer in another process.

    Args:
        metadata: RaQuet metadata dictionary
        geotiff_filename: Output GeoTIFF path
        include_overviews: If True, include RaQuet overviews in GeoTIFF

    Returns:
        Pipe for sending tile data to the writer process
    """
    # Create a communication pipe
    child_recv, parent_send = multiprocessing.Pipe(duplex=False)

    # Start worker process
    process = multiprocessing.Process(
        target=write_geotiff,
        args=(metadata, geotiff_filename, child_recv, include_overviews),
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


def main(raquet_filename, geotiff_filename, include_overviews: bool = False):
    """Read RaQuet file and write to a GeoTIFF datasource

    Args:
        raquet_filename: RaQuet filename
        geotiff_filename: GeoTIFF filename
        include_overviews: If True, include RaQuet overviews in the output GeoTIFF
    """
    table = pyarrow.parquet.read_table(raquet_filename)

    # Sort by block column
    table = table.sort_by("block")
    metadata = read_metadata(table)

    # Get max zoom (compatible with v0.2.0 and v0.3.0)
    max_zoom = get_metadata_compat(metadata, "maxresolution")

    pipe = open_geotiff_in_process(metadata, geotiff_filename, include_overviews)

    try:
        # Process table one row at a time
        for i in range(len(table)):
            if i > 0 and i % 1000 == 0:
                logging.info(f"Processed {i} of {len(table)} blocks...")
            block = table.column("block")[i].as_py()
            if block == 0:
                continue
            x, y, z = quadbin.cell_to_tile(block)

            # Skip tiles not at max zoom unless including overviews
            if z != max_zoom and not include_overviews:
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
