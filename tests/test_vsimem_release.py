#!/usr/bin/env python3
"""Tile datasets live in /vsimem; they must be unlinked or every tile stays resident.

Without release_tile_datasets() a conversion keeps one uncompressed tile per block in memory
until the worker exits, so memory grows with raster size (a 30-tile, 3 arcsec Europe mosaic was
killed for running out of memory at ~51 GB on a 64 GB machine).

These checks run in a child process: importing osgeo next to pyarrow in one process breaks
pyarrow's filesystem registry (https://github.com/apache/arrow/issues/44696).
"""

import subprocess
import sys
import textwrap



def _run(script: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_release_tile_datasets_unlinks_vsimem_files():
    out = _run(
        """
        import mercantile
        from osgeo import gdal, osr
        from raquet.raster2raquet import create_tile_ds, release_tile_datasets

        gdal.UseExceptions()
        src = gdal.GetDriverByName("MEM").Create("", 4, 4, 1, gdal.GDT_Float32)
        src.GetRasterBand(1).SetNoDataValue(-9999)
        wm = osr.SpatialReference()
        wm.ImportFromEPSG(3857)
        gtiff = gdal.GetDriverByName("GTiff")

        tiles = [mercantile.Tile(x, 0, 1) for x in (0, 1)]
        datasets = [create_tile_ds(gtiff, wm, src, t, 8) for t in tiles]
        paths = [ds.GetDescription() for ds in datasets]
        assert all(p.startswith("/vsimem/") for p in paths), paths
        assert all(gdal.VSIStatL(p) is not None for p in paths)

        release_tile_datasets(datasets)
        assert datasets == []
        print([gdal.VSIStatL(p) is None for p in paths])
        """
    )
    assert out.strip() == "[True, True]"


def test_conversion_leaves_no_tiles_in_vsimem(tmp_path):
    """After a full conversion with overviews, the worker's /vsimem holds no tile files."""
    out = _run(
        f"""
        import multiprocessing, os
        from osgeo import gdal, osr
        from raquet import raster2raquet

        gdal.UseExceptions()
        path = {str(tmp_path / "in.tif")!r}
        ds = gdal.GetDriverByName("GTiff").Create(path, 1024, 1024, 1, gdal.GDT_Float32)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        ds.SetGeoTransform((10.0, 0.001, 0.0, 1.0, 0.0, -0.001))
        ds.GetRasterBand(1).SetNoDataValue(-9999)
        ds.GetRasterBand(1).Fill(1.0)  # no numpy: CI installs the package without the [numpy] extra
        ds = None

        # Run the worker in this process and drain its pipe, then inspect /vsimem
        recv, send = multiprocessing.Pipe(duplex=False)
        import threading
        received = []
        def drain():
            while True:
                msg = recv.recv()
                received.append(msg)
                if msg is None:
                    break
        t = threading.Thread(target=drain)
        t.start()
        raster2raquet.read_raster(
            path,
            raster2raquet.ZoomStrategy.UPPER,
            raster2raquet.ResamplingAlgorithm.NearestNeighbour,
            8,
            send,
        )
        t.join()
        tiles_sent = sum(1 for m in received[1:] if m is not None)
        leftovers = [f for f in (gdal.ReadDir("/vsimem/") or []) if f.startswith("tile-")]
        print(tiles_sent, len(leftovers))
        """
    )
    tiles_sent, leftovers = map(int, out.split())
    assert tiles_sent > 1, "test raster must produce several tiles including overviews"
    assert leftovers == 0
