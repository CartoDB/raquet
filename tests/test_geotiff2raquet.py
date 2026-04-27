import glob
import itertools
import math
import multiprocessing
import os
import tempfile
import unittest

import pyarrow.parquet
from raquet import geotiff2raquet

PROJDIR = os.path.join(os.path.dirname(__file__), "..")


def _build_4band_byte_alpha_mismatch_geotiff(path: str) -> None:
    """Build a 4-band Byte GeoTIFF with photometric / SamplesPerPixel mismatch.

    Run in a `spawn`'d subprocess only — importing osgeo in the test parent
    process clashes with pyarrow's filesystem registry (apache/arrow#44696)
    and would mask the actual failure shape with a different exception.
    """
    from osgeo import gdal, osr

    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, 1024, 1024, 4, gdal.GDT_Byte, options=["COMPRESS=DEFLATE"])
    ds.SetGeoTransform([-10.0, 0.02, 0.0, 50.0, 0.0, -0.02])
    sref = osr.SpatialReference()
    sref.ImportFromEPSG(4326)
    ds.SetSpatialRef(sref)
    color_interps = [
        gdal.GCI_GrayIndex,
        gdal.GCI_Undefined,
        gdal.GCI_Undefined,
        gdal.GCI_Undefined,
    ]
    for i, ci in enumerate(color_interps, start=1):
        b = ds.GetRasterBand(i)
        b.SetColorInterpretation(ci)
        b.SetNoDataValue(255)
        b.Fill(42 + i)
    ds.FlushCache()


class TestGeotiff2Raquet(unittest.TestCase):
    def test_find_minzoom(self):
        rg = geotiff2raquet.RasterGeometry(
            [], [], [], None, 8, -85.0511287798066, -180.0, 85.0511287798066, 180.0
        )
        self.assertEqual(geotiff2raquet.find_minzoom(rg, 6), 1)
        self.assertEqual(geotiff2raquet.find_minzoom(rg, 7), 0)
        self.assertEqual(geotiff2raquet.find_minzoom(rg, 8), 0)

    def test_read_statistics_python(self):
        stats = geotiff2raquet.read_statistics_python(list(range(100)), 0)
        self.assertEqual(stats.count, 99)
        self.assertEqual(stats.min, 1)
        self.assertEqual(stats.max, 99)
        self.assertEqual(stats.mean, 50)
        self.assertEqual(stats.sum, 4950)
        self.assertEqual(stats.sum_squares, 328350)
        self.assertAlmostEqual(stats.stddev, 28.577380332)

    def test_read_statistics_python_nans(self):
        stats = geotiff2raquet.read_statistics_python([math.nan] + list(range(100)), 0)
        self.assertEqual(stats.count, 99)
        self.assertEqual(stats.min, 1)
        self.assertEqual(stats.max, 99)
        self.assertEqual(stats.mean, 50)
        self.assertEqual(stats.sum, 4950)
        self.assertEqual(stats.sum_squares, 328350)
        self.assertAlmostEqual(stats.stddev, 28.577380332)

    @unittest.skipIf(not geotiff2raquet.HAS_NUMPY, "Missing numpy")
    def test_read_statistics_numpy(self):
        arr = geotiff2raquet.numpy.arange(100)
        stats = geotiff2raquet.read_statistics_numpy(arr, 0)
        self.assertEqual(stats.count, 99)
        self.assertEqual(stats.min, 1)
        self.assertEqual(stats.max, 99)
        self.assertEqual(stats.mean, 50)
        self.assertEqual(stats.sum, 4950)
        self.assertEqual(stats.sum_squares, 328350)
        self.assertAlmostEqual(stats.stddev, 28.577380332)

    @unittest.skipIf(not geotiff2raquet.HAS_NUMPY, "Missing numpy")
    def test_read_statistics_numpy_nans(self):
        arr = geotiff2raquet.numpy.arange(101, dtype=float)
        arr[-1] = math.nan
        stats = geotiff2raquet.read_statistics_numpy(arr, 0)
        self.assertEqual(stats.count, 99)
        self.assertEqual(stats.min, 1)
        self.assertEqual(stats.max, 99)
        self.assertEqual(stats.mean, 50)
        self.assertEqual(stats.sum, 4950)
        self.assertEqual(stats.sum_squares, 328350)
        self.assertAlmostEqual(stats.stddev, 28.577380332)

    def test_europe_tif(self):
        geotiff_filename = os.path.join(PROJDIR, "examples/europe.tif")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.AUTO,
                geotiff2raquet.ResamplingAlgorithm.CubicSpline,
                8,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        self.assertEqual(len(table), 23)
        self.assertEqual(
            table.column_names,
            ["block", "metadata", "band_1", "band_2", "band_3", "band_4"],
        )

        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(metadata["compression"], "gzip")
        self.assertEqual(metadata["width"], 1024)
        self.assertEqual(metadata["height"], 1024)
        self.assertEqual(metadata["tiling"]["num_blocks"], 16)
        self.assertIsNone(metadata["bands"][0]["nodata"])
        self.assertEqual(metadata["tiling"]["max_zoom"], 5)
        self.assertEqual(metadata["tiling"]["pixel_zoom"], 13)
        self.assertEqual(metadata["tiling"]["min_zoom"], 2)
        self.assertEqual(
            [round(b, 3) for b in metadata["bounds"]], [0.0, 40.98, 45.0, 66.513]
        )
        self.assertEqual(
            {b["name"]: b["type"] for b in metadata["bands"]},
            {
                "band_1": "uint8",
                "band_2": "uint8",
                "band_3": "uint8",
                "band_4": "uint8",
            },
        )
        self.assertEqual(
            {b["name"]: b["colorinterp"] for b in metadata["bands"]},
            {"band_1": "red", "band_2": "green", "band_3": "blue", "band_4": "alpha"},
        )

        # Check band statistics (v0.3.0 format uses STATISTICS_* keys)
        band0 = metadata["bands"][0]
        self.assertEqual(band0["STATISTICS_MINIMUM"], 0)
        self.assertEqual(band0["STATISTICS_MAXIMUM"], 255)
        self.assertAlmostEqual(band0["STATISTICS_MEAN"], 106.36, places=1)
        self.assertAlmostEqual(band0["STATISTICS_STDDEV"], 78.79, places=1)

    def test_n37_w123_1arc_v2_tif(self):
        geotiff_filename = os.path.join(PROJDIR, "tests/n37_w123_1arc_v2.tif")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.LOWER,
                geotiff2raquet.ResamplingAlgorithm.CubicSpline,
                8,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        self.assertEqual(len(table), 7)
        self.assertEqual(table.column_names, ["block", "metadata", "band_1"])

        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(metadata["compression"], "gzip")
        self.assertEqual(metadata["width"], 512)
        self.assertEqual(metadata["height"], 512)
        self.assertEqual(metadata["tiling"]["num_blocks"], 4)
        self.assertEqual(metadata["bands"][0]["nodata"], -32767.0)
        self.assertEqual(metadata["tiling"]["max_zoom"], 11)
        self.assertEqual(metadata["tiling"]["pixel_zoom"], 19)
        self.assertEqual(metadata["tiling"]["min_zoom"], 10)
        self.assertEqual(
            [round(b, 3) for b in metadata["bounds"]],
            [-122.695, 37.579, -122.344, 37.858],
        )
        self.assertEqual(
            {b["name"]: b["type"] for b in metadata["bands"]}, {"band_1": "int16"}
        )

        # Check band statistics (v0.3.0 format uses STATISTICS_* keys)
        band0 = metadata["bands"][0]
        self.assertAlmostEqual(band0["STATISTICS_MEAN"], 38.1, places=0)
        self.assertAlmostEqual(band0["STATISTICS_STDDEV"], 54.6, places=0)

    def test_smalltile_Annual_NLCD_LndCov_2023_CU_C1V0_tif(self):
        geotiff_filename = os.path.join(
            PROJDIR, "tests/Annual_NLCD_LndCov_2023_CU_C1V0.tif"
        )
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.UPPER,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
                8,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        self.assertEqual(len(table), 63)
        self.assertEqual(table.column_names, ["block", "metadata", "band_1"])

        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(metadata["compression"], "gzip")
        self.assertEqual(metadata["width"], 1536)
        self.assertEqual(metadata["height"], 1792)
        self.assertEqual(metadata["tiling"]["num_blocks"], 42)
        self.assertEqual(metadata["bands"][0]["nodata"], 250.0)
        self.assertEqual(metadata["tiling"]["max_zoom"], 13)
        self.assertEqual(metadata["tiling"]["pixel_zoom"], 21)
        self.assertEqual(metadata["tiling"]["min_zoom"], 10)

        # Check band statistics (v0.3.0 format uses STATISTICS_* keys)
        band0 = metadata["bands"][0]
        self.assertAlmostEqual(band0["STATISTICS_MEAN"], 75.8, places=0)
        # Note: stddev values vary with block size due to resampling
        self.assertGreater(band0["STATISTICS_STDDEV"], 10)

    def test_medtile_Annual_NLCD_LndCov_2023_CU_C1V0_tif(self):
        geotiff_filename = os.path.join(
            PROJDIR, "tests/Annual_NLCD_LndCov_2023_CU_C1V0.tif"
        )
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.UPPER,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
                9,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        self.assertEqual(len(table), 22)
        self.assertEqual(table.column_names, ["block", "metadata", "band_1"])

        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(metadata["compression"], "gzip")
        self.assertEqual(metadata["width"], 1536)
        self.assertEqual(metadata["height"], 2048)
        self.assertEqual(metadata["tiling"]["num_blocks"], 12)
        self.assertEqual(metadata["bands"][0]["nodata"], 250.0)
        self.assertEqual(metadata["tiling"]["max_zoom"], 12)
        self.assertEqual(metadata["tiling"]["pixel_zoom"], 21)
        self.assertEqual(metadata["tiling"]["min_zoom"], 9)

        # Check band statistics (v0.3.0 format uses STATISTICS_* keys)
        band0 = metadata["bands"][0]
        self.assertAlmostEqual(band0["STATISTICS_MEAN"], 76, places=0)
        # Note: stddev values vary with block size due to resampling
        self.assertGreater(band0["STATISTICS_STDDEV"], 10)

    def test_bigtile_Annual_NLCD_LndCov_2023_CU_C1V0_tif(self):
        geotiff_filename = os.path.join(
            PROJDIR, "tests/Annual_NLCD_LndCov_2023_CU_C1V0.tif"
        )
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.UPPER,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
                10,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        self.assertEqual(len(table), 11)
        self.assertEqual(table.column_names, ["block", "metadata", "band_1"])

        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(metadata["compression"], "gzip")
        self.assertEqual(metadata["width"], 2048)
        self.assertEqual(metadata["height"], 3072)
        self.assertEqual(metadata["tiling"]["num_blocks"], 6)
        self.assertEqual(metadata["bands"][0]["nodata"], 250.0)
        self.assertEqual(metadata["tiling"]["max_zoom"], 11)
        self.assertEqual(metadata["tiling"]["pixel_zoom"], 21)
        self.assertEqual(metadata["tiling"]["min_zoom"], 8)

        # Check band statistics (v0.3.0 format uses STATISTICS_* keys)
        band0 = metadata["bands"][0]
        self.assertAlmostEqual(band0["STATISTICS_MEAN"], 75.8, places=0)
        # Note: stddev values vary with block size due to resampling
        self.assertGreater(band0["STATISTICS_STDDEV"], 10)

    def test_multipart_Annual_NLCD_LndCov_2023_CU_C1V0_tif(self):
        geotiff_filename = os.path.join(
            PROJDIR, "tests/Annual_NLCD_LndCov_2023_CU_C1V0.tif"
        )
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_destination = os.path.join(tempdir, "out")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_destination,
                geotiff2raquet.ZoomStrategy.UPPER,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
                target_size=64000,
                block_zoom=8,
            )
            tables = [
                pyarrow.parquet.read_table(name)
                for name in sorted(glob.glob(f"{raquet_destination}/*.parquet"))
            ]

        self.assertGreater(len(tables), 1)
        self.assertEqual(sum(len(table) for table in tables), 63)
        for table in tables:
            self.assertEqual(table.column_names, ["block", "metadata", "band_1"])

        metadata = geotiff2raquet.read_metadata(tables[-1])
        self.assertEqual(metadata["compression"], "gzip")
        self.assertEqual(metadata["width"], 1536)
        self.assertEqual(metadata["height"], 1792)
        self.assertEqual(metadata["tiling"]["num_blocks"], 42)
        self.assertEqual(metadata["bands"][0]["nodata"], 250.0)
        self.assertEqual(metadata["tiling"]["max_zoom"], 13)
        self.assertEqual(metadata["tiling"]["pixel_zoom"], 21)
        self.assertEqual(metadata["tiling"]["min_zoom"], 10)

        # Check band statistics (v0.3.0 format uses STATISTICS_* keys)
        band0 = metadata["bands"][0]
        self.assertAlmostEqual(band0["STATISTICS_MEAN"], 75.8, places=0)
        # Note: stddev values vary with block size due to resampling
        self.assertGreater(band0["STATISTICS_STDDEV"], 10)

    def test_geotiff_discreteloss_2023_tif(self):
        geotiff_filename = os.path.join(PROJDIR, "tests/geotiff-discreteloss_2023.tif")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.UPPER,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
                8,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        self.assertEqual(len(table), 40)
        self.assertEqual(table.column_names, ["block", "metadata", "band_1"])

        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(metadata["compression"], "gzip")
        self.assertEqual(metadata["width"], 1280)
        self.assertEqual(metadata["height"], 1280)
        self.assertEqual(metadata["tiling"]["num_blocks"], 25)
        self.assertEqual(metadata["bands"][0]["nodata"], 0.0)
        self.assertEqual(metadata["tiling"]["max_zoom"], 13)
        self.assertEqual(metadata["tiling"]["pixel_zoom"], 21)
        self.assertEqual(metadata["tiling"]["min_zoom"], 10)

        # Check band statistics (v0.3.0 format uses STATISTICS_* keys)
        band0 = metadata["bands"][0]
        self.assertEqual(band0["STATISTICS_MINIMUM"], 1)
        self.assertEqual(band0["STATISTICS_MAXIMUM"], 1)
        self.assertEqual(band0["STATISTICS_MEAN"], 1)
        self.assertEqual(band0["STATISTICS_STDDEV"], 0)

    def test_colored_tif(self):
        geotiff_filename = os.path.join(PROJDIR, "tests/colored.tif")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.AUTO,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
                8,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(
            {b["name"]: b["colorinterp"] for b in metadata["bands"]},
            {"band_1": "palette"},
        )

        color_dict = metadata["bands"][0]["colortable"]
        self.assertEqual(
            {k: list(v) for k, v in itertools.islice(color_dict.items(), 6)},
            {
                "0": [0, 0, 0, 0],
                "1": [0, 255, 0, 255],
                "2": [0, 0, 255, 255],
                "3": [255, 255, 0, 255],
                "4": [255, 165, 0, 255],
                "5": [255, 0, 0, 255],
            },
        )

    def test_big_world(self):
        geotiff_filename = os.path.join(PROJDIR, "tests/big-world.tif")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.AUTO,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
                8,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(metadata["width"], 1024)
        self.assertEqual(metadata["height"], 1024)
        self.assertEqual(metadata["tiling"]["num_blocks"], 16)
        self.assertIsNone(metadata["bands"][0]["nodata"])
        self.assertEqual(metadata["tiling"]["max_zoom"], 2)
        self.assertEqual(metadata["tiling"]["pixel_zoom"], 10)
        self.assertEqual(metadata["tiling"]["min_zoom"], 0)
        self.assertEqual(
            {b["name"]: b["colorinterp"] for b in metadata["bands"]},
            {"band_1": "red", "band_2": "green", "band_3": "blue", "band_4": "alpha"},
        )

    def test_milton_2024(self):
        geotiff_filename = os.path.join(PROJDIR, "tests/Milton_2024-excerpt.tiff")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.AUTO,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
                8,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(
            {b["name"]: b["colorinterp"] for b in metadata["bands"]},
            {"band_1": "gray"},
        )
        # Check band statistics (v0.3.0 format uses STATISTICS_* keys)
        band0 = metadata["bands"][0]
        self.assertAlmostEqual(band0["STATISTICS_MINIMUM"], 58.6, places=0)
        self.assertAlmostEqual(band0["STATISTICS_MAXIMUM"], 70.5, places=0)
        self.assertAlmostEqual(band0["STATISTICS_MEAN"], 63.6, places=0)
        self.assertEqual(f"{metadata['bounds'][0]:.3g}", "-78.8")
        self.assertEqual(f"{metadata['bounds'][1]:.3g}", "21.9")
        self.assertEqual(f"{metadata['bounds'][2]:.3g}", "-75.9")
        self.assertEqual(f"{metadata['bounds'][3]:.3g}", "24.5")

    def test_civ(self):
        geotiff_filename = os.path.join(PROJDIR, "tests/civ.tif")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.AUTO,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
                8,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(
            {b["name"]: b["colorinterp"] for b in metadata["bands"]},
            {"band_1": "gray"},
        )
        self.assertEqual(f"{metadata['bounds'][0]:.3g}", "-180")
        self.assertEqual(f"{metadata['bounds'][1]:.3g}", "-85.1")
        self.assertEqual(f"{metadata['bounds'][2]:.3g}", "180")
        self.assertEqual(f"{metadata['bounds'][3]:.3g}", "85.1")

    def test_4band_byte_alpha_mismatch(self):
        """Regression: 4-band Byte GeoTIFF where photometric metadata makes
        gdal.Warp implicitly add an alpha mask band. Used to crash with
        `RuntimeError: Destination dataset has 4 bands, but at least 5 are
        needed`. Should now convert cleanly with the original 4 data bands
        in the parquet output (the alpha mask is discarded).
        """
        with tempfile.TemporaryDirectory() as tempdir:
            tif = os.path.join(tempdir, "synth-4band-byte.tif")

            # Build the input in a spawn'd subprocess so osgeo never loads in
            # this test process — importing both osgeo and pyarrow in the same
            # process collides on the 'file' filesystem registration.
            ctx = multiprocessing.get_context("spawn")
            proc = ctx.Process(
                target=_build_4band_byte_alpha_mismatch_geotiff, args=(tif,)
            )
            proc.start()
            proc.join(timeout=60)
            self.assertEqual(proc.exitcode, 0, "GeoTIFF builder subprocess failed")

            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                tif,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.AUTO,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
                9,
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        # All four original data bands made it into the output; the implicit
        # alpha mask Warp may have added is intentionally not persisted.
        self.assertEqual(
            table.column_names,
            ["block", "metadata", "band_1", "band_2", "band_3", "band_4"],
        )
        metadata = geotiff2raquet.read_metadata(table)
        self.assertEqual(len(metadata["bands"]), 4)
        # Critical: at least one data tile was actually written. Without the
        # fix, the worker process crashes mid-warp and the parent emits a
        # parquet with the metadata row but zero data rows — visible only
        # by checking that the table is more than the metadata header.
        self.assertGreater(
            len(table),
            1,
            f"Expected at least one data tile in addition to the metadata row; "
            f"got {len(table)}. Worker likely crashed mid-warp.",
        )
