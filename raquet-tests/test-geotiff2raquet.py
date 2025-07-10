import glob
import itertools
import math
import os
import tempfile
import unittest

import pyarrow.parquet
from raquet import geotiff2raquet

PROJDIR = os.path.join(os.path.dirname(__file__), "..")


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
        self.assertAlmostEqual(stats.stddev, 28.722813233)

    def test_read_statistics_python_nans(self):
        stats = geotiff2raquet.read_statistics_python([math.nan] + list(range(100)), 0)
        self.assertEqual(stats.count, 99)
        self.assertEqual(stats.min, 1)
        self.assertEqual(stats.max, 99)
        self.assertEqual(stats.mean, 50)
        self.assertEqual(stats.sum, 4950)
        self.assertEqual(stats.sum_squares, 328350)
        self.assertAlmostEqual(stats.stddev, 28.722813233)

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
        self.assertEqual(metadata["num_blocks"], 16)
        self.assertEqual(metadata["num_pixels"], 1048576)
        self.assertEqual(metadata["nodata"], None)
        self.assertEqual(metadata["block_resolution"], 5)
        self.assertEqual(metadata["pixel_resolution"], 13)
        self.assertEqual(metadata["minresolution"], 2)
        self.assertEqual(metadata["maxresolution"], 5)
        self.assertEqual(
            [round(b, 3) for b in metadata["bounds"]], [0.0, 40.98, 45.0, 66.513]
        )
        self.assertEqual([round(b, 3) for b in metadata["center"]], [22.5, 53.747, 5])
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

        stats0 = metadata["bands"][0]["stats"]
        self.assertEqual(f"{stats0['count']:.3g}", "1.05e+06")
        self.assertEqual(f"{stats0['max']:.3g}", "255")
        self.assertEqual(f"{stats0['mean']:.3g}", "105")
        self.assertEqual(f"{stats0['min']:.3g}", "0")
        self.assertEqual(f"{stats0['stddev']:.3g}", "78.4")
        self.assertEqual(f"{stats0['sum']:.3g}", "1.1e+08")
        self.assertEqual(f"{stats0['sum_squares']:.3g}", "1.8e+10")

        stats1 = metadata["bands"][1]["stats"]
        self.assertEqual(f"{stats1['count']:.3g}", "1.05e+06")
        self.assertEqual(f"{stats1['max']:.3g}", "255")
        self.assertEqual(f"{stats1['mean']:.3g}", "91.3")
        self.assertEqual(f"{stats1['min']:.3g}", "0")
        self.assertEqual(f"{stats1['stddev']:.3g}", "84.3")
        self.assertEqual(f"{stats1['sum']:.3g}", "9.57e+07")
        self.assertEqual(f"{stats1['sum_squares']:.3g}", "1.62e+10")

        stats2 = metadata["bands"][2]["stats"]
        self.assertEqual(f"{stats2['count']:.3g}", "1.05e+06")
        self.assertEqual(f"{stats2['max']:.3g}", "255")
        self.assertEqual(f"{stats2['mean']:.3g}", "124")
        self.assertEqual(f"{stats2['min']:.3g}", "0")
        self.assertEqual(f"{stats2['stddev']:.3g}", "81.4")
        self.assertEqual(f"{stats2['sum']:.3g}", "1.3e+08")
        self.assertEqual(f"{stats2['sum_squares']:.3g}", "2.31e+10")

        stats3 = metadata["bands"][3]["stats"]
        self.assertEqual(f"{stats3['count']:.3g}", "1.05e+06")
        self.assertEqual(f"{stats3['max']:.3g}", "255")
        self.assertEqual(f"{stats3['mean']:.3g}", "190")
        self.assertEqual(f"{stats3['min']:.3g}", "0")
        self.assertEqual(f"{stats3['stddev']:.3g}", "110")
        self.assertEqual(f"{stats3['sum']:.3g}", "1.99e+08")
        self.assertEqual(f"{stats3['sum_squares']:.3g}", "5.06e+10")

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
        self.assertEqual(metadata["num_blocks"], 4)
        self.assertEqual(metadata["num_pixels"], 262144)
        self.assertEqual(metadata["nodata"], -32767.0)
        self.assertEqual(metadata["block_resolution"], 11)
        self.assertEqual(metadata["pixel_resolution"], 19)
        self.assertEqual(metadata["minresolution"], 10)
        self.assertEqual(metadata["maxresolution"], 11)
        self.assertEqual(
            [round(b, 3) for b in metadata["bounds"]],
            [-122.695, 37.579, -122.344, 37.858],
        )
        self.assertEqual(
            [round(b, 3) for b in metadata["center"]], [-122.52, 37.718, 11]
        )
        self.assertEqual(
            {b["name"]: b["type"] for b in metadata["bands"]}, {"band_1": "int16"}
        )

        stats = metadata["bands"][0]["stats"]
        self.assertEqual(f"{stats['count']:.3g}", "9.76e+04")
        self.assertEqual(f"{stats['max']:.3g}", "358")
        self.assertEqual(f"{stats['mean']:.3g}", "38.1")
        self.assertEqual(f"{stats['min']:.3g}", "-4")
        self.assertEqual(f"{stats['stddev']:.3g}", "54.6")
        self.assertEqual(f"{stats['sum']:.3g}", "3.71e+06")
        self.assertEqual(f"{stats['sum_squares']:.3g}", "4.5e+08")
        self.assertTrue(stats["approximated_stats"])

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
        self.assertEqual(metadata["num_blocks"], 42)
        self.assertEqual(metadata["num_pixels"], 1536 * 1792)
        self.assertEqual(metadata["nodata"], 250.0)
        self.assertEqual(metadata["block_resolution"], 13)
        self.assertEqual(metadata["pixel_resolution"], 21)
        self.assertEqual(metadata["minresolution"], 10)
        self.assertEqual(metadata["maxresolution"], 13)

        stats = metadata["bands"][0]["stats"]
        self.assertEqual(f"{stats['count']:.3g}", "1.22e+06")
        self.assertEqual(f"{stats['max']:.3g}", "95")
        self.assertEqual(f"{stats['mean']:.3g}", "75.8")
        self.assertEqual(f"{stats['min']:.3g}", "11")
        self.assertEqual(f"{stats['stddev']:.3g}", "18.3")
        self.assertEqual(f"{stats['sum']:.3g}", "9.22e+07")
        self.assertEqual(f"{stats['sum_squares']:.3g}", "7.41e+09")
        self.assertTrue(stats["approximated_stats"])

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
        self.assertEqual(metadata["num_blocks"], 12)
        self.assertEqual(metadata["num_pixels"], 1536 * 2048)
        self.assertEqual(metadata["nodata"], 250.0)
        self.assertEqual(metadata["block_resolution"], 12)
        self.assertEqual(metadata["pixel_resolution"], 21)
        self.assertEqual(metadata["minresolution"], 9)
        self.assertEqual(metadata["maxresolution"], 12)

        stats = metadata["bands"][0]["stats"]
        self.assertEqual(f"{stats['count']:.2g}", "1.2e+06")
        self.assertEqual(f"{stats['max']:.2g}", "95")
        self.assertEqual(f"{stats['mean']:.2g}", "76")
        self.assertEqual(f"{stats['min']:.2g}", "11")
        self.assertEqual(f"{stats['stddev']:.2g}", "18")
        self.assertEqual(f"{stats['sum']:.2g}", "9.2e+07")
        self.assertEqual(f"{stats['sum_squares']:.2g}", "7.4e+09")
        self.assertTrue(stats["approximated_stats"])

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
        self.assertEqual(metadata["num_blocks"], 6)
        self.assertEqual(metadata["num_pixels"], 2048 * 3072)
        self.assertEqual(metadata["nodata"], 250.0)
        self.assertEqual(metadata["block_resolution"], 11)
        self.assertEqual(metadata["pixel_resolution"], 21)
        self.assertEqual(metadata["minresolution"], 8)
        self.assertEqual(metadata["maxresolution"], 11)

        stats = metadata["bands"][0]["stats"]
        self.assertEqual(f"{stats['count']:.3g}", "1.22e+06")
        self.assertEqual(f"{stats['max']:.3g}", "95")
        self.assertEqual(f"{stats['mean']:.3g}", "75.8")
        self.assertEqual(f"{stats['min']:.3g}", "11")
        self.assertEqual(f"{stats['stddev']:.3g}", "18.5")
        self.assertEqual(f"{stats['sum']:.3g}", "9.24e+07")
        self.assertEqual(f"{stats['sum_squares']:.3g}", "7.42e+09")
        self.assertTrue(stats["approximated_stats"])

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
        self.assertEqual(metadata["num_blocks"], 42)
        self.assertEqual(metadata["num_pixels"], 2752512)
        self.assertEqual(metadata["nodata"], 250.0)
        self.assertEqual(metadata["block_resolution"], 13)
        self.assertEqual(metadata["pixel_resolution"], 21)
        self.assertEqual(metadata["minresolution"], 10)
        self.assertEqual(metadata["maxresolution"], 13)

        stats = metadata["bands"][0]["stats"]
        self.assertEqual(f"{stats['count']:.3g}", "1.22e+06")
        self.assertEqual(f"{stats['max']:.3g}", "95")
        self.assertEqual(f"{stats['mean']:.3g}", "75.8")
        self.assertEqual(f"{stats['min']:.3g}", "11")
        self.assertEqual(f"{stats['stddev']:.3g}", "18.3")
        self.assertEqual(f"{stats['sum']:.3g}", "9.22e+07")
        self.assertEqual(f"{stats['sum_squares']:.3g}", "7.41e+09")
        self.assertTrue(stats["approximated_stats"])

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
        self.assertEqual(metadata["num_blocks"], 25)
        self.assertEqual(metadata["num_pixels"], 1638400)
        self.assertEqual(metadata["nodata"], 0.0)
        self.assertEqual(metadata["block_resolution"], 13)
        self.assertEqual(metadata["pixel_resolution"], 21)
        self.assertEqual(metadata["minresolution"], 10)
        self.assertEqual(metadata["maxresolution"], 13)

        stats = metadata["bands"][0]["stats"]
        self.assertEqual(f"{stats['count']:.3g}", "2.7e+04")
        self.assertEqual(f"{stats['max']:.3g}", "1")
        self.assertEqual(f"{stats['mean']:.3g}", "1")
        self.assertEqual(f"{stats['min']:.3g}", "1")
        self.assertEqual(f"{stats['stddev']:.3g}", "0")
        self.assertEqual(f"{stats['sum']:.3g}", "2.7e+04")
        self.assertEqual(f"{stats['sum_squares']:.3g}", "2.7e+04")
        self.assertTrue(stats["approximated_stats"])

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
        self.assertEqual(metadata["num_blocks"], 16)
        self.assertEqual(metadata["num_pixels"], 1048576)
        self.assertIsNone(metadata["nodata"])
        self.assertEqual(metadata["block_resolution"], 2)
        self.assertEqual(metadata["pixel_resolution"], 10)
        self.assertEqual(metadata["minresolution"], 0)
        self.assertEqual(metadata["maxresolution"], 2)
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
        stats = metadata["bands"][0]["stats"]
        self.assertEqual(f"{stats['min']:.3g}", "58.6")
        self.assertEqual(f"{stats['max']:.3g}", "70.5")
        self.assertEqual(f"{stats['mean']:.3g}", "63.6")
        self.assertEqual(f"{metadata['bounds'][0]:.3g}", "-78.1")
        self.assertEqual(f"{metadata['bounds'][0]:.3g}", "24.4")
        self.assertEqual(f"{metadata['bounds'][0]:.3g}", "-77.0")
        self.assertEqual(f"{metadata['bounds'][0]:.3g}", "23.6")
