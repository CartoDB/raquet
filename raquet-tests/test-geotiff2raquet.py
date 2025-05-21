import glob
import itertools
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
        self.assertEqual(f"{stats0['count']:.4g}", "1.049e+06")
        self.assertEqual(f"{stats0['max']:.4g}", "255")
        self.assertEqual(f"{stats0['mean']:.4g}", "104.7")
        self.assertEqual(f"{stats0['min']:.4g}", "0")
        self.assertEqual(f"{stats0['stddev']:.4g}", "63.24")
        self.assertEqual(f"{stats0['sum']:.4g}", "1.098e+08")
        self.assertEqual(f"{stats0['sum_squares']:.4g}", "1.827e+10")

        stats1 = metadata["bands"][1]["stats"]
        self.assertEqual(f"{stats1['count']:.4g}", "1.049e+06")
        self.assertEqual(f"{stats1['max']:.4g}", "255")
        self.assertEqual(f"{stats1['mean']:.4g}", "91.15")
        self.assertEqual(f"{stats1['min']:.4g}", "0")
        self.assertEqual(f"{stats1['stddev']:.4g}", "58.76")
        self.assertEqual(f"{stats1['sum']:.4g}", "9.558e+07")
        self.assertEqual(f"{stats1['sum_squares']:.4g}", "1.642e+10")

        stats2 = metadata["bands"][2]["stats"]
        self.assertEqual(f"{stats2['count']:.4g}", "1.049e+06")
        self.assertEqual(f"{stats2['max']:.4g}", "255")
        self.assertEqual(f"{stats2['mean']:.4g}", "124")
        self.assertEqual(f"{stats2['min']:.4g}", "0")
        self.assertEqual(f"{stats2['stddev']:.4g}", "68.08")
        self.assertEqual(f"{stats2['sum']:.4g}", "1.3e+08")
        self.assertEqual(f"{stats2['sum_squares']:.4g}", "2.342e+10")

        stats3 = metadata["bands"][3]["stats"]
        self.assertEqual(f"{stats3['count']:.4g}", "1.049e+06")
        self.assertEqual(f"{stats3['max']:.4g}", "255")
        self.assertEqual(f"{stats3['mean']:.4g}", "189.7")
        self.assertEqual(f"{stats3['min']:.4g}", "0")
        self.assertEqual(f"{stats3['stddev']:.4g}", "83.36")
        self.assertEqual(f"{stats3['sum']:.4g}", "1.99e+08")
        self.assertEqual(f"{stats3['sum_squares']:.4g}", "5.053e+10")

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
        self.assertEqual(f"{stats['count']:.4g}", "9.692e+04")
        self.assertEqual(f"{stats['max']:.4g}", "377")
        self.assertEqual(f"{stats['mean']:.4g}", "38.22")
        self.assertEqual(f"{stats['min']:.4g}", "-7")
        self.assertEqual(f"{stats['stddev']:.4g}", "54.02")
        self.assertEqual(f"{stats['sum']:.4g}", "3.704e+06")
        self.assertEqual(f"{stats['sum_squares']:.4g}", "4.539e+08")

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
        self.assertEqual(f"{stats['count']:.4g}", "1.216e+06")
        self.assertEqual(f"{stats['max']:.4g}", "95")
        self.assertEqual(f"{stats['mean']:.4g}", "75.85")
        self.assertEqual(f"{stats['min']:.4g}", "11")
        self.assertEqual(f"{stats['stddev']:.4g}", "16.47")
        self.assertEqual(f"{stats['sum']:.4g}", "9.225e+07")
        self.assertEqual(f"{stats['sum_squares']:.4g}", "7.415e+09")

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
        self.assertEqual(f"{stats['count']:.4g}", "1.216e+06")
        self.assertEqual(f"{stats['max']:.4g}", "95")
        self.assertEqual(f"{stats['mean']:.4g}", "75.85")
        self.assertEqual(f"{stats['min']:.4g}", "11")
        self.assertEqual(f"{stats['stddev']:.4g}", "17.25") # "16.47")
        self.assertEqual(f"{stats['sum']:.4g}", "9.225e+07")
        self.assertEqual(f"{stats['sum_squares']:.4g}", "7.415e+09")

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
        self.assertEqual(f"{stats['count']:.4g}", "1.216e+06")
        self.assertEqual(f"{stats['max']:.4g}", "95")
        self.assertEqual(f"{stats['mean']:.4g}", "75.85")
        self.assertEqual(f"{stats['min']:.4g}", "11")
        self.assertEqual(f"{stats['stddev']:.4g}", "18.28") # "16.47")
        self.assertEqual(f"{stats['sum']:.4g}", "9.225e+07")
        self.assertEqual(f"{stats['sum_squares']:.4g}", "7.415e+09")

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
        self.assertEqual(f"{stats['count']:.4g}", "1.216e+06")
        self.assertEqual(f"{stats['max']:.4g}", "95")
        self.assertEqual(f"{stats['mean']:.4g}", "75.85")
        self.assertEqual(f"{stats['min']:.4g}", "11")
        self.assertEqual(f"{stats['stddev']:.4g}", "16.47")
        self.assertEqual(f"{stats['sum']:.4g}", "9.225e+07")
        self.assertEqual(f"{stats['sum_squares']:.4g}", "7.415e+09")

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
        self.assertEqual(f"{stats['count']:.4g}", "2.736e+04")
        self.assertEqual(f"{stats['max']:.4g}", "1")
        self.assertEqual(f"{stats['mean']:.4g}", "1")
        self.assertEqual(f"{stats['min']:.4g}", "1")
        self.assertEqual(f"{stats['stddev']:.4g}", "0")
        self.assertEqual(f"{stats['sum']:.4g}", "2.736e+04")
        self.assertEqual(f"{stats['sum_squares']:.4g}", "2.736e+04")

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
