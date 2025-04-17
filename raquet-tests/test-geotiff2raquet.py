import itertools
import os
import tempfile
import unittest

import pyarrow.parquet
from raquet import geotiff2raquet

PROJDIR = os.path.join(os.path.dirname(__file__), "..")


def str_stats(d):
    return " ".join([f"{k}={v:.4g}" for k, v in sorted(d.items()) if k != "blocks"])


class TestGeotiff2Raquet(unittest.TestCase):
    def test_europe_tif(self):
        geotiff_filename = os.path.join(PROJDIR, "examples/europe.tif")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.AUTO,
                geotiff2raquet.ResamplingAlgorithm.CubicSpline,
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
        self.assertEqual(
            str_stats(metadata["bands"][0]["stats"]),
            "count=1.049e+06 max=255 mean=104.7 min=0 stddev=63.24 sum=1.098e+08 sum_squares=1.827e+10",
        )
        self.assertEqual(
            str_stats(metadata["bands"][1]["stats"]),
            "count=1.049e+06 max=255 mean=91.15 min=0 stddev=58.76 sum=9.558e+07 sum_squares=1.642e+10",
        )
        self.assertEqual(
            str_stats(metadata["bands"][2]["stats"]),
            "count=1.049e+06 max=255 mean=124 min=0 stddev=68.08 sum=1.3e+08 sum_squares=2.342e+10",
        )
        self.assertEqual(
            str_stats(metadata["bands"][3]["stats"]),
            "count=1.049e+06 max=255 mean=189.7 min=0 stddev=83.36 sum=1.99e+08 sum_squares=5.053e+10",
        )

    def test_n37_w123_1arc_v2_tif(self):
        geotiff_filename = os.path.join(PROJDIR, "tests/n37_w123_1arc_v2.tif")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.LOWER,
                geotiff2raquet.ResamplingAlgorithm.CubicSpline,
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
        self.assertEqual(
            str_stats(metadata["bands"][0]["stats"]),
            "count=9.692e+04 max=377 mean=38.22 min=-7 stddev=54.02 sum=3.704e+06 sum_squares=4.539e+08",
        )

    def test_Annual_NLCD_LndCov_2023_CU_C1V0_tif(self):
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
            )
            table = pyarrow.parquet.read_table(raquet_filename)

        self.assertEqual(len(table), 63)
        self.assertEqual(table.column_names, ["block", "metadata", "band_1"])

        metadata = geotiff2raquet.read_metadata(table)
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
        self.assertEqual(
            str_stats(metadata["bands"][0]["stats"]),
            "count=1.216e+06 max=95 mean=75.85 min=11 stddev=16.47 sum=9.225e+07 sum_squares=7.415e+09",
        )

    def test_geotiff_discreteloss_2023_tif(self):
        geotiff_filename = os.path.join(PROJDIR, "tests/geotiff-discreteloss_2023.tif")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.UPPER,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
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
        self.assertEqual(
            str_stats(metadata["bands"][0]["stats"]),
            "count=2.736e+04 max=1 mean=1 min=1 stddev=0 sum=2.736e+04 sum_squares=2.736e+04",
        )

    def test_colored_tif(self):
        geotiff_filename = os.path.join(PROJDIR, "tests/colored.tif")
        with tempfile.TemporaryDirectory() as tempdir:
            raquet_filename = os.path.join(tempdir, "out.parquet")
            geotiff2raquet.main(
                geotiff_filename,
                raquet_filename,
                geotiff2raquet.ZoomStrategy.AUTO,
                geotiff2raquet.ResamplingAlgorithm.NearestNeighbour,
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
