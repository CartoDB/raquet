import os
import tempfile
import unittest

from raquet import raquet2geotiff

PROJDIR = os.path.join(os.path.dirname(__file__), "..")


class TestRaquet2Geotiff(unittest.TestCase):
    def test_europe_parquet(self):
        raquet_filename = os.path.join(PROJDIR, "examples/europe.parquet")
        with tempfile.TemporaryDirectory() as tempdir:
            geotiff_filename = os.path.join(tempdir, "out.tif")
            raquet2geotiff.main(raquet_filename, geotiff_filename)
            geotiff_info = raquet2geotiff.read_geotiff_info(geotiff_filename)

        self.assertEqual(geotiff_info["size"], [1024, 1024])
        self.assertEqual(
            [round(n, 8) for n in geotiff_info["geoTransform"]],
            [0.0, 4891.96981025, 0.0, 10018754.17139462, 0.0, -4891.96981025],
        )
        self.assertEqual(
            [(b["block"], b["type"]) for b in geotiff_info["bands"]],
            [
                ([256, 256], "Byte"),
                ([256, 256], "Byte"),
                ([256, 256], "Byte"),
                ([256, 256], "Byte"),
            ],
        )

    def test_colored_parquet(self):
        raquet_filename = os.path.join(PROJDIR, "tests/colored.parquet")
        with tempfile.TemporaryDirectory() as tempdir:
            geotiff_filename = os.path.join(tempdir, "out.tif")
            raquet2geotiff.main(raquet_filename, geotiff_filename)
            geotiff_info = raquet2geotiff.read_geotiff_info(geotiff_filename)

        band = geotiff_info["bands"][0]
        self.assertEqual(band["colorInterpretation"], "Palette")
        self.assertEqual(
            [colorEntry for colorEntry in band["colorTable"]["entries"][:6]],
            [
                [0, 0, 0, 0],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
                [255, 255, 0, 255],
                [255, 165, 0, 255],
                [255, 0, 0, 255],
            ],
        )
