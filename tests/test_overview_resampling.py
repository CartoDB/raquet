#!/usr/bin/env python3
"""Tests for --overview-resampling (how overview tiles aggregate child pixels)."""

import gzip
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import quadbin
from click.testing import CliRunner

from raquet.cli import cli

NODATA = -9999.0
PEAK = 10.0


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hazard_geotiff(temp_dir):
    """A float32 'flood depth' raster: shallow 1.0 everywhere, one deep 10.0 pixel, a nodata band.

    Average overviews dilute the deep pixel; max overviews must keep it. Written in a child
    process: importing osgeo alongside pyarrow in one process breaks pyarrow's filesystem
    registry (https://github.com/apache/arrow/issues/44696), which raquet also avoids.
    """
    path = temp_dir / "hazard.tif"
    script = textwrap.dedent(
        f"""
        import numpy
        from osgeo import gdal, osr
        gdal.UseExceptions()
        size = 1024
        data = numpy.full((size, size), 1.0, dtype=numpy.float32)
        data[:, : size // 4] = {NODATA}  # dry strip, ignored by every overview method
        data[600, 700] = {PEAK}
        ds = gdal.GetDriverByName("GTiff").Create({str(path)!r}, size, size, 1, gdal.GDT_Float32)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        ds.SetGeoTransform((10.0, 0.001, 0.0, 1.0, 0.0, -0.001))  # ~110 m pixels near the equator
        band = ds.GetRasterBand(1)
        band.SetNoDataValue({NODATA})
        band.WriteArray(data)
        ds = None
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    return path


def _convert(runner, src, out, *extra):
    result = runner.invoke(
        cli,
        ["convert", "raster", str(src), str(out), "--zoom-strategy", "upper", *extra],
    )
    assert result.exit_code == 0, f"Command failed: {result.output}"
    return pq.read_table(out)


def _values_at_min_zoom(table):
    """Decode every block at the table's lowest zoom; return valid (non-nodata) pixel values."""
    metadata = json.loads(
        table.filter(pc.equal(table.column("block"), 0)).column("metadata")[0].as_py()
    )
    min_zoom = metadata["tiling"]["min_zoom"]
    max_zoom = metadata["tiling"]["max_zoom"]
    assert min_zoom < max_zoom, "test raster must produce overviews"

    values = []
    for block, raw in zip(table.column("block").to_pylist(), table.column("band_1").to_pylist()):
        if block == 0 or quadbin.cell_to_tile(block)[2] != min_zoom:
            continue
        if metadata.get("compression") == "gzip":
            raw = gzip.decompress(raw)
        arr = numpy.frombuffer(raw, dtype=numpy.float32)
        values.append(arr[arr != NODATA])
    return numpy.concatenate(values)


class TestOverviewResampling:
    def test_default_is_average(self, runner, temp_dir, hazard_geotiff):
        """No flag keeps today's behaviour, identical to an explicit --overview-resampling average."""
        default = _values_at_min_zoom(_convert(runner, hazard_geotiff, temp_dir / "default.parquet"))
        average = _values_at_min_zoom(
            _convert(runner, hazard_geotiff, temp_dir / "avg.parquet", "--overview-resampling", "average")
        )
        numpy.testing.assert_array_equal(default, average)

    def test_max_keeps_peak_average_dilutes_it(self, runner, temp_dir, hazard_geotiff):
        average = _values_at_min_zoom(
            _convert(runner, hazard_geotiff, temp_dir / "avg.parquet", "--overview-resampling", "average")
        )
        maximum = _values_at_min_zoom(
            _convert(runner, hazard_geotiff, temp_dir / "max.parquet", "--overview-resampling", "max")
        )
        assert maximum.max() == pytest.approx(PEAK)
        assert average.max() < PEAK
        # Nodata is ignored either way: no value can fall below the shallow 1.0 depth
        assert maximum.min() >= 1.0
        assert average.min() >= 1.0

    def test_min(self, runner, temp_dir, hazard_geotiff):
        minimum = _values_at_min_zoom(
            _convert(runner, hazard_geotiff, temp_dir / "min.parquet", "--overview-resampling", "min")
        )
        assert minimum.max() == pytest.approx(1.0)

    def test_native_zoom_unchanged(self, runner, temp_dir, hazard_geotiff):
        """The flag only affects overviews: native-resolution blocks are byte-identical."""
        avg = _convert(runner, hazard_geotiff, temp_dir / "avg.parquet", "--overview-resampling", "average")
        mx = _convert(runner, hazard_geotiff, temp_dir / "max.parquet", "--overview-resampling", "max")
        max_zoom = json.loads(
            avg.filter(pc.equal(avg.column("block"), 0)).column("metadata")[0].as_py()
        )["tiling"]["max_zoom"]

        def native(t):
            return {
                b: d
                for b, d in zip(t.column("block").to_pylist(), t.column("band_1").to_pylist())
                if b != 0 and quadbin.cell_to_tile(b)[2] == max_zoom
            }

        native_avg = native(avg)
        native_max = native(mx)
        assert native_avg.keys() == native_max.keys()
        for block, data in native_avg.items():
            assert gzip.decompress(data) == gzip.decompress(native_max[block])

    def test_geotiff_command_accepts_flag(self, runner, temp_dir, hazard_geotiff):
        out = temp_dir / "geotiff.parquet"
        result = runner.invoke(
            cli,
            ["convert", "geotiff", str(hazard_geotiff), str(out), "--overview-resampling", "max"],
        )
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Overview resampling: max" in result.output

    def test_invalid_choice_rejected(self, runner, temp_dir, hazard_geotiff):
        result = runner.invoke(
            cli,
            ["convert", "raster", str(hazard_geotiff), str(temp_dir / "x.parquet"), "--overview-resampling", "bogus"],
        )
        assert result.exit_code != 0

    def test_help_lists_flag(self, runner):
        result = runner.invoke(cli, ["convert", "raster", "--help"])
        assert result.exit_code == 0
        assert "--overview-resampling" in result.output
