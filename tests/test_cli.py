#!/usr/bin/env python3
"""Tests for the Raquet CLI."""

import json
import os
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from raquet.cli import cli

# Test data directory
TEST_DIR = Path(__file__).parent


@pytest.fixture
def runner():
    """Create CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestInspect:
    """Tests for the inspect command."""

    def test_inspect_example_file(self, runner):
        """Test inspecting the example parquet file."""
        example_file = Path(__file__).parent.parent / "examples" / "example_data.parquet"
        if not example_file.exists():
            pytest.skip("Example file not found")

        result = runner.invoke(cli, ["inspect", str(example_file)])
        assert result.exit_code == 0
        assert "Raquet File" in result.output or "General Information" in result.output

    def test_inspect_nonexistent_file(self, runner, temp_dir):
        """Test inspect with nonexistent file."""
        result = runner.invoke(cli, ["inspect", str(temp_dir / "nonexistent.parquet")])
        assert result.exit_code != 0

    def test_inspect_verbose(self, runner):
        """Test inspect with verbose flag."""
        example_file = Path(__file__).parent.parent / "examples" / "example_data.parquet"
        if not example_file.exists():
            pytest.skip("Example file not found")

        result = runner.invoke(cli, ["inspect", str(example_file), "-v"])
        assert result.exit_code == 0


class TestConvertGeoTIFF:
    """Tests for GeoTIFF to Raquet conversion."""

    @pytest.fixture
    def sample_geotiff(self):
        """Return path to sample GeoTIFF."""
        tiff_files = list(TEST_DIR.glob("*.tif")) + list(TEST_DIR.glob("*.tiff"))
        if not tiff_files:
            pytest.skip("No GeoTIFF test files found")
        return tiff_files[0]

    def test_convert_geotiff_basic(self, runner, temp_dir, sample_geotiff):
        """Test basic GeoTIFF conversion."""
        output = temp_dir / "output.parquet"

        result = runner.invoke(
            cli,
            ["convert", "geotiff", str(sample_geotiff), str(output)],
        )

        # Check command succeeded
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists(), "Output file was not created"

        # Verify output is valid raquet
        table = pq.read_table(output)
        assert "block" in table.column_names
        assert "metadata" in table.column_names

    def test_convert_geotiff_with_options(self, runner, temp_dir, sample_geotiff):
        """Test GeoTIFF conversion with options."""
        output = temp_dir / "output.parquet"

        result = runner.invoke(
            cli,
            [
                "convert",
                "geotiff",
                str(sample_geotiff),
                str(output),
                "--resampling",
                "bilinear",
                "--block-size",
                "256",
                "-v",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

    def test_convert_geotiff_nonexistent_input(self, runner, temp_dir):
        """Test conversion with nonexistent input file."""
        output = temp_dir / "output.parquet"

        result = runner.invoke(
            cli,
            ["convert", "geotiff", "/nonexistent/file.tif", str(output)],
        )

        assert result.exit_code != 0


class TestExportGeoTIFF:
    """Tests for Raquet to GeoTIFF export."""

    def test_export_example_file(self, runner, temp_dir):
        """Test exporting the example raquet file to GeoTIFF."""
        example_file = Path(__file__).parent.parent / "examples" / "example_data.parquet"
        if not example_file.exists():
            pytest.skip("Example file not found")

        output = temp_dir / "output.tif"

        result = runner.invoke(
            cli,
            ["export", "geotiff", str(example_file), str(output)],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

    def test_export_nonexistent_input(self, runner, temp_dir):
        """Test export with nonexistent input file."""
        output = temp_dir / "output.tif"

        result = runner.invoke(
            cli,
            ["export", "geotiff", "/nonexistent/file.parquet", str(output)],
        )

        assert result.exit_code != 0


class TestRoundTrip:
    """Tests for round-trip conversion (GeoTIFF -> Raquet -> GeoTIFF)."""

    def test_roundtrip_conversion(self, runner, temp_dir):
        """Test converting GeoTIFF to Raquet and back."""
        # Find a test GeoTIFF
        tiff_files = list(TEST_DIR.glob("*.tif")) + list(TEST_DIR.glob("*.tiff"))
        if not tiff_files:
            pytest.skip("No GeoTIFF test files found")

        input_tiff = tiff_files[0]
        raquet_file = temp_dir / "intermediate.parquet"
        output_tiff = temp_dir / "output.tif"

        # Convert to Raquet
        result1 = runner.invoke(
            cli,
            ["convert", "geotiff", str(input_tiff), str(raquet_file)],
        )
        assert result1.exit_code == 0, f"GeoTIFF to Raquet failed: {result1.output}"
        assert raquet_file.exists()

        # Convert back to GeoTIFF
        result2 = runner.invoke(
            cli,
            ["export", "geotiff", str(raquet_file), str(output_tiff)],
        )
        assert result2.exit_code == 0, f"Raquet to GeoTIFF failed: {result2.output}"
        assert output_tiff.exists()


class TestHelp:
    """Tests for help output."""

    def test_main_help(self, runner):
        """Test main help output."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Raquet CLI" in result.output
        assert "convert" in result.output
        assert "export" in result.output
        assert "inspect" in result.output

    def test_convert_help(self, runner):
        """Test convert help output."""
        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code == 0
        assert "geotiff" in result.output
        assert "imageserver" in result.output

    def test_convert_geotiff_help(self, runner):
        """Test convert geotiff help output."""
        result = runner.invoke(cli, ["convert", "geotiff", "--help"])
        assert result.exit_code == 0
        assert "INPUT_FILE" in result.output
        assert "OUTPUT_FILE" in result.output
        assert "--resampling" in result.output

    def test_convert_imageserver_help(self, runner):
        """Test convert imageserver help output."""
        result = runner.invoke(cli, ["convert", "imageserver", "--help"])
        assert result.exit_code == 0
        assert "URL" in result.output
        assert "OUTPUT_FILE" in result.output
        assert "--bbox" in result.output
        assert "--token" in result.output

    def test_export_help(self, runner):
        """Test export help output."""
        result = runner.invoke(cli, ["export", "--help"])
        assert result.exit_code == 0
        assert "geotiff" in result.output

    def test_inspect_help(self, runner):
        """Test inspect help output."""
        result = runner.invoke(cli, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "FILE" in result.output


class TestVersion:
    """Tests for version output."""

    def test_version(self, runner):
        """Test version output."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or "0." in result.output


class TestMetadataValidation:
    """Tests for raquet metadata validation."""

    def test_converted_file_has_valid_metadata(self, runner, temp_dir):
        """Test that converted files have valid raquet metadata."""
        tiff_files = list(TEST_DIR.glob("*.tif")) + list(TEST_DIR.glob("*.tiff"))
        if not tiff_files:
            pytest.skip("No GeoTIFF test files found")

        input_tiff = tiff_files[0]
        output = temp_dir / "output.parquet"

        result = runner.invoke(
            cli,
            ["convert", "geotiff", str(input_tiff), str(output)],
        )

        assert result.exit_code == 0

        # Read and verify metadata
        table = pq.read_table(output)

        # Find metadata row (block=0)
        import pyarrow.compute as pc

        block_zero = table.filter(pc.equal(table.column("block"), 0))
        assert len(block_zero) == 1, "Should have exactly one metadata row"

        metadata_json = block_zero.column("metadata")[0].as_py()
        assert metadata_json is not None

        metadata = json.loads(metadata_json)

        # Verify required fields
        assert "bounds" in metadata
        assert "bands" in metadata
        assert "width" in metadata
        assert "height" in metadata
        assert "block_width" in metadata
        assert "block_height" in metadata

        # Verify bounds is valid
        bounds = metadata["bounds"]
        assert len(bounds) == 4
        assert bounds[0] <= bounds[2], "min longitude should be <= max longitude"
        assert bounds[1] <= bounds[3], "min latitude should be <= max latitude"

        # Verify bands info
        bands = metadata["bands"]
        assert len(bands) > 0, "Should have at least one band"
        for band in bands:
            assert "type" in band
            assert "name" in band
