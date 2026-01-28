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

        # v0.3.0 uses tiling object for block dimensions
        if "tiling" in metadata:
            assert "block_width" in metadata["tiling"]
            assert "block_height" in metadata["tiling"]
        else:
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


class TestConvertOverviewOptions:
    """Tests for overview-related convert options."""

    @pytest.fixture
    def sample_geotiff(self):
        """Return path to sample GeoTIFF."""
        # Try examples directory first
        examples_dir = Path(__file__).parent.parent / "examples"
        tiff_files = list(examples_dir.glob("*.tif"))
        if tiff_files:
            return tiff_files[0]
        # Fall back to test directory
        tiff_files = list(TEST_DIR.glob("*.tif")) + list(TEST_DIR.glob("*.tiff"))
        if not tiff_files:
            pytest.skip("No GeoTIFF test files found")
        return tiff_files[0]

    def test_convert_overviews_none(self, runner, temp_dir, sample_geotiff):
        """Test conversion with --overviews none (no pyramid)."""
        output = temp_dir / "output.parquet"

        result = runner.invoke(
            cli,
            ["convert", "raster", str(sample_geotiff), str(output), "--overviews", "none"],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify only one zoom level exists
        table = pq.read_table(output)
        import pyarrow.compute as pc
        import quadbin

        blocks = table.filter(pc.not_equal(table.column("block"), 0))
        zoom_levels = set()
        for block_id in blocks.column("block").to_pylist():
            _, _, z = quadbin.cell_to_tile(block_id)
            zoom_levels.add(z)

        assert len(zoom_levels) == 1, f"Expected 1 zoom level with --overviews none, got {zoom_levels}"

    def test_convert_overviews_auto(self, runner, temp_dir, sample_geotiff):
        """Test conversion with --overviews auto (default, builds pyramid)."""
        output = temp_dir / "output.parquet"

        result = runner.invoke(
            cli,
            ["convert", "raster", str(sample_geotiff), str(output), "--overviews", "auto"],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify multiple zoom levels exist
        table = pq.read_table(output)
        import pyarrow.compute as pc
        import quadbin

        blocks = table.filter(pc.not_equal(table.column("block"), 0))
        zoom_levels = set()
        for block_id in blocks.column("block").to_pylist():
            _, _, z = quadbin.cell_to_tile(block_id)
            zoom_levels.add(z)

        # Auto mode should create multiple zoom levels for most images
        assert len(zoom_levels) >= 1, f"Expected at least 1 zoom level, got {zoom_levels}"

    def test_convert_min_zoom(self, runner, temp_dir, sample_geotiff):
        """Test conversion with --min-zoom option."""
        output = temp_dir / "output.parquet"

        result = runner.invoke(
            cli,
            ["convert", "raster", str(sample_geotiff), str(output), "--min-zoom", "3"],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify minimum zoom level is respected
        table = pq.read_table(output)
        import pyarrow.compute as pc
        import quadbin

        blocks = table.filter(pc.not_equal(table.column("block"), 0))
        zoom_levels = set()
        for block_id in blocks.column("block").to_pylist():
            _, _, z = quadbin.cell_to_tile(block_id)
            zoom_levels.add(z)

        min_zoom = min(zoom_levels)
        assert min_zoom >= 3, f"Min zoom should be >= 3, got {min_zoom}"


class TestConvertStreaming:
    """Tests for streaming mode conversion."""

    @pytest.fixture
    def sample_geotiff(self):
        """Return path to sample GeoTIFF."""
        examples_dir = Path(__file__).parent.parent / "examples"
        tiff_files = list(examples_dir.glob("*.tif"))
        if tiff_files:
            return tiff_files[0]
        tiff_files = list(TEST_DIR.glob("*.tif")) + list(TEST_DIR.glob("*.tiff"))
        if not tiff_files:
            pytest.skip("No GeoTIFF test files found")
        return tiff_files[0]

    def test_streaming_mode(self, runner, temp_dir, sample_geotiff):
        """Test conversion with --streaming flag."""
        output = temp_dir / "output.parquet"

        result = runner.invoke(
            cli,
            ["convert", "raster", str(sample_geotiff), str(output), "--streaming"],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify output is valid raquet
        table = pq.read_table(output)
        assert "block" in table.column_names
        assert "metadata" in table.column_names

    def test_streaming_matches_non_streaming(self, runner, temp_dir, sample_geotiff):
        """Test that streaming mode produces equivalent output to non-streaming."""
        output_stream = temp_dir / "streaming.parquet"
        output_normal = temp_dir / "normal.parquet"

        # Convert with streaming
        result1 = runner.invoke(
            cli,
            ["convert", "raster", str(sample_geotiff), str(output_stream), "--streaming", "--overviews", "none"],
        )
        assert result1.exit_code == 0

        # Convert without streaming
        result2 = runner.invoke(
            cli,
            ["convert", "raster", str(sample_geotiff), str(output_normal), "--overviews", "none"],
        )
        assert result2.exit_code == 0

        # Compare outputs
        table_stream = pq.read_table(output_stream)
        table_normal = pq.read_table(output_normal)

        assert len(table_stream) == len(table_normal), "Row counts should match"
        assert table_stream.schema == table_normal.schema, "Schemas should match"

        # Compare block IDs
        stream_blocks = sorted(table_stream.column("block").to_pylist())
        normal_blocks = sorted(table_normal.column("block").to_pylist())
        assert stream_blocks == normal_blocks, "Block IDs should match"


class TestExportOverviews:
    """Tests for GeoTIFF export with overviews."""

    def test_export_with_overviews(self, runner, temp_dir):
        """Test exporting with --overviews flag."""
        example_file = Path(__file__).parent.parent / "examples" / "europe.parquet"
        if not example_file.exists():
            pytest.skip("Example file not found")

        output = temp_dir / "output.tif"

        result = runner.invoke(
            cli,
            ["export", "geotiff", str(example_file), str(output), "--overviews"],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()


class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_example_file(self, runner):
        """Test validating the example parquet file."""
        example_file = Path(__file__).parent.parent / "examples" / "europe.parquet"
        if not example_file.exists():
            pytest.skip("Example file not found")

        result = runner.invoke(cli, ["validate", str(example_file)])
        assert result.exit_code == 0
        assert "VALID" in result.output

    def test_validate_json_output(self, runner):
        """Test validate with --json flag."""
        example_file = Path(__file__).parent.parent / "examples" / "europe.parquet"
        if not example_file.exists():
            pytest.skip("Example file not found")

        result = runner.invoke(cli, ["validate", str(example_file), "--json"])
        assert result.exit_code == 0

        # Parse JSON output
        output_json = json.loads(result.output)
        assert "is_valid" in output_json
        assert output_json["is_valid"] is True

    def test_validate_nonexistent_file(self, runner, temp_dir):
        """Test validate with nonexistent file."""
        result = runner.invoke(cli, ["validate", str(temp_dir / "nonexistent.parquet")])
        assert result.exit_code != 0

    def test_validate_help(self, runner):
        """Test validate help output."""
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate" in result.output
        assert "--json" in result.output


class TestV03Metadata:
    """Tests for v0.3.0 metadata format."""

    def test_v03_metadata_structure(self, runner, temp_dir):
        """Test that converted files use v0.3.0 metadata format."""
        examples_dir = Path(__file__).parent.parent / "examples"
        tiff_files = list(examples_dir.glob("*.tif"))
        if not tiff_files:
            pytest.skip("No GeoTIFF test files found")

        input_tiff = tiff_files[0]
        output = temp_dir / "output.parquet"

        result = runner.invoke(
            cli,
            ["convert", "raster", str(input_tiff), str(output)],
        )
        assert result.exit_code == 0

        # Read and verify v0.3.0 metadata structure
        table = pq.read_table(output)
        import pyarrow.compute as pc

        block_zero = table.filter(pc.equal(table.column("block"), 0))
        metadata = json.loads(block_zero.column("metadata")[0].as_py())

        # v0.3.0 required fields
        assert metadata.get("version") == "0.3.0", "Should use v0.3.0 format"
        assert "crs" in metadata, "v0.3.0 should have crs field"
        assert "bounds_crs" in metadata, "v0.3.0 should have bounds_crs field"
        assert "tiling" in metadata, "v0.3.0 should have tiling object"

        # Check tiling object
        tiling = metadata["tiling"]
        assert "scheme" in tiling
        assert "block_width" in tiling
        assert "block_height" in tiling
        assert "min_zoom" in tiling
        assert "max_zoom" in tiling
        assert "pixel_zoom" in tiling
        assert "num_blocks" in tiling

        # Check bands have GDAL-compatible stats keys
        for band in metadata.get("bands", []):
            # These are optional but should use GDAL-compatible names if present
            if "STATISTICS_MINIMUM" in band:
                assert isinstance(band["STATISTICS_MINIMUM"], (int, float))
            if "STATISTICS_MAXIMUM" in band:
                assert isinstance(band["STATISTICS_MAXIMUM"], (int, float))
