#!/usr/bin/env python3
"""Tests for the Earth Engine conversion module."""

import os
import pytest

# Check if earthengine dependencies are available
try:
    from raquet.earthengine import (
        _gcs_to_vsigs,
        _vsigs_to_gcs,
        EarthEngineConfig,
        EarthEngineError,
        EarthEngineAuthError,
        EarthEngineTaskError,
    )

    HAS_EARTHENGINE = True
except ImportError:
    HAS_EARTHENGINE = False


pytestmark = pytest.mark.skipif(
    not HAS_EARTHENGINE,
    reason="Earth Engine dependencies not installed (pip install raquet[earthengine])",
)


class TestGCSPathConversion:
    """Tests for GCS path utilities."""

    def test_gs_to_vsigs(self):
        """Test converting gs:// to /vsigs/"""
        result = _gcs_to_vsigs("gs://my-bucket/path/to/file.tif")
        assert result == "/vsigs/my-bucket/path/to/file.tif"

    def test_gs_to_vsigs_root(self):
        """Test converting gs:// with file at root."""
        result = _gcs_to_vsigs("gs://bucket/file.tif")
        assert result == "/vsigs/bucket/file.tif"

    def test_gs_to_vsigs_passthrough(self):
        """Test that non-gs:// paths pass through."""
        result = _gcs_to_vsigs("/vsigs/bucket/file.tif")
        assert result == "/vsigs/bucket/file.tif"

    def test_vsigs_to_gcs(self):
        """Test converting /vsigs/ to gs://"""
        result = _vsigs_to_gcs("/vsigs/my-bucket/path/to/file.tif")
        assert result == "gs://my-bucket/path/to/file.tif"

    def test_vsigs_to_gcs_passthrough(self):
        """Test that non-/vsigs/ paths pass through."""
        result = _vsigs_to_gcs("gs://bucket/file.tif")
        assert result == "gs://bucket/file.tif"


class TestEarthEngineConfig:
    """Tests for configuration dataclass."""

    def test_config_defaults(self):
        """Test default values."""
        config = EarthEngineConfig(
            image_spec="COPERNICUS/DEM/GLO30",
            gcs_bucket="test-bucket",
            output_path="output.parquet",
        )
        assert config.scale == 10
        assert config.crs == "EPSG:4326"
        assert config.block_size == 256
        assert config.delete_temp is True
        assert config.max_pixels == int(1e13)
        assert config.file_format == "GeoTIFF"

    def test_config_custom_values(self):
        """Test custom values."""
        config = EarthEngineConfig(
            image_spec="COPERNICUS/S2_SR/image123",
            gcs_bucket="my-bucket",
            output_path="out.parquet",
            scale=30,
            crs="EPSG:32632",
            bands=["B4", "B3", "B2"],
            delete_temp=False,
        )
        assert config.scale == 30
        assert config.crs == "EPSG:32632"
        assert config.bands == ["B4", "B3", "B2"]
        assert config.delete_temp is False

    def test_get_gcs_path_explicit(self):
        """Test get_gcs_path with explicit path."""
        config = EarthEngineConfig(
            image_spec="test",
            gcs_bucket="bucket",
            output_path="out.parquet",
            gcs_path="custom/path/file.tif",
        )
        assert config.get_gcs_path() == "custom/path/file.tif"

    def test_get_gcs_path_auto(self):
        """Test get_gcs_path auto-generation."""
        config = EarthEngineConfig(
            image_spec="test",
            gcs_bucket="bucket",
            output_path="out.parquet",
        )
        path = config.get_gcs_path()
        assert path.startswith("raquet-temp/export-")
        assert path.endswith(".tif")


class TestExceptions:
    """Tests for custom exceptions."""

    def test_earth_engine_error(self):
        """Test base exception."""
        error = EarthEngineError("test error")
        assert str(error) == "test error"

    def test_auth_error(self):
        """Test auth exception inherits from base."""
        error = EarthEngineAuthError("auth failed")
        assert isinstance(error, EarthEngineError)
        assert str(error) == "auth failed"

    def test_task_error_with_status(self):
        """Test task exception with status dict."""
        status = {"state": "FAILED", "error_message": "quota exceeded"}
        error = EarthEngineTaskError("task failed", task_status=status)
        assert isinstance(error, EarthEngineError)
        assert error.task_status == status
        assert str(error) == "task failed"


class TestGDALPathCLI:
    """Tests for the GDALPath CLI parameter type."""

    def test_gdal_path_import(self):
        """Test that GDALPath can be imported from cli."""
        from raquet.cli import GDALPath

        assert GDALPath is not None

    def test_gdal_path_vsi_passthrough(self):
        """Test that /vsi* paths pass through without validation."""
        from raquet.cli import GDALPath

        param_type = GDALPath(exists=True)
        # Should not raise even though file doesn't exist
        result = param_type.convert("/vsigs/bucket/nonexistent.tif", None, None)
        assert result == "/vsigs/bucket/nonexistent.tif"
        assert isinstance(result, str)

    def test_gdal_path_vcurl_passthrough(self):
        """Test that /vsicurl/ paths pass through."""
        from raquet.cli import GDALPath

        param_type = GDALPath(exists=True)
        result = param_type.convert("/vsicurl/https://example.com/file.tif", None, None)
        assert result == "/vsicurl/https://example.com/file.tif"

    def test_gdal_path_vsis3_passthrough(self):
        """Test that /vsis3/ paths pass through."""
        from raquet.cli import GDALPath

        param_type = GDALPath(exists=True)
        result = param_type.convert("/vsis3/bucket/file.tif", None, None)
        assert result == "/vsis3/bucket/file.tif"


# Integration tests that require actual EE credentials
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("EE_PROJECT"),
    reason="EE_PROJECT env var not set for integration tests",
)
class TestEarthEngineIntegration:
    """Integration tests requiring actual EE credentials.

    To run these tests:
    1. Authenticate with Earth Engine: earthengine authenticate
    2. Set environment variables:
       - EE_PROJECT: Your GCP project ID
       - EE_TEST_BUCKET: GCS bucket for test exports
    3. Run: pytest tests/test_earthengine.py -v -m integration
    """

    def test_initialize_ee(self):
        """Test EE initialization."""
        from raquet.earthengine import _initialize_ee

        project = os.environ.get("EE_PROJECT")
        _initialize_ee(project=project)  # Should not raise

    def test_create_ee_image_asset(self):
        """Test creating image from asset ID."""
        from raquet.earthengine import _initialize_ee, create_ee_image

        project = os.environ.get("EE_PROJECT")
        _initialize_ee(project=project)

        image = create_ee_image("COPERNICUS/DEM/GLO30")
        assert image is not None

    def test_create_ee_image_with_bands(self):
        """Test creating image with band selection."""
        from raquet.earthengine import _initialize_ee, create_ee_image

        project = os.environ.get("EE_PROJECT")
        _initialize_ee(project=project)

        image = create_ee_image("COPERNICUS/DEM/GLO30", bands=["DEM"])
        assert image is not None
