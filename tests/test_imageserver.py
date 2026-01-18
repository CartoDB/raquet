#!/usr/bin/env python3
"""Tests for the ImageServer conversion module."""

import pytest

# Import will fail if dependencies not installed
try:
    from raquet.imageserver import (
        _calculate_target_resolution,
        _get_crs_from_spatial_reference,
        _transform_bounds_with_crs,
        ImageServerMetadata,
    )

    HAS_IMAGESERVER = True
except ImportError:
    HAS_IMAGESERVER = False


pytestmark = pytest.mark.skipif(
    not HAS_IMAGESERVER,
    reason="ImageServer dependencies not installed (pip install raquet[imageserver])",
)


class TestCRSParsing:
    """Tests for CRS parsing from ArcGIS spatial references."""

    def test_epsg_wkid(self):
        """Test parsing EPSG code from wkid."""
        spatial_ref = {"wkid": 4326}
        result = _get_crs_from_spatial_reference(spatial_ref)
        assert result == "EPSG:4326"

    def test_web_mercator_102100(self):
        """Test parsing Web Mercator 102100 (Esri code)."""
        spatial_ref = {"wkid": 102100}
        result = _get_crs_from_spatial_reference(spatial_ref)
        assert result == "EPSG:3857"

    def test_latest_wkid(self):
        """Test parsing latestWkid when present."""
        spatial_ref = {"wkid": 102100, "latestWkid": 3857}
        result = _get_crs_from_spatial_reference(spatial_ref)
        assert result == "EPSG:3857"

    def test_wkt_fallback(self):
        """Test WKT fallback when no wkid."""
        wkt = 'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]]'
        spatial_ref = {"wkt": wkt}
        result = _get_crs_from_spatial_reference(spatial_ref)
        assert result == wkt

    def test_default_to_wgs84(self):
        """Test default to WGS84 when nothing specified."""
        spatial_ref = {}
        result = _get_crs_from_spatial_reference(spatial_ref)
        assert result == "EPSG:4326"


class TestBoundsTransform:
    """Tests for bounds transformation."""

    def test_wgs84_to_web_mercator(self):
        """Test transforming WGS84 bounds to Web Mercator."""
        bounds = (-122.5, 37.5, -122.0, 38.0)
        result = _transform_bounds_with_crs(bounds, "EPSG:4326", "EPSG:3857")

        # Web Mercator x values should be negative (west)
        assert result[0] < 0
        assert result[2] < 0
        # Web Mercator y values should be positive (north)
        assert result[1] > 0
        assert result[3] > 0
        # Bounds should maintain ordering
        assert result[0] < result[2]
        assert result[1] < result[3]

    def test_web_mercator_to_wgs84(self):
        """Test transforming Web Mercator bounds to WGS84."""
        # Approximate Web Mercator coords for San Francisco area
        bounds = (-13639000, 4526000, -13583000, 4593000)
        result = _transform_bounds_with_crs(bounds, "EPSG:3857", "EPSG:4326")

        # Should be valid WGS84 coordinates
        assert -180 <= result[0] <= 180
        assert -90 <= result[1] <= 90
        assert -180 <= result[2] <= 180
        assert -90 <= result[3] <= 90


class TestResolutionCalculation:
    """Tests for QUADBIN resolution calculation."""

    def test_high_resolution(self):
        """Test calculation for high resolution imagery."""
        # Small area, high detail
        bounds = (-13630000, 4530000, -13620000, 4540000)  # ~10km square
        width = 10000  # 1m resolution
        height = 10000
        block_size = 256

        result = _calculate_target_resolution(bounds, width, height, block_size)

        # Should be a high resolution (QUADBIN pixel resolution 0-26)
        assert 15 <= result <= 26

    def test_low_resolution(self):
        """Test calculation for low resolution imagery."""
        # Large area, low detail
        bounds = (-20000000, -20000000, 20000000, 20000000)  # Global
        width = 1000
        height = 1000
        block_size = 256

        result = _calculate_target_resolution(bounds, width, height, block_size)

        # Should be a low zoom level
        assert 0 <= result <= 10

    def test_different_block_sizes(self):
        """Test that larger blocks result in lower resolution."""
        bounds = (-13630000, 4530000, -13620000, 4540000)
        width = 5000
        height = 5000

        res_256 = _calculate_target_resolution(bounds, width, height, 256)
        res_512 = _calculate_target_resolution(bounds, width, height, 512)

        # Larger blocks should result in same or lower resolution
        assert res_512 <= res_256


class TestImageServerMetadata:
    """Tests for ImageServerMetadata dataclass."""

    def test_dataclass_creation(self):
        """Test creating ImageServerMetadata instance."""
        metadata = ImageServerMetadata(
            name="Test Service",
            description="Test description",
            extent={"xmin": -180, "ymin": -90, "xmax": 180, "ymax": 90},
            spatial_reference={"wkid": 4326},
            pixel_type="uint8",
            band_count=3,
            min_values=[0, 0, 0],
            max_values=[255, 255, 255],
            mean_values=[128, 128, 128],
            stddev_values=[50, 50, 50],
            nodata=0,
            pixel_size_x=0.001,
            pixel_size_y=0.001,
            rows=1000,
            columns=1000,
        )

        assert metadata.name == "Test Service"
        assert metadata.band_count == 3
        assert metadata.pixel_type == "uint8"
        assert metadata.nodata == 0
