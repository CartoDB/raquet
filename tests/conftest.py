#!/usr/bin/env python3
"""Pytest configuration and shared fixtures."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_data_dir():
    """Return the test data directory."""
    return Path(__file__).parent


@pytest.fixture
def sample_geotiffs(test_data_dir):
    """Return list of sample GeoTIFF files."""
    tiffs = list(test_data_dir.glob("*.tif")) + list(test_data_dir.glob("*.tiff"))
    return tiffs


@pytest.fixture
def example_parquet():
    """Return path to example parquet file if it exists."""
    example = Path(__file__).parent.parent / "examples" / "example_data.parquet"
    if example.exists():
        return example
    return None
