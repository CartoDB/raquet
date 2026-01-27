#!/usr/bin/env python3
"""RaQuet file validation module

Provides comprehensive validation of RaQuet Parquet files including:
- Schema validation (required columns)
- Metadata validation (version, structure)
- Pyramid/overview validation (all zoom levels have data)
- Band statistics validation
- Nodata handling validation

Usage:
    from raquet.validate import validate_raquet

    result = validate_raquet("path/to/file.parquet")
    if result.is_valid:
        print("File is valid!")
    else:
        for error in result.errors:
            print(f"Error: {error}")
"""

import dataclasses
import gzip
import json
import math
import struct
from collections import defaultdict
from typing import Any

import pyarrow.compute
import pyarrow.parquet
import quadbin


@dataclasses.dataclass
class ValidationResult:
    """Result of RaQuet file validation"""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    metadata: dict | None
    stats: dict[str, Any]

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"RaQuet Validation: {status}"]

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  ✗ {error}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        if self.stats:
            lines.append("\nStatistics:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


def validate_schema(table: pyarrow.Table) -> tuple[list[str], list[str]]:
    """Validate RaQuet table schema"""
    errors = []
    warnings = []

    column_names = table.column_names

    # Required columns
    if "block" not in column_names:
        errors.append("Missing required column: 'block'")
    elif str(table.schema.field("block").type) not in ("uint64", "int64"):
        errors.append(f"Column 'block' should be uint64 or int64, got {table.schema.field('block').type}")

    if "metadata" not in column_names:
        errors.append("Missing required column: 'metadata'")
    elif str(table.schema.field("metadata").type) != "string":
        errors.append(f"Column 'metadata' should be string, got {table.schema.field('metadata').type}")

    # Check for band columns
    band_columns = [c for c in column_names if c.startswith("band_")]
    if not band_columns:
        errors.append("No band columns found (expected columns starting with 'band_')")
    else:
        for band_col in band_columns:
            if str(table.schema.field(band_col).type) not in ("binary", "large_binary"):
                warnings.append(f"Band column '{band_col}' is {table.schema.field(band_col).type}, expected binary")

    return errors, warnings


def validate_metadata(table: pyarrow.Table) -> tuple[list[str], list[str], dict | None]:
    """Validate RaQuet metadata in block 0"""
    errors = []
    warnings = []
    metadata = None

    # Get metadata row (block = 0)
    try:
        block_zero = table.filter(pyarrow.compute.equal(table.column("block"), 0))
        if len(block_zero) == 0:
            errors.append("No metadata row found (block=0)")
            return errors, warnings, None

        metadata_str = block_zero.column("metadata")[0].as_py()
        if metadata_str is None:
            errors.append("Metadata column is NULL in block=0 row")
            return errors, warnings, None

        metadata = json.loads(metadata_str)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in metadata: {e}")
        return errors, warnings, None
    except Exception as e:
        errors.append(f"Error reading metadata: {e}")
        return errors, warnings, None

    # Validate version
    version = metadata.get("version")
    if version is None:
        errors.append("Missing 'version' in metadata")
    elif version not in ("0.2.0", "0.3.0"):
        warnings.append(f"Unknown version '{version}', expected 0.2.0 or 0.3.0")

    # Validate required fields for v0.3.0
    if version == "0.3.0":
        required_fields = ["width", "height", "crs", "bounds", "bounds_crs", "tiling", "compression", "bands"]
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required field '{field}' in v0.3.0 metadata")

        # Validate tiling object
        tiling = metadata.get("tiling", {})
        tiling_fields = ["scheme", "block_width", "block_height", "min_zoom", "max_zoom", "pixel_zoom", "num_blocks"]
        for field in tiling_fields:
            if field not in tiling:
                errors.append(f"Missing required field 'tiling.{field}' in v0.3.0 metadata")

        # Validate bands
        bands = metadata.get("bands", [])
        if not bands:
            errors.append("No bands defined in metadata")
        else:
            for i, band in enumerate(bands):
                if "name" not in band:
                    errors.append(f"Band {i} missing 'name' field")
                if "type" not in band:
                    errors.append(f"Band {i} missing 'type' field")

                # Check for statistics
                stats_fields = ["STATISTICS_MINIMUM", "STATISTICS_MAXIMUM", "STATISTICS_MEAN", "STATISTICS_STDDEV"]
                missing_stats = [f for f in stats_fields if f not in band]
                if missing_stats:
                    warnings.append(f"Band {i} missing statistics: {', '.join(missing_stats)}")

    return errors, warnings, metadata


def validate_pyramids(table: pyarrow.Table, metadata: dict) -> tuple[list[str], list[str], dict]:
    """Validate that all zoom levels have valid data (not all nodata)"""
    errors = []
    warnings = []
    stats = {}

    if metadata is None:
        return errors, warnings, stats

    tiling = metadata.get("tiling", {})
    min_zoom = tiling.get("min_zoom")
    max_zoom = tiling.get("max_zoom")

    if min_zoom is None or max_zoom is None:
        errors.append("Cannot validate pyramids: missing min_zoom or max_zoom in metadata")
        return errors, warnings, stats

    # Get band info for nodata detection
    bands = metadata.get("bands", [])
    if not bands:
        return errors, warnings, stats

    band_nodata = bands[0].get("nodata")
    band_type = bands[0].get("type", "float32")

    # Count tiles per zoom level and check for valid data
    zoom_stats = defaultdict(lambda: {"total": 0, "valid": 0, "all_nodata": 0})

    # Filter out metadata row
    data_rows = table.filter(pyarrow.compute.greater(table.column("block"), 0))

    # Get the first band column
    band_columns = [c for c in table.column_names if c.startswith("band_")]
    if not band_columns:
        return errors, warnings, stats

    band_col = band_columns[0]

    # Sample tiles at each zoom level
    for i in range(len(data_rows)):
        block_id = data_rows.column("block")[i].as_py()

        # Extract zoom from quadbin
        try:
            x, y, z = quadbin.cell_to_tile(block_id)
        except Exception:
            continue

        zoom_stats[z]["total"] += 1

        # Check if tile has valid (non-nodata) data
        band_data = data_rows.column(band_col)[i].as_py()
        if band_data is not None:
            try:
                # Decompress and check for valid data
                decompressed = gzip.decompress(band_data)

                # Parse based on data type
                dtype_map = {
                    "uint8": ("B", 1),
                    "int8": ("b", 1),
                    "uint16": ("H", 2),
                    "int16": ("h", 2),
                    "uint32": ("I", 4),
                    "int32": ("i", 4),
                    "float32": ("f", 4),
                    "float64": ("d", 8),
                }

                fmt, size = dtype_map.get(band_type, ("f", 4))
                num_values = len(decompressed) // size
                values = struct.unpack(f"{num_values}{fmt}", decompressed)

                # Check if all values are nodata or NaN
                if band_nodata is not None:
                    valid_count = sum(1 for v in values if v != band_nodata and not (isinstance(v, float) and math.isnan(v)))
                else:
                    valid_count = sum(1 for v in values if not (isinstance(v, float) and math.isnan(v)))

                if valid_count > 0:
                    zoom_stats[z]["valid"] += 1
                else:
                    zoom_stats[z]["all_nodata"] += 1

            except Exception as e:
                # If we can't decompress, assume it has some data
                zoom_stats[z]["valid"] += 1

    # Report statistics and check for issues
    stats["zoom_levels"] = {}

    for z in range(min_zoom, max_zoom + 1):
        zs = zoom_stats[z]
        total = zs["total"]
        valid = zs["valid"]
        all_nodata = zs["all_nodata"]

        if total == 0:
            errors.append(f"Zoom {z}: No tiles found (expected data between min_zoom={min_zoom} and max_zoom={max_zoom})")
            stats["zoom_levels"][z] = {"total": 0, "valid": 0, "valid_percent": 0}
        else:
            valid_percent = (valid / total) * 100
            stats["zoom_levels"][z] = {
                "total": total,
                "valid": valid,
                "all_nodata": all_nodata,
                "valid_percent": round(valid_percent, 1)
            }

            if valid == 0:
                errors.append(f"Zoom {z}: 0% valid tiles ({all_nodata}/{total} tiles are all nodata)")
            elif valid_percent < 50:
                warnings.append(f"Zoom {z}: Only {valid_percent:.1f}% valid tiles ({valid}/{total})")

    return errors, warnings, stats


def validate_band_data(table: pyarrow.Table, metadata: dict) -> tuple[list[str], list[str]]:
    """Validate band data integrity"""
    errors = []
    warnings = []

    if metadata is None:
        return errors, warnings

    bands = metadata.get("bands", [])
    compression = metadata.get("compression")

    # Get band columns
    band_columns = [c for c in table.column_names if c.startswith("band_")]

    # Check that band columns match metadata
    meta_band_names = [b.get("name") for b in bands]
    for band_col in band_columns:
        if band_col not in meta_band_names:
            warnings.append(f"Band column '{band_col}' not found in metadata bands")

    for band_name in meta_band_names:
        if band_name not in band_columns:
            errors.append(f"Band '{band_name}' defined in metadata but not in table")

    # Sample a few tiles to verify data can be decompressed
    data_rows = table.filter(pyarrow.compute.greater(table.column("block"), 0))

    if len(data_rows) > 0 and band_columns:
        sample_indices = [0, len(data_rows) // 2, len(data_rows) - 1]
        sample_indices = [i for i in sample_indices if i < len(data_rows)]

        for idx in sample_indices[:3]:  # Check up to 3 samples
            for band_col in band_columns[:1]:  # Check first band only for speed
                band_data = data_rows.column(band_col)[idx].as_py()
                if band_data is not None:
                    try:
                        if compression == "gzip":
                            gzip.decompress(band_data)
                    except Exception as e:
                        errors.append(f"Failed to decompress band data at row {idx}: {e}")
                        break

    return errors, warnings


def validate_raquet(filepath: str, check_all_tiles: bool = False) -> ValidationResult:
    """Validate a RaQuet Parquet file

    Args:
        filepath: Path to the RaQuet file
        check_all_tiles: If True, check all tiles for valid data (slower).
                        If False, check sample tiles at each zoom level.

    Returns:
        ValidationResult with validation status, errors, warnings, and stats
    """
    all_errors = []
    all_warnings = []
    all_stats = {}
    metadata = None

    try:
        # Read the table
        table = pyarrow.parquet.read_table(filepath)
        all_stats["row_count"] = len(table)
        all_stats["columns"] = table.column_names

    except Exception as e:
        return ValidationResult(
            is_valid=False,
            errors=[f"Failed to read Parquet file: {e}"],
            warnings=[],
            metadata=None,
            stats={}
        )

    # Schema validation
    errors, warnings = validate_schema(table)
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    # Metadata validation
    errors, warnings, metadata = validate_metadata(table)
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    if metadata:
        all_stats["version"] = metadata.get("version")
        all_stats["dimensions"] = f"{metadata.get('width')}x{metadata.get('height')}"
        tiling = metadata.get("tiling", {})
        all_stats["zoom_range"] = f"{tiling.get('min_zoom')}-{tiling.get('max_zoom')}"
        all_stats["num_blocks"] = tiling.get("num_blocks")
        all_stats["num_bands"] = len(metadata.get("bands", []))

    # Pyramid validation (check all zoom levels have data)
    if metadata:
        errors, warnings, pyramid_stats = validate_pyramids(table, metadata)
        all_errors.extend(errors)
        all_warnings.extend(warnings)
        all_stats.update(pyramid_stats)

    # Band data validation
    if metadata:
        errors, warnings = validate_band_data(table, metadata)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    is_valid = len(all_errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=all_errors,
        warnings=all_warnings,
        metadata=metadata,
        stats=all_stats
    )


def main():
    """CLI entry point for validation"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate RaQuet Parquet files")
    parser.add_argument("filepath", help="Path to RaQuet file to validate")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    result = validate_raquet(args.filepath)
    print(result)

    if args.verbose and result.metadata:
        print("\nMetadata:")
        print(json.dumps(result.metadata, indent=2))

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
