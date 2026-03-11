#!/usr/bin/env python3
"""
Merge downloaded DEM tiles and compute slope raster.

Usage:
    python scripts/prepare_dem.py [--tiles-dir data/dem_tiles] [--output-dir data]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_tiles(tiles_dir: Path) -> list[Path]:
    """Find all GeoTIFF tiles in the directory."""
    tiles = sorted(tiles_dir.glob("*.tif"))
    if not tiles:
        print(f"No .tif files found in {tiles_dir}")
        sys.exit(1)
    total_gb = sum(t.stat().st_size for t in tiles) / 1024**3
    print(f"Found {len(tiles)} tiles ({total_gb:.2f} GB)")
    return tiles


def merge_tiles(tiles: list[Path], output_path: Path):
    """Merge tiles into a single GeoTIFF using gdal_merge.py."""
    if output_path.exists():
        size_gb = output_path.stat().st_size / 1024**3
        print(f"Merged file already exists: {output_path} ({size_gb:.2f} GB)")
        response = input("Overwrite? [y/N] ").strip().lower()
        if response != "y":
            return

    print(f"Merging {len(tiles)} tiles into {output_path}...")
    print("  This may take a while for large datasets...")

    # Use gdal_merge.py with COG output for efficiency
    cmd = [
        "gdal_merge.py",
        "-o", str(output_path),
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=512",
        "-co", "BLOCKYSIZE=512",
        "-co", "BIGTIFF=YES",
    ] + [str(t) for t in tiles]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: gdal_merge.py failed:\n{result.stderr}")
        sys.exit(1)

    size_gb = output_path.stat().st_size / 1024**3
    print(f"  Merged: {output_path} ({size_gb:.2f} GB)")


def compute_slope(elevation_path: Path, slope_path: Path):
    """Compute slope in degrees using gdaldem."""
    if slope_path.exists():
        size_gb = slope_path.stat().st_size / 1024**3
        print(f"Slope file already exists: {slope_path} ({size_gb:.2f} GB)")
        response = input("Overwrite? [y/N] ").strip().lower()
        if response != "y":
            return

    print(f"Computing slope from {elevation_path}...")
    print("  Output: slope in degrees (0=flat, 90=vertical)")

    cmd = [
        "gdaldem", "slope",
        str(elevation_path),
        str(slope_path),
        "-compute_edges",
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-co", "BLOCKXSIZE=512",
        "-co", "BLOCKYSIZE=512",
        "-co", "BIGTIFF=YES",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: gdaldem failed:\n{result.stderr}")
        sys.exit(1)

    size_gb = slope_path.stat().st_size / 1024**3
    print(f"  Slope: {slope_path} ({size_gb:.2f} GB)")


def mask_slope_nodata(slope_path: Path, elevation_path: Path, output_path: Path):
    """Mask slope pixels where elevation is 0 (merge gaps) as nodata."""
    if output_path.exists():
        size_gb = output_path.stat().st_size / 1024**3
        print(f"Masked slope file already exists: {output_path} ({size_gb:.2f} GB)")
        response = input("Overwrite? [y/N] ").strip().lower()
        if response != "y":
            return

    print(f"Masking slope nodata (where elevation == 0)...")
    print("  Setting slope = -9999 where elevation = 0 (tile gaps)")

    cmd = [
        "gdal_calc.py",
        "-A", str(slope_path),
        "-B", str(elevation_path),
        f"--outfile={output_path}",
        '--calc=numpy.where(B==0, -9999, A)',
        "--NoDataValue=-9999",
        "--co=COMPRESS=LZW",
        "--co=TILED=YES",
        "--co=BLOCKXSIZE=512",
        "--co=BLOCKYSIZE=512",
        "--co=BIGTIFF=YES",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: gdal_calc.py failed:\n{result.stderr}")
        sys.exit(1)

    size_gb = output_path.stat().st_size / 1024**3
    print(f"  Masked slope: {output_path} ({size_gb:.2f} GB)")


def show_info(path: Path):
    """Show gdalinfo summary."""
    print(f"\n--- {path.name} ---")
    result = subprocess.run(["gdalinfo", str(path)], capture_output=True, text=True)
    if result.returncode == 0:
        # Print key lines: size, CRS, bounds, type
        for line in result.stdout.splitlines():
            line_stripped = line.strip()
            if any(kw in line_stripped for kw in ["Size is", "Coordinate System", "Origin", "Pixel Size", "Type=", "Upper Left", "Lower Right"]):
                print(f"  {line_stripped}")


def main():
    parser = argparse.ArgumentParser(description="Merge DEM tiles and compute slope")
    parser.add_argument("--tiles-dir", type=str, default="data/dem_tiles",
                        help="Directory containing DEM tiles (default: data/dem_tiles)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip merging (use existing elevation.tif)")
    parser.add_argument("--skip-slope", action="store_true",
                        help="Skip slope computation")
    args = parser.parse_args()

    tiles_dir = Path(args.tiles_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    elevation_path = output_dir / "elevation.tif"
    slope_raw_path = output_dir / "slope_raw.tif"
    slope_path = output_dir / "slope.tif"

    # Step 1: Merge tiles
    if not args.skip_merge:
        tiles = find_tiles(tiles_dir)
        merge_tiles(tiles, elevation_path)
    else:
        if not elevation_path.exists():
            print(f"ERROR: {elevation_path} not found (--skip-merge requires existing file)")
            sys.exit(1)
        print(f"Skipping merge, using existing {elevation_path}")

    show_info(elevation_path)

    # Step 2: Compute slope
    if not args.skip_slope:
        compute_slope(elevation_path, slope_raw_path)
        show_info(slope_raw_path)

        # Step 3: Mask nodata (where elevation == 0, i.e. tile gaps)
        mask_slope_nodata(slope_raw_path, elevation_path, slope_path)
        show_info(slope_path)

    # Summary
    print("\n=== Summary ===")
    for p in [elevation_path, slope_path]:
        if p.exists():
            size_gb = p.stat().st_size / 1024**3
            print(f"  {p.name}: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
