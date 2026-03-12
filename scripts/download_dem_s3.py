#!/usr/bin/env python3
"""
Download USGS 3DEP 1-meter DEM tiles directly from the AWS S3 public bucket.

Bypasses the TNM API (which can be unreliable) and lists tiles directly from:
  s3://prd-tnm/StagedProducts/Elevation/1m/Projects/

Usage:
    .venv/bin/python scripts/download_dem_s3.py --max-gb 15
    .venv/bin/python scripts/download_dem_s3.py --max-gb 15 --bbox "-78.0,38.5,-76.8,39.5"
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen, Request

S3_BASE = "https://prd-tnm.s3.amazonaws.com"
S3_PREFIX = "StagedProducts/Elevation/1m/Projects"

# Projects covering the DC metro area (Northern Virginia + Maryland)
DC_METRO_PROJECTS = [
    "VA_NorthernVA_B22",
    "VA_Fairfax_County_2018",
    "MD_VA_Sandy_NCR_2014",
    "MD_VA_NorthChesapeakeBay_KGeorge_2020_D20",
]


def list_tiles_in_project(project: str) -> list[dict]:
    """List all GeoTIFF tiles in a project on S3."""
    prefix = f"{S3_PREFIX}/{project}/TIFF/"
    tiles = []
    continuation_token = None

    while True:
        url = f"{S3_BASE}/?list-type=2&prefix={prefix}&max-keys=1000"
        if continuation_token:
            url += f"&continuation-token={continuation_token}"

        resp = urlopen(url, timeout=60)
        xml_data = resp.read().decode()
        root = ET.fromstring(xml_data)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

        for content in root.findall(".//s3:Contents", ns):
            key = content.find("s3:Key", ns).text
            if key.endswith(".tif"):
                size = int(content.find("s3:Size", ns).text)
                tiles.append({
                    "key": key,
                    "url": f"{S3_BASE}/{key}",
                    "size": size,
                    "project": project,
                    "filename": key.split("/")[-1],
                })

        # Check for pagination
        is_truncated = root.find(".//s3:IsTruncated", ns)
        if is_truncated is not None and is_truncated.text == "true":
            next_token = root.find(".//s3:NextContinuationToken", ns)
            if next_token is not None:
                continuation_token = next_token.text
            else:
                break
        else:
            break

    return tiles


def filter_by_bbox(tiles: list[dict], bbox: str) -> list[dict]:
    """Filter tiles by bounding box using gdalinfo (if available) or filename heuristics."""
    # For USGS 1m tiles, the filename contains grid coordinates like x25y431
    # These don't directly map to lat/lon, so we keep all tiles from relevant projects
    # The actual bbox filtering happens at the merge/query stage
    return tiles


def select_tiles(tiles: list[dict], max_gb: float) -> list[dict]:
    """Select tiles up to the target size, preferring newer projects."""
    # Sort by project name (newer projects have later dates) then by filename
    tiles.sort(key=lambda t: (t["project"], t["filename"]))

    selected = []
    total_bytes = 0
    max_bytes = max_gb * 1024**3

    for tile in tiles:
        if total_bytes + tile["size"] > max_bytes:
            continue
        selected.append(tile)
        total_bytes += tile["size"]

    return selected


def download_tile(tile: dict, output_dir: Path) -> Path:
    """Download a single tile."""
    filepath = output_dir / tile["filename"]

    # Skip if already downloaded with correct size
    if filepath.exists() and filepath.stat().st_size == tile["size"]:
        print(f"  {tile['filename']}: already downloaded, skipping")
        return filepath

    req = Request(tile["url"])
    with urlopen(req, timeout=300) as resp:
        total = tile["size"]
        downloaded = 0
        chunk_size = 131072

        with open(filepath, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {tile['filename']}: {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
    print()
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Download USGS 3DEP 1m DEM tiles from S3")
    parser.add_argument("--max-gb", type=float, default=15.0,
                        help="Maximum total download size in GB (default: 15)")
    parser.add_argument("--output-dir", type=str, default="data/15gb/dem_tiles",
                        help="Output directory for tiles")
    parser.add_argument("--bbox", type=str, default=None,
                        help="Bounding box (currently unused, kept for compatibility)")
    parser.add_argument("--projects", type=str, nargs="+", default=DC_METRO_PROJECTS,
                        help="S3 project names to download from")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list available tiles, don't download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # List tiles from all projects
    all_tiles = []
    for project in args.projects:
        print(f"Listing tiles in {project}...")
        tiles = list_tiles_in_project(project)
        size_gb = sum(t["size"] for t in tiles) / 1024**3
        print(f"  Found {len(tiles)} tiles ({size_gb:.1f} GB)")
        all_tiles.extend(tiles)

    total_gb = sum(t["size"] for t in all_tiles) / 1024**3
    print(f"\nTotal available: {len(all_tiles)} tiles, {total_gb:.1f} GB")

    # Select tiles within budget
    selected = select_tiles(all_tiles, args.max_gb)
    selected_gb = sum(t["size"] for t in selected) / 1024**3
    print(f"Selected: {len(selected)} tiles, {selected_gb:.1f} GB (max: {args.max_gb} GB)")

    if args.list_only:
        print("\nTiles:")
        for t in selected:
            print(f"  {t['filename']} ({t['size'] / 1e6:.0f} MB) [{t['project']}]")
        return

    # Download
    print(f"\nDownloading {len(selected)} tiles to {output_dir}/...")
    manifest = []
    for i, tile in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {tile['filename']} ({tile['size'] / 1e6:.0f} MB)")
        try:
            filepath = download_tile(tile, output_dir)
            manifest.append({
                "title": tile["filename"],
                "file": str(filepath),
                "url": tile["url"],
                "project": tile["project"],
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone! Downloaded {len(manifest)} tiles. Manifest: {manifest_path}")

    actual_size = sum(p.stat().st_size for p in output_dir.glob("*.tif")) / 1024**3
    print(f"Total on disk: {actual_size:.2f} GB")


if __name__ == "__main__":
    main()
