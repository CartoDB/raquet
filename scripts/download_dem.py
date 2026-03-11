#!/usr/bin/env python3
"""
Download USGS 3DEP 1-meter DEM tiles for Loudoun County, Virginia
via The National Map (TNM) API.

Usage:
    uv run python scripts/download_dem.py [--max-gb 8] [--output-dir data/dem_tiles]
"""

import argparse
import gzip
import json
import sys
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError


def fetch_json(url: str) -> dict:
    """Fetch JSON from URL, handling gzip encoding transparently."""
    req = Request(url)
    req.add_header("Accept-Encoding", "identity")
    with urlopen(req, timeout=30) as resp:
        raw = resp.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Server may send gzip despite asking for identity
        return json.loads(gzip.decompress(raw))

# Loudoun County, Virginia bounding box (WGS84)
# Covers the "Data Center Alley" around Ashburn
BBOX = "-77.8,38.85,-77.3,39.35"

TNM_API_URL = "https://tnmaccess.nationalmap.gov/api/v1/products"
DATASET = "Digital Elevation Model (DEM) 1 meter"


def query_tiles(bbox: str, max_results: int = 200) -> list[dict]:
    """Query TNM API for available 1m DEM tiles within bounding box."""
    params = {
        "datasets": DATASET,
        "bbox": bbox,
        "prodFormats": "GeoTIFF",
        "max": str(max_results),
        "offset": "0",
    }
    url = f"{TNM_API_URL}?{urlencode(params, safe=',()')}"
    print(f"Querying TNM API for 1m DEM tiles in bbox: {bbox}")

    data = fetch_json(url)

    total = data.get("total", 0)
    items = data.get("items", [])
    print(f"Found {total} tiles, retrieved {len(items)}")

    # Paginate if needed
    while len(items) < total:
        params["offset"] = str(len(items))
        url = f"{TNM_API_URL}?{urlencode(params, safe=',()')}"
        page = fetch_json(url)
        new_items = page.get("items", [])
        if not new_items:
            break
        items.extend(new_items)
        print(f"  Retrieved {len(items)}/{total} tiles...")

    return items


def deduplicate_tiles(tiles: list[dict]) -> list[dict]:
    """When multiple tiles cover the same grid cell, keep the newest/largest one."""
    # Group by grid cell (extract x/y coordinates from title like "x25y431")
    import re
    cell_map: dict[str, dict] = {}
    for tile in tiles:
        # Extract grid coordinates from title
        match = re.search(r'x(\d+)y(\d+)', tile.get("title", ""))
        if not match:
            continue
        cell_key = f"x{match.group(1)}y{match.group(2)}"
        existing = cell_map.get(cell_key)
        if existing is None:
            cell_map[cell_key] = tile
        else:
            # Prefer newer publication date, then larger file (more data)
            new_date = tile.get("publicationDate", "")
            old_date = existing.get("publicationDate", "")
            if new_date > old_date:
                cell_map[cell_key] = tile
            elif new_date == old_date and tile.get("sizeInBytes", 0) > existing.get("sizeInBytes", 0):
                cell_map[cell_key] = tile

    deduped = list(cell_map.values())
    print(f"Deduplicated: {len(tiles)} tiles -> {len(deduped)} unique grid cells")
    return deduped


def select_tiles(tiles: list[dict], max_gb: float) -> list[dict]:
    """Select tiles up to the target size, sorted by location for spatial coherence."""
    tiles = deduplicate_tiles(tiles)
    tiles.sort(key=lambda t: (t["boundingBox"]["minY"], t["boundingBox"]["minX"]))

    selected = []
    total_bytes = 0
    max_bytes = max_gb * 1024**3

    for tile in tiles:
        size = tile.get("sizeInBytes", 0)
        if total_bytes + size > max_bytes:
            continue
        selected.append(tile)
        total_bytes += size

    return selected


def download_tile(tile: dict, output_dir: Path) -> Path:
    """Download a single tile."""
    url = tile["downloadURL"]
    filename = url.split("/")[-1]
    filepath = output_dir / filename

    # Skip if already downloaded with correct size
    expected_size = tile.get("sizeInBytes", 0)
    if filepath.exists() and filepath.stat().st_size == expected_size:
        print(f"  {filename}: already downloaded, skipping")
        return filepath

    # Download with progress
    req = Request(url)
    with urlopen(req, timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 131072  # 128KB

        with open(filepath, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {filename}: {downloaded / 1e6:.0f}/{total / 1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
    print()
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Download USGS 3DEP 1m DEM tiles")
    parser.add_argument("--max-gb", type=float, default=8.0,
                        help="Maximum total download size in GB (default: 8)")
    parser.add_argument("--output-dir", type=str, default="data/dem_tiles",
                        help="Output directory for tiles (default: data/dem_tiles)")
    parser.add_argument("--bbox", type=str, default=BBOX,
                        help=f"Bounding box as 'west,south,east,north' (default: {BBOX})")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list available tiles, don't download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Query available tiles
    tiles = query_tiles(args.bbox)
    if not tiles:
        print("No tiles found. Check bbox and try again.")
        sys.exit(1)

    total_size_gb = sum(t.get("sizeInBytes", 0) for t in tiles) / 1024**3
    print(f"\nTotal available: {len(tiles)} tiles, {total_size_gb:.1f} GB")

    # Select tiles within budget
    selected = select_tiles(tiles, args.max_gb)
    selected_size_gb = sum(t.get("sizeInBytes", 0) for t in selected) / 1024**3
    print(f"Selected: {len(selected)} tiles, {selected_size_gb:.1f} GB (max: {args.max_gb} GB)")

    if args.list_only:
        print("\nTiles:")
        for t in selected:
            bb = t["boundingBox"]
            size_mb = t.get("sizeInBytes", 0) / 1e6
            print(f"  {t['title']} ({size_mb:.0f} MB) [{bb['minX']:.3f},{bb['minY']:.3f} -> {bb['maxX']:.3f},{bb['maxY']:.3f}]")
        return

    # Download tiles
    print(f"\nDownloading {len(selected)} tiles to {output_dir}/...")
    manifest = []
    for i, tile in enumerate(selected, 1):
        title = tile["title"]
        size_mb = tile.get("sizeInBytes", 0) / 1e6
        print(f"[{i}/{len(selected)}] {title} ({size_mb:.0f} MB)")
        try:
            filepath = download_tile(tile, output_dir)
            manifest.append({"title": title, "file": str(filepath), "url": tile["downloadURL"]})
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone! Downloaded {len(manifest)} tiles. Manifest: {manifest_path}")

    # Summary
    actual_size = sum(p.stat().st_size for p in output_dir.glob("*.tif")) / 1024**3
    print(f"Total on disk: {actual_size:.2f} GB")


if __name__ == "__main__":
    main()
