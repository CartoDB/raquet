#!/usr/bin/env python3
"""
ArcGIS ImageServer to Raquet conversion functionality.

This module provides functions to convert ArcGIS ImageServer services
to Raquet parquet format with QUADBIN spatial indexing.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
from dataclasses import dataclass
from typing import Any

import mercantile
import pyarrow as pa
import pyarrow.parquet as pq
import quadbin

# Constants
RAQUET_VERSION = "0.1.0"
METADATA_BLOCK_ID = 0
DEFAULT_BLOCK_SIZE = 256


@dataclass
class ImageServerMetadata:
    """Metadata about an ArcGIS ImageServer service."""

    name: str
    description: str
    extent: dict
    spatial_reference: dict
    pixel_type: str
    band_count: int
    min_values: list[float] | None
    max_values: list[float] | None
    mean_values: list[float] | None
    stddev_values: list[float] | None
    nodata: float | int | None
    pixel_size_x: float
    pixel_size_y: float
    rows: int
    columns: int


def _get_http_client():
    """Get HTTP client for making requests."""
    try:
        import httpx

        return httpx.Client(timeout=120.0, follow_redirects=True)
    except ImportError as e:
        raise ValueError(
            "httpx is required for ImageServer conversion. Install with: pip install httpx"
        ) from e


def _make_imageserver_request(
    url: str,
    params: dict | None = None,
    token: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    return_bytes: bool = False,
) -> dict | bytes:
    """Make HTTP request to ImageServer with retry logic."""
    import httpx

    if params is None:
        params = {}

    if token:
        params["token"] = token

    last_exception = None

    for attempt in range(max_retries):
        try:
            with _get_http_client() as client:
                response = client.get(url, params=params)
                response.raise_for_status()

                if return_bytes:
                    return response.content
                return response.json()
        except httpx.TimeoutException as e:
            last_exception = e
            if attempt < max_retries - 1:
                import time

                time.sleep(retry_delay * (attempt + 1))
        except httpx.NetworkError as e:
            last_exception = e
            if attempt < max_retries - 1:
                import time

                time.sleep(retry_delay * (attempt + 1))
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429 or (500 <= status < 600):
                last_exception = e
                if attempt < max_retries - 1:
                    import time

                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        delay = retry_delay * (attempt + 1)
                    time.sleep(delay)
                    continue
            elif status == 401:
                raise ValueError(
                    "Authentication required. Use --token option."
                ) from None
            elif status == 403:
                raise ValueError(
                    "Access denied. Check your credentials and service permissions."
                ) from None
            elif status == 404:
                raise ValueError(f"Service not found (404). Check the URL: {url}") from None
            raise ValueError(f"HTTP error {status}: {e}") from e

    raise ValueError(f"Request failed after {max_retries} attempts: {last_exception}")


def get_imageserver_metadata(
    service_url: str,
    token: str | None = None,
) -> ImageServerMetadata:
    """
    Fetch metadata from an ArcGIS ImageServer service.

    Args:
        service_url: ImageServer REST URL (e.g., .../ImageServer)
        token: Optional authentication token

    Returns:
        ImageServerMetadata with service information
    """
    # Clean URL
    service_url = service_url.rstrip("/")

    logging.info(f"Fetching ImageServer metadata from {service_url}")

    # Fetch service info
    params = {"f": "json"}
    data = _make_imageserver_request(service_url, params, token)

    if "error" in data:
        error = data["error"]
        raise ValueError(f"ImageServer error: {error.get('message', 'Unknown error')}")

    # Parse extent
    extent = data.get("extent", {})
    spatial_ref = data.get("spatialReference", extent.get("spatialReference", {}))

    # Parse pixel type to numpy-compatible type
    pixel_type = data.get("pixelType", "U8")
    pixel_type_map = {
        "U1": "uint8",
        "U2": "uint8",
        "U4": "uint8",
        "U8": "uint8",
        "S8": "int8",
        "U16": "uint16",
        "S16": "int16",
        "U32": "uint32",
        "S32": "int32",
        "F32": "float32",
        "F64": "float64",
    }
    numpy_type = pixel_type_map.get(pixel_type, "float32")

    # Get band statistics if available
    min_vals = data.get("minValues")
    max_vals = data.get("maxValues")
    mean_vals = data.get("meanValues")
    stddev_vals = data.get("stdDevValues")

    # Get pixel size
    pixel_size_x = data.get("pixelSizeX", 1.0)
    pixel_size_y = data.get("pixelSizeY", 1.0)

    # Get nodata value
    nodata = None
    nodata_values = data.get("noDataValues")
    if nodata_values and len(nodata_values) > 0:
        nodata = nodata_values[0]

    # Get rows/columns or calculate from extent and pixel size
    rows = data.get("rows")
    columns = data.get("columns")

    if not rows or not columns:
        # Calculate from extent and pixel size
        xmin = extent.get("xmin", 0)
        xmax = extent.get("xmax", 0)
        ymin = extent.get("ymin", 0)
        ymax = extent.get("ymax", 0)

        if pixel_size_x > 0 and pixel_size_y > 0:
            columns = int((xmax - xmin) / pixel_size_x)
            rows = int((ymax - ymin) / pixel_size_y)
        else:
            columns = 0
            rows = 0

    return ImageServerMetadata(
        name=data.get("name", "Unknown"),
        description=data.get("description", ""),
        extent=extent,
        spatial_reference=spatial_ref,
        pixel_type=numpy_type,
        band_count=data.get("bandCount", 1),
        min_values=min_vals,
        max_values=max_vals,
        mean_values=mean_vals,
        stddev_values=stddev_vals,
        nodata=nodata,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        rows=rows,
        columns=columns,
    )


def _transform_bounds_with_crs(
    bounds: tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str,
) -> tuple[float, float, float, float]:
    """Transform bounds from source to destination CRS (accepts any CRS string)."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    xmin, ymin = transformer.transform(bounds[0], bounds[1])
    xmax, ymax = transformer.transform(bounds[2], bounds[3])

    return (xmin, ymin, xmax, ymax)


def _get_crs_from_spatial_reference(spatial_ref: dict) -> str:
    """Extract CRS string from ArcGIS spatial reference (EPSG or WKT)."""
    # Check for wkid (well-known ID)
    wkid = spatial_ref.get("wkid") or spatial_ref.get("latestWkid")

    if wkid:
        # Handle special cases (Web Mercator variants)
        wkid_to_epsg = {102100: 3857, 102113: 3785}
        epsg = wkid_to_epsg.get(wkid, wkid)
        return f"EPSG:{epsg}"

    # Check for WKT
    wkt = spatial_ref.get("wkt") or spatial_ref.get("wkt2")
    if wkt:
        return wkt

    # Default to WGS84 if no spatial reference
    return "EPSG:4326"


def _calculate_target_resolution(
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    block_size: int,
) -> int:
    """
    Calculate the appropriate QUADBIN resolution for the raster.

    Based on the raster's resolution in Web Mercator coordinates.

    Args:
        bounds: (west, south, east, north) in Web Mercator
        width: Raster width in pixels
        height: Raster height in pixels
        block_size: Block size (256 or 512)

    Returns:
        QUADBIN pixel resolution (0-26)
    """
    # Calculate meters per pixel
    x_res = (bounds[2] - bounds[0]) / width
    y_res = (bounds[3] - bounds[1]) / height
    resolution_m = (x_res + y_res) / 2

    # Web Mercator circumference
    circumference = mercantile.CE  # ~40075016.68 meters

    # Calculate zoom level based on resolution
    block_zoom = int(math.log2(block_size))

    raw_zoom = math.log2(circumference / (resolution_m * block_size))
    zoom = max(0, min(26 - block_zoom, round(raw_zoom)))

    # Pixel resolution is zoom + block_zoom
    return zoom + block_zoom


def _calculate_band_statistics(
    data,
    nodata: float | int | None,
) -> dict[str, Any]:
    """
    Calculate band statistics.

    Args:
        data: Numpy array of pixel values
        nodata: NoData value to exclude

    Returns:
        dict with min, max, mean, stddev, sum, sum_squares, count
    """
    import numpy as np

    # Create mask for valid data
    if nodata is not None:
        mask = (data != nodata) & ~np.isnan(data.astype(float))
    else:
        mask = ~np.isnan(data.astype(float))

    valid_data = data[mask]

    if len(valid_data) == 0:
        return {"count": 0}

    return {
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "stddev": float(np.std(valid_data)),
        "sum": float(np.sum(valid_data)),
        "sum_squares": float(np.sum(valid_data.astype(float) ** 2)),
        "count": int(len(valid_data)),
        "approximated_stats": False,
    }


def _compress_block_data(data, compression: str | None) -> bytes:
    """
    Compress block pixel data.

    Args:
        data: Numpy array (2D) in row-major order
        compression: "gzip" or None

    Returns:
        Binary data, optionally gzip compressed
    """
    import numpy as np

    # Ensure row-major order (C order)
    raw_bytes = np.ascontiguousarray(data).tobytes()

    if compression == "gzip":
        return gzip.compress(raw_bytes)
    return raw_bytes


def _is_block_empty(data, nodata: float | int | None) -> bool:
    """Check if block contains only nodata values."""
    import numpy as np

    if nodata is None:
        return False
    return np.all(data == nodata)


def fetch_imageserver_tile(
    service_url: str,
    bounds: tuple[float, float, float, float],
    size: int,
    token: str | None = None,
    bands: list[int] | None = None,
):
    """
    Fetch a single tile from an ArcGIS ImageServer.

    Args:
        service_url: ImageServer REST URL
        bounds: Bounding box in Web Mercator (xmin, ymin, xmax, ymax)
        size: Output size in pixels (width and height)
        token: Optional authentication token
        bands: Optional list of band indices to fetch (1-based)

    Returns:
        Numpy array of shape (bands, height, width), or None if empty
    """
    import io

    import rasterio

    export_url = f"{service_url}/exportImage"

    params = {
        "bbox": f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{size},{size}",
        "format": "tiff",
        "f": "image",
        "interpolation": "RSP_NearestNeighbor",
    }

    if bands:
        params["bandIds"] = ",".join(str(b) for b in bands)

    # Fetch tile as TIFF bytes
    image_bytes = _make_imageserver_request(export_url, params, token, return_bytes=True)

    if not image_bytes or len(image_bytes) < 100:
        return None

    # Read TIFF with rasterio
    try:
        with rasterio.open(io.BytesIO(image_bytes)) as src:
            data = src.read()
            return data
    except Exception:
        return None


def imageserver_to_raquet_table(
    service_url: str,
    *,
    token: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    compression: str | None = "gzip",
    target_resolution: int | None = None,
    skip_empty_blocks: bool = True,
    calculate_stats: bool = True,
) -> pa.Table:
    """
    Convert an ArcGIS ImageServer to raquet PyArrow Table.

    This fetches raster tiles from the ImageServer and converts them to
    the raquet format with QUADBIN spatial indexing.

    Args:
        service_url: ImageServer REST URL (e.g., .../ImageServer)
        token: Optional authentication token
        bbox: Optional bounding box filter (xmin, ymin, xmax, ymax) in WGS84
        block_size: Tile size in pixels (256 or 512, default 256)
        compression: Block compression ("gzip" or None)
        target_resolution: Target QUADBIN pixel resolution (auto if None)
        skip_empty_blocks: Skip blocks with all nodata values
        calculate_stats: Calculate per-band statistics

    Returns:
        PyArrow Table with raquet structure
    """
    import numpy as np
    from pyproj import Transformer

    if block_size % 16 != 0:
        raise ValueError("Block size must be divisible by 16")

    logging.info(f"Fetching metadata from {service_url}...")

    # Get service metadata
    metadata = get_imageserver_metadata(service_url, token)

    logging.info(
        f"ImageServer: {metadata.name}, {metadata.columns}x{metadata.rows}, "
        f"{metadata.band_count} bands, type: {metadata.pixel_type}"
    )

    # Get source CRS and bounds
    src_crs = _get_crs_from_spatial_reference(metadata.spatial_reference)
    src_extent = metadata.extent

    src_bounds = (
        src_extent.get("xmin", 0),
        src_extent.get("ymin", 0),
        src_extent.get("xmax", 0),
        src_extent.get("ymax", 0),
    )

    logging.info(f"Source CRS: {src_crs[:50]}..." if len(src_crs) > 50 else f"Source CRS: {src_crs}")
    logging.info(f"Source bounds: {src_bounds}")

    # Apply bbox filter if provided (bbox is in WGS84)
    if bbox:
        # Transform bbox from WGS84 to source CRS
        bbox_src = _transform_bounds_with_crs(bbox, "EPSG:4326", src_crs)
        # Intersect with service bounds
        src_bounds = (
            max(src_bounds[0], bbox_src[0]),
            max(src_bounds[1], bbox_src[1]),
            min(src_bounds[2], bbox_src[2]),
            min(src_bounds[3], bbox_src[3]),
        )

        if src_bounds[0] >= src_bounds[2] or src_bounds[1] >= src_bounds[3]:
            raise ValueError("Bounding box does not intersect with ImageServer extent")

    # Transform bounds to Web Mercator
    bounds_3857 = _transform_bounds_with_crs(src_bounds, src_crs, "EPSG:3857")

    logging.info(f"Web Mercator bounds: {bounds_3857}")

    # Calculate dimensions in Web Mercator
    width_m = bounds_3857[2] - bounds_3857[0]
    height_m = bounds_3857[3] - bounds_3857[1]

    # Estimate pixel resolution from service native resolution
    native_res_m = max(metadata.pixel_size_x, metadata.pixel_size_y)

    # Check if source CRS is geographic (degrees)
    src_crs_upper = src_crs.upper()
    is_projected = (
        src_crs_upper.startswith("PROJCS")
        or src_crs.startswith("EPSG:")
        and not src_crs.startswith("EPSG:4326")
    )
    is_geographic = not is_projected and (
        src_crs.startswith("EPSG:4326") or src_crs_upper.startswith("GEOGCS")
    )

    logging.info(f"CRS type: {'projected' if is_projected else 'geographic'}")

    if is_geographic:
        # Rough approximation: for geographic coords, convert degrees to meters at center
        center_lat = (src_bounds[1] + src_bounds[3]) / 2
        native_res_m = native_res_m * 111320 * math.cos(math.radians(center_lat))
        logging.info(f"Converted geographic resolution: {native_res_m:.2f}m")

    # Estimate dimensions
    est_width = int(width_m / native_res_m) if native_res_m > 0 else metadata.columns
    est_height = int(height_m / native_res_m) if native_res_m > 0 else metadata.rows

    # Calculate target resolution if not specified
    if target_resolution is None:
        target_resolution = _calculate_target_resolution(
            bounds_3857, est_width, est_height, block_size
        )

    block_zoom = int(math.log2(block_size))
    tile_zoom = target_resolution - block_zoom

    # Clamp zoom level
    tile_zoom = max(0, min(22, tile_zoom))
    target_resolution = tile_zoom + block_zoom

    logging.info(f"Target resolution: {target_resolution}, tile zoom: {tile_zoom}")

    # Convert bounds to WGS84 for mercantile
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(bounds_3857[0], bounds_3857[1])
    lon_max, lat_max = transformer.transform(bounds_3857[2], bounds_3857[3])

    # Clamp to valid ranges
    lon_min = max(-180, min(180, lon_min))
    lon_max = max(-180, min(180, lon_max))
    lat_min = max(-85.051129, min(85.051129, lat_min))
    lat_max = max(-85.051129, min(85.051129, lat_max))

    tiles = list(mercantile.tiles(lon_min, lat_min, lon_max, lat_max, zooms=tile_zoom))

    logging.info(f"Fetching {len(tiles)} tiles at zoom {tile_zoom}...")

    # Prepare band names
    band_names = [f"band_{i + 1}" for i in range(metadata.band_count)]
    dtype = np.dtype(metadata.pixel_type)
    nodata = metadata.nodata

    # Collect rows and stats
    rows = []
    band_stats = [[] for _ in range(metadata.band_count)]

    # Track bounds for metadata
    xmin_tile, ymin_tile, xmax_tile, ymax_tile = (
        float("inf"),
        float("inf"),
        float("-inf"),
        float("-inf"),
    )

    tiles_fetched = 0
    for tile in tiles:
        # Get tile bounds in Web Mercator
        tile_bounds = mercantile.xy_bounds(tile)
        tile_bounds_tuple = (
            tile_bounds.left,
            tile_bounds.bottom,
            tile_bounds.right,
            tile_bounds.top,
        )

        # Fetch tile from ImageServer
        tile_data = fetch_imageserver_tile(
            service_url,
            tile_bounds_tuple,
            block_size,
            token,
        )

        if tile_data is None:
            continue

        tiles_fetched += 1
        if tiles_fetched % 10 == 0:
            logging.info(f"Fetched {tiles_fetched}/{len(tiles)} tiles...")

        # Process each band
        block_data = {}
        all_empty = True

        for band_idx in range(min(tile_data.shape[0], metadata.band_count)):
            band_array = tile_data[band_idx]

            # Ensure correct size (pad or crop if needed)
            if band_array.shape != (block_size, block_size):
                resized = np.full((block_size, block_size), nodata if nodata else 0, dtype=dtype)
                h, w = min(band_array.shape[0], block_size), min(band_array.shape[1], block_size)
                resized[:h, :w] = band_array[:h, :w]
                band_array = resized

            # Check if empty
            if skip_empty_blocks and _is_block_empty(band_array, nodata):
                continue

            all_empty = False

            # Compress block data
            compressed = _compress_block_data(band_array.astype(dtype), compression)
            block_data[band_names[band_idx]] = compressed

            # Collect stats
            if calculate_stats:
                stats = _calculate_band_statistics(band_array, nodata)
                if stats.get("count", 0) > 0:
                    band_stats[band_idx].append(stats)

        if all_empty and skip_empty_blocks:
            continue

        # Get QUADBIN cell ID
        quadbin_id = quadbin.tile_to_cell((tile.x, tile.y, tile.z))

        # Add row
        row = {
            "block": quadbin_id,
            "metadata": None,
            **{name: block_data.get(name) for name in band_names},
        }
        rows.append(row)

        # Update tile bounds
        xmin_tile = min(xmin_tile, tile.x)
        ymin_tile = min(ymin_tile, tile.y)
        xmax_tile = max(xmax_tile, tile.x)
        ymax_tile = max(ymax_tile, tile.y)

    logging.info(f"Processed {len(rows)} blocks from {tiles_fetched} tiles")

    if not rows:
        logging.warning("No valid tiles were fetched from the ImageServer")

    # Sort rows by block ID for efficient row group pruning when querying.
    # This allows Parquet readers to skip entire row groups based on block
    # statistics, significantly reducing data transfer for remote file access.
    if rows:
        logging.info("Sorting %d rows by block ID for optimized row group pruning...", len(rows))
        rows.sort(key=lambda row: row["block"])

    # Aggregate band statistics
    aggregated_stats = []
    for band_idx in range(metadata.band_count):
        if band_stats[band_idx]:
            total_count = sum(s["count"] for s in band_stats[band_idx])
            if total_count > 0:
                aggregated_stats.append(
                    {
                        "min": min(s["min"] for s in band_stats[band_idx]),
                        "max": max(s["max"] for s in band_stats[band_idx]),
                        "mean": sum(s["mean"] * s["count"] for s in band_stats[band_idx])
                        / total_count,
                        "stddev": math.sqrt(
                            sum(s["stddev"] ** 2 * s["count"] for s in band_stats[band_idx])
                            / total_count
                        ),
                        "count": total_count,
                        "approximated_stats": True,
                    }
                )
            else:
                aggregated_stats.append(None)
        else:
            aggregated_stats.append(None)

    # Calculate final bounds
    if xmin_tile == float("inf"):
        final_bounds = [lon_min, lat_min, lon_max, lat_max]
    else:
        ul_tile = mercantile.Tile(x=int(xmin_tile), y=int(ymin_tile), z=tile_zoom)
        lr_tile = mercantile.Tile(x=int(xmax_tile), y=int(ymax_tile), z=tile_zoom)
        ul_bounds = mercantile.bounds(ul_tile)
        lr_bounds = mercantile.bounds(lr_tile)
        final_bounds = [ul_bounds.west, lr_bounds.south, lr_bounds.east, ul_bounds.north]

    # Create metadata
    metadata_dict = {
        "version": RAQUET_VERSION,
        "compression": compression,
        "block_resolution": tile_zoom,
        "minresolution": tile_zoom,
        "maxresolution": tile_zoom,
        "nodata": nodata,
        "bounds": final_bounds,
        "center": [
            (final_bounds[0] + final_bounds[2]) / 2,
            (final_bounds[1] + final_bounds[3]) / 2,
            tile_zoom,
        ],
        "width": est_width,
        "height": est_height,
        "block_width": block_size,
        "block_height": block_size,
        "num_blocks": len(rows),
        "num_pixels": len(rows) * block_size * block_size,
        "pixel_resolution": target_resolution,
        "source": {
            "type": "ArcGIS ImageServer",
            "url": service_url,
            "name": metadata.name,
        },
        "bands": [
            {
                "type": metadata.pixel_type,
                "name": band_names[i],
                "colorinterp": None,
                "nodata": str(nodata) if nodata is not None else None,
                "stats": aggregated_stats[i] if i < len(aggregated_stats) else None,
                "colortable": None,
            }
            for i in range(metadata.band_count)
        ],
    }

    # Add metadata row
    metadata_row = {
        "block": METADATA_BLOCK_ID,
        "metadata": json.dumps(metadata_dict),
        **dict.fromkeys(band_names),
    }
    rows.insert(0, metadata_row)

    # Create schema
    schema = pa.schema(
        [
            ("block", pa.uint64()),
            ("metadata", pa.string()),
            *[(name, pa.binary()) for name in band_names],
        ]
    )

    # Build table
    table_dict = {key: [row.get(key) for row in rows] for key in schema.names}
    table = pa.Table.from_pydict(table_dict, schema=schema)

    return table


def imageserver_to_raquet(
    service_url: str,
    output_parquet: str,
    *,
    token: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    compression: str | None = "gzip",
    parquet_compression: str = "ZSTD",
    target_resolution: int | None = None,
    skip_empty_blocks: bool = True,
    calculate_stats: bool = True,
    row_group_size: int = 200,
) -> dict[str, Any]:
    """
    Convert an ArcGIS ImageServer to raquet parquet format.

    Args:
        service_url: ImageServer REST URL (e.g., .../ImageServer)
        output_parquet: Path for output parquet file
        token: Optional authentication token
        bbox: Optional bounding box filter (xmin, ymin, xmax, ymax) in WGS84
        block_size: Tile size in pixels (256 or 512, default 256)
        compression: Block compression ("gzip" or None)
        parquet_compression: Parquet file compression (ZSTD, GZIP, etc.)
        target_resolution: Target QUADBIN pixel resolution (auto if None)
        skip_empty_blocks: Skip nodata-only blocks
        calculate_stats: Calculate band statistics
        row_group_size: Rows per Parquet row group (default 200 for efficient pruning)

    Returns:
        dict with conversion stats (num_blocks, num_bands, etc.)
    """
    logging.info("Converting ImageServer to raquet format...")

    table = imageserver_to_raquet_table(
        service_url,
        token=token,
        bbox=bbox,
        block_size=block_size,
        compression=compression,
        target_resolution=target_resolution,
        skip_empty_blocks=skip_empty_blocks,
        calculate_stats=calculate_stats,
    )

    # Write to parquet with specified row group size for efficient remote pruning
    # Enable page index for finer-grained filtering and sorting metadata
    from pyarrow.parquet import SortingColumn
    pq.write_table(
        table,
        output_parquet,
        compression=parquet_compression.lower() if parquet_compression else None,
        row_group_size=row_group_size,
        write_page_index=True,  # Enable page-level column indexes
        sorting_columns=[SortingColumn(0)],  # Column 0 (block) is sorted
    )

    logging.info(f"Written to {output_parquet}")

    # Read back metadata to get stats
    from . import raquet2geotiff

    metadata_table = pq.read_table(
        output_parquet, columns=["block", "metadata"], filters=[("block", "=", 0)]
    )
    if len(metadata_table) > 0:
        metadata_json = metadata_table.column("metadata")[0].as_py()
        meta = json.loads(metadata_json)
        return {
            "num_blocks": meta.get("num_blocks", len(table) - 1),
            "num_bands": len(meta.get("bands", [])),
            "num_pixels": meta.get("num_pixels", 0),
            "block_size": block_size,
            "compression": compression,
        }

    return {
        "num_blocks": len(table) - 1,
        "num_bands": 0,
        "num_pixels": 0,
        "block_size": block_size,
        "compression": compression,
    }
