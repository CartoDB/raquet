#!/usr/bin/env python3
"""Earth Engine to Raquet conversion module.

This module provides functionality to export Google Earth Engine images
to Raquet format via Google Cloud Storage intermediate storage.

Workflow:
    ee.Image -> Export.image.toCloudStorage -> GCS GeoTIFF -> /vsigs/ -> Raquet
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Lazy imports for optional dependencies
def _get_ee():
    """Lazily import Earth Engine API."""
    try:
        import ee

        return ee
    except ImportError as e:
        raise ImportError(
            "earthengine-api is required for Earth Engine conversion. "
            "Install with: pip install raquet[earthengine]"
        ) from e


def _get_gcs():
    """Lazily import Google Cloud Storage client."""
    try:
        from google.cloud import storage

        return storage
    except ImportError as e:
        raise ImportError(
            "google-cloud-storage is required for Earth Engine conversion. "
            "Install with: pip install raquet[earthengine]"
        ) from e


class EarthEngineError(Exception):
    """Base exception for Earth Engine operations."""

    pass


class EarthEngineAuthError(EarthEngineError):
    """Authentication or initialization failed."""

    pass


class EarthEngineTaskError(EarthEngineError):
    """Export task failed."""

    def __init__(self, message: str, task_status: dict | None = None):
        super().__init__(message)
        self.task_status = task_status


@dataclass
class EarthEngineConfig:
    """Configuration for Earth Engine export and conversion."""

    image_spec: str  # Asset ID or expression
    gcs_bucket: str  # GCS bucket name
    output_path: str  # Final raquet output path

    # Optional GCS path (auto-generated if None)
    gcs_path: str | None = None

    # Image parameters
    bands: list[str] | None = None  # Band selection
    region: dict | None = None  # GeoJSON geometry
    scale: float | None = 10  # Meters per pixel (None if using crs_transform)
    crs: str = "EPSG:4326"  # Output CRS
    crs_transform: list[float] | None = None  # [xScale, xShear, xTrans, yShear, yScale, yTrans]
    tile_zoom: int | None = None  # If set, compute crs_transform for Web Mercator tile alignment

    # Export options
    max_pixels: int = int(1e13)  # Max export pixels
    file_format: str = "GeoTIFF"  # Export format
    cloud_optimized: bool = True  # Export as COG

    # Conversion options
    block_size: int = 256
    resampling: str = "near"
    overviews: str = "auto"
    streaming: bool = False
    row_group_size: int = 200

    # Cleanup options
    delete_temp: bool = True  # Delete GCS file after conversion

    # GCP project
    project: str | None = None

    def get_gcs_path(self) -> str:
        """Get GCS path, generating one if not specified."""
        if self.gcs_path:
            return self.gcs_path
        # Auto-generate unique path
        unique_id = uuid.uuid4().hex[:8]
        return f"raquet-temp/export-{unique_id}.tif"


@dataclass
class ExportResult:
    """Result information from an export task."""

    task_id: str
    gcs_uri: str  # gs://bucket/path
    vsigs_path: str  # /vsigs/bucket/path
    elapsed_seconds: float
    task_status: dict = field(default_factory=dict)


def _initialize_ee(project: str | None = None) -> None:
    """Initialize Earth Engine with authentication.

    Handles multiple auth scenarios:
    1. Already authenticated (ee.Initialize() succeeds)
    2. Application Default Credentials (gcloud auth application-default login)
    3. Service account (GOOGLE_APPLICATION_CREDENTIALS env var)

    Args:
        project: Optional GCP project ID for EE

    Raises:
        EarthEngineAuthError: If authentication fails
    """
    ee = _get_ee()

    # First try standard initialization (works if already authenticated)
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        logger.info("Earth Engine initialized successfully")
        return
    except ee.EEException:
        pass  # Try ADC below

    # Try Application Default Credentials
    try:
        import google.auth

        credentials, adc_project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/earthengine"]
        )
        use_project = project or adc_project
        if not use_project:
            raise EarthEngineAuthError(
                "No project specified. Please provide --project or set a default project."
            )
        ee.Initialize(credentials=credentials, project=use_project)
        logger.info(f"Earth Engine initialized with ADC (project: {use_project})")
        return
    except ImportError:
        raise EarthEngineAuthError(
            "google-auth is required for Application Default Credentials. "
            "Install with: pip install google-auth"
        )
    except Exception as e:
        error_msg = str(e)
        if "reauthentication" in error_msg.lower() or "refresh" in error_msg.lower():
            raise EarthEngineAuthError(
                "Credentials expired. Please run:\n"
                "  gcloud auth application-default login --scopes=https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform"
            ) from e
        raise EarthEngineAuthError(
            f"Earth Engine authentication failed: {e}\n"
            "Please run 'earthengine authenticate' or "
            "'gcloud auth application-default login --scopes=...'"
        ) from e


def _compute_web_mercator_transform(zoom: int) -> tuple[list[float], dict]:
    """Compute crsTransform and region for Web Mercator at given zoom level.

    This aligns pixels to the standard Web Mercator tile grid, ensuring
    pixel-perfect alignment with tile boundaries.

    Args:
        zoom: Tile zoom level (e.g., 9 for ~305m resolution)

    Returns:
        Tuple of (crs_transform list, region as ee.Geometry-compatible dict)
    """
    # Web Mercator constants
    ORIGIN = 20037508.342789244  # Half the Earth's circumference in meters
    BASE_RES = 156543.03392804097  # Resolution at zoom 0

    resolution = BASE_RES / (2**zoom)

    # crsTransform: [xScale, xShear, xTranslate, yShear, yScale, yTranslate]
    crs_transform = [resolution, 0, -ORIGIN, 0, -resolution, ORIGIN]

    # Global extent in Web Mercator
    region = {
        "type": "Polygon",
        "coordinates": [
            [
                [-ORIGIN, -ORIGIN],
                [ORIGIN, -ORIGIN],
                [ORIGIN, ORIGIN],
                [-ORIGIN, ORIGIN],
                [-ORIGIN, -ORIGIN],
            ]
        ],
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
    }

    return crs_transform, region


def _gcs_to_vsigs(gcs_uri: str) -> str:
    """Convert gs:// URI to /vsigs/ path for GDAL.

    Args:
        gcs_uri: GCS URI like gs://bucket/path/file.tif

    Returns:
        GDAL virtual path like /vsigs/bucket/path/file.tif
    """
    if gcs_uri.startswith("gs://"):
        return "/vsigs/" + gcs_uri[5:]
    return gcs_uri


def _vsigs_to_gcs(vsigs_path: str) -> str:
    """Convert /vsigs/ path to gs:// URI.

    Args:
        vsigs_path: GDAL path like /vsigs/bucket/path/file.tif

    Returns:
        GCS URI like gs://bucket/path/file.tif
    """
    if vsigs_path.startswith("/vsigs/"):
        return "gs://" + vsigs_path[7:]
    return vsigs_path


def create_ee_image(
    image_spec: str,
    bands: list[str] | None = None,
) -> "ee.Image":
    """Create an ee.Image from specification.

    Args:
        image_spec: Either:
            - Asset ID: "COPERNICUS/S2_SR/20230101T100031"
            - Expression prefixed with 'expr:': "expr:ee.Image('...').normalizedDifference(['B4','B3'])"
        bands: Optional list of bands to select

    Returns:
        ee.Image ready for export

    Raises:
        ValueError: If image_spec is invalid
        EarthEngineError: If image creation fails
    """
    ee = _get_ee()

    try:
        if image_spec.startswith("expr:"):
            # Python expression - evaluate it
            expression = image_spec[5:]
            # Create a safe namespace with ee module
            namespace = {"ee": ee}
            image = eval(expression, namespace)
            if not isinstance(image, ee.Image):
                raise ValueError(
                    f"Expression must return an ee.Image, got {type(image).__name__}"
                )
        else:
            # Treat as asset ID
            image = ee.Image(image_spec)

        # Select bands if specified
        if bands:
            image = image.select(bands)

        return image

    except ee.EEException as e:
        raise EarthEngineError(f"Failed to create Earth Engine image: {e}") from e
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {e}") from e


def submit_export_task(
    image: "ee.Image",
    config: EarthEngineConfig,
) -> tuple["ee.batch.Task", str]:
    """Submit an export task to Google Cloud Storage.

    Args:
        image: The ee.Image to export
        config: Export configuration

    Returns:
        Tuple of (started ee.batch.Task, gcs_path used for export)
    """
    ee = _get_ee()

    gcs_path = config.get_gcs_path()

    # Handle tile_zoom: compute crs_transform and region for Web Mercator alignment
    crs = config.crs
    crs_transform = config.crs_transform
    region = config.region
    scale = config.scale

    if config.tile_zoom is not None:
        crs = "EPSG:3857"
        crs_transform, computed_region = _compute_web_mercator_transform(config.tile_zoom)
        if region is None:
            region = computed_region
        scale = None  # Use crs_transform instead of scale
        logger.info(f"Using tile zoom {config.tile_zoom} (res={crs_transform[0]:.2f}m)")

    # Build export parameters
    export_params = {
        "image": image,
        "description": f"raquet-export-{uuid.uuid4().hex[:8]}",
        "bucket": config.gcs_bucket,
        "fileNamePrefix": gcs_path.replace(".tif", ""),  # EE adds extension
        "crs": crs,
        "maxPixels": config.max_pixels,
        "fileFormat": config.file_format,
    }

    # Use crsTransform or scale (not both)
    if crs_transform:
        export_params["crsTransform"] = crs_transform
    elif scale:
        export_params["scale"] = scale

    # Add region if specified
    if region:
        if isinstance(region, dict):
            export_params["region"] = ee.Geometry(region)
        else:
            export_params["region"] = region

    # Add format options for COG
    if config.cloud_optimized and config.file_format == "GeoTIFF":
        export_params["formatOptions"] = {"cloudOptimized": True}

    logger.info(f"Submitting export task to gs://{config.gcs_bucket}/{gcs_path}")

    task = ee.batch.Export.image.toCloudStorage(**export_params)
    task.start()

    logger.info(f"Export task started: {task.id}")
    return task, gcs_path


def poll_task_status(
    task: "ee.batch.Task",
    poll_interval: float = 10.0,
    timeout: float | None = None,
    progress_callback: callable | None = None,
) -> dict:
    """Poll task until completion or failure.

    Args:
        task: The EE task to monitor
        poll_interval: Seconds between status checks
        timeout: Maximum wait time in seconds (None = infinite)
        progress_callback: Optional callback(status_str, elapsed_seconds)

    Returns:
        dict with task status information

    Raises:
        EarthEngineTaskError: If task fails
        TimeoutError: If timeout exceeded
    """
    ee = _get_ee()

    start_time = time.time()

    while True:
        status = task.status()
        state = status.get("state", "UNKNOWN")
        elapsed = time.time() - start_time

        if progress_callback:
            progress_callback(state, elapsed)

        if state == "COMPLETED":
            logger.info(f"Export completed in {elapsed:.1f}s")
            return status

        if state in ("FAILED", "CANCELLED"):
            error_msg = status.get("error_message", "Unknown error")
            raise EarthEngineTaskError(
                f"Export task {state.lower()}: {error_msg}",
                task_status=status,
            )

        if timeout and elapsed > timeout:
            raise TimeoutError(
                f"Export task timed out after {elapsed:.1f}s (state: {state})"
            )

        logger.debug(f"Task state: {state}, elapsed: {elapsed:.1f}s")
        time.sleep(poll_interval)


def delete_gcs_file(bucket_name: str, blob_path: str) -> bool:
    """Delete a file from Google Cloud Storage.

    Args:
        bucket_name: GCS bucket name
        blob_path: Path within the bucket

    Returns:
        True if deleted successfully, False otherwise
    """
    storage = _get_gcs()

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.delete()
        logger.info(f"Deleted temporary file: gs://{bucket_name}/{blob_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete temporary file: {e}")
        return False


def earthengine_to_raquet(
    image_spec: str,
    gcs_bucket: str,
    output_path: str,
    *,
    gcs_path: str | None = None,
    bands: list[str] | None = None,
    region: dict | None = None,
    scale: float | None = 10,
    crs: str = "EPSG:4326",
    tile_zoom: int | None = None,
    block_size: int = 256,
    resampling: str = "near",
    overviews: str = "auto",
    streaming: bool = False,
    row_group_size: int = 200,
    delete_temp: bool = True,
    project: str | None = None,
    poll_interval: float = 10.0,
    timeout: float | None = None,
    progress_callback: callable | None = None,
) -> dict:
    """Export Earth Engine image and convert to Raquet format.

    Complete workflow:
    1. Initialize EE authentication
    2. Create ee.Image from spec
    3. Submit export to GCS
    4. Poll until complete
    5. Convert via /vsigs/ path using raster2raquet
    6. Optionally delete temp GCS file

    Args:
        image_spec: EE image specification (asset ID or expr:... expression)
        gcs_bucket: GCS bucket for temporary export
        output_path: Path for output Raquet file
        gcs_path: Path within bucket (auto-generated if None)
        bands: Band names to export
        region: GeoJSON geometry for export region
        scale: Scale in meters per pixel (ignored if tile_zoom is set)
        crs: Output CRS (overridden to EPSG:3857 if tile_zoom is set)
        tile_zoom: Web Mercator tile zoom level for pixel-perfect tile alignment
        block_size: Raquet block size in pixels
        resampling: Resampling algorithm
        overviews: Overview mode (auto/none)
        streaming: Use streaming mode for conversion
        row_group_size: Parquet row group size
        delete_temp: Delete temp GCS file after conversion
        project: GCP project ID for Earth Engine
        poll_interval: Seconds between task status checks
        timeout: Maximum wait time for export (None = infinite)
        progress_callback: Optional callback(status, elapsed) for progress

    Returns:
        dict with conversion statistics

    Raises:
        EarthEngineAuthError: If authentication fails
        EarthEngineTaskError: If export fails
        EarthEngineError: For other EE errors
    """
    import math
    from . import raster2raquet

    # Build config
    config = EarthEngineConfig(
        image_spec=image_spec,
        gcs_bucket=gcs_bucket,
        output_path=output_path,
        gcs_path=gcs_path,
        bands=bands,
        region=region,
        scale=scale,
        crs=crs,
        tile_zoom=tile_zoom,
        block_size=block_size,
        resampling=resampling,
        overviews=overviews,
        streaming=streaming,
        row_group_size=row_group_size,
        delete_temp=delete_temp,
        project=project,
    )

    # Step 1: Initialize Earth Engine
    logger.info("Initializing Earth Engine...")
    _initialize_ee(project=config.project)

    # Step 2: Create image
    logger.info(f"Creating image from: {image_spec}")
    image = create_ee_image(image_spec, bands=bands)

    # Step 3: Submit export task
    task, export_gcs_path = submit_export_task(image, config)

    # Step 4: Poll until complete
    logger.info("Waiting for export to complete...")
    export_start = time.time()
    task_status = poll_task_status(
        task,
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
    )
    export_elapsed = time.time() - export_start

    # Build the GCS and VSIGS paths (use path from submit, not regenerated)
    actual_gcs_path = export_gcs_path
    # EE adds .tif extension if not present
    if not actual_gcs_path.endswith(".tif"):
        actual_gcs_path = actual_gcs_path + ".tif"

    gcs_uri = f"gs://{config.gcs_bucket}/{actual_gcs_path}"
    vsigs_path = _gcs_to_vsigs(gcs_uri)

    logger.info(f"Export complete: {gcs_uri}")
    logger.info(f"Converting to Raquet via {vsigs_path}...")

    # Step 5: Convert using raster2raquet
    convert_start = time.time()

    block_zoom = int(math.log(block_size) / math.log(2))
    overview_mode = raster2raquet.OverviewMode(overviews)

    raster2raquet.main(
        vsigs_path,
        output_path,
        raster2raquet.ZoomStrategy.AUTO,
        raster2raquet.ResamplingAlgorithm(resampling),
        block_zoom,
        None,  # target_size
        row_group_size,
        overview_mode,
        None,  # min_zoom
        streaming,
    )

    convert_elapsed = time.time() - convert_start
    logger.info(f"Conversion complete: {output_path}")

    # Step 6: Cleanup temp file if requested
    if delete_temp:
        logger.info("Cleaning up temporary GCS file...")
        delete_gcs_file(config.gcs_bucket, actual_gcs_path)

    return {
        "output_path": output_path,
        "gcs_uri": gcs_uri,
        "export_seconds": export_elapsed,
        "convert_seconds": convert_elapsed,
        "total_seconds": export_elapsed + convert_elapsed,
        "task_id": task.id,
        "task_status": task_status,
    }
