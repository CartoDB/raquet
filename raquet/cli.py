#!/usr/bin/env python3
"""Raquet CLI - Tools for working with Raquet (Raster + Parquet) files

Raquet stores raster data in Parquet format with QUADBIN spatial indexing.
"""

import json
import logging
import math
import sys
from pathlib import Path

import click
import pyarrow.parquet as pq

from . import raster2raquet, raquet2geotiff, imageserver, validate as validate_module

# Backwards compatibility alias
geotiff2raquet = raster2raquet


class GDALPath(click.ParamType):
    """Custom Click type that supports both local paths and GDAL virtual filesystem paths.

    Recognizes paths starting with /vsi (e.g., /vsigs/, /vsicurl/, /vsis3/) and
    passes them through without local filesystem validation, letting GDAL handle them.
    """

    name = "gdal_path"

    def __init__(self, exists: bool = False):
        self.exists = exists

    def convert(self, value, param, ctx):
        if value is None:
            return None

        # GDAL virtual filesystem paths - pass through without validation
        if value.startswith("/vsi"):
            return value  # Return as string, not Path

        # Local path - use standard Path validation
        path = Path(value)
        if self.exists and not path.exists():
            self.fail(f"Path '{value}' does not exist.", param, ctx)
        return path


# Configure logging
def setup_logging(verbose: bool):
    """Configure logging based on verbosity."""
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _validate_block_size_or_exit(block_size: int) -> None:
    if not _is_power_of_two(block_size):
        click.echo(
            "Error: Block size must be a power of two (e.g., 256, 512, 1024).",
            err=True,
        )
        sys.exit(1)


def _get_tiling_value(metadata: dict, tiling_key: str, legacy_key: str | None = None, default="N/A"):
    tiling = metadata.get("tiling")
    if isinstance(tiling, dict) and tiling_key in tiling:
        return tiling[tiling_key]
    if legacy_key is not None:
        return metadata.get(legacy_key, default)
    return metadata.get(tiling_key, default)


@click.group()
@click.version_option(package_name="raquet-io")
def cli():
    """Raquet CLI - Tools for working with Raquet (Raster + Parquet) files.

    Raquet stores raster data in Parquet format with QUADBIN spatial indexing.

    \b
    Examples:
        raquet inspect file.parquet
        raquet convert raster input.tif output.parquet
        raquet convert raster input.nc output.parquet
        raquet export geotiff input.parquet output.tif
    """
    pass


@cli.command("inspect")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def inspect_command(file: Path, verbose: bool):
    """Inspect a Raquet file and display its metadata.

    FILE is the path to a Raquet (.parquet) file.

    \b
    Examples:
        raquet inspect landcover.parquet
        raquet inspect /path/to/raster.parquet -v
    """
    setup_logging(verbose)

    try:
        # Try rich import for nice output
        from rich.console import Console
        from rich.table import Table

        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    try:
        # Read parquet file
        table = pq.read_table(file)

        # Get metadata from block=0
        import pyarrow.compute as pc

        block_zero = table.filter(pc.equal(table.column("block"), 0))
        if len(block_zero) == 0:
            click.echo("Error: No metadata block (block=0) found in file", err=True)
            sys.exit(1)

        metadata_json = block_zero.column("metadata")[0].as_py()
        metadata = json.loads(metadata_json)

        # Get file stats
        file_size = file.stat().st_size
        num_rows = len(table)
        num_blocks = num_rows - 1  # Exclude metadata row

        has_tiling = isinstance(metadata.get("tiling"), dict)
        block_width = _get_tiling_value(metadata, "block_width", "block_width")
        block_height = _get_tiling_value(metadata, "block_height", "block_height")
        min_res = _get_tiling_value(metadata, "min_zoom", "minresolution")
        max_res = _get_tiling_value(metadata, "max_zoom", "maxresolution")

        if use_rich:
            # Rich formatted output
            console.print(f"\n[bold blue]Raquet File:[/bold blue] {file.name}")
            console.print(f"[dim]Path: {file.absolute()}[/dim]\n")

            # General info table
            info_table = Table(title="General Information", show_header=False)
            info_table.add_column("Property", style="cyan")
            info_table.add_column("Value")

            info_table.add_row("File Size", f"{file_size / (1024*1024):.2f} MB")
            info_table.add_row("Total Rows", str(num_rows))
            info_table.add_row("Data Blocks", str(num_blocks))
            info_table.add_row("Width", f"{metadata.get('width', 'N/A')} px")
            info_table.add_row("Height", f"{metadata.get('height', 'N/A')} px")
            info_table.add_row("Block Size", f"{block_width} x {block_height} px")

            console.print(info_table)

            # Resolution info
            res_table = Table(title="Resolution", show_header=False)
            res_table.add_column("Property", style="cyan")
            res_table.add_column("Value")

            if has_tiling and "minresolution" not in metadata:
                res_table.add_row("Min Zoom", str(min_res))
                res_table.add_row("Max Zoom", str(max_res))
            else:
                res_table.add_row("Min Resolution", str(min_res))
                res_table.add_row("Max Resolution", str(max_res))
            res_table.add_row("Compression", metadata.get("compression", "none"))

            console.print(res_table)

            # Bounds
            bounds = metadata.get("bounds", [])
            if bounds:
                bounds_table = Table(title="Bounds (WGS84)", show_header=False)
                bounds_table.add_column("Property", style="cyan")
                bounds_table.add_column("Value")

                bounds_table.add_row("Min Longitude", f"{bounds[0]:.6f}")
                bounds_table.add_row("Min Latitude", f"{bounds[1]:.6f}")
                bounds_table.add_row("Max Longitude", f"{bounds[2]:.6f}")
                bounds_table.add_row("Max Latitude", f"{bounds[3]:.6f}")

                console.print(bounds_table)

            # Bands info
            bands = metadata.get("bands", [])
            if bands:
                bands_table = Table(title=f"Bands ({len(bands)} total)")
                bands_table.add_column("#", style="cyan")
                bands_table.add_column("Name")
                bands_table.add_column("Type")
                bands_table.add_column("Color Interp")
                bands_table.add_column("NoData")

                for i, band in enumerate(bands, 1):
                    bands_table.add_row(
                        str(i),
                        band.get("name", f"band_{i}"),
                        band.get("type", "N/A"),
                        band.get("colorinterp", "N/A"),
                        str(band.get("nodata", metadata.get("nodata", "N/A"))),
                    )

                console.print(bands_table)

            # Schema
            schema_table = Table(title="Parquet Schema")
            schema_table.add_column("Column", style="cyan")
            schema_table.add_column("Type")

            for field in table.schema:
                schema_table.add_row(field.name, str(field.type))

            console.print(schema_table)
            console.print()

        else:
            # Plain text output
            click.echo(f"\nRaquet File: {file.name}")
            click.echo(f"Path: {file.absolute()}\n")

            click.echo("General Information:")
            click.echo(f"  File Size: {file_size / (1024*1024):.2f} MB")
            click.echo(f"  Total Rows: {num_rows}")
            click.echo(f"  Data Blocks: {num_blocks}")
            click.echo(f"  Width: {metadata.get('width', 'N/A')} px")
            click.echo(f"  Height: {metadata.get('height', 'N/A')} px")
            click.echo(f"  Block Size: {block_width} x {block_height} px")

            click.echo("\nResolution:")
            if has_tiling and "minresolution" not in metadata:
                click.echo(f"  Min Zoom: {min_res}")
                click.echo(f"  Max Zoom: {max_res}")
            else:
                click.echo(f"  Min Resolution: {min_res}")
                click.echo(f"  Max Resolution: {max_res}")
            click.echo(f"  Compression: {metadata.get('compression', 'none')}")

            bounds = metadata.get("bounds", [])
            if bounds:
                click.echo("\nBounds (WGS84):")
                click.echo(f"  Min Longitude: {bounds[0]:.6f}")
                click.echo(f"  Min Latitude: {bounds[1]:.6f}")
                click.echo(f"  Max Longitude: {bounds[2]:.6f}")
                click.echo(f"  Max Latitude: {bounds[3]:.6f}")

            bands = metadata.get("bands", [])
            if bands:
                click.echo(f"\nBands ({len(bands)} total):")
                for i, band in enumerate(bands, 1):
                    nodata = band.get("nodata", metadata.get("nodata", "N/A"))
                    click.echo(f"  {i}. {band.get('name', f'band_{i}')} ({band.get('type', 'N/A')}) - {band.get('colorinterp', 'N/A')}, nodata={nodata}")

            click.echo("\nParquet Schema:")
            for field in table.schema:
                click.echo(f"  {field.name}: {field.type}")
            click.echo()

    except Exception as e:
        click.echo(f"Error reading file: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.group("convert")
def convert_group():
    """Convert various raster formats to Raquet.

    Supports any GDAL-readable raster format including GeoTIFF, COG, NetCDF, and more.

    \b
    Examples:
        raquet convert raster input.tif output.parquet
        raquet convert raster input.nc output.parquet
        raquet convert imageserver https://server/arcgis/rest/services/layer/ImageServer output.parquet
    """
    pass


def _convert_raster_impl(
    input_file: Path | str,
    output_file: Path,
    zoom_strategy: str,
    resampling: str,
    block_size: int,
    target_size: int | None,
    row_group_size: int,
    verbose: bool,
    overviews: str,
    min_zoom: int | None,
    streaming: bool,
):
    """Implementation for raster conversion (shared by raster and geotiff commands)."""
    setup_logging(verbose)

    _validate_block_size_or_exit(block_size)

    # Calculate block_zoom from block_size
    block_zoom = int(math.log(block_size) / math.log(2))

    # Convert overview mode
    overview_mode = raster2raquet.OverviewMode(overviews)

    try:
        click.echo(f"Converting {input_file} to Raquet format...")
        if overview_mode == raster2raquet.OverviewMode.NONE:
            click.echo("  Overview mode: none (native resolution only)")
        elif min_zoom is not None:
            click.echo(f"  Min zoom override: {min_zoom}")
        if streaming:
            click.echo("  Streaming mode: enabled (memory-safe two-pass conversion)")

        raster2raquet.main(
            str(input_file),
            str(output_file),
            raster2raquet.ZoomStrategy(zoom_strategy),
            raster2raquet.ResamplingAlgorithm(resampling),
            block_zoom,
            target_size,
            row_group_size,
            overview_mode,
            min_zoom,
            streaming,
        )

        click.echo(f"Successfully created {output_file}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@convert_group.command("raster")
@click.argument("input_file", type=GDALPath(exists=True))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--zoom-strategy",
    type=click.Choice(["auto", "lower", "upper"]),
    default="auto",
    help="Strategy for selecting zoom level (default: auto)",
)
@click.option(
    "--resampling",
    type=click.Choice(["near", "average", "bilinear", "cubic", "cubicspline", "lanczos", "mode", "max", "min", "med", "q1", "q3"]),
    default="near",
    help="Resampling algorithm (default: near)",
)
@click.option(
    "--block-size",
    type=int,
    default=256,
    help="Block size in pixels (default: 256)",
)
@click.option(
    "--target-size",
    type=int,
    default=None,
    help="Target size for auto zoom calculation",
)
@click.option(
    "--row-group-size",
    type=int,
    default=200,
    help="Rows per Parquet row group (default: 200, smaller = better remote pruning)",
)
@click.option(
    "--overviews",
    type=click.Choice(["auto", "none"]),
    default="auto",
    help="Overview generation: auto (full pyramid) or none (native resolution only)",
)
@click.option(
    "--min-zoom",
    type=int,
    default=None,
    help="Minimum zoom level for overviews (overrides auto calculation)",
)
@click.option(
    "--streaming",
    is_flag=True,
    help="Use two-pass streaming mode for memory-safe conversion of large files",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def convert_raster(
    input_file: Path,
    output_file: Path,
    zoom_strategy: str,
    resampling: str,
    block_size: int,
    target_size: int | None,
    row_group_size: int,
    overviews: str,
    min_zoom: int | None,
    streaming: bool,
    verbose: bool,
):
    """Convert a raster file to Raquet format.

    Supports any GDAL-readable raster format including GeoTIFF, COG, NetCDF, and more.
    For NetCDF files with time dimensions, time columns (time_cf, time_ts) are automatically added.

    INPUT_FILE is the path to the source raster file.
    OUTPUT_FILE is the path for the output Raquet (.parquet) file.

    \b
    Overview options (similar to COG):
        --overviews auto    Build full pyramid from min to max zoom (default)
        --overviews none    No overviews, only native resolution tiles (fastest)
        --min-zoom N        Limit overviews to zoom N and above

    \b
    Memory options:
        --streaming         Two-pass conversion for large files (lower memory usage)

    \b
    Examples:
        raquet convert raster landcover.tif landcover.parquet
        raquet convert raster temperature.nc temperature.parquet
        raquet convert raster dem.tif dem.parquet --resampling bilinear
        raquet convert raster large.tif output.parquet --overviews none -v
        raquet convert raster large.tif output.parquet --min-zoom 5 -v
        raquet convert raster huge.tif output.parquet --streaming -v
    """
    _convert_raster_impl(input_file, output_file, zoom_strategy, resampling, block_size, target_size, row_group_size, verbose, overviews, min_zoom, streaming)


@convert_group.command("geotiff")
@click.argument("input_file", type=GDALPath(exists=True))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--zoom-strategy",
    type=click.Choice(["auto", "lower", "upper"]),
    default="auto",
    help="Strategy for selecting zoom level (default: auto)",
)
@click.option(
    "--resampling",
    type=click.Choice(["near", "average", "bilinear", "cubic", "cubicspline", "lanczos", "mode", "max", "min", "med", "q1", "q3"]),
    default="near",
    help="Resampling algorithm (default: near)",
)
@click.option(
    "--block-size",
    type=int,
    default=256,
    help="Block size in pixels (default: 256)",
)
@click.option(
    "--target-size",
    type=int,
    default=None,
    help="Target size for auto zoom calculation",
)
@click.option(
    "--row-group-size",
    type=int,
    default=200,
    help="Rows per Parquet row group (default: 200, smaller = better remote pruning)",
)
@click.option(
    "--overviews",
    type=click.Choice(["auto", "none"]),
    default="auto",
    help="Overview generation: auto (full pyramid) or none (native resolution only)",
)
@click.option(
    "--min-zoom",
    type=int,
    default=None,
    help="Minimum zoom level for overviews (overrides auto calculation)",
)
@click.option(
    "--streaming",
    is_flag=True,
    help="Use two-pass streaming mode for memory-safe conversion of large files",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def convert_geotiff(
    input_file: Path,
    output_file: Path,
    zoom_strategy: str,
    resampling: str,
    block_size: int,
    target_size: int | None,
    row_group_size: int,
    overviews: str,
    min_zoom: int | None,
    streaming: bool,
    verbose: bool,
):
    """Convert a GeoTIFF file to Raquet format (alias for 'convert raster').

    INPUT_FILE is the path to the source GeoTIFF file.
    OUTPUT_FILE is the path for the output Raquet (.parquet) file.

    \b
    Examples:
        raquet convert geotiff landcover.tif landcover.parquet
        raquet convert geotiff dem.tif dem.parquet --resampling bilinear
        raquet convert geotiff large.tif output.parquet --overviews none
        raquet convert geotiff huge.tif output.parquet --streaming
    """
    _convert_raster_impl(input_file, output_file, zoom_strategy, resampling, block_size, target_size, row_group_size, verbose, overviews, min_zoom, streaming)


@convert_group.command("imageserver")
@click.argument("url")
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--token",
    type=str,
    default=None,
    help="ArcGIS authentication token",
)
@click.option(
    "--bbox",
    type=str,
    default=None,
    help="Bounding box filter in WGS84: xmin,ymin,xmax,ymax",
)
@click.option(
    "--block-size",
    type=int,
    default=256,
    help="Block size in pixels (default: 256)",
)
@click.option(
    "--resolution",
    type=int,
    default=None,
    help="Target QUADBIN pixel resolution (auto if not specified)",
)
@click.option(
    "--no-compression",
    is_flag=True,
    help="Disable gzip compression for block data",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def convert_imageserver(
    url: str,
    output_file: Path,
    token: str | None,
    bbox: str | None,
    block_size: int,
    resolution: int | None,
    no_compression: bool,
    verbose: bool,
):
    """Convert an ArcGIS ImageServer to Raquet format.

    URL is the ArcGIS ImageServer REST endpoint (e.g., .../ImageServer).
    OUTPUT_FILE is the path for the output Raquet (.parquet) file.

    \b
    Examples:
        raquet convert imageserver https://server/arcgis/rest/services/dem/ImageServer dem.parquet
        raquet convert imageserver https://server/arcgis/.../ImageServer output.parquet --resolution 12
        raquet convert imageserver https://server/ImageServer output.parquet --bbox "-122.5,37.5,-122.0,38.0"
    """
    setup_logging(verbose)
    _validate_block_size_or_exit(block_size)

    # Parse bbox if provided
    bbox_tuple = None
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) != 4:
                raise ValueError("Must have exactly 4 values")
            bbox_tuple = tuple(parts)
        except ValueError as e:
            click.echo(f"Error: Invalid bbox format. Expected xmin,ymin,xmax,ymax: {e}", err=True)
            sys.exit(1)

    compression = None if no_compression else "gzip"

    try:
        click.echo(f"Converting ImageServer {url} to Raquet format...")

        result = imageserver.imageserver_to_raquet(
            url,
            str(output_file),
            token=token,
            bbox=bbox_tuple,
            block_size=block_size,
            compression=compression,
            target_resolution=resolution,
        )

        click.echo(f"Successfully created {output_file}")
        click.echo(f"  Blocks: {result.get('num_blocks', 'N/A')}")
        click.echo(f"  Bands: {result.get('num_bands', 'N/A')}")
        click.echo(f"  Pixels: {result.get('num_pixels', 'N/A'):,}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@convert_group.command("earthengine")
@click.argument("image_spec")
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--gcs-bucket",
    required=True,
    help="GCS bucket for temporary export (e.g., 'my-bucket')",
)
@click.option(
    "--gcs-path",
    default=None,
    help="Path within bucket for temp file (default: auto-generated)",
)
@click.option(
    "--bands",
    default=None,
    help="Comma-separated band names to export (e.g., 'B4,B3,B2')",
)
@click.option(
    "--region",
    default=None,
    help="Export region as GeoJSON string or path to .geojson file",
)
@click.option(
    "--scale",
    type=float,
    default=10,
    help="Scale in meters per pixel (default: 10)",
)
@click.option(
    "--crs",
    default="EPSG:4326",
    help="Output CRS (default: EPSG:4326, ignored if --tile-zoom is set)",
)
@click.option(
    "--tile-zoom",
    type=int,
    default=None,
    help="Web Mercator tile zoom level for pixel-perfect tile alignment (overrides --scale and --crs)",
)
@click.option(
    "--block-size",
    type=int,
    default=256,
    help="Block size in pixels (default: 256)",
)
@click.option(
    "--resampling",
    type=click.Choice(["near", "average", "bilinear", "cubic"]),
    default="near",
    help="Resampling algorithm (default: near)",
)
@click.option(
    "--overviews",
    type=click.Choice(["auto", "none"]),
    default="auto",
    help="Overview generation: auto (full pyramid) or none (native only)",
)
@click.option(
    "--streaming",
    is_flag=True,
    help="Use two-pass streaming mode for memory-safe conversion",
)
@click.option(
    "--keep-temp",
    is_flag=True,
    help="Keep temporary GCS file after conversion",
)
@click.option(
    "--project",
    default=None,
    help="GCP project ID for Earth Engine",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Maximum wait time for export in seconds (default: no limit)",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def convert_earthengine(
    image_spec: str,
    output_file: Path,
    gcs_bucket: str,
    gcs_path: str | None,
    bands: str | None,
    region: str | None,
    scale: float,
    crs: str,
    tile_zoom: int | None,
    block_size: int,
    resampling: str,
    overviews: str,
    streaming: bool,
    keep_temp: bool,
    project: str | None,
    timeout: float | None,
    verbose: bool,
):
    """Convert a Google Earth Engine image to Raquet format.

    IMAGE_SPEC is the Earth Engine image specification:
    - Asset ID: COPERNICUS/S2_SR/20230101T100031_20230101T100027_T33UUP
    - Expression: expr:ee.Image('COPERNICUS/DEM/GLO30').select('DEM')

    OUTPUT_FILE is the path for the output Raquet (.parquet) file.

    Requires a Google Cloud Storage bucket for temporary export.
    The GCS file will be deleted after successful conversion unless --keep-temp is used.

    \b
    Examples:
        # Export Copernicus DEM
        raquet convert earthengine COPERNICUS/DEM/GLO30 dem.parquet \\
            --gcs-bucket my-bucket --scale 30

        # Export Sentinel-2 RGB bands with region
        raquet convert earthengine COPERNICUS/S2_SR/20230101T100031 s2.parquet \\
            --gcs-bucket my-bucket --bands B4,B3,B2 --scale 10 --region region.geojson

        # Export global MODIS LST at tile zoom 9 (~305m, aligned to Web Mercator tiles)
        raquet convert earthengine "expr:ee.ImageCollection('MODIS/061/MOD11A1').filterDate('2024-01-01','2024-02-01').select('LST_Day_1km').mean()" lst.parquet \\
            --gcs-bucket my-bucket --tile-zoom 9

        # Export with custom expression
        raquet convert earthengine "expr:ee.Image('COPERNICUS/DEM/GLO30')" output.parquet \\
            --gcs-bucket my-bucket
    """
    # Lazy import to handle optional dependency
    try:
        from . import earthengine as ee_module
    except ImportError as e:
        click.echo(
            "Error: Earth Engine support requires additional dependencies.\n"
            "Install with: pip install raquet[earthengine]",
            err=True,
        )
        sys.exit(1)

    setup_logging(verbose)
    _validate_block_size_or_exit(block_size)

    # Parse bands
    bands_list = None
    if bands:
        bands_list = [b.strip() for b in bands.split(",")]

    # Parse region (could be GeoJSON string or file path)
    region_dict = None
    if region:
        import json
        if region.strip().startswith("{"):
            # JSON string
            try:
                region_dict = json.loads(region)
            except json.JSONDecodeError as e:
                click.echo(f"Error: Invalid GeoJSON string: {e}", err=True)
                sys.exit(1)
        else:
            # File path
            region_path = Path(region)
            if not region_path.exists():
                click.echo(f"Error: Region file not found: {region}", err=True)
                sys.exit(1)
            try:
                region_dict = json.loads(region_path.read_text())
            except json.JSONDecodeError as e:
                click.echo(f"Error: Invalid GeoJSON in file {region}: {e}", err=True)
                sys.exit(1)

    # Progress callback for verbose mode
    def progress_callback(state: str, elapsed: float):
        if verbose:
            click.echo(f"  Export status: {state} ({elapsed:.0f}s elapsed)")

    try:
        click.echo(f"Converting Earth Engine image to Raquet format...")
        click.echo(f"  Image: {image_spec}")
        click.echo(f"  GCS bucket: {gcs_bucket}")
        if bands_list:
            click.echo(f"  Bands: {', '.join(bands_list)}")
        if tile_zoom is not None:
            # Calculate resolution for display
            res = 156543.03392804097 / (2**tile_zoom)
            click.echo(f"  Tile zoom: {tile_zoom} (~{res:.0f}m, EPSG:3857)")
        else:
            click.echo(f"  Scale: {scale}m")
            click.echo(f"  CRS: {crs}")

        result = ee_module.earthengine_to_raquet(
            image_spec=image_spec,
            gcs_bucket=gcs_bucket,
            output_path=str(output_file),
            gcs_path=gcs_path,
            bands=bands_list,
            region=region_dict,
            scale=scale,
            crs=crs,
            tile_zoom=tile_zoom,
            block_size=block_size,
            resampling=resampling,
            overviews=overviews,
            streaming=streaming,
            delete_temp=not keep_temp,
            project=project,
            timeout=timeout,
            progress_callback=progress_callback if verbose else None,
        )

        click.echo(f"\nSuccessfully created {output_file}")
        click.echo(f"  Export time: {result['export_seconds']:.1f}s")
        click.echo(f"  Conversion time: {result['convert_seconds']:.1f}s")
        click.echo(f"  Total time: {result['total_seconds']:.1f}s")
        if keep_temp:
            click.echo(f"  Temp file kept at: {result['gcs_uri']}")

    except ee_module.EarthEngineAuthError as e:
        click.echo(f"Authentication error: {e}", err=True)
        sys.exit(1)
    except ee_module.EarthEngineTaskError as e:
        click.echo(f"Export task failed: {e}", err=True)
        sys.exit(1)
    except ee_module.EarthEngineError as e:
        click.echo(f"Earth Engine error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.group("export")
def export_group():
    """Export Raquet files to other formats.

    \b
    Examples:
        raquet export geotiff input.parquet output.tif
    """
    pass


@export_group.command("geotiff")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--overviews",
    is_flag=True,
    help="Include RaQuet overview tiles as GeoTIFF overviews",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def export_geotiff(input_file: Path, output_file: Path, overviews: bool, verbose: bool):
    """Export a Raquet file to GeoTIFF format.

    INPUT_FILE is the path to the source Raquet (.parquet) file.
    OUTPUT_FILE is the path for the output GeoTIFF file.

    \b
    Examples:
        raquet export geotiff landcover.parquet landcover.tif
        raquet export geotiff raster.parquet output.tif --overviews
        raquet export geotiff raster.parquet output.tif -v
    """
    setup_logging(verbose)

    try:
        click.echo(f"Exporting {input_file} to GeoTIFF format...")
        if overviews:
            click.echo("  Including RaQuet overviews in output")

        raquet2geotiff.main(str(input_file), str(output_file), include_overviews=overviews)

        click.echo(f"Successfully created {output_file}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


@cli.command("split-zoom")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--row-group-size",
    type=int,
    default=200,
    help="Rows per Parquet row group (default: 200)",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def split_zoom_command(
    input_file: Path,
    output_dir: Path,
    row_group_size: int,
    verbose: bool,
):
    """Split a Raquet file by zoom level for optimized remote access.

    Creates separate files for each zoom level, enabling clients to query
    only the zoom level they need without downloading data for other zooms.

    INPUT_FILE is the path to a Raquet (.parquet) file.
    OUTPUT_DIR is the directory for output files (zoom_N.parquet).

    \b
    Examples:
        raquet split-zoom raster.parquet ./split_output/
        raquet split-zoom large.parquet ./by_zoom/ --row-group-size 100
    """
    import quadbin
    import pyarrow as pa
    from pyarrow.parquet import SortingColumn

    setup_logging(verbose)

    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read input file
        click.echo(f"Reading {input_file}...")
        table = pq.read_table(input_file)

        # Get metadata row
        import pyarrow.compute as pc
        metadata_mask = pc.equal(table.column("block"), 0)
        metadata_table = table.filter(metadata_mask)
        data_table = table.filter(pc.invert(metadata_mask))

        if len(metadata_table) == 0:
            click.echo("Error: No metadata block found", err=True)
            sys.exit(1)

        metadata_json = json.loads(metadata_table.column("metadata")[0].as_py())

        # Group blocks by zoom level
        blocks = data_table.column("block").to_pylist()
        zoom_indices = {}

        for i, block_id in enumerate(blocks):
            tile = quadbin.cell_to_tile(block_id)
            z = tile[2]
            if z not in zoom_indices:
                zoom_indices[z] = []
            zoom_indices[z].append(i)

        click.echo(f"Found {len(zoom_indices)} zoom levels: {sorted(zoom_indices.keys())}")

        # Write separate file for each zoom level
        files_written = []
        for zoom in sorted(zoom_indices.keys()):
            indices = zoom_indices[zoom]
            zoom_table = data_table.take(indices)

            # Sort by block ID
            sort_indices = pc.sort_indices(zoom_table.column("block"))
            zoom_table = zoom_table.take(sort_indices)

            # Update metadata for this zoom
            zoom_metadata = metadata_json.copy()
            zoom_metadata["minresolution"] = zoom
            zoom_metadata["maxresolution"] = zoom
            zoom_metadata["num_blocks"] = len(indices)

            # Create metadata row
            metadata_row = pa.table({
                "block": [0],
                "metadata": [json.dumps(zoom_metadata)],
                **{col: [None] for col in zoom_table.schema.names if col not in ["block", "metadata"]}
            }, schema=zoom_table.schema)

            # Combine metadata + data
            final_table = pa.concat_tables([metadata_row, zoom_table])

            # Write file with optimizations
            output_path = output_dir / f"zoom_{zoom}.parquet"
            pq.write_table(
                final_table,
                output_path,
                compression="zstd",
                row_group_size=row_group_size,
                write_page_index=True,
                write_statistics=True,
                sorting_columns=[SortingColumn(0)],
            )

            size_mb = output_path.stat().st_size / (1024 * 1024)
            click.echo(f"  zoom_{zoom}.parquet: {len(indices)} tiles, {size_mb:.1f} MB")
            files_written.append(output_path)

        # Summary
        total_size = sum(f.stat().st_size for f in files_written) / (1024 * 1024)
        orig_size = input_file.stat().st_size / (1024 * 1024)
        click.echo("\nSplit complete:")
        click.echo(f"  Original: {orig_size:.1f} MB")
        click.echo(f"  Split total: {total_size:.1f} MB")
        click.echo(f"  Files: {len(files_written)}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command("validate")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("-v", "--verbose", is_flag=True, help="Show detailed validation output")
@click.option("--json", "json_output", is_flag=True, help="Output results as JSON")
def validate_command(file: Path, verbose: bool, json_output: bool):
    """Validate a Raquet file for correctness.

    Performs comprehensive validation including:
    - Schema validation (required columns)
    - Metadata validation (version, structure)
    - Pyramid validation (all zoom levels have data)
    - Band statistics validation
    - Data integrity checks

    FILE is the path to a Raquet (.parquet) file.

    \b
    Examples:
        raquet validate raster.parquet
        raquet validate raster.parquet -v
        raquet validate raster.parquet --json
    """
    try:
        result = validate_module.validate_raquet(str(file))

        if json_output:
            import json as json_lib
            output = {
                "is_valid": result.is_valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "stats": result.stats,
            }
            click.echo(json_lib.dumps(output, indent=2, default=str))
        else:
            click.echo(str(result))

            if verbose and result.metadata:
                import json as json_lib
                click.echo("\nFull Metadata:")
                click.echo(json_lib.dumps(result.metadata, indent=2))

        sys.exit(0 if result.is_valid else 1)

    except Exception as e:
        click.echo(f"Error validating file: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
