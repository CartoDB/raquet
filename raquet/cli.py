#!/usr/bin/env python3
"""Raquet CLI - Tools for working with Raquet (Raster + Parquet) files

Raquet stores raster data in Parquet format with QUADBIN spatial indexing.
"""

import gzip
import json
import logging
import math
import sys
from pathlib import Path

import click
import pyarrow.parquet as pq

from . import geotiff2raquet, raquet2geotiff, imageserver


# Configure logging
def setup_logging(verbose: bool):
    """Configure logging based on verbosity."""
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )


@click.group()
@click.version_option()
def cli():
    """Raquet CLI - Tools for working with Raquet (Raster + Parquet) files.

    Raquet stores raster data in Parquet format with QUADBIN spatial indexing.

    \b
    Examples:
        raquet inspect file.parquet
        raquet convert geotiff input.tif output.parquet
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
            info_table.add_row("Block Size", f"{metadata.get('block_width', 'N/A')} x {metadata.get('block_height', 'N/A')} px")

            console.print(info_table)

            # Resolution info
            res_table = Table(title="Resolution", show_header=False)
            res_table.add_column("Property", style="cyan")
            res_table.add_column("Value")

            res_table.add_row("Min Resolution", str(metadata.get("minresolution", "N/A")))
            res_table.add_row("Max Resolution", str(metadata.get("maxresolution", "N/A")))
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
            click.echo(f"  Block Size: {metadata.get('block_width', 'N/A')} x {metadata.get('block_height', 'N/A')} px")

            click.echo("\nResolution:")
            click.echo(f"  Min Resolution: {metadata.get('minresolution', 'N/A')}")
            click.echo(f"  Max Resolution: {metadata.get('maxresolution', 'N/A')}")
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
    """Convert various formats to Raquet.

    \b
    Examples:
        raquet convert geotiff input.tif output.parquet
        raquet convert imageserver https://server/arcgis/rest/services/layer/ImageServer output.parquet
    """
    pass


@convert_group.command("geotiff")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
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
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def convert_geotiff(
    input_file: Path,
    output_file: Path,
    zoom_strategy: str,
    resampling: str,
    block_size: int,
    target_size: int | None,
    verbose: bool,
):
    """Convert a GeoTIFF file to Raquet format.

    INPUT_FILE is the path to the source GeoTIFF file.
    OUTPUT_FILE is the path for the output Raquet (.parquet) file.

    \b
    Examples:
        raquet convert geotiff landcover.tif landcover.parquet
        raquet convert geotiff dem.tif dem.parquet --resampling bilinear
        raquet convert geotiff large.tif output.parquet --block-size 512 -v
    """
    setup_logging(verbose)

    # Calculate block_zoom from block_size
    block_zoom = int(math.log(block_size) / math.log(2))

    try:
        click.echo(f"Converting {input_file} to Raquet format...")

        geotiff2raquet.main(
            str(input_file),
            str(output_file),
            geotiff2raquet.ZoomStrategy(zoom_strategy),
            geotiff2raquet.ResamplingAlgorithm(resampling),
            block_zoom,
            target_size,
        )

        click.echo(f"Successfully created {output_file}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


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
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def export_geotiff(input_file: Path, output_file: Path, verbose: bool):
    """Export a Raquet file to GeoTIFF format.

    INPUT_FILE is the path to the source Raquet (.parquet) file.
    OUTPUT_FILE is the path for the output GeoTIFF file.

    \b
    Examples:
        raquet export geotiff landcover.parquet landcover.tif
        raquet export geotiff raster.parquet output.tif -v
    """
    setup_logging(verbose)

    try:
        click.echo(f"Exporting {input_file} to GeoTIFF format...")

        raquet2geotiff.main(str(input_file), str(output_file))

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


if __name__ == "__main__":
    main()
