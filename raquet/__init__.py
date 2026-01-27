import logging
import math

from . import raster2raquet
from . import raquet2geotiff

# Backwards compatibility alias
geotiff2raquet = raster2raquet


def raster2raquet_main():
    """Entry point for raster to raquet conversion."""
    args = raster2raquet.parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Zoom offset from tiles to pixels, e.g. 8 = 256px tiles
    block_zoom = int(math.log(args.block_size) / math.log(2))

    raster2raquet.main(
        args.input_filename,
        args.raquet_destination,
        raster2raquet.ZoomStrategy(args.zoom_strategy),
        raster2raquet.ResamplingAlgorithm(args.resampling_algorithm),
        block_zoom,
        args.target_size,
    )


# Backwards compatibility alias
geotiff2raquet_main = raster2raquet_main


def raquet2geotiff_main():
    args = raquet2geotiff.parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    raquet2geotiff.main(args.raquet_filename, args.geotiff_filename)
