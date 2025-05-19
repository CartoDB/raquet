import logging
import math

from . import geotiff2raquet
from . import raquet2geotiff


def geotiff2raquet_main():
    args = geotiff2raquet.parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Zoom offset from tiles to pixels, e.g. 8 = 256px tiles
    block_zoom=int(math.log(args.block_size) / math.log(2))

    geotiff2raquet.main(
        args.geotiff_filename,
        args.raquet_destination,
        geotiff2raquet.ZoomStrategy(args.zoom_strategy),
        geotiff2raquet.ResamplingAlgorithm(args.resampling_algorithm),
        block_zoom,
        args.target_size,
    )


def raquet2geotiff_main():
    args = raquet2geotiff.parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    raquet2geotiff.main(args.raquet_filename, args.geotiff_filename)
