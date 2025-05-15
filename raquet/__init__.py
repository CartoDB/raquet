import logging

from . import geotiff2raquet
from . import raquet2geotiff


def geotiff2raquet_main():
    args = geotiff2raquet.parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    geotiff2raquet.main(
        args.geotiff_filename,
        args.raquet_destination,
        geotiff2raquet.ZoomStrategy(args.zoom_strategy),
        geotiff2raquet.ResamplingAlgorithm(args.resampling_algorithm),
        args.max_size,
    )


def raquet2geotiff_main():
    args = raquet2geotiff.parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    raquet2geotiff.main(args.raquet_filename, args.geotiff_filename)
