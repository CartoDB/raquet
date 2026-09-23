-- RAQUET_PIXEL
-- Gets a single pixel value from a band at tile coordinates (x, y)
--
-- Parameters:
--   band: BYTES - The compressed band data (or 'pixels' column for interleaved layout)
--   metadata: STRING - JSON metadata from the Raquet file
--   band_index: INT64 - The band index (0-based)
--   x: INT64 - Pixel X coordinate within tile (0 to block_width-1)
--   y: INT64 - Pixel Y coordinate within tile (0 to block_height-1)
--
-- Returns: FLOAT64 - The pixel value, or NULL if nodata or out of bounds
--
-- RaQuet v0.4.0 Support:
--   - Interleaved band layout: metadata.band_layout = 'interleaved'
--   - JPEG compression: SUPPORTED (via jpeg_decoder.js)
--   - WebP compression: NOT YET SUPPORTED

CREATE OR REPLACE FUNCTION `cartobq.raquet.RAQUET_PIXEL`(
    band BYTES,
    metadata STRING,
    band_index INT64,
    x INT64,
    y INT64
)
RETURNS FLOAT64
LANGUAGE js
OPTIONS (
    library = [
        "gs://cartobq-raquet-libs/raquet_lib.js",
        "gs://cartobq-raquet-libs/raquet_inflate.js",
        "gs://cartobq-raquet-libs/jpeg_decoder.js"
    ]
)
AS r"""
    if (band === null || metadata === null) {
        return null;
    }

    const meta = JSON.parse(metadata);
    const bandIdx = Number(band_index) || 0;

    // Get band info
    const bandInfo = meta.bands[bandIdx];
    if (!bandInfo) {
        throw new Error(`Band index ${bandIdx} not found in metadata`);
    }

    const tiling = meta.tiling || {};
    const blockWidth = Number(tiling.block_width) || 256;
    const blockHeight = Number(tiling.block_height) || 256;

    // Get band layout info (v0.4.0)
    const { bandLayout, bandCount } = raquetLib.parseBandLayout(meta);

    // Convert BigQuery INT64 values to numbers (they may be passed as strings)
    const px = Number(x);
    const py = Number(y);

    // Validate coordinates
    if (px < 0 || px >= blockWidth || py < 0 || py >= blockHeight) {
        return null;
    }

    const nodata = raquetLib.parseNodata(bandInfo.nodata);

    // Decode base64 to bytes
    let data = raquetLib.base64ToUint8Array(band);

    // Handle JPEG compression (v0.4.0)
    if (meta.compression === 'jpeg') {
        const decoded = jpegDecoderLib.decode(data);
        // JPEG returns RGB data, extract band by channel index
        const pixelIdx = py * decoded.width + px;
        const channels = decoded.channels;
        if (bandIdx >= channels) {
            return null;
        }
        const value = decoded.data[pixelIdx * channels + bandIdx];
        if (raquetLib.isNodata(value, nodata)) {
            return null;
        }
        return value;
    }

    // WebP not yet supported
    if (meta.compression === 'webp') {
        throw new Error("WebP compression is not yet supported in BigQuery.");
    }

    // Decompress gzip if needed
    if (meta.compression === 'gzip') {
        data = raquetInflateLib.inflate(data);
    }

    // Calculate pixel offset (row-major order)
    const pixelOffset = py * blockWidth + px;

    // Get pixel value with interleaved layout support (v0.4.0)
    const value = raquetLib.decodePixelValue(data, bandInfo.type, pixelOffset, {
        bandLayout: bandLayout,
        bandIndex: bandIdx,
        bandCount: bandCount
    });

    // Return null for nodata
    if (raquetLib.isNodata(value, nodata)) {
        return null;
    }

    return value;
""";
