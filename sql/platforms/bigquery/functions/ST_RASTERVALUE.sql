-- ST_RASTERVALUE
-- Gets the raster value at a geographic point
--
-- This function finds the correct block and pixel for a given point and returns the value.
-- It requires joining with the raster table to get the matching block data.
--
-- Parameters:
--   block: INT64 - The QUADBIN block identifier
--   band: BYTES - The compressed band data (or 'pixels' column for interleaved layout)
--   lon: FLOAT64 - Longitude of the point
--   lat: FLOAT64 - Latitude of the point
--   metadata: STRING - JSON metadata from the Raquet file
--   band_index: INT64 - The band index (0-based)
--
-- Returns: FLOAT64 - The pixel value at the point
--
-- RaQuet v0.4.0 Support:
--   - Interleaved band layout: metadata.band_layout = 'interleaved'
--   - JPEG compression: SUPPORTED (via jpeg_decoder.js)
--   - WebP compression: NOT YET SUPPORTED

CREATE OR REPLACE FUNCTION `cartobq.raquet.ST_RASTERVALUE`(
    block INT64,
    band BYTES,
    lon FLOAT64,
    lat FLOAT64,
    metadata STRING,
    band_index INT64
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
    if (band === null || metadata === null || lon === null || lat === null || block === null) {
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
    const nodata = raquetLib.parseNodata(bandInfo.nodata);

    // Get band layout info (v0.4.0)
    const { bandLayout, bandCount } = raquetLib.parseBandLayout(meta);

    // Convert coordinates to numbers
    const longitude = Number(lon);
    const latitude = Number(lat);

    // Convert block QUADBIN to tile coordinates
    const blockNum = BigInt(block);
    const blockZ = Number((blockNum >> 52n) & 0x1Fn);

    // Extract x, y from QUADBIN using bit de-interleaving
    const content = (blockNum >> (52n - 2n * BigInt(blockZ))) & ((1n << (2n * BigInt(blockZ))) - 1n);
    let blockX = 0n, blockY = 0n;
    for (let i = 0; i < blockZ; i++) {
        blockX |= ((content >> BigInt(2 * i)) & 1n) << BigInt(i);
        blockY |= ((content >> BigInt(2 * i + 1)) & 1n) << BigInt(i);
    }

    // Calculate tile bounds in Web Mercator coordinates
    const n = Math.pow(2, blockZ);
    const tileMinLon = (Number(blockX) / n) * 360 - 180;
    const tileMaxLon = ((Number(blockX) + 1) / n) * 360 - 180;

    // Web Mercator Y calculation
    const tileMaxLat = Math.atan(Math.sinh(Math.PI * (1 - 2 * Number(blockY) / n))) * 180 / Math.PI;
    const tileMinLat = Math.atan(Math.sinh(Math.PI * (1 - 2 * (Number(blockY) + 1) / n))) * 180 / Math.PI;

    // Calculate pixel coordinates within tile
    const pixelX = Math.floor((longitude - tileMinLon) / (tileMaxLon - tileMinLon) * blockWidth);
    const pixelY = Math.floor((tileMaxLat - latitude) / (tileMaxLat - tileMinLat) * blockHeight);

    // Validate pixel coordinates
    if (pixelX < 0 || pixelX >= blockWidth || pixelY < 0 || pixelY >= blockHeight) {
        return null;
    }

    // Calculate pixel offset (row-major order)
    const pixelOffset = pixelY * blockWidth + pixelX;

    // Decode base64 to bytes
    let data = raquetLib.base64ToUint8Array(band);

    // Handle JPEG compression (v0.4.0)
    if (meta.compression === 'jpeg') {
        const decoded = jpegDecoderLib.decode(data);
        const pixelIdx = pixelY * decoded.width + pixelX;
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

    // Ensure clean buffer
    data = raquetLib.ensureOwnBuffer(data);

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
