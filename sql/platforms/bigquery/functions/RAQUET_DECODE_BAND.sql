-- RAQUET_DECODE_BAND
-- Decodes a gzip-compressed band from a Raquet file to an array of pixel values
--
-- Parameters:
--   band: BYTES - The compressed band data (or 'pixels' column for interleaved layout)
--   metadata: STRING - JSON metadata from the Raquet file (block=0 row)
--   band_index: INT64 - The band index (0-based), used to get band type from metadata
--
-- Returns: ARRAY<FLOAT64> - Array of pixel values (nodata values become NULL)
--
-- RaQuet v0.4.0 Support:
--   - Interleaved band layout: metadata.band_layout = 'interleaved'
--   - JPEG compression: SUPPORTED (via jpeg_decoder.js)
--   - WebP compression: NOT YET SUPPORTED

CREATE OR REPLACE FUNCTION `cartobq.raquet.RAQUET_DECODE_BAND`(
    band BYTES,
    metadata STRING,
    band_index INT64
)
RETURNS ARRAY<FLOAT64>
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
    const pixelCount = blockWidth * blockHeight;
    const nodata = raquetLib.parseNodata(bandInfo.nodata);

    // Get band layout info (v0.4.0)
    const { bandLayout, bandCount } = raquetLib.parseBandLayout(meta);

    // Decode base64 to bytes
    let data = raquetLib.base64ToUint8Array(band);

    // Handle JPEG compression (v0.4.0)
    if (meta.compression === 'jpeg') {
        const decoded = jpegDecoderLib.decode(data);
        const channels = decoded.channels;
        if (bandIdx >= channels) {
            return null;
        }
        const result = new Array(decoded.width * decoded.height);
        for (let i = 0; i < result.length; i++) {
            const value = decoded.data[i * channels + bandIdx];
            result[i] = raquetLib.isNodata(value, nodata) ? null : value;
        }
        return result;
    }

    // WebP not yet supported
    if (meta.compression === 'webp') {
        throw new Error("WebP compression is not yet supported in BigQuery.");
    }

    // Decompress gzip if needed
    if (meta.compression === 'gzip') {
        data = raquetInflateLib.inflate(data);
    }

    // Decode band values with interleaved layout support (v0.4.0)
    const result = raquetLib.decodeBand(data, bandInfo.type, pixelCount, nodata, {
        bandLayout: bandLayout,
        bandIndex: bandIdx,
        bandCount: bandCount
    });

    return result;
""";
