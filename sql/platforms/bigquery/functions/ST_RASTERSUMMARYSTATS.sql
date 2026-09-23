-- ST_RASTERSUMMARYSTATS
-- Computes summary statistics for a raster band tile
--
-- Two overloads:
--
-- 1. Original (decompresses band data at query time):
--    ST_RASTERSUMMARYSTATS(band, metadata, band_index) -> STRUCT
--    Parameters:
--      band: BYTES - The compressed band data (or 'pixels' column for interleaved layout)
--      metadata: STRING - JSON metadata from the Raquet file
--      band_index: INT64 - The band index (0-based)
--
-- 2. Pre-computed stats (no decompression, pure SQL — much faster):
--    ST_RASTERSUMMARYSTATS(count, sum, min, max, mean, stddev) -> STRUCT
--    Parameters:
--      band_count: INT64 - Pre-computed count of valid pixels
--      band_sum: FLOAT64 - Pre-computed sum of valid pixel values
--      band_min: FLOAT64 - Pre-computed minimum value
--      band_max: FLOAT64 - Pre-computed maximum value
--      band_mean: FLOAT64 - Pre-computed mean value
--      band_stddev: FLOAT64 - Pre-computed standard deviation
--
-- Both return: STRUCT<count INT64, sum FLOAT64, mean FLOAT64, min FLOAT64, max FLOAT64, stddev FLOAT64>
-- Both work with RAQUET_AGGREGATE_STATS for region aggregation.
--
-- Usage with pre-computed stats (RaQuet files with tile_statistics):
--   SELECT `cartobq.raquet.ST_RASTERSUMMARYSTATS`(
--       band_1_count, band_1_sum, band_1_min, band_1_max, band_1_mean, band_1_stddev
--   ) AS stats
--   FROM `project.dataset.raster`
--   WHERE block != 0;
--
-- RaQuet v0.4.0 Support:
--   - Interleaved band layout: metadata.band_layout = 'interleaved'
--   - JPEG compression: SUPPORTED (via jpeg_decoder.js)
--   - WebP compression: NOT YET SUPPORTED

CREATE OR REPLACE FUNCTION `cartobq.raquet.ST_RASTERSUMMARYSTATS`(
    band BYTES,
    metadata STRING,
    band_index INT64
)
RETURNS STRUCT<count INT64, sum FLOAT64, mean FLOAT64, min FLOAT64, max FLOAT64, stddev FLOAT64>
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
        // Extract band data and compute stats
        let count = 0, sum = 0, sumSq = 0, min = Infinity, max = -Infinity;
        const totalPixels = decoded.width * decoded.height;
        for (let i = 0; i < totalPixels; i++) {
            const value = decoded.data[i * channels + bandIdx];
            if (raquetLib.isNodata(value, nodata)) continue;
            count++;
            sum += value;
            sumSq += value * value;
            if (value < min) min = value;
            if (value > max) max = value;
        }
        if (count === 0) {
            return { count: 0, sum: 0, mean: null, min: null, max: null, stddev: null };
        }
        const mean = sum / count;
        const variance = (sumSq / count) - (mean * mean);
        const stddev = Math.sqrt(Math.max(0, variance));
        return { count, sum, mean, min: min === Infinity ? null : min, max: max === -Infinity ? null : max, stddev };
    }

    // WebP not yet supported
    if (meta.compression === 'webp') {
        throw new Error("WebP compression is not yet supported in BigQuery.");
    }

    // Decompress gzip if needed
    if (meta.compression === 'gzip') {
        data = raquetInflateLib.inflate(data);
    }

    // Compute statistics with interleaved layout support (v0.4.0)
    const stats = raquetLib.computeBandStats(data, bandInfo.type, pixelCount, nodata, {
        bandLayout: bandLayout,
        bandIndex: bandIdx,
        bandCount: bandCount
    });

    return stats;
""";

-- 6-parameter overload: uses pre-computed tile statistics (no decompression)
-- Returns the same STRUCT format as the 3-parameter version.
CREATE OR REPLACE FUNCTION `cartobq.raquet.ST_RASTERSUMMARYSTATS`(
    band_count INT64,
    band_sum FLOAT64,
    band_min FLOAT64,
    band_max FLOAT64,
    band_mean FLOAT64,
    band_stddev FLOAT64
)
RETURNS STRUCT<count INT64, sum FLOAT64, mean FLOAT64, min FLOAT64, max FLOAT64, stddev FLOAT64>
AS (
    STRUCT(band_count, band_sum, band_mean, band_min, band_max, band_stddev)
);
