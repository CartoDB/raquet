-- ST_NORMALIZEDDIFFERENCE
-- Computes normalized difference index: (band1 - band2) / (band1 + band2)
--
-- Common uses:
--   - NDVI (Vegetation): NIR - Red
--   - NDWI (Water): Green - NIR
--   - NDSI (Snow): Green - SWIR
--
-- Parameters:
--   band1: BYTES - First band (e.g., NIR for NDVI, or 'pixels' for interleaved)
--   band2: BYTES - Second band (e.g., Red for NDVI, or same 'pixels' for interleaved)
--   metadata: STRING - JSON metadata from the Raquet file
--   band1_index: INT64 - Index of first band in metadata
--   band2_index: INT64 - Index of second band in metadata
--
-- Returns: ARRAY<FLOAT64> - Normalized difference values (-1 to 1), NULL for nodata
--
-- RaQuet v0.4.0 Support:
--   - Interleaved band layout: metadata.band_layout = 'interleaved'
--   - JPEG compression: SUPPORTED (via jpeg_decoder.js)
--   - WebP compression: NOT YET SUPPORTED
--
-- Usage:
--   SELECT `cartobq.raquet.ST_NORMALIZEDDIFFERENCE`(
--       r.band_nir, r.band_red, m.metadata, 0, 1
--   ) AS ndvi
--   FROM raster_table r, metadata m

CREATE OR REPLACE FUNCTION `cartobq.raquet.ST_NORMALIZEDDIFFERENCE`(
    band1 BYTES,
    band2 BYTES,
    metadata STRING,
    band1_index INT64,
    band2_index INT64
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
    if (band1 === null || band2 === null || metadata === null) {
        return null;
    }

    const meta = JSON.parse(metadata);
    const idx1 = Number(band1_index) || 0;
    const idx2 = Number(band2_index) || 0;

    // Get band info
    const bandInfo1 = meta.bands[idx1];
    const bandInfo2 = meta.bands[idx2];
    if (!bandInfo1 || !bandInfo2) {
        throw new Error(`Band index not found in metadata`);
    }

    const tiling = meta.tiling || {};
    const blockWidth = Number(tiling.block_width) || 256;
    const blockHeight = Number(tiling.block_height) || 256;
    const pixelCount = blockWidth * blockHeight;

    // Get band layout info (v0.4.0)
    const { bandLayout, bandCount } = raquetLib.parseBandLayout(meta);

    const nodata1 = raquetLib.parseNodata(bandInfo1.nodata);
    const nodata2 = raquetLib.parseNodata(bandInfo2.nodata);

    // Decode base64 to bytes
    let data1 = raquetLib.base64ToUint8Array(band1);
    let data2 = raquetLib.base64ToUint8Array(band2);

    // Handle JPEG compression (v0.4.0)
    if (meta.compression === 'jpeg') {
        const decoded1 = jpegDecoderLib.decode(data1);
        const decoded2 = jpegDecoderLib.decode(data2);
        const channels = decoded1.channels;
        if (idx1 >= channels || idx2 >= channels) {
            return null;
        }
        const totalPixels = decoded1.width * decoded1.height;
        const result = new Array(totalPixels);
        for (let i = 0; i < totalPixels; i++) {
            const v1 = decoded1.data[i * channels + idx1];
            const v2 = decoded2.data[i * channels + idx2];
            if (raquetLib.isNodata(v1, nodata1) || raquetLib.isNodata(v2, nodata2)) {
                result[i] = null;
                continue;
            }
            const sum = v1 + v2;
            result[i] = sum === 0 ? 0 : (v1 - v2) / sum;
        }
        return result;
    }

    // WebP not yet supported
    if (meta.compression === 'webp') {
        throw new Error("WebP compression is not yet supported in BigQuery.");
    }

    // Decompress gzip if needed
    if (meta.compression === 'gzip') {
        data1 = raquetInflateLib.inflate(data1);
        data2 = raquetInflateLib.inflate(data2);
    }
    data1 = raquetLib.ensureOwnBuffer(data1);
    data2 = raquetLib.ensureOwnBuffer(data2);

    // Get type info
    const type1 = bandInfo1.type.toLowerCase();
    const type2 = bandInfo2.type.toLowerCase();

    const TYPE_SIZES = raquetLib.TYPE_SIZES;
    const size1 = TYPE_SIZES[type1];
    const size2 = TYPE_SIZES[type2];

    const view1 = new DataView(data1.buffer);
    const view2 = new DataView(data2.buffer);

    // Type readers (including float16 for ML/inference use cases)
    const readers = {
        'uint8': (v, o) => v.getUint8(o),
        'int8': (v, o) => v.getInt8(o),
        'uint16': (v, o) => v.getUint16(o, true),
        'int16': (v, o) => v.getInt16(o, true),
        'uint32': (v, o) => v.getUint32(o, true),
        'int32': (v, o) => v.getInt32(o, true),
        'float16': (v, o) => {
            const bits = v.getUint16(o, true);
            const sign = (bits >> 15) ? -1 : 1;
            const exp = (bits >> 10) & 0x1F;
            const frac = bits & 0x3FF;
            if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
            if (exp === 31) return frac ? NaN : sign * Infinity;
            return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
        },
        'float32': (v, o) => v.getFloat32(o, true),
        'float64': (v, o) => v.getFloat64(o, true)
    };

    const read1 = readers[type1];
    const read2 = readers[type2];

    // Compute normalized difference
    const result = new Array(pixelCount);
    for (let i = 0; i < pixelCount; i++) {
        // Calculate byte offsets based on band layout (v0.4.0)
        let offset1, offset2;
        if (bandLayout === 'interleaved') {
            // Band Interleaved by Pixel (BIP): [R0,G0,B0,R1,G1,B1,...]
            offset1 = (i * bandCount + idx1) * size1;
            offset2 = (i * bandCount + idx2) * size2;
        } else {
            // Sequential layout (default)
            offset1 = i * size1;
            offset2 = i * size2;
        }

        const v1 = read1(view1, offset1);
        const v2 = read2(view2, offset2);

        // Check nodata (handles NaN nodata correctly)
        if (raquetLib.isNodata(v1, nodata1) || raquetLib.isNodata(v2, nodata2)) {
            result[i] = null;
            continue;
        }

        // Compute (b1 - b2) / (b1 + b2)
        const sum = v1 + v2;
        if (sum === 0) {
            result[i] = 0;  // Avoid division by zero
        } else {
            result[i] = (v1 - v2) / sum;
        }
    }

    return result;
""";
