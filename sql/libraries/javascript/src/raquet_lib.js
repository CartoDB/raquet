/**
 * Raquet Library for BigQuery JavaScript UDFs
 * Provides base64 decoding, gzip decompression, and band decoding
 *
 * RaQuet v0.4.0 Support:
 * - Interleaved band layout (BIP): metadata.band_layout = 'interleaved'
 * - JPEG/WebP compression: metadata.compression = 'jpeg' | 'webp'
 */

// Base64 decoding (from CARTO analytics-toolbox pattern)
const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=';

function atob(input) {
    const str = String(input).replace(/[=]+$/, '');
    if (str.length % 4 === 1) {
        throw new Error('Invalid base64 string');
    }
    let bc = 0, bs, buffer, idx = 0, output = '';
    while ((buffer = str.charAt(idx++))) {
        buffer = chars.indexOf(buffer);
        if (~buffer) {
            bs = bc % 4 ? bs * 64 + buffer : buffer;
            if (bc++ % 4) {
                output += String.fromCharCode(255 & bs >> (-2 * bc & 6));
            }
        }
    }
    return output;
}

function btoa(input) {
    const str = String(input);
    let block, charCode, idx = 0, map = chars, output = '';
    for (
        ;
        str.charAt(idx | 0) || (map = '=', idx % 1);
        output += map.charAt(63 & block >> 8 - idx % 1 * 8)
    ) {
        charCode = str.charCodeAt(idx += 3 / 4);
        if (charCode > 0xFF) {
            throw new Error('Invalid character');
        }
        block = block << 8 | charCode;
    }
    return output;
}

function base64ToUint8Array(base64) {
    const binary = atob(base64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
}

/**
 * Ensure data has its own ArrayBuffer (not a view into a larger buffer)
 */
function ensureOwnBuffer(data) {
    if (data.byteOffset === 0 && data.buffer.byteLength === data.length) {
        return data;
    }
    // Copy to a new buffer
    const copy = new Uint8Array(data.length);
    copy.set(data);
    return copy;
}

// Type readers using DataView for proper endianness handling
const TYPE_READERS = {
    'uint8': (view, offset) => view.getUint8(offset),
    'int8': (view, offset) => view.getInt8(offset),
    'uint16': (view, offset) => view.getUint16(offset, true),
    'int16': (view, offset) => view.getInt16(offset, true),
    'uint32': (view, offset) => view.getUint32(offset, true),
    'int32': (view, offset) => view.getInt32(offset, true),
    'float16': (view, offset) => {
        // IEEE 754 half-precision (16-bit) to JavaScript number
        const bits = view.getUint16(offset, true);
        const sign = (bits >> 15) ? -1 : 1;
        const exponent = (bits >> 10) & 0x1F;
        const fraction = bits & 0x3FF;

        if (exponent === 0) {
            // Subnormal or zero
            return sign * Math.pow(2, -14) * (fraction / 1024);
        } else if (exponent === 31) {
            // Infinity or NaN
            return fraction ? NaN : sign * Infinity;
        }
        // Normalized number
        return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    },
    'float32': (view, offset) => view.getFloat32(offset, true),
    'float64': (view, offset) => view.getFloat64(offset, true),
    'uint64': (view, offset) => {
        const low = view.getUint32(offset, true);
        const high = view.getUint32(offset + 4, true);
        return high * 0x100000000 + low;
    },
    'int64': (view, offset) => {
        const low = view.getUint32(offset, true);
        const high = view.getInt32(offset + 4, true);
        return high * 0x100000000 + low;
    }
};

const TYPE_SIZES = {
    'uint8': 1, 'int8': 1,
    'uint16': 2, 'int16': 2,
    'uint32': 4, 'int32': 4,
    'uint64': 8, 'int64': 8,
    'float16': 2, 'float32': 4, 'float64': 8
};

/**
 * Parse nodata value from metadata, handling Zarr v3 string conventions
 * @param {*} nodata - The nodata value from metadata (number or string)
 * @returns {number|null} - Parsed nodata value (NaN, Infinity, -Infinity, or number)
 */
function parseNodata(nodata) {
    if (nodata === null || nodata === undefined) {
        return null;
    }
    if (typeof nodata === 'string') {
        if (nodata === 'NaN') return NaN;
        if (nodata === 'Infinity') return Infinity;
        if (nodata === '-Infinity') return -Infinity;
        // Try parsing as number string
        const parsed = Number(nodata);
        return Number.isNaN(parsed) ? null : parsed;
    }
    return nodata;
}

/**
 * Check if a value matches the nodata sentinel
 * Handles NaN comparison (NaN !== NaN in JavaScript)
 */
function isNodata(value, nodata) {
    if (nodata === null) return false;
    if (Number.isNaN(nodata)) return Number.isNaN(value);
    return value === nodata;
}

/**
 * Decode a single pixel value from band data
 * @param {Uint8Array} data - The raw band data
 * @param {string} bandType - Data type (uint8, float32, etc.)
 * @param {number} pixelOffset - Pixel offset in sequential layout, or pre-calculated offset for interleaved
 * @param {Object} options - Optional parameters for v0.4.0 features
 * @param {string} options.bandLayout - 'sequential' (default) or 'interleaved'
 * @param {number} options.bandIndex - Band index (required for interleaved layout)
 * @param {number} options.bandCount - Total number of bands (required for interleaved layout)
 */
function decodePixelValue(data, bandType, pixelOffset, options = {}) {
    const dataType = bandType.toLowerCase();
    const typeSize = TYPE_SIZES[dataType];
    const reader = TYPE_READERS[dataType];

    if (!reader) {
        throw new Error(`Unknown data type: ${dataType}`);
    }

    // Ensure data has its own buffer (not a view into pako's internal buffer)
    const cleanData = ensureOwnBuffer(data);
    const view = new DataView(cleanData.buffer);

    // Calculate byte offset based on band layout (v0.4.0)
    let byteOffset;
    if (options.bandLayout === 'interleaved') {
        // Band Interleaved by Pixel (BIP): offset = (pixelIndex * bandCount + bandIndex) * typeSize
        const bandIndex = options.bandIndex || 0;
        const bandCount = options.bandCount || 1;
        byteOffset = (pixelOffset * bandCount + bandIndex) * typeSize;
    } else {
        // Sequential layout (default)
        byteOffset = pixelOffset * typeSize;
    }

    return reader(view, byteOffset);
}

/**
 * Decode entire band to array of values
 * @param {Uint8Array} data - The raw band data
 * @param {string} bandType - Data type (uint8, float32, etc.)
 * @param {number} pixelCount - Number of pixels in the tile
 * @param {*} nodata - Nodata value from metadata
 * @param {Object} options - Optional parameters for v0.4.0 features
 * @param {string} options.bandLayout - 'sequential' (default) or 'interleaved'
 * @param {number} options.bandIndex - Band index (required for interleaved layout)
 * @param {number} options.bandCount - Total number of bands (required for interleaved layout)
 */
function decodeBand(data, bandType, pixelCount, nodata, options = {}) {
    const dataType = bandType.toLowerCase();
    const typeSize = TYPE_SIZES[dataType];
    const reader = TYPE_READERS[dataType];

    if (!reader) {
        throw new Error(`Unknown data type: ${dataType}`);
    }

    // Ensure data has its own buffer (not a view into pako's internal buffer)
    const cleanData = ensureOwnBuffer(data);
    const view = new DataView(cleanData.buffer);
    const result = new Array(pixelCount);

    // Parse nodata value (handles Zarr v3 string conventions)
    const parsedNodata = parseNodata(nodata);

    // Handle interleaved band layout (v0.4.0)
    const bandLayout = options.bandLayout || 'sequential';
    const bandIndex = options.bandIndex || 0;
    const bandCount = options.bandCount || 1;

    for (let i = 0; i < pixelCount; i++) {
        let byteOffset;
        if (bandLayout === 'interleaved') {
            // Band Interleaved by Pixel (BIP): [R0,G0,B0,R1,G1,B1,...]
            byteOffset = (i * bandCount + bandIndex) * typeSize;
        } else {
            // Sequential layout (default)
            byteOffset = i * typeSize;
        }

        const value = reader(view, byteOffset);
        // Handle nodata - return null for nodata values
        if (isNodata(value, parsedNodata)) {
            result[i] = null;
        } else {
            result[i] = value;
        }
    }

    return result;
}

/**
 * Compute streaming statistics for a band
 * @param {Uint8Array} data - The raw band data
 * @param {string} bandType - Data type (uint8, float32, etc.)
 * @param {number} pixelCount - Number of pixels in the tile
 * @param {*} nodata - Nodata value from metadata
 * @param {Object} options - Optional parameters for v0.4.0 features
 * @param {string} options.bandLayout - 'sequential' (default) or 'interleaved'
 * @param {number} options.bandIndex - Band index (required for interleaved layout)
 * @param {number} options.bandCount - Total number of bands (required for interleaved layout)
 */
function computeBandStats(data, bandType, pixelCount, nodata, options = {}) {
    const dataType = bandType.toLowerCase();
    const typeSize = TYPE_SIZES[dataType];
    const reader = TYPE_READERS[dataType];

    if (!reader) {
        throw new Error(`Unknown data type: ${dataType}`);
    }

    // Ensure data has its own buffer (not a view into pako's internal buffer)
    const cleanData = ensureOwnBuffer(data);
    const view = new DataView(cleanData.buffer);

    // Parse nodata value (handles Zarr v3 string conventions)
    const parsedNodata = parseNodata(nodata);

    // Handle interleaved band layout (v0.4.0)
    const bandLayout = options.bandLayout || 'sequential';
    const bandIndex = options.bandIndex || 0;
    const bandCount = options.bandCount || 1;

    let count = 0;
    let sum = 0;
    let sumSq = 0;
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < pixelCount; i++) {
        let byteOffset;
        if (bandLayout === 'interleaved') {
            // Band Interleaved by Pixel (BIP): [R0,G0,B0,R1,G1,B1,...]
            byteOffset = (i * bandCount + bandIndex) * typeSize;
        } else {
            // Sequential layout (default)
            byteOffset = i * typeSize;
        }

        const value = reader(view, byteOffset);

        // Skip nodata values (handles NaN nodata correctly)
        if (isNodata(value, parsedNodata)) {
            continue;
        }
        // Also skip NaN values even if nodata is something else
        if (Number.isNaN(value)) {
            continue;
        }

        count++;
        sum += value;
        sumSq += value * value;
        if (value < min) min = value;
        if (value > max) max = value;
    }

    if (count === 0) {
        return {
            count: 0,
            sum: null,
            mean: null,
            min: null,
            max: null,
            stddev: null
        };
    }

    const mean = sum / count;
    const variance = (sumSq / count) - (mean * mean);
    const stddev = Math.sqrt(Math.max(0, variance));

    return {
        count: count,
        sum: sum,
        mean: mean,
        min: min === Infinity ? null : min,
        max: max === -Infinity ? null : max,
        stddev: stddev
    };
}

/**
 * Check if compression type is lossy (JPEG/WebP) - v0.4.0
 * @param {string} compression - Compression type from metadata
 * @returns {boolean} True if lossy compression
 */
function isLossyCompression(compression) {
    return compression === 'jpeg' || compression === 'webp';
}

/**
 * Validate compression type and throw error for unsupported lossy compression
 * JPEG/WebP decoders are not yet available in JavaScript UDFs
 * @param {string} compression - Compression type from metadata
 */
function validateCompression(compression) {
    if (isLossyCompression(compression)) {
        throw new Error(
            `Lossy compression '${compression}' is not yet supported in JavaScript UDFs. ` +
            `Use Databricks (Python) for JPEG/WebP compressed RaQuet files.`
        );
    }
}

/**
 * Parse band layout from metadata - v0.4.0
 * @param {Object} metadata - Parsed metadata object
 * @returns {Object} Layout info: { bandLayout, bandCount }
 */
function parseBandLayout(metadata) {
    return {
        bandLayout: metadata.band_layout || 'sequential',
        bandCount: metadata.bands ? metadata.bands.length : 1
    };
}

// Export for BigQuery UDF
export default {
    atob: atob,
    btoa: btoa,
    base64ToUint8Array: base64ToUint8Array,
    ensureOwnBuffer: ensureOwnBuffer,
    parseNodata: parseNodata,
    isNodata: isNodata,
    decodePixelValue: decodePixelValue,
    decodeBand: decodeBand,
    computeBandStats: computeBandStats,
    isLossyCompression: isLossyCompression,
    validateCompression: validateCompression,
    parseBandLayout: parseBandLayout,
    TYPE_SIZES: TYPE_SIZES
};
