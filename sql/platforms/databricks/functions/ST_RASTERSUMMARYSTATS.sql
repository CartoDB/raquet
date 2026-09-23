-- ST_RASTERSUMMARYSTATS for Databricks SQL Warehouse
-- Computes statistics for a raster band tile
-- Returns: JSON string with {count, sum, mean, min, max, stddev}
-- Parse with: FROM_JSON(result, 'STRUCT<count:BIGINT, sum:DOUBLE, mean:DOUBLE, min:DOUBLE, max:DOUBLE, stddev:DOUBLE>')
-- Supports: RaQuet v0.4.0 interleaved band layout and JPEG/WebP compression
--
-- Two overloads:
--
-- 1. Original (decompresses band data at query time):
--    ST_RASTERSUMMARYSTATS(band, metadata, band_index) -> STRING (JSON)
--
-- 2. Pre-computed stats (no decompression, pure SQL — much faster):
--    ST_RASTERSUMMARYSTATS(count, sum, min, max, mean, stddev) -> STRING (JSON)
--
-- Both return the same JSON format. Both work with RAQUET_AGGREGATE_STATS.
--
-- Usage with pre-computed stats:
--   SELECT ${catalog}.${schema}.ST_RASTERSUMMARYSTATS(
--       band_1_count, band_1_sum, band_1_min, band_1_max, band_1_mean, band_1_stddev
--   ) AS stats
--   FROM my_raquet_table WHERE block != 0;

CREATE OR REPLACE FUNCTION ${catalog}.${schema}.ST_RASTERSUMMARYSTATS(
    band BINARY,
    metadata STRING,
    band_index INT
)
RETURNS STRING
LANGUAGE PYTHON
DETERMINISTIC
COMMENT 'Computes raster band statistics. Supports interleaved layout and JPEG/WebP. Returns JSON: {count, sum, mean, min, max, stddev}'
ENVIRONMENT (
    dependencies = '["numpy", "pillow"]',
    environment_version = 'None'
)
AS $$
import numpy as np
import gzip
import json
import io

if band is None or metadata is None:
    return None

meta = json.loads(metadata)
bands_meta = meta.get('bands', [])
tiling = meta.get('tiling', {})
block_width = tiling.get('block_width', 256)
block_height = tiling.get('block_height', 256)

if band_index < 0 or band_index >= len(bands_meta):
    return None

band_meta = bands_meta[band_index]
compression = meta.get('compression', 'none')
band_layout = meta.get('band_layout', 'sequential')
dtype_str = band_meta.get('type', 'float32')
nodata = band_meta.get('nodata')

# Handle JPEG/WebP lossy compression (v0.4.0)
if compression in ('jpeg', 'webp'):
    from PIL import Image
    img = Image.open(io.BytesIO(bytes(band)))
    img_array = np.array(img)

    if len(img_array.shape) == 3:
        if band_index >= img_array.shape[2]:
            return None
        pixels = img_array[:, :, band_index].flatten().astype(np.float64)
    else:
        if band_index != 0:
            return None
        pixels = img_array.flatten().astype(np.float64)
else:
    # Standard decompression (gzip or none)
    if compression == 'gzip':
        data = gzip.decompress(band)
    else:
        data = bytes(band)

    # Map dtype string to numpy dtype
    dtype_map = {
        'uint8': np.uint8,
        'int8': np.int8,
        'uint16': np.uint16,
        'int16': np.int16,
        'uint32': np.uint32,
        'int32': np.int32,
        'uint64': np.uint64,
        'int64': np.int64,
        'float16': np.float16,
        'float32': np.float32,
        'float64': np.float64,
    }

    dtype = dtype_map.get(dtype_str, np.float32)
    all_pixels = np.frombuffer(data, dtype=dtype)

    # Handle interleaved band layout (v0.4.0)
    if band_layout == 'interleaved':
        band_count = len(bands_meta)
        pixels = all_pixels[band_index::band_count].astype(np.float64)
    else:
        pixels = all_pixels.astype(np.float64)

# Filter out nodata values
mask = ~np.isnan(pixels)
if nodata is not None:
    mask &= (pixels != nodata)

valid_pixels = pixels[mask]

if len(valid_pixels) == 0:
    return json.dumps({
        'count': 0,
        'sum': None,
        'mean': None,
        'min': None,
        'max': None,
        'stddev': None
    })

count = int(len(valid_pixels))
total = float(np.sum(valid_pixels))
mean = float(np.mean(valid_pixels))
min_val = float(np.min(valid_pixels))
max_val = float(np.max(valid_pixels))
stddev = float(np.std(valid_pixels, ddof=0))  # Population stddev

return json.dumps({
    'count': count,
    'sum': total,
    'mean': mean,
    'min': min_val,
    'max': max_val,
    'stddev': stddev
})
$$;

-- 6-parameter overload: uses pre-computed tile statistics (no decompression)
-- Returns the same JSON format as the 3-parameter version.
CREATE OR REPLACE FUNCTION ${catalog}.${schema}.ST_RASTERSUMMARYSTATS(
    band_count BIGINT,
    band_sum DOUBLE,
    band_min DOUBLE,
    band_max DOUBLE,
    band_mean DOUBLE,
    band_stddev DOUBLE
)
RETURNS STRING
LANGUAGE SQL
DETERMINISTIC
COMMENT 'Returns pre-computed tile statistics as JSON (no decompression needed)'
RETURN TO_JSON(NAMED_STRUCT(
    'count', band_count,
    'sum', band_sum,
    'mean', band_mean,
    'min', band_min,
    'max', band_max,
    'stddev', band_stddev
));
