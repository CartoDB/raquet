-- RAQUET_PIXEL for Databricks SQL Warehouse
-- Gets a single pixel value at tile coordinates (x, y)
-- Returns: DOUBLE (scalar)
-- Supports: RaQuet v0.4.0 interleaved band layout and JPEG/WebP compression

CREATE OR REPLACE FUNCTION ${catalog}.${schema}.RAQUET_PIXEL(
    band BINARY,
    metadata STRING,
    band_index INT,
    x INT,
    y INT
)
RETURNS DOUBLE
LANGUAGE PYTHON
DETERMINISTIC
COMMENT 'Gets pixel value at tile coordinates (x, y). Supports interleaved layout and JPEG/WebP. Tile is 256x256.'
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
tiling = meta.get('tiling', {})
block_width = tiling.get('block_width', 256)
block_height = tiling.get('block_height', 256)

if x < 0 or x >= block_width or y < 0 or y >= block_height:
    return None

bands_meta = meta.get('bands', [])

if band_index < 0 or band_index >= len(bands_meta):
    return None

band_meta = bands_meta[band_index]
compression = meta.get('compression', 'none')
band_layout = meta.get('band_layout', 'sequential')
dtype_str = band_meta.get('type', 'float32')

# Handle JPEG/WebP lossy compression (v0.4.0)
if compression in ('jpeg', 'webp'):
    from PIL import Image
    img = Image.open(io.BytesIO(bytes(band)))
    img_array = np.array(img)

    # For RGB/RGBA images, band_index selects the channel
    if len(img_array.shape) == 3:
        if band_index >= img_array.shape[2]:
            return None
        val = img_array[y, x, band_index]
    else:
        # Grayscale
        if band_index != 0:
            return None
        val = img_array[y, x]

    nodata = band_meta.get('nodata')
    if nodata is not None and val == nodata:
        return None
    return float(val)

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
pixels = np.frombuffer(data, dtype=dtype)

# Calculate index based on band layout (v0.4.0)
if band_layout == 'interleaved':
    # Band Interleaved by Pixel (BIP): [R0,G0,B0,R1,G1,B1,...]
    band_count = len(bands_meta)
    idx = (y * block_width + x) * band_count + band_index
else:
    # Sequential layout (default): separate band columns
    idx = y * block_width + x

if idx >= len(pixels):
    return None

val = pixels[idx]

# Check for nodata
nodata = band_meta.get('nodata')
if np.issubdtype(dtype, np.floating) and np.isnan(val):
    return None
if nodata is not None and val == nodata:
    return None

return float(val)
$$;
