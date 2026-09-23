-- RAQUET_DECODE_BAND for Databricks SQL Warehouse
-- Decodes a compressed raster band into pixel values
-- Returns: JSON string containing array of pixel values (due to scalar-only limitation)
-- Parse with: FROM_JSON(result, 'ARRAY<DOUBLE>')
-- Supports: RaQuet v0.4.0 interleaved band layout and JPEG/WebP compression

CREATE OR REPLACE FUNCTION ${catalog}.${schema}.RAQUET_DECODE_BAND(
    band BINARY,
    metadata STRING,
    band_index INT
)
RETURNS STRING
LANGUAGE PYTHON
DETERMINISTIC
COMMENT 'Decodes compressed raster band to pixel array. Supports interleaved layout and JPEG/WebP. Returns JSON array string.'
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

    result = []
    pixel_count = block_width * block_height

    if len(img_array.shape) == 3:
        # RGB/RGBA image - extract single channel
        if band_index >= img_array.shape[2]:
            return None
        channel = img_array[:, :, band_index].flatten()
    else:
        # Grayscale
        if band_index != 0:
            return None
        channel = img_array.flatten()

    for val in channel:
        if nodata is not None and val == nodata:
            result.append(None)
        else:
            result.append(float(val))

    return json.dumps(result)

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

# Handle interleaved band layout (v0.4.0)
if band_layout == 'interleaved':
    # Band Interleaved by Pixel (BIP): [R0,G0,B0,R1,G1,B1,...]
    band_count = len(bands_meta)
    pixel_count = block_width * block_height
    # Extract every band_count-th value starting at band_index
    band_pixels = pixels[band_index::band_count]
else:
    band_pixels = pixels

# Convert to Python list and handle NaN/nodata
result = []
for val in band_pixels:
    if np.isnan(val) if np.issubdtype(dtype, np.floating) else False:
        result.append(None)
    elif nodata is not None and val == nodata:
        result.append(None)
    else:
        result.append(float(val))

return json.dumps(result)
$$;
