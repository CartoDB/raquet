-- ST_NORMALIZEDDIFFERENCE for Databricks SQL Warehouse
-- Computes normalized difference: (band1 - band2) / (band1 + band2)
-- Used for NDVI, NDWI, and other spectral indices
-- Returns: JSON string containing array of result values
-- Supports: RaQuet v0.4.0 interleaved band layout and JPEG/WebP compression

CREATE OR REPLACE FUNCTION ${catalog}.${schema}.ST_NORMALIZEDDIFFERENCE(
    band1 BINARY,
    band2 BINARY,
    metadata STRING,
    band_index1 INT,
    band_index2 INT
)
RETURNS STRING
LANGUAGE PYTHON
DETERMINISTIC
COMMENT 'Computes (b1-b2)/(b1+b2) for indices like NDVI. Supports interleaved layout and JPEG/WebP. Returns JSON array.'
ENVIRONMENT (
    dependencies = '["numpy", "pillow"]',
    environment_version = 'None'
)
AS $$
import numpy as np
import gzip
import json
import io

if band1 is None or band2 is None or metadata is None:
    return None

meta = json.loads(metadata)
bands_meta = meta.get('bands', [])
compression = meta.get('compression', 'none')
band_layout = meta.get('band_layout', 'sequential')
tiling = meta.get('tiling', {})
block_width = tiling.get('block_width', 256)
block_height = tiling.get('block_height', 256)

def decode_band(band_data, band_idx):
    if band_idx < 0 or band_idx >= len(bands_meta):
        return None
    band_meta = bands_meta[band_idx]
    dtype_str = band_meta.get('type', 'float32')

    # Handle JPEG/WebP lossy compression (v0.4.0)
    if compression in ('jpeg', 'webp'):
        from PIL import Image
        img = Image.open(io.BytesIO(bytes(band_data)))
        img_array = np.array(img)

        if len(img_array.shape) == 3:
            if band_idx >= img_array.shape[2]:
                return None
            pixels = img_array[:, :, band_idx].flatten().astype(np.float64)
        else:
            if band_idx != 0:
                return None
            pixels = img_array.flatten().astype(np.float64)
    else:
        # Standard decompression (gzip or none)
        if compression == 'gzip':
            data = gzip.decompress(band_data)
        else:
            data = bytes(band_data)

        dtype_map = {
            'uint8': np.uint8, 'int8': np.int8,
            'uint16': np.uint16, 'int16': np.int16,
            'uint32': np.uint32, 'int32': np.int32,
            'uint64': np.uint64, 'int64': np.int64,
            'float16': np.float16, 'float32': np.float32, 'float64': np.float64,
        }
        dtype = dtype_map.get(dtype_str, np.float32)
        all_pixels = np.frombuffer(data, dtype=dtype)

        # Handle interleaved band layout (v0.4.0)
        if band_layout == 'interleaved':
            band_count = len(bands_meta)
            pixels = all_pixels[band_idx::band_count].astype(np.float64)
        else:
            pixels = all_pixels.astype(np.float64)

    # Handle nodata
    nodata = band_meta.get('nodata')
    if nodata is not None:
        pixels = np.where(pixels == nodata, np.nan, pixels)

    return pixels

pixels1 = decode_band(band1, band_index1)
pixels2 = decode_band(band2, band_index2)

if pixels1 is None or pixels2 is None:
    return None

if len(pixels1) != len(pixels2):
    return None

# Compute normalized difference: (b1 - b2) / (b1 + b2)
numerator = pixels1 - pixels2
denominator = pixels1 + pixels2

with np.errstate(divide='ignore', invalid='ignore'):
    result = np.where(denominator != 0, numerator / denominator, np.nan)

# Convert to list, replacing NaN with None
output = []
for val in result:
    if np.isnan(val):
        output.append(None)
    else:
        output.append(float(val))

return json.dumps(output)
$$;
