-- ST_RASTERVALUE for Databricks SQL Warehouse
-- Gets raster value at a geographic point (lon, lat)
-- Returns: DOUBLE (scalar)
-- Supports: RaQuet v0.4.0 interleaved band layout and JPEG/WebP compression

CREATE OR REPLACE FUNCTION ${catalog}.${schema}.ST_RASTERVALUE(
    block BIGINT,
    band BINARY,
    lon DOUBLE,
    lat DOUBLE,
    metadata STRING,
    band_index INT
)
RETURNS DOUBLE
LANGUAGE PYTHON
DETERMINISTIC
COMMENT 'Gets raster value at geographic coordinates (lon, lat). Supports interleaved layout and JPEG/WebP.'
ENVIRONMENT (
    dependencies = '["numpy", "pillow"]',
    environment_version = 'None'
)
AS $$
import numpy as np
import gzip
import json
import math
import io

# QUADBIN helper functions
def lon_to_tile(lon, zoom):
    return ((lon + 180.0) / 360.0) * (1 << zoom)

def lat_to_tile(lat, zoom):
    lat_rad = math.radians(lat)
    n = 1 << zoom
    return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n

def quadbin_to_tile(quadbin):
    """Extract zoom, x, y from quadbin."""
    # Quadbin format: mode (4 bits) | zoom (5 bits) | reserved (3 bits) | cell index
    mode = (quadbin >> 59) & 0xF
    zoom = (quadbin >> 52) & 0x1F

    # Extract x and y by deinterleaving the cell index
    cell = quadbin & ((1 << 52) - 1)

    x = 0
    y = 0
    for i in range(26):
        # In quadbin, bits are interleaved as: y_i at position 2*i, x_i at position 2*i+1
        y |= ((cell >> (2 * i + 1)) & 1) << i
        x |= ((cell >> (2 * i)) & 1) << i

    # Adjust for zoom level
    shift = 26 - zoom
    x = x >> shift
    y = y >> shift

    return zoom, x, y

if band is None or metadata is None or block is None:
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

# Get tile info from block quadbin
block_zoom, tile_x, tile_y = quadbin_to_tile(block)

# Calculate pixel position within tile
# Global pixel coordinates at this zoom
global_pixel_x = lon_to_tile(lon, block_zoom) * block_width
global_pixel_y = lat_to_tile(lat, block_zoom) * block_height

# Local pixel within this tile
local_x = int(global_pixel_x - tile_x * block_width)
local_y = int(global_pixel_y - tile_y * block_height)

# Bounds check
if local_x < 0 or local_x >= block_width or local_y < 0 or local_y >= block_height:
    return None

# Handle JPEG/WebP lossy compression (v0.4.0)
if compression in ('jpeg', 'webp'):
    from PIL import Image
    img = Image.open(io.BytesIO(bytes(band)))
    img_array = np.array(img)

    if len(img_array.shape) == 3:
        if band_index >= img_array.shape[2]:
            return None
        val = img_array[local_y, local_x, band_index]
    else:
        if band_index != 0:
            return None
        val = img_array[local_y, local_x]

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
    idx = (local_y * block_width + local_x) * band_count + band_index
else:
    # Sequential layout (default)
    idx = local_y * block_width + local_x

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
