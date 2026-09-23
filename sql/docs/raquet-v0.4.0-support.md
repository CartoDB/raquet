# RaQuet v0.4.0 Support

This document describes the RaQuet v0.4.0 features supported in this experimental branch.

## New Features in v0.4.0

### 1. Interleaved Band Layout

v0.4.0 introduces a new `band_layout` metadata field that supports two layouts:

- **`sequential`** (default): Traditional format with separate columns for each band (`band_1`, `band_2`, `band_3`, etc.)
- **`interleaved`**: All bands stored in a single `pixels` column using Band Interleaved by Pixel (BIP) format

#### BIP Format

In interleaved mode, pixels are stored as: `[R₀,G₀,B₀,R₁,G₁,B₁,...,Rₙ,Gₙ,Bₙ]`

The pixel offset calculation changes from:
```
// Sequential: offset = (y * width + x) * bytesPerPixel
// Interleaved: offset = ((y * width + x) * bandCount + bandIndex) * bytesPerPixel
```

#### Metadata Detection

```javascript
const bandLayout = metadata.band_layout || 'sequential';  // default: sequential
const columnName = bandLayout === 'interleaved' ? 'pixels' : `band_${bandIndex + 1}`;
```

### 2. Lossy Compression (JPEG/WebP)

v0.4.0 adds support for lossy compression formats:

- **`jpeg`**: JPEG compression for RGB imagery
- **`webp`**: WebP compression for RGB/RGBA imagery

#### Metadata Fields

```json
{
  "compression": "jpeg",  // or "webp"
  "compression_quality": 85,  // optional, 1-100
  "band_layout": "interleaved"  // required for lossy compression
}
```

#### Constraints

- Only valid with `band_layout: "interleaved"`
- Only valid with `uint8` data type
- Requires full tile decompression (cannot extract single pixels efficiently)

## Platform Support

| Feature | Databricks | BigQuery | Snowflake |
|---------|-----------|----------|-----------|
| Interleaved + gzip | ✅ Full | ✅ Full | ✅ Full |
| JPEG compression | ✅ Full (PIL) | ✅ Full (jpeg_decoder.js) | ✅ Full (inline jpeg_decoder) |
| WebP compression | ✅ Full (PIL) | ❌ Not yet | ❌ Not yet |

> **Note**: JPEG support was added using a pure JavaScript decoder (~10KB minified).
> WebP compression is not yet supported in BigQuery/Snowflake.

### Databricks

Full support for all v0.4.0 features using Pillow (PIL) for image decoding.

```sql
-- Interleaved + gzip: use 'pixels' column
SELECT RAQUET_PIXEL(pixels, metadata, 0, 100, 100) AS red_value
FROM raquet_table
WHERE block != 0;

-- JPEG/WebP: same syntax, automatically detected from metadata
SELECT RAQUET_PIXEL(pixels, metadata, 0, 100, 100) AS red_value
FROM raquet_jpeg_table
WHERE block != 0;
```

Dependencies added to ENVIRONMENT:
- `numpy`
- `pillow`

### BigQuery

JPEG compression is now fully supported using a pure JavaScript decoder (`jpeg_decoder.js`). WebP compression is not yet supported.

```sql
-- JPEG compressed data: same syntax, compression auto-detected from metadata
SELECT
    block,
    `cartobq.raquet.RAQUET_PIXEL`(pixels, metadata, 0, 100, 100) AS red,
    `cartobq.raquet.ST_RASTERSUMMARYSTATS`(pixels, metadata, 0) AS red_stats
FROM `project.dataset.jpeg_raster`
WHERE block != 0;
```

### Snowflake

JPEG compression is supported with the JPEG decoder inlined in each function. WebP compression is not yet supported.

When attempting to use WebP files, you'll receive an error:
```
Error: WebP compression is not yet supported in BigQuery/Snowflake.
```

## Test Files

Test files are available at:

| Format | File | Size | GCS URL |
|--------|------|------|---------|
| Sequential + gzip (baseline) | tci_sequential_gzip.parquet | 256 MB | gs://raquet_demo_data/experimental/tci_sequential_gzip.parquet |
| Interleaved + gzip | tci_interleaved_gzip.parquet | 286 MB | gs://raquet_demo_data/experimental/tci_interleaved_gzip.parquet |
| Interleaved + JPEG | tci_interleaved_jpeg.parquet | 27 MB | gs://raquet_demo_data/experimental/tci_interleaved_jpeg.parquet |
| Interleaved + WebP | tci_interleaved_webp.parquet | 17 MB | gs://raquet_demo_data/experimental/tci_interleaved_webp.parquet |

All files contain the same Sentinel-2 TCI imagery (Sudan/Nile region, 10980×10980 pixels, 3 bands RGB uint8).

## Usage Examples

### Databricks - Interleaved Layout

```sql
-- Read a single pixel from interleaved data
SELECT
    block,
    ${catalog}.${schema}.RAQUET_PIXEL(pixels, metadata, 0, 100, 100) AS red,
    ${catalog}.${schema}.RAQUET_PIXEL(pixels, metadata, 1, 100, 100) AS green,
    ${catalog}.${schema}.RAQUET_PIXEL(pixels, metadata, 2, 100, 100) AS blue
FROM interleaved_raster
WHERE block != 0
LIMIT 10;
```

### Databricks - JPEG/WebP Compressed

```sql
-- Same syntax as interleaved, compression is auto-detected
SELECT
    block,
    ${catalog}.${schema}.RAQUET_PIXEL(pixels, metadata, 0, 100, 100) AS red,
    ${catalog}.${schema}.ST_RASTERSUMMARYSTATS(pixels, metadata, 0) AS red_stats
FROM jpeg_compressed_raster
WHERE block != 0;
```

### BigQuery - Interleaved Layout

```sql
-- Read from interleaved data (pass 'pixels' column instead of band_N)
SELECT
    block,
    `cartobq.raquet.RAQUET_PIXEL`(pixels, metadata, 0, 100, 100) AS red,
    `cartobq.raquet.RAQUET_PIXEL`(pixels, metadata, 1, 100, 100) AS green,
    `cartobq.raquet.RAQUET_PIXEL`(pixels, metadata, 2, 100, 100) AS blue
FROM `project.dataset.interleaved_raster`
WHERE block != 0
LIMIT 10;
```

## Future Work

### WebP Support for BigQuery/Snowflake

JPEG support has been added. To add WebP support, we need to:

1. Find or create a pure JavaScript WebP decoder
2. Minify and add to the library files
3. Update the UDFs to use the decoder

Current library sizes (minified):
- `raquet_lib.js`: ~3.4KB
- `raquet_inflate.js`: ~20.9KB
- `jpeg_decoder.js`: ~9.9KB

Candidate WebP libraries:
- [libwebp.js](https://chromium.googlesource.com/webm/libwebp/) - Would need JS port or WebAssembly
- Consider using WebAssembly-compiled libwebp for performance

## Specification Reference

Full v0.4.0 specification: https://github.com/CartoDB/raquet/blob/experiment/unified-bands/format-specs/raquet.md
