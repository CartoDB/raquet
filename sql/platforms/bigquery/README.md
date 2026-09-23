# BigQuery Platform

## Requirements

- Google Cloud SDK with `bq` CLI
- GCS bucket for JavaScript libraries
- BigQuery dataset

## Deployment

```bash
# From the sql/ directory
# Build the JavaScript bundles first (not committed; deploy.sh uploads them to the bucket)
(cd libraries/javascript && npm ci && npm run build)

./deploy.sh bigquery --bucket gs://your-bucket --dataset yourproject.raquet

# Or using environment variables
export RAQUET_GCS_BUCKET=gs://your-bucket
export RAQUET_BQ_DATASET=yourproject.raquet
./deploy.sh bigquery
```

## Loading Data

```bash
# Single file
bq load --source_format=PARQUET yourproject:raquet.my_raster gs://bucket/raster.parquet

# Partitioned dataset (multiple files from raquet-io partition)
bq load --source_format=PARQUET yourproject:raquet.my_raster 'gs://bucket/raster_partitioned/*.parquet'
```

> **Note:** Partitioned datasets load multiple metadata rows (block=0). Use `LIMIT 1` in metadata CTEs to handle this correctly.

## Platform-Specific Notes

- **JavaScript UDFs**: Libraries loaded from external GCS bucket
- **Binary handling**: Data passed as base64-encoded strings
- **QUADBIN**: Requires CARTO Analytics Toolbox for BigQuery

## Functions

| Function | Status | Description |
|----------|--------|-------------|
| `RAQUET_DECODE_BAND` | ✅ | Decode compressed band to array |
| `RAQUET_PIXEL` | ✅ | Get pixel value at x,y |
| `ST_RASTERSUMMARYSTATS` | ✅ | Compute band statistics |
| `ST_RASTERVALUE` | ✅ | Get value at lon/lat |
| `ST_RASTERVALUE_GEOG` | ✅ | Get value at GEOGRAPHY point |
| `ST_BANDMATH` | ✅ | Pixel-wise band arithmetic |
| `ST_NORMALIZEDDIFFERENCE` | ✅ | NDVI/NDWI calculation |
| `ST_NORMALIZEDDIFFERENCESTATS` | ✅ | NDVI stats |
| `RAQUET_AGGREGATE_STATS` | ✅ | Aggregate stats across tiles |
| `__RAQUET_AUTO_ZOOM` | ✅ | Auto-detect zoom level |
| `__RAQUET_RESOLVE_ZOOM` | ✅ | Resolve target zoom |
| `__RAQUET_PIXEL_POSITIONS` | ✅ | Generate pixel grid |
| `__RAQUET_REGION_BLOCKS` | ✅ | Get blocks for region |
