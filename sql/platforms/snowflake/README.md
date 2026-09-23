# Snowflake Platform

## Requirements

- SnowSQL CLI configured with connection
- Database and schema created
- CARTO Analytics Toolbox for Snowflake (for QUADBIN functions)

## Deployment

```bash
# From repository root
./deploy.sh snowflake --connection myconn --database MYDB --schema RAQUET

# Or using environment variables
export RAQUET_SF_CONNECTION=myconn
export RAQUET_SF_DATABASE=MYDB
export RAQUET_SF_SCHEMA=RAQUET
./deploy.sh snowflake
```

## Loading Data

```sql
-- Single file
COPY INTO MY_RASTER FROM @my_stage/raster.parquet
  FILE_FORMAT = (TYPE = PARQUET BINARY_AS_TEXT = FALSE)
  MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- Partitioned dataset (multiple files from raquet-io partition)
COPY INTO MY_RASTER FROM @my_stage/raster_partitioned/
  FILE_FORMAT = (TYPE = PARQUET BINARY_AS_TEXT = FALSE)
  MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
```

> **Note:** `BINARY_AS_TEXT = FALSE` is required for Raquet files with binary band columns. Partitioned datasets load multiple metadata rows (block=0); use `LIMIT 1` in metadata CTEs (shown in all examples below).

## Platform-Specific Notes

- **JavaScript UDFs**: Libraries inlined in SQL files (~25KB per function)
- **Binary handling**: Snowflake passes BINARY directly as Uint8Array
- **Block precision**: Block IDs passed as VARCHAR to preserve BigInt precision
- **QUADBIN**: Requires CARTO Analytics Toolbox for Snowflake

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
| `ST_NORMALIZEDDIFFERENCESTATS` | ✅ | Stats for normalized difference |
| `RAQUET_AGGREGATE_STATS` | ✅ | Aggregate stats across tiles |
| `RAQUET_PIXEL_GEOGRAPHY` | ✅ | Get GEOGRAPHY for pixel center |
| `__RAQUET_AUTO_ZOOM` | ✅ | Auto-detect zoom level |
| `__RAQUET_RESOLVE_ZOOM` | ✅ | Resolve target zoom |
| `__RAQUET_PIXEL_POSITIONS` | ✅ | Generate 256x256 pixel grid (VIEW) |
| `__RAQUET_REGION_BLOCKS` | ✅ | Get blocks for region (ARRAY) |

## Key Differences from BigQuery

| Aspect | BigQuery | Snowflake |
|--------|----------|-----------|
| Parameter names | lowercase | UPPERCASE |
| Binary input | base64 string | Uint8Array direct |
| Block type | INT64 | VARCHAR (precision) |
| Return types | STRUCT | OBJECT |
| Library loading | GCS external | Inlined |
