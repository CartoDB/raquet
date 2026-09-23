# Databricks Feasibility Study for Raquet

## Executive Summary

Porting BigQuery Raquet to Databricks is **highly feasible** with a different technical approach. CARTO's Analytics Toolbox provides QUADBIN support for Databricks, and Databricks has excellent native geospatial capabilities with 80+ Spatial SQL functions. The main architectural change is that **JavaScript is not supported** - the implementation must use **Python UDFs** or leverage Databricks' native functions. Given Python's rich ecosystem (including native gzip support), this is actually an advantage.

**Overall Feasibility: HIGH** - Estimated effort: 3-5 weeks for a production-ready port.

---

## 1. Architecture Comparison

### Current BigQuery Architecture

```
User SQL Queries
        ↓
SQL UDF Layer (14 BigQuery Functions)
        ↓
JavaScript UDF Layer (External Libraries from GCS)
        ↓
Raquet/Parquet Data Layer
        ↓
QUADBIN Spatial Indexing (CARTO Analytics Toolbox)
```

### Proposed Databricks Architecture

```
User SQL Queries
        ↓
SQL UDF / Python UDF Layer (Databricks Functions)
        ↓
Python Processing (NumPy, gzip native)
        ↓
Raquet/Parquet Data Layer (Delta Lake / Iceberg)
        ↓
QUADBIN Spatial Indexing (CARTO Analytics Toolbox for Databricks)
```

---

## 2. Component-by-Component Analysis

### 2.1 UDF Support

| Feature | Databricks | Notes |
|---------|------------|-------|
| JavaScript UDFs | **No** | Not supported |
| Python UDFs | **Yes** | Full support, recommended |
| Pandas UDFs | **Yes** | 100x faster than row-by-row Python |
| Scala UDFs | Yes | Alternative option |
| SQL UDFs | Yes | For pure SQL logic |

**Key Advantage**: Python has native `gzip` module - no need for external Pako library!

### 2.2 Python UDF Capabilities

| Feature | Support | Notes |
|---------|---------|-------|
| NumPy | Yes | Pre-installed in runtime |
| gzip module | Yes | Native Python stdlib |
| struct module | Yes | Binary data parsing |
| Custom packages | Yes | Via `%pip install` or cluster config |
| Pandas UDFs | Yes | Vectorized, much faster |
| Type hints | Yes | Better optimization |

**Performance Note**: Pandas UDFs are up to 100x faster than standard Python UDFs because they use Apache Arrow to reduce serialization costs.

### 2.3 Native Geospatial Functions (80+ Functions)

Databricks now offers native Spatial SQL functions (public preview in DBR 17.1+):

| BigQuery Function | Databricks Equivalent | Notes |
|-------------------|----------------------|-------|
| `ST_GEOGPOINT(lon, lat)` | `ST_Point(lon, lat)` | Direct mapping |
| `ST_BOUNDINGBOX(geom)` | `ST_Envelope(geom)` | Returns geometry |
| `ST_X(point)` | `ST_X(point)` | **Identical** |
| `ST_Y(point)` | `ST_Y(point)` | **Identical** |
| `ST_CONTAINS(geom, point)` | `ST_Contains(geom, point)` | **Identical** |
| `GEOGRAPHY` type | `GEOGRAPHY` type | Native in DBR 17.1+ |
| `GEOMETRY` type | `GEOMETRY` type | Native in DBR 17.1+ |

**Additional Functions Available**:
- `ST_AsEWKB`, `ST_Dump`, `ST_ExteriorRing`, `ST_InteriorRingN`
- `ST_Azimuth`, `ST_Boundary`, `ST_ClosestPoint` (DBR 18.0+)
- `ST_Centroid`, `ST_Transform`, `ST_Distance`, `ST_Intersection`

### 2.4 QUADBIN / Spatial Indexing

CARTO Analytics Toolbox is available for Databricks:

| Function | Databricks Support | Status |
|----------|-------------------|--------|
| `QUADBIN_BBOX` | Yes | **Available** |
| `QUADBIN_BOUNDARY` | Yes | **Available** |
| `QUADBIN_CENTER` | Yes | **Available** |
| `QUADBIN_DISTANCE` | Yes | **Available** |
| `QUADBIN_FROMGEOGPOINT` | Yes | **Available** |
| `QUADBIN_FROMLONGLAT` | Yes | **Available** |
| `QUADBIN_FROMZXY` | Yes | **Available** |
| `QUADBIN_POLYFILL` | Yes | **Available** |
| `QUADBIN_RESOLUTION` | Yes | **Available** |
| `QUADBIN_TOCHILDREN` | Yes | **Available** |
| `QUADBIN_TOPARENT` | Yes | **Available** |
| `QUADBIN_TOZXY` | Yes | **Available** |

**Bonus**: Databricks also has native **H3 functions** built-in (since DBR 11.2):
- `h3_longlatash3`, `h3_pointash3`, `h3_boundaryaswkb`
- Full H3 Java library 3.7.0 included

### 2.5 SQL Syntax Differences

| Feature | BigQuery | Databricks | Migration Effort |
|---------|----------|------------|------------------|
| Data types | `INT64`, `FLOAT64`, `STRING` | `BIGINT`, `DOUBLE`, `STRING` | Simple |
| JSON extraction | `JSON_VALUE(col, '$.path')` | `col:path` or `get_json_object` | Simple |
| Array generation | `GENERATE_ARRAY(1, 10)` | `sequence(1, 10)` | Simple |
| STRUCT type | `STRUCT<field TYPE>` | `STRUCT<field: TYPE>` | Simple |
| UNNEST | `UNNEST(array)` | `explode(array)` | Simple |
| Table functions | `CREATE TABLE FUNCTION` | Python UDTFs | Moderate |
| Bytes type | `BYTES` | `BINARY` | Simple |

---

## 3. Function Migration Matrix

### Tier 1: Core Data Extraction (Moderate Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `RAQUET_DECODE_BAND` | Moderate | Python UDF with NumPy |
| `RAQUET_PIXEL` | Moderate | Python UDF |
| `ST_RASTERVALUE` | Moderate | Python UDF + native ST functions |
| `ST_RASTERVALUE_GEOG` | Low | SQL wrapper |

### Tier 2: Statistics (Low Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `ST_RASTERSUMMARYSTATS` | Low | NumPy makes this trivial |
| `RAQUET_AGGREGATE_STATS` | Low | SQL aggregation |

### Tier 3: Band Operations (Low Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `ST_BANDMATH` | Low | NumPy vectorized operations |
| `ST_NORMALIZEDDIFFERENCE` | Low | NumPy array math |
| `ST_NORMALIZEDDIFFERENCESTATS` | Low | NumPy + statistics |

### Tier 4: Resolution Management (Low Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `__RAQUET_RESOLVE_ZOOM` | Low | SQL scalar function |
| `__RAQUET_AUTO_ZOOM` | Low | SQL with sequence() |

### Tier 5: Spatial Operations (Moderate Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `RAQUET_PIXEL_GEOGRAPHY` | Low | Use ST_Point |
| `__RAQUET_PIXEL_POSITIONS` | Moderate | Python UDTF or explode |
| `__RAQUET_REGION_BLOCKS` | Moderate | CARTO QUADBIN_POLYFILL |

---

## 4. Technical Implementation Plan

### Phase 1: Environment Setup (0.5 week)

1. **Install CARTO Analytics Toolbox**
   - Available free in Databricks workspace
   - Follow CARTO installation guide

2. **Configure cluster with required packages**
   ```python
   # Cluster init script or notebook
   %pip install numpy
   # Note: gzip and struct are stdlib, no install needed
   ```

3. **Verify native Spatial SQL functions**
   ```sql
   -- Check DBR version supports native spatial
   SELECT ST_Point(0, 0);
   ```

### Phase 2: Python UDF Development (1-2 weeks)

1. **Create core decoding library**
   ```python
   # raquet_lib.py
   import gzip
   import struct
   import base64
   import numpy as np
   from typing import List, Optional

   def decode_band(band_bytes: bytes, metadata: dict, band_index: int) -> np.ndarray:
       """Decode a compressed raster band to numpy array."""
       band_meta = metadata['bands'][band_index]

       # Decompress if needed
       if metadata.get('compression') == 'gzip':
           band_bytes = gzip.decompress(band_bytes)

       # Get dtype from metadata
       dtype_map = {
           'uint8': np.uint8, 'int8': np.int8,
           'uint16': np.uint16, 'int16': np.int16,
           'uint32': np.uint32, 'int32': np.int32,
           'float32': np.float32, 'float64': np.float64,
       }
       dtype = dtype_map.get(band_meta['type'], np.float32)

       # Decode to numpy array
       pixels = np.frombuffer(band_bytes, dtype=dtype)
       return pixels.reshape(256, 256)
   ```

2. **Register as Databricks UDF**
   ```python
   from pyspark.sql.functions import udf
   from pyspark.sql.types import ArrayType, DoubleType

   @udf(returnType=ArrayType(DoubleType()))
   def raquet_decode_band(band: bytes, metadata: str, band_index: int):
       import json
       import gzip
       import numpy as np

       meta = json.loads(metadata)
       # ... decoding logic
       return pixels.flatten().tolist()

   spark.udf.register("RAQUET_DECODE_BAND", raquet_decode_band)
   ```

### Phase 3: Pandas UDF Optimization (1 week)

For better performance, use Pandas UDFs:

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf("array<double>")
def raquet_decode_band_vectorized(
    band_series: pd.Series,
    metadata_series: pd.Series,
    band_index_series: pd.Series
) -> pd.Series:
    results = []
    for band, metadata, band_index in zip(band_series, metadata_series, band_index_series):
        pixels = decode_band(band, json.loads(metadata), band_index)
        results.append(pixels.flatten().tolist())
    return pd.Series(results)
```

### Phase 4: SQL Function Migration (1 week)

1. **Migrate SQL-based functions**
   ```sql
   -- RAQUET_PIXEL_GEOGRAPHY
   CREATE OR REPLACE FUNCTION raquet.pixel_geography(
       block BIGINT,
       x INT,
       y INT,
       metadata STRING
   )
   RETURNS GEOMETRY
   RETURN (
       WITH tile_info AS (
           SELECT
               get_json_object(metadata, '$.tiling.pixel_zoom') AS pixel_zoom,
               QUADBIN_BBOX(block) AS bbox
       )
       SELECT ST_Point(
           bbox[0] + (x + 0.5) * (bbox[2] - bbox[0]) / 256,
           bbox[3] - (y + 0.5) * (bbox[3] - bbox[1]) / 256
       )
       FROM tile_info
   );
   ```

2. **Migrate table-generating functions**
   ```sql
   -- __RAQUET_PIXEL_POSITIONS using explode
   CREATE OR REPLACE FUNCTION raquet.pixel_positions()
   RETURNS TABLE(x INT, y INT)
   RETURN
       SELECT col1 AS x, col2 AS y
       FROM (
           SELECT explode(sequence(0, 255)) AS col1
       ) a
       CROSS JOIN (
           SELECT explode(sequence(0, 255)) AS col2
       ) b;
   ```

### Phase 5: Integration & Testing (0.5-1 week)

1. **Test with Raquet sample data**
2. **Benchmark Pandas UDFs vs standard UDFs**
3. **Verify CARTO QUADBIN integration**
4. **Document Databricks-specific usage**

---

## 5. Code Examples

### Example: Complete RAQUET_DECODE_BAND

```python
from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import ArrayType, DoubleType, StructType, StructField
import pandas as pd
import numpy as np
import gzip
import json

def _decode_band_impl(band_bytes: bytes, metadata_str: str, band_index: int) -> list:
    """Core decoding logic."""
    metadata = json.loads(metadata_str)
    band_meta = metadata['bands'][band_index]

    # Decompress
    if metadata.get('compression') == 'gzip':
        band_bytes = gzip.decompress(band_bytes)

    # Type mapping
    dtype_map = {
        'uint8': np.uint8, 'int8': np.int8,
        'uint16': np.uint16, 'int16': np.int16,
        'uint32': np.uint32, 'int32': np.int32,
        'uint64': np.uint64, 'int64': np.int64,
        'float16': np.float16, 'float32': np.float32, 'float64': np.float64,
    }
    dtype = dtype_map.get(band_meta['type'], np.float32)

    # Decode
    pixels = np.frombuffer(band_bytes, dtype=dtype)

    # Handle nodata
    nodata = band_meta.get('nodata')
    if nodata is not None:
        pixels = np.where(pixels == nodata, np.nan, pixels.astype(np.float64))

    return pixels.tolist()

# Pandas UDF for vectorized performance
@pandas_udf("array<double>")
def raquet_decode_band(
    band: pd.Series,
    metadata: pd.Series,
    band_index: pd.Series
) -> pd.Series:
    return pd.Series([
        _decode_band_impl(b, m, i)
        for b, m, i in zip(band, metadata, band_index)
    ])

# Register for SQL use
spark.udf.register("RAQUET_DECODE_BAND", raquet_decode_band)
```

### Example: ST_RASTERVALUE with Native Spatial

```python
@pandas_udf("double")
def st_rastervalue(
    block: pd.Series,
    band: pd.Series,
    lon: pd.Series,
    lat: pd.Series,
    metadata: pd.Series,
    band_index: pd.Series
) -> pd.Series:
    import json
    import gzip
    import numpy as np

    results = []
    for blk, bnd, x, y, meta_str, idx in zip(block, band, lon, lat, metadata, band_index):
        meta = json.loads(meta_str)

        # Get tile bounds from QUADBIN (simplified - use CARTO functions in practice)
        # In real implementation, call QUADBIN_BBOX via Spark SQL

        # Decompress band
        if meta.get('compression') == 'gzip':
            bnd = gzip.decompress(bnd)

        # Calculate pixel position from lon/lat
        # ... Web Mercator math ...

        # Extract pixel value
        # ...

        results.append(pixel_value)

    return pd.Series(results)

spark.udf.register("ST_RASTERVALUE", st_rastervalue)
```

### Example: SQL Usage

```sql
-- Query raster value at a point
SELECT
    r.block,
    ST_RASTERVALUE(r.block, r.band_1, -122.4, 37.8, m.metadata, 0) as elevation
FROM raquet_dem r
CROSS JOIN (SELECT metadata FROM raquet_dem WHERE block = 0) m
WHERE r.block IN (
    SELECT QUADBIN_FROMLONGLAT(-122.4, 37.8, 14)
);

-- Compute NDVI for a region
WITH region_blocks AS (
    SELECT QUADBIN_POLYFILL(ST_GeomFromText('POLYGON(...)'), 14) as blocks
)
SELECT
    r.block,
    ST_NORMALIZEDDIFFERENCE(r.band_nir, r.band_red, m.metadata, 3, 2) as ndvi
FROM raquet_satellite r
CROSS JOIN (SELECT metadata FROM raquet_satellite WHERE block = 0) m
WHERE r.block IN (SELECT explode(blocks) FROM region_blocks);
```

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pandas UDF serialization overhead | Medium | Medium | Batch operations; use Arrow |
| DBR version compatibility | Low | Medium | Pin to DBR 17.1+ |
| CARTO Toolbox version gaps | Low | Low | Track release notes |
| Native spatial preview issues | Medium | Low | Fall back to Sedona if needed |
| Float16 support in NumPy | Low | Low | NumPy supports float16 natively |
| Large tile memory usage | Low | Medium | Configure executor memory |

---

## 7. Performance Considerations

### Pandas UDFs vs Standard UDFs

| Approach | Performance | Use Case |
|----------|-------------|----------|
| Standard Python UDF | Baseline | Simple, low-volume |
| Pandas UDF | 10-100x faster | Production workloads |
| Scala UDF | Fastest | Maximum performance |

### Optimization Tips

1. **Use Pandas UDFs** for all production functions
2. **Leverage NumPy** for vectorized array operations
3. **Enable Apache Arrow** for efficient serialization
4. **Use Delta Lake clustering** on QUADBIN column for spatial queries
5. **Cache metadata** to avoid repeated parsing

---

## 8. Advantages Over BigQuery Implementation

| Aspect | BigQuery | Databricks |
|--------|----------|------------|
| Decompression | Requires Pako library | Native Python gzip |
| Array operations | JavaScript arrays | NumPy (much faster) |
| Statistics | Custom streaming | NumPy built-in |
| Type support | Custom readers | NumPy dtype |
| Float16 | Custom decoder | `np.float16` native |
| Debugging | Limited | Full Python debugging |
| Testing | Complex | Standard pytest |

---

## 9. Alternative Approaches

### Option A: Python UDFs with NumPy (Recommended)
- Best balance of performance and maintainability
- Native gzip, NumPy for arrays
- Pandas UDFs for production

### Option B: Apache Sedona Integration
- Use Sedona for advanced spatial operations
- May conflict with native Databricks functions
- More complex setup

### Option C: Scala UDFs
- Maximum performance
- Steeper learning curve
- Less readable

---

## 10. Conclusion

Porting BigQuery Raquet to Databricks is **highly feasible** and in some ways **easier than BigQuery**:

**Pros**:
1. CARTO Analytics Toolbox provides full QUADBIN support
2. Native 80+ Spatial SQL functions (DBR 17.1+)
3. Python has native gzip - no external library needed
4. NumPy provides faster array operations than JavaScript
5. Pandas UDFs offer excellent performance
6. Native GEOMETRY and GEOGRAPHY types
7. Better debugging and testing capabilities
8. Delta Lake integration for optimized storage

**Cons**:
1. No JavaScript support (but Python is arguably better for this use case)
2. Learning curve for teams unfamiliar with PySpark
3. Native spatial functions still in preview

**Recommended next steps**:
1. Install CARTO Analytics Toolbox in test workspace
2. Prototype RAQUET_DECODE_BAND with Pandas UDF
3. Benchmark against BigQuery implementation
4. Migrate functions incrementally

---

## References

- [Databricks Spatial SQL Functions (80+)](https://www.databricks.com/blog/introducing-spatial-sql-databricks-80-functions-high-performance-geospatial-analytics)
- [Databricks H3 Geospatial Functions](https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-h3-geospatial-functions)
- [Databricks Python UDFs](https://docs.databricks.com/aws/en/udf/python)
- [Databricks User-Defined Functions Overview](https://docs.databricks.com/aws/en/udf/)
- [CARTO Analytics Toolbox for Databricks](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-databricks)
- [CARTO for Databricks Announcement](https://carto.com/blog/carto-for-databricks-true-native-geospatial-for-the-lakehouse)
- [Apache Sedona on Databricks](https://sedona.apache.org/latest/setup/databricks/)
