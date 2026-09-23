# Snowflake Feasibility Study for Raquet

## Executive Summary

Porting BigQuery Raquet to Snowflake is **highly feasible** with moderate effort. CARTO's Analytics Toolbox already provides full QUADBIN support for Snowflake, and Snowflake has robust JavaScript UDF and geospatial capabilities. The main technical challenge is adapting to Snowflake's constraint of no external library imports in JavaScript UDFs, requiring the Pako (gzip) library to be inlined.

**Overall Feasibility: HIGH** - Estimated effort: 2-4 weeks for a production-ready port.

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

### Proposed Snowflake Architecture

```
User SQL Queries
        ↓
SQL UDF/UDTF Layer (Snowflake Functions)
        ↓
JavaScript UDF Layer (Inlined Libraries)
        ↓
Raquet/Parquet Data Layer
        ↓
QUADBIN Spatial Indexing (CARTO Analytics Toolbox for Snowflake)
```

---

## 2. Component-by-Component Analysis

### 2.1 JavaScript UDF Support

| Feature | BigQuery | Snowflake | Compatibility |
|---------|----------|-----------|---------------|
| JavaScript UDFs | Yes | Yes | **Compatible** |
| External library loading | Yes (from GCS) | **No** | **Requires workaround** |
| Max code size | N/A | ~100KB compressed | Sufficient |
| V8 engine | Yes | Yes | **Compatible** |
| DataView/TypedArrays | Yes | Yes | **Compatible** |
| Base64 (atob/btoa) | Yes | Yes | **Compatible** |

**Key Challenge**: Snowflake does not support importing external libraries. The Pako library (~30KB minified) must be inlined directly into the UDF code. This is well within the 100KB limit.

**Solution**: Bundle Pako directly into `raquet_lib.js` using Rollup, creating a single self-contained IIFE that can be embedded in Snowflake UDFs.

### 2.2 Geospatial Functions

| BigQuery Function | Snowflake Equivalent | Notes |
|-------------------|---------------------|-------|
| `ST_GEOGPOINT(lon, lat)` | `ST_MAKEPOINT(lon, lat)` | Direct mapping |
| `ST_BOUNDINGBOX(geom)` | `ST_ENVELOPE(geom)` | Returns geometry, needs extraction |
| `ST_X(point)` | `ST_X(point)` | **Identical** |
| `ST_Y(point)` | `ST_Y(point)` | **Identical** |
| `ST_CONTAINS(geom, point)` | `ST_CONTAINS(geom, point)` | **Identical** |
| `GEOGRAPHY` type | `GEOGRAPHY` type | **Identical** |

**Assessment**: Excellent compatibility. Most geospatial functions have direct equivalents.

### 2.3 QUADBIN / Spatial Indexing

CARTO Analytics Toolbox is available for **both** BigQuery and Snowflake with equivalent QUADBIN support:

| Function | BigQuery | Snowflake | Status |
|----------|----------|-----------|--------|
| `QUADBIN_POLYFILL` | Yes | Yes | **Available** |
| `QUADBIN_POLYFILL_MODE` | Yes | Yes | **Available** |
| `QUADBIN_BBOX` | Yes | Yes | **Available** |
| `QUADBIN_BOUNDARY` | Yes | Yes | **Available** |
| `QUADBIN_TOZXY` | Yes | Yes | **Available** |
| `QUADBIN_FROMZXY` | Yes | Yes | **Available** |

**Assessment**: Full compatibility. CARTO maintains parity between platforms.

### 2.4 Table Functions (UDTFs)

| Feature | BigQuery | Snowflake | Notes |
|---------|----------|-----------|-------|
| SQL Table Functions | Yes | Yes (UDTFs) | Different syntax |
| JavaScript UDTFs | Yes | Yes | Different syntax |
| Python UDTFs | No | Yes | Additional option |
| Max output columns | N/A | 500 | Sufficient |

**BigQuery syntax**:
```sql
CREATE TABLE FUNCTION my_function(param INT64)
RETURNS TABLE<col1 INT64, col2 STRING>
AS (SELECT ...)
```

**Snowflake syntax**:
```sql
CREATE FUNCTION my_function(param INT)
RETURNS TABLE(col1 INT, col2 VARCHAR)
AS 'SELECT ...'
```

**Assessment**: Compatible with minor syntax adjustments.

### 2.5 SQL Syntax Differences

| Feature | BigQuery | Snowflake | Migration Effort |
|---------|----------|-----------|------------------|
| Data types | `INT64`, `FLOAT64`, `STRING` | `INT`, `FLOAT`, `VARCHAR` | Simple mapping |
| JSON extraction | `JSON_VALUE(col, '$.path')` | `col:path::string` or `JSON_EXTRACT_PATH_TEXT` | Moderate |
| Array generation | `GENERATE_ARRAY(1, 10)` | `ARRAY_GENERATE_RANGE(1, 11)` | Simple |
| STRUCT type | `STRUCT<field TYPE>` | `OBJECT` or separate columns | Moderate |
| UNNEST | `UNNEST(array)` | `FLATTEN(array)` | Simple |
| OPTIONS clause | `OPTIONS(library=[...])` | N/A (inline code) | Architecture change |

---

## 3. Function Migration Matrix

### Tier 1: Core Data Extraction (Low Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `RAQUET_DECODE_BAND` | Low | JS logic unchanged, embed library |
| `RAQUET_PIXEL` | Low | JS logic unchanged |
| `ST_RASTERVALUE` | Low | Replace ST_GEOGPOINT with ST_MAKEPOINT |
| `ST_RASTERVALUE_GEOG` | Low | Wrapper function |

### Tier 2: Statistics (Low Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `ST_RASTERSUMMARYSTATS` | Low | Return OBJECT instead of STRUCT |
| `RAQUET_AGGREGATE_STATS` | Low | Adjust STRUCT syntax |

### Tier 3: Band Operations (Low Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `ST_BANDMATH` | Low | JS logic unchanged |
| `ST_NORMALIZEDDIFFERENCE` | Low | JS logic unchanged |
| `ST_NORMALIZEDDIFFERENCESTATS` | Low | JS logic unchanged |

### Tier 4: Resolution Management (Moderate Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `__RAQUET_RESOLVE_ZOOM` | Moderate | Convert to Snowflake UDTF syntax |
| `__RAQUET_AUTO_ZOOM` | Moderate | Adjust array operations |

### Tier 5: Spatial Operations (Moderate Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `RAQUET_PIXEL_GEOGRAPHY` | Low | Use ST_MAKEPOINT |
| `__RAQUET_PIXEL_POSITIONS` | Moderate | Convert to Snowflake UDTF |
| `__RAQUET_REGION_BLOCKS` | Moderate | Adapt GENERATE_ARRAY to ARRAY_GENERATE_RANGE |

---

## 4. Technical Implementation Plan

### Phase 1: JavaScript Library Preparation (1 week)

1. **Bundle Pako inline**
   - Modify `rollup.config.js` to produce a single bundle with Pako embedded
   - Target output: `raquet_snowflake.js` (~50KB combined)
   - Verify all functions work without browser-specific APIs

2. **Test JavaScript compatibility**
   - Validate DataView and TypedArray operations in Snowflake V8
   - Test base64 encoding/decoding
   - Verify float16 decoding works correctly

### Phase 2: Core UDF Migration (1 week)

1. **Create Snowflake UDF templates**
   - Adapt SQL syntax (data types, return types)
   - Embed JavaScript library inline in each UDF
   - Map geospatial function names

2. **Migrate core functions**
   - `RAQUET_DECODE_BAND`
   - `RAQUET_PIXEL`
   - `ST_RASTERVALUE` / `ST_RASTERVALUE_GEOG`

### Phase 3: Advanced Functions (1 week)

1. **Migrate statistics functions**
   - Convert STRUCT returns to OBJECT or multi-column returns
   - Adapt aggregation patterns

2. **Migrate band math functions**
   - `ST_BANDMATH`
   - `ST_NORMALIZEDDIFFERENCE`

3. **Migrate table functions**
   - Convert BigQuery TABLE FUNCTIONs to Snowflake UDTFs
   - Adapt GENERATE_ARRAY to ARRAY_GENERATE_RANGE

### Phase 4: Integration & Testing (0.5-1 week)

1. **CARTO Analytics Toolbox integration**
   - Verify QUADBIN functions work correctly
   - Test spatial join patterns

2. **End-to-end testing**
   - Test with sample Raquet datasets
   - Performance benchmarking

3. **Documentation**
   - Snowflake-specific README
   - Installation instructions
   - Usage examples

---

## 5. Code Examples

### Example: ST_RASTERVALUE Migration

**BigQuery (current)**:
```sql
CREATE OR REPLACE FUNCTION `project.dataset.ST_RASTERVALUE`(
    block INT64,
    band BYTES,
    lon FLOAT64,
    lat FLOAT64,
    metadata STRING,
    band_index INT64
)
RETURNS FLOAT64
LANGUAGE js
OPTIONS (
    library = [
        "gs://cartobq-raquet-libs/raquet_lib.js",
        "gs://cartobq-raquet-libs/raquet_inflate.js"
    ]
)
AS r"""
    // ... JavaScript code
""";
```

**Snowflake (proposed)**:
```sql
CREATE OR REPLACE FUNCTION RAQUET.ST_RASTERVALUE(
    block INT,
    band BINARY,
    lon FLOAT,
    lat FLOAT,
    metadata VARCHAR,
    band_index INT
)
RETURNS FLOAT
LANGUAGE JAVASCRIPT
AS
$$
    // Inlined Pako library (~30KB)
    var pako = (function() { /* ... */ })();

    // Inlined raquet_lib.js
    var raquetLib = (function() { /* ... */ })();

    // Function implementation
    // ... (same logic, using raquetLib and pako)
$$;
```

### Example: Table Function Migration

**BigQuery (current)**:
```sql
CREATE OR REPLACE TABLE FUNCTION `project.dataset.__RAQUET_PIXEL_POSITIONS`()
RETURNS TABLE<x INT64, y INT64>
AS (
    SELECT x, y
    FROM UNNEST(GENERATE_ARRAY(0, 255)) AS x
    CROSS JOIN UNNEST(GENERATE_ARRAY(0, 255)) AS y
);
```

**Snowflake (proposed)**:
```sql
CREATE OR REPLACE FUNCTION RAQUET.__RAQUET_PIXEL_POSITIONS()
RETURNS TABLE(x INT, y INT)
AS
$$
    SELECT f1.value::int AS x, f2.value::int AS y
    FROM TABLE(FLATTEN(ARRAY_GENERATE_RANGE(0, 256))) f1
    CROSS JOIN TABLE(FLATTEN(ARRAY_GENERATE_RANGE(0, 256))) f2
$$;
```

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pako doesn't work in Snowflake V8 | Low | High | Test early; alternative: Python UDF for decompression |
| JavaScript size exceeds limit | Low | Medium | Minify aggressively; split into multiple UDFs |
| QUADBIN function differences | Low | Medium | Test all CARTO functions; consult CARTO docs |
| Performance regression | Medium | Medium | Benchmark early; optimize hot paths |
| Float16 decoding issues | Low | Low | Already pure JS; well-tested |

---

## 7. Alternative Approaches

### Option A: Pure SQL + Python UDFs (Not Recommended)
- Use Python for decompression (has native gzip support)
- More complex; two-language solution
- Python UDFs have higher latency

### Option B: External Functions (Not Recommended)
- Call external service for processing
- Network latency; operational complexity
- Requires cloud infrastructure

### Option C: JavaScript with Inlined Libraries (Recommended)
- Single-language solution
- Minimal architecture changes
- Best performance
- Easiest to maintain parity with BigQuery version

---

## 8. Conclusion

Porting BigQuery Raquet to Snowflake is **highly feasible** for the following reasons:

1. **CARTO Analytics Toolbox availability**: Full QUADBIN support already exists for Snowflake
2. **JavaScript UDF compatibility**: Snowflake's V8 engine supports all required features
3. **Geospatial function parity**: Nearly identical function names and capabilities
4. **Inline library workaround**: Pako can be bundled directly (~50KB total, well under 100KB limit)
5. **UDTF support**: Table functions are supported with minor syntax changes

**Recommended next steps**:
1. Prototype `RAQUET_DECODE_BAND` with inlined Pako to validate the approach
2. Create automated build process for Snowflake-compatible bundles
3. Migrate functions incrementally, testing each one
4. Document Snowflake-specific deployment process

---

## References

- [Snowflake JavaScript UDF Documentation](https://docs.snowflake.com/developer-guide/udf/javascript/udf-javascript-introduction)
- [Snowflake JavaScript UDF Limitations](https://docs.snowflake.com/en/developer-guide/udf/javascript/udf-javascript-limitations)
- [Snowflake Geospatial Functions](https://docs.snowflake.com/en/sql-reference/functions-geospatial)
- [Snowflake UDTFs Documentation](https://docs.snowflake.com/en/developer-guide/udf/sql/udf-sql-tabular-functions)
- [CARTO Analytics Toolbox for Snowflake - QUADBIN](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-snowflake/sql-reference/quadbin)
- [CARTO Spatial Indexes](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-snowflake/key-concepts/spatial-indexes)
- [RaQuet by CARTO](https://www.raquet.io/)
