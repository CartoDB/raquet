# Amazon Redshift Feasibility Study for Raquet

## Executive Summary

Porting BigQuery Raquet to Amazon Redshift is **feasible with moderate complexity**. CARTO's Analytics Toolbox provides full QUADBIN support for Redshift, and Redshift has robust geospatial capabilities. However, the main challenge is that **Python UDFs are being deprecated** (end of support June 2026), and Redshift does not support native JavaScript UDFs. The recommended approach is to use **Lambda UDFs** with Node.js, which provides full JavaScript support but adds architectural complexity.

**Overall Feasibility: MODERATE** - Estimated effort: 4-6 weeks for a production-ready port.

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

### Proposed Redshift Architecture

```
User SQL Queries
        ↓
SQL UDF Layer (Redshift Functions)
        ↓
Lambda UDF Layer (Node.js JavaScript Runtime)
        ↓
Raquet/Parquet Data Layer
        ↓
QUADBIN Spatial Indexing (CARTO Analytics Toolbox for Redshift)
```

---

## 2. Component-by-Component Analysis

### 2.1 UDF Support Options

| Option | Status | JavaScript Support | Notes |
|--------|--------|-------------------|-------|
| Python UDFs | **Deprecated** | No | End of support June 30, 2026 |
| Lambda UDFs | **Recommended** | Yes (Node.js) | Full JavaScript via AWS Lambda |
| SQL UDFs | Supported | No | Limited to SQL expressions |

**Critical Note**: AWS announced that Python UDFs will reach end of support after June 30, 2026:
- November 1, 2025: Creation of new Python UDFs disabled
- January 31, 2026: Creation completely blocked
- June 30, 2026: Execution of existing Python UDFs suspended

**Recommended Approach**: Use Lambda UDFs with Node.js runtime for JavaScript support.

### 2.2 Lambda UDF Capabilities

| Feature | Support | Notes |
|---------|---------|-------|
| Node.js runtime | Yes | Multiple versions available |
| External npm packages | Yes | Include Pako in deployment package |
| DataView/TypedArrays | Yes | Full V8 engine support |
| Base64 encoding | Yes | Native `Buffer` support |
| Max payload | 5MB | Sufficient for raster tiles |
| Timeout | 15 minutes max | Configurable |

**Advantages of Lambda UDFs**:
- Enhanced integration with external services
- Multiple Python/Node.js runtime versions
- Independent scaling from Redshift cluster
- Security patches available within a month of release

### 2.3 Geospatial Functions

| BigQuery Function | Redshift Equivalent | Notes |
|-------------------|---------------------|-------|
| `ST_GEOGPOINT(lon, lat)` | `ST_Point(lon, lat)` | Direct mapping |
| `ST_BOUNDINGBOX(geom)` | `ST_Envelope(geom)` | Returns geometry |
| `ST_X(point)` | `ST_X(point)` | **Identical** |
| `ST_Y(point)` | `ST_Y(point)` | **Identical** |
| `ST_CONTAINS(geom, point)` | `ST_Contains(geom, point)` | **Identical** |
| `GEOGRAPHY` type | `GEOGRAPHY` type | **Identical** |
| `GEOMETRY` type | `GEOMETRY` type | **Identical** |

**Assessment**: Excellent geospatial compatibility. Redshift supports 40+ spatial functions.

### 2.4 QUADBIN / Spatial Indexing

CARTO Analytics Toolbox is available for Redshift with full QUADBIN support:

| Function | Redshift Support | Status |
|----------|------------------|--------|
| `QUADBIN_FROMGEOPOINT` | Yes | **Available** |
| `QUADBIN_POLYFILL` | Yes | **Available** |
| `QUADBIN_BBOX` | Yes | **Available** |
| `QUADBIN_BOUNDARY` | Yes | **Available** |
| `QUADBIN_FROMZXY` | Yes | **Available** |
| `QUADBIN_TOZXY` | Yes | **Available** |
| `QUADBIN_KRING` | Yes | **Available** |
| `QUADBIN_DISTANCE` | Yes | **Available** |
| `QUADBIN_ISVALID` | Yes | **Available** |

**Assessment**: Full QUADBIN compatibility via CARTO Analytics Toolbox.

### 2.5 SQL Syntax Differences

| Feature | BigQuery | Redshift | Migration Effort |
|---------|----------|----------|------------------|
| Data types | `INT64`, `FLOAT64`, `STRING` | `BIGINT`, `FLOAT8`, `VARCHAR` | Simple mapping |
| JSON extraction | `JSON_VALUE(col, '$.path')` | `JSON_EXTRACT_PATH_TEXT(col, 'path')` | Moderate |
| Array generation | `GENERATE_ARRAY(1, 10)` | Custom function needed | Moderate |
| STRUCT type | `STRUCT<field TYPE>` | `SUPER` type or composite | Moderate |
| UNNEST | `UNNEST(array)` | Different syntax | Moderate |
| Table functions | `CREATE TABLE FUNCTION` | Not supported natively | **Architecture change** |

---

## 3. Function Migration Matrix

### Tier 1: Core Data Extraction (Moderate Effort - Lambda Required)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `RAQUET_DECODE_BAND` | Moderate | Lambda UDF with Node.js |
| `RAQUET_PIXEL` | Moderate | Lambda UDF with Node.js |
| `ST_RASTERVALUE` | Moderate | Lambda UDF + geospatial SQL |
| `ST_RASTERVALUE_GEOG` | Moderate | Wrapper function |

### Tier 2: Statistics (Moderate Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `ST_RASTERSUMMARYSTATS` | Moderate | Lambda UDF, return JSON/SUPER |
| `RAQUET_AGGREGATE_STATS` | Moderate | SQL aggregation logic |

### Tier 3: Band Operations (Moderate Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `ST_BANDMATH` | Moderate | Lambda UDF |
| `ST_NORMALIZEDDIFFERENCE` | Moderate | Lambda UDF |
| `ST_NORMALIZEDDIFFERENCESTATS` | Moderate | Lambda UDF |

### Tier 4: Resolution Management (High Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `__RAQUET_RESOLVE_ZOOM` | High | No table functions; redesign needed |
| `__RAQUET_AUTO_ZOOM` | Moderate | SQL scalar function |

### Tier 5: Spatial Operations (High Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `RAQUET_PIXEL_GEOGRAPHY` | Moderate | SQL with ST_Point |
| `__RAQUET_PIXEL_POSITIONS` | High | No table functions; use CTEs |
| `__RAQUET_REGION_BLOCKS` | High | Redesign with CARTO QUADBIN |

---

## 4. Technical Implementation Plan

### Phase 1: Infrastructure Setup (1 week)

1. **Create Lambda function for raster processing**
   - Node.js runtime (18.x or later)
   - Include Pako and raquet_lib.js in deployment package
   - Configure IAM roles for Redshift access

2. **Set up Lambda UDF in Redshift**
   ```sql
   CREATE EXTERNAL FUNCTION raquet_decode_band(
       band VARCHAR(MAX),
       metadata VARCHAR(MAX),
       band_index INTEGER
   )
   RETURNS VARCHAR(MAX)
   STABLE
   LAMBDA 'raquet-processor'
   IAM_ROLE 'arn:aws:iam::account:role/redshift-lambda';
   ```

3. **Install CARTO Analytics Toolbox**
   - Follow CARTO installation guide for Redshift
   - Verify QUADBIN functions are available

### Phase 2: Lambda Function Development (1-2 weeks)

1. **Create Node.js Lambda package**
   ```javascript
   // index.js
   const pako = require('pako');
   const raquetLib = require('./raquet_lib');

   exports.handler = async (event) => {
       const results = event.arguments.map(([band, metadata, bandIndex]) => {
           return raquetLib.decodeBand(band, metadata, bandIndex);
       });
       return { results };
   };
   ```

2. **Package and deploy**
   - Bundle with npm dependencies
   - Deploy to AWS Lambda
   - Test with sample data

### Phase 3: SQL UDF Migration (2 weeks)

1. **Migrate scalar functions**
   - Create Lambda-backed UDFs for JavaScript logic
   - Create SQL UDFs for pure SQL operations

2. **Handle table functions**
   - Redshift doesn't support table functions
   - Use CTEs with recursive queries or materialized views
   - Example for pixel positions:
   ```sql
   WITH RECURSIVE pixel_positions AS (
       SELECT 0 AS x, 0 AS y
       UNION ALL
       SELECT
           CASE WHEN y = 255 THEN x + 1 ELSE x END,
           CASE WHEN y = 255 THEN 0 ELSE y + 1 END
       FROM pixel_positions
       WHERE x < 256
   )
   SELECT x, y FROM pixel_positions WHERE x < 256;
   ```

### Phase 4: Integration & Testing (1-2 weeks)

1. **End-to-end testing**
   - Test with sample Raquet datasets
   - Performance benchmarking (Lambda latency considerations)

2. **Optimize Lambda cold starts**
   - Use provisioned concurrency for critical functions
   - Consider Lambda SnapStart for Java alternative

3. **Documentation**
   - Redshift-specific README
   - Lambda deployment guide
   - IAM role configuration

---

## 5. Code Examples

### Example: Lambda UDF for RAQUET_DECODE_BAND

**Lambda Function (Node.js)**:
```javascript
const pako = require('pako');

// Inlined raquet_lib functions
function base64ToUint8Array(base64) {
    const binary = Buffer.from(base64, 'base64');
    return new Uint8Array(binary);
}

function decodeBand(bandBase64, metadataJson, bandIndex) {
    const metadata = JSON.parse(metadataJson);
    const bandMeta = metadata.bands[bandIndex];

    let data = base64ToUint8Array(bandBase64);

    if (metadata.compression === 'gzip') {
        data = pako.inflate(data);
    }

    // Decode based on data type
    const view = new DataView(data.buffer);
    const pixels = [];
    const bytesPerPixel = getTypeSize(bandMeta.type);

    for (let i = 0; i < 256 * 256; i++) {
        pixels.push(readTypedValue(view, i * bytesPerPixel, bandMeta.type));
    }

    return JSON.stringify(pixels);
}

exports.handler = async (event) => {
    const results = event.arguments.map(args => {
        try {
            return decodeBand(args[0], args[1], args[2]);
        } catch (e) {
            return null;
        }
    });
    return { results };
};
```

**Redshift UDF**:
```sql
CREATE OR REPLACE EXTERNAL FUNCTION raquet.raquet_decode_band(
    band VARCHAR(MAX),
    metadata VARCHAR(MAX),
    band_index INTEGER
)
RETURNS VARCHAR(MAX)
STABLE
LAMBDA 'raquet-decode-band'
IAM_ROLE 'arn:aws:iam::123456789:role/redshift-lambda-role';
```

### Example: Replacing Table Function with CTE

**BigQuery (original)**:
```sql
CREATE TABLE FUNCTION __RAQUET_PIXEL_POSITIONS()
RETURNS TABLE<x INT64, y INT64>
AS (SELECT x, y FROM UNNEST(GENERATE_ARRAY(0, 255)) AS x
    CROSS JOIN UNNEST(GENERATE_ARRAY(0, 255)) AS y);
```

**Redshift (adapted)**:
```sql
-- Create a materialized view for performance
CREATE MATERIALIZED VIEW raquet.pixel_positions AS
WITH numbers AS (
    SELECT ROW_NUMBER() OVER () - 1 AS n
    FROM stl_connection_log  -- Any table with 256+ rows
    LIMIT 256
)
SELECT a.n AS x, b.n AS y
FROM numbers a
CROSS JOIN numbers b;

-- Usage
SELECT x, y FROM raquet.pixel_positions;
```

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Lambda cold start latency | High | Medium | Provisioned concurrency |
| Lambda payload limits (5MB) | Low | High | Chunk large tiles |
| No table function support | Certain | High | Use CTEs/materialized views |
| Python UDF deprecation | Certain | High | Use Lambda UDFs only |
| Network latency to Lambda | Medium | Medium | Co-locate in same region |
| IAM configuration complexity | Medium | Medium | Document thoroughly |
| CARTO Toolbox version gaps | Low | Medium | Track release notes |

---

## 7. Cost Considerations

| Component | Cost Factor | Notes |
|-----------|-------------|-------|
| Lambda invocations | $0.20 per 1M requests | Per-row function calls add up |
| Lambda compute | $0.0000166667/GB-second | Memory × duration |
| Data transfer | Varies | Redshift ↔ Lambda in same VPC |
| Provisioned concurrency | ~$0.000004/GB-second | Optional for cold starts |

**Recommendation**: Batch operations where possible to minimize Lambda invocations.

---

## 8. Alternative Approaches

### Option A: Lambda UDFs with Node.js (Recommended)
- Full JavaScript support via Node.js
- Pako and all dependencies supported
- Moderate complexity; production-ready path

### Option B: SQL-Only Implementation (Limited)
- Implement decoding in pure SQL
- Very limited; no gzip decompression possible
- Not recommended

### Option C: External Processing Pipeline
- Process raster data outside Redshift
- Store results in Redshift tables
- Loses SQL query flexibility

---

## 9. Conclusion

Porting BigQuery Raquet to Amazon Redshift is **moderately feasible** with the following considerations:

**Pros**:
1. CARTO Analytics Toolbox provides full QUADBIN support
2. Lambda UDFs enable JavaScript (Node.js) execution
3. Good geospatial function coverage
4. Parquet/Raquet format natively supported

**Cons**:
1. No native JavaScript UDFs (requires Lambda infrastructure)
2. Python UDFs being deprecated (not a viable alternative)
3. No table function support (requires architectural workarounds)
4. Lambda adds latency and cost complexity
5. More infrastructure to manage (IAM, Lambda deployment)

**Recommended next steps**:
1. Set up proof-of-concept Lambda function with Node.js
2. Test CARTO QUADBIN functions in target Redshift cluster
3. Benchmark Lambda latency for raster operations
4. Design CTE-based alternatives for table functions

---

## References

- [Amazon Redshift Python UDF Deprecation Announcement](https://aws.amazon.com/blogs/big-data/amazon-redshift-python-user-defined-functions-will-reach-end-of-support-after-june-30-2026/)
- [Amazon Redshift Lambda UDFs](https://docs.aws.amazon.com/redshift/latest/dg/udf-creating-a-lambda-sql-udf.html)
- [Amazon Redshift Spatial Functions](https://docs.aws.amazon.com/redshift/latest/dg/geospatial-functions.html)
- [Amazon Redshift Querying Spatial Data](https://docs.aws.amazon.com/redshift/latest/dg/geospatial-overview.html)
- [CARTO Analytics Toolbox for Redshift - QUADBIN](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-redshift/sql-reference/quadbin)
- [CARTO Spatial Indexes for Redshift](https://docs.carto.com/data-and-analysis/analytics-toolbox-for-redshift/key-concepts/spatial-indexes)
