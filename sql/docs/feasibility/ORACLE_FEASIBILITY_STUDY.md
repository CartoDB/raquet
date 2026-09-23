# Oracle Database Feasibility Study for Raquet

## Executive Summary

Porting BigQuery Raquet to Oracle Database is **feasible but requires significant effort**. Oracle 23ai/26ai introduces the Multilingual Engine (MLE) which provides full JavaScript support via GraalVM. However, unlike BigQuery, Snowflake, Redshift, and Databricks, **CARTO Analytics Toolbox is NOT available for Oracle**, meaning QUADBIN spatial indexing functions must be implemented from scratch. Oracle has excellent native geospatial capabilities via Oracle Spatial (SDO_GEOMETRY).

**Overall Feasibility: MODERATE-LOW** - Estimated effort: 6-10 weeks for a production-ready port.

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

### Proposed Oracle Architecture

```
User SQL Queries
        ↓
PL/SQL + MLE JavaScript Layer (Oracle Functions)
        ↓
MLE JavaScript Modules (GraalVM)
        ↓
Raquet Data Layer (Oracle Tables with BLOB/RAW)
        ↓
Custom QUADBIN Implementation + Oracle Spatial (SDO_GEOMETRY)
```

---

## 2. Component-by-Component Analysis

### 2.1 JavaScript Support via MLE

Oracle 23ai/26ai provides JavaScript support through the Multilingual Engine (MLE):

| Feature | Support | Notes |
|---------|---------|-------|
| JavaScript execution | **Yes** | Via GraalVM |
| MLE Modules | **Yes** | Persistent, reusable code |
| Call Specifications | **Yes** | PL/SQL wrappers for JS |
| External npm packages | **Limited** | Must bundle manually |
| TypedArrays/DataView | **Yes** | GraalVM V8 compatible |
| Base64 | **Yes** | Standard JS APIs |

**Key Features of Oracle MLE**:
- Based on GraalVM polyglot runtime
- Persistent MLE modules stored in database
- Call specs allow SQL/PL/SQL to invoke JavaScript
- Version 23.9+ removed EXECUTE ON JAVASCRIPT privilege requirement

### 2.2 Creating MLE Modules

```sql
-- Grant required privileges
GRANT CREATE MLE TO raquet_user;
GRANT CREATE PROCEDURE TO raquet_user;

-- Create MLE module with JavaScript code
CREATE OR REPLACE MLE MODULE raquet_lib
LANGUAGE JAVASCRIPT AS
$$
    export function decodePixelValue(data, offset, dataType) {
        const view = new DataView(data.buffer);
        // ... decoding logic
    }

    export function decodeBand(bandBase64, compression, dataType) {
        // ... band decoding logic
    }
$$;

-- Create call specification
CREATE OR REPLACE FUNCTION raquet_decode_band(
    p_band BLOB,
    p_metadata CLOB,
    p_band_index NUMBER
) RETURN CLOB
AS MLE MODULE raquet_lib
SIGNATURE 'decodeBand(Uint8Array, string, number)';
```

### 2.3 MLE Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| No npm install | High | Bundle Pako inline |
| Module size limits | Medium | Split into multiple modules |
| Type conversion | Medium | Use Uint8Array for binary |
| Debugging | Medium | Limited compared to Node.js |
| Performance | Low | GraalVM is highly optimized |

### 2.4 Geospatial Support (Oracle Spatial)

Oracle Spatial provides excellent geospatial capabilities:

| BigQuery Function | Oracle Equivalent | Notes |
|-------------------|-------------------|-------|
| `ST_GEOGPOINT(lon, lat)` | `SDO_GEOMETRY(2001, 4326, SDO_POINT_TYPE(lon, lat, NULL), NULL, NULL)` | More verbose |
| `ST_BOUNDINGBOX(geom)` | `SDO_GEOM.SDO_MBR(geom)` | Returns SDO_GEOMETRY |
| `ST_X(point)` | `geom.SDO_POINT.X` | Object property |
| `ST_Y(point)` | `geom.SDO_POINT.Y` | Object property |
| `ST_CONTAINS(geom, point)` | `SDO_CONTAINS(geom, point, 'mask=INSIDE') = 'TRUE'` | Different syntax |
| `GEOGRAPHY` type | `SDO_GEOMETRY` with SRID 4326 | Unified type |

**SDO_GEOMETRY Constructor**:
```sql
-- Point at longitude -122.4, latitude 37.8
SDO_GEOMETRY(
    2001,                          -- SDO_GTYPE: 2D point
    4326,                          -- SRID: WGS84
    SDO_POINT_TYPE(-122.4, 37.8, NULL),  -- Point coordinates
    NULL,                          -- SDO_ELEM_INFO
    NULL                           -- SDO_ORDINATES
)
```

### 2.5 QUADBIN - CRITICAL GAP

**CARTO Analytics Toolbox is NOT available for Oracle.**

This means ALL QUADBIN functions must be implemented from scratch:

| Required Function | Implementation Effort | Notes |
|-------------------|----------------------|-------|
| `QUADBIN_FROMZXY` | Low | Bit manipulation |
| `QUADBIN_TOZXY` | Low | Bit manipulation |
| `QUADBIN_FROMLONGLAT` | Medium | Web Mercator projection |
| `QUADBIN_POLYFILL` | High | Recursive tile enumeration |
| `QUADBIN_BBOX` | Medium | Inverse projection |
| `QUADBIN_BOUNDARY` | Medium | Build SDO_GEOMETRY from bbox |
| `QUADBIN_KRING` | Medium | Neighbor calculation |
| `QUADBIN_RESOLUTION` | Low | Extract from index |
| `QUADBIN_TOPARENT` | Low | Bit shift |
| `QUADBIN_TOCHILDREN` | Low | Bit manipulation |

**Estimated effort for QUADBIN implementation: 2-3 weeks**

### 2.6 SQL Syntax Differences

| Feature | BigQuery | Oracle | Migration Effort |
|---------|----------|--------|------------------|
| Data types | `INT64`, `FLOAT64`, `STRING` | `NUMBER`, `BINARY_DOUBLE`, `VARCHAR2` | Simple |
| JSON extraction | `JSON_VALUE(col, '$.path')` | `JSON_VALUE(col, '$.path')` | **Identical** |
| Array generation | `GENERATE_ARRAY(1, 10)` | `SELECT LEVEL FROM DUAL CONNECT BY LEVEL <= 10` | Moderate |
| STRUCT type | `STRUCT<field TYPE>` | Object types or JSON | Moderate |
| UNNEST | `UNNEST(array)` | `TABLE(array)` with nested table | Moderate |
| Table functions | `CREATE TABLE FUNCTION` | Pipelined functions | Moderate |
| Binary data | `BYTES` | `BLOB` or `RAW` | Simple |
| External libraries | `OPTIONS(library=[...])` | MLE modules | Different |

---

## 3. Function Migration Matrix

### Tier 1: Core Data Extraction (Moderate Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `RAQUET_DECODE_BAND` | Moderate | MLE JavaScript module |
| `RAQUET_PIXEL` | Moderate | MLE JavaScript module |
| `ST_RASTERVALUE` | High | Requires custom QUADBIN |
| `ST_RASTERVALUE_GEOG` | Moderate | PL/SQL wrapper |

### Tier 2: Statistics (Moderate Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `ST_RASTERSUMMARYSTATS` | Moderate | MLE JavaScript |
| `RAQUET_AGGREGATE_STATS` | Moderate | PL/SQL aggregation |

### Tier 3: Band Operations (Moderate Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `ST_BANDMATH` | Moderate | MLE JavaScript |
| `ST_NORMALIZEDDIFFERENCE` | Moderate | MLE JavaScript |
| `ST_NORMALIZEDDIFFERENCESTATS` | Moderate | MLE JavaScript |

### Tier 4: Resolution Management (Moderate Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `__RAQUET_RESOLVE_ZOOM` | Moderate | PL/SQL function |
| `__RAQUET_AUTO_ZOOM` | Moderate | PL/SQL with hierarchical query |

### Tier 5: Spatial Operations (High Effort)

| Function | Migration Complexity | Notes |
|----------|---------------------|-------|
| `RAQUET_PIXEL_GEOGRAPHY` | Moderate | Uses SDO_GEOMETRY |
| `__RAQUET_PIXEL_POSITIONS` | Moderate | Pipelined function |
| `__RAQUET_REGION_BLOCKS` | **High** | Requires QUADBIN_POLYFILL |

---

## 4. Technical Implementation Plan

### Phase 1: QUADBIN Implementation (2-3 weeks)

This is the critical path - implement QUADBIN functions first.

1. **Core QUADBIN functions in MLE JavaScript**
   ```sql
   CREATE OR REPLACE MLE MODULE quadbin_lib
   LANGUAGE JAVASCRIPT AS
   $$
       // QUADBIN encoding based on Bing Maps Tile System
       const TILE_SIZE = 256;
       const MAX_ZOOM = 26;

       export function quadbinFromZXY(z, x, y) {
           // Interleave x and y bits with zoom level
           let quadbin = BigInt(0);
           quadbin |= BigInt(z) << BigInt(59);  // Zoom in upper bits
           quadbin |= BigInt(1) << BigInt(63);  // Mode bit

           for (let i = 0; i < z; i++) {
               quadbin |= BigInt((x >> i) & 1) << BigInt(2 * i);
               quadbin |= BigInt((y >> i) & 1) << BigInt(2 * i + 1);
           }
           return quadbin.toString();
       }

       export function quadbinToZXY(quadbinStr) {
           const quadbin = BigInt(quadbinStr);
           const z = Number((quadbin >> BigInt(59)) & BigInt(0x1F));
           let x = 0, y = 0;

           for (let i = 0; i < z; i++) {
               x |= Number((quadbin >> BigInt(2 * i)) & BigInt(1)) << i;
               y |= Number((quadbin >> BigInt(2 * i + 1)) & BigInt(1)) << i;
           }
           return { z, x, y };
       }

       export function quadbinFromLonLat(lon, lat, zoom) {
           // Web Mercator projection
           const x = Math.floor((lon + 180) / 360 * Math.pow(2, zoom));
           const latRad = lat * Math.PI / 180;
           const y = Math.floor((1 - Math.log(Math.tan(latRad) + 1/Math.cos(latRad)) / Math.PI) / 2 * Math.pow(2, zoom));
           return quadbinFromZXY(zoom, x, y);
       }

       export function quadbinBBox(quadbinStr) {
           const { z, x, y } = quadbinToZXY(quadbinStr);
           const n = Math.pow(2, z);

           const west = x / n * 360 - 180;
           const east = (x + 1) / n * 360 - 180;

           const north = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))) * 180 / Math.PI;
           const south = Math.atan(Math.sinh(Math.PI * (1 - 2 * (y + 1) / n))) * 180 / Math.PI;

           return [west, south, east, north];
       }
   $$;
   ```

2. **PL/SQL call specifications**
   ```sql
   CREATE OR REPLACE FUNCTION QUADBIN_FROMZXY(
       p_z NUMBER,
       p_x NUMBER,
       p_y NUMBER
   ) RETURN NUMBER
   AS MLE MODULE quadbin_lib
   SIGNATURE 'quadbinFromZXY(number, number, number)';

   CREATE OR REPLACE FUNCTION QUADBIN_FROMLONGLAT(
       p_lon NUMBER,
       p_lat NUMBER,
       p_zoom NUMBER
   ) RETURN NUMBER
   AS MLE MODULE quadbin_lib
   SIGNATURE 'quadbinFromLonLat(number, number, number)';
   ```

3. **Implement QUADBIN_POLYFILL** (most complex)
   ```sql
   CREATE OR REPLACE FUNCTION QUADBIN_POLYFILL(
       p_geometry SDO_GEOMETRY,
       p_resolution NUMBER
   ) RETURN SYS.ODCINUMBERLIST PIPELINED
   AS
       v_mbr SDO_GEOMETRY;
       v_min_x NUMBER;
       v_min_y NUMBER;
       v_max_x NUMBER;
       v_max_y NUMBER;
       v_quadbin NUMBER;
       v_bbox SYS.ODCINUMBERLIST;
       v_tile_geom SDO_GEOMETRY;
   BEGIN
       -- Get MBR
       v_mbr := SDO_GEOM.SDO_MBR(p_geometry);

       -- Extract bounds
       v_min_x := SDO_GEOM.SDO_MIN_MBR_ORDINATE(v_mbr, 1);
       v_min_y := SDO_GEOM.SDO_MIN_MBR_ORDINATE(v_mbr, 2);
       v_max_x := SDO_GEOM.SDO_MAX_MBR_ORDINATE(v_mbr, 1);
       v_max_y := SDO_GEOM.SDO_MAX_MBR_ORDINATE(v_mbr, 2);

       -- Iterate over potential tiles
       FOR tile_x IN (
           SELECT LEVEL - 1 AS x
           FROM DUAL
           CONNECT BY LEVEL <= POWER(2, p_resolution)
       ) LOOP
           FOR tile_y IN (
               SELECT LEVEL - 1 AS y
               FROM DUAL
               CONNECT BY LEVEL <= POWER(2, p_resolution)
           ) LOOP
               v_quadbin := QUADBIN_FROMZXY(p_resolution, tile_x.x, tile_y.y);
               v_tile_geom := QUADBIN_BOUNDARY(v_quadbin);

               -- Check intersection
               IF SDO_GEOM.RELATE(p_geometry, 'ANYINTERACT', v_tile_geom, 0.005) = 'TRUE' THEN
                   PIPE ROW(v_quadbin);
               END IF;
           END LOOP;
       END LOOP;

       RETURN;
   END;
   ```

### Phase 2: Pako Library Integration (1 week)

1. **Bundle Pako into MLE module**
   - Minify Pako (~30KB)
   - Include in MLE module definition
   - Test gzip decompression

   ```sql
   CREATE OR REPLACE MLE MODULE raquet_inflate
   LANGUAGE JAVASCRIPT AS
   $$
       // Minified Pako library here (~30KB)
       const pako = (function() {
           // ... pako source ...
       })();

       export function inflate(data) {
           return pako.inflate(data);
       }
   $$;
   ```

### Phase 3: Core Raquet Functions (2-3 weeks)

1. **Create main raquet_lib MLE module**
   ```sql
   CREATE OR REPLACE MLE MODULE raquet_lib
   LANGUAGE JAVASCRIPT AS
   $$
       // Import from other modules
       import { inflate } from 'raquet_inflate';

       const TYPE_READERS = {
           'uint8': (view, offset) => view.getUint8(offset),
           'int8': (view, offset) => view.getInt8(offset),
           'uint16': (view, offset) => view.getUint16(offset, true),
           'int16': (view, offset) => view.getInt16(offset, true),
           'uint32': (view, offset) => view.getUint32(offset, true),
           'int32': (view, offset) => view.getInt32(offset, true),
           'float32': (view, offset) => view.getFloat32(offset, true),
           'float64': (view, offset) => view.getFloat64(offset, true),
       };

       export function decodeBand(bandData, metadataJson, bandIndex) {
           const metadata = JSON.parse(metadataJson);
           const bandMeta = metadata.bands[bandIndex];

           let data = new Uint8Array(bandData);

           // Decompress if needed
           if (metadata.compression === 'gzip') {
               data = inflate(data);
           }

           // Decode pixels
           const view = new DataView(data.buffer);
           const reader = TYPE_READERS[bandMeta.type];
           const bytesPerPixel = getTypeSize(bandMeta.type);
           const pixels = [];

           for (let i = 0; i < 256 * 256; i++) {
               pixels.push(reader(view, i * bytesPerPixel));
           }

           return JSON.stringify(pixels);
       }

       export function getPixelValue(bandData, metadataJson, bandIndex, x, y) {
           // ... pixel extraction logic
       }
   $$;
   ```

2. **Create PL/SQL wrappers**
   ```sql
   CREATE OR REPLACE FUNCTION RAQUET_DECODE_BAND(
       p_band BLOB,
       p_metadata CLOB,
       p_band_index NUMBER
   ) RETURN CLOB
   AS MLE MODULE raquet_lib
   SIGNATURE 'decodeBand(Uint8Array, string, number)';

   CREATE OR REPLACE FUNCTION RAQUET_PIXEL(
       p_band BLOB,
       p_metadata CLOB,
       p_band_index NUMBER,
       p_x NUMBER,
       p_y NUMBER
   ) RETURN NUMBER
   AS MLE MODULE raquet_lib
   SIGNATURE 'getPixelValue(Uint8Array, string, number, number, number)';
   ```

### Phase 4: Spatial Integration (1-2 weeks)

1. **ST_RASTERVALUE with Oracle Spatial**
   ```sql
   CREATE OR REPLACE FUNCTION ST_RASTERVALUE(
       p_block NUMBER,
       p_band BLOB,
       p_lon NUMBER,
       p_lat NUMBER,
       p_metadata CLOB,
       p_band_index NUMBER
   ) RETURN NUMBER
   AS
       v_zxy JSON_OBJECT_T;
       v_bbox JSON_ARRAY_T;
       v_tile_width NUMBER;
       v_tile_height NUMBER;
       v_x NUMBER;
       v_y NUMBER;
       v_pixel_value NUMBER;
   BEGIN
       -- Get tile coordinates from QUADBIN
       v_zxy := JSON_OBJECT_T(QUADBIN_TOZXY_JSON(p_block));

       -- Get tile bounds
       v_bbox := JSON_ARRAY_T(QUADBIN_BBOX_JSON(p_block));

       -- Calculate pixel position
       v_tile_width := v_bbox.get_Number(2) - v_bbox.get_Number(0);
       v_tile_height := v_bbox.get_Number(3) - v_bbox.get_Number(1);

       v_x := FLOOR((p_lon - v_bbox.get_Number(0)) / v_tile_width * 256);
       v_y := FLOOR((v_bbox.get_Number(3) - p_lat) / v_tile_height * 256);

       -- Clamp to valid range
       v_x := GREATEST(0, LEAST(255, v_x));
       v_y := GREATEST(0, LEAST(255, v_y));

       -- Get pixel value
       v_pixel_value := RAQUET_PIXEL(p_band, p_metadata, p_band_index, v_x, v_y);

       RETURN v_pixel_value;
   END;
   /
   ```

2. **RAQUET_PIXEL_GEOGRAPHY**
   ```sql
   CREATE OR REPLACE FUNCTION RAQUET_PIXEL_GEOGRAPHY(
       p_block NUMBER,
       p_x NUMBER,
       p_y NUMBER
   ) RETURN SDO_GEOMETRY
   AS
       v_bbox JSON_ARRAY_T;
       v_west NUMBER;
       v_south NUMBER;
       v_east NUMBER;
       v_north NUMBER;
       v_lon NUMBER;
       v_lat NUMBER;
   BEGIN
       v_bbox := JSON_ARRAY_T(QUADBIN_BBOX_JSON(p_block));

       v_west := v_bbox.get_Number(0);
       v_south := v_bbox.get_Number(1);
       v_east := v_bbox.get_Number(2);
       v_north := v_bbox.get_Number(3);

       -- Calculate pixel center
       v_lon := v_west + (p_x + 0.5) * (v_east - v_west) / 256;
       v_lat := v_north - (p_y + 0.5) * (v_north - v_south) / 256;

       RETURN SDO_GEOMETRY(2001, 4326, SDO_POINT_TYPE(v_lon, v_lat, NULL), NULL, NULL);
   END;
   /
   ```

### Phase 5: Testing & Documentation (1 week)

1. **Create test suite**
2. **Performance benchmarking**
3. **Documentation**

---

## 5. Code Examples

### Example: Complete Query Flow

```sql
-- Query raster value at a geographic point
SELECT
    r.block,
    ST_RASTERVALUE(
        r.block,
        r.band_1,
        -122.4,  -- longitude
        37.8,    -- latitude
        m.metadata,
        0        -- band index
    ) AS elevation
FROM raquet_dem r
CROSS JOIN (
    SELECT metadata
    FROM raquet_dem
    WHERE block = 0
) m
WHERE r.block = QUADBIN_FROMLONGLAT(-122.4, 37.8, 14);

-- Get all pixels in a region
SELECT
    r.block,
    pp.x,
    pp.y,
    RAQUET_PIXEL(r.band_1, m.metadata, 0, pp.x, pp.y) AS value,
    RAQUET_PIXEL_GEOGRAPHY(r.block, pp.x, pp.y) AS location
FROM raquet_dem r
CROSS JOIN (SELECT metadata FROM raquet_dem WHERE block = 0) m
CROSS JOIN TABLE(RAQUET_PIXEL_POSITIONS()) pp
WHERE r.block IN (
    SELECT COLUMN_VALUE
    FROM TABLE(QUADBIN_POLYFILL(
        SDO_GEOMETRY(2003, 4326, NULL,
            SDO_ELEM_INFO_ARRAY(1, 1003, 1),
            SDO_ORDINATE_ARRAY(-122.5, 37.7, -122.3, 37.7, -122.3, 37.9, -122.5, 37.9, -122.5, 37.7)
        ),
        14
    ))
);
```

### Example: Pipelined Table Function

```sql
-- Equivalent to __RAQUET_PIXEL_POSITIONS
CREATE OR REPLACE FUNCTION RAQUET_PIXEL_POSITIONS
RETURN SYS.ODCINUMBERLIST PIPELINED
AS
    TYPE t_pixel_rec IS RECORD (x NUMBER, y NUMBER);
    v_rec t_pixel_rec;
BEGIN
    FOR v_x IN 0..255 LOOP
        FOR v_y IN 0..255 LOOP
            PIPE ROW(v_x * 256 + v_y);  -- Encode x,y in single number
        END LOOP;
    END LOOP;
    RETURN;
END;
/

-- Or with object type for proper x,y columns
CREATE OR REPLACE TYPE pixel_position_t AS OBJECT (
    x NUMBER,
    y NUMBER
);
/

CREATE OR REPLACE TYPE pixel_position_table_t AS TABLE OF pixel_position_t;
/

CREATE OR REPLACE FUNCTION RAQUET_PIXEL_POSITIONS
RETURN pixel_position_table_t PIPELINED
AS
BEGIN
    FOR v_x IN 0..255 LOOP
        FOR v_y IN 0..255 LOOP
            PIPE ROW(pixel_position_t(v_x, v_y));
        END LOOP;
    END LOOP;
    RETURN;
END;
/
```

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **No CARTO Toolbox** | **Certain** | **High** | Implement QUADBIN from scratch |
| MLE module complexity | Medium | Medium | Thorough testing |
| Pako integration issues | Low | High | Test early; consider PL/SQL gzip |
| SDO_GEOMETRY learning curve | Medium | Medium | Document patterns |
| Performance of custom QUADBIN | Medium | High | Optimize with spatial indexes |
| Oracle version requirements | Low | High | Require 23ai minimum |
| MLE debugging limitations | Medium | Medium | Extensive logging |

---

## 7. Prerequisites

### Oracle Version Requirements

| Feature | Minimum Version |
|---------|-----------------|
| MLE JavaScript | Oracle 21c |
| Persistent MLE modules | Oracle 23ai |
| Call specifications | Oracle 23ai |
| Improved MLE syntax | Oracle 23.9 |

**Recommended**: Oracle 23ai (23.9+) or Oracle 26ai

### Required Privileges

```sql
-- For the Raquet schema owner
GRANT CREATE MLE TO raquet_admin;
GRANT CREATE PROCEDURE TO raquet_admin;
GRANT CREATE TYPE TO raquet_admin;
GRANT CREATE TABLE TO raquet_admin;

-- For Raquet users
GRANT EXECUTE ON raquet_admin.RAQUET_DECODE_BAND TO raquet_user;
GRANT EXECUTE ON raquet_admin.ST_RASTERVALUE TO raquet_user;
-- etc.
```

---

## 8. Alternative Approaches

### Option A: MLE JavaScript (Recommended)
- Closest to BigQuery implementation
- GraalVM provides good performance
- TypedArray support for binary data

### Option B: Pure PL/SQL
- No MLE dependency
- Much more verbose code
- Difficult gzip decompression (would need UTL_COMPRESS)

### Option C: Java Stored Procedures
- Full Java ecosystem
- Better library support
- Higher complexity

### Option D: External Processing
- Process outside Oracle
- Lose SQL integration benefits
- Simpler implementation

---

## 9. Comparison with Other Platforms

| Aspect | BigQuery | Snowflake | Redshift | Databricks | **Oracle** |
|--------|----------|-----------|----------|------------|------------|
| CARTO Toolbox | Yes | Yes | Yes | Yes | **No** |
| JavaScript UDFs | Yes | Yes | Lambda | No | **MLE** |
| Native Spatial | Yes | Yes | Yes | Yes | **Yes (SDO)** |
| Implementation Effort | Baseline | Low | Moderate | Low | **High** |
| QUADBIN Available | Yes | Yes | Yes | Yes | **Must Build** |

---

## 10. Conclusion

Porting BigQuery Raquet to Oracle Database is **feasible but significantly more complex** than other platforms:

**Pros**:
1. MLE provides full JavaScript support (Oracle 23ai+)
2. Excellent spatial capabilities via Oracle Spatial
3. Mature, enterprise-grade database
4. JSON support comparable to other platforms
5. Pipelined functions for table-valued results

**Cons**:
1. **No CARTO Analytics Toolbox** - must implement QUADBIN from scratch
2. More verbose SDO_GEOMETRY syntax
3. MLE requires Oracle 23ai or later
4. Steeper learning curve for MLE modules
5. Less community support for spatial analytics use cases

**Estimated additional effort vs other platforms**: +3-4 weeks (primarily for QUADBIN implementation)

**Recommended next steps**:
1. Verify Oracle 23ai+ availability in target environment
2. Prototype QUADBIN_FROMZXY and QUADBIN_TOZXY first
3. Test Pako integration in MLE module
4. Consider whether full port is justified vs. other platforms

---

## References

- [Oracle MLE JavaScript Introduction](https://docs.oracle.com/en/database/oracle/oracle-database/26/mlejs/introduction-to-mle.html)
- [Oracle MLE JavaScript in 23c](https://blogs.oracle.com/developers/post/introduction-javascript-oracle-database-23c-free-developer-release)
- [Oracle MLE on GraalVM](https://www.graalvm.org/js/mle-oracle-db/)
- [Oracle Spatial SDO_GEOMETRY Methods](https://docs.oracle.com/en/database/oracle/oracle-database/26/spatl/sdo_geometry-methods.html)
- [Oracle Spatial Data Types](https://docs.oracle.com/en/database/oracle/oracle-database/19/spatl/spatial-datatypes-metadata.html)
- [MLE Module Best Practices](https://oracle-base.com/articles/23/multilingual-engine-for-javascript-23)
- [Bing Maps Tile System (QUADBIN basis)](https://docs.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system)
