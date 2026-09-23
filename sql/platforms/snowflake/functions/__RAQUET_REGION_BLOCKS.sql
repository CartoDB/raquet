-- __RAQUET_REGION_BLOCKS (Snowflake version)
-- Returns all block quadbins from a Raquet file that intersect a region
--
-- This handles multi-resolution Raquet files by polyfilling at each zoom level
-- from min_zoom to max_zoom and unioning the results.
--
-- Parameters:
--   GEOM: GEOGRAPHY - The region of interest
--   MIN_ZOOM: DOUBLE - Minimum zoom level in the Raquet file
--   MAX_ZOOM: DOUBLE - Maximum zoom level in the Raquet file
--   MODE: VARCHAR (optional) - 'center' (default) or 'intersects'
--     'center': returns tiles whose center falls within the geometry (faster, may miss edge tiles)
--     'intersects': returns all tiles that intersect the geometry (matches DuckDB behavior)
--     'contains': returns only tiles fully contained within the geometry
--
-- Returns: ARRAY - Array of QUADBIN block identifiers
--
-- Usage:
--   -- Default (center containment):
--   SELECT f.VALUE::NUMBER AS block
--   FROM TABLE(FLATTEN(
--       RAQUET_DB.RAQUET.__RAQUET_REGION_BLOCKS(
--           ST_GEOGRAPHYFROMWKT('POLYGON((...))'),
--           17, 17
--       )
--   )) f;
--
--   -- Intersects mode (matches DuckDB read_raquet behavior):
--   SELECT f.VALUE::NUMBER AS block
--   FROM TABLE(FLATTEN(
--       RAQUET_DB.RAQUET.__RAQUET_REGION_BLOCKS(
--           ST_GEOGRAPHYFROMWKT('POLYGON((...))'),
--           17, 17, 'intersects'
--       )
--   )) f;

-- JavaScript UDF: full polyfill with mode support
-- MUST be deployed before the 4-parameter __RAQUET_REGION_BLOCKS wrapper below.
-- Returns ARRAY of VARCHAR (string-encoded quadbin IDs) to avoid JS number precision loss.
-- Quadbin values exceed JS Number.MAX_SAFE_INTEGER (2^53-1), so we use string arithmetic.
--
-- Mirrors the DuckDB duckdb-raquet QUADBIN_POLYFILL algorithm:
--   1. Parse GeoJSON polygon
--   2. Extract bbox, convert to tile coordinates
--   3. Iterate all tiles in bbox range
--   4. For each tile, apply mode-specific test (ray-casting point-in-polygon)
--   5. Convert matching tiles to QUADBIN cell ID strings
CREATE OR REPLACE FUNCTION RAQUET_DB.RAQUET.RAQUET_POLYFILL(
    GEOG_JSON VARCHAR,
    MIN_ZOOM DOUBLE,
    MAX_ZOOM DOUBLE,
    MODE VARCHAR
)
RETURNS ARRAY
LANGUAGE JAVASCRIPT
AS
$$
var MAX_LAT = 85.051128779806604;
var PI = Math.PI;

// --- 64-bit unsigned integer as [hi, lo] pair (each 32 bits) ---
function u64(hi, lo) { return [hi >>> 0, lo >>> 0]; }
function u64or(a, b) { return u64(a[0] | b[0], a[1] | b[1]); }
function u64shl(v, n) {
    if (n === 0) return v;
    if (n >= 32) return u64(v[1] << (n - 32), 0);
    return u64((v[0] << n) | (v[1] >>> (32 - n)), v[1] << n);
}
function u64shr(v, n) {
    if (n === 0) return v;
    if (n >= 32) return u64(0, v[0] >>> (n - 32));
    return u64(v[0] >>> n, (v[1] >>> n) | (v[0] << (32 - n)));
}
function u64and(a, b) { return u64(a[0] & b[0], a[1] & b[1]); }

// Convert u64 to decimal string
function u64str(v) {
    // Split into high and low, convert via base-10 arithmetic
    var hi = v[0] >>> 0;
    var lo = v[1] >>> 0;
    if (hi === 0) return lo.toString();
    // hi * 2^32 + lo
    // Use repeated division by 10
    var digits = [];
    while (hi > 0 || lo > 0) {
        // Divide [hi, lo] by 10
        var r = hi % 10;
        hi = Math.floor(hi / 10);
        // (r * 2^32 + lo) / 10
        var combined = r * 4294967296 + lo;
        lo = Math.floor(combined / 10);
        var remainder = combined % 10;
        digits.push(remainder);
    }
    if (digits.length === 0) return "0";
    return digits.reverse().join('');
}

// Morton interleave: spread bits of a 32-bit value into even bit positions of 64-bit value
var _B0 = u64(0x55555555, 0x55555555);
var _B1 = u64(0x33333333, 0x33333333);
var _B2 = u64(0x0F0F0F0F, 0x0F0F0F0F);
var _B3 = u64(0x00FF00FF, 0x00FF00FF);
var _B4 = u64(0x0000FFFF, 0x0000FFFF);

function spread(val32) {
    var v = u64(0, val32 >>> 0);
    v = u64and(u64or(v, u64shl(v, 16)), _B4);
    v = u64and(u64or(v, u64shl(v, 8)), _B3);
    v = u64and(u64or(v, u64shl(v, 4)), _B2);
    v = u64and(u64or(v, u64shl(v, 2)), _B1);
    v = u64and(u64or(v, u64shl(v, 1)), _B0);
    return v;
}

// HEADER | MODE = 0x48000000_00000000
var HEADER_MODE = u64(0x48000000, 0x00000000);
var FOOTER_MASK = u64(0x000FFFFF, 0xFFFFFFFF);

function tileToCell(x, y, z) {
    var sx = (x << (32 - z)) >>> 0;
    var sy = (y << (32 - z)) >>> 0;
    var ux = spread(sx);
    var uy = spread(sy);
    // interleaved = ux | (uy << 1)
    var interleaved = u64or(ux, u64shl(uy, 1));
    // shifted = interleaved >> 12
    var shifted = u64shr(interleaved, 12);
    // resolution part: z << 52
    var resPart = u64shl(u64(0, z), 52);
    // footer: FOOTER_MASK >> (z * 2)
    var footer = u64shr(FOOTER_MASK, z * 2);
    // cell = HEADER_MODE | resPart | shifted | footer
    // But shifted and footer overlap in the quadkey region, so we need:
    // cell = HEADER_MODE | resPart | (shifted & ~footer) | footer
    // Actually the standard formula is: HEADER | MODE | (z << 52) | ((interleaved >> 12) & ~footer_region) | footer
    // Simpler: the shifted value already has the quadkey bits in the right position,
    // and footer fills the remaining lower bits with 1s
    // cell = HEADER_MODE | resPart | shifted | footer works because
    // shifted has 0s in the footer region (the bits below the quadkey)
    return u64or(u64or(u64or(HEADER_MODE, resPart), shifted), footer);
}

function lonlatToTile(lon, lat, z) {
    if (lat > MAX_LAT) lat = MAX_LAT;
    if (lat < -MAX_LAT) lat = -MAX_LAT;
    var n = Math.pow(2, z);
    var x = Math.floor((lon + 180.0) / 360.0 * n);
    var latRad = lat * PI / 180.0;
    var y = Math.floor((1.0 - Math.log(Math.tan(latRad) + 1.0 / Math.cos(latRad)) / PI) / 2.0 * n);
    if (x < 0) x = 0;
    if (x >= n) x = n - 1;
    if (y < 0) y = 0;
    if (y >= n) y = n - 1;
    return [x, y];
}

function tileToBboxWgs84(x, y, z) {
    var n = Math.pow(2, z);
    var minLon = x / n * 360.0 - 180.0;
    var maxLon = (x + 1) / n * 360.0 - 180.0;
    var minLatRad = Math.atan(Math.sinh(PI * (1.0 - 2.0 * (y + 1) / n)));
    var maxLatRad = Math.atan(Math.sinh(PI * (1.0 - 2.0 * y / n)));
    return [minLon, minLatRad * 180.0 / PI, maxLon, maxLatRad * 180.0 / PI];
}

// Ray-casting point-in-polygon
function pointInRing(px, py, ring) {
    var inside = false;
    for (var i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        var xi = ring[i][0], yi = ring[i][1];
        var xj = ring[j][0], yj = ring[j][1];
        if (((yi > py) !== (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

function pointInPolygon(px, py, rings) {
    if (!rings || rings.length === 0) return false;
    if (!pointInRing(px, py, rings[0])) return false;
    for (var h = 1; h < rings.length; h++) {
        if (pointInRing(px, py, rings[h])) return false;
    }
    return true;
}

function pointInGeometry(px, py, polygons) {
    for (var i = 0; i < polygons.length; i++) {
        if (pointInPolygon(px, py, polygons[i])) return true;
    }
    return false;
}

function parseGeoJSON(geojson) {
    var geo = typeof geojson === 'string' ? JSON.parse(geojson) : geojson;
    var polygons = [];
    if (geo.type === 'Polygon') {
        polygons.push(geo.coordinates);
    } else if (geo.type === 'MultiPolygon') {
        for (var i = 0; i < geo.coordinates.length; i++) {
            polygons.push(geo.coordinates[i]);
        }
    } else if (geo.type === 'GeometryCollection') {
        for (var i = 0; i < geo.geometries.length; i++) {
            polygons = polygons.concat(parseGeoJSON(geo.geometries[i]));
        }
        return polygons;
    }
    return polygons;
}

function extractBbox(polygons) {
    var minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
    for (var p = 0; p < polygons.length; p++) {
        var ring = polygons[p][0];
        for (var i = 0; i < ring.length; i++) {
            if (ring[i][0] < minLon) minLon = ring[i][0];
            if (ring[i][0] > maxLon) maxLon = ring[i][0];
            if (ring[i][1] < minLat) minLat = ring[i][1];
            if (ring[i][1] > maxLat) maxLat = ring[i][1];
        }
    }
    return [minLon, minLat, maxLon, maxLat];
}

// Main
var mode = (MODE || 'center').toLowerCase();
var polygons = parseGeoJSON(GEOG_JSON);
if (polygons.length === 0) return [];

var bbox = extractBbox(polygons);
var results = [];
var seen = {};

for (var z = Math.floor(MIN_ZOOM); z <= Math.floor(MAX_ZOOM); z++) {
    var nw = lonlatToTile(bbox[0], bbox[3], z);
    var se = lonlatToTile(bbox[2], bbox[1], z);

    for (var ty = nw[1]; ty <= se[1]; ty++) {
        for (var tx = nw[0]; tx <= se[0]; tx++) {
            var tb = tileToBboxWgs84(tx, ty, z);
            var include = false;

            if (mode === 'center') {
                var cx = (tb[0] + tb[2]) / 2.0;
                var cy = (tb[1] + tb[3]) / 2.0;
                include = pointInGeometry(cx, cy, polygons);
            } else if (mode === 'intersects') {
                include = pointInGeometry(tb[0], tb[1], polygons) ||
                          pointInGeometry(tb[2], tb[1], polygons) ||
                          pointInGeometry(tb[0], tb[3], polygons) ||
                          pointInGeometry(tb[2], tb[3], polygons);
                if (!include) {
                    var cx = (tb[0] + tb[2]) / 2.0;
                    var cy = (tb[1] + tb[3]) / 2.0;
                    include = pointInGeometry(cx, cy, polygons);
                }
                if (!include) {
                    for (var p = 0; p < polygons.length && !include; p++) {
                        var ring = polygons[p][0];
                        for (var i = 0; i < ring.length && !include; i++) {
                            if (ring[i][0] >= tb[0] && ring[i][0] <= tb[2] &&
                                ring[i][1] >= tb[1] && ring[i][1] <= tb[3]) {
                                include = true;
                            }
                        }
                    }
                }
            } else if (mode === 'contains') {
                include = pointInGeometry(tb[0], tb[1], polygons) &&
                          pointInGeometry(tb[2], tb[1], polygons) &&
                          pointInGeometry(tb[0], tb[3], polygons) &&
                          pointInGeometry(tb[2], tb[3], polygons);
            }

            if (include) {
                var cell = tileToCell(tx, ty, z);
                var key = u64str(cell);
                if (!seen[key]) {
                    seen[key] = true;
                    results.push(key);
                }
            }
        }
    }
}

return results;
$$;

-- 3-parameter version: center containment (backward compatible, uses CARTO QUADBIN_POLYFILL)
CREATE OR REPLACE FUNCTION RAQUET_DB.RAQUET.__RAQUET_REGION_BLOCKS(
    GEOM GEOGRAPHY,
    MIN_ZOOM DOUBLE,
    MAX_ZOOM DOUBLE
)
RETURNS ARRAY
AS
$$
    SELECT ARRAY_AGG(DISTINCT f2.VALUE::NUMBER)
    FROM TABLE(FLATTEN(ARRAY_GENERATE_RANGE(MIN_ZOOM::INT, MAX_ZOOM::INT + 1))) f1,
         TABLE(FLATTEN(CARTO_DEV_DATA.CARTO.QUADBIN_POLYFILL(GEOM, f1.VALUE::INT))) f2
$$;

-- 4-parameter version: SQL wrapper that calls RAQUET_POLYFILL JS UDF
-- Supports 'center', 'intersects', 'contains' modes
CREATE OR REPLACE FUNCTION RAQUET_DB.RAQUET.__RAQUET_REGION_BLOCKS(
    GEOM GEOGRAPHY,
    MIN_ZOOM DOUBLE,
    MAX_ZOOM DOUBLE,
    MODE VARCHAR
)
RETURNS ARRAY
AS
$$
    SELECT ARRAY_AGG(f.VALUE::NUMBER)
    FROM TABLE(FLATTEN(
        RAQUET_DB.RAQUET.RAQUET_POLYFILL(ST_ASGEOJSON(GEOM)::VARCHAR, MIN_ZOOM, MAX_ZOOM, MODE)
    )) f
$$;

-- RAQUET_BLOCKS: convenience table function that returns blocks as rows
-- Wraps __RAQUET_REGION_BLOCKS so users don't need FLATTEN boilerplate.
--
-- Usage:
--   SELECT RAQUET_AGGREGATE_STATS(
--       ARRAY_AGG(ST_RASTERSUMMARYSTATS(
--           BAND_1_COUNT, BAND_1_SUM, BAND_1_MIN, BAND_1_MAX, BAND_1_MEAN, BAND_1_STDDEV
--       ))
--   ) AS stats
--   FROM my_raquet_table
--   WHERE BAND_1_COUNT IS NOT NULL
--     AND BLOCK IN (SELECT BLOCK FROM TABLE(RAQUET_BLOCKS(
--         ST_GEOGRAPHYFROMWKT('POLYGON((...))'), 17, 17
--     )));

-- DOUBLE overload
CREATE OR REPLACE FUNCTION RAQUET_DB.RAQUET.RAQUET_BLOCKS(
    GEOM GEOGRAPHY,
    MIN_ZOOM DOUBLE,
    MAX_ZOOM DOUBLE
)
RETURNS TABLE(BLOCK NUMBER)
AS
$$
    SELECT f.VALUE::NUMBER AS BLOCK
    FROM TABLE(FLATTEN(
        RAQUET_DB.RAQUET.__RAQUET_REGION_BLOCKS(GEOM, MIN_ZOOM, MAX_ZOOM, 'intersects')
    )) f
$$;

-- NUMBER overload (avoids 17::DOUBLE casts)
CREATE OR REPLACE FUNCTION RAQUET_DB.RAQUET.RAQUET_BLOCKS(
    GEOM GEOGRAPHY,
    MIN_ZOOM NUMBER,
    MAX_ZOOM NUMBER
)
RETURNS TABLE(BLOCK NUMBER)
AS
$$
    SELECT f.VALUE::NUMBER AS BLOCK
    FROM TABLE(FLATTEN(
        RAQUET_DB.RAQUET.__RAQUET_REGION_BLOCKS(GEOM, MIN_ZOOM::DOUBLE, MAX_ZOOM::DOUBLE, 'intersects')
    )) f
$$;
