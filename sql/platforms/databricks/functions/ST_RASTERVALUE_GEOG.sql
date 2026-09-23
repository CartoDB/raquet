-- ST_RASTERVALUE_GEOG for Databricks SQL Warehouse
-- Gets raster value at a GEOGRAPHY point
-- This is a SQL wrapper that extracts lon/lat from the point
-- Returns: DOUBLE (scalar)

CREATE OR REPLACE FUNCTION ${catalog}.${schema}.ST_RASTERVALUE_GEOG(
    block BIGINT,
    band BINARY,
    point STRING,  -- WKT or GeoJSON point
    metadata STRING,
    band_index INT
)
RETURNS DOUBLE
LANGUAGE SQL
DETERMINISTIC
COMMENT 'Gets raster value at a GEOGRAPHY point. Wrapper for ST_RASTERVALUE.'
RETURN
    ${catalog}.${schema}.ST_RASTERVALUE(
        block,
        band,
        ST_X(ST_GEOMFROMTEXT(point)),
        ST_Y(ST_GEOMFROMTEXT(point)),
        metadata,
        band_index
    );
