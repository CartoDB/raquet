-- __RAQUET_PIXEL_POSITIONS for Databricks SQL Warehouse
-- View that generates all pixel positions for a 256x256 block
--
-- Returns 65,536 rows representing all (x, y) positions in a Raquet block.
-- Used for clipping operations to enumerate all pixels.
--
-- Columns: x INT, y INT
--   - x: 0-255 (left to right)
--   - y: 0-255 (top to bottom)
--
-- Note: BigQuery uses a TABLE FUNCTION for this. Databricks does not support
-- parameterless SQL table functions, so this is implemented as a VIEW.
--
-- Usage:
--   SELECT * FROM ${catalog}.${schema}.__RAQUET_PIXEL_POSITIONS
--   -- Returns 65,536 rows

CREATE OR REPLACE VIEW ${catalog}.${schema}.__RAQUET_PIXEL_POSITIONS AS
SELECT x, y
FROM (SELECT EXPLODE(SEQUENCE(0, 255)) AS x),
     (SELECT EXPLODE(SEQUENCE(0, 255)) AS y);
