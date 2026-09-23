-- __RAQUET_PIXEL_POSITIONS (Snowflake version)
-- View that generates all pixel positions for a 256x256 block
--
-- Returns 65,536 rows representing all (x, y) positions in a Raquet block.
-- Used for clipping operations to enumerate all pixels.
--
-- Columns: x NUMBER, y NUMBER
--   - x: 0-255 (left to right)
--   - y: 0-255 (top to bottom)
--
-- Note: BigQuery uses a TABLE FUNCTION for this. Snowflake does not support
-- parameterless SQL table functions, so this is implemented as a VIEW.
--
-- Usage:
--   SELECT * FROM RAQUET_DB.RAQUET.__RAQUET_PIXEL_POSITIONS
--   -- Returns 65,536 rows

CREATE OR REPLACE VIEW RAQUET_DB.RAQUET.__RAQUET_PIXEL_POSITIONS AS
SELECT f1.VALUE::INT AS x, f2.VALUE::INT AS y
FROM TABLE(FLATTEN(ARRAY_GENERATE_RANGE(0, 256))) f1,
     TABLE(FLATTEN(ARRAY_GENERATE_RANGE(0, 256))) f2;
