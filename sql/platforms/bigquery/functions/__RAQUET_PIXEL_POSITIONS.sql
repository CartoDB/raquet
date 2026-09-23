-- __RAQUET_PIXEL_POSITIONS
-- Table function that generates all pixel positions for a 256x256 block
--
-- Returns 65,536 rows representing all (x, y) positions in a Raquet block.
-- Used for clipping operations to enumerate all pixels.
--
-- Returns: TABLE<x INT64, y INT64>
--   - x: 0-255 (left to right)
--   - y: 0-255 (top to bottom)
--
-- Usage:
--   SELECT * FROM `cartobq.raquet.__RAQUET_PIXEL_POSITIONS`()
--   -- Returns 65,536 rows

CREATE OR REPLACE TABLE FUNCTION `cartobq.raquet.__RAQUET_PIXEL_POSITIONS`()
AS (
    SELECT x, y
    FROM UNNEST(GENERATE_ARRAY(0, 255)) AS x,
         UNNEST(GENERATE_ARRAY(0, 255)) AS y
);
