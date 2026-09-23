-- RASTER_ALGEBRA
--
-- Evaluates a user-defined expression pixel-by-pixel over one or more RaQuet
-- rasters and writes a new RaQuet raster table (sequential layout, gzip,
-- v0.5.0 per-tile statistics columns, metadata row at block = 0).
--
-- Usage:
--   CALL `cartobq.raquet.RASTER_ALGEBRA`(
--     ['project.dataset.before', 'project.dataset.after'],   -- $a, $b
--     '$b - $a',                                             -- expression
--     'project.dataset.delta',                               -- output table
--     '{"output_type": "float32"}'                           -- options (or NULL)
--   );
--
--   -- NDVI on a single multi-band raster, plus a second output band
--   CALL `cartobq.raquet.RASTER_ALGEBRA`(
--     ['project.dataset.sentinel2'],
--     'ndvi = ($a.band_4 - $a.band_3) / ($a.band_4 + $a.band_3); mask = $a.band_4 > 1000',
--     'project.dataset.sentinel2_ndvi', NULL);
--
-- Expression syntax: see libraries/javascript/src/raquet_algebra.js.
-- Options (JSON): output_type, output_nodata, overviews ('evaluate' | 'none'),
--   apply_scale_offset (default true), require_version (default '0.5.0'),
--   compression ('gzip' | 'none'),
--   compression_level (default 1).
--
-- Nodata: an output pixel is nodata when any referenced operand pixel is
-- nodata or the result is not finite. Blocks with no valid pixels are dropped.
--
-- Grid alignment: inputs must share block size and native zoom (max_zoom).
-- Extents may differ: each output band covers the blocks where all the
-- inputs it references exist.
--
-- The procedure fails if output_table already exists.

-- ---------------------------------------------------------------------------
-- Planner: parses + validates the expression against the inputs' metadata
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION `cartobq.raquet.__RASTER_ALGEBRA_PLAN`(
    expression STRING,
    metadatas ARRAY<STRING>,
    options STRING
)
RETURNS STRING
LANGUAGE js
OPTIONS (library = ["gs://cartobq-raquet-libs/raquet_algebra.js"])
AS r"""
    const plan = raquetAlgebraLib.plan(expression, metadatas, options);
    return JSON.stringify(plan);
""";

-- ---------------------------------------------------------------------------
-- Per-block evaluator: operands (one payload per plan operand) -> output bands
-- Returns NULL when every output pixel of the block is nodata.
-- (INT64 is not allowed in JS UDF signatures, so count is FLOAT64.)
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION `cartobq.raquet.__RASTER_ALGEBRA_BLOCK`(
    plan STRING,
    operands ARRAY<BYTES>
)
RETURNS ARRAY<STRUCT<data BYTES, count FLOAT64, min FLOAT64, max FLOAT64, sum FLOAT64, mean FLOAT64, stddev FLOAT64>>
LANGUAGE js
OPTIONS (library = ["gs://cartobq-raquet-libs/raquet_algebra.js", "gs://cartobq-raquet-libs/jpeg_decoder.js"])
AS r"""
    return raquetAlgebraLib.evalBlock(plan, operands);
""";

-- ---------------------------------------------------------------------------
-- SQL builders. All identifiers are validated in JS; the plan is embedded as
-- a base64 literal so no user text ever reaches the generated SQL verbatim.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION `cartobq.raquet.__RASTER_ALGEBRA_SQL`(
    plan STRING,
    inputs ARRAY<STRING>,
    output_table STRING,
    udf_dataset STRING
)
RETURNS STRING
LANGUAGE js
OPTIONS (library = ["gs://cartobq-raquet-libs/raquet_algebra.js"])
AS r"""
    return raquetAlgebraLib.buildBigQuerySql(plan, inputs, output_table, udf_dataset);
""";

CREATE OR REPLACE FUNCTION `cartobq.raquet.__RASTER_ALGEBRA_STATS_SQL`(
    plan STRING,
    output_table STRING
)
RETURNS STRING
LANGUAGE js
OPTIONS (library = ["gs://cartobq-raquet-libs/raquet_algebra.js"])
AS r"""
    return raquetAlgebraLib.buildBigQueryStatsSql(plan, output_table);
""";

CREATE OR REPLACE FUNCTION `cartobq.raquet.__RASTER_ALGEBRA_METADATA_INSERT_SQL`(
    plan STRING,
    stats STRING,
    inputs ARRAY<STRING>,
    output_table STRING
)
RETURNS STRING
LANGUAGE js
OPTIONS (library = ["gs://cartobq-raquet-libs/raquet_algebra.js"])
AS r"""
    return raquetAlgebraLib.buildBigQueryMetadataInsertSql(plan, stats, inputs, output_table);
""";

-- ---------------------------------------------------------------------------
-- Procedure
-- ---------------------------------------------------------------------------
CREATE OR REPLACE PROCEDURE `cartobq.raquet.RASTER_ALGEBRA`(
    inputs ARRAY<STRING>,
    expression STRING,
    output_table STRING,
    options STRING
)
BEGIN
    DECLARE metadatas ARRAY<STRING> DEFAULT [];
    DECLARE md STRING;
    DECLARE plan STRING;
    DECLARE stats STRING;
    DECLARE i INT64 DEFAULT 0;

    IF inputs IS NULL OR ARRAY_LENGTH(inputs) = 0 THEN
        RAISE USING MESSAGE = 'RASTER_ALGEBRA: at least one input raster is required';
    END IF;
    IF expression IS NULL OR TRIM(expression) = '' THEN
        RAISE USING MESSAGE = 'RASTER_ALGEBRA: expression cannot be empty';
    END IF;
    IF output_table IS NULL THEN
        RAISE USING MESSAGE = 'RASTER_ALGEBRA: output_table cannot be NULL';
    END IF;

    -- Read each input's metadata row. Names are validated before interpolation.
    WHILE i < ARRAY_LENGTH(inputs) DO
        IF inputs[OFFSET(i)] IS NULL
           OR NOT REGEXP_CONTAINS(inputs[OFFSET(i)], r'^`?[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+){1,2}`?$') THEN
            RAISE USING MESSAGE = FORMAT('RASTER_ALGEBRA: invalid input table name %T', inputs[OFFSET(i)]);
        END IF;
        SET md = NULL;
        BEGIN
            EXECUTE IMMEDIATE FORMAT(
                'SELECT metadata FROM `%s` WHERE block = 0 AND metadata IS NOT NULL LIMIT 1',
                REPLACE(inputs[OFFSET(i)], '`', '')
            ) INTO md;
        EXCEPTION WHEN ERROR THEN
            RAISE USING MESSAGE = FORMAT(
                'RASTER_ALGEBRA: input %s could not be read as a RaQuet raster: %s',
                inputs[OFFSET(i)], @@error.message
            );
        END;
        SET metadatas = ARRAY_CONCAT(metadatas, [IFNULL(md, 'null')]);
        SET i = i + 1;
    END WHILE;

    -- Validate everything before creating anything (no output on failure)
    SET plan = `cartobq.raquet.__RASTER_ALGEBRA_PLAN`(expression, metadatas, options);

    -- Fails if the output table already exists
    EXECUTE IMMEDIATE `cartobq.raquet.__RASTER_ALGEBRA_SQL`(plan, inputs, output_table, 'cartobq.raquet');

    -- Metadata row: aggregate tile stats (cheap), then a single-row INSERT.
    -- On failure the incomplete output is removed.
    BEGIN
        EXECUTE IMMEDIATE `cartobq.raquet.__RASTER_ALGEBRA_STATS_SQL`(plan, output_table) INTO stats;
        EXECUTE IMMEDIATE `cartobq.raquet.__RASTER_ALGEBRA_METADATA_INSERT_SQL`(plan, stats, inputs, output_table);
    EXCEPTION WHEN ERROR THEN
        EXECUTE IMMEDIATE FORMAT('DROP TABLE IF EXISTS `%s`', REPLACE(output_table, '`', ''));
        RAISE USING MESSAGE = FORMAT('RASTER_ALGEBRA: %s', @@error.message);
    END;
END;
