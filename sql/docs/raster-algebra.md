# RASTER_ALGEBRA

`RASTER_ALGEBRA` evaluates a user-defined expression pixel by pixel over one or more RaQuet rasters and writes a new RaQuet v0.5.0 raster, with gzip tiles, per-tile statistics and a metadata row. It runs inside BigQuery and Snowflake as JavaScript UDFs plus a procedure. The CARTO Analytics Toolbox ships a production version built from the same library.

## What's here

| Piece | File |
|---|---|
| Parser, planner, vectorized evaluator, codec, stats, SQL builders | `libraries/javascript/src/raquet_algebra.js` (bundled to `build/raquet_algebra.js`, ~72 KB incl. pako inflate + gzip) |
| BigQuery procedure + UDFs | `platforms/bigquery/functions/RASTER_ALGEBRA.sql` |
| Snowflake procedure + UDF (generated, libraries inlined) | `platforms/snowflake/templates/RASTER_ALGEBRA.sql.tmpl` → `scripts/build_snowflake_algebra.mjs` → `platforms/snowflake/functions/RASTER_ALGEBRA.sql` |
| Unit tests (34) | `libraries/javascript/test/raquet_algebra.test.mjs` (`npm test` in `libraries/javascript`) |

```sql
CALL `myproject.raquet.RASTER_ALGEBRA`(
  ['cartobq.raquet.world_elevation', 'cartobq.raquet.world_solar_pvout'],   -- $a, $b
  'pv_high = if($a > 1000, $b, 0); ratio = $b / ($a + 1)',                  -- 2 output bands
  'myproject.mydataset.pvout_elev',
  NULL);
```

## Design

**Expression language.** `$a`, `$b`, … are inputs, following the Custom SQL component convention. Bands can be referenced as `$a.band_1` or `$a.nir` (by metadata name), `$a[4]` or `$a.4` (1-based index), or `$a` for a single-band input. Supported syntax:

- Arithmetic: `+ - * / % ^`
- Comparisons and logic: `and or not`. Chained comparisons such as `1 < $a < 3` are rejected with a hint to use `and`.
- Functions: `if()` and an allowlist of math functions
- Multiple output bands: `name = expr; name = expr`.

The parser is a hand-written tokenizer plus a Pratt parser that produces an AST. Constant subtrees are folded, and the result is interpreted, never turned into code: there is no `eval` or `new Function`. It is also bounded:

- identifiers are allowlisted;
- expressions are limited to 10,000 characters, 2,000 AST nodes and 64 levels of nesting;
- non-finite literals such as `1e999` are rejected, because JSON would turn them into `null`.

The evaluator reuses intermediate buffers, so memory per block is bounded by the depth limit.

**Options** are validated strictly. Unknown keys, invalid values and malformed JSON all fail with a clear message.

**Identifiers and SQL.** No user text reaches the generated SQL verbatim:

- Table names are validated per identifier part.
- Band and column names come from metadata and are validated as identifiers.
- The plan is embedded as a base64 literal, and the metadata row goes in as a base64 literal (BigQuery) or a bind (Snowflake).
- On Snowflake:
  - input columns are resolved case-insensitively against the table's actual columns (RaQuet files loaded with `MATCH_BY_COLUMN_NAME` have upper-case `BAND_1`) and quoted;
  - output columns are quoted upper case (`"NDVI"`, `"NDVI_COUNT"`), so reserved words like `order` also work;
  - user quoting of table names is preserved.
- Tests cover injection through the expression, the table names on both warehouses (including `$$` on Snowflake) and the metadata band names.

**Flow.**

1. `PLAN(expression, input metadatas, options)` validates everything before any DDL, so a failure creates no output table .
2. `CREATE TABLE … CLUSTER BY block AS`, which fails if the output already exists. The output is also rejected if it names one of the inputs, even at a different qualification level (`ds.t` vs `proj.ds.t`). One JS UDF call per block does the work:
   - decode every operand, sharing a single decode of interleaved `pixels`;
   - evaluate the AST vectorized over Float64Arrays;
   - apply the nodata mask;
   - cast to the output type, compute tile stats (two-pass stddev), gzip, base64.
3. Aggregate native-zoom tile stats into a JSON string. This reads only the stats columns, and the variance is combined across tiles with the parallel algorithm.
4. Insert a single metadata row with v0.5.0 metadata: bands with STATISTICS_*, `tile_statistics`, and a `processing` block with the expression and inputs.

   If step 3 or 4 fails, the incomplete output table is dropped.

**Coverage and grid alignment.** QUADBIN tiles at the same zoom are pixel-aligned by construction, so inputs only need the same block size and native zoom (`max_zoom`). Extents can differ. Each output band covers the blocks where *all the inputs it references* exist:

- When every output references every input, the inputs are inner-joined on `block`.
- Otherwise the query takes the union of blocks plus left joins, so `x = $a; y = $a + $b` keeps all of `$a` in `x`. Verified end to end: 7,424 vs 3,029 blocks.

Grid mismatches fail with an actionable message.

**Nodata .** An output pixel is nodata when any operand it references is nodata or NaN, or when the result is not finite (x/0, log of a negative). An integer output that overflows is also nodata.

- Input sentinels are compared in the band's own precision. For example, a float32 band with `nodata: -3.4028235e38` matches the stored pixels, which a double-precision compare would miss.
- `output_nodata` is stored in the output precision.
- Blocks with no valid output in any band are dropped. Bands of kept blocks are always written with the declared compression.
- The default output is `float32` with nodata `NaN`.

## Results (BigQuery, on-demand)

**Correctness.** 50 random native-zoom blocks (3,276,800 pixels) were decoded with the independent `RAQUET_DECODE_BAND` and compared with the same formula computed in plain SQL:

- 0 nodata mismatches and 0 value mismatches on both output bands;
- maximum absolute error 4.8e-8, which is float32 rounding.

After the review fixes, the same check gives the same result: 0 mismatches, maximum error 4.8e-8.

**Error paths** (all verified in BigQuery; none left an output table behind):

- missing band;
- grid mismatch (z7 vs z9);
- unknown function;
- injection through the expression and through the table name;
- output table equal to an input, including the 2-part vs 3-part name;
- output table already exists;
- missing input table (clear "could not be read as a RaQuet raster" message);
- unknown option;
- chained comparison;
- multi-band input without a band reference;
- strict `require_version: 0.5.0` on a 0.3.0 input.

**Performance.** The pair is `world_elevation` × `world_solar_pvout`: 7,424 blocks each at z0–z7, 256×256 pixels. About 4,400 blocks are written, since solar only covers land.

| Run | Blocks out | CTAS wall | CTAS slot-s | Output | Metadata step |
|---|---|---|---|---|---|
| `$b - $a`, native zoom only | 3,030 | 14 s | 559 | 354 MB | — |
| `$b - $a`, all zooms, gzip 6 | 4,398 | 20 s | 587 | 508 MB | 110 slot-s (INSERT…SELECT) |
| `$b - $a`, all zooms, gzip 1 | 4,398 | 21 s | 458 | 512 MB | — |
| 2 outputs, gzip 6 (first run) | 4,398 | 33 s | 977 | — | 162 slot-s |
| 2 outputs, gzip 1, final metadata step | 4,398 | 14 s | 672 | — | 15 slot-s |
| Same, after review fixes (buffer reuse, two-pass stats, always gzip) | 4,398 | 13 s | 770 | — | 5 slot-s |

**At scale:** 15 GB LiDAR elevation × slope, 563K tiles each at z17, 256² px, RaQuet v0.5.0. The expression `suitable = $b < 5 and $a < 300` as `uint8` took 235 s wall time, about 9,700 slot-seconds and 27 GB billed.

It wrote 75,497 blocks. Slope is nodata outside that footprint: 200 random dropped blocks, checked independently, all had 0 valid slope pixels.

The join evaluates all ~563K blocks present in both tables, most of them all-nodata. That is about 0.017 slot-seconds per evaluated block, or about 0.13 per block written.

Local V8 profile per 256×256 block: about 19 ms for 2 operands and 1 output with gzip level 6, about 9 ms without compression.

- About 60% of block time is output gzip (pako `deflate_slow` / `longest_match`), because computed float32 rasters compress poorly.
- Level 1 is about 20% cheaper for less than 1% more bytes, so it is the default. The evaluation itself is a small share.

Bytes billed are the same in every variant (1.11 GB): the join reads each input's tile bytes once.

## Known behaviours and limitations

- **Overviews.** By default (`overviews: 'evaluate'`) the expression is evaluated at every zoom level the inputs share. This adds only about 5% slot time and produces a renderable pyramid immediately. For non-linear expressions (NDVI, thresholds), overview pixels are *f(downsampled inputs)*, not *downsample(f(inputs))*, which is fine for visualization. This is recorded in `processing.overviews`. `overviews: 'none'` computes the native resolution only.
- **Scale/offset.** Expressions see the stored (DN) values unless `apply_scale_offset: true`.
- **Versions.** RaQuet ≥ 0.3.0 is accepted by default; `require_version: '0.5.0'` makes it strict. Legacy CARTO rasters (with `block_resolution` metadata) are rejected.
- **Output nodata.** The default output is `float32` with nodata `"NaN"` (Zarr v3 encoding). For unsigned integer outputs the default nodata is the type maximum, so masks such as `if($a > 0, 255, 0)` as `uint8` should set `output_nodata` explicitly.
- **Snowflake.** The generated UDF body is about 80 KB with the libraries inlined, and it caches the library on `globalThis`.
