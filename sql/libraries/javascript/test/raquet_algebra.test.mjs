// Unit tests for the raster algebra library.  Run: node --test test/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import pako from 'pako';
import lib from '../src/raquet_algebra.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const N = 256 * 256;

function meta(overrides = {}) {
    return {
        file_format: 'raquet',
        version: '0.5.0',
        bounds: [-10, -10, 10, 10],
        compression: 'gzip',
        tiling: { scheme: 'quadbin', block_width: 256, block_height: 256, min_zoom: 3, max_zoom: 7, pixel_zoom: 15 },
        bands: [{ name: 'band_1', type: 'float32', nodata: -9999 }],
        ...overrides
    };
}

function tile(type, fill, { gzip = true } = {}) {
    const Ctor = { uint8: Uint8Array, int16: Int16Array, int32: Int32Array, float32: Float32Array, float64: Float64Array }[type];
    const arr = new Ctor(fill.length);
    fill.forEach((v, i) => { arr[i] = v; });
    const bytes = new Uint8Array(arr.buffer);
    return gzip ? pako.gzip(bytes) : bytes;
}

function decodeOut(result, type = 'float32') {
    const raw = pako.ungzip(lib.base64Decode(result.data));
    const Ctor = { float32: Float32Array, float64: Float64Array, int16: Int16Array, uint8: Uint8Array, int32: Int32Array }[type];
    return new Ctor(raw.buffer, raw.byteOffset, raw.length / Ctor.BYTES_PER_ELEMENT);
}

const range = (n, f) => Array.from({ length: n }, (_, i) => f(i));

// --------------------------------------------------------------------------
test('parse: precedence, unary, power right-assoc', () => {
    const [{ ast }] = lib.parse('-$a ^ 2 ^ 3 + 1 * 2');
    assert.equal(ast.t, 'bin');
    assert.equal(ast.op, '+');
    assert.equal(ast.l.t, 'un');                  // -( a^(2^3) )
    assert.equal(ast.l.x.op, '^');
    assert.equal(ast.l.x.r.op, '^');
});

test('parse: named multi-output', () => {
    const outs = lib.parse('ndvi = ($a.b4 - $a.b3) / ($a.b4 + $a.b3); diff = $b - $a.b4;');
    assert.deepEqual(outs.map(o => o.name), ['ndvi', 'diff']);
});

test('parse: rejects anything outside the grammar', () => {
    const bad = [
        '', '   ', '$a +', 'foo($a)', 'process.exit()', '$a; DROP TABLE x',
        "$a') ; SELECT 1 --", '`x`', '$a.band_1 + "1"', 'constructor', '__proto__',
        'this', '$a[0]', 'if($a, 1)', 'Math.random()', '$a = 1'
    ];
    for (const src of bad) {
        assert.throws(() => lib.plan(src, [meta()]), lib.AlgebraError, `should reject: ${src}`);
    }
});

// --------------------------------------------------------------------------
test('plan: resolves references, dedupes operands', () => {
    const p = lib.plan('($b.band_1 - $a) / $a', [meta(), meta()]);
    assert.equal(p.operands.length, 2);
    assert.deepEqual(p.operands.map(o => [o.input, o.column]), [[1, 'band_1'], [0, 'band_1']]);
    assert.deepEqual(p.zoom_range, [3, 7]);
});

test('plan: clear errors', () => {
    const four = meta({ bands: ['b1', 'b2', 'b3', 'b4'].map(n => ({ name: n, type: 'uint16', nodata: 0 })) });
    assert.throws(() => lib.plan('$a.band_9', [four]), /Band 'band_9' not found in \$a. Available bands: b1, b2, b3, b4/);
    assert.throws(() => lib.plan('$a[9]', [four]), /Band 9 not found/);
    assert.throws(() => lib.plan('$a', [four]), /has 4 bands/);
    assert.throws(() => lib.plan('$a + $c', [meta(), meta()]), /references \$c but only 2 input/);
    assert.throws(() => lib.plan('$a', [meta(), meta()]), /\$b are not referenced/);
    assert.throws(() => lib.plan('$a', [{ block_resolution: 5, bands: [] }]), /not a RaQuet raster/);
    assert.throws(() => lib.plan('$a', [meta({ version: '0.4.0' })], { require_version: '0.5.0' }), /0.5.0 or later/);
    assert.throws(() => lib.plan('$a', [meta({ time: { count: 3 } })]), /time dimension/);
    assert.throws(() => lib.plan('$a', [meta({ compression: 'webp' })]), /WebP/);
});

test('plan: grid alignment', () => {
    const other = meta({ tiling: { ...meta().tiling, max_zoom: 8 } });
    assert.throws(() => lib.plan('$a - $b', [meta(), other]), /not on the same grid: native zoom/);
    const small = meta({ tiling: { ...meta().tiling, block_width: 512, block_height: 512 } });
    assert.throws(() => lib.plan('$a - $b', [meta(), small]), /block size/);
    const far = meta({ bounds: [50, 50, 60, 60] });
    assert.throws(() => lib.plan('$a - $b', [meta(), far]), /do not overlap/);
    // Different extents & overview depth are fine: intersect
    const partial = meta({ bounds: [0, 0, 20, 20], tiling: { ...meta().tiling, min_zoom: 5 } });
    const p = lib.plan('$a - $b', [meta(), partial]);
    assert.deepEqual(p.bounds, [0, 0, 10, 10]);
    assert.deepEqual(p.zoom_range, [5, 7]);
    assert.deepEqual(lib.plan('$a - $b', [meta(), partial], { overviews: 'none' }).zoom_range, [7, 7]);
});

test('plan: rejects unsafe identifiers from metadata', () => {
    const evil = meta({ bands: [{ name: 'x`; DROP TABLE t; --', type: 'float32' }] });
    assert.throws(() => lib.plan('$a[1]', [evil]), /not a safe SQL identifier/);
});

// --------------------------------------------------------------------------
test('evalBlock: two-raster difference with nodata rule', () => {
    const p = lib.plan('$b - $a', [meta(), meta()]);
    const a = range(N, i => (i % 10 === 0 ? -9999 : i));        // nodata every 10th
    const b = range(N, i => (i % 7 === 0 ? -9999 : 2 * i));     // nodata every 7th
    // operands order: $b first (first referenced), then $a
    const payloads = p.operands.map(o => (o.input === 0 ? tile('float32', a) : tile('float32', b)));
    const [r] = lib.evalBlock(JSON.stringify(p), payloads);
    const out = decodeOut(r);
    let expectCount = 0, expectSum = 0;
    for (let i = 0; i < N; i++) {
        if (i % 10 === 0 || i % 7 === 0) {
            assert.ok(Number.isNaN(out[i]), `pixel ${i} should be nodata`);
        } else {
            assert.equal(out[i], Math.fround(i));
            expectCount++;
            expectSum += Math.fround(i);
        }
    }
    assert.equal(r.count, expectCount);
    assert.equal(r.sum, expectSum);
    assert.equal(r.min, 1);
});

test('evalBlock: NDVI on a 4-band input, div-by-zero -> nodata', () => {
    const four = meta({ bands: ['band_1', 'band_2', 'band_3', 'band_4'].map(n => ({ name: n, type: 'uint16', nodata: null })) });
    const p = lib.plan('($a.band_4 - $a.band_3) / ($a.band_4 + $a.band_3)', [four]);
    const red = range(N, i => i % 100);
    const nir = range(N, i => (i % 100 === 0 ? 0 : 200));
    const u16 = vals => pako.gzip(new Uint8Array(Uint16Array.from(vals).buffer));
    const payloads = p.operands.map(o => (o.column === 'band_4' ? u16(nir) : u16(red)));
    const [r] = lib.evalBlock(p, payloads);
    const out = decodeOut(r);
    assert.ok(Number.isNaN(out[0]));                       // 0/0
    assert.ok(Math.abs(out[1] - (199 / 201)) < 1e-6);
    assert.ok(r.min >= -1 && r.max <= 1);
    let expected = 0;
    for (let i = 0; i < N; i++) if (red[i] + nir[i] !== 0) expected++;
    assert.equal(r.count, expected);
});

test('evalBlock: interleaved input decoded once, int16 output clamps to nodata', () => {
    const rgb = meta({
        band_layout: 'interleaved',
        bands: ['r', 'g', 'b'].map(n => ({ name: n, type: 'uint8', nodata: null }))
    });
    const p = lib.plan('$a.r * 1000 - $a.b', [rgb], { output_type: 'int16' });
    assert.equal(p.operands.every(o => o.column === 'pixels'), true);
    const bip = new Uint8Array(N * 3);
    for (let i = 0; i < N; i++) { bip[i * 3] = i % 40; bip[i * 3 + 1] = 7; bip[i * 3 + 2] = 5; }
    const payload = pako.gzip(bip);
    const [r] = lib.evalBlock(p, p.operands.map(() => payload));
    const out = decodeOut(r, 'int16');
    assert.equal(out[1], 995);
    assert.equal(out[33], -32768);          // 33000 - 5 > int16 max -> nodata (type min)
    assert.equal(r.max, 31995);          // 32*1000-5, largest value that fits
});

test('evalBlock: multi-band output, all-nodata block dropped, empty payload', () => {
    const p = lib.plan('x = $a + 1; y = $a * 2', [meta()]);
    const res = lib.evalBlock(p, [tile('float32', range(N, () => 3))]);
    assert.equal(res.length, 2);
    assert.equal(decodeOut(res[0])[0], 4);
    assert.equal(decodeOut(res[1])[0], 6);
    assert.equal(lib.evalBlock(p, [tile('float32', range(N, () => -9999))]), null);
    assert.equal(lib.evalBlock(p, ['']), null);
});

test('evalBlock: comparisons, if(), functions', () => {
    const p = lib.plan('if($a > 10 and not ($a == 20), sqrt($a), clamp($a, 0, 5))', [meta()]);
    const [r] = lib.evalBlock(p, [tile('float32', range(N, i => [4, 16, 20, 100][i % 4]))]);
    const out = decodeOut(r);
    assert.deepEqual([...out.slice(0, 4)], [4, 4, 5, 10]);
});

test('evalBlock: physical values (scale/offset) by default, DN on request', () => {
    const m = meta({ bands: [{ name: 'band_1', type: 'int16', nodata: -1, scale: 0.5, offset: 10 }] });
    const t = tile('int16', range(N, () => 4));
    assert.equal(decodeOut(lib.evalBlock(lib.plan('$a', [m]), [t])[0])[0], 12);
    assert.equal(decodeOut(lib.evalBlock(lib.plan('$a', [m], { apply_scale_offset: false }), [t])[0])[0], 4);
});

test('plan: RaQuet v0.5.0 is required by default', () => {
    assert.throws(() => lib.plan('$a', [meta({ version: '0.4.0' })]), /RaQuet 0.4.0; version 0.5.0 or later is required/);
    assert.ok(lib.plan('$a', [meta({ version: '0.4.0' })], { require_version: '0.3.0' }));
});

// --------------------------------------------------------------------------
test('metadata: v0.5.0 with aggregated statistics', () => {
    const p = lib.plan('$a', [meta()]);
    const m = JSON.parse(lib.buildMetadata(p, { num_blocks: 5, min_zoom: 3, max_zoom: 7, bands: [{ count: 4, min: 1, max: 4, sum: 10, m2: 5 }] }));
    assert.equal(m.version, '0.5.0');
    assert.equal(m.tile_statistics, true);
    assert.equal(m.bands[0].nodata, 'NaN');
    assert.equal(m.bands[0].STATISTICS_MEAN, 2.5);
    assert.ok(Math.abs(m.bands[0].STATISTICS_STDDEV - Math.sqrt(1.25)) < 1e-12);
    assert.equal(m.tiling.pixel_zoom, 15);
    assert.equal(m.tiling.num_blocks, 5);
});

test('sql: safe literals and identifiers only', () => {
    const p = lib.plan('$b - $a', [meta(), meta()]);
    const sql = lib.buildBigQuerySql(p, ['proj.ds.a', '`proj.ds.b`'], 'proj.ds.out', 'proj.raquet');
    assert.match(sql, /^CREATE TABLE `proj.ds.out`/);
    assert.match(sql, /JOIN i1 USING \(block\)/);
    assert.match(sql, /__RASTER_ALGEBRA_BLOCK`\('[A-Za-z0-9+/=]+', \[/);
    assert.throws(() => lib.buildBigQuerySql(p, ['a.b; DROP', 'x.y'], 'p.d.o', 'p.r'), /Invalid input/);
    assert.throws(() => lib.buildBigQuerySql(p, ['p.d.a', 'p.d.b'], 'p.d.a', 'p.r'), /must differ/);
});

// --------------------------------------------------------------------------
test('built IIFE bundle works without require/Buffer (BigQuery/Snowflake-like sandbox)', () => {
    const code = fs.readFileSync(path.join(here, '..', 'build', 'raquet_algebra.js'), 'utf8');
    const ctx = vm.createContext({});
    vm.runInContext(code, ctx);
    const L = ctx.raquetAlgebraLib;
    const p = L.plan('$a * 2', [JSON.stringify(meta())]);
    const payload = lib.base64Encode(tile('float32', range(N, i => i)));
    const [r] = L.evalBlock(p, [payload]);
    assert.equal(decodeOut(r)[10], 20);
});

// --------------------------------------------------------------------------
// Snowflake: run the generated UDF / procedure bodies in a sandbox
// --------------------------------------------------------------------------
function snowflakeBodies() {
    const sql = fs.readFileSync(path.join(here, '..', '..', '..', 'platforms', 'snowflake', 'functions', 'RASTER_ALGEBRA.sql'), 'utf8');
    const bodies = [...sql.matchAll(/\$\$\n([\s\S]*?)\n\$\$;/g)].map(m => m[1]);
    assert.equal(bodies.length, 2);
    return { udf: bodies[0], proc: bodies[1] };
}

test('snowflake: generated UDF body evaluates a block', () => {
    const { udf } = snowflakeBodies();
    const ctx = vm.createContext({});
    const fn = vm.runInContext(`(function(PLAN, OPERANDS) {\n${udf}\n})`, ctx);
    const p = lib.plan('$a * 2', [meta()]);
    const payload = lib.base64Encode(tile('float32', range(N, i => i)));
    const [r] = fn(lib.planToBase64(p), [payload]);
    assert.equal(decodeOut(r)[10], 20);
    // second call reuses the cached library
    assert.ok(ctx.__raquetAlgebraLib);
    assert.equal(decodeOut(fn(lib.planToBase64(p), [payload])[0])[3], 6);
});

test('snowflake: generated procedure drives the expected statements', () => {
    const { proc } = snowflakeBodies();
    const executed = [];
    const failOn = { stats: false };
    const snowflake = {
        execute({ sqlText, binds }) {
            executed.push({ sqlText, binds });
            let rows = [];
            if (/^SELECT "METADATA" FROM/.test(sqlText)) rows = [[JSON.stringify(meta())]];
            if (/^WITH S AS/.test(sqlText)) {
                if (failOn.stats) throw new Error('boom');
                rows = [[JSON.stringify({ num_blocks: 2, min_zoom: 3, max_zoom: 7, bands: [{ count: 10, min: 0, max: 9, sum: 45, m2: 82.5 }] })]];
            }
            let k = -1;
            // Mirrors the Snowflake API: column metadata lives on Statement, not ResultSet
            return { next: () => ++k < rows.length, getColumnValue: c => rows[k][c - 1] };
        },
        createStatement({ sqlText }) {
            const self = this;
            let columns = [];
            return {
                execute() {
                    const rs = self.execute({ sqlText });
                    if (/LIMIT 0$/.test(sqlText)) columns = ['BLOCK', 'METADATA', 'BAND_1'];
                    return rs;
                },
                getColumnCount: () => columns.length,
                getColumnName: c => columns[c - 1]
            };
        }
    };
    const ctx = vm.createContext({ snowflake });
    const fn = vm.runInContext(`(function(INPUTS, EXPRESSION, OUTPUT_TABLE, OPTIONS) {\n${proc}\n})`, ctx);
    const res = JSON.parse(fn(['DB.S.A', 'DB.S.B'], '$b - $a', 'DB.S.OUT', null));
    assert.equal(res.num_blocks, 2);
    const create = executed.find(e => /^CREATE TABLE/.test(e.sqlText));
    assert.match(create.sqlText, /^CREATE TABLE DB.S.OUT/);
    assert.match(create.sqlText, /SELECT "BLOCK" AS BLOCK, "BAND_1" AS C0/);
    assert.match(create.sqlText, /RAQUET_DB.RAQUET.__RASTER_ALGEBRA_BLOCK\('[A-Za-z0-9+/=]+', ARRAY_CONSTRUCT\(COALESCE\(BASE64_ENCODE\(I1.C0\), ''\)/);
    assert.match(create.sqlText, /AS "BAND_1_COUNT"/);
    const insert = executed.find(e => /^INSERT/.test(e.sqlText));
    assert.equal(insert.sqlText, 'INSERT INTO DB.S.OUT ("BLOCK", "METADATA") VALUES (0, ?)');
    const md = JSON.parse(insert.binds[0]);
    assert.equal(md.version, '0.5.0');
    assert.equal(md.bands[0].STATISTICS_MEAN, 4.5);
    assert.equal(md.bands[0].STATISTICS_STDDEV, Math.sqrt(8.25));
    // validation errors surface before any DDL
    executed.length = 0;
    assert.throws(() => fn(['DB.S.A'], '$a.nope', 'DB.S.OUT', null), /Band 'nope' not found/);
    assert.ok(executed.every(e => !/CREATE/.test(e.sqlText)));
    assert.throws(() => fn(['DB.S.A; DROP TABLE X'], '$a', 'DB.S.OUT', null), /Invalid input/);
    assert.throws(() => fn(['DB.S.A$$'], '$a', 'DB.S.OUT', null), /Invalid input/);
    // a failure after the CTAS drops the incomplete output
    executed.length = 0;
    failOn.stats = true;
    assert.throws(() => fn(['DB.S.A'], '$a * 2', 'DB.S.OUT', null), /boom/);
    assert.equal(executed[executed.length - 1].sqlText, 'DROP TABLE IF EXISTS DB.S.OUT');
});

// --------------------------------------------------------------------------
// Regression tests for review findings (PR #4)
// --------------------------------------------------------------------------
test('review #1: float32 nodata written with few digits still matches', () => {
    for (const nd of [-3.4028235e38, 0.1, -3.4e38]) {
        const m = meta({ bands: [{ name: 'band_1', type: 'float32', nodata: nd }] });
        const vals = range(N, i => (i < 10 ? nd : i));
        const [r] = lib.evalBlock(lib.plan('$a', [m]), [tile('float32', vals)]);
        assert.equal(r.count, N - 10, `nodata ${nd}`);
        assert.ok(r.min >= 10);
    }
});

test('review #2: all-nodata band of a kept block is still gzip-compressed', () => {
    const m = meta({ bands: [{ name: 'b1', type: 'float32', nodata: -9999 }, { name: 'b2', type: 'float32', nodata: -9999 }] });
    const p = lib.plan('x = $a.b1; y = $a.b2', [m]);
    const payloads = p.operands.map(o => tile('float32', range(N, () => (o.column === 'b1' ? 1 : -9999))));
    const [x, y] = lib.evalBlock(p, payloads);
    assert.equal(x.count, N);
    assert.equal(y.count, 0);
    const raw = decodeOut(y); // throws if not gzip
    assert.ok(Number.isNaN(raw[0]));
});

test('review #3: outputs not referencing every input are not masked by the others', () => {
    const p = lib.plan('x = $a; y = $a + $b', [meta(), meta()]);
    assert.equal(p.join, 'outer');
    // block present in $a only: x is valid, y is nodata
    const payloads = p.operands.map(o => (o.input === 0 ? tile('float32', range(N, () => 5)) : ''));
    const [x, y] = lib.evalBlock(p, payloads);
    assert.equal(x.count, N);
    assert.equal(y.count, 0);
    const sql = lib.buildBigQuerySql(p, ['p.d.a', 'p.d.b'], 'p.d.out', 'p.r');
    assert.match(sql, /blocks AS \(\n  SELECT block FROM i0\n  UNION DISTINCT\n  SELECT block FROM i1\n\)/);
    assert.match(sql, /FROM blocks\n  LEFT JOIN i0 USING \(block\)\n  LEFT JOIN i1 USING \(block\)/);
    assert.equal(lib.plan('$b - $a', [meta(), meta()]).join, 'inner');
});

test('review #4: non-finite literals are rejected (they would not survive JSON)', () => {
    assert.throws(() => lib.plan('$a < 1e999', [meta()]), /out of range/);
    // folded constants that overflow are not folded
    const p = JSON.parse(JSON.stringify(lib.plan('$a * (1e300 * 1e300)', [meta()])));
    assert.equal(lib.evalBlock(p, [tile('float32', range(N, () => 1))]), null); // Infinity -> all nodata
});

test('review #5: Snowflake identifiers are quoted and resolved against actual columns', () => {
    const m = meta({ bands: [{ name: 'band_1', type: 'float32' }] });
    const p = lib.plan('order = $a; select = $a * 2', [m]);
    const sql = lib.buildSnowflakeSql(p, ['"MyDb"."Sch"."tbl"'], 'DB.S.OUT', 'DB.S.__RASTER_ALGEBRA_BLOCK', [['block', 'metadata', 'band_1']]);
    assert.match(sql, /FROM "MyDb"."Sch"."tbl"/);
    assert.match(sql, /SELECT "block" AS BLOCK, "band_1" AS C0/);
    assert.match(sql, /AS "ORDER"/);
    assert.match(sql, /AS "SELECT_COUNT"/);
    assert.throws(() => lib.buildSnowflakeSql(p, ['DB.S.T'], 'DB.S.OUT', 'DB.S.F', [['BLOCK', 'OTHER']]), /Column 'band_1' not found/);
});

test('review #6: output_nodata is stored in the output precision', () => {
    const p = lib.plan('$a', [meta()], { output_nodata: 0.1 });
    assert.equal(p.output.nodata, Math.fround(0.1));
    const [r] = lib.evalBlock(p, [tile('float32', range(N, i => (i < 5 ? 0.1 : 1)))]);
    assert.equal(r.count, N - 5);
});

test('review #7: output == input is detected across qualification levels', () => {
    const p = lib.plan('$a', [meta()]);
    assert.throws(() => lib.buildBigQuerySql(p, ['myproj.ds.t'], 'ds.t', 'p.r'), /must differ/);
    assert.throws(() => lib.buildSnowflakeSql(p, ['DB.S.T'], 's.t', 'DB.S.F'), /must differ/);
    assert.match(lib.buildBigQuerySql(p, ['myproj.ds.t'], 'ds.t2', 'carto'), /^CREATE TABLE `ds.t2`/);
});

test('review #9: output names are compared case-insensitively', () => {
    assert.throws(() => lib.plan('x = $a; X = $a * 2', [meta()]), /Duplicate/);
    assert.throws(() => lib.plan('Block = $a', [meta()]), /reserved/);
    assert.throws(() => lib.plan('x_COUNT = $a', [meta()]), /reserved/);
});

test('review #10: $a.N is a band index', () => {
    const four = meta({ bands: ['b1', 'b2', 'b3', 'b4'].map(n => ({ name: n, type: 'float32' })) });
    const p = lib.plan('$a.4 - $a[3]', [four]);
    assert.deepEqual(p.operands.map(o => o.column), ['b4', 'b3']);
});

test('review #11/#12: nesting depth is limited', () => {
    assert.throws(() => lib.plan('('.repeat(100) + '$a' + ')'.repeat(100), [meta()]), /nested too deeply/);
    const rightDeep = Array.from({ length: 200 }, () => '$a').join(' ^ ');
    assert.throws(() => lib.plan(rightDeep, [meta()]), /nested too deeply/);
    // long left-deep sums are fine (constant memory)
    const leftDeep = Array.from({ length: 400 }, (_, i) => `${i}`).join(' + ') + ' + $a';
    const [r] = lib.evalBlock(lib.plan(leftDeep, [meta()]), [tile('float32', range(N, () => 0))]);
    assert.equal(r.min, 79800);
});

test('review #13: tile stddev is numerically stable', () => {
    const m = meta({ bands: [{ name: 'band_1', type: 'float64', nodata: null }] });
    const [r] = lib.evalBlock(lib.plan('$a', [m], { output_type: 'float64' }),
        [tile('float64', range(N, i => 1e7 + (i % 2 ? 0.001 : -0.001)))]);
    assert.ok(Math.abs(r.stddev - 0.001) < 1e-9, `stddev ${r.stddev}`);
});

test('review #16: options are validated', () => {
    assert.deepEqual(lib.plan('$a', [meta()], 'null').output.type, 'float32');
    assert.throws(() => lib.plan('$a', [meta()], '{not json'), /options must be a JSON object/);
    assert.throws(() => lib.plan('$a', [meta()], { outputType: 'int16' }), /Unknown option 'outputType'/);
    assert.throws(() => lib.plan('$a', [meta()], { compression: 'zstd' }), /Invalid compression/);
});

test('review #17: chained comparisons are rejected; prerelease versions are older', () => {
    assert.throws(() => lib.plan('1 < $a < 3', [meta()]), /Chained comparison/);
    assert.ok(lib.plan('(1 < $a) < 3', [meta()]));
    assert.throws(() => lib.plan('$a', [meta({ version: '0.5.0-rc1' })], { require_version: '0.5.0' }), /0.5.0 or later/);
});

test('review #18: interleaved tile with too few channels is an error, not silent nodata', () => {
    const rgb = meta({ band_layout: 'interleaved', bands: ['r', 'g', 'b'].map(n => ({ name: n, type: 'uint8' })) });
    const p = lib.plan('$a.b', [rgb]);
    assert.throws(() => lib.evalBlock(p, [pako.gzip(new Uint8Array(N * 2))]), /too short|channel/);
});

test('min/max do not clobber later arguments when reusing buffers', () => {
    const p = lib.plan('max($a * 1, $a * 2, $a * 3)', [meta()]);
    const [r] = lib.evalBlock(p, [tile('float32', range(N, () => 2))]);
    assert.equal(decodeOut(r)[0], 6);
});

test('bundle has no $$ / replacement-pattern sequences (Snowflake inlining)', () => {
    const code = fs.readFileSync(path.join(here, '..', 'build', 'raquet_algebra.js'), 'utf8');
    for (const seq of ['$$', '$&', '$`', "$'"]) assert.ok(!code.includes(seq), `bundle contains ${seq}`);
});
