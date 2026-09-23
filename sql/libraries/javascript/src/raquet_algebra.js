/**
 * Raquet Raster Algebra Library
 *
 * Pixel-by-pixel evaluation of user-defined expressions over one or more
 * RaQuet rasters, producing a new RaQuet raster (sequential layout, gzip,
 * v0.5.0 per-tile statistics).
 *
 * Pieces:
 *   parse()        expression text -> AST (whitelisted grammar, never eval'd)
 *   plan()         AST + input metadata -> validated, serializable plan
 *   evalBlock()    plan + operand tile payloads -> encoded output bands + stats
 *   buildMetadata  plan + aggregated stats -> RaQuet v0.5.0 metadata JSON
 *   buildBigQuery* / buildSnowflake*  plan -> SQL statements
 *
 * Expression syntax:
 *   $a, $b, ...            inputs, in the order they are passed
 *   $a.band_1, $a.nir      band by metadata name
 *   $a[1], $a.1            band by 1-based index
 *   $a                     only band of a single-band input
 *   + - * / % ^ **         arithmetic (^ and ** are power, right-assoc)
 *   < <= > >= == !=        comparisons (1 / 0); chaining (a < b < c) is rejected
 *   and or not && || !     logical (1 / 0)
 *   if(cond, a, b), abs, sqrt, exp, log, log10, log2, pow, min, max,
 *   floor, ceil, round, clamp, sin, cos, tan, atan, atan2
 *   name = expr; name = expr     several output bands (optional names)
 *
 * Nodata rule (PRD): an output pixel is nodata when any operand pixel it
 * references is nodata, or when the result is not finite (x/0, log(-1), ...).
 */
import { inflate, gzip } from 'pako';

// "$" is never written next to "${", a quote or a backtick in this file:
// Snowflake delimits function bodies with $$, and builds that inline the
// library with String.replace interpret $$, $&, $` and $' specially.
const DOLLAR = String.fromCharCode(36);
const inputLabel = i => String.fromCharCode(97 + i); // a, b, c...
const inputRef = i => DOLLAR + inputLabel(i);
const refName = name => DOLLAR + name;

// ---------------------------------------------------------------------------
// base64 (fast table-based; BigQuery passes BYTES to JS UDFs as base64)
// ---------------------------------------------------------------------------
const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
const B64_LOOKUP = new Uint8Array(256).fill(255);
for (let i = 0; i < B64.length; i++) B64_LOOKUP[B64.charCodeAt(i)] = i;
B64_LOOKUP['-'.charCodeAt(0)] = 62; // tolerate url-safe
B64_LOOKUP['_'.charCodeAt(0)] = 63;

function base64Decode(str) {
    let len = str.length;
    while (len > 0 && (str.charCodeAt(len - 1) === 61 /* = */ || str.charCodeAt(len - 1) <= 32)) len--;
    const out = new Uint8Array((len * 3) >> 2);
    let o = 0, acc = 0, bits = 0;
    for (let i = 0; i < len; i++) {
        const v = B64_LOOKUP[str.charCodeAt(i)];
        if (v === 255) continue; // skip whitespace/newlines
        acc = ((acc << 6) | v) & 0xffffff;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out[o++] = (acc >> bits) & 0xff;
        }
    }
    return o === out.length ? out : out.subarray(0, o);
}

function base64Encode(bytes) {
    const parts = [];
    const CHUNK = 0x8000 * 3;
    for (let start = 0; start < bytes.length; start += CHUNK) {
        const end = Math.min(start + CHUNK, bytes.length);
        let s = '';
        let i = start;
        for (; i + 2 < end; i += 3) {
            const n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
            s += B64[n >> 18] + B64[(n >> 12) & 63] + B64[(n >> 6) & 63] + B64[n & 63];
        }
        if (i < end) {
            const rem = end - i;
            const n = (bytes[i] << 16) | (rem > 1 ? bytes[i + 1] << 8 : 0);
            s += B64[n >> 18] + B64[(n >> 12) & 63] + (rem > 1 ? B64[(n >> 6) & 63] : '=') + '=';
        }
        parts.push(s);
    }
    return parts.join('');
}

function toBytes(value) {
    if (value === null || value === undefined) return null;
    if (typeof value === 'string') return base64Decode(value);
    if (value instanceof Uint8Array) return value;
    if (value instanceof ArrayBuffer) return new Uint8Array(value);
    if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    throw new Error('Unsupported operand payload type');
}

// ---------------------------------------------------------------------------
// Tokenizer + Pratt parser
// ---------------------------------------------------------------------------
const FUNCTIONS = {
    // name: [minArgs, maxArgs]
    if: [3, 3], abs: [1, 1], sqrt: [1, 1], exp: [1, 1], log: [1, 1], log10: [1, 1],
    log2: [1, 1], pow: [2, 2], min: [2, 16], max: [2, 16], floor: [1, 1], ceil: [1, 1],
    round: [1, 1], clamp: [3, 3], sin: [1, 1], cos: [1, 1], tan: [1, 1], atan: [1, 1],
    atan2: [2, 2]
};
const CONSTANTS = { pi: Math.PI, e: Math.E };
const MAX_EXPRESSION_LENGTH = 10000;
const MAX_AST_NODES = 2000;
const MAX_NESTING = 64; // parser recursion and evaluation depth

class AlgebraError extends Error {
    constructor(message) {
        super(message);
        this.name = 'RasterAlgebraError';
    }
}

function tokenize(src) {
    const tokens = [];
    let i = 0;
    const n = src.length;
    const isDigit = c => c >= '0' && c <= '9';
    const isIdStart = c => /[A-Za-z_]/.test(c);
    const isId = c => /[A-Za-z0-9_]/.test(c);
    while (i < n) {
        const c = src[i];
        if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i++; continue; }
        const pos = i;
        if (isDigit(c) || (c === '.' && isDigit(src[i + 1] || ''))) {
            let j = i;
            while (j < n && isDigit(src[j])) j++;
            if (src[j] === '.') { j++; while (j < n && isDigit(src[j])) j++; }
            if (src[j] === 'e' || src[j] === 'E') {
                let k = j + 1;
                if (src[k] === '+' || src[k] === '-') k++;
                if (isDigit(src[k] || '')) { j = k; while (j < n && isDigit(src[j])) j++; }
            }
            const value = Number(src.slice(i, j));
            if (!Number.isFinite(value)) {
                throw new AlgebraError(`Number '${src.slice(i, j)}' at position ${pos} is out of range`);
            }
            tokens.push({ type: 'num', value, pos });
            i = j;
            continue;
        }
        if (c === DOLLAR) {
            let j = i + 1;
            while (j < n && /[A-Za-z]/.test(src[j])) j++;
            const name = src.slice(i + 1, j);
            if (!name) throw new AlgebraError(`Expected input name after '${DOLLAR}' at position ${pos}`);
            tokens.push({ type: 'input', value: name.toLowerCase(), pos });
            // $a.1 is a band index, like $a[1]
            if (src[j] === '.' && isDigit(src[j + 1] || '')) {
                let k = j + 1;
                while (k < n && isDigit(src[k])) k++;
                tokens.push({ type: '[', pos: j });
                tokens.push({ type: 'num', value: Number(src.slice(j + 1, k)), pos: j + 1 });
                tokens.push({ type: ']', pos: k });
                j = k;
            }
            i = j;
            continue;
        }
        if (isIdStart(c)) {
            let j = i;
            while (j < n && isId(src[j])) j++;
            const word = src.slice(i, j);
            const lower = word.toLowerCase();
            if (lower === 'and' || lower === 'or' || lower === 'not') {
                tokens.push({ type: 'op', value: lower, pos });
            } else {
                tokens.push({ type: 'id', value: word, pos });
            }
            i = j;
            continue;
        }
        const two = src.slice(i, i + 2);
        if (['**', '<=', '>=', '==', '!=', '&&', '||'].includes(two)) {
            const map = { '&&': 'and', '||': 'or', '**': '^' };
            tokens.push({ type: 'op', value: map[two] || two, pos });
            i += 2;
            continue;
        }
        if ('+-*/%^<>!'.includes(c)) {
            tokens.push({ type: 'op', value: c === '!' ? 'not' : c, pos });
            i++;
            continue;
        }
        if ('(),.[]=;'.includes(c)) {
            tokens.push({ type: c, pos });
            i++;
            continue;
        }
        throw new AlgebraError(`Unexpected character '${c}' at position ${pos}`);
    }
    tokens.push({ type: 'eof', pos: n });
    return tokens;
}

const BINARY_PRECEDENCE = {
    or: 1, and: 2,
    '==': 4, '!=': 4, '<': 5, '<=': 5, '>': 5, '>=': 5,
    '+': 6, '-': 6, '*': 7, '/': 7, '%': 7, '^': 9
};
const COMPARISONS = new Set(['==', '!=', '<', '<=', '>', '>=']);
const RIGHT_ASSOC = { '^': true };

function parseExpressionTokens(tokens, start) {
    let p = start;
    let nodes = 0;
    let depth = 0;
    const peek = () => tokens[p];
    const next = () => tokens[p++];
    const expect = (type) => {
        const t = next();
        if (t.type !== type) {
            throw new AlgebraError(`Expected '${type}' at position ${t.pos}` + (t.type === 'eof' ? ' (unexpected end of expression)' : ''));
        }
        return t;
    };
    const node = (obj) => {
        if (++nodes > MAX_AST_NODES) throw new AlgebraError('Expression is too complex');
        return obj;
    };
    const enter = () => {
        if (++depth > MAX_NESTING) throw new AlgebraError(`Expression is nested too deeply (maximum ${MAX_NESTING} levels)`);
    };

    function parsePrimary() {
        const t = next();
        switch (t.type) {
            case 'num':
                return node({ t: 'num', v: t.value });
            case 'input': {
                const ref = { t: 'ref', input: t.value, band: null, pos: t.pos };
                if (peek().type === '.') {
                    next();
                    const b = next();
                    if (b.type !== 'id') {
                        throw new AlgebraError(`Expected band name after '${refName(t.value)}.' at position ${b.pos}`);
                    }
                    ref.band = { name: b.value };
                } else if (peek().type === '[') {
                    next();
                    const b = expect('num');
                    if (!Number.isInteger(b.value) || b.value < 1) {
                        throw new AlgebraError(`Band index must be a positive integer at position ${b.pos}`);
                    }
                    expect(']');
                    ref.band = { index: b.value };
                }
                return node(ref);
            }
            case 'id': {
                const name = t.value.toLowerCase();
                if (peek().type === '(') {
                    if (!Object.prototype.hasOwnProperty.call(FUNCTIONS, name)) {
                        throw new AlgebraError(`Unknown function '${t.value}' at position ${t.pos}. Allowed: ${Object.keys(FUNCTIONS).join(', ')}`);
                    }
                    next();
                    const args = [];
                    if (peek().type !== ')') {
                        args.push(parseBinary(0));
                        while (peek().type === ',') { next(); args.push(parseBinary(0)); }
                    }
                    expect(')');
                    const [lo, hi] = FUNCTIONS[name];
                    if (args.length < lo || args.length > hi) {
                        throw new AlgebraError(`Function '${name}' expects ${lo === hi ? lo : `${lo}-${hi}`} arguments, got ${args.length} (position ${t.pos})`);
                    }
                    return node({ t: 'call', fn: name, args });
                }
                if (Object.prototype.hasOwnProperty.call(CONSTANTS, name)) {
                    return node({ t: 'num', v: CONSTANTS[name] });
                }
                throw new AlgebraError(`Unknown identifier '${t.value}' at position ${t.pos}. Reference inputs as ${refName('a')}, ${refName('b')}, ... and bands as ${refName('a')}.band_1`);
            }
            case '(': {
                const e = parseBinary(0);
                expect(')');
                e.paren = true;
                return e;
            }
            case 'op':
                if (t.value === '-' || t.value === '+' || t.value === 'not') {
                    // unary binds tighter than * but looser than ^  (-x^2 == -(x^2))
                    const operand = parseBinary(t.value === 'not' ? 3 : 8);
                    if (t.value === '+') return operand;
                    return node({ t: 'un', op: t.value === '-' ? 'neg' : 'not', x: operand });
                }
                throw new AlgebraError(`Unexpected operator '${t.value}' at position ${t.pos}`);
            case 'eof':
                throw new AlgebraError('Unexpected end of expression');
            default:
                throw new AlgebraError(`Unexpected '${t.type}' at position ${t.pos}`);
        }
    }

    function parseBinary(minPrec) {
        enter();
        let left = parsePrimary();
        for (;;) {
            const t = peek();
            if (t.type !== 'op' || !(t.value in BINARY_PRECEDENCE)) break;
            const prec = BINARY_PRECEDENCE[t.value];
            if (prec < minPrec) break;
            if (COMPARISONS.has(t.value) && left.t === 'bin' && COMPARISONS.has(left.op) && !left.paren) {
                throw new AlgebraError(`Chained comparison at position ${t.pos} is not supported; combine comparisons with 'and', e.g. (x > 1) and (x < 3)`);
            }
            next();
            const right = parseBinary(RIGHT_ASSOC[t.value] ? prec : prec + 1);
            left = node({ t: 'bin', op: t.value, l: left, r: right });
        }
        depth--;
        return left;
    }

    const ast = parseBinary(0);
    return { ast, end: p };
}

/**
 * Parse one or more expressions separated by ';', each optionally named:
 *   "ndvi = ($a.band_4 - $a.band_3) / ($a.band_4 + $a.band_3); $b - $a"
 * Returns [{ name|null, ast }]
 */
function parse(src) {
    if (typeof src !== 'string' || !src.trim()) throw new AlgebraError('Expression cannot be empty');
    if (src.length > MAX_EXPRESSION_LENGTH) throw new AlgebraError('Expression is too long');
    const tokens = tokenize(src);
    const outputs = [];
    let p = 0;
    for (;;) {
        while (tokens[p].type === ';') p++;
        if (tokens[p].type === 'eof') break;
        let name = null;
        if (tokens[p].type === 'id' && tokens[p + 1].type === '=') {
            name = tokens[p].value;
            p += 2;
        }
        const { ast, end } = parseExpressionTokens(tokens, p);
        p = end;
        const t = tokens[p];
        if (t.type !== ';' && t.type !== 'eof') {
            throw new AlgebraError(`Unexpected token at position ${t.pos}`);
        }
        outputs.push({ name, ast });
    }
    if (!outputs.length) throw new AlgebraError('Expression cannot be empty');
    return outputs;
}

// ---------------------------------------------------------------------------
// Scalar semantics (shared by constant folding and the vectorized evaluator)
// ---------------------------------------------------------------------------
const UNARY_FNS = {
    abs: Math.abs, sqrt: Math.sqrt, exp: Math.exp, log: Math.log, log10: Math.log10,
    log2: Math.log2, floor: Math.floor, ceil: Math.ceil, round: Math.round,
    sin: Math.sin, cos: Math.cos, tan: Math.tan, atan: Math.atan
};

function scalarBinary(op, a, b) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return a / b;
        case '%': return a % b;
        case '^': return Math.pow(a, b);
        case '<': return a < b ? 1 : 0;
        case '<=': return a <= b ? 1 : 0;
        case '>': return a > b ? 1 : 0;
        case '>=': return a >= b ? 1 : 0;
        case '==': return a === b ? 1 : 0;
        case '!=': return a !== b ? 1 : 0;
        case 'and': return (a && b) ? 1 : 0;
        case 'or': return (a || b) ? 1 : 0;
        default: throw new AlgebraError(`Unknown operator ${op}`);
    }
}

function scalarCall(fn, args) {
    if (UNARY_FNS[fn]) return UNARY_FNS[fn](args[0]);
    switch (fn) {
        case 'if': return args[0] ? args[1] : args[2];
        case 'pow': return Math.pow(args[0], args[1]);
        case 'atan2': return Math.atan2(args[0], args[1]);
        case 'clamp': return Math.min(Math.max(args[0], args[1]), args[2]);
        case 'min': return Math.min(...args);
        case 'max': return Math.max(...args);
        default: throw new AlgebraError(`Unknown function ${fn}`);
    }
}

// ---------------------------------------------------------------------------
// Metadata helpers
// ---------------------------------------------------------------------------
const TYPE_SIZES = {
    uint8: 1, int8: 1, uint16: 2, int16: 2, uint32: 4, int32: 4,
    uint64: 8, int64: 8, float16: 2, float32: 4, float64: 8
};
const OUTPUT_TYPES = ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'float32', 'float64'];
const INT_RANGES = {
    uint8: [0, 255], int8: [-128, 127], uint16: [0, 65535], int16: [-32768, 32767],
    uint32: [0, 4294967295], int32: [-2147483648, 2147483647]
};
const IDENT_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const STAT_SUFFIXES = ['count', 'min', 'max', 'sum', 'mean', 'stddev'];

function parseNodata(nodata) {
    if (nodata === null || nodata === undefined) return null;
    if (typeof nodata === 'string') {
        const s = nodata.trim();
        if (s === 'NaN' || s === 'nan') return NaN;
        if (s === 'Infinity' || s === 'inf') return Infinity;
        if (s === '-Infinity' || s === '-inf') return -Infinity;
        const v = Number(s);
        return s === '' || Number.isNaN(v) ? null : v;
    }
    return typeof nodata === 'number' ? nodata : null;
}

function encodeNodata(v) {
    if (v === null || v === undefined) return null;
    if (Number.isNaN(v)) return 'NaN';
    if (v === Infinity) return 'Infinity';
    if (v === -Infinity) return '-Infinity';
    return v;
}

function floatToHalf(value) {
    const f32 = new Float32Array(1);
    const u32 = new Uint32Array(f32.buffer);
    f32[0] = value;
    const x = u32[0];
    const sign = (x >> 16) & 0x8000;
    const exp = ((x >> 23) & 0xff) - 127 + 15;
    const mant = x & 0x7fffff;
    if (((x >> 23) & 0xff) === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // Inf / NaN
    if (exp >= 31) return sign | 0x7c00; // overflow -> Inf
    if (exp <= 0) {
        if (exp < -10) return sign; // underflow -> 0
        const m = (mant | 0x800000) >> (1 - exp);
        return sign | ((m + 0x1000) >> 13);
    }
    return sign | (exp << 10) | ((mant + 0x1000) >> 13);
}

function halfToFloat(bits) {
    const sign = (bits >> 15) ? -1 : 1;
    const exp = (bits >> 10) & 0x1f;
    const frac = bits & 0x3ff;
    if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
    if (exp === 31) return frac ? NaN : sign * Infinity;
    return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/**
 * Nodata sentinels come from JSON as doubles, but pixels are stored in the band
 * type: compare in the band's precision (e.g. -3.4028235e38 as float32).
 */
function normalizeNodata(value, type) {
    if (value === null || Number.isNaN(value) || !Number.isFinite(value)) return value;
    switch (type) {
        case 'float32': return Math.fround(value);
        case 'float16': return halfToFloat(floatToHalf(value));
        default: return value;
    }
}

function parseMeta(meta, label) {
    if (meta === null || meta === undefined) throw new AlgebraError(`Input ${label}: metadata row (block = 0) not found`);
    let m;
    try {
        m = typeof meta === 'string' ? JSON.parse(meta) : meta;
    } catch (e) {
        throw new AlgebraError(`Input ${label}: metadata is not valid JSON`);
    }
    if (m === null) throw new AlgebraError(`Input ${label}: metadata row (block = 0) not found`);
    if (typeof m !== 'object' || Array.isArray(m)) throw new AlgebraError(`Input ${label}: invalid metadata`);
    return m;
}

function compareVersions(a, b) {
    const parse = v => {
        const [core, pre] = String(v).split('-');
        return { nums: core.split('.').map(Number), pre: pre !== undefined };
    };
    const pa = parse(a);
    const pb = parse(b);
    for (let i = 0; i < 3; i++) {
        const d = (pa.nums[i] || 0) - (pb.nums[i] || 0);
        if (d) return d;
    }
    if (pa.pre !== pb.pre) return pa.pre ? -1 : 1; // 0.5.0-rc1 < 0.5.0
    return 0;
}

// Web Mercator pixel coordinates at a given zoom
function lonToPixelX(lon, zoom) {
    return ((lon + 180) / 360) * Math.pow(2, zoom);
}
function latToPixelY(lat, zoom) {
    const clamped = Math.max(-85.05112877980659, Math.min(85.05112877980659, lat));
    const s = Math.sin((clamped * Math.PI) / 180);
    return (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * Math.pow(2, zoom);
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
const OPTION_KEYS = [
    'output_type', 'output_nodata', 'overviews', 'apply_scale_offset',
    'require_version', 'compression', 'compression_level'
];

function parseOptions(options) {
    let opts = options;
    if (typeof options === 'string') {
        if (!options.trim()) return {};
        try {
            opts = JSON.parse(options);
        } catch (e) {
            throw new AlgebraError('options must be a JSON object, e.g. {"output_type": "float32"}');
        }
    }
    if (opts === null || opts === undefined) return {};
    if (typeof opts !== 'object' || Array.isArray(opts)) {
        throw new AlgebraError('options must be a JSON object, e.g. {"output_type": "float32"}');
    }
    for (const key of Object.keys(opts)) {
        if (!OPTION_KEYS.includes(key)) {
            throw new AlgebraError(`Unknown option '${key}'. Allowed: ${OPTION_KEYS.join(', ')}`);
        }
    }
    if (opts.overviews !== undefined && !['evaluate', 'none'].includes(opts.overviews)) {
        throw new AlgebraError(`Invalid overviews '${opts.overviews}'. Allowed: evaluate, none`);
    }
    if (opts.compression !== undefined && !['gzip', 'none'].includes(opts.compression)) {
        throw new AlgebraError(`Invalid compression '${opts.compression}'. Allowed: gzip, none`);
    }
    if (opts.compression_level !== undefined &&
        !(Number.isInteger(opts.compression_level) && opts.compression_level >= 1 && opts.compression_level <= 9)) {
        throw new AlgebraError('compression_level must be an integer between 1 and 9');
    }
    if (opts.apply_scale_offset !== undefined && typeof opts.apply_scale_offset !== 'boolean') {
        throw new AlgebraError('apply_scale_offset must be true or false');
    }
    if (opts.require_version !== undefined && !/^\d+\.\d+\.\d+$/.test(String(opts.require_version))) {
        throw new AlgebraError('require_version must be a version like "0.5.0"');
    }
    return opts;
}

// ---------------------------------------------------------------------------
// Planner: resolve references against input metadata, validate grids
// ---------------------------------------------------------------------------
function intersectBounds(list) {
    return list.reduce((acc, b) => (acc
        ? [Math.max(acc[0], b[0]), Math.max(acc[1], b[1]), Math.min(acc[2], b[2]), Math.min(acc[3], b[3])]
        : b.slice()), null);
}

function unionBounds(list) {
    return list.reduce((acc, b) => (acc
        ? [Math.min(acc[0], b[0]), Math.min(acc[1], b[1]), Math.max(acc[2], b[2]), Math.max(acc[3], b[3])]
        : b.slice()), null);
}

function astDepth(n) {
    if (n.t === 'num' || n.t === 'op') return 1;
    if (n.t === 'un') return 1 + astDepth(n.x);
    if (n.t === 'bin') return 1 + Math.max(astDepth(n.l), astDepth(n.r));
    return 1 + Math.max(...n.args.map(astDepth));
}

/**
 * @param {string} expression
 * @param {Array<string|object>} metadatas  one per input, in $a,$b.. order
 * @param {object|string} options
 *   output_type:        'float32' (default) | float64 | int32 | ...
 *   output_nodata:      number | 'NaN' (default NaN for float, type limit for int)
 *   overviews:          'evaluate' (default) | 'none'
 *   apply_scale_offset: false (default: expressions see stored DN values)
 *   require_version:    minimum RaQuet version accepted (default '0.3.0')
 *   compression:        'gzip' (default) | 'none'
 *   compression_level:  1-9 (default 1: ~20% less CPU than 6 for <1% larger tiles)
 */
function plan(expression, metadatas, options) {
    const opts = parseOptions(options);
    if (!Array.isArray(metadatas) || metadatas.length === 0) {
        throw new AlgebraError('At least one input raster is required');
    }
    if (metadatas.length > 26) throw new AlgebraError('At most 26 input rasters are supported');

    const metas = metadatas.map((m, i) => parseMeta(m, inputRef(i)));
    const minVersion = opts.require_version || '0.3.0';

    metas.forEach((m, i) => {
        const label = inputRef(i);
        // RaQuet < 0.5 has no file_format field; legacy CARTO rasters use block_resolution
        const isRaquet = m.file_format === 'raquet' ||
            (m.file_format === undefined && m.tiling && typeof m.tiling === 'object' && m.block_resolution === undefined);
        if (!isRaquet || !m.tiling) {
            throw new AlgebraError(`Input ${label} is not a RaQuet raster (legacy CARTO raster format is not supported; re-import it to produce a RaQuet v0.5.0 table)`);
        }
        if (!m.version || compareVersions(m.version, minVersion) < 0) {
            throw new AlgebraError(`Input ${label} is RaQuet ${m.version || 'unknown'}; version ${minVersion} or later is required`);
        }
        if (m.time) {
            throw new AlgebraError(`Input ${label} has a time dimension; multi-temporal raster algebra is not supported`);
        }
        if ((m.tiling.scheme || 'quadbin') !== 'quadbin') {
            throw new AlgebraError(`Input ${label} uses unsupported tiling scheme '${m.tiling.scheme}'`);
        }
        if (!Array.isArray(m.bands) || m.bands.length === 0) {
            throw new AlgebraError(`Input ${label} has no bands in its metadata`);
        }
        if (m.compression === 'webp') {
            throw new AlgebraError(`Input ${label} uses WebP compression, which raster algebra cannot decode in SQL UDFs`);
        }
    });

    // --- Grid alignment ---------------------------------------------------
    // QUADBIN tiles at the same zoom are pixel-aligned by construction, so
    // inputs align when block size and native zoom match. Extents may differ.
    const g0 = metas[0].tiling;
    metas.forEach((m, i) => {
        if (i === 0) return;
        const g = m.tiling;
        const problems = [];
        if (g.block_width !== g0.block_width || g.block_height !== g0.block_height) {
            problems.push(`block size ${g.block_width}x${g.block_height} vs ${g0.block_width}x${g0.block_height}`);
        }
        if (g.max_zoom !== g0.max_zoom) {
            problems.push(`native zoom (max_zoom) ${g.max_zoom} vs ${g0.max_zoom}`);
        }
        if (problems.length) {
            throw new AlgebraError(
                `Inputs ${inputRef(0)} and ${inputRef(i)} are not on the same grid: ${problems.join('; ')}. ` +
                'Re-import them with the same block size and resolution (automatic resampling is not supported).'
            );
        }
    });

    const maxZoom = g0.max_zoom;
    const minZoomCommon = Math.max(...metas.map(m => (m.tiling.min_zoom ?? m.tiling.max_zoom)));
    const overviews = opts.overviews || 'evaluate';
    const zoomRange = [overviews === 'none' ? maxZoom : minZoomCommon, maxZoom];

    // --- Output type & nodata --------------------------------------------
    const outType = String(opts.output_type || 'float32').toLowerCase();
    if (!OUTPUT_TYPES.includes(outType)) {
        throw new AlgebraError(`Invalid output_type '${opts.output_type}'. Allowed: ${OUTPUT_TYPES.join(', ')}`);
    }
    let outNodata;
    if (opts.output_nodata !== undefined && opts.output_nodata !== null) {
        outNodata = parseNodata(opts.output_nodata);
        if (outNodata === null) throw new AlgebraError(`Invalid output_nodata '${opts.output_nodata}'`);
    } else {
        outNodata = INT_RANGES[outType] ? (outType.startsWith('u') ? INT_RANGES[outType][1] : INT_RANGES[outType][0]) : NaN;
    }
    if (INT_RANGES[outType] && (!Number.isInteger(outNodata) || outNodata < INT_RANGES[outType][0] || outNodata > INT_RANGES[outType][1])) {
        throw new AlgebraError(`output_nodata ${opts.output_nodata} is not representable as ${outType}`);
    }
    // Store the sentinel exactly as it will be written (0.1 -> fround(0.1) for float32)
    outNodata = normalizeNodata(outNodata, outType);

    // --- Parse & resolve references -------------------------------------
    const outputsParsed = parse(expression);
    const operands = [];
    const operandKey = new Map();

    function resolveRef(ref) {
        const idx = ref.input.charCodeAt(0) - 97;
        if (ref.input.length !== 1 || idx < 0 || idx >= metas.length) {
            throw new AlgebraError(
                `Expression references ${refName(ref.input)} but only ${metas.length} input(s) were provided (${metas.map((_, i) => inputRef(i)).join(', ')})`
            );
        }
        const m = metas[idx];
        const label = inputRef(idx);
        let bandIndex;
        if (ref.band === null) {
            if (m.bands.length !== 1) {
                throw new AlgebraError(`${label} has ${m.bands.length} bands; specify one, e.g. ${label}.${m.bands[0].name}`);
            }
            bandIndex = 0;
        } else if (ref.band.name !== undefined) {
            bandIndex = m.bands.findIndex(b => b.name === ref.band.name);
            if (bandIndex < 0) {
                bandIndex = m.bands.findIndex(b => String(b.name).toLowerCase() === ref.band.name.toLowerCase());
            }
            if (bandIndex < 0) {
                throw new AlgebraError(`Band '${ref.band.name}' not found in ${label}. Available bands: ${m.bands.map(b => b.name).join(', ')}`);
            }
        } else {
            bandIndex = ref.band.index - 1;
            if (bandIndex < 0 || bandIndex >= m.bands.length) {
                throw new AlgebraError(`Band ${ref.band.index} not found in ${label}, which has ${m.bands.length} band(s)`);
            }
        }
        const key = `${idx}:${bandIndex}`;
        if (!operandKey.has(key)) {
            const band = m.bands[bandIndex];
            const layout = m.band_layout || 'sequential';
            const column = layout === 'interleaved' ? 'pixels' : band.name;
            if (!IDENT_RE.test(column)) {
                throw new AlgebraError(`Band column name '${column}' in ${label} is not a safe SQL identifier`);
            }
            const type = String(band.type).toLowerCase();
            if (!TYPE_SIZES[type]) throw new AlgebraError(`Unsupported band type '${band.type}' in ${label}`);
            operandKey.set(key, operands.length);
            operands.push({
                input: idx,
                band: bandIndex,
                column,
                type,
                nodata: encodeNodata(normalizeNodata(parseNodata(band.nodata), type)),
                layout,
                band_count: m.bands.length,
                compression: m.compression || null,
                scale: opts.apply_scale_offset && band.scale != null ? band.scale : null,
                offset: opts.apply_scale_offset && band.offset != null ? band.offset : null
            });
        }
        return operandKey.get(key);
    }

    // Lower to the evaluation program, folding constant subtrees
    function lower(n) {
        switch (n.t) {
            case 'num': return { t: 'num', v: n.v };
            case 'ref': return { t: 'op', i: resolveRef(n) };
            case 'un': {
                const x = lower(n.x);
                if (x.t === 'num') {
                    const v = n.op === 'neg' ? -x.v : (x.v ? 0 : 1);
                    if (Number.isFinite(v)) return { t: 'num', v };
                }
                return { t: 'un', op: n.op, x };
            }
            case 'bin': {
                const l = lower(n.l);
                const r = lower(n.r);
                if (l.t === 'num' && r.t === 'num') {
                    const v = scalarBinary(n.op, l.v, r.v);
                    if (Number.isFinite(v)) return { t: 'num', v };
                }
                return { t: 'bin', op: n.op, l, r };
            }
            case 'call': {
                const args = n.args.map(lower);
                if (args.every(a => a.t === 'num')) {
                    const v = scalarCall(n.fn, args.map(a => a.v));
                    if (Number.isFinite(v)) return { t: 'num', v };
                }
                return { t: 'call', fn: n.fn, args };
            }
            default: throw new AlgebraError('Invalid expression');
        }
    }

    const usedNames = new Set();
    const outputs = outputsParsed.map((o, k) => {
        const name = o.name || `band_${k + 1}`;
        const lowerName = name.toLowerCase();
        if (!IDENT_RE.test(name)) throw new AlgebraError(`Invalid output band name '${name}'`);
        if (STAT_SUFFIXES.some(s => lowerName.endsWith(`_${s}`)) || lowerName === 'block' || lowerName === 'metadata') {
            throw new AlgebraError(`Output band name '${name}' clashes with a reserved column name`);
        }
        if (usedNames.has(lowerName)) throw new AlgebraError(`Duplicate output band name '${name}' (names are case-insensitive)`);
        usedNames.add(lowerName);
        const program = lower(o.ast);
        if (astDepth(program) > MAX_NESTING) {
            throw new AlgebraError(`Expression '${name}' is nested too deeply (maximum ${MAX_NESTING} levels)`);
        }
        const refs = new Set();
        (function collect(n) {
            if (n.t === 'op') refs.add(n.i);
            if (n.x) collect(n.x);
            if (n.l) { collect(n.l); collect(n.r); }
            if (n.args) n.args.forEach(collect);
        })(program);
        const opList = [...refs].sort((x, y) => x - y);
        const inputs = [...new Set(opList.map(i => operands[i].input))].sort((x, y) => x - y);
        return { name, program, operands: opList, inputs };
    });

    // Suffix collisions between outputs (e.g. "x" and "x_count" are both reserved above;
    // here "x" vs a derived column of another band)
    const derived = new Set();
    outputs.forEach(o => STAT_SUFFIXES.forEach(s => derived.add(`${o.name.toLowerCase()}_${s}`)));
    outputs.forEach(o => {
        if (derived.has(o.name.toLowerCase())) {
            throw new AlgebraError(`Output band name '${o.name}' clashes with a statistics column of another band`);
        }
    });

    const inputsUsed = [...new Set(operands.map(o => o.input))].sort((x, y) => x - y);
    if (inputsUsed.length !== metas.length) {
        const unused = metas.map((_, i) => i).filter(i => !inputsUsed.includes(i)).map(inputRef);
        throw new AlgebraError(`Input(s) ${unused.join(', ')} are not referenced by the expression`);
    }

    // --- Coverage ----------------------------------------------------------
    // Each output covers the blocks where all the inputs it references exist.
    // When every output references every input, an inner join computes exactly
    // that; otherwise blocks from any input are evaluated (outer join).
    const inputBounds = metas.map(m => m.bounds || [-180, -85.05112877980659, 180, 85.05112877980659]);
    const outputBounds = outputs.map(o => {
        const b = intersectBounds(o.inputs.map(i => inputBounds[i]));
        if (b[0] >= b[2] || b[1] >= b[3]) {
            throw new AlgebraError(o.inputs.length > 1
                ? `The rasters referenced by output '${o.name}' (${o.inputs.map(inputRef).join(', ')}) do not overlap`
                : `Input ${inputRef(o.inputs[0])} has an empty extent`);
        }
        return b;
    });
    const join = outputs.every(o => o.inputs.length === metas.length) ? 'inner' : 'outer';
    const bounds = unionBounds(outputBounds);

    const compression = opts.compression === 'none' ? null : 'gzip';

    return {
        version: 1,
        expression,
        block_width: g0.block_width,
        block_height: g0.block_height,
        max_zoom: maxZoom,
        zoom_range: zoomRange,
        overviews,
        bounds,
        join,
        operands,
        outputs,
        output: {
            type: outType,
            nodata: encodeNodata(outNodata),
            compression,
            compression_level: opts.compression_level || 1
        },
        apply_scale_offset: !!opts.apply_scale_offset,
        num_inputs: metas.length
    };
}

// ---------------------------------------------------------------------------
// Tile decoding
// ---------------------------------------------------------------------------
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

function readTypedArray(bytes, type, count) {
    const size = TYPE_SIZES[type];
    const needed = count * size;
    if (bytes.length < needed) {
        throw new AlgebraError(`Tile payload too short: expected ${needed} bytes of ${type}, got ${bytes.length}`);
    }
    // Copy into an aligned buffer
    const buf = new Uint8Array(needed);
    buf.set(bytes.subarray(0, needed));
    if (LITTLE_ENDIAN) {
        switch (type) {
            case 'uint8': return buf;
            case 'int8': return new Int8Array(buf.buffer);
            case 'uint16': return new Uint16Array(buf.buffer);
            case 'int16': return new Int16Array(buf.buffer);
            case 'uint32': return new Uint32Array(buf.buffer);
            case 'int32': return new Int32Array(buf.buffer);
            case 'float32': return new Float32Array(buf.buffer);
            case 'float64': return new Float64Array(buf.buffer);
        }
    }
    const view = new DataView(buf.buffer);
    const out = new Float64Array(count);
    for (let i = 0; i < count; i++) {
        const o = i * size;
        switch (type) {
            case 'uint8': out[i] = view.getUint8(o); break;
            case 'int8': out[i] = view.getInt8(o); break;
            case 'uint16': out[i] = view.getUint16(o, true); break;
            case 'int16': out[i] = view.getInt16(o, true); break;
            case 'uint32': out[i] = view.getUint32(o, true); break;
            case 'int32': out[i] = view.getInt32(o, true); break;
            case 'float32': out[i] = view.getFloat32(o, true); break;
            case 'float64': out[i] = view.getFloat64(o, true); break;
            case 'float16': out[i] = halfToFloat(view.getUint16(o, true)); break;
            case 'uint64': out[i] = view.getUint32(o + 4, true) * 4294967296 + view.getUint32(o, true); break;
            case 'int64': out[i] = view.getInt32(o + 4, true) * 4294967296 + view.getUint32(o, true); break;
        }
    }
    return out;
}

/**
 * Decode one operand tile into Float64 values + validity mask.
 * Returns null for a missing/empty payload (treated as all-nodata).
 */
function decodeOperand(payload, op, pixelCount, cache) {
    let bytes = toBytes(payload);
    if (!bytes || bytes.length === 0) return null;

    // Interleaved tiles referenced by several operands are decoded once
    const cacheKey = op.layout === 'interleaved' ? `${op.input}` : null;
    let raw = cacheKey && cache ? cache.get(cacheKey) : undefined;
    const bandCount = op.layout === 'interleaved' ? op.band_count : 1;

    if (raw === undefined) {
        if (op.compression === 'jpeg') {
            const jpeg = (typeof jpegDecoderLib !== 'undefined') ? jpegDecoderLib : null; // eslint-disable-line no-undef
            if (!jpeg) throw new AlgebraError('JPEG-compressed rasters are not supported by raster algebra');
            const img = jpeg.decode(bytes);
            raw = { values: img.data, channels: img.channels };
        } else {
            if (op.compression === 'gzip') bytes = inflate(bytes);
            raw = { values: readTypedArray(bytes, op.type, pixelCount * bandCount), channels: bandCount };
        }
        if (cacheKey && cache) cache.set(cacheKey, raw);
    }

    const src = raw.values;
    const stride = raw.channels;
    const offset = op.layout === 'interleaved' ? op.band : 0;
    if (offset >= stride) {
        throw new AlgebraError(`Tile of ${inputRef(op.input)} has ${stride} channel(s) but band ${op.band + 1} was requested`);
    }
    if (src.length < pixelCount * stride) {
        throw new AlgebraError(`Tile of ${inputRef(op.input)} has ${src.length} values; expected ${pixelCount * stride}`);
    }
    const values = new Float64Array(pixelCount);
    const valid = new Uint8Array(pixelCount);
    const nodata = parseNodata(op.nodata);
    const hasNodata = nodata !== null && !Number.isNaN(nodata);
    const scale = op.scale == null ? 1 : op.scale;
    const add = op.offset == null ? 0 : op.offset;
    const scaled = scale !== 1 || add !== 0;
    for (let i = 0, j = offset; i < pixelCount; i++, j += stride) {
        const v = src[j];
        if (v !== v || (hasNodata && v === nodata)) continue; // NaN or nodata
        values[i] = scaled ? v * scale + add : v;
        valid[i] = 1;
    }
    return { values, valid };
}

// ---------------------------------------------------------------------------
// Vectorized evaluation. Intermediate results are reused in place; operand
// and constant arrays are read-only.
// ---------------------------------------------------------------------------
function evaluate(program, operandValues, n) {
    const temps = new Set(); // intermediate results (writable)
    const pool = [];         // intermediates no longer referenced
    const constants = new Map();
    const constant = v => {
        let a = constants.get(v);
        if (!a) {
            a = new Float64Array(n).fill(v);
            constants.set(v, a);
        }
        return a;
    };
    // Output buffer for a node: reuse an intermediate input (element-wise ops
    // read index i before writing it), else a pooled one, else allocate.
    const target = (...xs) => {
        for (const x of xs) if (temps.has(x)) return x;
        const a = pool.pop() || new Float64Array(n);
        temps.add(a);
        return a;
    };
    const release = (xs, out) => {
        for (const x of xs) {
            if (x !== out && temps.has(x)) {
                temps.delete(x);
                pool.push(x);
            }
        }
    };

    function ev(node) {
        switch (node.t) {
            case 'num':
                return constant(node.v);
            case 'op':
                return operandValues[node.i];
            case 'un': {
                const x = ev(node.x);
                const out = target(x);
                if (node.op === 'neg') for (let i = 0; i < n; i++) out[i] = -x[i];
                else for (let i = 0; i < n; i++) out[i] = x[i] ? 0 : 1;
                release([x], out);
                return out;
            }
            case 'bin': {
                const l = ev(node.l);
                const r = ev(node.r);
                const out = target(l, r);
                switch (node.op) {
                    case '+': for (let i = 0; i < n; i++) out[i] = l[i] + r[i]; break;
                    case '-': for (let i = 0; i < n; i++) out[i] = l[i] - r[i]; break;
                    case '*': for (let i = 0; i < n; i++) out[i] = l[i] * r[i]; break;
                    case '/': for (let i = 0; i < n; i++) out[i] = l[i] / r[i]; break;
                    case '%': for (let i = 0; i < n; i++) out[i] = l[i] % r[i]; break;
                    case '^': for (let i = 0; i < n; i++) out[i] = Math.pow(l[i], r[i]); break;
                    case '<': for (let i = 0; i < n; i++) out[i] = l[i] < r[i] ? 1 : 0; break;
                    case '<=': for (let i = 0; i < n; i++) out[i] = l[i] <= r[i] ? 1 : 0; break;
                    case '>': for (let i = 0; i < n; i++) out[i] = l[i] > r[i] ? 1 : 0; break;
                    case '>=': for (let i = 0; i < n; i++) out[i] = l[i] >= r[i] ? 1 : 0; break;
                    case '==': for (let i = 0; i < n; i++) out[i] = l[i] === r[i] ? 1 : 0; break;
                    case '!=': for (let i = 0; i < n; i++) out[i] = l[i] !== r[i] ? 1 : 0; break;
                    case 'and': for (let i = 0; i < n; i++) out[i] = (l[i] && r[i]) ? 1 : 0; break;
                    case 'or': for (let i = 0; i < n; i++) out[i] = (l[i] || r[i]) ? 1 : 0; break;
                    default: throw new AlgebraError(`Unknown operator ${node.op}`);
                }
                release([l, r], out);
                return out;
            }
            case 'call': {
                const args = node.args.map(ev);
                // min/max read later arguments after writing: only the first may be reused
                const out = (node.fn === 'min' || node.fn === 'max') ? target(args[0]) : target(...args);
                const f = UNARY_FNS[node.fn];
                if (f) {
                    const x = args[0];
                    for (let i = 0; i < n; i++) out[i] = f(x[i]);
                } else {
                    switch (node.fn) {
                        case 'if': {
                            const [c, a, b] = args;
                            for (let i = 0; i < n; i++) out[i] = c[i] ? a[i] : b[i];
                            break;
                        }
                        case 'pow': {
                            const [a, b] = args;
                            for (let i = 0; i < n; i++) out[i] = Math.pow(a[i], b[i]);
                            break;
                        }
                        case 'atan2': {
                            const [a, b] = args;
                            for (let i = 0; i < n; i++) out[i] = Math.atan2(a[i], b[i]);
                            break;
                        }
                        case 'clamp': {
                            const [x, lo, hi] = args;
                            for (let i = 0; i < n; i++) out[i] = Math.min(Math.max(x[i], lo[i]), hi[i]);
                            break;
                        }
                        case 'min':
                        case 'max': {
                            const pick = node.fn === 'min' ? Math.min : Math.max;
                            const first = args[0];
                            for (let i = 0; i < n; i++) out[i] = first[i];
                            for (let k = 1; k < args.length; k++) {
                                const a = args[k];
                                for (let i = 0; i < n; i++) out[i] = pick(out[i], a[i]);
                            }
                            break;
                        }
                        default: throw new AlgebraError(`Unknown function ${node.fn}`);
                    }
                }
                release(args, out);
                return out;
            }
            default:
                throw new AlgebraError('Invalid program');
        }
    }
    return ev(program);
}

// ---------------------------------------------------------------------------
// Encoding + statistics
// ---------------------------------------------------------------------------
const TYPED_CTORS = {
    uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int16: Int16Array,
    uint32: Uint32Array, int32: Int32Array, float32: Float32Array, float64: Float64Array
};

function encodeBand(result, valid, output) {
    const n = valid.length;
    const Ctor = TYPED_CTORS[output.type];
    const typed = new Ctor(n);
    const ok = new Uint8Array(n);
    const nodata = parseNodata(output.nodata);
    const fill = nodata === null ? NaN : nodata;
    const hasNodata = nodata !== null && !Number.isNaN(nodata);
    const range = INT_RANGES[output.type];
    const isFloat32 = output.type === 'float32';
    let count = 0, min = Infinity, max = -Infinity, sum = 0;
    for (let i = 0; i < n; i++) {
        let v = result ? result[i] : NaN;
        let good = valid[i] === 1 && Number.isFinite(v);
        if (good && range) {
            v = Math.round(v);
            if (v < range[0] || v > range[1]) good = false;
        }
        if (good && isFloat32) {
            v = Math.fround(v);
            if (!Number.isFinite(v)) good = false; // overflow to +-Inf
        }
        if (good && hasNodata && v === nodata) good = false; // would collide with the sentinel
        if (!good) {
            typed[i] = fill;
            continue;
        }
        typed[i] = v;
        ok[i] = 1;
        count++;
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
    }
    // Second pass for a numerically stable standard deviation
    const mean = count ? sum / count : null;
    let m2 = 0;
    if (count) {
        for (let i = 0; i < n; i++) {
            if (ok[i]) {
                const d = typed[i] - mean;
                m2 += d * d;
            }
        }
    }
    let bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
    if (!LITTLE_ENDIAN && TYPE_SIZES[output.type] > 1) {
        const le = new Uint8Array(bytes.length);
        const view = new DataView(le.buffer);
        const size = TYPE_SIZES[output.type];
        for (let i = 0; i < n; i++) {
            const o = i * size;
            switch (output.type) {
                case 'uint16': view.setUint16(o, typed[i], true); break;
                case 'int16': view.setInt16(o, typed[i], true); break;
                case 'uint32': view.setUint32(o, typed[i], true); break;
                case 'int32': view.setInt32(o, typed[i], true); break;
                case 'float32': view.setFloat32(o, typed[i], true); break;
                case 'float64': view.setFloat64(o, typed[i], true); break;
            }
        }
        bytes = le;
    }
    // Always honour the declared compression, also for all-nodata bands
    if (output.compression === 'gzip') {
        bytes = gzip(bytes, { level: output.compression_level || 1 });
    }
    return {
        data: bytes,
        count,
        min: count ? min : null,
        max: count ? max : null,
        sum: count ? sum : null,
        mean,
        stddev: count ? Math.sqrt(m2 / count) : null
    };
}

// ---------------------------------------------------------------------------
// Block evaluation (the per-row UDF entry point)
// ---------------------------------------------------------------------------
const PLAN_CACHE = new Map();

function decodeBase64Text(text) {
    const bytes = base64Decode(text);
    let out = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
        out += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return decodeURIComponent(escape(out)); // utf-8
}

function loadPlan(planArg) {
    if (typeof planArg === 'object' && planArg !== null) return planArg;
    let p = PLAN_CACHE.get(planArg);
    if (!p) {
        // Plans travel either as JSON or as base64 JSON (trivially safe SQL literals)
        const text = planArg[0] === '{' ? planArg : decodeBase64Text(planArg);
        p = JSON.parse(text);
        if (PLAN_CACHE.size > 16) PLAN_CACHE.clear();
        PLAN_CACHE.set(planArg, p);
    }
    return p;
}

/**
 * Evaluate all output bands for one block.
 * @param {string|object} planArg  plan (JSON or base64 JSON)
 * @param {Array} payloads         one tile payload per plan.operands entry
 * @param {object} [opts]          { encoding: 'base64' | 'bytes' }
 * @returns {Array|null} one {data,count,min,max,sum,mean,stddev} per output
 *          band, or null when every output pixel is nodata (block dropped)
 */
function evalBlock(planArg, payloads, opts) {
    const p = loadPlan(planArg);
    const n = p.block_width * p.block_height;
    if (!Array.isArray(payloads) || payloads.length !== p.operands.length) {
        throw new AlgebraError(`Expected ${p.operands.length} operand payloads, got ${payloads ? payloads.length : 0}`);
    }
    const cache = new Map();
    const decoded = p.operands.map((op, i) => decodeOperand(payloads[i], op, n, cache));
    const values = decoded.map(d => (d ? d.values : null));

    const results = [];
    let anyValid = false;
    for (const out of p.outputs) {
        const valid = new Uint8Array(n);
        let result = null;
        if (out.operands.every(i => decoded[i])) {
            valid.fill(1);
            for (const i of out.operands) {
                const v = decoded[i].valid;
                for (let k = 0; k < n; k++) valid[k] &= v[k];
            }
            result = evaluate(out.program, values, n);
        }
        const enc = encodeBand(result, valid, p.output);
        if (enc.count > 0) anyValid = true;
        results.push(enc);
    }
    if (!anyValid) return null;
    const asBase64 = !opts || opts.encoding !== 'bytes';
    return results.map(r => ({
        data: asBase64 ? base64Encode(r.data) : r.data,
        count: r.count,
        min: r.min,
        max: r.max,
        sum: r.sum,
        mean: r.mean,
        stddev: r.stddev
    }));
}

// ---------------------------------------------------------------------------
// Output metadata (block = 0 row)
// ---------------------------------------------------------------------------
/**
 * @param {string|object} planArg
 * @param {string|object} stats  { num_blocks, min_zoom, max_zoom,
 *        bands: [{count, min, max, sum, m2}] }  (native-zoom aggregates;
 *        m2 = sum of squared deviations from the band mean)
 * @param {object} [extra]  { inputs: [table names], created_by }
 */
function buildMetadata(planArg, stats, extra) {
    const p = loadPlan(planArg);
    const s = (typeof stats === 'string' ? JSON.parse(stats) : stats) || {};
    const pixelZoom = p.max_zoom + Math.round(Math.log2(p.block_width));
    const [w, south, e, north] = p.bounds;
    const width = Math.max(1, Math.round(lonToPixelX(e, pixelZoom) - lonToPixelX(w, pixelZoom)));
    const height = Math.max(1, Math.round(latToPixelY(south, pixelZoom) - latToPixelY(north, pixelZoom)));
    const totalPixels = width * height;

    const bands = p.outputs.map((o, k) => {
        const b = (s.bands && s.bands[k]) || {};
        const count = Number(b.count) || 0;
        return {
            name: o.name,
            description: o.name,
            type: p.output.type,
            nodata: p.output.nodata,
            unit: null,
            scale: null,
            offset: null,
            colorinterp: 'undefined',
            colortable: null,
            STATISTICS_MINIMUM: count ? Number(b.min) : null,
            STATISTICS_MAXIMUM: count ? Number(b.max) : null,
            STATISTICS_MEAN: count ? Number(b.sum) / count : null,
            STATISTICS_STDDEV: count ? Math.sqrt(Math.max(0, Number(b.m2)) / count) : null,
            STATISTICS_VALID_PERCENT: totalPixels ? Math.min(100, (100 * count) / totalPixels) : null
        };
    });

    return JSON.stringify({
        file_format: 'raquet',
        version: '0.5.0',
        width,
        height,
        crs: 'EPSG:3857',
        bounds: p.bounds,
        bounds_crs: 'EPSG:4326',
        compression: p.output.compression,
        band_layout: 'sequential',
        tiling: {
            scheme: 'quadbin',
            block_width: p.block_width,
            block_height: p.block_height,
            min_zoom: s.min_zoom != null ? Number(s.min_zoom) : p.zoom_range[0],
            max_zoom: p.max_zoom,
            pixel_zoom: pixelZoom,
            num_blocks: Number(s.num_blocks) || 0
        },
        tile_statistics: true,
        tile_statistics_columns: STAT_SUFFIXES,
        bands,
        processing: {
            created_by: (extra && extra.created_by) || 'RASTER_ALGEBRA',
            created_at: new Date().toISOString(),
            operation: 'raster_algebra',
            expression: p.expression,
            inputs: (extra && extra.inputs) || undefined,
            overviews: p.overviews === 'evaluate' ? 'expression evaluated per zoom level' : 'none',
            apply_scale_offset: p.apply_scale_offset
        }
    });
}

// ---------------------------------------------------------------------------
// SQL generation — shared helpers
// ---------------------------------------------------------------------------
function planToBase64(p) {
    const json = typeof p === 'string' ? p : JSON.stringify(p);
    const utf8 = unescape(encodeURIComponent(json));
    const bytes = new Uint8Array(utf8.length);
    for (let i = 0; i < utf8.length; i++) bytes[i] = utf8.charCodeAt(i);
    return base64Encode(bytes);
}

/** True when two (possibly partially qualified) names may denote the same table */
function sameTable(a, b) {
    const pa = a.split('.').map(x => x.replace(/["`]/g, '').toLowerCase());
    const pb = b.split('.').map(x => x.replace(/["`]/g, '').toLowerCase());
    const k = Math.min(pa.length, pb.length);
    return pa.slice(-k).join('.') === pb.slice(-k).join('.');
}

// ---------------------------------------------------------------------------
// BigQuery
// ---------------------------------------------------------------------------
const BQ_TABLE_RE = /^`?[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+){1,2}`?$/;
const BQ_DATASET_RE = /^`?[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)?`?$/;

function bigQueryName(name, re, what) {
    if (typeof name !== 'string' || !re.test(name.trim())) {
        throw new AlgebraError(`Invalid ${what} name '${name}'`);
    }
    return name.trim().replace(/`/g, '');
}

/**
 * Build the BigQuery CREATE TABLE ... AS SELECT for a plan. Fails if the
 * output table already exists.
 * @param {object|string} planArg
 * @param {string[]} inputs     input table names ($a, $b, ...)
 * @param {string} outputTable
 * @param {string} dataset      dataset hosting the UDFs, e.g. 'cartobq.raquet' or 'carto'
 */
function buildBigQuerySql(planArg, inputs, outputTable, dataset) {
    const p = loadPlan(planArg);
    const out = bigQueryName(outputTable, BQ_TABLE_RE, 'output table');
    const tables = inputs.map((t, i) => bigQueryName(t, BQ_TABLE_RE, `input ${inputRef(i)} table`));
    if (tables.length !== p.num_inputs) throw new AlgebraError('Input count does not match plan');
    if (tables.some(t => sameTable(t, out))) throw new AlgebraError('Output table must differ from the input tables');
    const ds = bigQueryName(dataset, BQ_DATASET_RE, 'UDF dataset');
    const b64 = planToBase64(p);
    const [zmin, zmax] = p.zoom_range;

    const ctes = tables.map((t, i) => {
        const cols = [...new Set(p.operands.filter(o => o.input === i).map(o => o.column))];
        return `i${i} AS (\n  SELECT block, ${cols.map(c => `\`${c}\``).join(', ')}\n  FROM \`${t}\`\n  WHERE block != 0 AND ((block >> 52) & 31) BETWEEN ${zmin} AND ${zmax}\n)`;
    });
    let from;
    if (p.join === 'inner') {
        from = tables.map((_, i) => (i === 0 ? 'i0' : `JOIN i${i} USING (block)`)).join('\n  ');
    } else {
        ctes.push(`blocks AS (\n  ${tables.map((_, i) => `SELECT block FROM i${i}`).join('\n  UNION DISTINCT\n  ')}\n)`);
        from = ['blocks', ...tables.map((_, i) => `LEFT JOIN i${i} USING (block)`)].join('\n  ');
    }
    const payloads = p.operands.map(o => `IFNULL(i${o.input}.\`${o.column}\`, b'')`).join(', ');
    const cols = p.outputs.map((o, k) => [
        `r[OFFSET(${k})].data AS \`${o.name}\``,
        `CAST(r[OFFSET(${k})].count AS INT64) AS \`${o.name}_count\``,
        `r[OFFSET(${k})].min AS \`${o.name}_min\``,
        `r[OFFSET(${k})].max AS \`${o.name}_max\``,
        `r[OFFSET(${k})].sum AS \`${o.name}_sum\``,
        `r[OFFSET(${k})].mean AS \`${o.name}_mean\``,
        `r[OFFSET(${k})].stddev AS \`${o.name}_stddev\``
    ].join(',\n  ')).join(',\n  ');

    return `CREATE TABLE \`${out}\`
CLUSTER BY block AS
WITH ${ctes.join(',\n')},
computed AS (
  SELECT block, \`${ds}.__RASTER_ALGEBRA_BLOCK\`('${b64}', [${payloads}]) AS r
  FROM ${from}
)
SELECT
  block,
  CAST(NULL AS STRING) AS metadata,
  ${cols}
FROM computed
WHERE ARRAY_LENGTH(r) > 0`;
}

/**
 * Query returning the native-zoom statistics of the output table as one JSON
 * string (reads only the per-tile statistics columns, never the tile bytes).
 * The variance is combined across tiles with the parallel algorithm.
 */
function buildBigQueryStatsSql(planArg, outputTable) {
    const p = loadPlan(planArg);
    const out = bigQueryName(outputTable, BQ_TABLE_RE, 'output table');
    const c = (o, s) => `\`${o.name}_${s}\``;
    const statCols = p.outputs.flatMap(o => STAT_SUFFIXES.map(s => c(o, s)));
    const means = p.outputs.map((o, k) =>
        `SAFE_DIVIDE(SUM(IF(z = mz, ${c(o, 'sum')}, 0)), SUM(IF(z = mz, ${c(o, 'count')}, 0))) AS gm${k}`);
    const bandAggs = p.outputs.map((o, k) => `STRUCT(
      SUM(IF(z = mz, ${c(o, 'count')}, 0)) AS count,
      MIN(IF(z = mz, ${c(o, 'min')}, NULL)) AS min,
      MAX(IF(z = mz, ${c(o, 'max')}, NULL)) AS max,
      SUM(IF(z = mz, ${c(o, 'sum')}, 0)) AS sum,
      SUM(IF(z = mz AND ${c(o, 'count')} > 0, ${c(o, 'count')} * (POW(${c(o, 'stddev')}, 2) + POW(${c(o, 'mean')} - gm${k}, 2)), 0)) AS m2
    )`).join(',\n    ');
    return `WITH s AS (
  SELECT ${statCols.join(', ')}, ((block >> 52) & 31) AS z FROM \`${out}\` WHERE block != 0
),
m AS (SELECT MAX(z) AS mz FROM s),
g AS (SELECT ${means.join(', ')} FROM s CROSS JOIN m)
SELECT TO_JSON_STRING(STRUCT(
    COUNT(*) AS num_blocks,
    MIN(z) AS min_zoom,
    MAX(z) AS max_zoom,
    [${bandAggs}] AS bands
  ))
FROM s CROSS JOIN m CROSS JOIN g`;
}

/** Single-row INSERT of the metadata row; the JSON travels as a base64 literal. */
function buildBigQueryMetadataInsertSql(planArg, stats, inputs, outputTable, createdBy) {
    const out = bigQueryName(outputTable, BQ_TABLE_RE, 'output table');
    const metadata = buildMetadata(planArg, stats, {
        inputs: inputs.map(t => String(t).trim().replace(/`/g, '')),
        created_by: createdBy
    });
    return `INSERT INTO \`${out}\` (block, metadata)
VALUES (0, SAFE_CONVERT_BYTES_TO_STRING(FROM_BASE64('${planToBase64(metadata)}')))`;
}

// ---------------------------------------------------------------------------
// Snowflake. RaQuet files loaded with MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
// have upper-case columns (BLOCK, BAND_1) while metadata keeps the band names:
// input columns are resolved case-insensitively against the actual table
// columns and every identifier is quoted; output columns are upper case.
// ---------------------------------------------------------------------------
const SF_PART = '(?:[A-Za-z_][A-Za-z0-9_]*|"[^"]+")';
// Built by concatenation: a "$" next to a backtick in the minified bundle would be
// read as a replacement pattern by builds that inline libraries with String.replace
const SF_NAME_RE = new RegExp('^' + SF_PART + '(?:\\.' + SF_PART + '){0,2}' + DOLLAR);

function snowflakeTableName(name, what) {
    if (typeof name !== 'string' || !SF_NAME_RE.test(name.trim())) {
        throw new AlgebraError(`Invalid ${what || 'table'} name '${name}'`);
    }
    return name.trim(); // user quoting is kept: it changes identifier semantics
}

const sfQuote = id => `"${String(id).replace(/"/g, '""')}"`;

function resolveSnowflakeColumn(columns, name, i) {
    if (!columns) return name.toUpperCase(); // unquoted-load convention
    const exact = columns.find(c => c === name);
    if (exact) return exact;
    const matches = columns.filter(c => c.toLowerCase() === name.toLowerCase());
    if (matches.length === 1) return matches[0];
    if (matches.length > 1) throw new AlgebraError(`Column '${name}' of ${inputRef(i)} is ambiguous (${matches.join(', ')})`);
    throw new AlgebraError(`Column '${name}' not found in ${inputRef(i)}. Columns: ${columns.join(', ')}`);
}

/**
 * @param {string} blockFunction       qualified name of the per-block UDF,
 *        e.g. 'MYDB.RAQUET.__RASTER_ALGEBRA_BLOCK'
 * @param {string[][]} [inputColumns]  actual column names of each input table
 *        (from the procedure); defaults to the upper-case load convention
 */
function buildSnowflakeSql(planArg, inputs, outputTable, blockFunction, inputColumns) {
    const p = loadPlan(planArg);
    const out = snowflakeTableName(outputTable, 'output table');
    const tables = inputs.map((t, i) => snowflakeTableName(t, `input ${inputRef(i)} table`));
    if (tables.length !== p.num_inputs) throw new AlgebraError('Input count does not match plan');
    if (tables.some(t => sameTable(t, out))) throw new AlgebraError('Output table must differ from the input tables');
    const udf = snowflakeTableName(blockFunction, 'block function');
    const b64 = planToBase64(p);
    const [zmin, zmax] = p.zoom_range;
    const alias = o => `C${o}`;

    const ctes = tables.map((t, i) => {
        const cols = inputColumns ? inputColumns[i] : null;
        const blockCol = sfQuote(resolveSnowflakeColumn(cols, 'block', i));
        const selected = [];
        const seen = new Set();
        p.operands.forEach((o, k) => {
            if (o.input !== i || seen.has(o.column)) return;
            seen.add(o.column);
            selected.push(`${sfQuote(resolveSnowflakeColumn(cols, o.column, i))} AS ${alias(k)}`);
        });
        return `I${i} AS (\n  SELECT ${blockCol} AS BLOCK, ${selected.join(', ')}\n  FROM ${t}\n  WHERE ${blockCol} != 0 AND BITAND(BITSHIFTRIGHT(${blockCol}, 52), 31) BETWEEN ${zmin} AND ${zmax}\n)`;
    });
    // Operands sharing a column (interleaved bands) read the alias of the first one
    const firstAlias = p.operands.map((o, k) => {
        const first = p.operands.findIndex(x => x.input === o.input && x.column === o.column);
        return alias(first === -1 ? k : first);
    });
    let from;
    if (p.join === 'inner') {
        from = tables.map((_, i) => (i === 0 ? 'I0' : `JOIN I${i} USING (BLOCK)`)).join('\n  ');
    } else {
        ctes.push(`BLOCKS AS (\n  ${tables.map((_, i) => `SELECT BLOCK FROM I${i}`).join('\n  UNION\n  ')}\n)`);
        from = ['BLOCKS', ...tables.map((_, i) => `LEFT JOIN I${i} USING (BLOCK)`)].join('\n  ');
    }
    const payloads = p.operands.map((o, k) => `COALESCE(BASE64_ENCODE(I${o.input}.${firstAlias[k]}), '')`).join(', ');
    const col = s => sfQuote(s.toUpperCase());
    const cols = p.outputs.map((o, k) => [
        `BASE64_DECODE_BINARY(R[${k}]:data::VARCHAR) AS ${col(o.name)}`,
        `R[${k}]:count::INTEGER AS ${col(`${o.name}_count`)}`,
        `R[${k}]:min::FLOAT AS ${col(`${o.name}_min`)}`,
        `R[${k}]:max::FLOAT AS ${col(`${o.name}_max`)}`,
        `R[${k}]:sum::FLOAT AS ${col(`${o.name}_sum`)}`,
        `R[${k}]:mean::FLOAT AS ${col(`${o.name}_mean`)}`,
        `R[${k}]:stddev::FLOAT AS ${col(`${o.name}_stddev`)}`
    ].join(',\n  ')).join(',\n  ');
    return `CREATE TABLE ${out}
CLUSTER BY (BLOCK) AS
WITH ${ctes.join(',\n')},
COMPUTED AS (
  SELECT BLOCK, ${udf}('${b64}', ARRAY_CONSTRUCT(${payloads})) AS R
  FROM ${from}
)
SELECT
  BLOCK AS "BLOCK",
  CAST(NULL AS VARCHAR) AS "METADATA",
  ${cols}
FROM COMPUTED
WHERE R IS NOT NULL`;
}

function buildSnowflakeStatsSql(planArg, outputTable) {
    const p = loadPlan(planArg);
    const out = snowflakeTableName(outputTable, 'output table');
    const c = (o, s) => sfQuote(`${o.name}_${s}`.toUpperCase());
    const statCols = p.outputs.flatMap(o => STAT_SUFFIXES.map(s => c(o, s)));
    const means = p.outputs.map((o, k) =>
        `DIV0NULL(SUM(IFF(Z = MZ, ${c(o, 'sum')}, 0)), SUM(IFF(Z = MZ, ${c(o, 'count')}, 0))) AS GM${k}`);
    const bands = p.outputs.map((o, k) => `OBJECT_CONSTRUCT_KEEP_NULL(
      'count', SUM(IFF(Z = MZ, ${c(o, 'count')}, 0)),
      'min', MIN(IFF(Z = MZ, ${c(o, 'min')}, NULL)),
      'max', MAX(IFF(Z = MZ, ${c(o, 'max')}, NULL)),
      'sum', SUM(IFF(Z = MZ, ${c(o, 'sum')}, 0)),
      'm2', SUM(IFF(Z = MZ AND ${c(o, 'count')} > 0, ${c(o, 'count')} * (SQUARE(${c(o, 'stddev')}) + SQUARE(${c(o, 'mean')} - GM${k})), 0))
    )`).join(',\n    ');
    return `WITH S AS (
  SELECT ${statCols.join(', ')}, BITAND(BITSHIFTRIGHT("BLOCK", 52), 31) AS Z FROM ${out} WHERE "BLOCK" != 0
),
M AS (SELECT MAX(Z) AS MZ FROM S),
G AS (SELECT ${means.join(', ')} FROM S CROSS JOIN M)
SELECT TO_JSON(OBJECT_CONSTRUCT_KEEP_NULL(
    'num_blocks', COUNT(*),
    'min_zoom', MIN(Z),
    'max_zoom', MAX(Z),
    'bands', ARRAY_CONSTRUCT(${bands})
  ))
FROM S CROSS JOIN M CROSS JOIN G`;
}

/** INSERT of the metadata row; bind the metadata JSON as the single parameter */
function buildSnowflakeMetadataInsertSql(outputTable) {
    const out = snowflakeTableName(outputTable, 'output table');
    return `INSERT INTO ${out} ("BLOCK", "METADATA") VALUES (0, ?)`;
}

function decodeExtra(extra) {
    if (!extra) return undefined;
    if (typeof extra === 'object') return extra;
    return JSON.parse(extra[0] === '{' ? extra : decodeBase64Text(extra));
}

export default {
    parse,
    plan,
    evalBlock,
    buildMetadata: (planArg, stats, extra) => buildMetadata(planArg, stats, decodeExtra(extra)),
    buildBigQuerySql,
    buildBigQueryStatsSql,
    buildBigQueryMetadataInsertSql,
    buildSnowflakeSql,
    buildSnowflakeStatsSql,
    buildSnowflakeMetadataInsertSql,
    snowflakeTableName,
    inputRef,
    planToBase64,
    base64Encode,
    base64Decode,
    AlgebraError
};
