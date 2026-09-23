#!/usr/bin/env node
// Inline the built JS bundles into the Snowflake RASTER_ALGEBRA SQL.
// Snowflake JS UDFs cannot load external libraries, and bodies are delimited
// by $$: a bundle containing that sequence is rejected (rewriting it is not
// semantics-preserving in regex literals or identifiers).
//
// Usage: (cd libraries/javascript && npm run build) && node scripts/build_snowflake_algebra.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const build = path.join(root, 'libraries/javascript/build');
const template = path.join(root, 'platforms/snowflake/templates/RASTER_ALGEBRA.sql.tmpl');
const output = path.join(root, 'platforms/snowflake/functions/RASTER_ALGEBRA.sql');

function load(file) {
    const code = fs.readFileSync(path.join(build, file), 'utf8');
    // $$ ends a Snowflake function body; $&, $` and $' are replacement patterns
    // for builds that inline libraries with String.prototype.replace
    const bad = ['$$', '$&', '$`', "$'"].filter(seq => code.includes(seq));
    if (bad.length) {
        throw new Error(`${file} contains ${bad.join(', ')}, which break Snowflake inlining`);
    }
    new Function(code); // syntax check
    return code;
}

const sql = fs.readFileSync(template, 'utf8')
    .replaceAll('/*__RAQUET_ALGEBRA_LIB__*/', () => load('raquet_algebra.js'));

// Sanity: exactly one opening and one closing $$ per object
const delimiters = (sql.match(/\$\$/g) || []).length;
if (delimiters !== 4) throw new Error(`Expected 4 $$ delimiters, found ${delimiters}`);

fs.writeFileSync(output, sql);
console.log(`wrote ${path.relative(root, output)} (${(sql.length / 1024).toFixed(0)} KiB)`);
