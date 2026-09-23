import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import terser from '@rollup/plugin-terser';

export default [
    // Raquet core library (base64, decoding, stats)
    {
        input: 'src/raquet_lib.js',
        output: {
            file: 'build/raquet_lib.js',
            format: 'iife',
            name: 'raquetLib'
        },
        plugins: [
            resolve(),
            commonjs(),
            terser()
        ]
    },
    // Raquet inflate library (pako)
    {
        input: 'libs/raquet_inflate.js',
        output: {
            file: 'build/raquet_inflate.js',
            format: 'iife',
            name: 'raquetInflateLib'
        },
        plugins: [
            resolve(),
            commonjs(),
            terser()
        ]
    },
    // JPEG decoder library (v0.4.0)
    {
        input: 'src/jpeg_decoder.js',
        output: {
            file: 'build/jpeg_decoder.js',
            format: 'iife',
            name: 'jpegDecoderLib'
        },
        plugins: [
            resolve(),
            commonjs(),
            terser()
        ]
    },
    // Raster algebra library (parser, evaluator, codec incl. pako inflate+gzip)
    {
        input: 'src/raquet_algebra.js',
        output: {
            file: 'build/raquet_algebra.js',
            format: 'iife',
            name: 'raquetAlgebraLib'
        },
        plugins: [
            resolve(),
            commonjs(),
            terser()
        ]
    }
];
