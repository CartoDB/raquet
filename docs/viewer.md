---
layout: default
title: Viewer
permalink: /viewer/
---

# RaQuet Viewer

The RaQuet Viewer is a client-side application that lets you explore RaQuet files directly in your browser.

**[Launch the Viewer]({{ site.baseurl }}/viewer.html)**

## Features

- **No server required** - Runs entirely in your browser using DuckDB-WASM
- **Load from URL** - Works with any publicly accessible RaQuet file
- **Interactive map** - Pan, zoom, and explore raster data
- **Efficient loading** - Uses batched queries for optimal performance

## How It Works

The viewer uses:
- **DuckDB-WASM** for querying Parquet files via HTTP range requests
- **deck.gl** for WebGL-accelerated map rendering
- **pako** for decompressing gzip-compressed band data

## Usage

1. Enter a URL to a RaQuet file (must be CORS-enabled)
2. Click "Load Dataset"
3. Pan and zoom to explore

You can also pass a URL parameter: `viewer.html?url=https://example.com/file.parquet`

## CORS Requirements

The RaQuet file must be served with CORS headers allowing browser access. For Google Cloud Storage:

```json
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD"],
    "responseHeader": ["Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"],
    "maxAgeSeconds": 3600
  }
]
```

Apply with: `gsutil cors set cors.json gs://your-bucket`
