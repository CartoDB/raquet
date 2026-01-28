---
layout: default
title: Viewer
page_header: true
page_description: Explore RaQuet files directly in your browser with no server required
---

<div style="text-align: center; margin: 2rem 0;">
<a href="{{ site.baseurl }}/viewer.html" class="btn btn-primary" style="font-size: 1.1rem; padding: 1rem 2rem;">Launch the Viewer</a>
</div>

---

## Features

<div class="feature-grid">
  <div class="feature-card">
    <h3>No Server Required</h3>
    <p>Runs entirely in your browser using DuckDB-WASM. Query Parquet files via HTTP range requests — no backend needed.</p>
  </div>
  <div class="feature-card">
    <h3>Load from URL</h3>
    <p>Works with any publicly accessible RaQuet file. Just paste a URL and explore.</p>
  </div>
  <div class="feature-card">
    <h3>Interactive Map</h3>
    <p>Pan, zoom, and explore raster data with WebGL-accelerated rendering powered by deck.gl.</p>
  </div>
</div>

---

## How It Works

The viewer uses modern web technologies to query and render RaQuet files client-side:

| Component | Purpose |
|-----------|---------|
| **DuckDB-WASM** | Query Parquet files via HTTP range requests |
| **deck.gl** | WebGL-accelerated map rendering |
| **pako** | Decompress gzip-compressed band data |

---

## Usage

1. Enter a URL to a RaQuet file
2. Click **Load Dataset**
3. Pan and zoom to explore

### URL Parameter

Pass a file URL directly:

```
viewer.html?url=https://storage.googleapis.com/raquet_demo_data/spain_solar_ghi.parquet
```

---

## Sample Files

Try these example datasets:

| Dataset | URL |
|---------|-----|
| Spain Solar GHI | [Load in Viewer]({{ site.baseurl }}/viewer.html?url=https://storage.googleapis.com/raquet_demo_data/spain_solar_ghi.parquet) |
| World Elevation | [Load in Viewer]({{ site.baseurl }}/viewer.html?url=https://storage.googleapis.com/raquet_demo_data/world_elevation.parquet) |
| World Solar PVOUT | [Load in Viewer]({{ site.baseurl }}/viewer.html?url=https://storage.googleapis.com/raquet_demo_data/world_solar_pvout.parquet) |
| TCI (Sentinel-2) | [Load in Viewer]({{ site.baseurl }}/viewer.html?url=https://storage.googleapis.com/raquet_demo_data/TCI.parquet) |

---

## CORS Requirements

The RaQuet file must be served with CORS headers allowing browser access.

### Google Cloud Storage

Create a `cors.json` file:

```json
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD"],
    "responseHeader": [
      "Content-Type",
      "Content-Length",
      "Content-Range",
      "Accept-Ranges"
    ],
    "maxAgeSeconds": 3600
  }
]
```

Apply with:

```bash
gsutil cors set cors.json gs://your-bucket
```

### Amazon S3

Add a CORS configuration to your bucket:

```json
{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedOrigins": ["*"],
      "ExposeHeaders": ["Content-Range", "Accept-Ranges"],
      "MaxAgeSeconds": 3600
    }
  ]
}
```

### Azure Blob Storage

Configure CORS in the Azure portal or via CLI:

```bash
az storage cors add --services b --methods GET HEAD \
  --origins '*' --allowed-headers '*' \
  --exposed-headers 'Content-Range,Accept-Ranges' \
  --max-age 3600 --account-name YOUR_ACCOUNT
```

---

## Limitations

- **File size** — Very large files (>1 GB) may be slow to load initially
- **CORS** — Files must be served with appropriate CORS headers
- **Browser memory** — Limited by available browser memory for decompression

For large-scale analysis, use [DuckDB]({{ site.baseurl }}/engines) or a data warehouse instead of the browser viewer.

