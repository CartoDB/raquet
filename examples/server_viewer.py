#!/usr/bin/env python3
"""
Example server-side RaQuet tile server using DuckDB.

This demonstrates how to serve RaQuet tiles from a backend server,
which provides better performance than client-side DuckDB-WASM for
production applications.

Usage:
    pip install fastapi uvicorn duckdb quadbin pillow
    python server_viewer.py

Then open http://localhost:8000 in your browser.
"""

import io
import json
import gzip
import struct
from pathlib import Path

try:
    import duckdb
    import quadbin
    from PIL import Image
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import Response, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install fastapi uvicorn duckdb quadbin pillow")
    exit(1)

app = FastAPI(title="RaQuet Tile Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Cache for DuckDB connections and metadata
_connections = {}
_metadata_cache = {}


def get_connection(file_path: str):
    """Get or create a DuckDB connection for a file."""
    if file_path not in _connections:
        conn = duckdb.connect()
        if file_path.startswith("http"):
            conn.execute("INSTALL httpfs; LOAD httpfs;")
        _connections[file_path] = conn
    return _connections[file_path]


def get_metadata(file_path: str) -> dict:
    """Get cached metadata for a RaQuet file."""
    if file_path not in _metadata_cache:
        conn = get_connection(file_path)
        result = conn.execute(f"""
            SELECT metadata FROM read_parquet('{file_path}') WHERE block = 0
        """).fetchone()
        if not result:
            raise HTTPException(status_code=404, detail="Metadata not found")
        _metadata_cache[file_path] = json.loads(result[0])
    return _metadata_cache[file_path]


def decode_band(data: bytes, dtype: str, width: int, height: int) -> bytes:
    """Decode band data from RaQuet format."""
    # Decompress if gzipped
    try:
        data = gzip.decompress(data)
    except:
        pass

    # Convert to uint8 for PNG output
    if dtype == "uint16":
        values = struct.unpack(f"<{width*height}H", data)
        return bytes(min(255, v >> 8) for v in values)
    elif dtype in ("float32", "float64"):
        fmt = "f" if dtype == "float32" else "d"
        values = struct.unpack(f"<{width*height}{fmt}", data)
        min_v, max_v = min(values), max(values)
        if max_v > min_v:
            return bytes(int(255 * (v - min_v) / (max_v - min_v)) for v in values)
        return bytes(width * height)
    return data


@app.get("/")
async def index():
    """Simple HTML viewer."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RaQuet Server Viewer</title>
        <script src="https://unpkg.com/deck.gl@9.0.16/dist.min.js"></script>
        <style>
            body { margin: 0; font-family: sans-serif; }
            #map { width: 100vw; height: 100vh; }
            #controls { position: absolute; top: 10px; left: 10px; background: white; padding: 15px; border-radius: 8px; }
            input { width: 300px; padding: 8px; margin: 5px 0; }
            button { padding: 10px 20px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div id="map"></div>
        <div id="controls">
            <h3>RaQuet Server Viewer</h3>
            <input type="text" id="fileUrl" placeholder="RaQuet file URL or path">
            <button onclick="load()">Load</button>
            <div id="status"></div>
        </div>
        <script>
            let deckgl;
            async function load() {
                const file = document.getElementById('fileUrl').value;
                const res = await fetch('/metadata?file=' + encodeURIComponent(file));
                const meta = await res.json();
                document.getElementById('status').textContent = `Loaded: ${meta.num_blocks} tiles`;

                const center = meta.center || [(meta.bounds[0]+meta.bounds[2])/2, (meta.bounds[1]+meta.bounds[3])/2];
                const layer = new deck.TileLayer({
                    id: 'tiles',
                    minZoom: meta.minresolution,
                    maxZoom: meta.maxresolution,
                    tileSize: meta.block_width || 256,
                    extent: meta.bounds,
                    getTileData: ({x, y, z}) => fetch(`/tile/${z}/${x}/${y}?file=${encodeURIComponent(file)}`).then(r => r.ok ? r.blob() : null),
                    renderSubLayers: props => {
                        if (!props.data) return null;
                        return new deck.BitmapLayer({
                            ...props,
                            image: props.data,
                            bounds: [props.tile.bbox.west, props.tile.bbox.south, props.tile.bbox.east, props.tile.bbox.north]
                        });
                    }
                });

                if (deckgl) deckgl.setProps({ layers: [layer] });
                else deckgl = new deck.DeckGL({
                    container: 'map',
                    initialViewState: { longitude: center[0], latitude: center[1], zoom: meta.center?.[2] || meta.minresolution },
                    controller: true,
                    layers: [layer]
                });
            }
        </script>
    </body>
    </html>
    """)


@app.get("/metadata")
async def metadata(file: str = Query(..., description="Path or URL to RaQuet file")):
    """Get metadata for a RaQuet file."""
    try:
        return get_metadata(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tile/{z}/{x}/{y}")
async def tile(
    z: int,
    x: int,
    y: int,
    file: str = Query(..., description="Path or URL to RaQuet file"),
    bands: str = Query("band_1,band_2,band_3", description="Comma-separated band names"),
):
    """Get a tile as PNG."""
    try:
        meta = get_metadata(file)
        conn = get_connection(file)

        # Convert tile coordinates to QUADBIN
        block_id = quadbin.tile_to_cell((x, y, z))

        # Query the tile
        band_list = [b.strip() for b in bands.split(",")]
        band_cols = ", ".join(band_list)
        result = conn.execute(f"""
            SELECT {band_cols} FROM read_parquet('{file}') WHERE block = {block_id}
        """).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Tile not found")

        # Get band info
        band_info = {b["name"]: b for b in meta["bands"]}
        width = meta.get("block_width", 256)
        height = meta.get("block_height", 256)

        # Decode bands
        decoded = []
        for i, band_name in enumerate(band_list[:3]):
            data = result[i]
            if data:
                dtype = band_info.get(band_name, {}).get("type", "uint8")
                decoded.append(decode_band(data, dtype, width, height))
            else:
                decoded.append(bytes(width * height))

        # Pad to 3 bands for RGB
        while len(decoded) < 3:
            decoded.append(decoded[0] if decoded else bytes(width * height))

        # Create PNG
        img = Image.new("RGB", (width, height))
        pixels = list(zip(decoded[0], decoded[1], decoded[2]))
        img.putdata(pixels)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return Response(content=buf.read(), media_type="image/png")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting RaQuet tile server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
