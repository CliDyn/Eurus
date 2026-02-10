#!/usr/bin/env python3
"""
Eurus Tile Server — Custom lightweight tile renderer
=====================================================
Renders XYZ map tiles from xarray datasets using scipy interpolation + matplotlib colormaps.
No dependency on xpublish-tiles (which has rendering bugs in v0.4.0).

Usage:
    python scripts/tile_server.py                     # Default: air tutorial on port 8080
    python scripts/tile_server.py --port 9090         # Custom port
    python scripts/tile_server.py --dataset ersstv5   # Different tutorial dataset
"""

import argparse
import io
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse
import uvicorn

# Load .env for ARRAYLAKE_API_KEY
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logger = logging.getLogger("tile_server")

# ────────────────────────────────────────
# Dataset management
# ────────────────────────────────────────
DATASETS: dict = {}  # {name: xr.Dataset}


def load_dataset(name: str):
    """Load an xarray tutorial dataset, normalize coords for tile rendering."""
    import xarray as xr

    if name == "air":
        ds = xr.tutorial.open_dataset("air_temperature")
        # Normalize lon from 200-330 to -160 to -30
        lon_attrs = dict(ds.lon.attrs)
        ds = ds.assign_coords(lon=(ds.lon.values - 360))
        ds.lon.attrs.update(lon_attrs)
        ds = ds.sortby("lat")
        ds = ds.isel(time=0).drop_vars("time")
        return ds

    elif name == "ersstv5":
        ds = xr.tutorial.open_dataset("ersstv5")
        ds = ds.sortby("lat")
        ds = ds.isel(time=0).drop_vars("time")
        return ds

    elif name == "rasm":
        ds = xr.tutorial.open_dataset("rasm")
        ds = ds.isel(time=0).drop_vars("time")
        return ds

    else:
        raise ValueError(f"Unknown dataset: {name}")


# ────────────────────────────────────────
# Tile math (Web Mercator ↔ lat/lon)
# ────────────────────────────────────────
def tile_to_bbox(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Convert tile z/x/y to bounding box (west, south, east, north) in EPSG:4326."""
    n = 2.0 ** z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def render_tile_png(
    ds,
    variable: str,
    z: int, x: int, y: int,
    width: int = 256,
    height: int = 256,
    colorscalerange: str = "0,1",
    style: str = "raster/viridis",
) -> bytes | None:
    """Render a single tile as PNG bytes."""
    from scipy.interpolate import RegularGridInterpolator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    west, south, east, north = tile_to_bbox(z, x, y)

    # Get data array
    da = ds[variable]
    lats = da.coords["lat"].values.astype(float)
    lons = da.coords["lon"].values.astype(float)

    # Check if tile intersects dataset bounds
    lat_min, lat_max = float(lats.min()), float(lats.max())
    lon_min, lon_max = float(lons.min()), float(lons.max())

    if west > lon_max or east < lon_min or south > lat_max or north < lat_min:
        # Tile outside dataset bounds → transparent
        return None

    # Parse color range
    try:
        parts = colorscalerange.split(",")
        vmin, vmax = float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        vmin, vmax = float(np.nanmin(da.values)), float(np.nanmax(da.values))

    # Parse colormap name from style (e.g. "raster/viridis" → "viridis")
    cmap_name = style.split("/")[-1] if "/" in style else style
    try:
        cmap = plt.get_cmap(cmap_name)
    except ValueError:
        cmap = plt.get_cmap("viridis")

    # Ensure lat/lon are sorted for RegularGridInterpolator
    data = da.values.astype(float)
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[::-1, :]
    if lons[0] > lons[-1]:
        lons = lons[::-1]
        data = data[:, ::-1]

    # Create interpolator
    interp = RegularGridInterpolator(
        (lats, lons), data,
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Generate pixel grid in geographic coords
    tile_lons = np.linspace(west, east, width)
    tile_lats = np.linspace(north, south, height)  # north→south for image coords
    grid_lons, grid_lats = np.meshgrid(tile_lons, tile_lats)
    points = np.column_stack([grid_lats.ravel(), grid_lons.ravel()])

    # Interpolate
    values = interp(points).reshape(height, width)

    # Check if all NaN (tile fully outside data)
    if np.all(np.isnan(values)):
        return None

    # Normalize values to 0-1 range
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    normalized = norm(values)

    # Apply colormap → RGBA
    rgba = cmap(normalized)

    # Set NaN pixels to transparent
    nan_mask = np.isnan(values)
    rgba[nan_mask] = [0, 0, 0, 0]

    # Convert to uint8
    rgba_uint8 = (rgba * 255).astype(np.uint8)

    # Save as PNG
    from PIL import Image
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


# ────────────────────────────────────────
# FastAPI application
# ────────────────────────────────────────
app = FastAPI(title="Eurus Tile Server", description="Custom tile renderer for ERA5 data")

TRANSPARENT_PNG = None  # lazy-init


def get_transparent_tile() -> bytes:
    """Generate a 256×256 fully transparent PNG."""
    global TRANSPARENT_PNG
    if TRANSPARENT_PNG is None:
        from PIL import Image
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        TRANSPARENT_PNG = buf.getvalue()
    return TRANSPARENT_PNG


@app.get("/datasets")
async def list_datasets():
    """List available datasets."""
    return list(DATASETS.keys())


@app.get("/datasets/{dataset_id}/variables")
async def list_variables(dataset_id: str):
    """List variables in a dataset."""
    if dataset_id not in DATASETS:
        return JSONResponse({"error": f"Dataset '{dataset_id}' not found"}, status_code=404)
    ds = DATASETS[dataset_id]
    return {
        "variables": list(ds.data_vars),
        "coords": list(ds.coords),
        "dims": {str(k): v for k, v in ds.sizes.items()},
    }


@app.get("/datasets/{dataset_id}/tiles/WebMercatorQuad/{z}/{y}/{x}")
async def get_tile(
    dataset_id: str,
    z: int, y: int, x: int,
    variables: str = Query(default="air"),
    colorscalerange: str = Query(default="240,310"),
    style: str = Query(default="raster/viridis"),
    width: int = Query(default=256),
    height: int = Query(default=256),
):
    """Render a map tile."""
    if dataset_id not in DATASETS:
        return JSONResponse({"error": f"Dataset '{dataset_id}' not found"}, status_code=404)

    ds = DATASETS[dataset_id]
    variable = variables  # single variable name

    if variable not in ds.data_vars:
        return JSONResponse(
            {"error": f"Variable '{variable}' not found. Available: {list(ds.data_vars)}"},
            status_code=422,
        )

    try:
        png_data = render_tile_png(
            ds, variable, z, x, y,
            width=width, height=height,
            colorscalerange=colorscalerange,
            style=style,
        )
    except Exception as e:
        logger.exception(f"Tile render error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

    if png_data is None:
        # Outside data bounds → transparent tile
        return Response(content=get_transparent_tile(), media_type="image/png")

    return Response(content=png_data, media_type="image/png")


@app.get("/docs-info")
async def docs_info():
    """API info."""
    return {
        "title": "Eurus Tile Server",
        "datasets": list(DATASETS.keys()),
        "tile_url_template": "/datasets/{dataset_id}/tiles/WebMercatorQuad/{z}/{y}/{x}?variables={var}&colorscalerange={min},{max}&style=raster/{cmap}",
    }


# ────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Eurus Tile Server — custom tile renderer for xarray datasets"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument(
        "--dataset", type=str, default="air",
        help="Dataset to serve: air (default), ersstv5, rasm",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    # Load dataset
    print(f"📦 Loading dataset: {args.dataset}...")
    ds = load_dataset(args.dataset)
    DATASETS[args.dataset] = ds
    print(f"   Variables: {list(ds.data_vars)}")
    print(f"   Lat: {float(ds.lat.min()):.1f} → {float(ds.lat.max()):.1f}")
    print(f"   Lon: {float(ds.lon.min()):.1f} → {float(ds.lon.max()):.1f}")

    print(f"""
╔══════════════════════════════════════════════════════╗
║          Eurus Tile Server (Custom Renderer)         ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║   Tiles: http://localhost:{args.port}/datasets/{args.dataset}/tiles/     ║
║   Vars:  {', '.join(ds.data_vars):<43s} ║
║                                                      ║
║   Example tile URL:                                  ║
║   /datasets/{args.dataset}/tiles/WebMercatorQuad/2/1/1       ║
║     ?variables={list(ds.data_vars)[0]}&colorscalerange=240,310           ║
║     &style=raster/viridis                             ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
