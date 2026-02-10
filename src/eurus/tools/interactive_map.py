"""
Interactive Map Tile Tool
==========================
LangChain StructuredTool for rendering interactive map tiles via xpublish-tiles.

⚠️  THIS TOOL IS FOR SPECIAL CASES ONLY:
    - User explicitly requests an interactive/zoomable map
    - User asks for a "live" or "dynamic" spatial visualization
    - User wants to explore data spatially by panning/zooming

For regular plots and static maps, use the Python REPL tool with matplotlib.
"""

import json
import logging
from typing import Optional, Literal

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# Default tile server URL (proxied through Eurus web on same port)
TILE_PROXY_BASE = "/tiles"
TILE_SERVER_BASE = "http://localhost:8080"


# ============================================================================
# ARGUMENT SCHEMA
# ============================================================================

class InteractiveMapArgs(BaseModel):
    """Arguments for rendering an interactive map tile layer."""

    variable: str = Field(
        description=(
            "ERA5 variable short name to render on the map. Common variables:\n"
            "  sst  — Sea Surface Temperature (K)\n"
            "  t2   — 2m Air Temperature (K)\n"
            "  d2   — 2m Dewpoint Temperature (K)\n"
            "  skt  — Skin Temperature (K)\n"
            "  u10  — 10m U-wind (m/s)\n"
            "  v10  — 10m V-wind (m/s)\n"
            "  sp   — Surface Pressure (Pa)\n"
            "  mslp — Mean Sea Level Pressure (Pa)\n"
            "  tp   — Total Precipitation (m)\n"
            "  tcc  — Total Cloud Cover (0-1)\n"
            "  ssr  — Net Solar Radiation (J/m²)\n"
            "  tcwv — Total Column Water Vapour (kg/m²)\n"
            "Use the same short names as the ERA5 retrieval tool."
        )
    )

    colorscale_min: float = Field(
        description=(
            "Minimum value for the color scale. Examples:\n"
            "  SST: 270 (K) for polar, 290 for tropics\n"
            "  Temperature: 240-280 (K)\n"
            "  Wind: 0 (m/s)\n"
            "  Pressure: 98000 (Pa)"
        )
    )

    colorscale_max: float = Field(
        description=(
            "Maximum value for the color scale. Examples:\n"
            "  SST: 305 (K)\n"
            "  Temperature: 310 (K)\n"
            "  Wind: 25 (m/s)\n"
            "  Pressure: 103000 (Pa)"
        )
    )

    style: str = Field(
        default="raster/viridis",
        description=(
            "Colormap style in format 'raster/{colormap_name}'. Options:\n"
            "  raster/viridis   — sequential (default, good for temperature)\n"
            "  raster/plasma    — sequential warm\n"
            "  raster/inferno   — sequential hot\n"
            "  raster/coolwarm  — diverging (anomalies)\n"
            "  raster/RdBu_r    — diverging red-blue\n"
            "  raster/Blues     — sequential blues (precipitation)\n"
            "  raster/YlOrRd    — sequential yellow-red"
        )
    )

    min_latitude: float = Field(
        ge=-85.0, le=85.0,
        description="Southern latitude bound for the map view (-85 to 85)"
    )

    max_latitude: float = Field(
        ge=-85.0, le=85.0,
        description="Northern latitude bound for the map view (-85 to 85)"
    )

    min_longitude: float = Field(
        ge=-180.0, le=180.0,
        description="Western longitude bound for the map view (-180 to 180)"
    )

    max_longitude: float = Field(
        ge=-180.0, le=180.0,
        description="Eastern longitude bound for the map view (-180 to 180)"
    )

    dataset_id: str = Field(
        default="air",
        description=(
            "Dataset ID on the tile server. Use 'air' for the demo air temperature dataset. "
            "Other datasets depend on what is currently loaded in the tile server."
        )
    )

    label: Optional[str] = Field(
        default=None,
        description="Human-readable label for the map (e.g. 'SST — North Atlantic, Jan 2024')"
    )


# ============================================================================
# TOOL FUNCTION
# ============================================================================

def render_interactive_map(
    variable: str,
    colorscale_min: float,
    colorscale_max: float,
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
    style: str = "raster/viridis",
    dataset_id: str = "air",
    label: Optional[str] = None,
) -> str:
    """
    Generate an interactive map tile URL for the Eurus web interface.

    Returns a JSON payload that the web interface renders as a zoomable
    Leaflet map with ERA5 data tiles overlaid on a base map.
    """
    # Resolve variable name for the actual dataset
    # Demo datasets have different variable names than ERA5
    resolved_variable = variable
    try:
        import httpx
        resp = httpx.get(
            f"{TILE_SERVER_BASE}/datasets/{dataset_id}/tiles/WebMercatorQuad/tilejson.json",
            params={"variables": variable, "width": 256, "height": 256},
            timeout=5.0
        )
        if resp.status_code == 422:
            # Variable doesn't exist — query dataset to find correct one
            logger.warning(f"Variable '{variable}' not in dataset '{dataset_id}', resolving...")
            # For known demo datasets, use known mappings
            DEMO_VARS = {"air": "air", "ersstv5": "sst", "rasm": "Tair"}
            if dataset_id in DEMO_VARS:
                resolved_variable = DEMO_VARS[dataset_id]
                logger.info(f"Mapped to demo variable: {resolved_variable}")
    except Exception as e:
        logger.warning(f"Variable resolution failed: {e}")

    # Build tile URL template (uses {z}/{y}/{x} Leaflet placeholders)
    colorscalerange = f"{colorscale_min},{colorscale_max}"
    tile_url = (
        f"{TILE_PROXY_BASE}/datasets/{dataset_id}/tiles/WebMercatorQuad"
        f"/{{z}}/{{y}}/{{x}}"
        f"?variables={resolved_variable}"
        f"&colorscalerange={colorscalerange}"
        f"&style={style}"
        f"&width=256&height=256"
    )

    # Check if tile server is reachable
    server_status = "unknown"
    try:
        import httpx
        resp = httpx.get(f"{TILE_SERVER_BASE}/datasets", timeout=3.0)
        if resp.status_code == 200:
            server_status = "online"
            available_datasets = resp.json()
            logger.info(f"Tile server online. Available datasets: {available_datasets}")
        else:
            server_status = "error"
    except Exception as e:
        server_status = "offline"
        logger.warning(f"Tile server not reachable: {e}")

    # Build the response
    map_label = label or f"{variable.upper()} ({colorscale_min}–{colorscale_max})"
    bbox = [min_longitude, min_latitude, max_longitude, max_latitude]

    result = {
        "type": "interactive_map",
        "tile_url": tile_url,
        "options": {
            "variable": variable,
            "label": map_label,
            "bbox": bbox,
            "colorscalerange": f"{colorscale_min}–{colorscale_max}",
            "style": style,
            "zoom": _estimate_zoom(bbox),
        },
        "server_status": server_status,
        "instructions": (
            "The web interface will render this as an interactive Leaflet map. "
            "The user can zoom, pan, and explore the data spatially. "
            f"Tile server status: {server_status}."
        ),
    }

    if server_status == "offline":
        result["warning"] = (
            "⚠️ Tile server is not running! "
            "Start it with: python scripts/tile_server.py --dataset {dataset_id}"
        )

    return json.dumps(result, indent=2)


def _estimate_zoom(bbox: list) -> int:
    """Estimate a reasonable zoom level from bbox size."""
    lon_span = abs(bbox[2] - bbox[0])
    lat_span = abs(bbox[3] - bbox[1])
    span = max(lon_span, lat_span)

    if span > 100:
        return 2
    elif span > 50:
        return 3
    elif span > 20:
        return 4
    elif span > 10:
        return 5
    elif span > 5:
        return 6
    else:
        return 7


# ============================================================================
# LANGCHAIN TOOL
# ============================================================================

interactive_map_tool = StructuredTool.from_function(
    func=render_interactive_map,
    name="render_interactive_map",
    description=(
        "⚠️ SPECIAL USE ONLY — call this ONLY when the user explicitly requests "
        "an interactive, zoomable, or dynamic map visualization.\n\n"
        "DO NOT use for regular plots — use the Python REPL with matplotlib instead.\n\n"
        "USE WHEN the user says things like:\n"
        "  - 'show me an interactive map of SST'\n"
        "  - 'I want to explore temperature data on a map'\n"
        "  - 'can I zoom into this region?'\n"
        "  - 'render a live/dynamic/interactive visualization'\n\n"
        "DO NOT USE when the user asks for:\n"
        "  - static plots, charts, or figures\n"
        "  - time series analysis\n"
        "  - computations or statistics\n\n"
        "Returns a JSON payload that the web interface renders as a Leaflet map "
        "with ERA5 data tiles. Requires the tile server to be running "
        "(python scripts/tile_server.py).\n\n"
        "Supported variables: sst, t2, d2, skt, u10, v10, sp, mslp, tp, tcc, ssr, tcwv"
    ),
    args_schema=InteractiveMapArgs,
)
