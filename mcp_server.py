#!/usr/bin/env python3
"""
ERA5 MCP Server
================
Model Context Protocol server for ERA5 climate data retrieval.

Run with:
    python mcp_server.py

Or configure in Claude Code settings:
    "mcpServers": {
        "vostok": {
            "command": "python",
            "args": ["/path/to/era_5_agent/mcp_server.py"],
            "env": {
                "ARRAYLAKE_API_KEY": "your_key_here"
            }
        }
    }
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

from config import (
    DATA_DIR, PLOTS_DIR,
    get_variable_info, get_short_name, list_available_variables,
    GEOGRAPHIC_REGIONS
)
from memory import get_memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("vostok-climate-data")


# ============================================================================
# ERA5 RETRIEVAL FUNCTION
# ============================================================================

def retrieve_era5_data_sync(
    query_type: str,
    variable_id: str,
    start_date: str,
    end_date: str,
    min_latitude: float = -90.0,
    max_latitude: float = 90.0,
    min_longitude: float = 0.0,
    max_longitude: float = 359.75,
    region: Optional[str] = None
) -> str:
    """
    Synchronous ERA5 data retrieval function.
    """
    import time
    import shutil
    from datetime import datetime
    import xarray as xr

    def _format_coord(value: float) -> str:
        if abs(value) < 0.005:
            value = 0.0
        return f"{value:.2f}"

    memory = get_memory()

    # Get API key
    api_key = os.environ.get("ARRAYLAKE_API_KEY")
    if not api_key:
        return "Error: ARRAYLAKE_API_KEY not found in environment."

    # Check icechunk is available
    try:
        import icechunk
    except ImportError:
        return "Error: The 'icechunk' library is required. Install with: pip install icechunk"

    # Apply region bounds if specified
    region_tag = None
    if region and region.lower() in GEOGRAPHIC_REGIONS:
        r = GEOGRAPHIC_REGIONS[region.lower()]
        min_latitude = r["min_lat"]
        max_latitude = r["max_lat"]
        min_longitude = r["min_lon"]
        max_longitude = r["max_lon"]
        region_tag = region.lower()
        logger.info(f"Using region '{region}'")

    # Resolve variable name
    short_var = get_short_name(variable_id)
    var_info = get_variable_info(variable_id)

    # Setup paths
    output_dir = Path(DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_var = short_var.replace('_', '')
    clean_start = start_date.replace('-', '')
    clean_end = end_date.replace('-', '')
    if not region_tag:
        region_tag = (
            f"lat{_format_coord(min_latitude)}_{_format_coord(max_latitude)}"
            f"_lon{_format_coord(min_longitude)}_{_format_coord(max_longitude)}"
        )
    filename = f"era5_{clean_var}_{query_type}_{clean_start}_{clean_end}_{region_tag}.zarr"
    local_path = str(output_dir / filename)

    # Check cache
    if os.path.exists(local_path):
        return (
            f"CACHE HIT - Data already available\n"
            f"  Variable: {short_var}\n"
            f"  Path: {local_path}\n\n"
            f"Load with: ds = xr.open_dataset('{local_path}', engine='zarr')"
        )

    # Download data
    max_retries = 3
    retry_delay = 2.0

    for attempt in range(max_retries):
        try:
            from arraylake import Client

            logger.info(f"Connecting to Earthmover (attempt {attempt + 1})...")
            client = Client(token=api_key)
            repo = client.get_repo("earthmover-public/era5-surface-aws")
            session = repo.readonly_session("main")

            logger.info(f"Opening {query_type} dataset...")
            ds = xr.open_dataset(
                session.store,
                engine="zarr",
                consolidated=False,
                zarr_format=3,
                chunks=None,
                group=query_type
            )

            if short_var not in ds:
                available = list(ds.data_vars)
                return (
                    f"Error: Variable '{short_var}' not found.\n"
                    f"Available: {', '.join(available)}"
                )

            # Slice data (ERA5 latitude is 90 -> -90)
            lat_slice = slice(max_latitude, min_latitude)
            req_min = min_longitude % 360
            req_max = max_longitude % 360
            lon_slice = slice(req_min, 359.75) if req_min > req_max else slice(req_min, req_max)

            logger.info("Subsetting data...")
            subset = ds[short_var].sel(
                time=slice(start_date, end_date),
                latitude=lat_slice,
                longitude=lon_slice
            )

            ds_out = subset.to_dataset(name=short_var)
            for var in ds_out.variables:
                ds_out[var].encoding = {}

            ds_out.attrs['source'] = 'ERA5 Reanalysis via Earthmover Arraylake'
            ds_out.attrs['download_date'] = datetime.now().isoformat()

            if os.path.exists(local_path):
                shutil.rmtree(local_path)

            logger.info(f"Saving to {local_path}...")
            start_time = time.time()
            ds_out.to_zarr(local_path, mode="w", consolidated=True, compute=True)
            download_time = time.time() - start_time

            file_size = sum(f.stat().st_size for f in Path(local_path).rglob('*') if f.is_file())
            shape = tuple(ds_out[short_var].shape)

            # Register in memory
            memory.register_dataset(
                path=local_path,
                variable=short_var,
                query_type=query_type,
                start_date=start_date,
                end_date=end_date,
                lat_bounds=(min_latitude, max_latitude),
                lon_bounds=(min_longitude, max_longitude),
                file_size_bytes=file_size,
                shape=shape
            )

            size_mb = file_size / (1024 * 1024)
            result = (
                f"SUCCESS - Data downloaded\n"
                f"{'='*50}\n"
                f"  Variable: {short_var}"
            )
            if var_info:
                result += f" ({var_info.long_name})"
            result += (
                f"\n  Units: {var_info.units if var_info else 'Unknown'}\n"
                f"  Period: {start_date} to {end_date}\n"
                f"  Shape: {shape}\n"
                f"  Size: {size_mb:.2f} MB\n"
                f"  Time: {download_time:.1f}s\n"
                f"  Path: {local_path}\n"
                f"{'='*50}\n\n"
                f"Load with:\n"
                f"  ds = xr.open_dataset('{local_path}', engine='zarr')"
            )
            return result

        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if os.path.exists(local_path):
                shutil.rmtree(local_path, ignore_errors=True)

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
            else:
                return f"Error after {max_retries} attempts: {str(e)}"

    return "Error: Unexpected failure"


# ============================================================================
# MCP TOOL DEFINITIONS
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="retrieve_era5_data",
            description=(
                "Retrieve ERA5 climate reanalysis data from Earthmover's cloud archive.\n\n"
                "QUERY TYPES:\n"
                "- 'temporal': For time series (long time, small area)\n"
                "- 'spatial': For maps (large area, short time)\n\n"
                "VARIABLES:\n"
                "- sst: Sea Surface Temperature (K)\n"
                "- t2: 2m Air Temperature (K)\n"
                "- u10, v10: 10m Wind Components (m/s)\n"
                "- mslp: Mean Sea Level Pressure (Pa)\n"
                "- tcc: Total Cloud Cover (0-1)\n"
                "- tp: Total Precipitation (m)\n\n"
                "REGIONS (optional):\n"
                "north_atlantic, north_pacific, california_coast, mediterranean,\n"
                "gulf_of_mexico, nino34, arctic, antarctic, global\n\n"
                "Returns the file path to load with xarray."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["temporal", "spatial"],
                        "description": "Use 'temporal' for time series, 'spatial' for maps"
                    },
                    "variable_id": {
                        "type": "string",
                        "description": "ERA5 variable (sst, t2, u10, v10, mslp, tcc, tp)"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD)"
                    },
                    "min_latitude": {
                        "type": "number",
                        "default": -90.0,
                        "description": "Southern bound (-90 to 90)"
                    },
                    "max_latitude": {
                        "type": "number",
                        "default": 90.0,
                        "description": "Northern bound (-90 to 90)"
                    },
                    "min_longitude": {
                        "type": "number",
                        "default": 0.0,
                        "description": "Western bound (0 to 360)"
                    },
                    "max_longitude": {
                        "type": "number",
                        "default": 359.75,
                        "description": "Eastern bound (0 to 360)"
                    },
                    "region": {
                        "type": "string",
                        "description": "Predefined region (overrides lat/lon)"
                    }
                },
                "required": ["query_type", "variable_id", "start_date", "end_date"]
            }
        ),
        Tool(
            name="list_era5_variables",
            description="List all available ERA5 variables with their descriptions.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_cached_datasets",
            description="List all cached ERA5 datasets that have been downloaded.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_regions",
            description="List all predefined geographic regions.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    """Handle tool calls."""

    if name == "retrieve_era5_data":
        # Run synchronous function in thread pool
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: retrieve_era5_data_sync(
                query_type=arguments["query_type"],
                variable_id=arguments["variable_id"],
                start_date=arguments["start_date"],
                end_date=arguments["end_date"],
                min_latitude=arguments.get("min_latitude", -90.0),
                max_latitude=arguments.get("max_latitude", 90.0),
                min_longitude=arguments.get("min_longitude", 0.0),
                max_longitude=arguments.get("max_longitude", 359.75),
                region=arguments.get("region")
            )
        )
        return CallToolResult(content=[TextContent(type="text", text=result)])

    elif name == "list_era5_variables":
        result = list_available_variables()
        return CallToolResult(content=[TextContent(type="text", text=result)])

    elif name == "list_cached_datasets":
        memory = get_memory()
        result = memory.list_datasets()
        return CallToolResult(content=[TextContent(type="text", text=result)])

    elif name == "list_regions":
        lines = ["Available Geographic Regions:", "=" * 50]
        for region, bounds in GEOGRAPHIC_REGIONS.items():
            lines.append(
                f"  {region:20} | lat: [{bounds['min_lat']:6.1f}, {bounds['max_lat']:6.1f}] "
                f"| lon: [{bounds['min_lon']:6.1f}, {bounds['max_lon']:6.1f}]"
            )
        result = "\n".join(lines)
        return CallToolResult(content=[TextContent(type="text", text=result)])

    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True
        )


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Run the MCP server."""
    logger.info("Starting ERA5 MCP Server...")

    # Check for API key
    if not os.environ.get("ARRAYLAKE_API_KEY"):
        logger.warning("ARRAYLAKE_API_KEY not set - data retrieval will fail")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
