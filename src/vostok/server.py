#!/usr/bin/env python3
"""
ERA5 MCP Server
===============

Model Context Protocol server for ERA5 climate data retrieval.

Usage:
    vostok-mcp                          # If installed as package
    python -m vostok.server         # Direct execution

Configuration via environment variables:
    ARRAYLAKE_API_KEY    - Required for data access
    ERA5_DATA_DIR        - Data storage directory (default: ./data)
    ERA5_MEMORY_DIR      - Memory storage directory (default: ./.memory)
    ERA5_MAX_RETRIES     - Download retry attempts (default: 3)
    ERA5_LOG_LEVEL       - Logging level (default: INFO)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Configure logging
log_level = os.environ.get("ERA5_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import MCP components
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        CallToolResult,
        TextContent,
        Tool,
    )
except ImportError:
    logger.error("MCP library not found. Install with: pip install mcp")
    sys.exit(1)

# Import ERA5 components
from vostok.config import (
    list_available_variables,
)
from vostok.memory import get_memory
from vostok.tools.era5 import retrieve_era5_data, ERA5RetrievalArgs

# Import Maritime Routing tool
from vostok.tools.routing import (
    calculate_maritime_route,
    RouteArgs,
    HAS_ROUTING_DEPS,
)

# Create MCP server
server = Server("era5-climate-data")

# Alias for compatibility
app = server


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    tools = [
        Tool(
            name="retrieve_era5_data",
            description=(
                "Retrieve ERA5 climate reanalysis data from Earthmover's cloud archive.\n\n"
                "⚠️ QUERY TYPE is AUTO-DETECTED based on time/area:\n"
                "- 'temporal': time > 1 day AND region < 30°×30° (time series, small area)\n"
                "- 'spatial': time ≤ 1 day OR region ≥ 30°×30° (maps, snapshots, large area)\n\n"
                "VARIABLES: sst, t2, u10, v10, mslp, tcc, tp\n"
                "NOTE: swh (waves) is NOT available in this dataset!\n\n"
                "COORDINATES: Always specify lat/lon bounds explicitly.\n"
                "Longitude: Use 0-360 format (e.g., -74°W = 286°E)\n\n"
                "Returns file path. Load: xr.open_dataset('PATH', engine='zarr')"
            ),
            inputSchema=ERA5RetrievalArgs.model_json_schema()
        ),
        Tool(
            name="list_era5_variables",
            description=(
                "List all available ERA5 variables with their descriptions, units, "
                "and short names for use with retrieve_era5_data."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="list_cached_datasets",
            description=(
                "List all ERA5 datasets that have been downloaded and cached locally. "
                "Shows variable, date range, file path, and size."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
    ]

    # ========== MARITIME ROUTING TOOL (if dependencies available) ==========
    if HAS_ROUTING_DEPS:
        tools.append(
            Tool(
                name="calculate_maritime_route",
                description=(
                    "Calculate a realistic maritime shipping route between two ports. "
                    "Uses global shipping lane graph to avoid land and find optimal path.\n\n"
                    "RETURNS: Waypoint coordinates, bounding box, and INSTRUCTIONS for "
                    "climatological risk assessment protocol.\n\n"
                    "DOES NOT: Check weather itself. The Agent must follow the returned "
                    "protocol to assess route safety using ERA5 data.\n\n"
                    "WORKFLOW:\n"
                    "1. Call this tool → get waypoints + instructions\n"
                    "2. Download ERA5 wind data (u10, v10) for the region\n"
                    "3. Call get_visualization_guide(viz_type='maritime_risk_assessment')\n"
                    "4. Execute analysis in python_repl"
                ),
                inputSchema=RouteArgs.model_json_schema()
            )
        )

    return tools


# ============================================================================
# TOOL HANDLERS
# ============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle tool calls."""

    try:
        if name == "retrieve_era5_data":
            # Run synchronous function in thread pool (query_type auto-detected)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: retrieve_era5_data(
                    variable_id=arguments["variable_id"],
                    start_date=arguments["start_date"],
                    end_date=arguments["end_date"],
                    min_latitude=arguments["min_latitude"],
                    max_latitude=arguments["max_latitude"],
                    min_longitude=arguments["min_longitude"],
                    max_longitude=arguments["max_longitude"],
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

        # ========== MARITIME ROUTING HANDLER ==========
        elif name == "calculate_maritime_route":
            if not HAS_ROUTING_DEPS:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Error: Maritime routing dependencies not installed.\n"
                             "Install with: pip install scgraph geopy"
                    )],
                    isError=True
                )
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: calculate_maritime_route(
                    origin_lat=arguments["origin_lat"],
                    origin_lon=arguments["origin_lon"],
                    dest_lat=arguments["dest_lat"],
                    dest_lon=arguments["dest_lon"],
                    month=arguments["month"],
                    year=arguments.get("year"),
                    speed_knots=arguments.get("speed_knots", 14.0)
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True
            )

    except Exception as e:
        logger.exception(f"Error executing tool {name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


# ============================================================================
# SERVER STARTUP
# ============================================================================

async def run_server() -> None:
    """Run the MCP server using stdio transport."""
    logger.info("Starting ERA5 MCP Server...")

    # Check for API key
    if not os.environ.get("ARRAYLAKE_API_KEY"):
        logger.warning(
            "ARRAYLAKE_API_KEY not set. Data retrieval will fail. "
            "Set it via environment variable or .env file."
        )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
