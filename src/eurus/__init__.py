"""
Eurus - ERA5 Climate Analysis Agent
====================================

A scientific climate analysis platform powered by ERA5 reanalysis data from
Earthmover's cloud-optimized archive via Icechunk.

Features:
- ERA5 reanalysis data retrieval (SST, temperature, wind, pressure, etc.)
- Interactive Python REPL with pre-loaded scientific libraries
- Maritime route calculation with weather risk assessment
- Analysis methodology guides for climate science
- Intelligent caching with persistent memory
- Predefined geographic regions (El Niño, Atlantic, Pacific, etc.)
- Full MCP protocol support for Claude and other AI assistants

Example usage as MCP server:
    # In .mcp.json
    {
        "mcpServers": {
            "era5": {
                "command": "era5-mcp",
                "env": {"ARRAYLAKE_API_KEY": "your_key"}
            }
        }
    }

Example usage as Python library:
    from eurus import retrieve_era5_data, list_available_variables
    from eurus.tools import get_all_tools

    # Download SST data
    result = retrieve_era5_data(
        query_type="temporal",
        variable_id="sst",
        start_date="2024-01-01",
        end_date="2024-01-07",
        region="california_coast"
    )

    # Get all tools for agent (only core tools, no science clutter)
    tools = get_all_tools(enable_routing=True)
"""

__version__ = "1.1.0"
__author__ = "Eurus Team"

from eurus.config import (
    ERA5_VARIABLES,
    GEOGRAPHIC_REGIONS,
    AGENT_SYSTEM_PROMPT,
    get_variable_info,
    get_short_name,
    list_available_variables,
)
from eurus.retrieval import retrieve_era5_data
from eurus.memory import MemoryManager, get_memory
from eurus.tools import get_all_tools

__all__ = [
    # Version
    "__version__",
    # Config
    "ERA5_VARIABLES",
    "GEOGRAPHIC_REGIONS",
    "AGENT_SYSTEM_PROMPT",
    "get_variable_info",
    "get_short_name",
    "list_available_variables",
    # Retrieval
    "retrieve_era5_data",
    # Memory
    "MemoryManager",
    "get_memory",
    # Tools
    "get_all_tools",
]
