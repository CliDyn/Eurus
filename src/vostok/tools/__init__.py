"""
Vostok Tools Registry
=====================
Central hub for all agent tools.

Tools:
- Data Retrieval: ERA5 data access
- Analysis: Python REPL for custom analysis  
- Guides: Methodology and visualization guidance
- Routing: Maritime navigation (optional)
"""

from typing import List
from langchain_core.tools import BaseTool

# Import core tools
from .era5 import era5_tool
from .repl import PythonREPLTool
from .routing import routing_tool
from .analysis_guide import analysis_guide_tool, visualization_guide_tool

# Optional dependency check for routing
try:
    import scgraph
    HAS_ROUTING_DEPS = True
except ImportError:
    HAS_ROUTING_DEPS = False


def get_all_tools(
    enable_routing: bool = True,
    enable_guide: bool = True
) -> List[BaseTool]:
    """
    Return a list of all available tools for the agent.

    Args:
        enable_routing: If True, includes the maritime routing tool (default: True).
        enable_guide: If True, includes the guide tools (default: True).

    Returns:
        List of LangChain tools for the agent.
    """
    # Core tools: data retrieval + Python analysis
    tools = [
        era5_tool,
        PythonREPLTool(working_dir=".")
    ]

    # Guide tools: methodology and visualization guidance
    if enable_guide:
        tools.append(analysis_guide_tool)
        tools.append(visualization_guide_tool)

    # Routing tools: maritime navigation
    if enable_routing:
        if HAS_ROUTING_DEPS:
            tools.append(routing_tool)
        else:
            print("WARNING: Routing tools requested but dependencies (scgraph) are missing.")

    return tools


# Alias for backward compatibility
get_tools = get_all_tools