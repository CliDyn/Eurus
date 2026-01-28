# Vostok - ERA5 Climate Analysis Agent

<div align="center">
  <img src="assets/vostok_logo.jpeg" alt="Vostok Logo" width="300"/>
  
  <h3><b>Next-Generation Oceanographic & Climate Data Intelligence</b></h3>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![MCP Protocol](https://img.shields.io/badge/MCP-1.0-orange.svg)](https://modelcontextprotocol.io)
  [![Built with Earthmover](https://img.shields.io/badge/Built%20with-Earthmover-blue.svg)](https://earthmover.io)
</div>

---

**Vostok** is a high-performance, intelligent climate analysis agent designed for oceanographers, climate scientists, and data engineers. Built on the cutting-edge **Icechunk** transactional storage engine, Vostok bridges Earthmover's cloud-optimized ERA5 archives with advanced LLM reasoning, enabling seamless, natural language-driven exploration of planetary-scale climate data.

### ❄️ Powered By

This project is made possible by the incredible open-source work from the **[Earthmover](https://earthmover.io)** team:
- **[Icechunk](https://github.com/earth-mover/icechunk)**: The transactional storage engine for Zarr that provides the backbone for our high-performance data access.
- **Arraylake**: The cloud-native data lake that hosts the global ERA5 reanalysis archives used by this agent.

### 🚀 Core Pillars

- **Intelligence-First Analysis**: Leveraging LLMs to translate complex natural language queries into precise data retrieval and scientific analysis.
- **Cloud-Native Performance**: Direct integration with Earthmover's Arraylake and Icechunk/Zarr storage for lightning-fast, subsetted data access without local heavy lifting.
- **Persistent Context**: A sophisticated memory system that tracks conversation history, cached datasets, and analytical insights across sessions.
- **Universal Integration**: Fully compliant with the Model Context Protocol (MCP), allowing Vostok to act as a powerful brain for Claude, IDEs, and custom scientific workflows.

---

## Features

- **Cloud-Optimized Data Retrieval**: Downloads ERA5 reanalysis data directly from Earthmover's Arraylake using Icechunk/Zarr format
- **Persistent Memory**: Conversation history and cached datasets survive between sessions
- **Interactive Python Analysis**: Jupyter kernel with pre-loaded scientific computing libraries
- **Automatic Visualization**: Matplotlib plots with auto-save to `./data/plots/`
- **Intelligent Caching**: Re-uses previously downloaded data automatically
- **MCP Server**: Can be used as a Model Context Protocol server for Claude and other AI assistants

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key
ARRAYLAKE_API_KEY=your_arraylake_api_key
```

## Usage

### Interactive Agent

```bash
python main.py
```

#### Commands
- `/help` - Show help message
- `/clear` - Clear conversation history
- `/cache` - List cached datasets
- `/memory` - Show memory summary
- `/quit` or `q` - Exit

#### Example Queries
- "Show me the sea surface temperature off California for 2023"
- "What's the wind pattern in the Gulf of Mexico this January?"
- "Plot a time series of temperature anomalies in the North Atlantic"
- "Compare SST between El Nino region and the California coast"

### MCP Server

Configure in your Claude Code settings (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "vostok": {
      "command": "python",
      "args": ["/path/to/era_5_agent/mcp_server.py"],
      "env": {
        "ARRAYLAKE_API_KEY": "your_key_here"
      }
    }
  }
}
```

Or run directly:

```bash
python mcp_server.py
```

## Available Variables

| Variable | Description | Units |
|----------|-------------|-------|
| `sst` | Sea Surface Temperature | K |
| `t2` | 2m Air Temperature | K |
| `u10` | 10m U-Wind Component | m/s |
| `v10` | 10m V-Wind Component | m/s |
| `mslp` | Mean Sea Level Pressure | Pa |
| `sp` | Surface Pressure | Pa |
| `tcc` | Total Cloud Cover | 0-1 |
| `tp` | Total Precipitation | m |
| `cp` | Convective Precipitation | m |
| `lsp` | Large-scale Precipitation | m |

## Predefined Regions

- `north_atlantic`, `south_atlantic`
- `north_pacific`, `south_pacific`
- `indian_ocean`
- `california_coast`, `east_coast_us`
- `gulf_of_mexico`, `caribbean`
- `mediterranean`
- `europe`, `asia_east`, `australia`
- `arctic`, `antarctic`
- `nino34`, `nino3`, `nino4` (El Niño regions)
- `global`

## Project Structure

```
era_5_agent/
├── main.py          # Interactive agent entry point
├── config.py        # Configuration and variable catalog
├── memory.py        # Persistent memory system
├── era5_tool.py     # ERA5 data retrieval tool
├── repl_tool.py     # Python REPL with Jupyter kernel
├── mcp_server.py    # MCP server for Claude integration
├── requirements.txt # Python dependencies
├── .env             # API keys (create this)
├── data/            # Downloaded datasets
│   └── plots/       # Generated visualizations
└── .memory/         # Conversation history and cache
```

## Data Storage

- **Downloaded Data**: `./data/*.zarr` - Zarr format datasets
- **Plots**: `./data/plots/*.png` - Generated visualizations
- **Memory**: `./.memory/` - JSON files for conversation history and dataset registry

## Python REPL

The agent includes a persistent Jupyter kernel with pre-loaded libraries:

```python
# Pre-loaded
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load ERA5 data
ds = xr.open_dataset('./data/era5_sst_temporal_20240101_20240107.zarr', engine='zarr')

# Analyze
sst_celsius = ds['sst'] - 273.15
print(f"Mean SST: {float(sst_celsius.mean()):.2f}°C")

# Visualize
ds['sst'].mean(dim='time').plot()
plt.savefig('./data/plots/my_plot.png')
plt.close()
```

## API Reference

### ERA5 Tool Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query_type` | `"temporal"` \| `"spatial"` | Optimization mode |
| `variable_id` | string | ERA5 variable name |
| `start_date` | string | Start date (YYYY-MM-DD) |
| `end_date` | string | End date (YYYY-MM-DD) |
| `min_latitude` | float | Southern bound (-90 to 90) |
| `max_latitude` | float | Northern bound (-90 to 90) |
| `min_longitude` | float | Western bound (0 to 360) |
| `max_longitude` | float | Eastern bound (0 to 360) |
| `region` | string | Predefined region name |

### Query Types

- **`temporal`**: Optimized for time series analysis (long time periods, small geographic area)
- **`spatial`**: Optimized for spatial maps (large geographic area, short time periods)

## License

MIT License

---

<div align="center">
  <p>Special thanks to the <b>Icechunk</b> and <b>Earthmover</b> teams for their pioneering work in cloud-native scientific data storage. Vostok stands on the shoulders of giants.</p>
</div>
