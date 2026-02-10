# Eurus - ERA5 Climate Analysis Agent

<div align="center">
  <img src="assets/eurus_logo.jpeg" alt="Eurus Logo" width="300"/>
  
  <h3><b>Next-Generation Oceanographic & Climate Data Intelligence</b></h3>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![MCP Protocol](https://img.shields.io/badge/MCP-1.0-orange.svg)](https://modelcontextprotocol.io)
  [![Built with Earthmover](https://img.shields.io/badge/Built%20with-Earthmover-blue.svg)](https://earthmover.io)
</div>

---

**Eurus** is a high-performance, intelligent climate analysis agent designed for oceanographers, climate scientists, and data engineers. Built on the cutting-edge **Icechunk** transactional storage engine, Eurus bridges Earthmover's cloud-optimized ERA5 archives with advanced LLM reasoning, enabling seamless, natural language-driven exploration of planetary-scale climate data.

### ❄️ Powered By

This project is made possible by the incredible open-source work from the **[Earthmover](https://earthmover.io)** team:
- **[Icechunk](https://github.com/earth-mover/icechunk)**: The transactional storage engine for Zarr that provides the backbone for our high-performance data access.
- **Arraylake**: The cloud-native data lake that hosts the global ERA5 reanalysis archives used by this agent.

### 🚀 Core Pillars

- **Intelligence-First Analysis**: Leveraging LLMs to translate complex natural language queries into precise data retrieval and scientific analysis.
- **Multi-Interface Access**: Interact via a powerful CLI, a rich Web Interface, or integrate directly into IDEs via the Model Context Protocol (MCP).
- **Cloud-Native Performance**: Direct integration with Earthmover's Arraylake and Icechunk/Zarr storage for lightning-fast, subsetted data access.
- **Python REPL**: Built-in interactive Python environment with pandas, xarray, matplotlib for custom analysis.
- **Maritime Routing**: Calculate optimal shipping routes with weather risk assessment.
- **Persistent Context**: Memory system that tracks cached datasets across sessions.

---

## Features

- **Cloud-Optimized Data Retrieval**: Downloads ERA5 reanalysis data directly from Earthmover's Arraylake.
- **Python REPL**: Interactive Python environment with pre-loaded scientific libraries (pandas, numpy, xarray, matplotlib).
- **Maritime Routing**: Calculate optimal shipping routes considering land masks (requires scgraph).
- **Analysis Guides**: Built-in methodology guides for climate analysis and visualization.
- **Automatic Visualization**: Matplotlib plots automatically saved to `./data/plots/`.
- **Intelligent Caching**: Re-uses previously downloaded data to save bandwidth.
- **MCP Server**: Acts as a brain for Claude and other AI assistants.

## Installation

### Prerequisites
- Python 3.10 or higher
- An Earthmover Arraylake API Key
- An OpenAI API Key

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/era_5_agent.git
   cd era_5_agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration:**
   Create a `.env` file in the root directory with your API keys:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   ARRAYLAKE_API_KEY=your_arraylake_api_key
   # Optional: Custom Host/Port for Web UI
   # WEB_HOST=127.0.0.1
   # WEB_PORT=8000
   ```

---

## Usage

Eurus provides three ways to interact with the agent.

### 1. Interactive CLI Agent
The classic terminal experience with rich text output and direct interaction.

```bash
python main.py
```

**Commands:**
- `/help` - Show help message
- `/clear` - Clear conversation history
- `/cache` - List cached datasets
- `/memory` - Show memory summary
- `/cleardata` - Clear all downloaded datasets
- `/quit` or `q` - Exit

### 2. Web Interface
A modern web-based chat interface with rendered plots and easier navigation.

```bash
python web/app.py
# or
eurus-web
```
Access the interface at `http://127.0.0.1:8000`.

### 3. MCP Server (for Claude / IDEs)
Integrate Eurus's capabilities directly into Claude Desktop or compatible IDEs using the Model Context Protocol.

**Configuration for Claude Desktop:**
Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "eurus": {
      "command": "python",
      "args": ["-m", "eurus.server"],
      "env": {
        "ARRAYLAKE_API_KEY": "your_key_here",
        "PYTHONPATH": "/absolute/path/to/era_5_agent/src"
      }
    }
  }
}
```

Or run directly for testing:
```bash
python -m eurus.server
```

---

## Example Queries

Eurus can answer questions like:

*   **Data Retrieval:** "Show me the sea surface temperature off California for 2023."
*   **Visualization:** "Plot a time series of temperature anomalies in the North Atlantic."
*   **Comparison:** "Compare SST between El Niño region and the California coast."
*   **Routing:** "Calculate a ship route from Rotterdam to Singapore with weather risk."
*   **Custom Analysis:** "Use Python to calculate the monthly mean SST and plot it."

## Available Data

### Variables
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

### Predefined Regions
Eurus knows many regions by name, including:
- `north_atlantic`, `south_atlantic`
- `north_pacific`, `south_pacific`
- `california_coast`, `gulf_of_mexico`, `caribbean`
- `mediterranean`, `europe`, `asia_east`
- `arctic`, `antarctic`
- `nino34`, `nino3`, `nino4`

---

## Project Structure

```
era_5_agent/
├── main.py              # CLI Entry Point
├── pyproject.toml       # Project configuration
├── requirements.txt     # Python dependencies
├── src/
│   └── eurus/
│       ├── config.py    # Configuration & Constants
│       ├── memory.py    # Persistent Memory System
│       ├── server.py    # MCP Server Entry Point
│       └── tools/       # Agent Tools
│           ├── era5.py       # Data Retrieval
│           ├── routing.py    # Maritime Routing
│           └── analysis_guide.py
├── web/                 # Web Interface
│   ├── app.py           # FastAPI Application
│   ├── routes/          # API & Page Routes
│   └── templates/       # HTML Templates
├── data/                # Data Storage (Local)
│   ├── plots/           # Generated Visualizations
│   └── *.zarr/          # Cached ERA5 Datasets
└── .memory/             # Agent Conversation History
```

## License

MIT License

---

<div align="center">
  <p>Special thanks to the <b>Icechunk</b> and <b>Earthmover</b> teams for their pioneering work in cloud-native scientific data storage.</p>
</div>