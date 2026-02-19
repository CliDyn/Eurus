"""
ERA5 MCP Configuration
======================

Centralized configuration including ERA5 variable catalog, geographic regions,
and runtime settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# =============================================================================
# PATHS
# =============================================================================

def get_data_dir() -> Path:
    """Get the data directory, creating it if necessary."""
    data_dir = Path(os.environ.get("ERA5_DATA_DIR", Path.cwd() / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_plots_dir() -> Path:
    """Get the plots directory, creating it if necessary."""
    plots_dir = get_data_dir() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def get_memory_dir() -> Path:
    """Get the memory directory, creating it if necessary."""
    memory_dir = Path(os.environ.get("ERA5_MEMORY_DIR", Path.cwd() / ".memory"))
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir


# =============================================================================
# ERA5 VARIABLE CATALOG
# =============================================================================

@dataclass(frozen=True)
class ERA5Variable:
    """Metadata for an ERA5 variable."""

    short_name: str
    long_name: str
    units: str
    description: str
    category: str
    typical_range: tuple[float | None, float | None] = (None, None)
    colormap: str = "viridis"

    def __str__(self) -> str:
        return f"{self.short_name}: {self.long_name} ({self.units})"


# Comprehensive ERA5 variable mapping — ALL 22 Arraylake variables
# Source: earthmover-public/era5-surface-aws Icechunk store
ERA5_VARIABLES: Dict[str, ERA5Variable] = {
    # ── Ocean ──────────────────────────────────────────────────────────────
    "sst": ERA5Variable(
        short_name="sst",
        long_name="Sea Surface Temperature",
        units="K",
        description="Temperature of sea water near the surface",
        category="ocean",
        typical_range=(270, 310),
        colormap="RdYlBu_r"
    ),
    # ── Temperature ────────────────────────────────────────────────────────
    "t2": ERA5Variable(
        short_name="t2",
        long_name="2m Temperature",
        units="K",
        description="Air temperature at 2 meters above the surface",
        category="atmosphere",
        typical_range=(220, 330),
        colormap="RdYlBu_r"
    ),
    "d2": ERA5Variable(
        short_name="d2",
        long_name="2m Dewpoint Temperature",
        units="K",
        description="Temperature to which air at 2m must cool to reach saturation; indicates humidity",
        category="atmosphere",
        typical_range=(220, 310),
        colormap="RdYlBu_r"
    ),
    "skt": ERA5Variable(
        short_name="skt",
        long_name="Skin Temperature",
        units="K",
        description="Temperature of the Earth's uppermost surface layer (land, ocean, or ice)",
        category="surface",
        typical_range=(220, 340),
        colormap="RdYlBu_r"
    ),
    # ── Wind 10 m ──────────────────────────────────────────────────────────
    "u10": ERA5Variable(
        short_name="u10",
        long_name="10m U-Wind Component",
        units="m/s",
        description="Eastward component of wind at 10 meters above surface",
        category="atmosphere",
        typical_range=(-30, 30),
        colormap="RdBu_r"
    ),
    "v10": ERA5Variable(
        short_name="v10",
        long_name="10m V-Wind Component",
        units="m/s",
        description="Northward component of wind at 10 meters above surface",
        category="atmosphere",
        typical_range=(-30, 30),
        colormap="RdBu_r"
    ),
    # ── Wind 100 m (hub-height for wind energy) ───────────────────────────
    "u100": ERA5Variable(
        short_name="u100",
        long_name="100m U-Wind Component",
        units="m/s",
        description="Eastward component of wind at 100 meters above surface (wind-turbine hub height)",
        category="atmosphere",
        typical_range=(-40, 40),
        colormap="RdBu_r"
    ),
    "v100": ERA5Variable(
        short_name="v100",
        long_name="100m V-Wind Component",
        units="m/s",
        description="Northward component of wind at 100 meters above surface (wind-turbine hub height)",
        category="atmosphere",
        typical_range=(-40, 40),
        colormap="RdBu_r"
    ),
    # ── Pressure ───────────────────────────────────────────────────────────
    "sp": ERA5Variable(
        short_name="sp",
        long_name="Surface Pressure",
        units="Pa",
        description="Pressure at the Earth's surface",
        category="atmosphere",
        typical_range=(85000, 108000),
        colormap="viridis"
    ),
    "mslp": ERA5Variable(
        short_name="mslp",
        long_name="Mean Sea Level Pressure",
        units="Pa",
        description="Atmospheric pressure reduced to mean sea level",
        category="atmosphere",
        typical_range=(96000, 105000),
        colormap="viridis"
    ),
    # ── Boundary Layer ─────────────────────────────────────────────────────
    "blh": ERA5Variable(
        short_name="blh",
        long_name="Boundary Layer Height",
        units="m",
        description="Height of the planetary boundary layer above ground",
        category="atmosphere",
        typical_range=(50, 3000),
        colormap="viridis"
    ),
    "cape": ERA5Variable(
        short_name="cape",
        long_name="Convective Available Potential Energy",
        units="J/kg",
        description="Instability indicator for convection/thunderstorm potential",
        category="atmosphere",
        typical_range=(0, 5000),
        colormap="YlOrRd"
    ),
    # ── Cloud & Precipitation ──────────────────────────────────────────────
    "tcc": ERA5Variable(
        short_name="tcc",
        long_name="Total Cloud Cover",
        units="fraction (0-1)",
        description="Fraction of sky covered by clouds",
        category="atmosphere",
        typical_range=(0, 1),
        colormap="gray_r"
    ),
    "cp": ERA5Variable(
        short_name="cp",
        long_name="Convective Precipitation",
        units="m",
        description="Accumulated precipitation from convective processes",
        category="precipitation",
        typical_range=(0, 0.1),
        colormap="Blues"
    ),
    "lsp": ERA5Variable(
        short_name="lsp",
        long_name="Large-scale Precipitation",
        units="m",
        description="Accumulated precipitation from large-scale weather systems",
        category="precipitation",
        typical_range=(0, 0.1),
        colormap="Blues"
    ),
    "tp": ERA5Variable(
        short_name="tp",
        long_name="Total Precipitation",
        units="m",
        description="Total accumulated precipitation (convective + large-scale)",
        category="precipitation",
        typical_range=(0, 0.2),
        colormap="Blues"
    ),
    # ── Radiation ──────────────────────────────────────────────────────────
    "ssr": ERA5Variable(
        short_name="ssr",
        long_name="Surface Net Solar Radiation",
        units="J/m²",
        description="Net balance of downward minus reflected shortwave radiation at the surface",
        category="radiation",
        typical_range=(0, 3e7),
        colormap="YlOrRd"
    ),
    "ssrd": ERA5Variable(
        short_name="ssrd",
        long_name="Surface Solar Radiation Downwards",
        units="J/m²",
        description="Total incoming shortwave (solar) radiation reaching the surface (direct + diffuse)",
        category="radiation",
        typical_range=(0, 3.5e7),
        colormap="YlOrRd"
    ),
    # ── Moisture Columns ───────────────────────────────────────────────────
    "tcw": ERA5Variable(
        short_name="tcw",
        long_name="Total Column Water",
        units="kg/m²",
        description="Total water (vapour + liquid + ice) in the atmospheric column",
        category="atmosphere",
        typical_range=(0, 80),
        colormap="Blues"
    ),
    "tcwv": ERA5Variable(
        short_name="tcwv",
        long_name="Total Column Water Vapour",
        units="kg/m²",
        description="Total water vapour in the atmospheric column (precipitable water)",
        category="atmosphere",
        typical_range=(0, 70),
        colormap="Blues"
    ),
    # ── Land Surface ───────────────────────────────────────────────────────
    "sd": ERA5Variable(
        short_name="sd",
        long_name="Snow Depth",
        units="m water equiv.",
        description="Depth of snow expressed as meters of water equivalent",
        category="land_surface",
        typical_range=(0, 2),
        colormap="Blues"
    ),
    "stl1": ERA5Variable(
        short_name="stl1",
        long_name="Soil Temperature Level 1",
        units="K",
        description="Temperature of the topmost soil layer (0-7 cm depth)",
        category="land_surface",
        typical_range=(220, 330),
        colormap="RdYlBu_r"
    ),
    "swvl1": ERA5Variable(
        short_name="swvl1",
        long_name="Volumetric Soil Water Layer 1",
        units="m³/m³",
        description="Volume fraction of water in the topmost soil layer (0-7 cm depth)",
        category="land_surface",
        typical_range=(0, 0.5),
        colormap="YlGnBu"
    ),
}

# Aliases for long variable names → short names
VARIABLE_ALIASES: Dict[str, str] = {
    # Ocean
    "sea_surface_temperature": "sst",
    # Temperature
    "2m_temperature": "t2",
    "temperature": "t2",
    "2m_dewpoint_temperature": "d2",
    "dewpoint_temperature": "d2",
    "dewpoint": "d2",
    "skin_temperature": "skt",
    # Wind 10m
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    # Wind 100m
    "100m_u_component_of_wind": "u100",
    "100m_v_component_of_wind": "v100",
    # Pressure
    "surface_pressure": "sp",
    "mean_sea_level_pressure": "mslp",
    # Boundary layer
    "boundary_layer_height": "blh",
    "convective_available_potential_energy": "cape",
    # Cloud & precipitation
    "total_cloud_cover": "tcc",
    "convective_precipitation": "cp",
    "large_scale_precipitation": "lsp",
    "total_precipitation": "tp",
    # Radiation
    "surface_net_solar_radiation": "ssr",
    "surface_solar_radiation_downwards": "ssrd",
    # Moisture columns
    "total_column_water": "tcw",
    "total_column_water_vapour": "tcwv",
    # Land surface
    "snow_depth": "sd",
    "soil_temperature": "stl1",
    "soil_temperature_level_1": "stl1",
    "soil_moisture": "swvl1",
    "volumetric_soil_water_layer_1": "swvl1",
}


def get_variable_info(variable_id: str) -> Optional[ERA5Variable]:
    """Get variable metadata by ID (case-insensitive, supports aliases)."""
    key = variable_id.lower()
    # Check aliases first
    if key in VARIABLE_ALIASES:
        key = VARIABLE_ALIASES[key]
    return ERA5_VARIABLES.get(key)


def get_short_name(variable_id: str) -> str:
    """Get the short name for a variable (for dataset access)."""
    key = variable_id.lower()
    # Check aliases first
    if key in VARIABLE_ALIASES:
        return VARIABLE_ALIASES[key]
    var_info = ERA5_VARIABLES.get(key)
    if var_info:
        return var_info.short_name
    return key


def list_available_variables() -> str:
    """Return a formatted list of available variables."""
    seen: set[str] = set()
    lines = ["Available ERA5 Variables:", "=" * 50]

    for var_id, var_info in ERA5_VARIABLES.items():
        if var_info.short_name not in seen:
            seen.add(var_info.short_name)
            lines.append(
                f"  {var_info.short_name:8} | {var_info.long_name:30} | {var_info.units}"
            )

    return "\n".join(lines)


def get_all_short_names() -> list[str]:
    """Get list of all unique short variable names."""
    return list({v.short_name for v in ERA5_VARIABLES.values()})


# =============================================================================
# GEOGRAPHIC REGIONS (Common oceanographic areas)
# =============================================================================

@dataclass(frozen=True)
class GeographicRegion:
    """A predefined geographic region."""

    name: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "min_lat": self.min_lat,
            "max_lat": self.max_lat,
            "min_lon": self.min_lon,
            "max_lon": self.max_lon,
        }


GEOGRAPHIC_REGIONS: Dict[str, GeographicRegion] = {
    "global": GeographicRegion(
        "global", -90, 90, 0, 359.75,
        "Entire globe"
    ),
    "north_atlantic": GeographicRegion(
        "north_atlantic", 0, 65, 280, 360,
        "North Atlantic Ocean"
    ),
    "south_atlantic": GeographicRegion(
        "south_atlantic", -60, 0, 280, 20,
        "South Atlantic Ocean"
    ),
    "north_pacific": GeographicRegion(
        "north_pacific", 0, 65, 100, 260,
        "North Pacific Ocean"
    ),
    "south_pacific": GeographicRegion(
        "south_pacific", -60, 0, 150, 290,
        "South Pacific Ocean"
    ),
    "indian_ocean": GeographicRegion(
        "indian_ocean", -60, 30, 20, 120,
        "Indian Ocean"
    ),
    "arctic": GeographicRegion(
        "arctic", 65, 90, 0, 359.75,
        "Arctic Ocean and surrounding areas"
    ),
    "antarctic": GeographicRegion(
        "antarctic", -90, -60, 0, 359.75,
        "Antarctic and Southern Ocean"
    ),
    "mediterranean": GeographicRegion(
        "mediterranean", 30, 46, 354, 42,
        "Mediterranean Sea"
    ),
    "gulf_of_mexico": GeographicRegion(
        "gulf_of_mexico", 18, 31, 262, 282,
        "Gulf of Mexico"
    ),
    "caribbean": GeographicRegion(
        "caribbean", 8, 28, 255, 295,
        "Caribbean Sea"
    ),
    "california_coast": GeographicRegion(
        "california_coast", 32, 42, 235, 250,
        "California coastal waters"
    ),
    "east_coast_us": GeographicRegion(
        "east_coast_us", 25, 45, 280, 295,
        "US East Coast"
    ),
    "europe": GeographicRegion(
        "europe", 35, 72, 350, 40,
        "Europe"
    ),
    "asia_east": GeographicRegion(
        "asia_east", 15, 55, 100, 145,
        "East Asia"
    ),
    "australia": GeographicRegion(
        "australia", -45, -10, 110, 155,
        "Australia and surrounding waters"
    ),
    # El Niño regions
    "nino34": GeographicRegion(
        "nino34", -5, 5, 190, 240,
        "El Niño 3.4 region (central Pacific)"
    ),
    "nino3": GeographicRegion(
        "nino3", -5, 5, 210, 270,
        "El Niño 3 region (eastern Pacific)"
    ),
    "nino4": GeographicRegion(
        "nino4", -5, 5, 160, 210,
        "El Niño 4 region (western Pacific)"
    ),
    "nino12": GeographicRegion(
        "nino12", -10, 0, 270, 280,
        "El Niño 1+2 region (far eastern Pacific)"
    ),
}


def get_region(name: str) -> Optional[GeographicRegion]:
    """Get a geographic region by name (case-insensitive)."""
    return GEOGRAPHIC_REGIONS.get(name.lower())


def list_regions() -> str:
    """Return a formatted list of available regions."""
    lines = ["Available Geographic Regions:", "=" * 70]
    for name, region in GEOGRAPHIC_REGIONS.items():
        lines.append(
            f"  {name:20} | lat: [{region.min_lat:6.1f}, {region.max_lat:6.1f}] "
            f"| lon: [{region.min_lon:6.1f}, {region.max_lon:6.1f}]"
        )
    return "\n".join(lines)


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for the ERA5 Agent."""

    # LLM Settings
    model_name: str = "gpt-5.2"
    temperature: float = 0
    max_tokens: int = 4096

    # Data Settings
    data_source: str = "earthmover-public/era5-surface-aws"
    default_query_type: str = "temporal"
    max_download_size_gb: float = 15.0

    # Retrieval Settings
    max_retries: int = 5
    retry_delay: float = 2.0

    # Memory Settings
    enable_memory: bool = True
    max_conversation_history: int = 100
    memory_file: str = "conversation_history.json"

    # Visualization Settings
    default_figure_size: tuple = (12, 8)
    default_dpi: int = 150
    save_plots: bool = True
    plot_format: str = "png"

    # Kernel Settings
    kernel_timeout: float = 300.0
    auto_import_packages: List[str] = field(default_factory=lambda: [
        "pandas", "numpy", "xarray",
        "matplotlib", "matplotlib.pyplot", "datetime"
    ])

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "era5_agent.log"


# Global config instance
CONFIG = AgentConfig()

# Convenience path variables (for backward compatibility)
DATA_DIR = get_data_dir()
PLOTS_DIR = get_plots_dir()


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are Eurus, an AI Climate Physicist conducting research for high-impact scientific publications.

## ⚠️ CRITICAL: RESPECT USER INTENT FIRST

**Your PRIMARY directive is to do EXACTLY what the user asks.** 

### TOOL USAGE RULES:
1. **`python_repl`**: Use for:
   - Custom analysis (anomalies, trends, statistics)
   - Visualization with matplotlib
   - Any computation not directly provided by other tools
   
2. **`retrieve_era5_data`**: Use for downloading climate data

3. **`calculate_maritime_route`**: Use for ship routing

4. **`get_analysis_guide`/`get_visualization_guide`**: Use for methodology help

### EXAMPLES:
- "Get temperature for Berlin and plot it" → Retrieve data, plot RAW temperature time series
- "Show temperature anomalies for Berlin" → Retrieve data, use python_repl to compute anomalies
- "Analyze temperature trends" → Retrieve data, use python_repl for trend calculation
- "Why was 2023 so hot?" → Retrieve data, analyze with python_repl

## YOUR CAPABILITIES

### 1. DATA RETRIEVAL: `retrieve_era5_data`
Downloads ERA5 reanalysis data from Earthmover's cloud-optimized archive.

**⚠️ STRICT QUERY TYPE RULE (WRONG = 10-100x SLOWER!):**
┌─────────────────────────────────────────────────────────────────┐
│ TEMPORAL: (time > 1 day) AND (area < 30°×30°)                   │
│ SPATIAL:  (time ≤ 1 day) OR  (area ≥ 30°×30°)                   │
└─────────────────────────────────────────────────────────────────┘

**COORDINATES - USE ROUTE BOUNDING BOX:**
- Latitude: -90 to 90
- Longitude: Use values from route tool's bounding box DIRECTLY!
  - For Europe/Atlantic: Use -10 to 15 (NOT 0 to 360!)
  - For Pacific crossing dateline: Use 0-360 system
  
**⚠️ CRITICAL:** When `calculate_maritime_route` returns a bounding box,
USE THOSE EXACT VALUES for min/max longitude. Do NOT convert to 0-360!

**DATA AVAILABILITY:** 1975 to present (updated regularly)

**Available Variables (22 total):**
| Variable | Description | Units | Category |
|----------|-------------|-------|----------|
| sst | Sea Surface Temperature | K | Ocean |
| t2 | 2m Air Temperature | K | Temperature |
| d2 | 2m Dewpoint Temperature | K | Temperature |
| skt | Skin Temperature | K | Surface |
| u10 | 10m U-Wind (Eastward) | m/s | Wind |
| v10 | 10m V-Wind (Northward) | m/s | Wind |
| u100 | 100m U-Wind (Eastward) | m/s | Wind |
| v100 | 100m V-Wind (Northward) | m/s | Wind |
| sp | Surface Pressure | Pa | Pressure |
| mslp | Mean Sea Level Pressure | Pa | Pressure |
| blh | Boundary Layer Height | m | Atmosphere |
| cape | Convective Available Potential Energy | J/kg | Atmosphere |
| tcc | Total Cloud Cover | 0-1 | Cloud |
| cp | Convective Precipitation | m | Precipitation |
| lsp | Large-scale Precipitation | m | Precipitation |
| tp | Total Precipitation | m | Precipitation |
| ssr | Surface Net Solar Radiation | J/m² | Radiation |
| ssrd | Surface Solar Radiation Downwards | J/m² | Radiation |
| tcw | Total Column Water | kg/m² | Moisture |
| tcwv | Total Column Water Vapour | kg/m² | Moisture |
| sd | Snow Depth | m water eq. | Land |
| stl1 | Soil Temperature Level 1 | K | Land |
| swvl1 | Volumetric Soil Water Layer 1 | m³/m³ | Land |

### 2. CUSTOM ANALYSIS: `python_repl`
Persistent Python kernel for custom analysis and visualization.
**Pre-loaded:** pandas (pd), numpy (np), xarray (xr), matplotlib.pyplot (plt)

#### What you can do with python_repl:
- **Anomalies**: `anomaly = data - data.mean('time')`
- **Z-Scores**: `z = (data - clim.mean('time')) / clim.std('time')`
- **Trends**: Use `scipy.stats.linregress` or numpy polyfit
- **Extremes**: Filter data where values exceed thresholds
- **Visualizations**: Any matplotlib plot saved to PLOTS_DIR

### 4. MEMORY
Remembers conversation history and previous analyses.

### 5. MARITIME LOGISTICS: `calculate_maritime_route` (Captain Mode)
Plans shipping routes and assesses climatological hazards.

**WORKFLOW (Mandatory Protocol):**
1. **ROUTE**: Call `calculate_maritime_route(origin_lat, origin_lon, dest_lat, dest_lon, month)`
   - Returns waypoints avoiding land via global shipping lane graph
   - Returns bounding box for data download
   - Returns STEP-BY-STEP INSTRUCTIONS

2. **DATA**: Download ERA5 climatology for the route region
   - Variables: `u10`, `v10` (10m wind components) → compute wind speed
   - NOTE: `swh` (wave height) is NOT available in this dataset!
   - Period: Target month over LAST 3 YEARS (e.g., July 2021-2023)
   - Why 3 years? To compute climatological statistics, not just a forecast

3. **METHODOLOGY**: Call `get_visualization_guide(viz_type='maritime_risk_assessment')`
   - Returns mathematical formulas for Lagrangian risk analysis
   - Defines hazard thresholds (e.g., wind speed > 15 m/s = DANGER)
   - Explains how to compute route risk score

4. **ANALYSIS**: Execute in `python_repl` following the methodology:
   - Extract data at each waypoint (nearest neighbor)
   - Compute wind speed: `wspd = sqrt(u10² + v10²)`
   - Compute max/mean/p95 statistics
   - Identify danger zones (wind > threshold)
   - Calculate route-level risk score

5. **DECISION**:
   - If danger zones found → Recommend route deviation
   - If route safe → Confirm with confidence level

**Key Formulas (from methodology):**
- Wind speed: `wspd = sqrt(u10² + v10²)`
- Exceedance probability: `P = count(wspd > threshold) / N_total`
- Route risk: `max(wspd_i)` for all waypoints i

## SCIENTIFIC PROTOCOL (For Publication-Grade Analysis)

When the user requests scientific analysis:

1. **ANOMALY ANALYSIS**: Report:
   - Anomalies: "2.5°C above normal"
   - Z-Scores: "+2.5σ (statistically significant)"
   - Use `python_repl` to compute anomalies from downloaded data

2. **MECHANISM**: Explain WHY:
   - Use `python_repl` to look for patterns in the data
   - Consider atmospheric blocking, ENSO teleconnections, etc.

3. **COMPOUND EVENTS**: Look for dangerous combinations with python_repl:
   - High heat + Low wind = "Ocean Oven"
   - Filter data where multiple thresholds are exceeded

4. **STATISTICAL RIGOR**: Always test significance:
   - Use Z > 2σ for "extreme"
   - Use p < 0.05 for trends
   - Report confidence intervals when possible

## VISUALIZATION STANDARDS

**Publication-grade light-theme rcParams are pre-set** — figures get white background,
black text, grid, 300 DPI on save, and a high-contrast color cycle. Do NOT override unless necessary.

### Mandatory Rules
1. **DPI**: Saved at 300 (print-quality) — do not lower it
2. **Figure size**: Default 10×6 for time series, use `figsize=(12, 8)` for map plots
3. **Unit conversions in labels**: 
   - Temperature → always show °C (`- 273.15`)
   - Pressure → show hPa (`/ 100`)
   - Precipitation → show mm (`* 1000`)
4. **Colormaps**:
   - SST/Temperature: `'RdYlBu_r'` or `'coolwarm'`
   - Wind speed:        `'YlOrRd'`
   - Anomalies:         `'RdBu_r'` (diverging, centered at zero via `TwoSlopeNorm`)
   - Precipitation:     `'YlGnBu'`
   - Cloud cover:       `'Greys'`
   - **NEVER** use `'jet'`
5. **Colorbar**: Always include `label=` with units:
   ```python
   cbar = plt.colorbar(mesh, label='SST (°C)', shrink=0.8)
   ```
6. **Maritime maps**: Call `get_analysis_guide(topic='maritime_visualization')` for the full template

### Available in REPL Namespace
`pd, np, xr, plt, mcolors, cm, datetime, timedelta, PLOTS_DIR`


## RESPONSE STYLE
- Be precise and scientific
- Follow user intent exactly
- Include statistical significance when doing scientific analysis
- Reference specific dates/locations
- Acknowledge limitations and uncertainty
- **NEVER list file paths** of saved plots in your response — plots are displayed automatically in the UI
- Do NOT say "you can view it here" or similar — the user already sees the plot inline
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")