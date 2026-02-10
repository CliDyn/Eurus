"""
Analysis Guide Tool
====================
Provides methodological guidance for climate data analysis using python_repl.

This tool returns TEXT INSTRUCTIONS (not executable code!) for:
- What libraries to import
- How to structure the analysis
- How to interpret results
- Best practices for visualization

The agent uses python_repl to execute the actual analysis.
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


# =============================================================================
# ANALYSIS GUIDES
# =============================================================================

ANALYSIS_GUIDES = {
    # -------------------------------------------------------------------------
    # DATA OPERATIONS
    # -------------------------------------------------------------------------
    "load_data": """
## Loading ERA5 Data

### Required Imports
```python
import xarray as xr
```

### Method
1. Use `xr.open_zarr(dataset_path)` to load the dataset
2. The dataset contains dimensions: time, latitude, longitude
3. Data variables depend on what was downloaded (sst, t2m, u10, v10, etc.)

### Key Operations
- `ds.info()` - See dataset structure
- `ds.coords` - Check coordinate names and ranges
- `ds.data_vars` - List available variables
- `ds['variable_name']` - Access specific variable

### Unit Conversions
- Temperature (K → °C): Subtract 273.15
- Pressure (Pa → hPa): Divide by 100
- Precipitation (m → mm): Multiply by 1000
""",

    "spatial_subset": """
## Spatial Subsetting

### Method
Use xarray's `.sel()` with slice for geographic regions:
```python
subset = ds.sel(
    latitude=slice(north, south),  # Note: often reversed!
    longitude=slice(west, east)
)
```

### Important Notes
- Check if latitude is ascending or descending first
- For global datasets crossing dateline, may need special handling
- Use `.where()` for irregular regions with masks
""",

    "temporal_subset": """
## Temporal Subsetting

### Method
```python
# Specific date range
subset = ds.sel(time=slice('2024-01-01', '2024-01-31'))

# Specific month across years
january_data = ds.sel(time=ds.time.dt.month == 1)

# Specific season (DJF, MAM, JJA, SON)
winter = ds.sel(time=ds.time.dt.season == 'DJF')
```

### Aggregations
- `.mean(dim='time')` - Temporal mean
- `.std(dim='time')` - Standard deviation
- `.resample(time='1M').mean()` - Monthly means
- `.groupby('time.month').mean()` - Climatological monthly means
""",

    # -------------------------------------------------------------------------
    # STATISTICAL ANALYSIS
    # -------------------------------------------------------------------------
    "anomalies": """
## Calculating Anomalies

### Concept
Anomaly = Actual Value - Climatological Mean

### Method
1. Calculate climatology (long-term mean by month/day)
2. Subtract from actual values

```python
# Monthly climatology
climatology = ds.groupby('time.month').mean(dim='time')
anomalies = ds.groupby('time.month') - climatology
```

### Interpretation
- Positive anomaly: Warmer/wetter than normal
- Negative anomaly: Cooler/drier than normal
- Units remain the same as original variable
""",

    "zscore": """
## Z-Score (Standardized Anomalies)

### Concept
Z = (Value - Mean) / Standard Deviation

### Interpretation
- Z = 0: Average conditions
- Z = ±1: Within 1 standard deviation (68% of data)
- Z = ±2: Unusual (only ~5% of data)
- Z = ±3: Extreme (only ~0.3% of data)

### Method
```python
mean = ds.mean(dim='time')
std = ds.std(dim='time')
zscore = (ds - mean) / std
```

### Use Cases
- Comparing extremity across different variables
- Comparing regions with different variability
- Identifying statistically significant departures
""",

    "trend_analysis": """
## Linear Trend Analysis

### Required Imports
```python
from scipy import stats
import numpy as np
```

### Method
For each grid point, fit a linear regression of value vs time.

### Key Outputs
- Slope: Rate of change (units/year or units/decade)
- P-value: Statistical significance (p < 0.05 is significant)
- R²: Variance explained by trend

### Interpretation
- Report trends per decade for climate (more intuitive)
- Only trust trends where p < 0.05
- Consider using stippling for significant areas on maps

### Common Pitfalls
- Short time series have uncertain trends
- Autocorrelation can inflate significance
- Nonlinear changes may be missed
""",

    "eof_analysis": """
## EOF/PCA Analysis (Empirical Orthogonal Functions)

### Concept
Decomposes spatiotemporal data into:
1. Spatial patterns (EOFs/loadings)
2. Time series (Principal Components)
3. Variance explained by each mode

### Required Imports
```python
from sklearn.decomposition import PCA
# or
from eofs.xarray import Eof
```

### Interpretation
- EOF1: Dominant mode of variability
- PC1: How EOF1 varies in time
- Variance %: Importance of each mode

### Use Cases
- Finding dominant climate patterns (ENSO, NAO)
- Data compression/filtering
- Identifying teleconnections
""",

    # -------------------------------------------------------------------------
    # CLIMATE INDICES
    # -------------------------------------------------------------------------
    "climate_indices": """
## Climate Indices (ENSO, NAO, etc.)

### ENSO (Niño 3.4 Index)
- Region: 5°S-5°N, 170°W-120°W
- Calculate: SST anomaly averaged over region
- El Niño: Index > +0.5°C for 5+ months
- La Niña: Index < -0.5°C for 5+ months

### NAO (North Atlantic Oscillation)
- Pressure difference: Azores High minus Icelandic Low
- Positive NAO: Strong pressure gradient → mild European winters
- Negative NAO: Weak gradient → cold European winters

### PDO (Pacific Decadal Oscillation)
- Leading EOF of North Pacific SST (north of 20°N)
- Multidecadal oscillation (20-30 year phases)

### AMO (Atlantic Multidecadal Oscillation)
- Detrended North Atlantic SST average
- ~60-70 year cycle
""",

    # -------------------------------------------------------------------------
    # EXTREME EVENTS
    # -------------------------------------------------------------------------
    "extremes": """
## Extreme Event Analysis

### Percentile-Based
- Heat extreme: Values > 95th percentile
- Cold extreme: Values < 5th percentile
- Calculate percentiles from historical baseline period

### Threshold-Based
- Marine heatwave: SST > climatology + 1 standard deviation
- Drought: Precipitation < 10th percentile for 3+ months

### Return Periods (Extreme Value Theory)
- Fit GEV distribution to annual maxima
- Calculate return levels (e.g., 100-year event magnitude)
- Requires 20+ years of data for reliability

### Compound Extremes
- Multiple hazards co-occurring (hot + dry, high waves + strong winds)
- Assess joint probability
""",

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------
    "visualization_spatial": """
## Spatial Map Visualization

### Ready-to-Adapt Template
The REPL has a dark theme pre-set. All plots automatically get dark background,
white labels, and grid. Just focus on the data.

```python
import numpy as np

# ── 1. Load data ──
ds = xr.open_zarr('path/to/data.zarr')
var = ds['sst']  # or t2, u10, etc.

# ── 2. Select time slice and convert units ──
data = var.isel(time=0) - 273.15  # K → °C

# ── 3. Choose colormap by variable type ──
# SST / Temperature: 'RdYlBu_r' or 'coolwarm'
# Wind speed:        'YlOrRd'
# Anomalies:         'RdBu_r' (with TwoSlopeNorm for centering at 0)
# Precipitation:     'YlGnBu'
# Cloud cover:       'Greys'
# NEVER use 'jet'!

# ── 4. Plot ──
fig, ax = plt.subplots(figsize=(12, 8))

lons, lats = np.meshgrid(data.longitude.values, data.latitude.values)
mesh = ax.pcolormesh(lons, lats, data.values,
                     cmap='RdYlBu_r', shading='auto')

# Colorbar
cbar = plt.colorbar(mesh, ax=ax, label='SST (°C)', shrink=0.8, pad=0.02)

# Labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Sea Surface Temperature — July 15, 2023', fontweight='bold')

plt.savefig(f'{PLOTS_DIR}/sst_map.png')
plt.close()
```

### Colormaps Reference
| Variable | Colormap | Units |
|----------|----------|-------|
| SST, T2m | `RdYlBu_r` | °C (subtract 273.15) |
| Wind speed | `YlOrRd` | m/s |
| Anomalies | `RdBu_r` | same as original |
| Precipitation | `YlGnBu` | mm (multiply by 1000) |
| Pressure | `viridis` | hPa (divide by 100) |
| Cloud cover | `Greys` | fraction (0-1) |

### With Cartopy (optional, for coastlines)
```python
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig, ax = plt.subplots(figsize=(12, 8),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#c0c0c0')
    ax.add_feature(cfeature.LAND, facecolor='#2a2e36', zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='#555')
    mesh = ax.pcolormesh(lons, lats, data.values,
                         cmap='RdYlBu_r', shading='auto',
                         transform=ccrs.PlateCarree())
except ImportError:
    pass  # Fall back to plain axes
```
""",

    "visualization_timeseries": """
## Time Series Visualization

### Ready-to-Adapt Template
Dark theme is pre-set. The Eurus color cycle starts with sky-blue (#4fc3f7),
then coral, green, amber — all vivid on dark backgrounds.

```python
import matplotlib.dates as mdates
import numpy as np

# ── 1. Load and prepare data ──
ds = xr.open_zarr('path/to/data.zarr')
var = ds['t2'] - 273.15  # K → °C

# ── 2. Area-average if spatial data ──
ts = var.mean(dim=['latitude', 'longitude'])

# ── 3. Plot ──
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts.time.values, ts.values, linewidth=1.5, label='2m Temperature')

# ── 4. Optional: rolling mean overlay ──
if len(ts) > 48:  # Enough data for smoothing
    rolling = ts.rolling(time=24, center=True).mean()
    ax.plot(rolling.time.values, rolling.values,
            color='#ff7043', linewidth=2.5, alpha=0.9, label='24h rolling mean')

# ── 5. Date formatting ──
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
fig.autofmt_xdate(rotation=30)

# ── 6. Labels ──
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.set_title('2m Air Temperature — Berlin, August 2019', fontweight='bold')
ax.legend()

plt.savefig(f'{PLOTS_DIR}/temperature_timeseries.png')
plt.close()
```

### Enhancements
- **Trend line**: `from scipy.stats import linregress` → overlay with dashed line
- **Uncertainty band**: `ax.fill_between(time, mean-std, mean+std, alpha=0.2)`
- **Event markers**: `ax.axvline(event_date, color='#ef5350', ls='--', alpha=0.7)`
- **Multi-variable**: Plot on twin axes with `ax2 = ax.twinx()`

### Date Formatting Cheatsheet
| Period | Locator | Formatter |
|--------|---------|----------|
| Hours (1 day) | `HourLocator(interval=3)` | `%H:%M` |
| Days (1 week) | `DayLocator()` | `%b %d` |
| Days (1 month) | `WeekdayLocator()` | `%b %d` |
| Months (1 year) | `MonthLocator()` | `%b %Y` |
""",

    # -------------------------------------------------------------------------
    # MARITIME ANALYSIS
    # -------------------------------------------------------------------------
    "maritime_route": """
## Maritime Route Analysis

### Workflow
1. **Calculate route** using `calculate_maritime_route` tool
   - Provides waypoints along shipping lanes
   - Returns coordinates and distances

2. **Download climate data** for route region
   - Extend bounding box by 1° around route
   - Get relevant variables: wind (u10, v10), waves if available

3. **Extract conditions along route**
   - Sample data at each waypoint
   - Calculate statistics (mean, max, percentiles)

### Risk Assessment
- Wind speed thresholds:
  - Safe: < 10 m/s
  - Caution: 10-17 m/s
  - Danger: 17-24 m/s
  - Extreme: > 24 m/s

- Calculate percentage of route in each category
- Identify highest-risk segments

### Visualization
- Plot route on map with risk color coding
- Add wind vectors along route
- Show climatological wind roses at key points
""",

    "maritime_visualization": """
## Maritime Route Risk Visualization

### Ready-to-Adapt Template
Copy and modify this code in `python_repl`. Replace variable names with your actual data.

```python
import numpy as np

# ── 1. Load wind data ──
ds = xr.open_zarr('path/to/wind_data.zarr')
u10 = ds['u10']
v10 = ds['v10']
wspd = np.sqrt(u10**2 + v10**2)  # Wind speed

# ── 2. Compute climatological mean wind speed (spatial map) ──
wspd_mean = wspd.mean(dim='time')

# ── 3. Define waypoints (from routing tool) ──
waypoint_lats = [53.5, 54.0, 53.8, 52.4]  # example
waypoint_lons = [8.5, 6.0, 5.2, 4.9]      # example

# ── 4. Extract wind speed at each waypoint ──
wp_wspd = []
for lat, lon in zip(waypoint_lats, waypoint_lons):
    val = float(wspd_mean.sel(latitude=lat, longitude=lon, method='nearest').values)
    wp_wspd.append(val)

# ── 5. Risk categories ──
def risk_color(speed):
    if speed < 10:   return '#66bb6a'  # Safe - green
    if speed < 17:   return '#ffca28'  # Caution - amber
    if speed < 24:   return '#ff7043'  # Danger - orange/coral
    return '#ef5350'                    # Extreme - red

colors = [risk_color(w) for w in wp_wspd]

# ── 6. Plot ──
fig, ax = plt.subplots(figsize=(12, 8))

# Background heatmap of mean wind speed
lons_2d, lats_2d = np.meshgrid(wspd_mean.longitude.values, wspd_mean.latitude.values)
mesh = ax.pcolormesh(lons_2d, lats_2d, wspd_mean.values,
                     cmap='YlOrRd', shading='auto', alpha=0.7)
cbar = plt.colorbar(mesh, ax=ax, label='Mean Wind Speed (m/s)', shrink=0.8, pad=0.02)

# Route line
ax.plot(waypoint_lons, waypoint_lats, '--', color='#1f77b4', linewidth=1.5,
        alpha=0.7, zorder=5)

# Risk-colored waypoint dots
for lon, lat, c, wspd_val in zip(waypoint_lons, waypoint_lats, colors, wp_wspd):
    ax.scatter(lon, lat, c=c, s=80, edgecolors='black', linewidths=0.8,
               zorder=10)

# Origin / Destination labels
ax.annotate('ORIGIN', (waypoint_lons[0], waypoint_lats[0]),
            textcoords='offset points', xytext=(8, 8),
            fontsize=9, color='black', fontweight='bold')
ax.annotate('DEST', (waypoint_lons[-1], waypoint_lats[-1]),
            textcoords='offset points', xytext=(8, -12),
            fontsize=9, color='black', fontweight='bold')

# Legend for risk levels
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#66bb6a',
           markersize=10, label='Safe (< 10 m/s)', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffca28',
           markersize=10, label='Caution (10-17 m/s)', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7043',
           markersize=10, label='Danger (17-24 m/s)', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ef5350',
           markersize=10, label='Extreme (> 24 m/s)', linestyle='None'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

# Try adding coastlines (requires cartopy)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    # If using cartopy, create the axes with projection instead:
    # fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#c0c0c0')
    # ax.add_feature(cfeature.LAND, facecolor='#2a2e36')
    # ax.add_feature(cfeature.OCEAN, facecolor='#1a1d23')
except ImportError:
    pass  # Cartopy optional - plot still works without it

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Route Risk: Origin → Destination | Wind Speed Assessment',
             fontsize=14, fontweight='bold')

plt.savefig(f'{PLOTS_DIR}/route_risk_map.png')
plt.close()
```

### Key Rules
- **ALWAYS** use `YlOrRd` or similar warm colormap for wind speed
- **ALWAYS** add the risk legend with the 4 categories
- **ALWAYS** label Origin and Destination
- Use `shading='auto'` for `pcolormesh` to avoid deprecation warnings
- Use `method='nearest'` when extracting data at waypoints
""",
}


# =============================================================================
# ARGUMENT SCHEMA
# =============================================================================

class AnalysisGuideArgs(BaseModel):
    """Arguments for analysis guide retrieval."""
    
    topic: Literal[
        # Data operations
        "load_data",
        "spatial_subset", 
        "temporal_subset",
        # Statistical analysis
        "anomalies",
        "zscore",
        "trend_analysis",
        "eof_analysis",
        # Climate indices
        "climate_indices",
        # Extreme events
        "extremes",
        # Visualization
        "visualization_spatial",
        "visualization_timeseries",
        # Maritime
        "maritime_route",
        "maritime_visualization",
    ] = Field(
        description="Analysis topic to get guidance for"
    )


# =============================================================================
# TOOL FUNCTION
# =============================================================================

def get_analysis_guide(topic: str) -> str:
    """
    Get methodological guidance for climate data analysis.
    
    Returns text instructions for using python_repl to perform the analysis.
    """
    guide = ANALYSIS_GUIDES.get(topic)
    
    if not guide:
        available = ", ".join(ANALYSIS_GUIDES.keys())
        return f"Unknown topic: {topic}. Available: {available}"
    
    return f"""
# Analysis Guide: {topic.replace('_', ' ').title()}

{guide}

---
Use python_repl to implement this analysis with your downloaded ERA5 data.
"""


# =============================================================================
# TOOL DEFINITION
# =============================================================================

analysis_guide_tool = StructuredTool.from_function(
    func=get_analysis_guide,
    name="get_analysis_guide",
    description="""
    Get methodological guidance for climate data analysis.
    
    Returns instructions on:
    - What libraries to import
    - How to structure the analysis  
    - Key methods and parameters
    - How to interpret results
    - Best practices for visualization
    
    Use this BEFORE writing analysis code in python_repl.
    
    Available topics:
    - Data: load_data, spatial_subset, temporal_subset
    - Statistics: anomalies, zscore, trend_analysis, eof_analysis
    - Climate: climate_indices, extremes
    - Visualization: visualization_spatial, visualization_timeseries
    - Maritime: maritime_route, maritime_visualization
    """,
    args_schema=AnalysisGuideArgs,
)


# Visualization guide - alias for backward compatibility
visualization_guide_tool = StructuredTool.from_function(
    func=get_analysis_guide,
    name="get_visualization_guide",
    description="""
    Get publication-grade visualization instructions for ERA5 climate data.
    
    CALL THIS BEFORE creating any plot to get:
    - Correct colormap choices
    - Standard value ranges  
    - Required map elements
    - Best practices
    
    Available visualization topics:
    - visualization_spatial: Maps with proper projections
    - visualization_timeseries: Time series plots
    - maritime_visualization: Route risk maps
    """,
    args_schema=AnalysisGuideArgs,
)
