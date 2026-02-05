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

### Required Imports
```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
```

### Essential Elements
1. **Projection**: Use appropriate map projection
   - Regional: `ccrs.PlateCarree()`
   - Global: `ccrs.Robinson()`
   - Polar: `ccrs.NorthPolarStereo()`

2. **Coastlines**: Always add with appropriate resolution
   - Global: '110m' (coarse)
   - Regional: '50m' (medium)
   - Local: '10m' (high detail)

3. **Colorbar**: Horizontal, with proper label and units

4. **Colormaps**:
   - Absolute values: Sequential (viridis, thermal)
   - Anomalies: Diverging, centered at zero (RdBu_r)
   - NEVER use jet for scientific figures!

### Units & Ranges
- Temperature: Convert to °C, use variable-appropriate ranges
- SST global: -2 to 32°C
- Anomalies: Symmetric around zero (e.g., ±3°C)
""",

    "visualization_timeseries": """
## Time Series Visualization

### Required Imports
```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
```

### Structure
1. Area-average the data first (if spatial)
2. Convert time coordinate to proper datetime
3. Use date formatters for x-axis

### Essential Elements
- Clear axis labels with units
- Grid lines (alpha=0.3)
- Legend if multiple lines
- Title with region/variable info

### Enhancements
- Trend line with statistics
- Uncertainty shading (±1 std)
- Event markers
- Smoothed line over noisy data
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
## Maritime Route Visualization

### Map Requirements
1. Route line with waypoints
2. Background: Wind speed or wave height
3. Risk segments color-coded (green/yellow/red)
4. Port locations marked

### Essential Elements
- Clear legend for risk categories
- Scale bar for distances
- Wind barbs or vectors along route
- Coastlines and bathymetry

### Color Scheme for Risk
- Green: Safe conditions
- Yellow: Caution advised
- Orange: Dangerous
- Red: Extreme hazard
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
