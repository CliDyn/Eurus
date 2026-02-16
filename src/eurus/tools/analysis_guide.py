"""
Analysis Guide Tool
====================
Provides methodological guidance for climate data analysis using python_repl.

This tool returns TEXT INSTRUCTIONS (not executable code!) for:
- What approach to take
- How to structure the analysis
- Quality checks and pitfalls
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

### When to use
- Initializing any analysis
- Loading downloaded Zarr data

### Workflow
1. **Load data** — Use `xr.open_dataset('path', engine='zarr')` or `xr.open_zarr('path')`.
2. **Inspect dataset** — Check coordinates and available variables.
3. **Convert units** before any analysis:
   - Temp (`t2`, `d2`, `skt`, `sst`, `stl1`): subtract 273.15 → °C
   - Precip (`tp`, `cp`, `lsp`): multiply by 1000 → mm
   - Pressure (`sp`, `mslp`): divide by 100 → hPa

### Quality Checklist
- [ ] Data loaded lazily (avoid `.load()` on large datasets)
- [ ] Units converted before aggregations
- [ ] Coordinate names verified (latitude vs lat, etc.)

### Common Pitfalls
- ⚠️ Loading multi-year global data into memory causes OOM. Keep operations lazy until subsetted.
- ⚠️ Some Zarr stores have `valid_time` instead of `time` — check with `.coords`.
""",

    "spatial_subset": """
## Spatial Subsetting

### When to use
- Focusing on a specific region, country, or routing bounding box
- Reducing data size before heavy analysis

### Workflow
1. **Determine bounds** — Find min/max latitude and longitude.
2. **Check coordinate orientation** — ERA5 latitude is often descending (90 to -90).
3. **Slice data** — `.sel(latitude=slice(north, south), longitude=slice(west, east))`.

### Quality Checklist
- [ ] Latitude sliced from North to South (max to min) for descending coords
- [ ] Longitudes match dataset format (convert -180/180 ↔ 0/360 if needed)
- [ ] Result is not empty — verify with `.shape`

### Common Pitfalls
- ⚠️ Slicing `slice(south, north)` on descending coords → empty result.
- ⚠️ Crossing the prime meridian in 0-360 coords requires concatenating two slices.
- ⚠️ Use `.sel(method='nearest')` for point extraction, not exact matching.
""",

    "temporal_subset": """
## Temporal Subsetting & Aggregation

### When to use
- Isolating specific events, months, or seasons
- Downsampling hourly data to daily/monthly

### Workflow
1. **Time slice** — `.sel(time=slice('2023-01-01', '2023-12-31'))`.
2. **Filter** — Seasons: `.sel(time=ds.time.dt.season == 'DJF')`.
3. **Resample** — `.resample(time='1D').mean()` for daily means.

### Quality Checklist
- [ ] Aggregation matches variable: `.mean()` for T/wind, `.sum()` for precip
- [ ] Leap years handled if using day-of-year grouping

### Common Pitfalls
- ⚠️ DJF wraps across years — verify start/end boundaries.
- ⚠️ `.resample()` (continuous) ≠ `.groupby()` (climatological). Don't mix them up.
- ⚠️ Radiation variables (`ssr`, `ssrd`) are accumulated — need differencing, not averaging.
""",

    # -------------------------------------------------------------------------
    # STATISTICAL ANALYSIS
    # -------------------------------------------------------------------------
    "anomalies": """
## Anomaly Analysis

### When to use
- "How unusual was this period?"
- Comparing current conditions to "normal"
- Any "above/below average" question

### Workflow
1. **Define baseline** — ≥10 years (30 ideal). E.g. 1991-2020.
2. **Compute climatology** — `clim = ds.groupby('time.month').mean('time')`.
3. **Subtract** — `anomaly = ds.groupby('time.month') - clim`.
4. **Convert units** — Report in °C, mm, m/s (not K, m, Pa).
5. **Assess magnitude** — Compare to σ of the baseline period.

### Quality Checklist
- [ ] Baseline ≥10 years
- [ ] Same calendar grouping for clim and analysis
- [ ] Units converted for readability
- [ ] Spatial context: is anomaly regional or localized?

### Common Pitfalls
- ⚠️ Short baselines amplify noise.
- ⚠️ Daily climatologies with <30yr baseline are noisy → use monthly grouping.
- ⚠️ Be explicit: spatial anomaly vs temporal anomaly.

### Interpretation
- Positive = warmer/wetter/windier than normal.
- ±1σ = common, ±2σ = unusual (5%), ±3σ = extreme (0.3%).
- Maps: MUST use `RdBu_r` centered at zero via `TwoSlopeNorm`.
""",

    "zscore": """
## Z-Score Analysis (Standardized Anomalies)

### When to use
- Comparing extremity across different variables
- Standardizing for regions with different variability
- Identifying statistically significant departures

### Workflow
1. **Compute baseline mean** — Grouped by month for seasonality.
2. **Compute baseline std** — Same period, same grouping.
3. **Standardize** — `z = (value - mean) / std`.

### Quality Checklist
- [ ] Standard deviation is non-zero everywhere
- [ ] Baseline period matches for mean and std

### Common Pitfalls
- ⚠️ Precipitation is NOT normally distributed — use SPI or percentiles instead of raw Z-scores.
- ⚠️ Z-scores near coastlines can be extreme due to mixed land/ocean std.

### Interpretation
- Z = 0: average. ±1: normal (68%). ±2: unusual (5%). ±3: extreme (0.3%).
""",

    "trend_analysis": """
## Linear Trend Analysis

### When to use
- "Is it getting warmer/wetter over time?"
- Detecting long-term climate change signals

### Workflow
1. **Downsample** — Convert to annual/seasonal means first.
2. **Regress** — `scipy.stats.linregress` or `np.polyfit(degree=1)`.
3. **Significance** — Extract p-value for the slope.
4. **Scale** — Multiply annual slope by 10 → "per decade".

### Quality Checklist
- [ ] Period ≥20-30 years for meaningful trends
- [ ] Seasonal cycle removed before fitting
- [ ] Significance tested (p < 0.05)
- [ ] Report trend as units/decade

### Common Pitfalls
- ⚠️ Trend on daily data without removing seasonality → dominated by summer/winter swings.
- ⚠️ Short series have uncertain trends — report confidence intervals.
- ⚠️ Autocorrelation can inflate significance — consider using Mann-Kendall test.

### Interpretation
- Report as °C/decade. Use stippling on maps for significant areas.
""",

    "eof_analysis": """
## EOF/PCA Analysis

### When to use
- Finding dominant spatial patterns (ENSO, NAO, PDO)
- Dimensionality reduction of spatiotemporal data

### Workflow
1. **Deseasonalize** — Compute anomalies to remove the seasonal cycle.
2. **Latitude weighting** — Multiply by `np.sqrt(np.cos(np.deg2rad(lat)))`.
3. **Decompose** — PCA on flattened space dimensions.
4. **Reconstruct** — Map PCs back to spatial grid (EOFs).

### Quality Checklist
- [ ] Seasonal cycle removed
- [ ] Latitude weighting applied
- [ ] Variance explained (%) calculated per mode
- [ ] Physical interpretation attempted for leading modes

### Common Pitfalls
- ⚠️ Unweighted EOFs inflate polar regions artificially.
- ⚠️ EOFs are mathematical constructs — not guaranteed to correspond to physical modes.

### Interpretation
- EOF1: dominant spatial pattern. PC1: its temporal evolution.
- If EOF1 explains >20% variance, it's highly dominant.
""",

    "correlation_analysis": """
## Correlation Analysis

### When to use
- Spatial/temporal correlation mapping
- Lead-lag analysis (e.g., SST vs downstream precipitation)
- Teleconnection exploration

### Workflow
1. **Deseasonalize both variables** — Remove seasonal cycle from both.
2. **Align time coordinates** — Ensure identical time axes.
3. **Correlate** — `xr.corr(var1, var2, dim='time')`.
4. **Lead-lag** — Use `.shift(time=N)` month offsets to test delayed responses.
5. **Significance** — Compute p-values, mask insignificant areas.

### Quality Checklist
- [ ] Both variables deseasonalized
- [ ] p-values computed (p < 0.05 for significance)
- [ ] Sample size adequate (≥30 time points)

### Common Pitfalls
- ⚠️ Correlating raw data captures the seasonal cycle — everything correlates with summer.
- ⚠️ Spatial autocorrelation inflates field significance — apply Bonferroni or FDR correction.

### Interpretation
- R² gives variance explained. Lead-lag peak indicates response time.
- Plot spatial R maps with `RdBu_r`, stipple significant areas.
""",

    "composite_analysis": """
## Composite Analysis

### When to use
- Average conditions during El Niño vs La Niña years
- Spatial fingerprint of specific extreme events
- "What does the atmosphere look like when X happens?"

### Workflow
1. **Define events** — Boolean mask of times exceeding a threshold (e.g., Niño3.4 > 0.5°C).
2. **Subset data** — `.where(mask, drop=True)`.
3. **Average** — Time mean of the subset = composite.
4. **Compare** — Subtract climatological mean → composite anomaly.

### Quality Checklist
- [ ] Sample size ≥10 events for robustness
- [ ] Baseline climatology matches the season of the events
- [ ] Significance tested via bootstrap or t-test

### Common Pitfalls
- ⚠️ Compositing n=2 events → noise, not a physical signal.
- ⚠️ Mixing seasons in composite (El Niño in DJF vs JJA) obscures the signal.

### Interpretation
- Shows the typical anomaly expected when event occurs.
- Plot with `RdBu_r` diverging colormap. Stipple significant areas.
""",

    "diurnal_cycle": """
## Diurnal Cycle Analysis

### When to use
- Hourly variability within days (afternoon convection, nighttime cooling)
- Solar radiation patterns

### Workflow
1. **Group by hour** — `ds.groupby('time.hour').mean('time')`.
2. **Convert to local time** — ERA5 is UTC. `Local = UTC + Longitude/15`.
3. **Calculate amplitude** — `diurnal_range = max('hour') - min('hour')`.

### Quality Checklist
- [ ] Input data is hourly (not daily/monthly)
- [ ] UTC → local time conversion applied before labeling "afternoon"/"morning"

### Common Pitfalls
- ⚠️ Averaging global data by UTC hour mixes day and night across longitudes.
- ⚠️ Cloud cover (`tcc`) and radiation (`ssrd`) have strong diurnal signals — always check.

### Interpretation
- `blh` and `t2` peak mid-afternoon. Convective precip (`cp`) peaks late afternoon over land, early morning over oceans.
""",

    "seasonal_decomposition": """
## Seasonal Decomposition

### When to use
- Separating the seasonal cycle from interannual variability
- Visualizing how a specific year deviates from the normal curve

### Workflow
1. **Compute climatology** — `.groupby('time.month').mean('time')`.
2. **Extract anomalies** — Subtract climatology from raw data.
3. **Smooth trend** — Apply 12-month rolling mean to extract multi-year trends.

### Quality Checklist
- [ ] Baseline robust (≥10 years)
- [ ] Residual = raw - seasonal - trend (should be ~white noise)

### Common Pitfalls
- ⚠️ Day-of-year climatologies over short baselines are noisy — smooth with 15-day window.

### Interpretation
- Separates variance into: seasonal (predictable), trend (long-term), residual (weather noise).
""",

    "spectral_analysis": """
## Spectral Analysis

### When to use
- Periodicity detection (ENSO 3-7yr, MJO 30-60d, annual/semi-annual)
- Confirming suspected oscillatory behavior

### Workflow
1. **Prepare 1D series** — Spatial average or single point.
2. **Detrend** — Remove linear trend AND seasonal cycle.
3. **Compute spectrum** — `scipy.signal.welch` or `periodogram`.
4. **Plot as Period** — X-axis = 1/frequency (years or days), not raw frequency.

### Quality Checklist
- [ ] No NaNs in time series (interpolate or drop)
- [ ] Time coordinate evenly spaced
- [ ] Seasonal cycle removed

### Common Pitfalls
- ⚠️ Seasonal cycle dominates spectrum if not removed — drowns everything else.
- ⚠️ Short records can't resolve low-frequency oscillations (need ≥3× the period).

### Interpretation
- Peaks = dominant cycles. ENSO: 3-7yr. QBO: ~28mo. MJO: 30-60d. Annual: 12mo.
""",

    "spatial_statistics": """
## Spatial Statistics & Area Averaging

### When to use
- Computing a single time series for a geographic region
- Area-weighted means for reporting
- Field significance testing

### Workflow
1. **Latitude weights** — `weights = np.cos(np.deg2rad(ds.latitude))`.
2. **Apply** — `ds.weighted(weights).mean(dim=['latitude', 'longitude'])`.
3. **Land/sea mask** — Apply if needed (e.g., ocean-only SST average).

### Quality Checklist
- [ ] Latitude weighting applied BEFORE spatial averaging
- [ ] Land-sea mask applied where relevant
- [ ] Units preserved correctly

### Common Pitfalls
- ⚠️ Unweighted averages bias toward poles (smaller grid cells over-counted).
- ⚠️ Global mean SST must exclude land points.

### Interpretation
- Produces physically accurate area-averaged time series.
""",

    "multi_variable": """
## Multi-Variable Derived Quantities

### When to use
- Combining ERA5 variables for derived metrics

### Common Derivations
1. **Wind speed** — `wspd = np.sqrt(u10**2 + v10**2)` (or u100/v100 for hub-height).
2. **Wind direction** — `wdir = (270 - np.degrees(np.arctan2(v10, u10))) % 360`.
3. **Relative humidity** — From `t2` and `d2` using Magnus formula.
4. **Heat index** — Combine `t2` and `d2` (Steadman formula).
5. **Vapour transport** — `IVT ≈ tcwv * wspd` (surface proxy).
6. **Total precip check** — `tp ≈ cp + lsp`.

### Quality Checklist
- [ ] Variables share identical grids (time, lat, lon)
- [ ] Units matched before combining (both in °C, both in m/s, etc.)

### Common Pitfalls
- ⚠️ `mean(speed) ≠ speed_of_means` — always compute speed FIRST, then average.
- ⚠️ Wind direction requires proper 4-quadrant atan2, not naive arctan.

### Interpretation
- Derived metrics often better represent human/environmental impact than raw fields.
""",

    "climatology_normals": """
## Climatology Normals (WMO Standard)

### When to use
- Computing 30-year normals
- Calculating "departure from normal"

### Workflow
1. **Select base period** — Standard WMO epoch: 1991-2020 (or 1981-2010).
2. **Compute monthly averages** — `normals = baseline.groupby('time.month').mean('time')`.
3. **Departure** — `departure = current.groupby('time.month') - normals`.

### Quality Checklist
- [ ] Exactly 30 years used
- [ ] Same months compared (don't mix Feb normals with March data)

### Common Pitfalls
- ⚠️ Moving baselines make comparisons with WMO climate reports inconsistent.

### Interpretation
- "Normal" = statistical baseline. Departures express how much current conditions deviate.
""",

    # -------------------------------------------------------------------------
    # CLIMATE INDICES & EXTREMES
    # -------------------------------------------------------------------------
    "climate_indices": """
## Climate Indices

### When to use
- Assessing ENSO, NAO, PDO, AMO teleconnections
- Correlating local weather with large-scale modes

### Key Indices
- **ENSO (Niño 3.4)**: `sst` anomaly, 5°S-5°N, 170°W-120°W. El Niño > +0.5°C, La Niña < -0.5°C.
- **NAO**: `mslp` difference, Azores High minus Icelandic Low. Positive → mild European winters.
- **PDO**: Leading EOF of North Pacific `sst` (north of 20°N). 20-30yr phases.
- **AMO**: Detrended North Atlantic `sst` average. ~60-70yr cycle.

### Workflow
1. **Extract region** — Use standard geographic bounds.
2. **Compute anomaly** — Area-averaged, against 30yr baseline.
3. **Smooth** — 3-to-5 month rolling mean.

### Quality Checklist
- [ ] Standard geographic bounds strictly followed
- [ ] Rolling mean applied to filter weather noise
- [ ] Latitude-weighted area average

### Common Pitfalls
- ⚠️ Without rolling mean, the index is too noisy for classification.
- ⚠️ Using incorrect region bounds produces a different (invalid) index.
""",

    "extremes": """
## Extreme Event Analysis

### When to use
- Heat/cold extremes, heavy precipitation, tail-risk assessment
- Threshold exceedance frequency

### Workflow
1. **Define threshold** — Absolute (e.g., T > 35°C) or percentile-based (> 95th pctl of baseline).
2. **Create mask** — Boolean where condition is met.
3. **Count** — Sum over time for extreme days per year/month.
4. **Trend** — Check if frequency is increasing over time.

### Quality Checklist
- [ ] Percentiles from robust baseline (≥30 years)
- [ ] Use daily data, not monthly averages
- [ ] Units converted before applying thresholds

### Common Pitfalls
- ⚠️ 99th percentile on monthly averages misses true daily extremes entirely.
- ⚠️ Absolute thresholds (e.g., 35°C) are region-dependent — 35°C is normal in Sahara, extreme in London.

### Interpretation
- Increasing frequency of extremes = non-linear climate change impact.
- Report as "N days/year exceeding threshold" or "return period shortened from X to Y years".
""",

    "drought_analysis": """
## Drought Analysis

### When to use
- Prolonged precipitation deficits
- Agricultural/hydrological impact assessment
- SPI (Standardized Precipitation Index) proxy

### Workflow
1. **Extract precip** — Use `tp` in mm (×1000 from meters).
2. **Accumulate** — Rolling sums: `tp.rolling(time=3).sum()` for 3-month SPI.
3. **Standardize** — `(accumulated - mean) / std` → SPI proxy.
4. **Cross-check** — Verify with `swvl1` (soil moisture) for ground-truth.

### Quality Checklist
- [ ] Monthly data used (not hourly)
- [ ] Baseline ≥30 years for stable statistics
- [ ] Multiple accumulation periods tested (1, 3, 6, 12 months)

### Common Pitfalls
- ⚠️ Absolute precipitation deficits are meaningless in deserts — always standardize.
- ⚠️ Gamma distribution fit (proper SPI) is better than raw Z-score for precip.

### Interpretation
- SPI < -1.0: Moderate drought. < -1.5: Severe. < -2.0: Extreme.
""",

    "heatwave_detection": """
## Heatwave Detection

### When to use
- Identifying heatwave events using standard definitions
- Assessing heat-related risk periods

### Workflow
1. **Daily data** — Must be daily resolution (resample hourly if needed).
2. **Threshold** — 90th percentile of `t2` per calendar day from baseline.
3. **Exceedance mask** — `is_hot = t2_daily > threshold_90`.
4. **Streak detection** — Find ≥3 consecutive hot days using rolling sum ≥ 3.

### Quality Checklist
- [ ] Daily data (not monthly!)
- [ ] `t2` converted to °C
- [ ] Threshold is per-calendar-day (not a single annual value)
- [ ] Duration criterion applied (≥3 days)

### Common Pitfalls
- ⚠️ Monthly data — physically impossible to detect heatwaves.
- ⚠️ A single hot day is not a heatwave — duration matters.
- ⚠️ Nighttime temperatures (`t2` at 00/06 UTC) also matter for health impact.

### Interpretation
- Heatwaves require BOTH intensity (high T) AND duration (consecutive days).
- Report: number of events per year, mean duration, max intensity.
""",

    "atmospheric_rivers": """
## Atmospheric Rivers Detection

### When to use
- Detecting AR events from integrated vapour transport proxy
- Extreme precipitation risk at landfall

### Workflow
1. **Extract** — `tcwv` + `u10`, `v10`.
2. **Compute IVT proxy** — `ivt = tcwv * np.sqrt(u10**2 + v10**2)`.
3. **Threshold** — IVT proxy > 250 kg/m/s (approximate).
4. **Shape check** — Feature should be elongated (>2000km long, <1000km wide).

### Quality Checklist
- [ ] Acknowledge this is surface-wind proxy (true IVT needs pressure-level data)
- [ ] Cross-validate with heavy `tp` at landfall
- [ ] Check for persistent (≥24h) plume features

### Common Pitfalls
- ⚠️ Tropical moisture pools are NOT ARs — wind-speed multiplier is essential to distinguish.
- ⚠️ This surface proxy underestimates true IVT — use conservative thresholds.

### Interpretation
- High `tcwv` + strong directed wind at coast = extreme flood risk.
- Map with `YlGnBu` for moisture intensity.
""",

    "blocking_events": """
## Atmospheric Blocking Detection

### When to use
- Identifying persistent high-pressure blocks from MSLP
- Explaining prolonged heatwaves, droughts, or cold spells

### Workflow
1. **Extract** — `mslp` in hPa (÷100 from Pa).
2. **Compute anomalies** — Daily anomalies from climatology.
3. **Detect** — Find positive anomalies > 1.5σ persisting ≥5 days.
4. **Location** — Focus on mid-to-high latitudes (40-70°N typically).

### Quality Checklist
- [ ] 3-5 day rolling mean applied to filter transient ridges
- [ ] Persistence criterion enforced (≥5 days)
- [ ] Mid-latitude focus

### Common Pitfalls
- ⚠️ Fast-moving ridges are NOT blocks — persistence is key.
- ⚠️ Blocks in the Southern Hemisphere are rarer and weaker.

### Interpretation
- Blocks force storms to detour, causing prolonged rain on flanks and drought/heat underneath.
""",

    "energy_budget": """
## Surface Energy Budget

### When to use
- Analyzing radiation balance and surface heating
- Solar energy potential assessment

### Workflow
1. **Extract radiation** — `ssrd` (incoming solar), `ssr` (net solar after reflection).
2. **Convert units** — J/m² to W/m² by dividing by accumulation period (3600s for hourly).
3. **Compute albedo proxy** — `albedo ≈ 1 - (ssr / ssrd)` where ssrd > 0.
4. **Seasonal patterns** — Group by month to see radiation cycle.

### Quality Checklist
- [ ] Accumulation period properly accounted for (hourly vs daily sums)
- [ ] Division by zero protected (nighttime ssrd = 0)
- [ ] Units clearly stated: W/m² or MJ/m²/day

### Common Pitfalls
- ⚠️ ERA5 radiation is ACCUMULATED over the forecast step — must difference consecutive steps for instantaneous values.
- ⚠️ `ssr` already accounts for clouds and albedo — don't double-correct.

### Interpretation
- Higher `ssrd` - High solar potential. Low `ssr/ssrd` ratio → high cloudiness or reflective surface (snow/ice).
""",

    "wind_energy": """
## Wind Energy Assessment

### When to use
- Wind power density analysis
- Turbine hub-height wind resource mapping

### Workflow
1. **Use hub-height winds** — `u100`, `v100` (100m, not 10m surface winds).
2. **Compute speed** — `wspd100 = np.sqrt(u100**2 + v100**2)`.
3. **Power density** — `P = 0.5 * rho * wspd100**3` where rho ≈ 1.225 kg/m³.
4. **Capacity factor** — Fraction of time wind exceeds cut-in speed (~3 m/s) and stays below cut-out (~25 m/s).
5. **Weibull fit** — Fit shape (k) and scale (A) parameters to the wind speed distribution.

### Quality Checklist
- [ ] Using 100m winds, NOT 10m (turbines don't operate at surface)
- [ ] Power density in W/m²
- [ ] Seasonal variation checked (winter vs summer)

### Common Pitfalls
- ⚠️ Using 10m winds severely underestimates wind energy potential.
- ⚠️ Mean wind speed misleads — power depends on speed CUBED, so variability matters enormously.

### Interpretation
- Power density >400 W/m² = excellent wind resource.
- Report Weibull k parameter: k < 2 = gusty/variable, k > 3 = steady flow.
""",

    "moisture_budget": """
## Moisture Budget Analysis

### When to use
- Understanding precipitation sources
- Tracking moisture plumes and convergence zones

### Workflow
1. **Extract** — `tcwv` (precipitable water), `tcw` (total column water incl. liquid/ice).
2. **Temporal evolution** — Track `tcwv` changes to infer moisture convergence.
3. **Relate to precip** — Compare `tcwv` peaks with `tp` to see conversion efficiency.
4. **Spatial patterns** — Map `tcwv` to identify moisture corridors.

### Quality Checklist
- [ ] Distinguish `tcwv` (vapour only) from `tcw` (vapour + liquid + ice)
- [ ] Units: kg/m² (equivalent to mm of water)

### Common Pitfalls
- ⚠️ High `tcwv` doesn't guarantee rain — need a lifting mechanism.
- ⚠️ `tcw - tcwv` gives cloud water + ice content (proxy for cloud thickness).

### Interpretation
- `tcwv` > 50 kg/m² in tropics = moisture-laden atmosphere primed for heavy precip.
""",

    "convective_potential": """
## Convective Potential (Thunderstorm Risk)

### When to use
- Thunderstorm forecasting and climatology
- Severe weather risk assessment

### Workflow
1. **Extract CAPE** — Already available as `cape` variable (J/kg).
2. **Classify risk** — Low (<300), Moderate (300-1000), High (1000-2500), Extreme (>2500 J/kg).
3. **Combine with moisture** — High CAPE + high `tcwv` → heavy convective storms.
4. **Check trigger** — Fronts, orography, or strong daytime heating (`t2` diurnal cycle).

### Quality Checklist
- [ ] CAPE alone is insufficient — need a trigger mechanism
- [ ] Check `blh` (boundary layer height) — deep BLH aids convective initiation

### Common Pitfalls
- ⚠️ CAPE = potential energy, not a guarantee. High CAPE + strong capping inversion = no storms.
- ⚠️ CAPE is most meaningful in afternoon hours — avoid pre-dawn values.

### Interpretation
- CAPE > 1000 J/kg with deep BLH (>2km) and high `tcwv` = significant thunderstorm risk.
""",

    "snow_cover": """
## Snow Cover & Melt Analysis

### When to use
- Tracking snow accumulation and melt timing
- Climate change impacts on snowpack

### Workflow
1. **Extract** — `sd` (Snow Depth in m water equivalent).
2. **Seasonal cycle** — Track start/end of snow season per grid point.
3. **Melt timing** — Find the date when `sd` drops below threshold.
4. **Trend** — Check if snow season is shortening over decades.
5. **Compare with `stl1`/`t2`** — Warming soil accelerates melt.

### Quality Checklist
- [ ] Units: meters of water equivalent
- [ ] Focus on mid/high latitudes and mountain regions
- [ ] Inter-annual variability large — use multi-year analysis

### Common Pitfalls
- ⚠️ ERA5 snow depth is modeled, not observed — cross-reference with station data.
- ⚠️ Rain-on-snow events can cause rapid melt not captured well in reanalysis.

### Interpretation
- Earlier melt = less summer water supply. Map with `Blues`, reversed for snowless areas.
""",

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------
    "visualization_spatial": """
## Spatial Map Visualization

### When to use
- Mapping absolute climate fields (Temp, Wind, Precip, Pressure)

### Workflow
1. **Figure** — `fig, ax = plt.subplots(figsize=(12, 8))`.
2. **Meshgrid** — `lons, lats = np.meshgrid(data.longitude, data.latitude)`.
3. **Plot** — `ax.pcolormesh(lons, lats, data, cmap=..., shading='auto')`.
4. **Colorbar** — ALWAYS: `plt.colorbar(mesh, ax=ax, label='Units', shrink=0.8)`.
5. **Cartopy** — Optional: add coastlines, land fill. Graceful fallback if not installed.

### Quality Checklist
- [ ] Figure 12×8 for maps
- [ ] Colormap matches variable:
  - Temp: `RdYlBu_r` | Wind: `YlOrRd` | Precip: `YlGnBu`
  - Pressure: `viridis` | Cloud: `Greys` | Anomalies: `RdBu_r`
- [ ] NEVER use `jet`
- [ ] Colorbar has label with units

### Common Pitfalls
- ⚠️ Diverging cmap on absolute data is misleading — diverging only for anomalies.
- ⚠️ Missing `shading='auto'` triggers deprecation warning.
""",

    "visualization_timeseries": """
## Time Series Visualization

### When to use
- Temporal evolution of a variable at a point or region

### Workflow
1. **Area average** — `ts = data.mean(dim=['latitude', 'longitude'])` (with lat weighting!).
2. **Figure** — `fig, ax = plt.subplots(figsize=(10, 6))`.
3. **Raw line** — `ax.plot(ts.time, ts, linewidth=1.5)`.
4. **Smoothing** — Add rolling mean overlay with contrasting color.
5. **Date formatting** — `fig.autofmt_xdate(rotation=30)`.

### Quality Checklist
- [ ] Figure 10×6
- [ ] Y-axis has explicit units
- [ ] Legend included if multiple lines
- [ ] Trend line if requested: dashed with slope annotation

### Enhancements
- **Uncertainty band**: `ax.fill_between(time, mean-std, mean+std, alpha=0.2)`
- **Event markers**: `ax.axvline(date, color='red', ls='--')`
- **Twin axis**: `ax2 = ax.twinx()` for second variable

### Common Pitfalls
- ⚠️ Hourly data over 10+ years → unreadable block of ink. Resample to daily first.
""",

    "visualization_anomaly_map": """
## Anomaly Map Visualization

### When to use
- Diverging data: departures, trends, z-scores
- Any map that has positive AND negative values

### Workflow
1. **Center at zero** — `from matplotlib.colors import TwoSlopeNorm`.
2. **Norm** — `norm = TwoSlopeNorm(vmin=data.min(), vcenter=0, vmax=data.max())`.
3. **Plot** — `pcolormesh(..., cmap='RdBu_r', norm=norm)`.
4. **Stippling** — Overlay significance: `contourf(..., levels=[0, 0.05], hatches=['...'], colors='none')`.

### Quality Checklist
- [ ] Zero is EXACTLY white/neutral in the colorbar
- [ ] Warm/dry = Red; Cool/wet = Blue
- [ ] Precip anomalies: consider `BrBG` instead of `RdBu_r`

### Common Pitfalls
- ⚠️ Without `TwoSlopeNorm`, skewed data makes 0 appear colored → reader is misled.
- ⚠️ Symmetric vmin/vmax (`vmax = max(abs(data))`) can also work but wastes color range.
""",

    "visualization_wind": """
## Wind & Vector Visualization

### When to use
- Circulation patterns, wind fields, quiver/streamline plots

### Workflow
1. **Speed background** — `wspd` with `pcolormesh` + `YlOrRd`.
2. **Subsample vectors** — `skip = (slice(None, None, 5), slice(None, None, 5))` to avoid solid black.
3. **Quiver** — `ax.quiver(lons[skip], lats[skip], u[skip], v[skip], color='black')`.
4. **Alternative** — `ax.streamplot()` for flow visualization (less cluttered).

### Quality Checklist
- [ ] Background heatmap shows magnitude
- [ ] Vectors sparse enough to be readable
- [ ] Wind barbs: `ax.barbs()` for meteorological display

### Common Pitfalls
- ⚠️ Full-resolution quiver = completely black, unreadable mess.
- ⚠️ Check arrow scaling — default autoscale can make light winds invisible.

### Interpretation
- Arrows = direction, background color = magnitude. Cyclonic rotation = storm.
""",

    "visualization_comparison": """
## Multi-Panel Comparison

### When to use
- Before/after, two periods, difference maps
- Multi-variable side-by-side

### Workflow
1. **Grid** — `fig, axes = plt.subplots(1, 3, figsize=(18, 6))`.
2. **Panels 1 & 2** — Absolute values with SHARED `vmin`/`vmax`.
3. **Panel 3** — Difference (A-B) with diverging cmap centered at zero.

### Quality Checklist
- [ ] Panels 1 & 2 share EXACT same vmin/vmax (otherwise visual comparison is invalid)
- [ ] Panel 3 has its own divergent colorbar centered at zero
- [ ] Titles clearly label what each panel shows

### Common Pitfalls
- ⚠️ Auto-scaled panels = impossible to compare visually. Always lock limits.
""",

    "visualization_profile": """
## Hovmöller Diagrams

### When to use
- Lat-time or lon-time cross-sections
- Tracking wave propagation, ITCZ migration, monsoon onset

### Workflow
1. **Average out one dimension** — e.g., average across latitudes to get (lon, time).
2. **Transpose** — X=Time, Y=Lon/Lat.
3. **Plot** — `contourf` or `pcolormesh`, figure 12×6.  

### Quality Checklist
- [ ] X-axis uses date formatting
- [ ] Y-axis labels state the averaged geographic slice
- [ ] Colormap matches variable type

### Common Pitfalls
- ⚠️ Swapping axes makes the diagram unintuitive. Time → X-axis convention.

### Interpretation
- Diagonal banding = propagating waves/systems. Vertical banding = stationary patterns.
""",

    "visualization_distribution": """
## Distribution Visualization

### When to use
- Histograms, PDFs, box plots
- Comparing two time periods or regions

### Workflow
1. **Flatten** — `.values.flatten()`, drop NaNs.
2. **Shared bins** — `np.linspace(min, max, 50)`.
3. **Plot** — `ax.hist(data, bins=bins, alpha=0.5, density=True, label='Period')`.
4. **Median/mean markers** — Vertical lines with annotation.

### Quality Checklist
- [ ] `density=True` for comparing different-sized samples
- [ ] `alpha=0.5` for overlapping distributions
- [ ] Legend when comparing multiple distributions

### Common Pitfalls
- ⚠️ Raw counts (not density) skew comparison between periods with different sample sizes.
- ⚠️ Too few bins = lost detail. Too many = noisy. 30-50 bins is usually good.

### Interpretation
- Rightward shift = warming. Flatter + wider = more variability = more extremes.
""",

    "visualization_animation": """
## Animated/Sequential Maps

### When to use
- Monthly/seasonal evolution of a field
- Event lifecycle (genesis → peak → decay)

### Workflow
1. **Global limits** — Find absolute vmin/vmax across ALL timesteps.
2. **Multi-panel grid** — `fig, axes = plt.subplots(2, 3, figsize=(18, 12))` for 6 timesteps.
3. **Lock colorbars** — Same vmin/vmax on every panel.
4. **Shared colorbar** — Remove per-panel colorbars, add one at the bottom.

### Quality Checklist
- [ ] Colorbar limits LOCKED across all panels (no jumping colors)
- [ ] Timestamps clearly labeled on each panel
- [ ] Static grid preferred over video (headless environment)

### Common Pitfalls
- ⚠️ Auto-scaled panels flash/jump between frames — always lock limits.
- ⚠️ MP4/GIF generation may fail in headless — use PNG grids instead.
""",

    "visualization_dashboard": """
## Summary Dashboard

### When to use
- Comprehensive overview: map + time series + statistics in one figure
- Publication-ready event summaries

### Workflow
1. **Layout** — `fig = plt.figure(figsize=(16, 10))` + `matplotlib.gridspec`.
2. **Top row** — Large spatial map (anomaly or mean field).
3. **Bottom left** — Time series of regional mean.
4. **Bottom right** — Distribution histogram or box plot.

### Quality Checklist
- [ ] `plt.tight_layout()` or `constrained_layout=True` to prevent overlap
- [ ] Consistent color theme across all panels
- [ ] Clear panel labels (a, b, c)

### Common Pitfalls
- ⚠️ Cramming too much into small figure → illegible text. Scale figure size up.
- ⚠️ Different aspect ratios between map and time series need explicit gridspec ratios.
""",

    "visualization_contour": """
## Contour & Isobar Plots

### When to use
- Pressure maps with isobars
- Temperature isotherms
- Any smoothly varying field where specific levels matter

### Workflow
1. **Define levels** — `levels = np.arange(990, 1040, 4)` for MSLP isobars.
2. **Filled contour** — `ax.contourf(lons, lats, data, levels=levels, cmap=...)`.
3. **Contour lines** — `cs = ax.contour(lons, lats, data, levels=levels, colors='black', linewidths=0.5)`.
4. **Labels** — `ax.clabel(cs, inline=True, fontsize=8)`.

### Quality Checklist
- [ ] Level spacing is physically meaningful (e.g., 4 hPa for MSLP)
- [ ] Contour labels don't overlap
- [ ] Filled + line contours combined for best readability

### Common Pitfalls
- ⚠️ Too many levels → cluttered, unreadable. 10-15 levels max.
- ⚠️ Non-uniform level spacing requires manual colorbar ticks.

### Interpretation
- Tightly packed isobars = strong pressure gradient = high winds.
""",

    "visualization_correlation_map": """
## Spatial Correlation Maps

### When to use
- Showing where a variable correlates with an index (e.g., ENSO vs global precip)
- Teleconnection mapping

### Workflow
1. **Compute index** — 1D time series (e.g., Niño3.4 SST anomaly).
2. **Correlate** — `xr.corr(index, spatial_field, dim='time')` → 2D R-map.
3. **Significance** — Compute p-values from sample size and R.
4. **Plot** — Map R values with `RdBu_r` centered at zero. Stipple p < 0.05.

### Quality Checklist
- [ ] Both index and field deseasonalized
- [ ] R-map centered at zero (TwoSlopeNorm or symmetric limits)
- [ ] Significant areas stippled or hatched
- [ ] Sample size ≥30 stated

### Common Pitfalls
- ⚠️ Raw data correlations dominated by shared seasonal cycle.
- ⚠️ Field significance: many grid points → some will be significant by chance. Apply FDR correction.

### Interpretation
- R > 0: in-phase with index. R < 0: out-of-phase. |R| > 0.5 = strong relationship.
""",

    # -------------------------------------------------------------------------
    # MARITIME ANALYSIS
    # -------------------------------------------------------------------------
    "maritime_route": """
## Maritime Route Risk Analysis

### When to use
- Analyzing weather risks along calculated shipping lanes
- Voyage planning and hazard assessment

### Workflow
1. **Route** — Call `calculate_maritime_route` → waypoints + bounding box.
2. **Data** — Download `u10`, `v10` for route bbox, target month, last 3 years.
3. **Wind speed** — `wspd = np.sqrt(u10**2 + v10**2)`.
4. **Extract** — Loop waypoints: `.sel(lat=lat, lon=lon, method='nearest')`.
5. **Risk classify** — Safe (<10), Caution (10-17), Danger (17-24), Extreme (>24 m/s).
6. **Statistics** — P95 wind speed at each waypoint, % time in each risk category.

### Quality Checklist
- [ ] Bounding box from route tool used DIRECTLY (don't convert coords)
- [ ] 3-year period for climatological context, not just one date
- [ ] Risk categories applied at waypoint level

### Common Pitfalls
- ⚠️ Global hourly downloads → timeout. Subset tightly to route bbox.
- ⚠️ Don't use bounding box mean — extract AT waypoints for route-specific risk.
""",

    "maritime_visualization": """
## Maritime Route Risk Visualization

### When to use
- Plotting route risk maps with waypoint-level risk coloring

### Workflow
1. **Background** — Map mean `wspd` with `pcolormesh` + `YlOrRd`.
2. **Route line** — Dashed line connecting waypoints.
3. **Waypoint scatter** — Color by risk: Green (<10), Amber (10-17), Coral (17-24), Red (>24 m/s).
4. **Labels** — "ORIGIN" and "DEST" annotations.
5. **Legend** — Custom 4-category legend (mandatory).

### Quality Checklist
- [ ] 4-category risk legend ALWAYS included
- [ ] Origin/Destination labeled
- [ ] Colormap: `YlOrRd` for wind speed
- [ ] Saved to PLOTS_DIR

### Common Pitfalls
- ⚠️ No legend → colored dots are meaningless to the user.
- ⚠️ Route line + waypoints must be on top (high zorder) to not be hidden by background.
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
        # Advanced analysis
        "correlation_analysis",
        "composite_analysis",
        "diurnal_cycle",
        "seasonal_decomposition",
        "spectral_analysis",
        "spatial_statistics",
        "multi_variable",
        "climatology_normals",
        # Climate indices & extremes
        "climate_indices",
        "extremes",
        "drought_analysis",
        "heatwave_detection",
        "atmospheric_rivers",
        "blocking_events",
        # Domain-specific
        "energy_budget",
        "wind_energy",
        "moisture_budget",
        "convective_potential",
        "snow_cover",
        # Visualization
        "visualization_spatial",
        "visualization_timeseries",
        "visualization_anomaly_map",
        "visualization_wind",
        "visualization_comparison",
        "visualization_profile",
        "visualization_distribution",
        "visualization_animation",
        "visualization_dashboard",
        "visualization_contour",
        "visualization_correlation_map",
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
        available = ", ".join(sorted(ANALYSIS_GUIDES.keys()))
        return f"Unknown topic: {topic}. Available: {available}"

    return f"""
# Analysis Guide: {topic.replace('_', ' ').title()}

{guide}

---
Use python_repl to implement this analysis with your downloaded ERA5 data.
"""


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

analysis_guide_tool = StructuredTool.from_function(
    func=get_analysis_guide,
    name="get_analysis_guide",
    description="""
    Get methodological guidance for climate data analysis.

    Returns workflow steps, quality checklists, and pitfall warnings for:
    - Data: load_data, spatial_subset, temporal_subset
    - Statistics: anomalies, zscore, trend_analysis, eof_analysis
    - Advanced: correlation_analysis, composite_analysis, diurnal_cycle,
      seasonal_decomposition, spectral_analysis, spatial_statistics,
      multi_variable, climatology_normals
    - Climate: climate_indices, extremes, drought_analysis, heatwave_detection,
      atmospheric_rivers, blocking_events
    - Domain: energy_budget, wind_energy, moisture_budget, convective_potential, snow_cover
    - Visualization: visualization_spatial, visualization_timeseries,
      visualization_anomaly_map, visualization_wind, visualization_comparison,
      visualization_profile, visualization_distribution, visualization_animation,
      visualization_dashboard, visualization_contour, visualization_correlation_map
    - Maritime: maritime_route, maritime_visualization

    Use this BEFORE writing analysis code in python_repl.
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
    - visualization_anomaly_map: Diverging anomaly maps
    - visualization_wind: Quiver/streamline plots
    - visualization_comparison: Multi-panel comparisons
    - visualization_profile: Hovmöller diagrams
    - visualization_distribution: Histograms/PDFs
    - visualization_animation: Sequential map grids
    - visualization_dashboard: Multi-panel summaries
    - visualization_contour: Isobar/isotherm plots
    - visualization_correlation_map: Spatial correlation maps
    - maritime_visualization: Route risk maps
    """,
    args_schema=AnalysisGuideArgs,
)
