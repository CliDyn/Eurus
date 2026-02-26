"""
Copernicus Marine Data Retrieval
=================================

Retrieves ocean reanalysis / analysis data from the Copernicus Marine Service
using the copernicusmarine Python toolbox.

Credentials: assumes `copernicusmarine login` has been run once in the shell,
which writes ~/.copernicusmarine. No explicit credential handling here.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from eurus.config import get_data_dir, get_region
from eurus.memory import get_memory

logger = logging.getLogger(__name__)


# =============================================================================
# VARIABLE CATALOG
# =============================================================================

@dataclass(frozen=True)
class CopernicusVariable:
    dataset_id: str
    var_name: str       # internal name inside the CMEMS dataset
    long_name: str
    units: str
    has_depth: bool = False


_GLORYS = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
_DUACS  = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D"
_WAVES  = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"

COPERNICUS_CATALOG: dict[str, CopernicusVariable] = {
    # ── GLORYS12 physics reanalysis ────────────────────────────────────────
    "thetao":  CopernicusVariable(_GLORYS, "thetao",  "Sea water potential temperature", "°C",    has_depth=True),
    "so":      CopernicusVariable(_GLORYS, "so",      "Sea water salinity",               "PSU",   has_depth=True),
    "uo":      CopernicusVariable(_GLORYS, "uo",      "Eastward sea water velocity",      "m/s",   has_depth=True),
    "vo":      CopernicusVariable(_GLORYS, "vo",      "Northward sea water velocity",     "m/s",   has_depth=True),
    "zos":     CopernicusVariable(_GLORYS, "zos",     "Sea surface height above geoid",   "m"),
    "mlotst":  CopernicusVariable(_GLORYS, "mlotst",  "Mixed layer depth (sigma-theta)",  "m"),
    "siconc":  CopernicusVariable(_GLORYS, "siconc",  "Sea ice area fraction",            "1"),
    "sithick": CopernicusVariable(_GLORYS, "sithick", "Sea ice thickness",                "m"),
    "bottomT": CopernicusVariable(_GLORYS, "bottomT", "Sea floor potential temperature",  "°C"),
    # ── DUACS altimetry ────────────────────────────────────────────────────
    "sla":     CopernicusVariable(_DUACS,  "sla",     "Sea level anomaly",                "m"),
    "adt":     CopernicusVariable(_DUACS,  "adt",     "Absolute dynamic topography",      "m"),
    "ugos":    CopernicusVariable(_DUACS,  "ugos",    "Geostrophic eastward velocity",    "m/s"),
    "vgos":    CopernicusVariable(_DUACS,  "vgos",    "Geostrophic northward velocity",   "m/s"),
    # ── WAVERYS wave reanalysis ────────────────────────────────────────────
    "VHM0":    CopernicusVariable(_WAVES,  "VHM0",    "Significant wave height",          "m"),
    "VMDR":    CopernicusVariable(_WAVES,  "VMDR",    "Mean wave direction",              "°"),
    "VTM10":   CopernicusVariable(_WAVES,  "VTM10",   "Mean wave period",                 "s"),
}

COPERNICUS_ALIASES: dict[str, str] = {
    # temperature
    "temperature":               "thetao",
    "ocean_temperature":         "thetao",
    "sea_temperature":           "thetao",
    "potential_temperature":     "thetao",
    # salinity
    "salinity":                  "so",
    "ocean_salinity":            "so",
    # currents
    "u_current":                 "uo",
    "v_current":                 "vo",
    "eastward_velocity":         "uo",
    "northward_velocity":        "vo",
    # sea level
    "sea_level":                 "zos",
    "ssh":                       "zos",
    "sea_surface_height":        "zos",
    "sea_level_anomaly":         "sla",
    "absolute_dynamic_topography": "adt",
    # mixed layer
    "mld":                       "mlotst",
    "mixed_layer_depth":         "mlotst",
    # sea ice
    "sea_ice":                   "siconc",
    "ice_concentration":         "siconc",
    "ice_thickness":             "sithick",
    # bottom
    "bottom_temperature":        "bottomT",
    # waves
    "wave_height":               "VHM0",
    "significant_wave_height":   "VHM0",
    "wave_direction":            "VMDR",
    "wave_period":               "VTM10",
    # geostrophic
    "geostrophic_u":             "ugos",
    "geostrophic_v":             "vgos",
}


def resolve_variable(variable_id: str) -> Optional[CopernicusVariable]:
    """Return CopernicusVariable for a user-supplied name (alias-aware)."""
    key = COPERNICUS_ALIASES.get(variable_id.lower(), variable_id)
    return COPERNICUS_CATALOG.get(key)


def list_available_variables() -> str:
    lines = ["Available Copernicus Marine Variables:", "=" * 60]
    for key, v in COPERNICUS_CATALOG.items():
        depth_tag = " [has depth]" if v.has_depth else ""
        lines.append(f"  {key:10} | {v.long_name:42} | {v.units}{depth_tag}")
    return "\n".join(lines)


# =============================================================================
# FILENAME HELPERS
# =============================================================================

def _fmt(value: float) -> str:
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:.2f}"


def generate_filename(
    variable: str,
    start: str,
    end: str,
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
    depth_m: Optional[float],
    region: Optional[str] = None,
) -> str:
    clean_var   = variable.replace("_", "").lower()
    clean_start = start.replace("-", "")
    clean_end   = end.replace("-", "")
    depth_tag   = f"_d{depth_m:.0f}m" if depth_m is not None else "_surf"
    if region:
        geo_tag = region.lower()
    else:
        geo_tag = (
            f"lat{_fmt(min_latitude)}_{_fmt(max_latitude)}"
            f"_lon{_fmt(min_longitude)}_{_fmt(max_longitude)}"
        )
    return f"cmems_{clean_var}_{clean_start}_{clean_end}_{geo_tag}{depth_tag}.zarr"


def format_file_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve_copernicus_data(
    variable_id: str,
    start_date: str,
    end_date: str,
    min_latitude: float = -90.0,
    max_latitude: float = 90.0,
    min_longitude: float = -180.0,
    max_longitude: float = 180.0,
    depth_m: Optional[float] = None,
    region: Optional[str] = None,
) -> str:
    """
    Retrieve Copernicus Marine data and cache it locally as Zarr.

    Args:
        variable_id:   Variable name or alias (e.g. "thetao", "temperature", "VHM0").
        start_date:    YYYY-MM-DD
        end_date:      YYYY-MM-DD
        min_latitude:  Southern bound
        max_latitude:  Northern bound
        min_longitude: Western bound (-180 to 180)
        max_longitude: Eastern bound (-180 to 180)
        depth_m:       Depth in metres for 3-D variables (None = surface ~0.5 m).
                       Ignored for variables without a depth dimension.
        region:        Optional predefined region name (overrides lat/lon).

    Returns:
        Success message with local file path, or error string.
    """
    memory = get_memory()

    # Dependency check
    try:
        import copernicusmarine as cm
    except ImportError:
        return (
            "Error: 'copernicusmarine' is not installed.\n"
            "Install with: pip install copernicusmarine"
        )

    try:
        import xarray as xr
    except ImportError:
        return "Error: 'xarray' is not installed.\nInstall with: pip install xarray"

    # Apply region bounds
    region_tag = None
    if region:
        region_info = get_region(region)
        if region_info:
            min_latitude  = region_info.min_lat
            max_latitude  = region_info.max_lat
            # Regions are stored in 0-360; convert to -180/180 for Copernicus
            min_longitude = region_info.min_lon if region_info.min_lon <= 180 else region_info.min_lon - 360
            max_longitude = region_info.max_lon if region_info.max_lon <= 180 else region_info.max_lon - 360
            region_tag    = region.lower()
            logger.info("Using region '%s'", region)
        else:
            logger.warning("Unknown region '%s', using provided coordinates", region)

    # Resolve variable
    var_info = resolve_variable(variable_id)
    if var_info is None:
        return (
            f"Error: Variable '{variable_id}' not found in Copernicus catalog.\n\n"
            f"{list_available_variables()}"
        )
    var_key = COPERNICUS_ALIASES.get(variable_id.lower(), variable_id)

    # Depth only applies to 3-D variables
    effective_depth = depth_m if var_info.has_depth else None

    # Cache check
    output_dir = get_data_dir()
    filename   = generate_filename(
        var_key, start_date, end_date,
        min_latitude, max_latitude,
        min_longitude, max_longitude,
        effective_depth, region_tag,
    )
    local_path = str(output_dir / filename)

    if os.path.exists(local_path):
        existing = memory.get_dataset(local_path)
        if existing:
            logger.info("Cache hit: %s", local_path)
            return (
                f"CACHE HIT - Data already downloaded\n"
                f"  Variable: {var_key} ({var_info.long_name})\n"
                f"  Period: {existing.start_date} to {existing.end_date}\n"
                f"  Path: {local_path}\n\n"
                f"Load with: ds = xr.open_dataset('{local_path}', engine='zarr')"
            )
        else:
            try:
                file_size = sum(f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file())
                memory.register_dataset(
                    path=local_path,
                    variable=var_key,
                    query_type="copernicus",
                    start_date=start_date,
                    end_date=end_date,
                    lat_bounds=(min_latitude, max_latitude),
                    lon_bounds=(min_longitude, max_longitude),
                    file_size_bytes=file_size,
                )
            except Exception as exc:
                logger.warning("Could not register existing dataset: %s", exc)
            return (
                f"CACHE HIT - Found existing data\n"
                f"  Variable: {var_key}\n"
                f"  Path: {local_path}\n\n"
                f"Load with: ds = xr.open_dataset('{local_path}', engine='zarr')"
            )

    # Choose service based on query shape (mirrors ERA5 temporal/spatial logic)
    req_start = datetime.strptime(start_date, "%Y-%m-%d")
    req_end   = datetime.strptime(end_date,   "%Y-%m-%d")
    time_days = (req_end - req_start).days + 1
    area      = abs(max_latitude - min_latitude) * abs(max_longitude - min_longitude)
    service   = "arco-time-series" if (time_days > 1 and area < 900) else "arco-geo-series"
    logger.info("Auto-selected service=%s (time=%dd, area=%.0f sq°)", service, time_days, area)

    # Build open_dataset kwargs
    depth_val = effective_depth if effective_depth is not None else 0.5
    kwargs: dict = dict(
        dataset_id        = var_info.dataset_id,
        variables         = [var_info.var_name],
        minimum_longitude = min_longitude,
        maximum_longitude = max_longitude,
        minimum_latitude  = min_latitude,
        maximum_latitude  = max_latitude,
        start_datetime    = start_date,
        end_datetime      = end_date,
        service           = service,
    )
    if var_info.has_depth:
        kwargs["minimum_depth"] = depth_val
        kwargs["maximum_depth"] = depth_val

    # Download with retry
    for attempt in range(3):
        try:
            logger.info("Connecting to Copernicus Marine (attempt %d)...", attempt + 1)
            ds = cm.open_dataset(**kwargs)

            # Validate non-empty
            if ds.dims.get("time", 0) == 0:
                return (
                    f"Error: No data returned for the requested time range "
                    f"({start_date} to {end_date}).\n"
                    f"Check data availability for dataset '{var_info.dataset_id}'."
                )

            # Size guard
            import numpy as np  # noqa: F401 (ensure numpy available in scope)
            estimated_gb = ds.nbytes / (1024 ** 3)
            if estimated_gb > 15.0:
                return (
                    f"Error: Estimated download size ({estimated_gb:.1f} GB) exceeds the 15 GB limit.\n"
                    f"Try narrowing the time range or spatial area."
                )
            if estimated_gb > 1.0:
                logger.info("Large download (%.1f GB) — this may take a while...", estimated_gb)

            # Clear encoding, add metadata
            for var in ds.variables:
                ds[var].encoding = {}
            ds.attrs["source"]        = f"Copernicus Marine Service — {var_info.dataset_id}"
            ds.attrs["download_date"] = datetime.now().isoformat()
            if var_info.has_depth and effective_depth is not None:
                ds.attrs["depth_m"] = effective_depth

            # Save
            if os.path.exists(local_path):
                shutil.rmtree(local_path)

            logger.info("Saving to %s ...", local_path)
            t0 = time.time()
            ds.to_zarr(local_path, mode="w", consolidated=True)
            elapsed = time.time() - t0

            file_size = sum(f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file())
            shape     = tuple(ds[var_info.var_name].shape)

            memory.register_dataset(
                path           = local_path,
                variable       = var_key,
                query_type     = "copernicus",
                start_date     = start_date,
                end_date       = end_date,
                lat_bounds     = (min_latitude, max_latitude),
                lon_bounds     = (min_longitude, max_longitude),
                file_size_bytes= file_size,
                shape          = shape,
            )

            result = (
                f"SUCCESS - Data downloaded\n{'=' * 50}\n"
                f"  Variable: {var_key} ({var_info.long_name})\n"
                f"  Units:    {var_info.units}\n"
                f"  Period:   {start_date} to {end_date}\n"
                f"  Shape:    {shape}\n"
                f"  Size:     {format_file_size(file_size)}\n"
                f"  Time:     {elapsed:.1f}s\n"
                f"  Path:     {local_path}\n"
                f"{'=' * 50}\n\n"
                f"Load with:\n"
                f"  ds = xr.open_dataset('{local_path}', engine='zarr')"
            )
            return result

        except Exception as exc:
            logger.error("Attempt %d failed: %s", attempt + 1, exc)
            if os.path.exists(local_path):
                shutil.rmtree(local_path, ignore_errors=True)
            if attempt < 2:
                wait = 2.0 * (2 ** attempt)
                logger.info("Retrying in %.1fs...", wait)
                time.sleep(wait)
            else:
                return (
                    f"Error: Failed after 3 attempts.\n"
                    f"Last error: {exc}\n\n"
                    f"Troubleshooting:\n"
                    f"1. Run 'copernicusmarine login' in your shell if not done already\n"
                    f"2. Check your internet connection\n"
                    f"3. Try a smaller date range or region\n"
                    f"4. Verify variable '{var_key}' is available for dataset '{var_info.dataset_id}'"
                )

    return "Error: Unexpected failure in retrieval logic."
