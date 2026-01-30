"""
ERA5 Data Retrieval
===================

Cloud-optimized data retrieval from Earthmover's ERA5 archive.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from vostok.config import (
    CONFIG,
    get_data_dir,
    get_region,
    get_short_name,
    get_variable_info,
    list_available_variables,
)
from vostok.memory import get_memory

logger = logging.getLogger(__name__)


def _format_coord(value: float) -> str:
    """Format coordinates for stable, filename-safe identifiers."""
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:.2f}"


def generate_filename(
    variable: str,
    query_type: str,
    start: str,
    end: str,
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
    region: Optional[str] = None,
) -> str:
    """Generate a descriptive filename for the dataset."""
    clean_var = variable.replace("_", "")
    clean_start = start.replace("-", "")
    clean_end = end.replace("-", "")
    if region:
        region_tag = region.lower()
    else:
        region_tag = (
            f"lat{_format_coord(min_latitude)}_{_format_coord(max_latitude)}"
            f"_lon{_format_coord(min_longitude)}_{_format_coord(max_longitude)}"
        )
    return f"era5_{clean_var}_{query_type}_{clean_start}_{clean_end}_{region_tag}.zarr"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def retrieve_era5_data(
    query_type: str,
    variable_id: str,
    start_date: str,
    end_date: str,
    min_latitude: float = -90.0,
    max_latitude: float = 90.0,
    min_longitude: float = 0.0,
    max_longitude: float = 359.75,
    region: Optional[str] = None,
) -> str:
    """
    Retrieve ERA5 reanalysis data from Earthmover's cloud-optimized archive.

    Args:
        query_type: Either "temporal" (time series) or "spatial" (maps)
        variable_id: ERA5 variable name (e.g., "sst", "t2", "u10")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        min_latitude: Southern bound (-90 to 90)
        max_latitude: Northern bound (-90 to 90)
        min_longitude: Western bound (0 to 360)
        max_longitude: Eastern bound (0 to 360)
        region: Optional predefined region name (overrides lat/lon)

    Returns:
        Success message with file path, or error message.

    Raises:
        No exceptions raised - errors returned as strings.
    """
    memory = get_memory()

    # Get API key
    api_key = os.environ.get("ARRAYLAKE_API_KEY")
    if not api_key:
        return (
            "Error: ARRAYLAKE_API_KEY not found in environment.\n"
            "Please set it via environment variable or .env file."
        )

    # Check dependencies
    try:
        import icechunk  # noqa: F401
    except ImportError:
        return (
            "Error: The 'icechunk' library is required.\n"
            "Install with: pip install icechunk"
        )

    try:
        import xarray as xr
    except ImportError:
        return (
            "Error: The 'xarray' library is required.\n"
            "Install with: pip install xarray"
        )

    # Apply region bounds if specified
    region_tag = None
    if region:
        region_info = get_region(region)
        if region_info:
            min_latitude = region_info.min_lat
            max_latitude = region_info.max_lat
            min_longitude = region_info.min_lon
            max_longitude = region_info.max_lon
            region_tag = region.lower()
            logger.info(f"Using region '{region}'")
        else:
            logger.warning(f"Unknown region '{region}', using provided coordinates")

    # Resolve variable name
    short_var = get_short_name(variable_id)
    var_info = get_variable_info(variable_id)

    # Setup paths
    output_dir = get_data_dir()
    filename = generate_filename(
        short_var,
        query_type,
        start_date,
        end_date,
        min_latitude,
        max_latitude,
        min_longitude,
        max_longitude,
        region_tag,
    )
    local_path = str(output_dir / filename)

    # Check cache first
    if os.path.exists(local_path):
        existing = memory.get_dataset(local_path)
        if existing:
            logger.info(f"Cache hit: {local_path}")
            var_name = f"{short_var} ({var_info.long_name})" if var_info else short_var
            return (
                f"CACHE HIT - Data already downloaded\n"
                f"  Variable: {var_name}\n"
                f"  Period: {existing.start_date} to {existing.end_date}\n"
                f"  Path: {local_path}\n\n"
                f"Load with: ds = xr.open_dataset('{local_path}', engine='zarr')"
            )
        else:
            # File exists but not registered - register it
            try:
                file_size = sum(f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file())
                memory.register_dataset(
                    path=local_path,
                    variable=short_var,
                    query_type=query_type,
                    start_date=start_date,
                    end_date=end_date,
                    lat_bounds=(min_latitude, max_latitude),
                    lon_bounds=(min_longitude, max_longitude),
                    file_size_bytes=file_size,
                )
            except Exception as e:
                logger.warning(f"Could not register existing dataset: {e}")

            return (
                f"CACHE HIT - Found existing data\n"
                f"  Variable: {short_var}\n"
                f"  Path: {local_path}\n\n"
                f"Load with: ds = xr.open_dataset('{local_path}', engine='zarr')"
            )

    # Download with retry logic
    for attempt in range(CONFIG.max_retries):
        try:
            from arraylake import Client

            logger.info(f"Connecting to Earthmover (attempt {attempt + 1})...")

            client = Client(token=api_key)
            repo = client.get_repo(CONFIG.data_source)
            session = repo.readonly_session("main")

            logger.info(f"Opening {query_type} dataset...")
            ds = xr.open_dataset(
                session.store,
                engine="zarr",
                consolidated=False,
                zarr_format=3,
                chunks=None,
                group=query_type,
            )

            # Validate variable exists
            if short_var not in ds:
                available = list(ds.data_vars)
                return (
                    f"Error: Variable '{short_var}' not found in dataset.\n"
                    f"Available variables: {', '.join(available)}\n\n"
                    f"Variable reference:\n{list_available_variables()}"
                )

            # ERA5 latitude is stored 90 -> -90 (descending)
            lat_slice = slice(max_latitude, min_latitude)

            # Handle longitude wrapping (ERA5 uses 0-360)
            req_min = min_longitude % 360
            req_max = max_longitude % 360

            if req_min > req_max:
                # Crosses prime meridian
                lon_slice = slice(req_min, 359.75)
                logger.warning("Region crosses prime meridian - taking eastern portion")
            else:
                lon_slice = slice(req_min, req_max)

            # Subset the data
            logger.info("Subsetting data...")
            subset = ds[short_var].sel(
                time=slice(start_date, end_date),
                latitude=lat_slice,
                longitude=lon_slice,
            )

            # Convert to dataset
            ds_out = subset.to_dataset(name=short_var)

            # Clear encoding for clean serialization
            for var in ds_out.variables:
                ds_out[var].encoding = {}

            # Add metadata
            ds_out.attrs["source"] = "ERA5 Reanalysis via Earthmover Arraylake"
            ds_out.attrs["download_date"] = datetime.now().isoformat()
            ds_out.attrs["query_type"] = query_type
            if var_info:
                ds_out[short_var].attrs["long_name"] = var_info.long_name
                ds_out[short_var].attrs["units"] = var_info.units

            # Clean up existing file
            if os.path.exists(local_path):
                shutil.rmtree(local_path)

            # Save to Zarr
            logger.info(f"Saving to {local_path}...")
            start_time = time.time()
            ds_out.to_zarr(local_path, mode="w", consolidated=True, compute=True)
            download_time = time.time() - start_time

            # Get actual file size
            file_size = sum(f.stat().st_size for f in Path(local_path).rglob("*") if f.is_file())
            shape = tuple(ds_out[short_var].shape)

            # Register in memory
            memory.register_dataset(
                path=local_path,
                variable=short_var,
                query_type=query_type,
                start_date=start_date,
                end_date=end_date,
                lat_bounds=(min_latitude, max_latitude),
                lon_bounds=(min_longitude, max_longitude),
                file_size_bytes=file_size,
                shape=shape,
            )

            # Build success message
            result = f"SUCCESS - Data downloaded\n{'='*50}\n  Variable: {short_var}"
            if var_info:
                result += f" ({var_info.long_name})"
            result += (
                f"\n  Units: {var_info.units if var_info else 'Unknown'}\n"
                f"  Period: {start_date} to {end_date}\n"
                f"  Shape: {shape}\n"
                f"  Size: {format_file_size(file_size)}\n"
                f"  Time: {download_time:.1f}s\n"
                f"  Path: {local_path}\n"
                f"{'='*50}\n\n"
                f"Load with:\n"
                f"  ds = xr.open_dataset('{local_path}', engine='zarr')"
            )
            return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Attempt {attempt + 1} failed: {error_msg}")

            # Clean up partial download
            if os.path.exists(local_path):
                shutil.rmtree(local_path, ignore_errors=True)

            if attempt < CONFIG.max_retries - 1:
                wait_time = CONFIG.retry_delay * (2**attempt)
                logger.info(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                return (
                    f"Error: Failed after {CONFIG.max_retries} attempts.\n"
                    f"Last error: {error_msg}\n\n"
                    f"Troubleshooting:\n"
                    f"1. Check your ARRAYLAKE_API_KEY\n"
                    f"2. Verify internet connection\n"
                    f"3. Try a smaller date range or region\n"
                    f"4. Check if variable '{short_var}' is available"
                )

    return "Error: Unexpected failure in retrieval logic."
