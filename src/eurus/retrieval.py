"""
ERA5 Data Retrieval
===================

Cloud-optimized data retrieval from Earthmover's ERA5 archive.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

from eurus.config import (
    CONFIG,
    get_data_dir,
    get_region,
    get_short_name,
    get_variable_info,
    list_available_variables,
)
from eurus.memory import get_memory

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


_aws_region_lock = threading.Lock()
_aws_region_set = False


def _ensure_aws_region(api_key: str, repo_name: Optional[str] = None) -> None:
    """
    Populate AWS S3 region/endpoint env vars from Arraylake repo metadata.

    Some environments fail S3 resolution unless region/endpoint are explicit.
    """
    global _aws_region_set
    if _aws_region_set:
        return  # Only run once per process

    with _aws_region_lock:
        if _aws_region_set:
            return  # Double-checked locking

        repo = repo_name or CONFIG.data_source
        try:
            req = Request(
                f"https://api.earthmover.io/repos/{repo}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urlopen(req, timeout=30) as resp:
                payload = resp.read().decode("utf-8")
            repo_meta = json.loads(payload)
        except Exception as exc:
            logger.debug("Could not auto-detect AWS region from Arraylake metadata: %s", exc)
            _aws_region_set = True  # Don't retry on failure
            return

    if not isinstance(repo_meta, dict):
        return

    bucket = repo_meta.get("bucket")
    if not isinstance(bucket, dict):
        return

    extra_cfg = bucket.get("extra_config")
    if not isinstance(extra_cfg, dict):
        return

    region_name = extra_cfg.get("region_name")
    if not isinstance(region_name, str) or not region_name:
        return

    endpoint = f"https://s3.{region_name}.amazonaws.com"
    desired_values = {
        "AWS_REGION": region_name,
        "AWS_DEFAULT_REGION": region_name,
        "AWS_ENDPOINT_URL": endpoint,
        "AWS_S3_ENDPOINT": endpoint,
    }
    updated = False
    for key, value in desired_values.items():
        if not os.environ.get(key):
            os.environ[key] = value
            updated = True

        if updated:
            logger.info(
                "Auto-set AWS region/endpoint for Arraylake: region=%s endpoint=%s",
                region_name,
                endpoint,
            )
        _aws_region_set = True


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
    _ensure_aws_region(api_key)

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

    # Check for future / too-recent dates (ERA5T has a ~5-day processing lag)
    req_start = datetime.strptime(start_date, '%Y-%m-%d')
    if req_start > datetime.now() - timedelta(days=5):
        return (
            f"Error: Requested start date ({start_date}) is too recent or in the future.\n"
            f"ERA5 data has a ~5-day processing lag. Please request dates at least 5 days ago."
        )

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

    # Guard: spatial queries are chunked for map access — multi-year ranges
    # cause thousands of S3 chunk fetches and streaming errors.
    # Limit spatial queries to 1 year max; suggest splitting or using temporal mode.
    req_end = datetime.strptime(end_date, '%Y-%m-%d')
    date_span_days = (req_end - req_start).days
    if query_type == "spatial" and date_span_days > 366:
        return (
            f"Error: Spatial queries are limited to 1 year max ({date_span_days} days requested).\n"
            f"The spatial dataset is optimised for maps, not long time series.\n\n"
            f"Options:\n"
            f"1. Split into yearly requests (e.g. one call per year)\n"
            f"2. Use query_type='temporal' for multi-year time-series analysis\n"
            f"3. Narrow the date range to ≤ 366 days"
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
            # Auto-compute tp = cp + lsp if tp is not directly available
            compute_tp = False
            if short_var not in ds:
                if short_var == "tp" and "cp" in ds and "lsp" in ds:
                    logger.info("Variable 'tp' not in store — will compute tp = cp + lsp")
                    compute_tp = True
                else:
                    available = list(ds.data_vars)
                    return (
                        f"Error: Variable '{short_var}' not found in dataset.\n"
                        f"Available variables: {', '.join(available)}\n\n"
                        f"Variable reference:\n{list_available_variables()}"
                    )

            # ERA5 latitude is stored 90 -> -90 (descending)
            lat_slice = slice(max_latitude, min_latitude)

            # Handle longitude - ERA5 uses 0-360 but we accept -180 to 180
            # CRITICAL: If coordinates are in Europe (-10 to 30), we need to 
            # convert to 0-360 for ERA5's coordinate system
            
            # Special case: Full world range (-180 to 180)
            # Both become 180 after % 360, which creates empty slice!
            if min_longitude == -180 and max_longitude == 180:
                req_min = 0.0
                req_max = 360.0
            elif min_longitude > max_longitude and min_longitude >= 0 and max_longitude >= 0:
                # Already in 0-360 format but wraps around 0° (e.g., Mediterranean: 354 to 42)
                # This comes from predefined regions — go directly to two-slice logic
                req_min = min_longitude
                req_max = max_longitude
            elif min_longitude < 0:
                # Convert -180/+180 to 0-360 for ERA5
                # e.g., -0.9 becomes 359.1
                req_min = min_longitude % 360
                req_max = max_longitude if max_longitude >= 0 else max_longitude % 360
            else:
                req_min = min_longitude
                req_max = max_longitude if max_longitude >= 0 else max_longitude % 360
            
            # Now handle the actual slicing
            # If min > max after conversion, it means we span the prime meridian (0°)
            # e.g., req_min=359.1 (was -0.9) and req_max=25.9 means we need 359.1->360 + 0->25.9
            if req_min > req_max:
                # Crosses prime meridian in ERA5's 0-360 system
                # We need to get two slices and concatenate
                logger.info(f"Region spans prime meridian: {req_min:.1f}° to {req_max:.1f}° (ERA5 coords)")
                
                # Get western portion (from req_min to 360)
                west_slice = slice(req_min, 360.0)
                # Get eastern portion (from 0 to req_max)
                east_slice = slice(0.0, req_max)
                
                # Subset both portions
                logger.info("Subsetting data (two-part: west + east of prime meridian)...")
                fetch_vars = ["cp", "lsp"] if compute_tp else [short_var]
                subsets_all = []
                for fv in fetch_vars:
                    subset_west = ds[fv].sel(
                        time=slice(start_date, end_date),
                        latitude=lat_slice,
                        longitude=west_slice,
                    )
                    subset_east = ds[fv].sel(
                        time=slice(start_date, end_date),
                        latitude=lat_slice,
                        longitude=east_slice,
                    )
                    
                    # Convert western longitudes from 360+ to negative (for -180/+180 output)
                    # e.g., 359.1 -> -0.9
                    subset_west = subset_west.assign_coords(
                        longitude=subset_west.longitude - 360
                    )
                    
                    # Concatenate along longitude
                    subsets_all.append(xr.concat([subset_west, subset_east], dim='longitude'))
                
                if compute_tp:
                    subset = (subsets_all[0] + subsets_all[1]).rename("tp")
                else:
                    subset = subsets_all[0]
            else:
                # Normal case - no prime meridian crossing
                lon_slice = slice(req_min, req_max)

                # Subset the data
                logger.info("Subsetting data...")
                fetch_vars = ["cp", "lsp"] if compute_tp else [short_var]
                subsets_all = []
                for fv in fetch_vars:
                    subsets_all.append(ds[fv].sel(
                        time=slice(start_date, end_date),
                        latitude=lat_slice,
                        longitude=lon_slice,
                    ))
                
                if compute_tp:
                    subset = (subsets_all[0] + subsets_all[1]).rename("tp")
                else:
                    subset = subsets_all[0]

            # Convert to dataset
            ds_out = subset.to_dataset(name=short_var)

            # Check for empty time dimension (no data in requested range)
            if ds_out.dims.get('time', 0) == 0:
                # Get actual data availability
                time_max = ds['time'].max().values
                import numpy as np
                last_available = str(np.datetime_as_string(time_max, unit='D'))
                return (
                    f"Error: No data available for the requested time range.\n"
                    f"Requested: {start_date} to {end_date}\n"
                    f"ERA5 data on Arraylake is available until {last_available}.\n\n"
                    f"Please request dates up to {last_available}."
                )

            # Check for empty data (all NaNs) — only check 1st timestep
            # Guard: skip the check for very large spatial slices to prevent OOM
            first_step = ds_out[short_var].isel(time=0)
            if first_step.size < 500_000 and first_step.isnull().all().compute():
                 return (
                    f"Error: The downloaded data for '{short_var}' is entirely empty (NaNs).\n"
                    f"Possible causes:\n"
                    f"1. The requested date/region has no data (e.g., SST over land).\n"
                    f"2. The request is too recent (ERA5T has a 5-day delay).\n"
                    f"3. Region bounds might be invalid or cross the prime meridian incorrectly."
                )

            # Size guard — prevent downloading datasets larger than the configured limit
            estimated_gb = ds_out.nbytes / (1024 ** 3)
            if estimated_gb > CONFIG.max_download_size_gb:
                return (
                    f"Error: Estimated download size ({estimated_gb:.1f} GB) exceeds the "
                    f"{CONFIG.max_download_size_gb} GB limit.\n"
                    f"Try narrowing the time range or spatial area."
                )

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
