"""
ERA5 Data Retrieval Tool (Wrapper)
==================================
LangChain tool definition. Imports core logic from ..retrieval

This is a THIN WRAPPER - all retrieval logic lives in eurus/retrieval.py

QUERY_TYPE IS AUTO-DETECTED based on time/area rules:
- TEMPORAL: time > 1 day AND area < 30°×30°
- SPATIAL:  time ≤ 1 day OR  area ≥ 30°×30°
"""

import logging
from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool

# IMPORT CORE LOGIC FROM RETRIEVAL MODULE - SINGLE SOURCE OF TRUTH
from ..retrieval import retrieve_era5_data as _retrieve_era5_data
from ..config import get_short_name

logger = logging.getLogger(__name__)


# ============================================================================
# ARGUMENT SCHEMA (NO query_type - it's auto-detected!)
# ============================================================================

class ERA5RetrievalArgs(BaseModel):
    """Arguments for ERA5 data retrieval. query_type is AUTO-DETECTED."""

    variable_id: str = Field(
        description=(
            "ERA5 variable short name. Available variables (22 total):\n"
            "Ocean: sst (Sea Surface Temperature)\n"
            "Temperature: t2 (2m Air Temp), d2 (2m Dewpoint), skt (Skin Temp)\n"
            "Wind 10m: u10 (Eastward), v10 (Northward)\n"
            "Wind 100m: u100 (Eastward), v100 (Northward)\n"
            "Pressure: sp (Surface), mslp (Mean Sea Level)\n"
            "Boundary Layer: blh (BL Height), cape (CAPE)\n"
            "Cloud/Precip: tcc (Cloud Cover), cp (Convective), lsp (Large-scale), tp (Total Precip)\n"
            "Radiation: ssr (Net Solar), ssrd (Solar Downwards)\n"
            "Moisture: tcw (Total Column Water), tcwv (Water Vapour)\n"
            "Land: sd (Snow Depth), stl1 (Soil Temp L1), swvl1 (Soil Water L1)"
        )
    )

    start_date: str = Field(
        description="Start date in YYYY-MM-DD format (e.g., '2021-02-01')"
    )

    end_date: str = Field(
        description="End date in YYYY-MM-DD format (e.g., '2023-02-28')"
    )

    min_latitude: float = Field(
        ge=-90.0, le=90.0,
        description="Southern latitude bound (-90 to 90)"
    )

    max_latitude: float = Field(
        ge=-90.0, le=90.0,
        description="Northern latitude bound (-90 to 90)"
    )

    min_longitude: float = Field(
        ge=-180.0, le=360.0,
        description="Western longitude bound. Use -180 to 180 for Europe/Atlantic."
    )

    max_longitude: float = Field(
        ge=-180.0, le=360.0,
        description="Eastern longitude bound. Use -180 to 180 for Europe/Atlantic."
    )

    region: Optional[str] = Field(
        default=None,
        description=(
            "Optional predefined region (overrides lat/lon if specified):\n"
            "north_atlantic, mediterranean, nino34, global"
        )
    )

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
        return v

    @field_validator('variable_id')
    @classmethod
    def validate_variable(cls, v: str) -> str:
        from ..config import get_all_short_names
        short_name = get_short_name(v)
        valid_vars = get_all_short_names()  # DRY: use config as single source of truth
        if short_name not in valid_vars:
            logger.warning(f"Variable '{v}' may not be available. Will attempt anyway.")
        return v


# ============================================================================
# AUTO-DETECT QUERY TYPE
# ============================================================================

def _auto_detect_query_type(
    start_date: str,
    end_date: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
) -> str:
    """
    Auto-detect optimal query_type based on time/area rules.
    
    RULES:
    - TEMPORAL: time > 1 day AND area < 30°×30° (900 sq degrees)
    - SPATIAL:  time ≤ 1 day OR  area ≥ 30°×30°
    """
    # Calculate time span in days
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    time_days = (end - start).days + 1  # inclusive
    
    # Calculate area in square degrees
    lat_span = abs(max_lat - min_lat)
    lon_span = abs(max_lon - min_lon)
    area = lat_span * lon_span
    
    # Decision logic
    if time_days > 1 and area < 900:
        query_type = "temporal"
    else:
        query_type = "spatial"
    
    logger.info(f"Auto-detected query_type: {query_type} "
                f"(time={time_days}d, area={area:.0f}sq°)")
    
    return query_type


# ============================================================================
# WRAPPER FUNCTION (auto-adds query_type)
# ============================================================================

def retrieve_era5_data(
    variable_id: str,
    start_date: str,
    end_date: str,
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
    region: Optional[str] = None
) -> str:
    """
    Wrapper that auto-detects query_type and calls the real retrieval function.
    """
    # Auto-detect query type
    query_type = _auto_detect_query_type(
        start_date, end_date,
        min_latitude, max_latitude,
        min_longitude, max_longitude
    )
    
    # Call the real retrieval function
    return _retrieve_era5_data(
        query_type=query_type,
        variable_id=variable_id,
        start_date=start_date,
        end_date=end_date,
        min_latitude=min_latitude,
        max_latitude=max_latitude,
        min_longitude=min_longitude,
        max_longitude=max_longitude,
        region=region
    )


# ============================================================================
# LANGCHAIN TOOL CREATION
# ============================================================================

era5_tool = StructuredTool.from_function(
    func=retrieve_era5_data,
    name="retrieve_era5_data",
    description=(
        "Retrieves ERA5 climate reanalysis data from Earthmover's cloud archive.\n\n"
        "⚠️ query_type is AUTO-DETECTED - you don't need to specify it!\n\n"
        "Just provide:\n"
        "- variable_id: one of 22 ERA5 variables (sst, t2, d2, skt, u10, v10, u100, v100, "
        "sp, mslp, blh, cape, tcc, cp, lsp, tp, ssr, ssrd, tcw, tcwv, sd, stl1, swvl1)\n"
        "- start_date, end_date: YYYY-MM-DD format\n"
        "- lat/lon bounds: Use values from maritime route bounding box!\n\n"
        "DATA: 1975-2024.\n"
        "Returns file path. Load with: xr.open_zarr('PATH')"
    ),
    args_schema=ERA5RetrievalArgs
)
