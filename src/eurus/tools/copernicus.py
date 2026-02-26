"""
Copernicus Marine Retrieval Tool (LangChain wrapper)
=====================================================
Thin wrapper over eurus.copernicus.retrieve_copernicus_data.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool

from ..copernicus import retrieve_copernicus_data as _retrieve, list_available_variables

logger = logging.getLogger(__name__)


class CopernicusRetrievalArgs(BaseModel):
    variable_id: str = Field(
        description=(
            "Copernicus Marine variable. Available variables:\n"
            "GLORYS12 reanalysis (1993-present, daily):\n"
            "  thetao  — Sea water potential temperature (°C) [has depth]\n"
            "  so      — Sea water salinity (PSU) [has depth]\n"
            "  uo      — Eastward current (m/s) [has depth]\n"
            "  vo      — Northward current (m/s) [has depth]\n"
            "  zos     — Sea surface height (m)\n"
            "  mlotst  — Mixed layer depth (m)\n"
            "  siconc  — Sea ice concentration (0-1)\n"
            "  sithick — Sea ice thickness (m)\n"
            "  bottomT — Sea floor temperature (°C)\n"
            "DUACS altimetry (1993-present, daily):\n"
            "  sla  — Sea level anomaly (m)\n"
            "  adt  — Absolute dynamic topography (m)\n"
            "  ugos — Geostrophic eastward velocity (m/s)\n"
            "  vgos — Geostrophic northward velocity (m/s)\n"
            "WAVERYS waves (1993-present, 3-hourly):\n"
            "  VHM0  — Significant wave height (m)\n"
            "  VMDR  — Mean wave direction (°)\n"
            "  VTM10 — Mean wave period (s)\n"
            "Common aliases accepted: 'temperature', 'salinity', 'mld', "
            "'wave_height', 'sea_level', 'sea_ice', 'bottom_temperature'."
        )
    )

    start_date: str = Field(description="Start date YYYY-MM-DD (data available from 1993-01-01)")
    end_date:   str = Field(description="End date YYYY-MM-DD")

    min_latitude:  float = Field(ge=-90.0,  le=90.0,  description="Southern bound (-90 to 90)")
    max_latitude:  float = Field(ge=-90.0,  le=90.0,  description="Northern bound (-90 to 90)")
    min_longitude: float = Field(ge=-180.0, le=180.0, description="Western bound (-180 to 180)")
    max_longitude: float = Field(ge=-180.0, le=180.0, description="Eastern bound (-180 to 180)")

    depth_m: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Depth in metres for 3-D variables (thetao, so, uo, vo). "
            "None = full water column (all depth levels). "
            "Use 0.5 to retrieve only the surface layer (~0.5 m). "
            "Ignored for 2-D variables. "
            "For bottom temperature use variable='bottomT' instead. "
            "For mixed-layer averages retrieve mlotst first, then use python_repl."
        )
    )

    region: Optional[str] = Field(
        default=None,
        description="Optional predefined region name (overrides lat/lon). E.g. 'north_atlantic', 'mediterranean'."
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD, got: {v}")
        return v


def _run(
    variable_id: str,
    start_date: str,
    end_date: str,
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
    depth_m: Optional[float] = None,
    region: Optional[str] = None,
) -> str:
    return _retrieve(
        variable_id   = variable_id,
        start_date    = start_date,
        end_date      = end_date,
        min_latitude  = min_latitude,
        max_latitude  = max_latitude,
        min_longitude = min_longitude,
        max_longitude = max_longitude,
        depth_m       = depth_m,
        region        = region,
    )


copernicus_tool = StructuredTool.from_function(
    func        = _run,
    name        = "retrieve_copernicus_data",
    description = (
        "Retrieve ocean reanalysis / analysis data from the Copernicus Marine Service (CMEMS).\n\n"
        "Covers 1993-present. Datasets: GLORYS12 physics reanalysis, DUACS altimetry, WAVERYS waves.\n\n"
        "Provide variable_id, date range, and bounding box. "
        "For depth variables (thetao, so, uo, vo): depth_m=None retrieves the full water column; "
        "use depth_m=0.5 for surface only, or specify a depth in metres for a single level. "
        "Returns local Zarr file path. Load with: xr.open_dataset('PATH', engine='zarr')"
    ),
    args_schema = CopernicusRetrievalArgs,
)
