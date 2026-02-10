"""
Maritime Routing Tool
=====================
Strictly calculates maritime routes using global shipping lane graphs.
Does NOT perform weather analysis. Returns waypoints for the Agent to analyze.

Dependencies:
- scgraph (for maritime pathfinding)
"""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Any
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# Check for optional dependencies
HAS_ROUTING_DEPS = False
try:
    import scgraph
    from scgraph.geographs.marnet import marnet_geograph
    HAS_ROUTING_DEPS = True
except ImportError:
    pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _normalize_lon(lon: float) -> float:
    """Convert longitude to -180 to 180 range (scgraph format)."""
    # Efficient modulo operation - prevents infinite loop on extreme values
    return ((lon + 180) % 360) - 180





def _get_maritime_path(origin: Tuple[float, float], dest: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Calculate shortest maritime path using scgraph."""
    if not HAS_ROUTING_DEPS:
        raise ImportError("Dependency 'scgraph' is missing.")

    # Normalize longitudes for scgraph (-180 to 180)
    origin_lon = _normalize_lon(origin[1])
    dest_lon = _normalize_lon(dest[1])

    graph = marnet_geograph
    path_dict = graph.get_shortest_path(
        origin_node={"latitude": origin[0], "longitude": origin_lon},
        destination_node={"latitude": dest[0], "longitude": dest_lon}
    )
    return [(p[0], p[1]) for p in path_dict.get('coordinate_path', [])]


def _interpolate_route(
    path: List[Tuple[float, float]],
    speed_knots: float,
    departure: datetime
) -> List[dict]:
    """Convert path to waypoints with timestamps. Keeps ALL points for risk assessment."""
    try:
        from geopy.distance import great_circle
    except ImportError:
        # Proper Haversine fallback for accurate distance at all latitudes
        import math
        from collections import namedtuple
        Distance = namedtuple('Distance', ['km'])
        def great_circle(p1, p2):
            lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
            lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return Distance(km=6371 * c)  # Earth radius in km

    speed_kmh = speed_knots * 1.852
    waypoints = []
    current_time = departure

    # Add ALL points from scgraph - each is a navigation waypoint
    # Risk assessment needs every geographic point, not time-filtered ones
    for i, point in enumerate(path):
        if i == 0:
            step = "Origin"
        elif i == len(path) - 1:
            step = "Destination"
        else:
            step = f"Waypoint {i}"

        # Calculate time to reach this point
        if i > 0:
            prev = path[i-1]
            dist = great_circle(prev, point).km
            hours = dist / speed_kmh if speed_kmh > 0 else 0
            current_time += timedelta(hours=hours)

        waypoints.append({
            "lat": point[0],
            "lon": point[1],
            "time": current_time.strftime("%Y-%m-%d %H:%M"),
            "step": step
        })
        
    return waypoints


# ============================================================================
# TOOL FUNCTION
# ============================================================================

def calculate_maritime_route(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    month: int,
    year: int = None,
    speed_knots: float = 14.0
) -> str:
    """
    Calculates the detailed maritime route waypoints.
    """
    if not HAS_ROUTING_DEPS:
        return "Error: 'scgraph' not installed."

    try:
        path = _get_maritime_path((origin_lat, origin_lon), (dest_lat, dest_lon))
        
        # Use provided year or calculate based on current date
        if year is None:
            now = datetime.now()
            year = now.year if month >= now.month else now.year + 1
        departure = datetime(year, month, 15)
        
        waypoints = _interpolate_route(path, speed_knots, departure)
        
        # Calculate bounding box with buffer for weather data
        lats = [w['lat'] for w in waypoints]
        lons = [w['lon'] for w in waypoints]
        
        min_lat = max(-90, min(lats) - 5)
        max_lat = min(90, max(lats) + 5)
        
        # Detect dateline crossing: if lon range > 180°, the route crosses -180/+180
        lon_range = max(lons) - min(lons)
        if lon_range > 180:
            # Route crosses dateline - need to recalculate
            # Split lons into positive and negative, find the gap
            pos_lons = [l for l in lons if l >= 0]
            neg_lons = [l for l in lons if l < 0]
            if pos_lons and neg_lons:
                # Route goes from ~+179 to ~-179 - use 0-360 system
                lons_360 = [(l + 360) if l < 0 else l for l in lons]
                min_lon = max(0, min(lons_360) - 5)
                max_lon = min(360, max(lons_360) + 5)
            else:
                min_lon = max(-180, min(lons) - 5)
                max_lon = min(180, max(lons) + 5)
        else:
            min_lon = max(-180, min(lons) - 5)
            max_lon = min(180, max(lons) + 5)

        # Format waypoints as Python-ready list (keep original -180/+180 format)
        waypoint_list = "[\n" + ",\n".join([
            f"    ({w['lat']:.2f}, {w['lon']:.2f})"
            for w in waypoints
        ]) + "\n]"

        # Calculate total distance
        total_nm = 0
        try:
            from geopy.distance import great_circle
            for i in range(1, len(waypoints)):
                d = great_circle(
                    (waypoints[i-1]['lat'], waypoints[i-1]['lon']),
                    (waypoints[i]['lat'], waypoints[i]['lon'])
                ).nautical
                total_nm += d
        except ImportError:
            # Haversine fallback for distance calculation
            import math
            for i in range(1, len(waypoints)):
                lat1, lon1 = math.radians(waypoints[i-1]['lat']), math.radians(waypoints[i-1]['lon'])
                lat2, lon2 = math.radians(waypoints[i]['lat']), math.radians(waypoints[i]['lon'])
                dlat, dlon = lat2 - lat1, lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                total_nm += 6371 * c / 1.852  # km to nm

        eta_days = total_nm / (speed_knots * 24)

        output = f"""
================================================================================
                        MARITIME ROUTE CALCULATION COMPLETE
================================================================================

ROUTE SUMMARY:
  Origin:      ({origin_lat:.2f}, {origin_lon:.2f})
  Destination: ({dest_lat:.2f}, {dest_lon:.2f})
  Distance:    ~{total_nm:.0f} nautical miles
  Speed:       {speed_knots} knots
  ETA:         ~{eta_days:.1f} days
  Waypoints:   {len(waypoints)} checkpoints

WAYPOINT COORDINATES (for risk analysis):
{waypoint_list}

DATA REGION (with 5° buffer):
  Latitude:  [{min_lat:.1f}, {max_lat:.1f}]
  Longitude: [{min_lon:.1f}, {max_lon:.1f}]

================================================================================
                        MANDATORY RISK ASSESSMENT PROTOCOL
================================================================================

STEP 1: DOWNLOAD CLIMATOLOGICAL DATA
  Call `retrieve_era5_data` with:
    - variable: 'u10' and 'v10' (10m wind components) for wind speed analysis
    - query_type: 'spatial'
    - region bounds: lat=[{min_lat:.1f}, {max_lat:.1f}], lon=[{min_lon:.1f}, {max_lon:.1f}]
    - dates: Month {month} for LAST 3 YEARS (e.g., {month}/2021, {month}/2022, {month}/2023)

  ⚠️ WARNING: Large bounding boxes can cause OOM/timeout!
  If (max_lon - min_lon) > 60° or (max_lat - min_lat) > 40°:
    - Do NOT download spatial data for the whole route at once
    - Instead, iterate through waypoints and download small chunks
    - Or sample every Nth waypoint for point-based temporal queries

  WHY 3 YEARS? To build climatological statistics, not just one snapshot.

STEP 2: GET ANALYSIS PROTOCOL
  Call `get_analysis_guide(topic='maritime_visualization')`
  
  Or for full workflow: `get_analysis_guide(topic='maritime_route')`

  This will provide methodology for:
    - Lagrangian risk assessment (ship vs. stationary climate data)
    - Threshold definitions (what wind speed is dangerous)
    - Risk aggregation formulas
    - Route deviation recommendations

STEP 3: EXECUTE ANALYSIS
  Use python_repl to:
    1. Load the downloaded data
    2. Extract values at each waypoint
    3. Calculate risk metrics per the methodology
    4. Generate risk map and report

================================================================================
"""
        return output

    except Exception as e:
        return f"Routing Calculation Failed: {str(e)}"


# ============================================================================
# ARGUMENT SCHEMA
# ============================================================================

class RouteArgs(BaseModel):
    origin_lat: float = Field(description="Latitude of origin")
    origin_lon: float = Field(description="Longitude of origin")
    dest_lat: float = Field(description="Latitude of destination")
    dest_lon: float = Field(description="Longitude of destination")
    month: int = Field(description="Month of travel (1-12)")
    year: int = Field(default=None, description="Year for analysis. Defaults to upcoming occurrence of month.")
    speed_knots: float = Field(default=14.0, description="Speed in knots")


# ============================================================================
# LANGCHAIN TOOL
# ============================================================================

routing_tool = StructuredTool.from_function(
    func=calculate_maritime_route,
    name="calculate_maritime_route",
    description="Calculates a realistic maritime route (avoiding land). Returns a list of time-stamped waypoints. DOES NOT check weather.",
    args_schema=RouteArgs
)
