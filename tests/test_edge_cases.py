"""
Edge-Case & Hardening Tests for Eurus
=======================================
Focused on retrieval edge cases discovered during manual testing:
prime-meridian crossing, future dates, invalid variables, filename
generation, cache behaviour, and routing with real dependencies.

Run with: pytest tests/test_edge_cases.py -v -s
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# RETRIEVAL HELPERS — pure-logic, no API calls
# ============================================================================

class TestFilenameGeneration:
    """Tests for generate_filename edge cases."""

    def test_negative_longitude_in_filename(self):
        from eurus.retrieval import generate_filename
        name = generate_filename(
            "sst", "temporal", "2023-01-01", "2023-01-31",
            min_latitude=30.0, max_latitude=46.0,
            min_longitude=-6.0, max_longitude=36.0,
        )
        assert name.endswith(".zarr")
        assert "lat30.00_46.00" in name
        assert "lon-6.00_36.00" in name

    def test_region_tag_overrides_coords(self):
        from eurus.retrieval import generate_filename
        name = generate_filename(
            "sst", "temporal", "2023-07-01", "2023-07-31",
            min_latitude=30, max_latitude=46,
            min_longitude=354, max_longitude=42,
            region="mediterranean",
        )
        assert "mediterranean" in name
        assert "lat" not in name  # region tag replaces coord string

    def test_filename_includes_data_source(self):
        """Current source must appear in the name so caches can't collide."""
        from eurus.retrieval import generate_filename
        from eurus.config import CONFIG
        name = generate_filename(
            "sst", "temporal", "2023-01-01", "2023-01-31",
            min_latitude=0, max_latitude=10,
            min_longitude=0, max_longitude=10,
        )
        assert name.startswith("earthmover-public-era5_")
        assert CONFIG.data_source == "earthmover-public/era5"

    def test_different_sources_do_not_share_a_cache_entry(self):
        """The same query against the old and new store must not collide."""
        from eurus.retrieval import generate_filename
        kwargs = dict(
            variable="sst", query_type="temporal",
            start="2023-01-01", end="2023-01-31",
            min_latitude=0, max_latitude=10,
            min_longitude=0, max_longitude=10,
        )
        new = generate_filename(**kwargs, source="earthmover-public/era5")
        old = generate_filename(**kwargs, source="earthmover-public/era5-surface-aws")
        assert new != old

    def test_source_tag_is_filename_safe(self):
        from eurus.retrieval import source_tag
        assert source_tag("earthmover-public/era5") == "earthmover-public-era5"
        assert source_tag("Org/Repo_Name.v2") == "org-repo-name-v2"
        assert "/" not in source_tag("a/b/c")


class TestGroupPath:
    """The zarr group path must track CONFIG, not be hardcoded."""

    def test_snippet_group_follows_config(self, monkeypatch):
        from eurus.config import CONFIG
        from eurus.retrieval import _arraylake_snippet
        monkeypatch.setattr(CONFIG, "data_group", "pressure")
        snippet = _arraylake_snippet(
            "t", "t", "spatial", "2020-01-01", "2020-01-02", 0, 10, 0, 10,
        )
        assert "group='pressure/spatial'" in snippet

    def test_snippet_group_defaults_to_single(self):
        from eurus.config import CONFIG
        from eurus.retrieval import _arraylake_snippet
        assert CONFIG.data_group == "single"
        snippet = _arraylake_snippet(
            "t2", "t2m", "temporal", "2020-01-01", "2020-01-02", 0, 10, 0, 10,
        )
        assert "group='single/temporal'" in snippet

    def test_format_coord_near_zero(self):
        from eurus.retrieval import _format_coord
        assert _format_coord(0.003) == "0.00"
        assert _format_coord(-0.004) == "0.00"
        assert _format_coord(0.01) == "0.01"


class TestFutureDateRejection:
    """Ensure retrieval rejects future start dates without touching the API."""

    def test_future_date_returns_error(self):
        from eurus.retrieval import retrieve_era5_data
        result = retrieve_era5_data(
            query_type="temporal",
            variable_id="sst",
            start_date="2099-01-01",
            end_date="2099-01-31",
            min_latitude=0, max_latitude=10,
            min_longitude=250, max_longitude=260,
        )
        assert "future" in result.lower()
        assert "Error" in result


# ============================================================================
# E2E RETRIEVAL — require ARRAYLAKE_API_KEY
# ============================================================================

@pytest.fixture(scope="module")
def has_arraylake_key():
    key = os.environ.get("ARRAYLAKE_API_KEY")
    if not key:
        pytest.skip("ARRAYLAKE_API_KEY not set")
    return True


class TestPrimeMeridianCrossing:
    """Verify data integrity when the request spans the 0° meridian."""

    @pytest.mark.slow
    def test_cross_meridian_longitude_continuity(self, has_arraylake_key):
        """
        Request u10 from -10°E to 15°E and check that the returned
        longitude axis has no gaps (step ≈ 0.25° everywhere).
        """
        import numpy as np
        import xarray as xr
        from eurus.retrieval import retrieve_era5_data
        from eurus.memory import reset_memory

        reset_memory()
        result = retrieve_era5_data(
            query_type="temporal",
            variable_id="u10",
            start_date="2024-01-15",
            end_date="2024-01-17",  # small window
            min_latitude=50.0,
            max_latitude=55.0,
            min_longitude=-10.0,
            max_longitude=15.0,
        )
        assert "SUCCESS" in result or "CACHE HIT" in result

        # Extract path and load
        path = None
        for line in result.split("\n"):
            if "Path:" in line:
                path = line.split("Path:")[-1].strip()
                break
        assert path and os.path.exists(path)

        ds = xr.open_dataset(path, engine="zarr")
        lons = ds["u10"].longitude.values
        diffs = np.diff(lons)
        # uniform step — no jump across 0°
        assert diffs.max() < 1.0, f"Gap in longitude: max step = {diffs.max()}"
        ds.close()


class TestInvalidVariableHandling:
    """Ensure retrieval returns a clear error for unavailable variables."""

    @pytest.mark.slow
    def test_swh_not_available(self, has_arraylake_key):
        from eurus.retrieval import retrieve_era5_data
        from eurus.memory import reset_memory

        reset_memory()
        result = retrieve_era5_data(
            query_type="temporal",
            variable_id="swh",
            start_date="2023-06-01",
            end_date="2023-06-07",
            min_latitude=40, max_latitude=50,
            min_longitude=0, max_longitude=10,
        )
        assert "not found" in result.lower() or "Error" in result
        assert "Available variables" in result or "available" in result.lower()


class TestCacheHitBehaviour:
    """Verify that repeated identical requests return CACHE HIT."""

    @pytest.mark.slow
    def test_second_request_is_cache_hit(self, has_arraylake_key):
        from eurus.retrieval import retrieve_era5_data
        from eurus.memory import reset_memory

        reset_memory()
        params = dict(
            query_type="temporal",
            variable_id="sst",
            start_date="2023-08-01",
            end_date="2023-08-03",
            min_latitude=35.0, max_latitude=37.0,
            min_longitude=15.0, max_longitude=18.0,
        )
        first = retrieve_era5_data(**params)
        assert "SUCCESS" in first or "CACHE HIT" in first

        second = retrieve_era5_data(**params)
        assert "CACHE HIT" in second


# ============================================================================
# ROUTING WITH REAL DEPENDENCIES
# ============================================================================

class TestRoutingIntegration:
    """Tests that use real scgraph (if installed)."""

    def test_hamburg_rotterdam_route(self):
        from eurus.tools.routing import HAS_ROUTING_DEPS, calculate_maritime_route
        if not HAS_ROUTING_DEPS:
            pytest.skip("scgraph not installed")

        result = calculate_maritime_route(
            origin_lat=53.5, origin_lon=8.5,
            dest_lat=52.4, dest_lon=4.9,
            month=6,
        )
        assert "MARITIME ROUTE CALCULATION COMPLETE" in result
        assert "Waypoints" in result or "waypoints" in result.lower()
        # distance should be reasonable (100–500 nm)
        assert "nautical miles" in result.lower()

    def test_long_route_across_atlantic(self):
        from eurus.tools.routing import HAS_ROUTING_DEPS, calculate_maritime_route
        if not HAS_ROUTING_DEPS:
            pytest.skip("scgraph not installed")

        result = calculate_maritime_route(
            origin_lat=40.7, origin_lon=-74.0,   # New York
            dest_lat=51.9, dest_lon=4.5,          # Rotterdam
            month=1,
        )
        assert "MARITIME ROUTE CALCULATION COMPLETE" in result
        # trans-Atlantic should produce plenty of waypoints
        assert "nautical miles" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
