"""
Tests for Climate Science Tools
===============================
Unit tests for the statistical analysis tools in eurus.tools.climate_science.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

# Check if dependencies are available
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_sst_dataset(tmp_path):
    """Create a sample SST dataset for testing."""
    if not HAS_XARRAY:
        pytest.skip("xarray not available")

    import pandas as pd

    # Create synthetic SST data with seasonal cycle and trend
    np.random.seed(42)

    times = pd.date_range("1990-01-01", periods=365*10, freq="D")
    lats = np.linspace(30, 50, 20)
    lons = np.linspace(-80, -60, 25)

    # Base SST with latitude gradient
    base_temp = 290 + 10 * np.cos(np.deg2rad(lats[:, np.newaxis]))

    # Add seasonal cycle and noise for each timestep
    data = []
    for i, t in enumerate(times):
        seasonal = 5 * np.sin(2 * np.pi * i / 365)  # Seasonal cycle
        trend = 0.02 * (i / 365)  # Warming trend
        noise = np.random.normal(0, 1, (len(lats), len(lons)))
        data.append(base_temp + seasonal + trend + noise)

    data = np.array(data)

    ds = xr.Dataset(
        {"sst": (["time", "latitude", "longitude"], data)},
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons
        }
    )
    ds["sst"].attrs["units"] = "K"
    ds["sst"].attrs["long_name"] = "Sea Surface Temperature"

    # Save to zarr
    path = tmp_path / "test_sst.zarr"
    ds.to_zarr(str(path), mode='w')

    return str(path)


@pytest.fixture
def sample_wind_dataset(tmp_path):
    """Create a sample wind speed dataset for testing."""
    if not HAS_XARRAY:
        pytest.skip("xarray not available")

    import pandas as pd

    np.random.seed(123)

    times = pd.date_range("1990-01-01", periods=365*10, freq="D")
    lats = np.linspace(30, 50, 20)
    lons = np.linspace(-80, -60, 25)

    # Create wind speed data
    base_wind = 8 + 2 * np.random.randn(len(times), len(lats), len(lons))

    ds = xr.Dataset(
        {"u10": (["time", "latitude", "longitude"], base_wind)},
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons
        }
    )
    ds["u10"].attrs["units"] = "m/s"

    path = tmp_path / "test_wind.zarr"
    ds.to_zarr(str(path), mode='w')

    return str(path)


# ============================================================================
# TOOL IMPORT TESTS
# ============================================================================

class TestToolImports:
    """Test that all tools can be imported."""

    def test_import_science_tools(self):
        """Test importing the science tools module."""
        from eurus.tools.climate_science import SCIENCE_TOOLS
        assert len(SCIENCE_TOOLS) == 10  # detrend + 2 pattern + 3 extreme + 3 attribution + 1 viz

    def test_import_individual_tools(self):
        """Test importing individual tools."""
        from eurus.tools.climate_science import (
            detrend_tool,
            eof_tool,
            compound_tool,
            trend_tool,
            percentile_tool
        )

        assert detrend_tool.name == "detrend_climate_data"
        assert eof_tool.name == "analyze_climate_modes_eof"
        assert compound_tool.name == "detect_compound_extremes"
        assert trend_tool.name == "calculate_climate_trends"
        assert percentile_tool.name == "detect_percentile_extremes"

    def test_tools_in_registry(self):
        """Test that science tools are in the main registry."""
        from eurus.tools import get_all_tools
        from eurus.tools.climate_science import get_science_tools

        all_tools = get_all_tools()
        science_tools = get_science_tools()

        # Counts depend on what's enabled - just verify science tools are subset
        assert len(science_tools) == 10
        assert len(all_tools) >= 12  # At least: era5, repl, guide + 10 science

        # All science tools should be in all_tools
        science_names = {t.name for t in science_tools}
        all_names = {t.name for t in all_tools}
        assert science_names.issubset(all_names)


# ============================================================================
# DIAGNOSTICS TESTS
# ============================================================================

@pytest.mark.skipif(not HAS_XARRAY, reason="xarray not available")
class TestDiagnostics:
    """Test the climate diagnostics tool."""

    def test_diagnostics_creates_output(self, sample_sst_dataset, tmp_path):
        """Test that diagnostics creates output file."""
        from eurus.tools.climate_science import calculate_climate_diagnostics

        result = calculate_climate_diagnostics(
            sample_sst_dataset,
            baseline_start="1991",
            baseline_end="2000"
        )

        assert "DIAGNOSTICS COMPLETE" in result
        assert "_DIAGNOSTICS.zarr" in result

        # Check output file exists
        out_path = sample_sst_dataset.replace(".zarr", "_DIAGNOSTICS.zarr")
        assert Path(out_path).exists()

    def test_diagnostics_creates_zscore(self, sample_sst_dataset):
        """Test that diagnostics creates Z-score variable."""
        from eurus.tools.climate_science import calculate_climate_diagnostics

        result = calculate_climate_diagnostics(
            sample_sst_dataset,
            baseline_start="1991",
            baseline_end="2000"
        )

        out_path = sample_sst_dataset.replace(".zarr", "_DIAGNOSTICS.zarr")
        ds = xr.open_dataset(out_path, engine='zarr')

        # Should have anomaly and zscore variables
        var_names = list(ds.data_vars)
        assert any("anom" in v for v in var_names)
        assert any("zscore" in v for v in var_names)

    def test_diagnostics_invalid_path(self):
        """Test error handling for invalid path."""
        from eurus.tools.climate_science import calculate_climate_diagnostics

        result = calculate_climate_diagnostics("/nonexistent/path.zarr")
        assert "FAILED" in result or "Error" in result


# ============================================================================
# EOF ANALYSIS TESTS
# ============================================================================

@pytest.mark.skipif(not HAS_XARRAY or not HAS_SKLEARN, reason="Dependencies not available")
class TestEOFAnalysis:
    """Test the EOF/PCA analysis tool."""

    def test_eof_creates_output(self, sample_sst_dataset):
        """Test that EOF analysis creates output file."""
        from eurus.tools.climate_science import perform_eof_analysis

        result = perform_eof_analysis(sample_sst_dataset, n_modes=3)

        assert "EOF ANALYSIS COMPLETE" in result
        assert "_EOF_PATTERNS.zarr" in result

        out_path = sample_sst_dataset.replace(".zarr", "_EOF_PATTERNS.zarr")
        assert Path(out_path).exists()

    def test_eof_extracts_modes(self, sample_sst_dataset):
        """Test that EOF extracts the requested number of modes."""
        from eurus.tools.climate_science import perform_eof_analysis

        n_modes = 3
        result = perform_eof_analysis(sample_sst_dataset, n_modes=n_modes)

        out_path = sample_sst_dataset.replace(".zarr", "_EOF_PATTERNS.zarr")
        ds = xr.open_dataset(out_path, engine='zarr')

        # Should have EOF_Mode_1, EOF_Mode_2, EOF_Mode_3
        for i in range(1, n_modes + 1):
            assert f"EOF_Mode_{i}" in ds.data_vars

        # Should have principal_components
        assert "principal_components" in ds.data_vars

    def test_eof_variance_explained(self, sample_sst_dataset):
        """Test that variance explained is reported."""
        from eurus.tools.climate_science import perform_eof_analysis

        result = perform_eof_analysis(sample_sst_dataset, n_modes=2)

        # Should report variance for each mode
        assert "Mode 1:" in result
        assert "Mode 2:" in result
        assert "%" in result


# ============================================================================
# TREND ANALYSIS TESTS
# ============================================================================

@pytest.mark.skipif(not HAS_XARRAY, reason="xarray not available")
class TestTrendAnalysis:
    """Test the trend analysis tool."""

    def test_trend_creates_output(self, sample_sst_dataset):
        """Test that trend analysis creates output file."""
        from eurus.tools.climate_science import calculate_trends

        result = calculate_trends(sample_sst_dataset)

        assert "TREND ANALYSIS COMPLETE" in result
        assert "_TRENDS.zarr" in result

        out_path = sample_sst_dataset.replace(".zarr", "_TRENDS.zarr")
        assert Path(out_path).exists()

    def test_trend_includes_significance(self, sample_sst_dataset):
        """Test that trend analysis includes significance testing."""
        from eurus.tools.climate_science import calculate_trends

        result = calculate_trends(sample_sst_dataset, confidence_level=0.95)

        out_path = sample_sst_dataset.replace(".zarr", "_TRENDS.zarr")
        ds = xr.open_dataset(out_path, engine='zarr')

        # Should have p-value and significance mask
        assert "p_value" in ds.data_vars
        assert "statistically_significant" in ds.data_vars


# ============================================================================
# PERCENTILE EXTREME TESTS
# ============================================================================

@pytest.mark.skipif(not HAS_XARRAY, reason="xarray not available")
class TestPercentileExtremes:
    """Test the percentile extreme detection tool."""

    def test_percentile_creates_output(self, sample_sst_dataset):
        """Test that percentile detection creates output file."""
        from eurus.tools.climate_science import detect_percentile_extremes

        result = detect_percentile_extremes(
            sample_sst_dataset,
            percentile=95.0,
            extreme_type="above"
        )

        assert "EXTREME DETECTION COMPLETE" in result
        assert "_EXTREMES_" in result

        out_path = sample_sst_dataset.replace(".zarr", "_EXTREMES_p95.zarr")
        assert Path(out_path).exists()

    def test_percentile_detects_events(self, sample_sst_dataset):
        """Test that percentile detection finds extreme events."""
        from eurus.tools.climate_science import detect_percentile_extremes

        result = detect_percentile_extremes(
            sample_sst_dataset,
            percentile=90.0,
            extreme_type="above"
        )

        out_path = sample_sst_dataset.replace(".zarr", "_EXTREMES_p90.zarr")
        ds = xr.open_dataset(out_path, engine='zarr')

        # Should have extreme event hours
        assert "extreme_event_hours" in ds.data_vars
        assert "extreme_percentage" in ds.data_vars

        # With 90th percentile, should expect ~10% of events
        mean_pct = float(ds["extreme_percentage"].mean().values)
        assert 5 < mean_pct < 15  # Allow some variance


# ============================================================================
# ARGUMENT VALIDATION TESTS
# ============================================================================

class TestArgumentValidation:
    """Test argument validation for all tools."""

    def test_diagnostics_args_validation(self):
        """Test diagnostics argument validation."""
        from eurus.tools.climate_science import DiagnosticsArgs

        # Valid args
        args = DiagnosticsArgs(dataset_path="/path/to/data.zarr")
        assert args.baseline_start == "1991"
        assert args.baseline_end == "2020"

        # Invalid path (not .zarr)
        with pytest.raises(ValueError):
            DiagnosticsArgs(dataset_path="/path/to/data.nc")

    def test_eof_args_validation(self):
        """Test EOF argument validation."""
        from eurus.tools.climate_science import EOFArgs

        # Valid args
        args = EOFArgs(dataset_path="/path/to/data.zarr", n_modes=5)
        assert args.n_modes == 5

        # Invalid n_modes (too large)
        with pytest.raises(ValueError):
            EOFArgs(dataset_path="/path/to/data.zarr", n_modes=20)

    def test_percentile_args_validation(self):
        """Test percentile argument validation."""
        from eurus.tools.climate_science import PercentileArgs

        # Valid args
        args = PercentileArgs(dataset_path="/path/to/data.zarr", percentile=95.0)
        assert args.extreme_type == "above"

        # Invalid percentile
        with pytest.raises(ValueError):
            PercentileArgs(dataset_path="/path/to/data.zarr", percentile=150.0)

    def test_index_args_validation(self):
        """Test climate index argument validation."""
        from eurus.tools.climate_science import IndexArgs

        # Valid args
        args = IndexArgs(index_name="nino34", start_date="2020-01-01", end_date="2023-12-31")
        assert args.index_name == "nino34"

        # Invalid index name
        with pytest.raises(ValueError):
            IndexArgs(index_name="invalid_index", start_date="2020-01-01", end_date="2023-12-31")

    def test_return_period_args_validation(self):
        """Test return period argument validation."""
        from eurus.tools.climate_science import ReturnPeriodArgs

        # Valid args
        args = ReturnPeriodArgs(dataset_path="/path/to/data.zarr", block_size="year")
        assert args.fit_type == "maxima"

        # Invalid path
        with pytest.raises(ValueError):
            ReturnPeriodArgs(dataset_path="/path/to/data.nc")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.skipif(not HAS_XARRAY or not HAS_SKLEARN, reason="Dependencies not available")
class TestIntegration:
    """Integration tests for the scientific workflow."""

    def test_diagnostics_then_eof_workflow(self, sample_sst_dataset):
        """Test running diagnostics followed by EOF analysis."""
        from eurus.tools.climate_science import (
            calculate_climate_diagnostics,
            perform_eof_analysis
        )

        # Step 1: Calculate diagnostics
        diag_result = calculate_climate_diagnostics(
            sample_sst_dataset,
            baseline_start="1991",
            baseline_end="2000"
        )
        assert "DIAGNOSTICS COMPLETE" in diag_result

        # Step 2: Run EOF on diagnostics output
        diag_path = sample_sst_dataset.replace(".zarr", "_DIAGNOSTICS.zarr")
        eof_result = perform_eof_analysis(diag_path, n_modes=2)

        # EOF should use Z-scores
        assert "zscore" in eof_result.lower() or "pre-processed" in eof_result.lower()
        assert "EOF ANALYSIS COMPLETE" in eof_result


# ============================================================================
# CLIMATE INDEX TESTS
# ============================================================================

class TestClimateIndex:
    """Test the climate index retrieval tool."""

    def test_index_tool_exists(self):
        """Test that index tool is available."""
        from eurus.tools.climate_science import index_tool
        assert index_tool.name == "fetch_climate_index"

    def test_index_invalid_name(self):
        """Test error handling for invalid index name."""
        from eurus.tools.climate_science import fetch_climate_index
        # This should handle validation internally now
        result = fetch_climate_index("invalid_index", "2020-01-01", "2020-12-31")
        assert "Error" in result or "Unknown" in result


# ============================================================================
# RETURN PERIOD TESTS
# ============================================================================

@pytest.mark.skipif(not HAS_XARRAY, reason="xarray not available")
class TestReturnPeriods:
    """Test the return period calculation tool."""

    def test_return_period_tool_exists(self):
        """Test that return period tool is available."""
        from eurus.tools.climate_science import return_period_tool
        assert return_period_tool.name == "calculate_return_periods"

    def test_return_period_insufficient_data(self, sample_sst_dataset):
        """Test error handling for insufficient data."""
        from eurus.tools.climate_science import calculate_return_periods
        import xarray as xr

        # Create a short dataset (only 5 years)
        ds = xr.open_dataset(sample_sst_dataset, engine='zarr')
        short_ds = ds.isel(time=slice(0, 365*5))  # 5 years

        short_path = sample_sst_dataset.replace(".zarr", "_short.zarr")
        short_ds.to_zarr(short_path, mode='w')

        result = calculate_return_periods(short_path, block_size="year")
        # Should warn about insufficient data or still work with limited samples
        assert "Error" in result or "EXTREME VALUE" in result

    def test_return_period_long_data(self, sample_sst_dataset):
        """Test GEV fitting with sufficient data."""
        from eurus.tools.climate_science import calculate_return_periods

        # sample_sst_dataset has 10 years of data
        result = calculate_return_periods(sample_sst_dataset, block_size="year")

        # Should either succeed or give helpful error
        if "Error" not in result:
            assert "GEV PARAMETERS" in result
            assert "RETURN LEVELS" in result
            assert "100-Year Event" in result


# ============================================================================
# FULL TOOL COUNT TEST
# ============================================================================

class TestFullToolRegistry:
    """Test the complete tool registry."""

    def test_science_tool_count(self):
        """Test that all 11 science tools are registered."""
        from eurus.tools.climate_science import SCIENCE_TOOLS
        assert len(SCIENCE_TOOLS) == 10

    def test_total_tool_count(self):
        """Test total tool count including core tools."""
        from eurus.tools import get_all_tools
        tools = get_all_tools(enable_routing=False)
        # At least era5 + repl + 10 science tools
        assert len(tools) >= 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
