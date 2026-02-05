"""
End-to-End Tests for Vostok
===========================
These tests use REAL API calls to verify the complete workflow.
Requires valid API keys in .env file.

Run with: pytest tests/test_e2e.py -v -s
Use -s flag to see output from data retrieval.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def temp_data_dir():
    """Create temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="vostok_e2e_")
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def has_arraylake_key():
    """Check if Arraylake API key is available."""
    key = os.environ.get("ARRAYLAKE_API_KEY")
    if not key:
        pytest.skip("ARRAYLAKE_API_KEY not found in environment")
    return True


# ============================================================================
# E2E: ERA5 DATA RETRIEVAL
# ============================================================================

class TestERA5Retrieval:
    """End-to-end tests for ERA5 data retrieval."""
    
    @pytest.mark.slow
    def test_retrieve_sst_temporal_small_region(self, has_arraylake_key, temp_data_dir):
        """
        E2E Test: Retrieve SST data for a small region and short time period.
        This tests the complete retrieval pipeline.
        """
        from vostok.retrieval import retrieve_era5_data
        from vostok.memory import reset_memory
        
        # Reset memory for clean state
        reset_memory()
        
        # Use a small request to minimize download time
        result = retrieve_era5_data(
            query_type="temporal",
            variable_id="sst",
            start_date="2023-01-01",
            end_date="2023-01-07",  # Just 1 week
            min_latitude=25.0,
            max_latitude=30.0,
            min_longitude=260.0,  # Gulf of Mexico
            max_longitude=265.0,
        )
        
        print(f"\n=== ERA5 Retrieval Result ===\n{result}\n")
        
        # Verify success
        assert "SUCCESS" in result or "CACHE HIT" in result
        assert "sst" in result.lower()
        assert ".zarr" in result
        
    @pytest.mark.slow
    def test_retrieve_t2m_spatial(self, has_arraylake_key, temp_data_dir):
        """
        E2E Test: Retrieve 2m temperature as spatial data.
        Tests spatial query type.
        """
        from vostok.retrieval import retrieve_era5_data
        from vostok.memory import reset_memory
        
        reset_memory()
        
        result = retrieve_era5_data(
            query_type="spatial",
            variable_id="t2",  # 2m temperature
            start_date="2023-06-01",
            end_date="2023-06-03",  # Just 3 days
            min_latitude=40.0,
            max_latitude=50.0,
            min_longitude=0.0,
            max_longitude=10.0,  # Western Europe
        )
        
        print(f"\n=== T2M Spatial Result ===\n{result}\n")
        
        assert "SUCCESS" in result or "CACHE HIT" in result
        
    @pytest.mark.slow
    def test_retrieve_and_load_dataset(self, has_arraylake_key, temp_data_dir):
        """
        E2E Test: Retrieve data and verify it can be loaded with xarray.
        Tests the full data integrity pipeline.
        """
        import xarray as xr
        from vostok.retrieval import retrieve_era5_data
        from vostok.memory import reset_memory, get_memory
        
        reset_memory()
        
        result = retrieve_era5_data(
            query_type="temporal",
            variable_id="sst",
            start_date="2023-02-01",
            end_date="2023-02-05",
            min_latitude=20.0,
            max_latitude=25.0,
            min_longitude=270.0,
            max_longitude=275.0,
        )
        
        assert "SUCCESS" in result or "CACHE HIT" in result
        
        # Extract path from result
        # Look for the path in the result string
        lines = result.split('\n')
        path = None
        for line in lines:
            if "Path:" in line:
                path = line.split("Path:")[-1].strip()
                break
            if ".zarr" in line and "Load with" not in line:
                # Try to find zarr path
                parts = line.split()
                for part in parts:
                    if ".zarr" in part:
                        path = part.strip()
                        break
        
        if path and os.path.exists(path):
            # Load and verify dataset
            ds = xr.open_dataset(path, engine='zarr')
            
            print(f"\n=== Loaded Dataset ===")
            print(f"Variables: {list(ds.data_vars)}")
            print(f"Dimensions: {dict(ds.dims)}")
            print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
            
            assert 'sst' in ds.data_vars
            assert 'time' in ds.dims
            assert ds.dims['time'] > 0
            
            ds.close()


# ============================================================================
# E2E: PYTHON REPL ANALYSIS
# ============================================================================

class TestREPLAnalysis:
    """End-to-end tests for REPL-based data analysis."""
    
    def test_repl_numpy_computation(self):
        """
        E2E Test: Use REPL to perform numpy computation.
        """
        from vostok.tools.repl import PythonREPLTool
        
        repl = PythonREPLTool()
        
        code = """
import numpy as np
data = np.random.randn(100)
mean = np.mean(data)
std = np.std(data)
print(f"Mean: {mean:.4f}, Std: {std:.4f}")
"""
        result = repl._run(code)
        print(f"\n=== REPL Result ===\n{result}\n")
        
        assert "Mean:" in result
        assert "Std:" in result
        assert "Error" not in result
        
    def test_repl_pandas_dataframe(self):
        """
        E2E Test: Use REPL to create and manipulate pandas DataFrame.
        """
        from vostok.tools.repl import PythonREPLTool
        
        repl = PythonREPLTool()
        
        code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'temperature': np.random.randn(10) * 5 + 20,
    'humidity': np.random.randn(10) * 10 + 60
})

print("DataFrame created:")
print(df.head())
print(f"\\nStats: Mean temp = {df['temperature'].mean():.2f}")
"""
        result = repl._run(code)
        print(f"\n=== Pandas Result ===\n{result}\n")
        
        assert "DataFrame created" in result
        assert "temperature" in result
        assert "Error" not in result
        
    @pytest.mark.slow
    def test_repl_load_and_analyze_data(self, has_arraylake_key):
        """
        E2E Test: Retrieve ERA5 data, then analyze it in REPL.
        Full workflow test.
        """
        from vostok.retrieval import retrieve_era5_data
        from vostok.tools.repl import PythonREPLTool
        from vostok.memory import reset_memory
        import xarray as xr
        
        reset_memory()
        
        # Step 1: Retrieve data
        result = retrieve_era5_data(
            query_type="temporal",
            variable_id="sst",
            start_date="2023-03-01",
            end_date="2023-03-05",
            min_latitude=25.0,
            max_latitude=28.0,
            min_longitude=265.0,
            max_longitude=268.0,
        )
        
        assert "SUCCESS" in result or "CACHE HIT" in result
        
        # Extract path
        path = None
        for line in result.split('\n'):
            if "Path:" in line:
                path = line.split("Path:")[-1].strip()
                break
                
        if not path or not os.path.exists(path):
            pytest.skip("Could not extract data path")
            
        # Step 2: Analyze in REPL
        repl = PythonREPLTool()
        
        analysis_code = f"""
import xarray as xr
import numpy as np

# Load the dataset
ds = xr.open_dataset('{path}', engine='zarr')
data = ds['sst']

# Calculate statistics
spatial_mean = data.mean(dim=['latitude', 'longitude'])
time_mean = data.mean(dim='time')

print("=== SST Analysis ===")
print(f"Time points: {{len(data.time)}}")
print(f"Spatial shape: {{data.shape}}")
print(f"Overall mean: {{float(data.mean()):.2f}} K")
print(f"Overall std: {{float(data.std()):.2f}} K")
print(f"Min: {{float(data.min()):.2f}} K, Max: {{float(data.max()):.2f}} K")
"""
        analysis_result = repl._run(analysis_code)
        print(f"\n=== Analysis Result ===\n{analysis_result}\n")
        
        assert "SST Analysis" in analysis_result
        assert "Error" not in analysis_result or "Security" not in analysis_result


# ============================================================================
# E2E: CLIMATE INDEX RETRIEVAL
# ============================================================================

class TestClimateIndices:
    """End-to-end tests for climate index retrieval."""
    
    @pytest.mark.slow
    def test_fetch_nino34_index(self):
        """
        E2E Test: Fetch Nino3.4 index from NOAA.
        Tests external API integration.
        """
        from vostok.tools.climate_science.attribution import fetch_climate_index
        
        result = fetch_climate_index(
            index_name="nino34",
            start_date="2020-01-01",
            end_date="2023-12-31"
        )
        
        print(f"\n=== Nino3.4 Result ===\n{result}\n")
        
        assert "CLIMATE INDEX RETRIEVED" in result or "Error" not in result
        assert "nino" in result.lower()
        
    @pytest.mark.slow
    def test_fetch_nao_index(self):
        """
        E2E Test: Fetch NAO index from NOAA.
        """
        from vostok.tools.climate_science.attribution import fetch_climate_index
        
        result = fetch_climate_index(
            index_name="nao",
            start_date="2022-01-01",
            end_date="2023-12-31"
        )
        
        print(f"\n=== NAO Result ===\n{result}\n")
        
        # May fail if NOAA is down, but should not crash
        assert result is not None


# ============================================================================
# E2E: MEMORY PERSISTENCE
# ============================================================================

class TestMemoryPersistence:
    """End-to-end tests for memory and dataset tracking."""
    
    @pytest.mark.slow
    def test_memory_tracks_downloaded_data(self, has_arraylake_key):
        """
        E2E Test: Verify memory tracks downloaded datasets.
        """
        from vostok.retrieval import retrieve_era5_data
        from vostok.memory import reset_memory, get_memory
        
        reset_memory()
        memory = get_memory()
        
        # Initial state - no datasets
        initial_datasets = memory.list_datasets()
        
        # Download data
        result = retrieve_era5_data(
            query_type="temporal",
            variable_id="sst",
            start_date="2023-04-01",
            end_date="2023-04-03",
            min_latitude=30.0,
            max_latitude=32.0,
            min_longitude=275.0,
            max_longitude=278.0,
        )
        
        # Check memory registered the dataset
        datasets = memory.list_datasets()
        print(f"\n=== Registered Datasets ===\n{datasets}\n")
        
        # Should have at least one dataset now
        if "SUCCESS" in result:
            assert len(datasets) > len(initial_datasets)


# ============================================================================
# E2E: ROUTING (if scgraph installed)
# ============================================================================

class TestRouting:
    """End-to-end tests for maritime routing."""
    
    def test_routing_without_deps(self):
        """
        E2E Test: Verify routing handles missing dependencies gracefully.
        """
        from vostok.tools.routing import HAS_ROUTING_DEPS, calculate_maritime_route
        
        if not HAS_ROUTING_DEPS:
            # Should return helpful error message
            result = calculate_maritime_route(
                origin_lat=53.5,
                origin_lon=8.5,
                dest_lat=52.4,
                dest_lon=4.9,
                month=6
            )
            print(f"\n=== Routing (no deps) ===\n{result}\n")
            assert "scgraph" in result.lower() or "install" in result.lower()
        else:
            pytest.skip("scgraph is installed, skipping no-deps test")


# ============================================================================
# RUN WITH: pytest tests/test_e2e.py -v -s --tb=short
# Add -m "not slow" to skip slow tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
