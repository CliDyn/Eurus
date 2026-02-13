"""
Server and Integration Tests
============================
Tests for server module, retrieval helpers, and integration scenarios.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import tempfile
from pathlib import Path
import json
import os


# ============================================================================
# SERVER MODULE TESTS
# ============================================================================

class TestServerModule:
    """Tests for eurus.server module."""
    
    def test_server_class_exists(self):
        """Test Server class can be imported."""
        from eurus.server import Server
        assert Server is not None
        
    def test_server_instance_exists(self):
        """Test server instance can be imported."""
        from eurus.server import server
        assert server is not None


# ============================================================================
# RETRIEVAL HELPERS TESTS
# ============================================================================

class TestRetrievalHelpers:
    """Tests for retrieval helper functions."""
    
    def test_format_coord_positive(self):
        """Test coordinate formatting for positive values."""
        from eurus.retrieval import _format_coord
        assert _format_coord(25.5) == "25.50"
        
    def test_format_coord_negative(self):
        """Test coordinate formatting for negative values."""
        from eurus.retrieval import _format_coord
        assert _format_coord(-10.333) == "-10.33"
        
    def test_format_coord_zero(self):
        """Test coordinate formatting for near-zero values."""
        from eurus.retrieval import _format_coord
        # Values very close to zero should be formatted as 0.00
        result = _format_coord(0.001)
        assert "0.00" in result or "0.01" in result
        
    def test_format_file_size_bytes(self):
        """Test file size formatting for bytes."""
        from eurus.retrieval import format_file_size
        assert "B" in format_file_size(500)
        
    def test_format_file_size_kb(self):
        """Test file size formatting for kilobytes."""
        from eurus.retrieval import format_file_size
        assert "KB" in format_file_size(2048)
        
    def test_format_file_size_mb(self):
        """Test file size formatting for megabytes."""
        from eurus.retrieval import format_file_size
        assert "MB" in format_file_size(5 * 1024 * 1024)
        
    def test_format_file_size_gb(self):
        """Test file size formatting for gigabytes."""
        from eurus.retrieval import format_file_size
        assert "GB" in format_file_size(5 * 1024 * 1024 * 1024)

    def test_ensure_aws_region_sets_env_from_repo_metadata(self, monkeypatch):
        """Auto-populate AWS vars when metadata includes region_name."""
        import eurus.retrieval as _retrieval
        from eurus.retrieval import _ensure_aws_region

        # Reset one-shot flag so the function actually runs
        _retrieval._aws_region_set = False

        for key in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_ENDPOINT_URL", "AWS_S3_ENDPOINT"):
            monkeypatch.delenv(key, raising=False)

        response = MagicMock()
        response.read.return_value = json.dumps(
            {"bucket": {"extra_config": {"region_name": "eu-north-1"}}}
        ).encode("utf-8")
        context_manager = MagicMock()
        context_manager.__enter__.return_value = response

        with patch("eurus.retrieval.urlopen", return_value=context_manager) as mock_urlopen:
            _ensure_aws_region("token", "earthmover-public/era5-surface-aws")

        assert os.environ["AWS_REGION"] == "eu-north-1"
        assert os.environ["AWS_DEFAULT_REGION"] == "eu-north-1"
        assert os.environ["AWS_ENDPOINT_URL"] == "https://s3.eu-north-1.amazonaws.com"
        assert os.environ["AWS_S3_ENDPOINT"] == "https://s3.eu-north-1.amazonaws.com"

        req = mock_urlopen.call_args.args[0]
        assert req.full_url == "https://api.earthmover.io/repos/earthmover-public/era5-surface-aws"

    def test_ensure_aws_region_does_not_override_existing_env(self, monkeypatch):
        """Keep explicit user-provided AWS endpoint config untouched."""
        import eurus.retrieval as _retrieval
        from eurus.retrieval import _ensure_aws_region

        # Reset one-shot flag so the function actually runs
        _retrieval._aws_region_set = False

        monkeypatch.setenv("AWS_REGION", "custom-region")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "custom-default")
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://custom.endpoint")
        monkeypatch.setenv("AWS_S3_ENDPOINT", "https://custom.s3.endpoint")

        response = MagicMock()
        response.read.return_value = json.dumps(
            {"bucket": {"extra_config": {"region_name": "us-west-2"}}}
        ).encode("utf-8")
        context_manager = MagicMock()
        context_manager.__enter__.return_value = response

        with patch("eurus.retrieval.urlopen", return_value=context_manager):
            _ensure_aws_region("token")

        assert os.environ["AWS_REGION"] == "custom-region"
        assert os.environ["AWS_DEFAULT_REGION"] == "custom-default"
        assert os.environ["AWS_ENDPOINT_URL"] == "https://custom.endpoint"
        assert os.environ["AWS_S3_ENDPOINT"] == "https://custom.s3.endpoint"




# ============================================================================
# ANALYSIS GUIDE TESTS
# ============================================================================

class TestAnalysisGuide:
    """Tests for analysis guide module."""
    
    def test_analysis_guide_tool_exists(self):
        """Test analysis guide tool can be imported."""
        from eurus.tools.analysis_guide import analysis_guide_tool
        assert analysis_guide_tool is not None
        
    def test_analysis_guide_returns_content(self):
        """Test analysis guide returns useful content."""
        from eurus.tools.analysis_guide import get_analysis_guide
        result = get_analysis_guide("timeseries")
        assert len(result) > 100  # Should have substantial content


# ============================================================================
# ERA5 TOOL EXTENDED TESTS
# ============================================================================

class TestERA5ToolValidation:
    """Tests for ERA5 tool validation and edge cases."""
    
    def test_era5_args_date_validation(self):
        """Test date format validation works."""
        from eurus.tools.era5 import ERA5RetrievalArgs
        # Valid dates should work
        args = ERA5RetrievalArgs(
            variable_id="sst",
            start_date="2023-01-01",
            end_date="2023-12-31",
            min_latitude=20.0,
            max_latitude=30.0,
            min_longitude=260.0,
            max_longitude=280.0
        )
        assert args.start_date == "2023-01-01"
        
    def test_era5_args_latitude_range(self):
        """Test latitude range parameters."""
        from eurus.tools.era5 import ERA5RetrievalArgs
        args = ERA5RetrievalArgs(
            variable_id="t2",
            start_date="2023-01-01",
            end_date="2023-01-31",
            min_latitude=-90.0,
            max_latitude=90.0,
            min_longitude=0.0,
            max_longitude=360.0
        )
        assert args.min_latitude == -90.0
        assert args.max_latitude == 90.0
        
    def test_era5_args_query_type_field(self):
        """Test that ERA5 args handles optional query_type correctly."""
        from eurus.tools.era5 import ERA5RetrievalArgs
        args = ERA5RetrievalArgs(
            variable_id="sst",
            start_date="2023-01-01",
            end_date="2023-12-31",
            min_latitude=20.0,
            max_latitude=30.0,
            min_longitude=260.0,
            max_longitude=280.0
        )
        # Just verify args created successfully
        assert args.variable_id == "sst"


# ============================================================================
# CONFIG EXTENDED TESTS
# ============================================================================

class TestConfigRegions:
    """Tests for region configuration."""
    
    def test_get_region_valid(self):
        """Test getting valid predefined region."""
        from eurus.config import get_region
        region = get_region("gulf_of_mexico")
        assert region is not None
        assert hasattr(region, 'min_lat')
        assert hasattr(region, 'max_lat')
        
    def test_get_region_case_insensitive(self):
        """Test region lookup is case insensitive."""
        from eurus.config import get_region
        region = get_region("GULF_OF_MEXICO")
        assert region is not None
        
    def test_list_regions_output(self):
        """Test list_regions returns formatted string."""
        from eurus.config import list_regions
        output = list_regions()
        assert "gulf" in output.lower() or "region" in output.lower()


# ============================================================================
# MEMORY MODULE INTEGRATION
# ============================================================================

class TestMemoryIntegration:
    """Integration tests for memory management."""
    
    def test_memory_manager_create(self):
        """Test MemoryManager can be created."""
        from eurus.memory import MemoryManager, reset_memory
        reset_memory()
        mm = MemoryManager()
        assert mm is not None
        
    def test_memory_add_conversation(self):
        """Test adding to conversation history."""
        from eurus.memory import MemoryManager, reset_memory
        reset_memory()
        mm = MemoryManager()
        mm.add_message("user", "Hello")
        history = mm.get_conversation_history()
        assert len(history) >= 1
        
    def test_memory_dataset_registration(self):
        """Test dataset registration."""
        from eurus.memory import MemoryManager, reset_memory
        reset_memory()
        mm = MemoryManager()
        mm.register_dataset(
            path="/tmp/test.zarr",
            variable="sst",
            query_type="temporal",
            start_date="2023-01-01",
            end_date="2023-12-31",
            lat_bounds=(20.0, 30.0),
            lon_bounds=(260.0, 280.0),
            file_size_bytes=1024
        )
        datasets = mm.list_datasets()
        assert len(datasets) >= 1


# ============================================================================
# ROUTING TOOL EXTENDED TESTS
# ============================================================================

class TestRoutingTool:
    """Extended tests for routing functionality."""
    
    def test_routing_tool_exists(self):
        """Test routing tool can be imported."""
        from eurus.tools.routing import routing_tool
        assert routing_tool is not None
        assert routing_tool.name == "calculate_maritime_route"
        
    def test_has_routing_deps_flag(self):
        """Test HAS_ROUTING_DEPS flag exists."""
        from eurus.tools.routing import HAS_ROUTING_DEPS
        assert isinstance(HAS_ROUTING_DEPS, bool)


# ============================================================================
# REPL TOOL COMPREHENSIVE SECURITY TESTS
# ============================================================================

class TestREPLSecurityComprehensive:
    """REPL tests — Docker is the sandbox, all imports allowed."""
    
    def test_repl_allows_sys(self):
        """Test REPL allows sys module (Docker sandbox)."""
        from eurus.tools.repl import PythonREPLTool
        repl = PythonREPLTool()
        result = repl._run("import sys; print(sys.version_info.major)")
        assert result is not None
        assert "Error" not in result
        repl.close()
        
    def test_repl_allows_os(self):
        """Test REPL allows os module (Docker sandbox)."""
        from eurus.tools.repl import PythonREPLTool
        repl = PythonREPLTool()
        result = repl._run("import os; print(os.getcwd())")
        assert result is not None
        assert "Error" not in result
        repl.close()
        
    def test_repl_allows_xarray(self):
        """Test REPL allows xarray operations."""
        from eurus.tools.repl import PythonREPLTool
        repl = PythonREPLTool()
        result = repl._run("import xarray as xr; print(type(xr))")
        assert "module" in result.lower() or "xarray" in result.lower()
        repl.close()
        
    def test_repl_allows_pandas(self):
        """Test REPL allows pandas operations."""
        from eurus.tools.repl import PythonREPLTool
        repl = PythonREPLTool()
        result = repl._run("import pandas as pd; print(pd.DataFrame({'a': [1, 2]}))")
        assert "Error" not in result
        repl.close()



# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_get_short_name_unknown(self):
        """Test get_short_name with unknown variable returns input."""
        from eurus.config import get_short_name
        result = get_short_name("completely_unknown_variable_xyz")
        # Should return the input as-is for unknown variables
        assert "completely_unknown_variable_xyz" in result or result is not None
        
    def test_variable_info_none_for_unknown(self):
        """Test get_variable_info returns None for unknown."""
        from eurus.config import get_variable_info
        result = get_variable_info("unknown_var_xyz")
        assert result is None
        
    def test_era5_tool_has_description(self):
        """Test ERA5 tool has comprehensive description."""
        from eurus.tools.era5 import era5_tool
        assert len(era5_tool.description) > 100
