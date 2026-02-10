import pytest
from eurus.config import ERA5_VARIABLES, VARIABLE_ALIASES, get_variable_info, get_short_name


# All 22 variables in the Arraylake dataset
ALL_ARRAYLAKE_VARS = [
    "blh", "cape", "cp", "d2", "lsp", "mslp", "sd", "skt", "sp",
    "ssr", "ssrd", "sst", "stl1", "swvl1", "t2", "tcc", "tcw",
    "tcwv", "u10", "u100", "v10", "v100",
]

# tp is a derived/accumulated variable kept for convenience
ALL_CATALOG_VARS = sorted(ALL_ARRAYLAKE_VARS + ["tp"])


def test_variable_catalog_has_all_22():
    """Every Arraylake variable must appear in ERA5_VARIABLES."""
    for var in ALL_ARRAYLAKE_VARS:
        assert var in ERA5_VARIABLES, f"Missing variable: {var}"


def test_total_variable_count():
    """Catalog should contain at least 22 variables (22 Arraylake + tp)."""
    assert len(ERA5_VARIABLES) >= 22


def test_variable_loading():
    """Test that ERA5 variables are loaded correctly."""
    assert "sst" in ERA5_VARIABLES
    assert "t2" in ERA5_VARIABLES
    assert "u10" in ERA5_VARIABLES

    sst_info = ERA5_VARIABLES["sst"]
    assert sst_info.units == "K"
    assert sst_info.short_name == "sst"


def test_new_variables_metadata():
    """Spot-check metadata on newly added variables."""
    # Boundary layer height
    blh = ERA5_VARIABLES["blh"]
    assert blh.units == "m"
    assert blh.category == "atmosphere"

    # Dewpoint
    d2 = ERA5_VARIABLES["d2"]
    assert d2.units == "K"

    # Soil moisture
    swvl1 = ERA5_VARIABLES["swvl1"]
    assert "m³/m³" in swvl1.units
    assert swvl1.category == "land_surface"

    # 100m wind
    u100 = ERA5_VARIABLES["u100"]
    assert u100.units == "m/s"

    # Radiation
    ssrd = ERA5_VARIABLES["ssrd"]
    assert "J/m²" in ssrd.units
    assert ssrd.category == "radiation"


def test_get_variable_info():
    """Test helper function for retrieving variable info."""
    # Test case insensitive
    assert get_variable_info("SST") == ERA5_VARIABLES["sst"]
    assert get_variable_info("Sea_Surface_Temperature") == ERA5_VARIABLES["sst"]
    assert get_variable_info("non_existent_var") is None

    # Test new aliases
    assert get_variable_info("dewpoint") == ERA5_VARIABLES["d2"]
    assert get_variable_info("soil_moisture") == ERA5_VARIABLES["swvl1"]
    assert get_variable_info("boundary_layer_height") == ERA5_VARIABLES["blh"]
    assert get_variable_info("snow_depth") == ERA5_VARIABLES["sd"]


def test_get_short_name():
    """Test retrieval of short names."""
    assert get_short_name("SST") == "sst"
    assert get_short_name("Sea_Surface_Temperature") == "sst"
    # Fallback to lower case input
    assert get_short_name("UNKNOWN_VAR") == "unknown_var"

    # New aliases
    assert get_short_name("skin_temperature") == "skt"
    assert get_short_name("100m_u_component_of_wind") == "u100"
    assert get_short_name("total_column_water_vapour") == "tcwv"


def test_agent_prompt_branding():
    """Test that the system prompt contains the Eurus branding."""
    from eurus.config import AGENT_SYSTEM_PROMPT
    assert "Eurus" in AGENT_SYSTEM_PROMPT
    assert "Comrade Copernicus" not in AGENT_SYSTEM_PROMPT
    assert "PANGAEA" not in AGENT_SYSTEM_PROMPT


def test_agent_prompt_lists_all_variables():
    """System prompt should mention all 22 Arraylake variable short names."""
    from eurus.config import AGENT_SYSTEM_PROMPT
    for var in ALL_ARRAYLAKE_VARS:
        assert var in AGENT_SYSTEM_PROMPT, (
            f"System prompt missing variable: {var}"
        )
