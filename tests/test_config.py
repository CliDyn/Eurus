import pytest
from eurus.config import (
    ERA5_VARIABLES,
    VARIABLE_ALIASES,
    get_variable_info,
    get_short_name,
    get_zarr_name,
)


# The 38 array names actually present in the "single" group of the
# earthmover-public/era5 Icechunk store. Catalog short names map onto these
# via ERA5Variable.zarr_name; most are identity, a few are renamed.
STORE_ARRAY_NAMES = {
    "blh", "cape", "cp", "d2m", "fdir", "fg10", "fsr", "hcc", "ie", "lcc",
    "lsp", "mcc", "msl", "sd", "sf", "skt", "slhf", "sp", "ssr", "ssrd",
    "sst", "stl1", "stl2", "stl3", "stl4", "swvl1", "t2m", "tcc", "tcw",
    "tcwv", "tisr", "tp", "tsr", "u10", "u100", "v10", "v100", "zust",
}

# Catalog short names whose store array name differs. These are the renames
# that would silently break retrieval if zarr_name were wrong.
RENAMED_VARS = {"t2": "t2m", "d2": "d2m", "mslp": "msl"}

ALL_CATALOG_VARS = sorted(ERA5_VARIABLES)


def test_catalog_covers_every_store_array():
    """Every array in the store must be reachable from some catalog entry."""
    reachable = {v.zarr_name or v.short_name for v in ERA5_VARIABLES.values()}
    assert reachable == STORE_ARRAY_NAMES


def test_total_variable_count():
    """Catalog should expose all 38 single-level store variables."""
    assert len(ERA5_VARIABLES) == len(STORE_ARRAY_NAMES) == 38


@pytest.mark.parametrize("short_name,expected_zarr", sorted(RENAMED_VARS.items()))
def test_renamed_variables_resolve_to_store_names(short_name, expected_zarr):
    """Renamed vars must keep their catalog name but index the store correctly."""
    assert get_short_name(short_name) == short_name
    assert get_zarr_name(short_name) == expected_zarr


@pytest.mark.parametrize("short_name", ALL_CATALOG_VARS)
def test_zarr_name_exists_in_store(short_name):
    """get_zarr_name must always return a real array name in the store."""
    assert get_zarr_name(short_name) in STORE_ARRAY_NAMES


@pytest.mark.parametrize("short_name", ALL_CATALOG_VARS)
def test_zarr_name_only_set_when_it_differs(short_name):
    """Don't carry a redundant zarr_name equal to the short name."""
    var = ERA5_VARIABLES[short_name]
    assert var.zarr_name != var.short_name, (
        f"{short_name}: zarr_name is redundant, leave it as None"
    )


def test_zarr_name_defaults_to_short_name():
    """Variables without a rename index the store under their short name."""
    assert get_zarr_name("sst") == "sst"
    assert get_zarr_name("u10") == "u10"
    assert get_zarr_name("tp") == "tp"


def test_zarr_name_resolves_through_aliases():
    """Aliases must resolve all the way to the store array name."""
    assert get_zarr_name("2m_temperature") == "t2m"
    assert get_zarr_name("dewpoint") == "d2m"
    assert get_zarr_name("mean_sea_level_pressure") == "msl"


def test_zarr_name_passes_through_unknown_variables():
    """Unknown names fall through unchanged rather than raising."""
    assert get_zarr_name("not_a_real_var") == "not_a_real_var"


@pytest.mark.parametrize("short_name", ALL_CATALOG_VARS)
def test_diverging_colormap_only_on_signed_variables(short_name):
    """RdBu_r centres on zero, so it's wrong for one-sided quantities.

    Wind components and fluxes are signed and diverge meaningfully; gust
    speed, roughness and radiation totals are non-negative and must not.
    """
    var = ERA5_VARIABLES[short_name]
    if var.colormap != "RdBu_r":
        return
    low, high = var.typical_range
    assert low is not None and high is not None, f"{short_name}: RdBu_r needs a range"
    assert low < 0 < high, (
        f"{short_name}: diverging RdBu_r but typical_range {var.typical_range} "
        f"does not straddle zero — use a sequential colormap"
    )


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


@pytest.mark.parametrize("var", ALL_CATALOG_VARS)
def test_agent_prompt_lists_all_variables(var):
    """System prompt's variable table should have a row for every catalog entry.

    Matches the table row rather than a bare substring — short names like
    "sd", "sp" and "ie" occur inside ordinary prompt prose.
    """
    from eurus.config import AGENT_SYSTEM_PROMPT
    assert f"| {var} |" in AGENT_SYSTEM_PROMPT, (
        f"System prompt variable table missing row for: {var}"
        )
