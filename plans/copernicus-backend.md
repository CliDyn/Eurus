# Copernicus Marine Backend

## Goal

Add a second data retrieval backend using the [Copernicus Marine Toolbox](https://toolbox-docs.marine.copernicus.eu/en/stable/)
(`copernicusmarine` pip package). This gives the agent access to CMEMS ocean
reanalysis products (GLORYS12, DUACS altimetry, wave models, etc.) that are
not in the Earthmover/ERA5 archive.

_feedback_:

---

## How the toolbox works

```python
import copernicusmarine as cm

ds = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["thetao", "so"],
    minimum_longitude=-30.0,
    maximum_longitude=10.0,
    minimum_latitude=20.0,
    maximum_latitude=60.0,
    start_datetime="2020-01-01",
    end_datetime="2020-12-31",
    minimum_depth=0.5,
    maximum_depth=0.5,   # surface only
)
# ds is a lazy xarray.Dataset — call .load() then .to_zarr() to cache
```

**Credentials** are resolved in order:
1. Env vars `COPERNICUSMARINE_SERVICE_USERNAME` / `COPERNICUSMARINE_SERVICE_PASSWORD`
2. Credentials file at `~/.copernicusmarine` (created by `cm.login()`)
3. Explicit `username=` / `password=` params

_feedback_:

For now, just assume the user ran copernicusmarine login on their shell once.

In the example aboce, note that chunking (set via service=) kwarg is essential to get problem-adapted chunking.

---

## Proposed hard-coded catalog

The `dataset_id` is the only tricky part — we map user-friendly variable names
to the right CMEMS product. For now, hard-code the catalog; `cm.describe()` can
be added later for dynamic discovery.

| User variable | `dataset_id` | Internal var | Has depth | Notes |
|---|---|---|---|---|
| `thetao` | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | `thetao` | yes | GLORYS12 reanalysis |
| `so` | same | `so` | yes | Salinity |
| `uo` | same | `uo` | yes | Eastward current |
| `vo` | same | `vo` | yes | Northward current |
| `zos` | same | `zos` | no | Sea surface height |
| `mlotst` | same | `mlotst` | no | Mixed layer depth |
| `siconc` | same | `siconc` | no | Sea ice concentration |
| `sithick` | same | `sithick` | no | Sea ice thickness |
| `bottomT` | same | `bottomT` | no | Sea floor temperature |
| `sla` | `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D` | `sla` | no | Altimetry SLA |
| `adt` | same | `adt` | no | Absolute dynamic topo |
| `ugos` | same | `ugos` | no | Geostrophic current E |
| `vgos` | same | `vgos` | no | Geostrophic current N |
| `VHM0` | `cmems_mod_glo_wav_my_0.2deg_PT3H-i` | `VHM0` | no | Significant wave height |
| `VMDR` | same | `VMDR` | no | Mean wave direction |
| `VTM10` | same | `VTM10` | no | Mean wave period |

GLORYS12 covers 1993–present (with a ~1 month interim lag). DUACS covers
1993–present. WAVERYS covers 1993–present.

_feedback_:

Good start. I just want to get an MVP up quickly.

---

## Files to create / modify

### New files

**`src/eurus/backends/__init__.py`** — empty package marker

**`src/eurus/backends/copernicus.py`** — core retrieval:
- `COPERNICUS_CATALOG` dataclass + dict (table above)
- `generate_filename()` — `cmems_{var}_{start}_{end}_{region_tag}.zarr`
- `retrieve_copernicus_data(variable_id, start_date, end_date, min/max lat/lon, depth_m, region)` → `str`
  - Checks creds (`COPERNICUSMARINE_SERVICE_USERNAME` present)
  - Looks up `variable_id` in catalog → gets `dataset_id`, internal var name
  - Checks local Zarr cache (same pattern as ERA5)
  - Calls `cm.open_dataset(...)`, selects nearest depth if `has_depth=True`
  - Validates non-empty, checks size guard
  - Saves to Zarr, registers in `MemoryManager`, returns success string

**`src/eurus/tools/copernicus.py`** — LangChain StructuredTool:
- `CopernicusRetrievalArgs` (Pydantic schema): `variable_id`, `start_date`, `end_date`, `min/max lat/lon`, `depth_m` (optional, default surface), `region` (optional)
- Thin wrapper calling `retrieve_copernicus_data`
- `copernicus_tool = StructuredTool.from_function(...)`

### Modified files

**`src/eurus/tools/__init__.py`**
- Import `copernicus_tool`
- Add it to `get_all_tools()` list (always enabled; graceful ImportError if `copernicusmarine` not installed)

**`pyproject.toml`**
- Add optional dep group `[project.optional-dependencies]` entry `copernicus = ["copernicusmarine>=1.0.0"]`

**`src/eurus/config.py`** (minimal)
- Possibly add `COPERNICUSMARINE_SERVICE_USERNAME` / `PASSWORD` to documented env vars (comment only)

_feedback_:

is backend submod really the way to go. How is era5 currently wired? And hos is arraylake wired? Let's look at structure of this first. --> Create some sequence diagram for what happens when a user prompts which highlights how credentials and arraylake are used? And let's create a flow chart or class diag or entity rel diag (whatever works best) for understanding modules? (mermaid.) Then we decide.

---

## Open questions / things to discuss

1. **Depth handling** — default to surface (nearest to 0.5 m)? Or always require the user / agent to specify? Exposing `depth_m` as an optional tool param seems right; default `None` → surface.

_feedback_:

How's the LLM part wired to understand things like "surface temperature", "bottom temperature", "mixed-layer average oxygen" etc. ?

2. **Interim dataset fallback** — GLORYS12 has two IDs: `my` (multi-year, stable reanalysis) and `myint` (interim, near-present, updated weekly). Should we automatically try `myint` if `my` returns empty data for recent dates? Or just document that `myint` is the variant to use for recent data and let the agent pick?

_feedback_:

Long term: Wire them together. They extend each other. For now, use my for first tests / MVP.

3. **Variable aliases** — Add Copernicus vars to the existing `VARIABLE_ALIASES` in `config.py`, or keep a separate alias table in the backend? Separate is cleaner since they're a different namespace.

_feedback_:

Tell me how this is done for ERA currently first.

4. **System prompt** — The `AGENT_SYSTEM_PROMPT` in `config.py` currently only mentions ERA5. We'll need to add a Copernicus section documenting available variables, the `retrieve_copernicus_data` tool, and the `depth_m` param. Defer until after wiring works?

_feedback_:

Adapt soon so we can test wiring in a real session. Note I don't want a good final desing of the internals. Quik / dirty MVP to see _that_ this works is fine.

5. **Credential error message** — Should we surface a helpful URL to the Copernicus registration page when the creds env vars are missing?

_feedback_:

Just fail. See above.