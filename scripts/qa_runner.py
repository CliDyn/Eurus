#!/usr/bin/env python3
"""
QA Runner — Automated End-to-End Agent Testing
===============================================
Runs test queries through the Eurus agent, captures ALL intermediate steps
(tool calls, tool outputs, reasoning, plots) and saves structured results
to data/qa_results/q{NN}_{slug}/.

Usage:
    PYTHONPATH=src OPENAI_API_KEY=... python3 scripts/qa_runner.py
    
Or run a single query:
    PYTHONPATH=src OPENAI_API_KEY=... python3 scripts/qa_runner.py --query 2
"""

import os
import sys
import json
import shutil
import base64
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Ensure eurus package is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env (API keys)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from eurus.config import AGENT_SYSTEM_PROMPT, CONFIG, get_plots_dir
from eurus.tools import get_all_tools

# ============================================================================
# QA TEST QUERIES  — progressive difficulty
#
#  TIER 1  (Q01-Q07)  Simple single-variable retrieve → plot / stats
#  TIER 2  (Q08-Q14)  Derived quantities, multi-variable, comparisons
#  TIER 3  (Q15-Q20)  Real scientific analysis: indices, correlations,
#                      composite events, physical reasoning
# ============================================================================

QA_QUERIES = [
    # ═════════════════════════════════════════════════════════════════════
    #  TIER 1 — simple retrieval + visualisation
    # ═════════════════════════════════════════════════════════════════════
    {
        "id": 1,
        "slug": "sst_snapshot",
        "query": "Retrieve sea surface temperature (sst) for the Mediterranean Sea "
                 "(30-46N, -6 to 36E) on 2023-08-01 at 12:00 UTC. "
                 "Plot a spatial map in °C with a colorbar.",
        "type": "spatial_map",
        "variables": ["sst"],
        "region": "Mediterranean",
    },
    {
        "id": 2,
        "slug": "t2m_timeseries",
        "query": "Retrieve 2m temperature (t2) for Berlin (52-53N, 13-14E) "
                 "for all of January 2024. Plot the hourly time series in °C "
                 "and report the monthly mean, min, and max.",
        "type": "time_series",
        "variables": ["t2"],
        "region": "Berlin",
    },
    {
        "id": 3,
        "slug": "mslp_contour",
        "query": "Retrieve mean sea level pressure (mslp) for Europe "
                 "(35-72N, -10 to 40E) on 2024-01-10 at 00 UTC. "
                 "Create a contour map in hPa.",
        "type": "contour_map",
        "variables": ["mslp"],
        "region": "Europe",
    },
    {
        "id": 4,
        "slug": "precip_histogram",
        "query": "Retrieve total precipitation (tp) for India (8-35N, 68-90E) "
                 "on 2023-07-15. Plot a histogram of the precipitation values "
                 "in mm and report the 90th and 99th percentiles.",
        "type": "histogram",
        "variables": ["tp"],
        "region": "India",
    },
    {
        "id": 5,
        "slug": "cloud_cover_map",
        "query": "Get total cloud cover (tcc) for the North Atlantic "
                 "(30-65N, -60 to 0E) on 2023-12-01 at 12 UTC. "
                 "Plot a spatial map using a grayscale colormap (0 = clear, 1 = overcast).",
        "type": "spatial_map",
        "variables": ["tcc"],
        "region": "North Atlantic",
    },
    {
        "id": 6,
        "slug": "blh_diurnal",
        "query": "Retrieve boundary layer height (blh) for the Moscow region "
                 "(55-56N, 37-38E) for 2023-06-21. Plot the diurnal cycle "
                 "(area-mean BLH for each of the 24 hours).",
        "type": "diurnal_cycle",
        "variables": ["blh"],
        "region": "Moscow",
    },
    {
        "id": 7,
        "slug": "snow_depth_map",
        "query": "Get snow depth (sd) for Scandinavia (58-70N, 5-30E) "
                 "on 2024-01-15. Plot a spatial map in meters of water equivalent. "
                 "Report the area-average snow depth.",
        "type": "spatial_map",
        "variables": ["sd"],
        "region": "Scandinavia",
    },

    # ═════════════════════════════════════════════════════════════════════
    #  TIER 2 — derived quantities, multi-variable, comparisons
    # ═════════════════════════════════════════════════════════════════════
    {
        "id": 8,
        "slug": "wind_speed_derived",
        "query": "Get u10 and v10 for the North Sea (51-56N, 3-9E) on 2023-12-01. "
                 "Compute wind speed as sqrt(u10² + v10²). Plot the wind speed "
                 "spatial map in m/s, overlay wind direction arrows with quiver().",
        "type": "spatial_map",
        "variables": ["u10", "v10"],
        "region": "North Sea",
    },
    {
        "id": 9,
        "slug": "relative_humidity",
        "query": "Retrieve t2 and d2 for Florida (24-31N, -88 to -80W) "
                 "on 2023-07-15 at 15 UTC. Compute relative humidity using the "
                 "Magnus formula: RH = 100 * exp(17.625*Td/(243.04+Td)) / exp(17.625*T/(243.04+T)) "
                 "where T and Td are in °C. Plot the RH field as a % map.",
        "type": "spatial_map",
        "variables": ["t2", "d2"],
        "region": "Florida",
    },
    {
        "id": 10,
        "slug": "sst_anomaly_blacksea",
        "query": "Retrieve SST for the Black Sea (41-46N, 27-42E) on 2023-08-01 "
                 "AND on 2022-08-01. Compute the anomaly (SST_2023 − SST_2022). "
                 "Plot a diverging anomaly map (blue=cooler, red=warmer) in °C.",
        "type": "anomaly_map",
        "variables": ["sst"],
        "region": "Black Sea",
    },
    {
        "id": 11,
        "slug": "radiation_budget",
        "query": "Retrieve ssrd (incoming solar) and ssr (net solar) for the Sahara "
                 "(20-30N, 0-10E) on 2023-06-21. Compute surface albedo as "
                 "α = 1 − (ssr / ssrd). Plot a map of albedo (0–1) and report "
                 "the area-mean albedo. What physical processes explain the pattern?",
        "type": "derived_map",
        "variables": ["ssrd", "ssr"],
        "region": "Sahara",
    },
    {
        "id": 12,
        "slug": "precip_partitioning",
        "query": "Retrieve tp, cp, and lsp for Western Europe (43-55N, -5 to 15E) "
                 "on 2023-08-15. Compute the convective fraction = cp / tp. "
                 "Plot two subplots: (1) total precipitation in mm, "
                 "(2) convective fraction (0–1). Where was convection dominant?",
        "type": "multi_panel",
        "variables": ["tp", "cp", "lsp"],
        "region": "Western Europe",
    },
    {
        "id": 13,
        "slug": "wind_shear_10_100",
        "query": "Retrieve u10, v10 AND u100, v100 for the German Bight "
                 "(53-56N, 6-10E) on 2023-09-01. Compute wind speed at 10m and 100m. "
                 "Calculate the wind shear exponent α from the power law: "
                 "V100/V10 = (100/10)^α → α = log(V100/V10) / log(10). "
                 "Plot a map of the shear exponent and report the area mean.",
        "type": "derived_map",
        "variables": ["u10", "v10", "u100", "v100"],
        "region": "German Bight",
    },
    {
        "id": 14,
        "slug": "soil_moisture_temp",
        "query": "Retrieve soil moisture (swvl1) and soil temperature (stl1) for "
                 "central France (45-49N, 0-5E) on 2023-08-01 at 12 UTC. "
                 "Plot a scatter plot of soil temperature (°C, x-axis) vs soil "
                 "moisture (m³/m³, y-axis) for all grid points. "
                 "Compute the Pearson correlation. Is there a negative relationship?",
        "type": "scatter_correlation",
        "variables": ["swvl1", "stl1"],
        "region": "Central France",
    },

    # ═════════════════════════════════════════════════════════════════════
    #  TIER 3 — real scientific computations
    # ═════════════════════════════════════════════════════════════════════
    {
        "id": 15,
        "slug": "heat_index",
        "query": "Retrieve t2 and d2 for the US Southeast (25-36N, -95 to -75W) "
                 "on 2023-07-20 at 18 UTC. "
                 "1) Convert t2 to °F: F = C*9/5 + 32. "
                 "2) Compute RH (%) from the Magnus formula. "
                 "3) Compute the NWS Heat Index using the Rothfusz regression: "
                 "   HI = -42.379 + 2.04901523*T + 10.14333127*RH "
                 "        - 0.22475541*T*RH - 6.83783e-3*T² - 5.481717e-2*RH² "
                 "        + 1.22874e-3*T²*RH + 8.5282e-4*T*RH² - 1.99e-6*T²*RH² "
                 "   (T in °F, RH in %). "
                 "4) Convert back to °C and plot. Highlight cells > 40°C as 'extreme danger'.",
        "type": "scientific_index",
        "variables": ["t2", "d2"],
        "region": "US Southeast",
    },
    {
        "id": 16,
        "slug": "wind_power_density",
        "query": "Retrieve u100 and v100 for the Danish North Sea (54-58N, 3-9E) "
                 "for 2023-09-01 (full day, all hours). "
                 "1) Compute hourly hub-height wind speed V = sqrt(u100² + v100²). "
                 "2) Compute wind power density: WPD = 0.5 * 1.225 * V³  (W/m²). "
                 "3) Compute the daily-mean WPD at each grid point. "
                 "4) Estimate capacity factor assuming a 15 MW turbine with "
                 "   rated wind speed 12 m/s: CF = min(1, (V_mean/12)³). "
                 "5) Plot two panels: (a) mean WPD (W/m²), (b) capacity factor (0–1).",
        "type": "scientific_analysis",
        "variables": ["u100", "v100"],
        "region": "Danish North Sea",
    },
    {
        "id": 17,
        "slug": "bowen_ratio",
        "query": "Retrieve ssr (net solar), ssrd (incoming solar), t2, and d2 for "
                 "the Iberian Peninsula (36-44N, -10 to 3E) on 2023-08-01. "
                 "1) Compute net radiation Q* ≈ ssr (J/m²). "
                 "2) Estimate latent heat flux using the Priestley-Taylor "
                 "   approach: compute the slope of the saturation vapour "
                 "   pressure curve s = 4098 * (0.6108 * exp(17.27*T/(T+237.3))) "
                 "   / (T+237.3)² (T in °C), then LE ≈ 1.26 * s/(s+0.067) * Q*. "
                 "3) Sensible heat H ≈ Q* − LE. "
                 "4) Compute Bowen ratio β = H / LE. "
                 "5) Plot the Bowen ratio map. Values > 5 indicate arid conditions.",
        "type": "energy_balance",
        "variables": ["ssr", "ssrd", "t2", "d2"],
        "region": "Iberian Peninsula",
    },
    {
        "id": 18,
        "slug": "thermal_front_detection",
        "query": "Retrieve SST for the Gulf Stream region (30-45N, -80 to -50W) "
                 "on 2024-01-15 at 12 UTC. "
                 "1) Compute the spatial SST gradient magnitude: "
                 "   |∇SST| = sqrt((dSST/dx)² + (dSST/dy)²) using np.gradient. "
                 "2) Convert from K/gridcell to °C/100km (grid spacing ≈ 0.25° ≈ 28 km). "
                 "3) Plot the SST field, then overlay contours of |∇SST| > 1°C/100km "
                 "   to highlight thermal fronts. "
                 "4) Report maximum gradient value and location — this is the Gulf Stream front.",
        "type": "gradient_analysis",
        "variables": ["sst"],
        "region": "Gulf Stream",
    },
    {
        "id": 19,
        "slug": "thunderstorm_composite",
        "query": "Retrieve CAPE, total column water vapour (tcwv), and wind shear "
                 "(u10, v10, u100, v100) for the US Great Plains (30-45N, -105 to -90W) "
                 "on 2023-05-15 at 18 UTC. "
                 "1) Compute 0–100m bulk wind shear: "
                 "   ΔV = sqrt((u100-u10)² + (v100-v10)²). "
                 "2) Define a Composite Severe Weather Index: "
                 "   SWI = (CAPE/1000) * (tcwv/30) * (ΔV/10). "
                 "   Each factor is normalised so that 1.0 ≈ moderate threat. "
                 "3) Plot: (a) CAPE (J/kg), (b) tcwv (kg/m²), (c) shear (m/s), "
                 "   (d) SWI. Highlight SWI > 2 as 'high risk' in panel (d). "
                 "4) Report the area fraction where SWI > 2.",
        "type": "composite_extreme",
        "variables": ["cape", "tcwv", "u10", "v10", "u100", "v100"],
        "region": "US Great Plains",
    },
    {
        "id": 20,
        "slug": "urban_heat_island",
        "query": "Retrieve skin temperature (skt) and 2m temperature (t2) for "
                 "greater Paris (48-49.5N, 1.5-3.5E) on 2023-08-24 at 00 UTC (nighttime). "
                 "Note: use .squeeze() to remove any singleton dimensions before computing. "
                 "1) Compute the surface-air temperature difference: ΔT = skt - t2 (°C). "
                 "2) Compute spatial statistics: mean, std, and the 95th percentile of ΔT. "
                 "3) Plot the ΔT map with a diverging colormap. "
                 "4) The urban heat island appears as a warm ΔT anomaly over the city. "
                 "   Identify the grid cell with the maximum ΔT and report its "
                 "   coordinates and value. Discuss the physical mechanism.",
        "type": "uhi_analysis",
        "variables": ["skt", "t2"],
        "region": "Paris",
    },
]


# ============================================================================
# AGENT SETUP  (mirrors main.py exactly)
# ============================================================================

def build_agent():
    """Build a LangChain agent with full tool suite."""
    llm = ChatOpenAI(
        model=CONFIG.model_name,
        temperature=CONFIG.temperature,
    )
    
    tools = get_all_tools(enable_routing=False, enable_guide=True)
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        debug=False,
    )
    
    return agent


# ============================================================================
# STEP CAPTURE
# ============================================================================

def extract_steps(messages) -> list:
    """
    Extract ALL intermediate steps from agent message history.
    Returns list of step dicts with type, content, tool_name, etc.
    """
    steps = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            steps.append({
                "step": len(steps) + 1,
                "type": "user_query",
                "content": msg.content[:2000],
            })
        elif isinstance(msg, AIMessage):
            # AI thinking / tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    # Capture tool call request
                    args = tc.get("args", {})
                    # Truncate large args
                    args_str = json.dumps(args, indent=2, default=str)
                    if len(args_str) > 5000:
                        args_str = args_str[:5000] + "\n... [TRUNCATED]"
                    
                    steps.append({
                        "step": len(steps) + 1,
                        "type": "tool_call",
                        "tool_name": tc.get("name", "unknown"),
                        "tool_id": tc.get("id", ""),
                        "arguments": json.loads(args_str) if len(args_str) <= 5000 else args_str,
                        "reasoning": msg.content[:1000] if msg.content else "",
                    })
            elif msg.content:
                # Final response or intermediate reasoning
                steps.append({
                    "step": len(steps) + 1,
                    "type": "ai_response",
                    "content": msg.content[:5000],
                })
        elif isinstance(msg, ToolMessage):
            # Tool output
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 3000:
                content = content[:3000] + "\n... [TRUNCATED]"
            
            steps.append({
                "step": len(steps) + 1,
                "type": "tool_output",
                "tool_name": msg.name if hasattr(msg, 'name') else "unknown",
                "tool_call_id": msg.tool_call_id if hasattr(msg, 'tool_call_id') else "",
                "content": content,
            })
    
    return steps


# ============================================================================
# QA RUNNER
# ============================================================================

def run_single_query(agent, query_def: dict, output_dir: Path) -> dict:
    """
    Run a single QA query and capture everything.
    
    Returns: metadata dict
    """
    qid = query_def["id"]
    slug = query_def["slug"]
    query = query_def["query"]
    
    folder = output_dir / f"q{qid:02d}_{slug}"
    folder.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  Q{qid:02d}: {query[:70]}...")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Snapshot existing plots BEFORE running so we only copy NEW ones
        plots_dir = get_plots_dir()
        existing_plots = set()
        if plots_dir.exists():
            existing_plots = {f.name for f in plots_dir.glob("*.png")}
        
        # Invoke agent
        config = {"recursion_limit": 80}
        messages = [HumanMessage(content=query)]
        
        result = agent.invoke({"messages": messages}, config=config)
        
        elapsed = time.time() - start_time
        result_messages = result["messages"]
        
        # Extract intermediate steps
        steps = extract_steps(result_messages)
        
        # Get final response
        final_response = ""
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_response = msg.content
                break
        
        # Save steps.json
        steps_path = folder / "steps.json"
        with open(steps_path, "w") as f:
            json.dump(steps, f, indent=2, default=str, ensure_ascii=False)
        
        # Save final response
        response_path = folder / "response.md"
        with open(response_path, "w") as f:
            f.write(f"# Q{qid:02d}: {slug}\n\n")
            f.write(f"**Query:** {query}\n\n")
            f.write(f"**Elapsed:** {elapsed:.1f}s\n\n")
            f.write("---\n\n")
            f.write(final_response)
        
        # Copy only NEW plots (diff against pre-query snapshot)
        plot_files = []
        if plots_dir.exists():
            for f_path in sorted(plots_dir.glob("*.png")):
                if f_path.name not in existing_plots:
                    dest = folder / f_path.name
                    shutil.copy2(f_path, dest)
                    plot_files.append(f_path.name)
                    print(f"   📊 Plot saved: {f_path.name}")
        
        # Count tool calls
        tool_calls = [s for s in steps if s["type"] == "tool_call"]
        tools_used = list(set(s["tool_name"] for s in tool_calls))
        
        # Build metadata
        metadata = {
            "query_id": qid,
            "slug": slug,
            "query": query,
            "type": query_def.get("type", "unknown"),
            "variables": query_def.get("variables", []),
            "region": query_def.get("region", ""),
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "status": "success",
            "tools_used": tools_used,
            "num_tool_calls": len(tool_calls),
            "num_steps": len(steps),
            "plot_files": plot_files,
            "notes": "",
        }
        
        # Save metadata.json
        meta_path = folder / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ SUCCESS in {elapsed:.1f}s | Tools: {', '.join(tools_used)} | Steps: {len(steps)}")
        
        return metadata
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ FAILED in {elapsed:.1f}s: {e}")
        
        metadata = {
            "query_id": qid,
            "slug": slug,
            "query": query,
            "type": query_def.get("type", "unknown"),
            "variables": query_def.get("variables", []),
            "region": query_def.get("region", ""),
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "status": "error",
            "error": str(e),
            "tools_used": [],
            "num_tool_calls": 0,
            "num_steps": 0,
            "plot_files": [],
            "notes": f"Error: {e}",
        }
        
        meta_path = folder / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata


def main():
    parser = argparse.ArgumentParser(description="Eurus QA Runner")
    parser.add_argument("--query", type=int, help="Run a single query by ID (1-10)")
    parser.add_argument("--start", type=int, default=1, help="Start from query ID")
    parser.add_argument("--end", type=int, default=10, help="End at query ID (inclusive)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if folder already has metadata.json")
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set!")
        sys.exit(1)
    
    output_dir = PROJECT_ROOT / "data" / "qa_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║          Eurus QA Runner v1.0                       ║
║          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                        ║
╚══════════════════════════════════════════════════════╝
Output: {output_dir}
""")
    
    # Build agent once
    print("🏗️  Building agent...")
    agent = build_agent()
    print("✅ Agent ready\n")
    
    # Select queries
    if args.query:
        queries = [q for q in QA_QUERIES if q["id"] == args.query]
    else:
        queries = [q for q in QA_QUERIES if args.start <= q["id"] <= args.end]
    
    results = []
    for q in queries:
        folder = output_dir / f"q{q['id']:02d}_{q['slug']}"
        if args.skip_existing and (folder / "metadata.json").exists():
            print(f"⏭️  Skipping Q{q['id']:02d} (already exists)")
            continue
        
        result = run_single_query(agent, q, output_dir)
        results.append(result)
    
    # Print summary
    print(f"\n{'='*70}")
    print("QA SUMMARY")
    print(f"{'='*70}")
    
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    total_time = sum(r["elapsed_seconds"] for r in results)
    
    for r in results:
        status = "✅" if r["status"] == "success" else "❌"
        print(f"  {status} Q{r['query_id']:02d} ({r['slug']:20s}) | "
              f"{r['elapsed_seconds']:5.1f}s | Tools: {', '.join(r['tools_used'])}")
    
    print(f"\nTotal: {success} passed, {failed} failed, {total_time:.1f}s total")
    
    # Save summary
    summary_path = output_dir / "qa_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
            "passed": success,
            "failed": failed,
            "total_time_seconds": round(total_time, 1),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
