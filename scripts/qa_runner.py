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
# QA TEST QUERIES — 36 research-grade demo queries
#
#  §1  Synoptic Meteorology & Case Studies         (Q01–Q05)
#  §2  Climate Variability & Teleconnections        (Q06–Q10)
#  §3  Trends & Climate Change Signals              (Q11–Q15)
#  §4  Extreme Events & Risk                        (Q16–Q20)
#  §5  Maritime & Shipping                          (Q21–Q24)
#  §6  Energy Assessment                            (Q25–Q28)
#  §7  Diurnal & Sub-Daily Processes                (Q29–Q30)
#  §8  Multi-Variable & Diagnostics                 (Q31–Q33)
#  §9  Quick Lookups                                (Q34–Q36)
# ============================================================================

QA_QUERIES = [
    # ═══════════════════════════════════════════════════════════════
    #  §1 — Synoptic Meteorology & Case Studies
    # ═══════════════════════════════════════════════════════════════
    {
        "id": 1,
        "slug": "europe_heatwave_anomaly",
        "query": "Show me a spatial map of 2m temperature anomalies across Europe "
                 "during the June 2023 heatwave compared to June 2022.",
        "type": "anomaly_map",
        "variables": ["t2"],
        "region": "Europe",
    },
    {
        "id": 2,
        "slug": "storm_isha_mslp_wind",
        "query": "Plot MSLP isobars and 10m wind vectors over the North Atlantic "
                 "for 2024-01-22 — I want to see Storm Isha's structure.",
        "type": "contour_quiver",
        "variables": ["mslp", "u10", "v10"],
        "region": "North Atlantic",
    },
    {
        "id": 3,
        "slug": "atmospheric_river_jan2023",
        "query": "Download total column water vapour for the US West Coast, Jan 2023, "
                 "and show the atmospheric river event around Jan 9th.",
        "type": "ar_detection",
        "variables": ["tcwv"],
        "region": "US West Coast",
    },
    {
        "id": 4,
        "slug": "sahara_heat_july2024",
        "query": "Plot the daily mean 2m temperature time series averaged over "
                 "the Sahara (20-30°N, 0 to 15°E) for July 2024 and compare "
                 "it to July 2023 on the same chart.",
        "type": "time_series",
        "variables": ["t2"],
        "region": "Sahara",
    },
    {
        "id": 5,
        "slug": "great_plains_wind_may2024",
        "query": "Plot a map of mean 10m wind speed over the US Great Plains "
                 "(30-45°N, -105 to -90°W) for May 2024 and highlight areas exceeding 5 m/s.",
        "type": "threshold_map",
        "variables": ["u10", "v10"],
        "region": "US Great Plains",
    },

    # ═══════════════════════════════════════════════════════════════
    #  §2 — Climate Variability & Teleconnections
    # ═══════════════════════════════════════════════════════════════
    {
        "id": 6,
        "slug": "nino34_index",
        "query": "Calculate the Niño 3.4 index from ERA5 SST for 2015-2024 and "
                 "classify El Niño / La Niña episodes.",
        "type": "climate_index",
        "variables": ["sst"],
        "region": "Tropical Pacific",
    },
    {
        "id": 7,
        "slug": "elnino_vs_lanina_tropical_belt",
        "query": "Compare SST anomalies across the entire tropical belt "
                 "(30°S-30°N, global) for December 2023 (peak El Niño) vs December 2022 "
                 "(La Niña). Show the full basin-wide pattern across the Pacific, "
                 "Atlantic, and Indian oceans in a single anomaly difference map.",
        "type": "anomaly_comparison",
        "variables": ["sst"],
        "region": "Tropical Belt (global)",
    },
    {
        "id": 8,
        "slug": "nao_index",
        "query": "Compute the NAO index from MSLP (Azores minus Iceland) for 2000-2024 "
                 "and plot it with a 3-month rolling mean.",
        "type": "climate_index",
        "variables": ["mslp"],
        "region": "North Atlantic",
    },
    {
        "id": 9,
        "slug": "australia_enso_rainfall",
        "query": "Compare precipitation over Eastern Australia (25-45°S, 145-155°E) "
                 "between the La Niña year 2022 and El Niño year 2023. "
                 "Show a two-panel map of annual total precipitation for each year "
                 "and a difference map (2023 minus 2022).",
        "type": "multi_year_anomaly",
        "variables": ["tp"],
        "region": "Australia",
    },
    {
        "id": 10,
        "slug": "med_eof_sst",
        "query": "Perform an EOF analysis on Mediterranean SST anomalies "
                 "(30-46°N, -6 to 36°E) for 2019-2024 and show the first 3 modes "
                 "with variance explained. Interpret the dominant patterns.",
        "type": "eof_analysis",
        "variables": ["sst"],
        "region": "Mediterranean",
    },

    # ═══════════════════════════════════════════════════════════════
    #  §3 — Trends & Climate Change Signals
    # ═══════════════════════════════════════════════════════════════
    {
        "id": 11,
        "slug": "arctic_polar_amplification",
        "query": "Compare January mean 2m temperature across the entire Arctic "
                 "(north of 70°N) for 2024 vs 2000. Show both maps side by side, "
                 "compute the area-weighted temperature difference, and quantify "
                 "polar amplification.",
        "type": "decadal_comparison",
        "variables": ["t2"],
        "region": "Arctic (>70°N)",
    },
    {
        "id": 12,
        "slug": "med_marine_heatwave_2023",
        "query": "Map the summer (JJA) 2023 mean SST anomaly across the entire "
                 "Mediterranean basin (30-46°N, -6 to 36°E) compared to the 2018-2022 "
                 "summer mean. Identify marine heatwave hotspots where SST exceeded "
                 "+2°C above normal.",
        "type": "marine_heatwave",
        "variables": ["sst"],
        "region": "Mediterranean",
    },
    {
        "id": 13,
        "slug": "paris_decadal_comparison",
        "query": "Compare the average summer (JJA) temperature in Paris between the "
                 "decades 2000-2009 and 2014-2023 — show a difference map and time series.",
        "type": "multi_panel_comparison",
        "variables": ["t2"],
        "region": "Paris",
    },
    {
        "id": 14,
        "slug": "alps_snow_trend",
        "query": "Has the snow depth over the Alps decreased over the last 30 years? "
                 "Show me the December-February trend.",
        "type": "trend_analysis",
        "variables": ["sd"],
        "region": "Alps",
    },
    {
        "id": 15,
        "slug": "uk_precip_anomaly_winter2024",
        "query": "Map the total precipitation anomaly over the British Isles "
                 "(49-60°N, 11°W-2°E) for January 2024 compared to the 2019-2023 "
                 "January mean. Highlight regions receiving more than 150% of normal "
                 "rainfall. Save the map as a PNG file.",
        "type": "anomaly_map",
        "variables": ["tp"],
        "region": "British Isles",
    },

    # ═══════════════════════════════════════════════════════════════
    #  §4 — Extreme Events & Risk
    # ═══════════════════════════════════════════════════════════════
    {
        "id": 16,
        "slug": "delhi_heatwave_detection",
        "query": "Detect heatwave events in Delhi from 2010-2024 using the 90th "
                 "percentile threshold with a 3-day duration criterion — how has the "
                 "frequency changed?",
        "type": "heatwave_detection",
        "variables": ["t2"],
        "region": "Delhi",
    },
    {
        "id": 17,
        "slug": "horn_africa_drought",
        "query": "Calculate a 3-month SPI proxy for the Horn of Africa "
                 "(Ethiopia/Somalia) for 2020-2024 — when were the worst drought periods?",
        "type": "drought_analysis",
        "variables": ["tp"],
        "region": "Horn of Africa",
    },
    {
        "id": 18,
        "slug": "baghdad_hot_days",
        "query": "How many days per year exceeded 35°C in Baghdad from 1980 to 2024? "
                 "Plot as a bar chart with a trend line.",
        "type": "exceedance_frequency",
        "variables": ["t2"],
        "region": "Baghdad",
    },
    {
        "id": 19,
        "slug": "sea_p95_precip",
        "query": "Show me the 95th percentile daily precipitation map for Southeast Asia "
                 "for 2010-2023.",
        "type": "extreme_percentile",
        "variables": ["tp"],
        "region": "Southeast Asia",
    },
    {
        "id": 20,
        "slug": "scandinavia_blocking_2018",
        "query": "Analyse the blocking event over Scandinavia in July 2018 — show MSLP "
                 "anomalies persisting for 5+ days.",
        "type": "blocking_detection",
        "variables": ["mslp"],
        "region": "Scandinavia",
    },

    # ═══════════════════════════════════════════════════════════════
    #  §5 — Maritime & Shipping
    # ═══════════════════════════════════════════════════════════════
    {
        "id": 21,
        "slug": "rotterdam_shanghai_route",
        "query": "Calculate the maritime route from Rotterdam to Shanghai and analyse "
                 "wind risk along the route for December.",
        "type": "maritime_route_risk",
        "variables": ["u10", "v10"],
        "region": "Europe-Asia",
    },
    {
        "id": 22,
        "slug": "indian_ocean_sst_dipole",
        "query": "Map the SST anomaly across the Indian Ocean (30°S-25°N, 30-120°E) "
                 "for October 2023 relative to the 2019-2022 October mean. "
                 "Show the Indian Ocean Dipole pattern. Save the map as PNG.",
        "type": "anomaly_map",
        "variables": ["sst"],
        "region": "Indian Ocean",
    },
    {
        "id": 23,
        "slug": "japan_typhoon_season_wind",
        "query": "Map the mean and maximum 10m wind speed over the seas around Japan "
                 "(20-45°N, 120-150°E) during typhoon season (August-October) 2023. "
                 "Show two-panel spatial maps highlighting areas where mean wind "
                 "exceeded 8 m/s. Save as PNG.",
        "type": "multi_panel_map",
        "variables": ["u10", "v10"],
        "region": "Japan",
    },
    {
        "id": 24,
        "slug": "south_atlantic_sst_gradient",
        "query": "Map the mean SST field across the South Atlantic (40°S-5°N, 50°W-15°E) "
                 "for March 2024. Overlay SST isotherms and highlight the "
                 "Brazil-Malvinas confluence zone. Save as PNG.",
        "type": "sst_map",
        "variables": ["sst"],
        "region": "South Atlantic",
    },

    # ═══════════════════════════════════════════════════════════════
    #  §6 — Energy Assessment
    # ═══════════════════════════════════════════════════════════════
    {
        "id": 25,
        "slug": "north_sea_wind_power",
        "query": "Map the mean 100m wind power density across the North Sea for "
                 "2020-2024 — where are the best offshore wind sites?",
        "type": "wind_energy",
        "variables": ["u100", "v100"],
        "region": "North Sea",
    },
    {
        "id": 26,
        "slug": "german_bight_weibull",
        "query": "Fit a Weibull distribution to 100m wind speed at 54°N, 7°E "
                 "(German Bight) for 2023 and estimate the capacity factor for a "
                 "3-25 m/s turbine range. Plot the histogram with Weibull fit overlay "
                 "and save as PNG.",
        "type": "weibull_analysis",
        "variables": ["u100", "v100"],
        "region": "German Bight",
    },
    {
        "id": 27,
        "slug": "solar_sahara_vs_germany",
        "query": "Compare incoming solar radiation (SSRD) between the Sahara and "
                 "northern Germany across 2023 — show monthly means.",
        "type": "comparison_timeseries",
        "variables": ["ssrd"],
        "region": "Sahara / Germany",
    },
    {
        "id": 28,
        "slug": "persian_gulf_sst_summer",
        "query": "Map the mean SST across the Persian Gulf and Arabian Sea "
                 "(12-32°N, 44-70°E) for August 2023. Highlight areas where SST "
                 "exceeded 32°C in a spatial map. Save as PNG.",
        "type": "threshold_map",
        "variables": ["sst"],
        "region": "Persian Gulf",
    },

    # ═══════════════════════════════════════════════════════════════
    #  §7 — Diurnal & Sub-Daily Processes
    # ═══════════════════════════════════════════════════════════════
    {
        "id": 29,
        "slug": "sahara_diurnal_t2_blh",
        "query": "Show the diurnal cycle of 2m temperature and boundary layer height "
                 "in the Sahara for July 2024 — dual-axis plot.",
        "type": "diurnal_cycle",
        "variables": ["t2", "blh"],
        "region": "Sahara",
    },
    {
        "id": 30,
        "slug": "amazon_convective_peak",
        "query": "When does convective precipitation peak over the Amazon basin during "
                 "DJF? Hourly climatology please.",
        "type": "diurnal_cycle",
        "variables": ["cp"],
        "region": "Amazon",
    },

    # ═══════════════════════════════════════════════════════════════
    #  §8 — Multi-Variable & Diagnostics
    # ═══════════════════════════════════════════════════════════════
    {
        "id": 31,
        "slug": "europe_rh_august",
        "query": "Compute relative humidity from 2m temperature and dewpoint for "
                 "central Europe, August 2023, and map the spatial mean.",
        "type": "derived_variable",
        "variables": ["t2", "d2"],
        "region": "Central Europe",
    },
    {
        "id": 32,
        "slug": "hovmoller_equator_skt",
        "query": "Create a Hovmöller diagram of 850 hPa equivalent — use skin "
                 "temperature as proxy — along the equator for 2023 to visualise the MJO.",
        "type": "hovmoller",
        "variables": ["skt"],
        "region": "Equatorial",
    },
    {
        "id": 33,
        "slug": "hurricane_otis_dashboard",
        "query": "Plot a summary dashboard for Hurricane Otis (Oct 2023, Acapulco): "
                 "SST map, wind speed time series, and TCWV distribution in one figure.",
        "type": "dashboard",
        "variables": ["sst", "u10", "v10", "tcwv"],
        "region": "East Pacific / Mexico",
    },

    # ═══════════════════════════════════════════════════════════════
    #  §9 — Quick Lookups
    # ═══════════════════════════════════════════════════════════════
    {
        "id": 34,
        "slug": "california_sst_jan",
        "query": "What was the average SST off the coast of California in January 2024? "
                 "Also plot a spatial map of the SST field for that month and save as PNG.",
        "type": "point_retrieval",
        "variables": ["sst"],
        "region": "California",
    },
    {
        "id": 35,
        "slug": "berlin_monthly_temp",
        "query": "Plot the 2023 monthly mean temperature for Berlin as a seasonal curve.",
        "type": "time_series",
        "variables": ["t2"],
        "region": "Berlin",
    },
    {
        "id": 36,
        "slug": "biscay_wind_stats",
        "query": "Download 10m wind speed for the Bay of Biscay, last 3 years, and "
                 "give me basic statistics. Also plot a wind speed histogram or time "
                 "series and save as PNG.",
        "type": "stats_retrieval",
        "variables": ["u10", "v10"],
        "region": "Bay of Biscay",
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
        config = {"recursion_limit": 35}
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
    parser.add_argument("--query", type=int, help="Run a single query by ID (1-36)")
    parser.add_argument("--start", type=int, default=1, help="Start from query ID")
    parser.add_argument("--end", type=int, default=36, help="End at query ID (inclusive)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: data/qa_results)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if folder already has metadata.json")
    args = parser.parse_args()
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set!")
        sys.exit(1)
    
    if args.output:
        output_dir = Path(args.output)
    else:
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
