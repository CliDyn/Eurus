#!/usr/bin/env python3
"""
QA Image Reviewer — Uses Gemini 3 Pro Preview (Vertex AI Express) to review
all generated plots from a QA run and checks whether each plot matches its
task requirements.

Usage:
    python scripts/qa_image_review.py [--run RUN_DIR] [--query N] [--output FILE]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── project root ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Gemini Config ───────────────────────────────────────────────────
PRIMARY_MODEL = "gemini-3-pro-preview"

# ── The query definitions (mirrored from qa_runner.py) ──────────────
QA_QUERIES = {
    1:  {"slug": "europe_heatwave_anomaly",
         "task": "Spatial map of 2m temperature anomalies across Europe during June 2023 heatwave vs June 2022."},
    2:  {"slug": "storm_isha_mslp_wind",
         "task": "MSLP isobars and 10m wind vectors over the North Atlantic for 2024-01-22 showing Storm Isha."},
    3:  {"slug": "atmospheric_river_jan2023",
         "task": "Total column water vapour for the US West Coast, Jan 2023, showing the atmospheric river event around Jan 9th."},
    4:  {"slug": "sahara_heat_july2024",
         "task": "Daily mean 2m temperature time series over the Sahara for July 2024 vs July 2023 on the same chart."},
    5:  {"slug": "great_plains_wind_may2024",
         "task": "Map of mean 10m wind speed over US Great Plains for May 2024, highlighting areas >5 m/s."},
    6:  {"slug": "nino34_index",
         "task": "Niño 3.4 index from ERA5 SST for 2015-2024 classifying El Niño / La Niña episodes."},
    7:  {"slug": "elnino_vs_lanina_tropical_belt",
         "task": "SST anomaly difference map: Dec 2023 (El Niño) minus Dec 2022 (La Niña) across the tropical belt."},
    8:  {"slug": "nao_index",
         "task": "NAO index from MSLP (Azores minus Iceland) for 2000-2024 with 3-month rolling mean."},
    9:  {"slug": "australia_enso_rainfall",
         "task": "Two-panel map of annual total precipitation over Eastern Australia for La Niña 2022 vs El Niño 2023, plus difference map."},
    10: {"slug": "med_eof_sst",
         "task": "EOF analysis on Mediterranean SST anomalies for 2019-2024: first 3 modes with variance explained."},
    11: {"slug": "arctic_polar_amplification",
         "task": "January mean 2m temperature maps for the Arctic (>70°N): 2024 vs 2000 side by side, with polar amplification quantification."},
    12: {"slug": "med_marine_heatwave_2023",
         "task": "Summer JJA 2023 SST anomaly map over the Mediterranean vs 2018-2022 mean, highlighting marine heatwave hotspots >+2°C."},
    13: {"slug": "paris_decadal_comparison",
         "task": "Average summer (JJA) temperature difference map for Paris: 2014-2023 vs 2000-2009, plus time series."},
    14: {"slug": "alps_snow_trend",
         "task": "December-February snow depth trend over the Alps for the last 30 years."},
    15: {"slug": "uk_precip_anomaly_winter2024",
         "task": "Total precipitation anomaly map over the British Isles for January 2024 vs 2019-2023 January mean, highlighting >150% normal."},
    16: {"slug": "delhi_heatwave_detection",
         "task": "Heatwave events in Delhi 2010-2024 using 90th percentile threshold with 3-day criterion; frequency change analysis."},
    17: {"slug": "horn_africa_drought",
         "task": "3-month SPI proxy for the Horn of Africa 2020-2024, identifying worst drought periods."},
    18: {"slug": "baghdad_hot_days",
         "task": "Bar chart of days per year >35°C in Baghdad from 1980-2024 with trend line."},
    19: {"slug": "sea_p95_precip",
         "task": "95th percentile daily precipitation map for Southeast Asia 2010-2023."},
    20: {"slug": "scandinavia_blocking_2018",
         "task": "Blocking event over Scandinavia July 2018: MSLP anomalies persisting 5+ days."},
    21: {"slug": "rotterdam_shanghai_route",
         "task": "Maritime route from Rotterdam to Shanghai with wind risk analysis for December."},
    22: {"slug": "indian_ocean_sst_dipole",
         "task": "SST anomaly map across the Indian Ocean for October 2023 relative to 2019-2022 October mean, showing IOD pattern."},
    23: {"slug": "japan_typhoon_season_wind",
         "task": "Mean and maximum 10m wind speed maps around Japan during typhoon season (Aug-Oct) 2023, highlighting areas >8 m/s."},
    24: {"slug": "south_atlantic_sst_gradient",
         "task": "Mean SST field across the South Atlantic for March 2024 with SST isotherms and Brazil-Malvinas confluence zone."},
    25: {"slug": "north_sea_wind_power",
         "task": "Mean 100m wind power density map across the North Sea for 2020-2024 identifying best offshore wind sites."},
    26: {"slug": "german_bight_weibull",
         "task": "Weibull distribution fit to 100m wind speed at German Bight for 2023 with histogram and fit overlay."},
    27: {"slug": "solar_sahara_vs_germany",
         "task": "Monthly mean incoming solar radiation (SSRD) comparison: Sahara vs Northern Germany for 2023."},
    28: {"slug": "persian_gulf_sst_summer",
         "task": "Mean SST map across Persian Gulf and Arabian Sea for August 2023, highlighting areas where SST >32°C."},
    29: {"slug": "sahara_diurnal_t2_blh",
         "task": "Diurnal cycle of 2m temperature and boundary layer height in the Sahara for July 2024, dual-axis plot."},
    30: {"slug": "amazon_convective_peak",
         "task": "Hourly climatology of convective precipitation peak over the Amazon basin during DJF."},
    31: {"slug": "europe_rh_august",
         "task": "Relative humidity map from 2m temperature and dewpoint for central Europe, August 2023."},
    32: {"slug": "hovmoller_equator_skt",
         "task": "Hovmöller diagram of skin temperature along the equator for 2023 to visualize MJO."},
    33: {"slug": "hurricane_otis_dashboard",
         "task": "Summary dashboard for Hurricane Otis (Oct 2023): SST map, wind speed time series, TCWV distribution in one figure."},
    34: {"slug": "california_sst_jan",
         "task": "Average SST off California coast in January 2024 with spatial map of the SST field."},
    35: {"slug": "berlin_monthly_temp",
         "task": "2023 monthly mean temperature for Berlin as a seasonal curve."},
    36: {"slug": "biscay_wind_stats",
         "task": "10m wind speed stats for Bay of Biscay (last 3 years) with histogram or time series plot."},
}


REVIEW_SYSTEM_PROMPT = """\
You are a RUTHLESS, METICULOUS senior scientific visualization reviewer for a climate/weather data agent.
You will receive one or more PNG plots generated by an AI agent and the TASK that the agent was asked to complete.

YOUR #1 JOB: For EVERY issue you find, describe it with EXACT SPECIFICITY.
Do NOT say "labels are unclear" — say EXACTLY which label, where it is, and what is wrong with it.
Do NOT say "colorbar could be better" — say EXACTLY what the colorbar shows, what it should show, and what the specific problem is.
Do NOT give vague feedback. Every single issue MUST pinpoint the EXACT location and EXACT problem in the figure.

CRITICAL: Be EXTREMELY SPECIFIC about problems. Point to EXACT elements:
- "The y-axis label says 'Value' but should say 'Temperature (°C)'"
- "The colorbar range is 270-310K but should be converted to °C for readability"
- "Coastlines are missing from the spatial map — there is no land/ocean boundary visible"
- "The title says 'January 2024' but the x-axis data only covers December 2023"
- "The legend overlaps with the data in the upper-right quadrant, obscuring the January peak"
- "Wind vectors are plotted but have no reference arrow showing the scale"
- "The projection is PlateCarree but should be a polar stereographic for Arctic data above 70°N"

For EACH problem: describe WHERE in the figure it is, WHAT exactly is wrong, and WHAT it should be instead.

Review each plot against the task and provide a structured assessment:

1. **Task Compliance** (1-10): Does the plot address EXACTLY what was asked? Check every single requirement in the task description. If the task says "two-panel" and there's only one panel, that is a major failure. If the task says "vs" comparison and only one dataset is shown, that is a failure. Be strict.

2. **Scientific Accuracy** (1-10): Are ALL axes labeled with correct units? Is the colorbar present with proper units and range? Are values physically reasonable (e.g., SST not showing 0K)? Are projections appropriate for the region? Check EVERY axis, EVERY label, EVERY unit.

3. **Visual Quality** (1-10): Is it publication-quality? Check: font sizes readable? Labels not overlapping data? Grid lines appropriate? Color scheme suitable (e.g., diverging for anomalies, sequential for absolute values)? Title descriptive and correct?

4. **Spatial/Map Quality** (1-10): For maps — are coastlines drawn? Is the projection correct for the region? Are lat/lon gridlines present? Are geographic features identifiable? For non-maps — is the chart type appropriate?

5. **Overall Score** (1-10): Weighted average. Be HARSH — a score of 8+ means near-perfect.

Also provide:
- **Summary**: 1-2 sentence factual summary of what the plot actually shows.
- **Strengths**: Specific things done well. Be precise — not "good colors" but "diverging RdBu colormap correctly centered at zero for anomaly data".
- **Issues**: LIST EVERY SINGLE PROBLEM. Each issue MUST describe the EXACT element, its EXACT location in the figure, WHAT is wrong, and WHAT it should be. DO NOT BE VAGUE. This is the MOST IMPORTANT part of your review. Be exhaustive. Miss nothing.

I REPEAT: The "issues" field is the MOST CRITICAL part. Every issue must be SPECIFIC and ACTIONABLE. Generic feedback like "could be improved" is UNACCEPTABLE. Say EXACTLY what needs to change and WHERE.

Respond ONLY in valid JSON with this exact structure:
{
  "task_compliance": <int>,
  "scientific_accuracy": <int>,
  "visual_quality": <int>,
  "spatial_quality": <int>,
  "overall_score": <int>,
  "summary": "<string>",
  "strengths": ["<string>", ...],
  "issues": ["<string — MUST be specific and exact, describing WHERE and WHAT>", ...]
}
"""


def create_client() -> genai.Client:
    """Create Gemini API client using Vertex AI Express (same pattern as cmip6 project)."""
    api_key = os.environ.get("vertex_api_key")
    if not api_key:
        print("❌ vertex_api_key not found in .env!")
        sys.exit(1)
    print(f"  Using Vertex AI Express (API key auth)")
    return genai.Client(vertexai=True, api_key=api_key)


def review_single_question(client: genai.Client, qid: int, task: str,
                           image_paths: list[Path], model: str) -> dict:
    """Send images + task to Gemini and get structured review."""

    # Build content parts: text prompt + inline images
    prompt_text = (
        f"**TASK (Q{qid:02d}):** {task}\n\n"
        f"Below are {len(image_paths)} plot(s) generated by the agent. "
        f"Review them against the task."
    )
    parts = [types.Part.from_text(text=prompt_text)]

    for img_path in image_paths:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

    for attempt in range(6):
        try:
            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=REVIEW_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=4096,
                ),
            )
            raw = response.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"error": f"Failed to parse JSON: {raw[:500]}"}
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = min(2 ** attempt * 15, 120)
                print(f"\n  Rate limited, waiting {wait}s (attempt {attempt+1}/6)...", end="", flush=True)
                time.sleep(wait)
            else:
                if attempt < 5:
                    time.sleep(3)
                    continue
                return {"error": str(e)[:300]}

    return {"error": "Max retries exceeded"}


def main():
    parser = argparse.ArgumentParser(description="QA Image Reviewer using Gemini 3 Pro Preview")
    parser.add_argument("--run", type=str, default=None,
                        help="Path to QA run directory (default: latest in data/qa_runs/)")
    parser.add_argument("--query", type=int, default=None,
                        help="Review only a specific query ID")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: <run_dir>/image_review.json)")
    parser.add_argument("--model", type=str, default=PRIMARY_MODEL,
                        help=f"Gemini model to use (default: {PRIMARY_MODEL})")
    args = parser.parse_args()

    # Find run directory
    if args.run:
        run_dir = Path(args.run)
    else:
        qa_runs = PROJECT_ROOT / "data" / "qa_runs"
        runs = sorted(qa_runs.glob("run_*"))
        if not runs:
            print("❌ No QA runs found in data/qa_runs/")
            sys.exit(1)
        run_dir = runs[-1]

    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)

    # Gemini client (Vertex AI Express)
    client = create_client()

    print(f"""
╔══════════════════════════════════════════════════════╗
║    QA Image Reviewer (Gemini 3 Pro Preview)         ║
║    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                          ║
╚══════════════════════════════════════════════════════╝
Run directory: {run_dir}
Model: {args.model}
""")

    # Collect questions to review
    all_reviews = {}
    question_dirs = sorted(run_dir.glob("q*_*"))

    for qdir in question_dirs:
        # Extract question ID from folder name (e.g., q01_xxx -> 1)
        try:
            qid = int(qdir.name.split("_")[0][1:])
        except (ValueError, IndexError):
            continue

        if args.query and qid != args.query:
            continue

        if qid not in QA_QUERIES:
            print(f"⚠️  Q{qid:02d}: Unknown query ID, skipping")
            continue

        # Find PNG files
        pngs = sorted(qdir.glob("*.png"))
        if not pngs:
            print(f"⏭️  Q{qid:02d} ({QA_QUERIES[qid]['slug']}): No PNG files, skipping")
            all_reviews[qid] = {"status": "no_images", "slug": QA_QUERIES[qid]["slug"]}
            continue

        task_desc = QA_QUERIES[qid]["task"]
        png_names = [p.name for p in pngs]

        print(f"🔍 Q{qid:02d} ({QA_QUERIES[qid]['slug']}): Reviewing {len(pngs)} image(s)...", end=" ", flush=True)

        try:
            start = time.time()
            review = review_single_question(client, qid, task_desc, pngs, args.model)
            elapsed = time.time() - start

            review["slug"] = QA_QUERIES[qid]["slug"]
            review["task"] = task_desc
            review["images"] = png_names
            review["status"] = "reviewed"
            review["review_time_s"] = round(elapsed, 1)

            score = review.get("overall_score", "?")
            if isinstance(score, int):
                icon = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
            else:
                icon = "❓"
            print(f"{icon} Score: {score}/10 ({elapsed:.1f}s)")

            all_reviews[qid] = review

        except Exception as e:
            print(f"❌ Error: {e}")
            all_reviews[qid] = {
                "status": "error",
                "slug": QA_QUERIES[qid]["slug"],
                "error": str(e),
            }

        # Rate limit: pause between calls
        time.sleep(1)

    # ── Summary ──────────────────────────────────────────────────────
    reviewed = [v for v in all_reviews.values() if v.get("status") == "reviewed"]
    scores = [v["overall_score"] for v in reviewed if isinstance(v.get("overall_score"), int)]

    print(f"\n{'='*70}")
    print("REVIEW SUMMARY")
    print(f"{'='*70}")

    # Score table
    for qid in sorted(all_reviews.keys()):
        r = all_reviews[qid]
        if r.get("status") == "reviewed":
            s = r.get("overall_score", 0)
            if isinstance(s, int):
                icon = "✅" if s >= 7 else "⚠️" if s >= 5 else "❌"
            else:
                icon = "❓"
            tc = r.get("task_compliance", "?")
            sa = r.get("scientific_accuracy", "?")
            vq = r.get("visual_quality", "?")
            sq = r.get("spatial_quality", "?")
            print(f"  {icon} Q{qid:02d} {r['slug']:35s} | Overall: {s:>2}/10 | "
                  f"Task:{tc} Sci:{sa} Vis:{vq} Spa:{sq}")
        elif r.get("status") == "no_images":
            print(f"  ⏭️  Q{qid:02d} {r['slug']:35s} | No images")
        else:
            print(f"  ❌ Q{qid:02d} {r['slug']:35s} | Error: {r.get('error', 'unknown')[:50]}")

    if scores:
        avg = sum(scores) / len(scores)
        excellent = sum(1 for s in scores if s >= 8)
        good = sum(1 for s in scores if 6 <= s < 8)
        needs_work = sum(1 for s in scores if s < 6)

        print(f"\n📊 Average score: {avg:.1f}/10 across {len(scores)} reviewed plots")
        print(f"   🟢 Excellent (8-10): {excellent}")
        print(f"   🟡 Good (6-7):       {good}")
        print(f"   🔴 Needs work (<6):  {needs_work}")

    # ── Save results ─────────────────────────────────────────────────
    output_path = Path(args.output) if args.output else run_dir / "image_review.json"

    # Convert int keys to strings for JSON
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "run_directory": str(run_dir),
        "model": args.model,
        "total_reviewed": len(reviewed),
        "average_score": round(avg, 2) if scores else None,
        "reviews": {f"q{k:02d}": v for k, v in sorted(all_reviews.items())},
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Full review saved to: {output_path}")


if __name__ == "__main__":
    main()
