#!/usr/bin/env python3
"""
Eurus - ERA5 Climate Analysis Agent
======================================
An intelligent oceanography and climate data analysis assistant.

Features:
- Persistent memory across sessions
- Cloud-optimized ERA5 data retrieval
- Interactive Python analysis with visualization
- Conversation history and context awareness

Usage:
    python main.py

Commands:
    q, quit, exit  - Exit the agent
    /clear         - Clear conversation history
    /cache         - List cached datasets
    /memory        - Show memory summary
    /cleardata     - Clear all downloaded ERA5 datasets
    /help          - Show help message
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime

# Suppress noisy warnings from xarray/zarr
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Consolidated metadata", category=UserWarning)

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Setup centralized logging
from eurus.logging_config import setup_logging, cleanup_old_logs
setup_logging(mode="cli")
cleanup_old_logs(keep=20)

logger = logging.getLogger(__name__)

# Import after logging is configured
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from eurus.config import CONFIG, AGENT_SYSTEM_PROMPT, DATA_DIR, PLOTS_DIR
from eurus.memory import get_memory, MemoryManager
from eurus.tools import get_all_tools


# ============================================================================
# BANNER AND HELP
# ============================================================================

BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ███████╗██╗   ██╗██████╗ ██╗   ██╗███████╗                              ║
║    ██╔════╝██║   ██║██╔══██╗██║   ██║██╔════╝                              ║
║    █████╗  ██║   ██║██████╔╝██║   ██║███████╗                              ║
║    ██╔══╝  ██║   ██║██╔══██╗██║   ██║╚════██║                              ║
║    ███████╗╚██████╔╝██║  ██║╚██████╔╝███████║                              ║
║    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝                              ║
║                                                                           ║
║                  AI Climate Physicist v2.0                                ║
║           ─────────────────────────────────────────                       ║
║                                                                           ║
║   Scientific Capabilities:                                                ║
║   • ERA5 reanalysis data retrieval (SST, wind, temperature, pressure)     ║
║   • Climate Diagnostics: Anomalies, Z-Scores, Statistical Significance    ║
║   • Pattern Discovery: EOF/PCA analysis for climate modes                 ║
║   • Compound Extremes: "Ocean Oven" detection (Heat + Stagnation)         ║
║   • Trend Analysis: Decadal trends with p-value significance              ║
║   • Teleconnections: Correlation and lead-lag analysis                    ║
║   • Maritime Routing & Lagrangian Risk Assessment                         ║
║                                                                           ║
║   Commands: /help, /clear, /cache, /memory, /quit                         ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                          EURUS HELP - AI Climate Physicist               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  COMMANDS:                                                                ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    /help       - Show this help message                                   ║
║    /clear      - Clear conversation history (fresh start)                 ║
║    /cache      - List all cached ERA5 datasets                            ║
║    /memory     - Show memory summary (datasets, analyses)                 ║
║    /cleardata  - Clear all downloaded ERA5 datasets                       ║
║    /quit       - Exit the agent (also: q, quit, exit)                     ║
║                                                                           ║
║  SCIENTIFIC ANALYSIS (Publication-Grade):                                 ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    "Analyze marine heatwaves in the North Atlantic summer 2023"           ║
║    "Find compound extremes where high SST coincides with low wind"        ║
║    "Perform EOF analysis on SST anomalies to find climate modes"          ║
║    "Calculate SST trends with statistical significance"                   ║
║    "Detect Ocean Ovens in the Mediterranean"                              ║
║                                                                           ║
║  SCIENCE TOOLS (The "Physics Brain"):                                     ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    analyze_climate_modes_eof    - Pattern discovery via EOF/PCA           ║
║    detect_compound_extremes     - "Ocean Oven" detection                  ║
║    calculate_climate_trends     - Trends with p-value significance        ║
║    detrend_climate_data         - Remove warming trend for analysis       ║
║    detect_percentile_extremes   - Percentile-based extreme detection      ║
║    fetch_climate_index          - NOAA indices (Nino3.4, NAO, PDO, AMO)   ║
║    calculate_return_periods     - GEV/EVT (1-in-100 year events)          ║
║    analyze_granger_causality    - Prove X causes Y (not just correlated)  ║
║                                                                           ║
║  AVAILABLE VARIABLES:                                                     ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    sst  - Sea Surface Temperature (K)                                     ║
║    t2   - 2m Air Temperature (K)                                          ║
║    u10  - 10m U-Wind Component (m/s)                                      ║
║    v10  - 10m V-Wind Component (m/s)                                      ║
║    mslp - Mean Sea Level Pressure (Pa)                                    ║
║    tcc  - Total Cloud Cover (0-1)                                         ║
║    tp   - Total Precipitation (m)                                         ║
║                                                                           ║
║  PREDEFINED REGIONS:                                                      ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    north_atlantic, north_pacific, california_coast, mediterranean         ║
║    gulf_of_mexico, caribbean, nino34, nino3, nino4, arctic, antarctic     ║
║                                                                           ║
║  SCIENTIFIC WORKFLOW:                                                     ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    1. RETRIEVE data → 2. DIAGNOSE (Z-scores) → 3. DISCOVER (EOF)          ║
║    4. DETECT (extremes) → 5. ATTRIBUTE (correlation) → 6. VISUALIZE       ║
║                                                                           ║
║  TIPS:                                                                    ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    • Always report in anomalies/Z-scores, not raw values                  ║
║    • Z > 2σ means statistically significant extreme                       ║
║    • Use diverging colormaps (RdBu_r) centered at 0 for anomalies         ║
║    • Add stippling for p < 0.05 significance                              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""



def clear_data_directory(data_dir: Path = None) -> tuple[int, float]:
    """
    Remove all downloaded ERA5 datasets (zarr directories) from the data folder.
    
    Args:
        data_dir: Data directory path. Defaults to DATA_DIR from config.
        
    Returns:
        Tuple of (datasets_removed, total_size_mb_freed)
    """
    import shutil
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    datasets_removed = 0
    total_bytes = 0
    
    if not data_dir.exists():
        return 0, 0.0
    
    # Find and remove all .zarr directories
    for zarr_dir in data_dir.glob('*.zarr'):
        if zarr_dir.is_dir():
            # Calculate size before removing
            dir_size = sum(f.stat().st_size for f in zarr_dir.rglob('*') if f.is_file())
            total_bytes += dir_size
            shutil.rmtree(zarr_dir)
            datasets_removed += 1
            logger.debug(f"Removed dataset: {zarr_dir}")
    
    total_mb = total_bytes / (1024 * 1024)
    return datasets_removed, total_mb


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def handle_command(command: str, memory: MemoryManager) -> tuple[bool, str]:
    """
    Handle slash commands.

    Returns:
        (should_continue, response_message)
    """
    cmd = command.lower().strip()

    if cmd in ('/quit', '/exit', '/q', 'quit', 'exit', 'q'):
        return False, "Goodbye! Your conversation has been saved."

    elif cmd == '/help':
        return True, HELP_TEXT

    elif cmd == '/clear':
        memory.clear_conversation()
        return True, "Conversation history cleared. Starting fresh!"

    elif cmd == '/cache':
        cache_info = memory.list_datasets()
        return True, f"\n{cache_info}\n"

    elif cmd == '/memory':
        summary = memory.get_context_summary()
        datasets = len([p for p in memory.datasets if os.path.exists(p)])
        analyses = len(memory.analyses)
        convos = len(memory.conversations)

        response = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                         MEMORY SUMMARY                                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Conversation messages: {convos:<5}                                        ║
║  Cached datasets: {datasets:<5}                                             ║
║  Recorded analyses: {analyses:<5}                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

{summary}
"""
        return True, response

    elif cmd == '/cleardata':
        datasets_removed, size_freed = clear_data_directory(DATA_DIR)
        # Also clear memory references
        memory.datasets.clear()
        memory._save_datasets()
        response = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                       ERA5 DATA CLEARED                                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Datasets removed: {datasets_removed:<5}                                               ║
║  Space freed: {size_freed:>8.2f} MB                                              ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""
        return True, response

    elif cmd.startswith('/'):
        return True, f"Unknown command: {cmd}\nType /help for available commands."

    return True, None  # Not a command


# ============================================================================
# MAIN AGENT LOOP
# ============================================================================

def main():
    """Main entry point for the Eurus agent."""

    # Print banner
    print(BANNER)

    # Check for required API keys
    if not os.environ.get("ARRAYLAKE_API_KEY"):
        print("ERROR: ARRAYLAKE_API_KEY not found in environment.")
        print("Please add it to your .env file:")
        print("  ARRAYLAKE_API_KEY=your_api_key_here")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment.")
        print("Please add it to your .env file:")
        print("  OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Initialize memory
    print("Initializing memory system...")
    memory = get_memory()

    # Load recent conversation context
    recent_messages = memory.get_langchain_messages(n_messages=10)
    logger.info(f"Loaded {len(recent_messages)} messages from history")

    # Initialize tools
    print("Starting Python kernel...")

    # All capabilities enabled by default (including maritime routing)
    tools = get_all_tools(enable_routing=True, enable_guide=True)
    logger.info(f"Loaded {len(tools)} tools")

    # Initialize LLM
    print("Connecting to LLM...")
    llm = ChatOpenAI(
        model=CONFIG.model_name,
        temperature=CONFIG.temperature,
        streaming=True  # Enable streaming for real-time output
    )

    # Create enhanced system prompt with context
    context_summary = memory.get_context_summary()
    enhanced_prompt = AGENT_SYSTEM_PROMPT

    if context_summary and context_summary != "No context available.":
        enhanced_prompt += f"\n\n## CURRENT CONTEXT\n{context_summary}"

    # Create agent
    print("Creating agent...")
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=enhanced_prompt,
        debug=False
    )

    # Initialize messages with history
    messages = recent_messages.copy()

    print("\n" + "=" * 75)
    print("READY! Type your question or /help for commands.")
    print("=" * 75 + "\n")

    # Main interaction loop
    try:
        while True:
            # Get user input
            try:
                user_input = input(">> You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle commands
            should_continue, response = handle_command(user_input, memory)

            if response:
                print(response)

            if not should_continue:
                break

            if response:  # Command was handled, skip agent
                continue

            # Save user message to memory
            memory.add_message("user", user_input)
            messages.append({"role": "user", "content": user_input})

            # Get agent response
            print("\nThinking...\n")

            try:
                # Stream the response for real-time output
                print("\n" + "─" * 75)
                
                full_response = ""
                tool_executed = False
                
                # Limit recursion to prevent infinite loops (~15 tool calls max)
                config = {"recursion_limit": 35}
                for event in agent.stream({"messages": messages}, stream_mode="updates", config=config):
                    # Handle different event types
                    for node_name, node_output in event.items():
                        if node_name == "agent":
                            # LLM is producing output
                            if "messages" in node_output:
                                for msg in node_output["messages"]:
                                    # Check for tool calls (only if list is non-empty)
                                    has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
                                    has_content = hasattr(msg, 'content') and msg.content
                                    
                                    if has_tool_calls:
                                        for tc in msg.tool_calls:
                                            tool_name = tc.get('name', 'unknown')
                                            print(f"🔧 Calling: {tool_name}...", flush=True)
                                            tool_executed = True
                                    
                                    # Also check for content (not elif — a message can have both)
                                    if has_content:
                                        if not tool_executed:
                                            print("Eurus: ", end="", flush=True)
                                        else:
                                            print("\n\n📝 Response:", flush=True)
                                        print(msg.content, flush=True)
                                        full_response = msg.content
                        
                        elif node_name == "tools":
                            # Tool execution completed
                            if "messages" in node_output:
                                for msg in node_output["messages"]:
                                    if hasattr(msg, 'name'):
                                        print(f"   ✓ {msg.name} done", flush=True)
                
                print("─" * 75 + "\n")
                
                # Update messages for the next turn
                if full_response:
                    messages.append({"role": "assistant", "content": full_response})
                    memory.add_message("assistant", full_response)
                else:
                    # Streaming completed but no text response was captured.
                    # This can happen if the LLM only used tools without a final summary.
                    # Do NOT re-invoke — that would re-execute all tools.
                    logger.warning("Streaming completed without capturing final text response")
                    fallback_text = "Analysis complete. Let me know if you need anything else!"
                    messages.append({"role": "assistant", "content": fallback_text})
                    memory.add_message("assistant", fallback_text)
                    print(f"Eurus: {fallback_text}")
                    print("─" * 75 + "\n")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit or continue with a new question.")

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                print(f"\nError during processing: {error_msg}")
                print("Please try again or rephrase your question.\n")

    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal.")

    finally:
        # Cleanup
        print("\nShutting down...")

        # Clean up missing dataset records
        removed = memory.cleanup_missing_datasets()
        if removed:
            logger.info(f"Cleaned up {removed} missing dataset records")

        print("Session saved. Goodbye!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
