"""
Agent Wrapper for Web Interface
===============================
Wraps the LangChain agent for WebSocket streaming.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable, Any, List, Dict
from queue import Queue

# Add src directory to path for eurus package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# IMPORT FROM EURUS PACKAGE - SINGLE SOURCE OF TRUTH
from eurus.config import CONFIG, AGENT_SYSTEM_PROMPT
from eurus.memory import get_memory  # Use SINGLETON so tools can register datasets!
from eurus.tools import get_all_tools
from eurus.tools.repl import PythonREPLTool

logger = logging.getLogger(__name__)


class AgentSession:
    """
    Manages a single agent session with streaming support.
    """

    def __init__(self):
        self._agent = None
        self._repl_tool: Optional[PythonREPLTool] = None
        self._messages: List[Dict] = []
        self._initialized = False
        
        # Use global memory singleton (so tools like retrieve_era5_data can register datasets!)
        # But clear conversation history for fresh session (datasets cache remains)
        self._memory = get_memory()
        self._memory.clear_conversation()  # Fresh chat, keep cached datasets

        # Queue for captured plots (thread-safe)
        self._plot_queue: Queue = Queue()

        self._initialize()

    def _initialize(self):
        """Initialize the agent and tools."""
        logger.info("Initializing agent session...")

        if not os.environ.get("ARRAYLAKE_API_KEY"):
            logger.warning("ARRAYLAKE_API_KEY not found")

        if not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not found")
            return

        try:
            # Initialize REPL tool with working directory
            logger.info("Starting Python kernel...")
            self._repl_tool = PythonREPLTool(working_dir=os.getcwd())

            # Set up plot callback using the proper method
            def on_plot_captured(base64_data: str, filepath: str, code: str = ""):
                logger.info(f"Plot captured, adding to queue: {filepath}")
                self._plot_queue.put((base64_data, filepath, code))

            self._repl_tool.set_plot_callback(on_plot_captured)
            logger.info("Plot callback registered")

            # Get ALL tools from centralized registry (no SCIENCE_TOOLS!)
            tools = get_all_tools(enable_routing=True, enable_guide=True)
            # Replace the default REPL with our configured one
            tools = [t for t in tools if t.name != "python_repl"] + [self._repl_tool]

            # Initialize LLM
            logger.info("Connecting to LLM...")
            llm = ChatOpenAI(
                model=CONFIG.model_name,
                temperature=CONFIG.temperature
            )

            # Use session-local memory for datasets (NOT global!)
            datasets = self._memory.list_datasets()
            enhanced_prompt = AGENT_SYSTEM_PROMPT
            
            if datasets != "No datasets in cache.":
                enhanced_prompt += f"\n\n## CACHED DATASETS\n{datasets}"

            # Create agent
            logger.info("Creating agent...")
            self._agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=enhanced_prompt,
                debug=False
            )

            # FRESH conversation - no old messages!
            self._messages = []

            self._initialized = True
            logger.info("Agent session initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize agent: {e}")
            self._initialized = False

    def is_ready(self) -> bool:
        """Check if the agent is ready."""
        return self._initialized and self._agent is not None

    def clear_messages(self):
        """Clear conversation messages."""
        self._messages = []

    def get_pending_plots(self) -> List[tuple]:
        """Get all pending plots from queue."""
        plots = []
        while not self._plot_queue.empty():
            try:
                plots.append(self._plot_queue.get_nowait())
            except:
                break
        return plots

    async def process_message(
        self,
        user_message: str,
        stream_callback: Callable
    ) -> str:
        """
        Process a user message and stream the response.
        """
        if not self.is_ready():
            raise RuntimeError("Agent not initialized")

        # Clear any old plots from queue
        self.get_pending_plots()

        # Add user message to history (session-local memory)
        self._memory.add_message("user", user_message)
        self._messages.append({"role": "user", "content": user_message})

        try:
            # Send status: analyzing
            await stream_callback("status", "🔍 Analyzing your request...")
            await asyncio.sleep(0.3)

            # Invoke the agent in executor (~15 tool calls max)
            config = {"recursion_limit": 35}
            
            # Stream status updates while agent is working
            await stream_callback("status", "🤖 Processing with AI...")
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._agent.invoke({"messages": self._messages}, config=config)
            )

            # Update messages
            self._messages = result["messages"]
            
            # Parse messages to show tool calls made
            tool_calls_made = []
            for msg in self._messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        if tool_name not in tool_calls_made:
                            tool_calls_made.append(tool_name)
                            
            if tool_calls_made:
                tools_str = ", ".join(tool_calls_made)
                await stream_callback("status", f"🛠️ Used tools: {tools_str}")
                await asyncio.sleep(0.5)

            # Extract response
            last_message = self._messages[-1]

            if hasattr(last_message, 'content') and last_message.content:
                response_text = last_message.content
            elif isinstance(last_message, dict) and last_message.get('content'):
                response_text = last_message['content']
            else:
                response_text = str(last_message)

            # Send status: generating response
            await stream_callback("status", "✍️ Generating response...")
            await asyncio.sleep(0.2)

            # Stream the response in chunks
            chunk_size = 50
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                await stream_callback("chunk", chunk)
                await asyncio.sleep(0.01)

            # Detect interactive map tool results and send as tile_map events
            import json as _json
            for msg in self._messages:
                # Check for ToolMessage from render_interactive_map
                if hasattr(msg, 'name') and msg.name == 'render_interactive_map':
                    try:
                        map_data = _json.loads(msg.content)
                        if map_data.get("type") == "interactive_map":
                            logger.info(f"Sending interactive map: {map_data['options'].get('label', 'map')}")
                            await stream_callback(
                                "tile_map", "",
                                tile_url=map_data["tile_url"],
                                options=map_data.get("options", {})
                            )
                    except (_json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse interactive map result: {e}")

            # Send any captured media (plots and videos)
            plots = self.get_pending_plots()
            # NOTE: Only use session-specific _plot_queue, NOT shared folder scan (privacy!)
            
            if plots:
                await stream_callback("status", f"📊 Rendering {len(plots)} visualization(s)...")
                await asyncio.sleep(0.3)
                
            logger.info(f"Sending {len(plots)} media items to client")
            for plot_data in plots:
                base64_data, filepath = plot_data[0], plot_data[1]
                code = plot_data[2] if len(plot_data) > 2 else ""
                
                # Determine if this is a video or image
                ext = filepath.lower().split('.')[-1] if filepath else ''
                if ext in ('gif',):
                    await stream_callback("video", "", data=base64_data, path=filepath, mimetype="image/gif")
                elif ext in ('webm',):
                    await stream_callback("video", "", data=base64_data, path=filepath, mimetype="video/webm")
                elif ext in ('mp4',):
                    await stream_callback("video", "", data=base64_data, path=filepath, mimetype="video/mp4")
                else:
                    # Default to plot (png, jpg, etc.)
                    await stream_callback("plot", "", data=base64_data, path=filepath, code=code)

            # Save to memory
            self._memory.add_message("assistant", response_text)

            return response_text

        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            raise

    def close(self):
        """Clean up resources."""
        logger.info("Closing agent session...")
        if self._repl_tool:
            try:
                self._repl_tool.close()
            except Exception as e:
                logger.error(f"Error closing REPL: {e}")


# Per-connection sessions (NOT global singleton!)
# Key: unique connection ID, Value: AgentSession
_sessions: Dict[str, AgentSession] = {}


def create_session(connection_id: str) -> AgentSession:
    """Create a new session for a connection."""
    if connection_id in _sessions:
        # Close existing session first
        _sessions[connection_id].close()
    session = AgentSession()
    _sessions[connection_id] = session
    logger.info(f"Created session for connection: {connection_id}")
    return session


def get_session(connection_id: str) -> Optional[AgentSession]:
    """Get session for a connection."""
    return _sessions.get(connection_id)


def close_session(connection_id: str):
    """Close and remove session for a connection."""
    if connection_id in _sessions:
        _sessions[connection_id].close()
        del _sessions[connection_id]
        logger.info(f"Closed session for connection: {connection_id}")


# DEPRECATED: Keep for backward compatibility during migration
def get_agent_session() -> AgentSession:
    """DEPRECATED: Use create_session/get_session with connection_id instead."""
    logger.warning("get_agent_session() is deprecated - use create_session(connection_id)")
    # Create default session for CLI/testing
    if "_default" not in _sessions:
        _sessions["_default"] = AgentSession()
    return _sessions["_default"]


def shutdown_agent_session():
    """Shutdown all agent sessions."""
    for conn_id in list(_sessions.keys()):
        close_session(conn_id)
    logger.info(f"Shutdown {len(_sessions)} sessions")
