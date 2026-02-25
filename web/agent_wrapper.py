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
from eurus.retrieval import _arraylake_snippet
from eurus.memory import get_memory, SmartConversationMemory  # Singleton for datasets, per-session for chat
from eurus.tools import get_all_tools
from eurus.tools.repl import PythonREPLTool

logger = logging.getLogger(__name__)


class AgentSession:
    """
    Manages a single agent session with streaming support.
    """

    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self._agent = None
        self._repl_tool: Optional[PythonREPLTool] = None
        self._messages: List[Dict] = []
        self._initialized = False
        self._api_keys = api_keys or {}

        # Global singleton keeps the dataset cache (shared across sessions)
        self._memory = get_memory()
        # Per-session conversation memory — never touches other sessions
        self._conversation = SmartConversationMemory()

        # Queue for captured plots (thread-safe)
        self._plot_queue: Queue = Queue()

        self._initialize()

    def _initialize(self):
        """Initialize the agent and tools."""
        logger.info("Initializing agent session...")

        # Resolve API keys: user-provided take priority over env vars
        openai_key = self._api_keys.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        arraylake_key = self._api_keys.get("arraylake_api_key") or os.environ.get("ARRAYLAKE_API_KEY")
        hf_token = self._api_keys.get("hf_token") or os.environ.get("HF_TOKEN")

        if not arraylake_key:
            logger.warning("ARRAYLAKE_API_KEY not found")
        elif not os.environ.get("ARRAYLAKE_API_KEY"):
            # Only set env var if not already configured (avoid overwriting
            # server-configured keys with user-provided ones in multi-user scenarios)
            os.environ["ARRAYLAKE_API_KEY"] = arraylake_key

        if hf_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = hf_token

        if not openai_key:
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

            # Initialize LLM with resolved key
            logger.info("Connecting to LLM...")
            llm = ChatOpenAI(
                model=CONFIG.model_name,
                temperature=CONFIG.temperature,
                api_key=openai_key,
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

    def reinitialize(self):
        """Retry initialization (e.g., after transient failure)."""
        logger.warning("Attempting agent reinitialization...")
        self._initialized = False
        self._agent = None
        self._initialize()

    def clear_messages(self):
        """Clear conversation messages."""
        self._messages = []

    def get_pending_plots(self) -> List[tuple]:
        """Get all pending plots from queue."""
        plots = []
        while not self._plot_queue.empty():
            try:
                plots.append(self._plot_queue.get_nowait())
            except Exception:
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
            # Try to reinitialize once before giving up
            logger.warning("Agent not ready, attempting reinitialization...")
            self.reinitialize()
            if not self.is_ready():
                raise RuntimeError("Agent not initialized")

        # Clear any old plots from queue
        self.get_pending_plots()

        # Add user message to history (session-local memory)
        self._conversation.add_message("user", user_message)
        self._messages.append({"role": "user", "content": user_message})

        try:
            # Send status: analyzing
            await stream_callback("status", "🔍 Analyzing your request...")
            await asyncio.sleep(0.3)

            # Invoke the agent in executor (~15 tool calls max)
            config = {"recursion_limit": 35}
            
            # Stream status updates while agent is working
            await stream_callback("status", "🤖 Processing with AI...")

            # Save message state before invoke (protect against corruption)
            messages_backup = list(self._messages)
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._agent.invoke({"messages": self._messages}, config=config)
            )

            # Only scan NEW messages from this turn
            prev_count = len(self._messages)
            self._messages = result["messages"]
            new_messages = self._messages[prev_count:]
            
            # Parse NEW messages to show tool calls made
            tool_calls_made = []
            for msg in new_messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        if tool_name not in tool_calls_made:
                            tool_calls_made.append(tool_name)
                            
            if tool_calls_made:
                tools_str = ", ".join(tool_calls_made)
                await stream_callback("status", f"🛠️ Used tools: {tools_str}")
                await asyncio.sleep(0.5)

            # Collect Arraylake snippet from NEW messages only
            # Only emit ONE snippet per unique (variable, region) — skip failed calls
            arraylake_snippets = []
            seen_snippet_keys = set()
            for i, msg in enumerate(new_messages):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.get('name') == 'retrieve_era5_data':
                            # Check if tool call succeeded by looking at the next message
                            # (ToolMessage with same tool_call_id)
                            tc_id = tc.get('id', '')
                            succeeded = True
                            for later_msg in new_messages[i+1:]:
                                if (hasattr(later_msg, 'tool_call_id') and
                                        later_msg.tool_call_id == tc_id):
                                    content = getattr(later_msg, 'content', '') or ''
                                    if any(kw in content.lower() for kw in
                                           ['error', 'failed', 'exception', 'limit',
                                            'exceeded', 'rejected', 'too large']):
                                        succeeded = False
                                    break

                            if not succeeded:
                                continue

                            args = tc.get('args', {})
                            # Dedup key: variable + rounded region
                            dedup_key = (
                                args.get('variable_id', 'sst'),
                                round(args.get('min_latitude', -90)),
                                round(args.get('max_latitude', 90)),
                                round(args.get('min_longitude', 0)),
                                round(args.get('max_longitude', 360)),
                            )
                            if dedup_key in seen_snippet_keys:
                                continue
                            seen_snippet_keys.add(dedup_key)

                            arraylake_snippets.append(_arraylake_snippet(
                                variable=args.get('variable_id', 'sst'),
                                query_type=args.get('query_type', 'spatial'),
                                start_date=args.get('start_date', ''),
                                end_date=args.get('end_date', ''),
                                min_lat=args.get('min_latitude', -90),
                                max_lat=args.get('max_latitude', 90),
                                min_lon=args.get('min_longitude', 0),
                                max_lon=args.get('max_longitude', 360),
                            ))

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

            # Send Arraylake snippets AFTER response + plots exist in DOM
            for snippet in arraylake_snippets:
                await stream_callback("arraylake_snippet", snippet)

            # Save to memory
            self._conversation.add_message("assistant", response_text)

            return response_text

        except Exception as e:
            # Restore clean message state to prevent corruption on next call
            self._messages = messages_backup
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


def create_session(connection_id: str, api_keys: Optional[Dict[str, str]] = None) -> AgentSession:
    """Create a new session for a connection (reuses if already ready)."""
    if connection_id in _sessions:
        existing = _sessions[connection_id]
        if existing.is_ready():
            logger.info(f"Reusing existing ready session for: {connection_id}")
            return existing
        # Close broken session before replacing
        existing.close()
    session = AgentSession(api_keys=api_keys)
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
    count = len(_sessions)
    for conn_id in list(_sessions.keys()):
        close_session(conn_id)
    logger.info(f"Shutdown {count} sessions")
