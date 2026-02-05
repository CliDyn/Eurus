"""
ERA5 MCP Memory System
======================

Session-based memory with smart compression for conversation history.
Dataset cache persists across sessions, but conversations are fresh each session.
"""

from __future__ import annotations

import json
import logging
import os
import tiktoken
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from vostok.config import get_memory_dir, CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Token limits for smart memory management
MAX_CONTEXT_TOKENS = 8000  # Max tokens to keep in active memory
COMPRESSION_THRESHOLD = 6000  # Start compressing when we hit this
SUMMARY_TARGET_TOKENS = 500  # Target tokens for compressed summary


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DatasetRecord:
    """Record of a downloaded dataset."""

    path: str
    variable: str
    query_type: str
    start_date: str
    end_date: str
    lat_bounds: tuple[float, float]
    lon_bounds: tuple[float, float]
    file_size_bytes: int
    download_timestamp: str
    shape: Optional[tuple[int, ...]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetRecord":
        if isinstance(data.get("lat_bounds"), list):
            data["lat_bounds"] = tuple(data["lat_bounds"])
        if isinstance(data.get("lon_bounds"), list):
            data["lon_bounds"] = tuple(data["lon_bounds"])
        if isinstance(data.get("shape"), list):
            data["shape"] = tuple(data["shape"])
        return cls(**data)


@dataclass
class Message:
    """A conversation message."""

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_compressed: bool = False  # Flag for compressed summary messages

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        valid_keys = {'role', 'content', 'timestamp', 'is_compressed'}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_langchain(self) -> dict:
        """Convert to LangChain message format."""
        return {"role": self.role, "content": self.content}


@dataclass
class AnalysisRecord:
    """Record of an analysis performed."""

    description: str
    code: str
    output: str
    timestamp: str
    datasets_used: List[str] = field(default_factory=list)
    plots_generated: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisRecord":
        return cls(**data)


# ============================================================================
# TOKEN COUNTER
# ============================================================================

class TokenCounter:
    """Efficient token counting using tiktoken."""
    
    _encoder = None
    
    @classmethod
    def get_encoder(cls):
        if cls._encoder is None:
            try:
                cls._encoder = tiktoken.encoding_for_model("gpt-4")
            except Exception:
                cls._encoder = tiktoken.get_encoding("cl100k_base")
        return cls._encoder
    
    @classmethod
    def count(cls, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(cls.get_encoder().encode(text))
        except Exception:
            # Fallback: rough estimate
            return len(text) // 4


# ============================================================================
# SMART CONVERSATION MEMORY
# ============================================================================

class SmartConversationMemory:
    """
    Session-based conversation memory with smart compression.
    
    Features:
    - Fresh start each session (no persistent history)
    - Automatic compression when context gets too long
    - Preserves recent messages in full, compresses older ones
    - Token-aware memory management
    """
    
    def __init__(self):
        self.messages: List[Message] = []
        self.compressed_summary: Optional[str] = None
        self._token_count = 0
        logger.info("SmartConversationMemory initialized (fresh session)")
    
    def add_message(self, role: str, content: str) -> Message:
        """Add a message and check if compression is needed."""
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        
        # Update token count
        self._token_count += TokenCounter.count(content)
        
        # Check if we need to compress
        if self._token_count > COMPRESSION_THRESHOLD:
            self._compress_history()
        
        return msg
    
    def _compress_history(self) -> None:
        """Compress older messages into a summary."""
        if len(self.messages) < 6:
            return  # Not enough messages to compress
        
        # Keep the last 4 messages in full
        keep_count = 4
        to_compress = self.messages[:-keep_count]
        to_keep = self.messages[-keep_count:]
        
        if not to_compress:
            return
        
        # Create a concise summary of compressed messages
        summary_parts = []
        for msg in to_compress:
            role = msg.role.upper()
            # Truncate long content for summary
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            summary_parts.append(f"[{role}]: {content}")
        
        summary = "[Previous conversation summary]\n" + "\n".join(summary_parts)
        
        # Replace old messages with summary + recent
        summary_msg = Message(
            role="system",
            content=summary[:SUMMARY_TARGET_TOKENS * 4],  # Rough limit
            is_compressed=True
        )
        
        self.messages = [summary_msg] + to_keep
        
        # Recalculate token count
        self._token_count = sum(
            TokenCounter.count(m.content) for m in self.messages
        )
        
        logger.info(f"Compressed {len(to_compress)} messages. Current tokens: {self._token_count}")
    
    def get_messages(self, n_messages: Optional[int] = None) -> List[Message]:
        """Get conversation messages."""
        if n_messages is None:
            return list(self.messages)
        return list(self.messages)[-n_messages:]
    
    def get_langchain_messages(self, n_messages: Optional[int] = None) -> List[dict]:
        """Get messages in LangChain format."""
        messages = self.get_messages(n_messages)
        return [m.to_langchain() for m in messages]
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self.compressed_summary = None
        self._token_count = 0
        logger.info("Conversation memory cleared")
    
    def get_token_count(self) -> int:
        """Get current token count."""
        return self._token_count


# ============================================================================
# MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """
    Manages memory for ERA5 MCP.

    Features:
    - Dataset cache registry (persists across sessions)
    - Session-based conversation history (fresh each restart)
    - Smart compression for long conversations
    - NO persistent conversation history to avoid stale context
    """

    def __init__(self, memory_dir: Optional[Path] = None, persist_conversations: bool = False):
        self.memory_dir = memory_dir or get_memory_dir()
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.persist_conversations = persist_conversations

        # File paths (only datasets persist)
        self.datasets_file = self.memory_dir / "datasets.json"
        self.analyses_file = self.memory_dir / "analyses.json"

        # In-memory storage
        self.datasets: Dict[str, DatasetRecord] = {}
        self.analyses: List[AnalysisRecord] = []
        
        # Session-based conversation memory (FRESH each time!)
        self.conversation_memory = SmartConversationMemory()

        # Load persistent data (only datasets)
        self._load_datasets()
        self._load_analyses()

        logger.info(
            f"MemoryManager initialized: {len(self.datasets)} datasets, "
            f"FRESH conversation (session-based)"
        )

    # ========================================================================
    # PERSISTENCE (Datasets only)
    # ========================================================================

    def _load_datasets(self) -> None:
        """Load dataset registry from disk."""
        if self.datasets_file.exists():
            try:
                with open(self.datasets_file, "r") as f:
                    data = json.load(f)
                    for path, record_data in data.items():
                        self.datasets[path] = DatasetRecord.from_dict(record_data)
            except Exception as e:
                logger.warning(f"Failed to load datasets: {e}")

    def _save_datasets(self) -> None:
        """Save dataset registry to disk."""
        try:
            with open(self.datasets_file, "w") as f:
                json.dump({p: r.to_dict() for p, r in self.datasets.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save datasets: {e}")

    def _load_analyses(self) -> None:
        """Load analysis history from disk."""
        if self.analyses_file.exists():
            try:
                with open(self.analyses_file, "r") as f:
                    data = json.load(f)
                    self.analyses = [AnalysisRecord.from_dict(r) for r in data[-20:]]  # Keep last 20
            except Exception as e:
                logger.warning(f"Failed to load analyses: {e}")

    def _save_analyses(self) -> None:
        """Save analysis history to disk."""
        try:
            with open(self.analyses_file, "w") as f:
                json.dump([a.to_dict() for a in self.analyses[-20:]], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save analyses: {e}")

    # ========================================================================
    # DATASET MANAGEMENT
    # ========================================================================

    def register_dataset(
        self,
        path: str,
        variable: str,
        query_type: str,
        start_date: str,
        end_date: str,
        lat_bounds: tuple[float, float],
        lon_bounds: tuple[float, float],
        file_size_bytes: int = 0,
        shape: Optional[tuple[int, ...]] = None,
    ) -> DatasetRecord:
        """Register a downloaded dataset."""
        record = DatasetRecord(
            path=path,
            variable=variable,
            query_type=query_type,
            start_date=start_date,
            end_date=end_date,
            lat_bounds=lat_bounds,
            lon_bounds=lon_bounds,
            file_size_bytes=file_size_bytes,
            download_timestamp=datetime.now().isoformat(),
            shape=shape,
        )
        self.datasets[path] = record
        self._save_datasets()
        logger.info(f"Registered dataset: {path}")
        return record

    def get_dataset(self, path: str) -> Optional[DatasetRecord]:
        """Get dataset record by path."""
        return self.datasets.get(path)

    def list_datasets(self) -> str:
        """Return formatted list of cached datasets."""
        if not self.datasets:
            return "No datasets in cache."

        lines = ["Cached Datasets:", "=" * 70]
        for path, record in self.datasets.items():
            if os.path.exists(path):
                size_str = self._format_size(record.file_size_bytes)
                lines.append(
                    f"  {record.variable:5} | {record.start_date} to {record.end_date} | "
                    f"{record.query_type:8} | {size_str:>10}"
                )
                lines.append(f"        Path: {path}")
            else:
                lines.append(f"  [MISSING] {path}")

        return "\n".join(lines)

    def cleanup_missing_datasets(self) -> int:
        """Remove records for datasets that no longer exist."""
        missing = [p for p in self.datasets if not os.path.exists(p)]
        for path in missing:
            del self.datasets[path]
            logger.info(f"Removed missing dataset: {path}")
        if missing:
            self._save_datasets()
        return len(missing)

    # ========================================================================
    # CONVERSATION MANAGEMENT (Session-based)
    # ========================================================================

    def add_message(self, role: str, content: str) -> Message:
        """Add a message to conversation history."""
        return self.conversation_memory.add_message(role, content)

    def get_conversation_history(self, n_messages: Optional[int] = None) -> List[Message]:
        """Get recent conversation history."""
        return self.conversation_memory.get_messages(n_messages)

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_memory.clear()
        logger.info("Conversation history cleared")

    def get_langchain_messages(self, n_messages: Optional[int] = None) -> List[dict]:
        """Get messages in LangChain format."""
        return self.conversation_memory.get_langchain_messages(n_messages)

    # Legacy property for compatibility
    @property
    def conversations(self) -> List[Message]:
        return self.conversation_memory.messages

    # ========================================================================
    # ANALYSIS TRACKING
    # ========================================================================

    def record_analysis(
        self,
        description: str,
        code: str,
        output: str,
        datasets_used: Optional[List[str]] = None,
        plots_generated: Optional[List[str]] = None,
    ) -> AnalysisRecord:
        """Record an analysis for history."""
        record = AnalysisRecord(
            description=description,
            code=code,
            output=output[:2000],  # Truncate long output
            timestamp=datetime.now().isoformat(),
            datasets_used=datasets_used or [],
            plots_generated=plots_generated or [],
        )
        self.analyses.append(record)
        self._save_analyses()
        return record

    def get_recent_analyses(self, n: int = 10) -> List[AnalysisRecord]:
        """Get recent analyses."""
        return self.analyses[-n:]

    # ========================================================================
    # CONTEXT SUMMARY
    # ========================================================================

    def get_context_summary(self) -> str:
        """Get a summary of current context for the agent."""
        lines = []

        # Token usage
        tokens = self.conversation_memory.get_token_count()
        if tokens > 0:
            lines.append(f"Session tokens: {tokens}/{MAX_CONTEXT_TOKENS}")

        # Recent conversation (brief)
        recent = self.get_conversation_history(3)
        if recent:
            lines.append("\nRecent in this session:")
            for msg in recent:
                preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
                lines.append(f"  [{msg.role}]: {preview}")

        # Available datasets
        valid_datasets = {p: r for p, r in self.datasets.items() if os.path.exists(p)}
        if valid_datasets:
            lines.append(f"\nCached Datasets ({len(valid_datasets)}):")
            for path, record in list(valid_datasets.items())[:5]:
                lines.append(f"  - {record.variable}: {record.start_date} to {record.end_date}")

        return "\n".join(lines) if lines else "Fresh session - no context yet."

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_memory_instance: Optional[MemoryManager] = None


def get_memory() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = MemoryManager()
    return _memory_instance


def reset_memory() -> None:
    """Reset the global memory instance (new session)."""
    global _memory_instance
    _memory_instance = None
    logger.info("Memory reset - next get_memory() will create fresh session")
