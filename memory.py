"""
ERA5 Agent Memory Module
=========================
Persistent memory for conversation history, data cache, and agent knowledge.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from collections import deque

from config import MEMORY_DIR, CACHE_DIR, DATA_DIR, CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Message:
    """A conversation message."""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        return cls(**data)

    def to_langchain(self) -> dict:
        """Convert to LangChain message format."""
        return {"role": self.role, "content": self.content}


@dataclass
class DatasetRecord:
    """Record of a downloaded dataset."""
    path: str
    variable: str
    query_type: str
    start_date: str
    end_date: str
    lat_bounds: tuple
    lon_bounds: tuple
    file_size_bytes: int
    download_timestamp: str
    shape: Optional[tuple] = None
    checksum: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'DatasetRecord':
        # Convert list to tuple for lat/lon bounds
        if isinstance(data.get('lat_bounds'), list):
            data['lat_bounds'] = tuple(data['lat_bounds'])
        if isinstance(data.get('lon_bounds'), list):
            data['lon_bounds'] = tuple(data['lon_bounds'])
        if isinstance(data.get('shape'), list):
            data['shape'] = tuple(data['shape'])
        return cls(**data)


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
    def from_dict(cls, data: dict) -> 'AnalysisRecord':
        return cls(**data)


# ============================================================================
# MEMORY MANAGER
# ============================================================================

class MemoryManager:
    """
    Manages persistent memory for the ERA5 Agent.

    Features:
    - Conversation history with timestamps
    - Dataset cache registry
    - Analysis history
    - Knowledge base (learned facts)
    """

    def __init__(self, memory_dir: Path = MEMORY_DIR):
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.conversation_file = self.memory_dir / "conversations.json"
        self.datasets_file = self.memory_dir / "datasets.json"
        self.analyses_file = self.memory_dir / "analyses.json"
        self.knowledge_file = self.memory_dir / "knowledge.json"

        # In-memory structures
        self.conversations: deque = deque(maxlen=CONFIG.max_conversation_history)
        self.datasets: Dict[str, DatasetRecord] = {}
        self.analyses: List[AnalysisRecord] = []
        self.knowledge: Dict[str, Any] = {}

        # Load existing data
        self._load_all()

        logger.info(f"MemoryManager initialized: {len(self.conversations)} messages, {len(self.datasets)} datasets")

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _load_all(self):
        """Load all memory from disk."""
        self._load_conversations()
        self._load_datasets()
        self._load_analyses()
        self._load_knowledge()

    def _save_all(self):
        """Save all memory to disk."""
        self._save_conversations()
        self._save_datasets()
        self._save_analyses()
        self._save_knowledge()

    def _load_conversations(self):
        """Load conversation history."""
        if self.conversation_file.exists():
            try:
                with open(self.conversation_file, 'r') as f:
                    data = json.load(f)
                    for msg_data in data:
                        self.conversations.append(Message.from_dict(msg_data))
            except Exception as e:
                logger.warning(f"Failed to load conversations: {e}")

    def _save_conversations(self):
        """Save conversation history."""
        try:
            with open(self.conversation_file, 'w') as f:
                json.dump([m.to_dict() for m in self.conversations], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversations: {e}")

    def _load_datasets(self):
        """Load dataset registry."""
        if self.datasets_file.exists():
            try:
                with open(self.datasets_file, 'r') as f:
                    data = json.load(f)
                    for path, record_data in data.items():
                        self.datasets[path] = DatasetRecord.from_dict(record_data)
            except Exception as e:
                logger.warning(f"Failed to load datasets: {e}")

    def _save_datasets(self):
        """Save dataset registry."""
        try:
            with open(self.datasets_file, 'w') as f:
                json.dump({p: r.to_dict() for p, r in self.datasets.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save datasets: {e}")

    def _load_analyses(self):
        """Load analysis history."""
        if self.analyses_file.exists():
            try:
                with open(self.analyses_file, 'r') as f:
                    data = json.load(f)
                    self.analyses = [AnalysisRecord.from_dict(r) for r in data]
            except Exception as e:
                logger.warning(f"Failed to load analyses: {e}")

    def _save_analyses(self):
        """Save analysis history."""
        try:
            with open(self.analyses_file, 'w') as f:
                json.dump([a.to_dict() for a in self.analyses[-50:]], f, indent=2)  # Keep last 50
        except Exception as e:
            logger.error(f"Failed to save analyses: {e}")

    def _load_knowledge(self):
        """Load knowledge base."""
        if self.knowledge_file.exists():
            try:
                with open(self.knowledge_file, 'r') as f:
                    self.knowledge = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load knowledge: {e}")

    def _save_knowledge(self):
        """Save knowledge base."""
        try:
            with open(self.knowledge_file, 'w') as f:
                json.dump(self.knowledge, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")

    # ========================================================================
    # CONVERSATION MANAGEMENT
    # ========================================================================

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to conversation history."""
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.conversations.append(msg)
        self._save_conversations()
        return msg

    def get_conversation_history(self, n_messages: int = None) -> List[Message]:
        """Get recent conversation history."""
        if n_messages is None:
            return list(self.conversations)
        return list(self.conversations)[-n_messages:]

    def get_langchain_messages(self, n_messages: int = None) -> List[dict]:
        """Get messages in LangChain format."""
        messages = self.get_conversation_history(n_messages)
        return [m.to_langchain() for m in messages]

    def clear_conversation(self):
        """Clear conversation history."""
        self.conversations.clear()
        self._save_conversations()
        logger.info("Conversation history cleared")

    def search_conversations(self, query: str) -> List[Message]:
        """Search conversation history for a query."""
        query_lower = query.lower()
        results = []
        for msg in self.conversations:
            if query_lower in msg.content.lower():
                results.append(msg)
        return results

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
        lat_bounds: tuple,
        lon_bounds: tuple,
        file_size_bytes: int = 0,
        shape: tuple = None
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
            shape=shape
        )
        self.datasets[path] = record
        self._save_datasets()
        logger.info(f"Registered dataset: {path}")
        return record

    def get_dataset(self, path: str) -> Optional[DatasetRecord]:
        """Get dataset record by path."""
        return self.datasets.get(path)

    def find_cached_dataset(
        self,
        variable: str,
        query_type: str,
        start_date: str,
        end_date: str,
        lat_bounds: tuple,
        lon_bounds: tuple,
        tolerance_days: int = 0
    ) -> Optional[DatasetRecord]:
        """Find a cached dataset that matches or covers the request."""
        for path, record in self.datasets.items():
            # Check if file still exists
            if not os.path.exists(path):
                continue

            # Check variable and query type
            if record.variable != variable or record.query_type != query_type:
                continue

            # Check if cached data covers requested bounds (with some flexibility)
            if (record.lat_bounds[0] <= lat_bounds[0] and
                record.lat_bounds[1] >= lat_bounds[1] and
                record.lon_bounds[0] <= lon_bounds[0] and
                record.lon_bounds[1] >= lon_bounds[1] and
                record.start_date <= start_date and
                record.end_date >= end_date):
                return record

        return None

    def list_datasets(self) -> str:
        """Return formatted list of cached datasets."""
        if not self.datasets:
            return "No datasets in cache."

        lines = ["Cached Datasets:", "=" * 70]
        for path, record in self.datasets.items():
            if os.path.exists(path):
                size = os.path.getsize(path) if os.path.isfile(path) else self._get_dir_size(path)
                size_str = self._format_size(size)
                lines.append(
                    f"  {record.variable:5} | {record.start_date} to {record.end_date} | "
                    f"{record.query_type:8} | {size_str:>10}"
                )
                lines.append(f"        Path: {path}")
            else:
                lines.append(f"  [MISSING] {path}")

        return "\n".join(lines)

    def cleanup_missing_datasets(self):
        """Remove records for datasets that no longer exist."""
        missing = [p for p in self.datasets if not os.path.exists(p)]
        for path in missing:
            del self.datasets[path]
            logger.info(f"Removed missing dataset record: {path}")
        self._save_datasets()
        return len(missing)

    # ========================================================================
    # ANALYSIS TRACKING
    # ========================================================================

    def record_analysis(
        self,
        description: str,
        code: str,
        output: str,
        datasets_used: List[str] = None,
        plots_generated: List[str] = None
    ):
        """Record an analysis for history."""
        record = AnalysisRecord(
            description=description,
            code=code,
            output=output[:2000],  # Truncate long output
            timestamp=datetime.now().isoformat(),
            datasets_used=datasets_used or [],
            plots_generated=plots_generated or []
        )
        self.analyses.append(record)
        self._save_analyses()
        return record

    def get_recent_analyses(self, n: int = 10) -> List[AnalysisRecord]:
        """Get recent analyses."""
        return self.analyses[-n:]

    # ========================================================================
    # KNOWLEDGE BASE
    # ========================================================================

    def learn(self, key: str, value: Any):
        """Store a piece of knowledge."""
        self.knowledge[key] = {
            "value": value,
            "learned_at": datetime.now().isoformat()
        }
        self._save_knowledge()
        logger.info(f"Learned: {key}")

    def recall(self, key: str) -> Optional[Any]:
        """Recall a piece of knowledge."""
        if key in self.knowledge:
            return self.knowledge[key]["value"]
        return None

    def get_all_knowledge(self) -> Dict[str, Any]:
        """Get all stored knowledge."""
        return {k: v["value"] for k, v in self.knowledge.items()}

    # ========================================================================
    # SUMMARY & CONTEXT
    # ========================================================================

    def get_context_summary(self) -> str:
        """Get a summary of current context for the agent."""
        lines = []

        # Recent conversation summary
        recent = self.get_conversation_history(5)
        if recent:
            lines.append("Recent Conversation:")
            for msg in recent[-3:]:
                content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                lines.append(f"  [{msg.role}]: {content_preview}")

        # Available datasets
        valid_datasets = {p: r for p, r in self.datasets.items() if os.path.exists(p)}
        if valid_datasets:
            lines.append(f"\nCached Datasets ({len(valid_datasets)}):")
            for path, record in list(valid_datasets.items())[:5]:
                lines.append(f"  - {record.variable}: {record.start_date} to {record.end_date}")

        # Recent analyses
        if self.analyses:
            lines.append(f"\nRecent Analyses ({len(self.analyses)} total):")
            for analysis in self.analyses[-2:]:
                lines.append(f"  - {analysis.description[:60]}...")

        return "\n".join(lines) if lines else "No context available."

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format file size."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    @staticmethod
    def _get_dir_size(path: str) -> int:
        """Get total size of a directory."""
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total


# ============================================================================
# GLOBAL MEMORY INSTANCE
# ============================================================================

_memory_instance: Optional[MemoryManager] = None

def get_memory() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = MemoryManager()
    return _memory_instance

def reset_memory():
    """Reset the global memory instance."""
    global _memory_instance
    _memory_instance = None
