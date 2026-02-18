"""
REST API Routes
===============
Health checks, cache management, and configuration endpoints.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Add project root and src/ to path for eurus package
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# IMPORT FROM EURUS PACKAGE
from eurus.config import CONFIG, ERA5_VARIABLES, GEOGRAPHIC_REGIONS

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    agent_ready: bool


class DatasetInfo(BaseModel):
    variable: str
    query_type: str
    start_date: str
    end_date: str
    lat_bounds: tuple
    lon_bounds: tuple
    file_size_bytes: int
    path: str


class CacheResponse(BaseModel):
    datasets: List[Dict[str, Any]]
    total_size_bytes: int


class ConfigResponse(BaseModel):
    variables: List[Dict[str, str]]
    regions: List[str]
    model: str


@router.get("/keys-status")
async def keys_status():
    """Check which API keys are configured via environment variables."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "arraylake": bool(os.environ.get("ARRAYLAKE_API_KEY")),
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the server and agent are healthy."""
    from web.agent_wrapper import get_agent_session

    try:
        session = get_agent_session()
        agent_ready = session is not None and session.is_ready()
    except Exception:
        agent_ready = False

    return HealthResponse(
        status="ok",
        version="1.0.0",
        agent_ready=agent_ready
    )


@router.get("/cache", response_model=CacheResponse)
async def list_cache():
    """List all cached datasets."""
    from eurus.memory import get_memory

    memory = get_memory()
    datasets = []
    total_size = 0

    for path, record in memory.datasets.items():
        if os.path.exists(path):
            size = record.file_size_bytes
            if size == 0:
                # Calculate size if not recorded
                if os.path.isdir(path):
                    size = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, files in os.walk(path)
                        for f in files
                    )
                else:
                    size = os.path.getsize(path)

            datasets.append({
                "variable": record.variable,
                "query_type": record.query_type,
                "start_date": record.start_date,
                "end_date": record.end_date,
                "lat_bounds": record.lat_bounds,
                "lon_bounds": record.lon_bounds,
                "file_size_bytes": size,
                "path": path
            })
            total_size += size

    return CacheResponse(datasets=datasets, total_size_bytes=total_size)


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get available variables and regions."""
    # Get unique variables
    seen_vars = set()
    variables = []
    for var_id, var_info in ERA5_VARIABLES.items():
        if var_info.short_name not in seen_vars:
            seen_vars.add(var_info.short_name)
            variables.append({
                "name": var_info.short_name,
                "long_name": var_info.long_name,
                "units": var_info.units,
                "description": var_info.description
            })

    regions = list(GEOGRAPHIC_REGIONS.keys())

    return ConfigResponse(
        variables=variables,
        regions=regions,
        model=CONFIG.model_name
    )


@router.delete("/conversation")
async def clear_conversation():
    """Clear the conversation history."""
    from eurus.memory import get_memory
    from web.agent_wrapper import get_agent_session

    memory = get_memory()
    memory.clear_conversation()

    # Also clear the agent session messages
    session = get_agent_session()
    if session:
        session.clear_messages()

    return {"status": "ok", "message": "Conversation cleared"}


@router.get("/memory")
async def get_memory_summary():
    """Get memory summary."""
    from eurus.memory import get_memory

    memory = get_memory()

    return {
        "conversation_count": len(memory.conversations),
        "dataset_count": len([p for p in memory.datasets if os.path.exists(p)]),
        "analysis_count": len(memory.analyses),
        "context_summary": memory.get_context_summary()
    }
