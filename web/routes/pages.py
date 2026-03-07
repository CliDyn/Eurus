"""
Page Routes
===========
Serves React frontend build in production, redirects to Vite dev server in development.
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

router = APIRouter()

# React production build directory
BUILD_DIR = Path(__file__).parent.parent.parent / "frontend" / "dist"


@router.get("/")
async def index():
    """Serve React build in production, redirect to Vite in dev."""
    index_file = BUILD_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    # Dev mode fallback: redirect to Vite dev server
    return RedirectResponse(url="http://localhost:5182/")

