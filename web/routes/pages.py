"""
Page Routes
===========
Serves React frontend build in production, redirects to Vite dev server in development.
"""

import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

router = APIRouter()

# React production build directory
BUILD_DIR = Path(__file__).parent.parent.parent / "frontend" / "dist"

# Detect dev mode: EURUS_DEV=1 or no dist folder AND localhost access
_DEV_MODE = os.environ.get("EURUS_DEV") == "1"

# Minimal HTML fallback — returns 200 so HF Spaces health check passes
_FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Eurus</title>
<style>body{font-family:system-ui;display:flex;align-items:center;
justify-content:center;height:100vh;margin:0;background:#0f172a;color:#e2e8f0}
.c{text-align:center}h1{font-size:2rem;margin-bottom:.5rem}
p{opacity:.7}</style></head>
<body><div class="c"><h1>🌊 Eurus</h1><p>Climate Agent is running.
Connect via WebSocket to start a session.</p></div></body></html>"""


@router.get("/{full_path:path}")
async def catch_all(request: Request, full_path: str = ""):
    """Serve React SPA — all non-API routes go to index.html."""
    index_file = BUILD_DIR / "index.html"

    # Production: serve from built React app
    if index_file.exists():
        # Try to serve the exact static file first (e.g. /assets/index-abc.js)
        requested = BUILD_DIR / full_path
        if full_path and requested.is_file() and requested.resolve().is_relative_to(BUILD_DIR.resolve()):
            return FileResponse(requested)
        # Otherwise serve index.html for SPA routing
        return FileResponse(index_file)

    # Dev mode on localhost only — redirect to Vite dev server
    if _DEV_MODE:
        return RedirectResponse(url=f"http://localhost:5182/{full_path}")

    # No React build available — return 200 OK fallback so HF health check passes
    return HTMLResponse(content=_FALLBACK_HTML, status_code=200)
