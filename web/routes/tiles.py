"""
Tile proxy routes — forwards requests to xpublish-tiles server.
Avoids CORS issues when tile server runs on a different port.
"""

import logging
from fastapi import APIRouter, Request, Response

router = APIRouter()
logger = logging.getLogger(__name__)

TILE_SERVER_URL = "http://localhost:8080"


@router.get("/tiles/{path:path}")
async def proxy_tiles(path: str, request: Request):
    """Proxy tile requests to xpublish-tiles server."""
    import httpx

    query_string = str(request.query_params)
    target_url = f"{TILE_SERVER_URL}/{path}"
    if query_string:
        target_url += f"?{query_string}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(target_url)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "image/png"),
        )
    except httpx.ConnectError:
        return Response(
            content=b'{"error": "Tile server not running. Start with: python scripts/tile_server.py"}',
            status_code=503,
            media_type="application/json",
        )
    except Exception as e:
        logger.error(f"Tile proxy error: {e}")
        return Response(
            content=f'{{"error": "{str(e)}"}}'.encode(),
            status_code=500,
            media_type="application/json",
        )


@router.get("/tiles-status")
async def tile_server_status():
    """Check if tile server is running."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{TILE_SERVER_URL}/docs")
        return {"status": "online", "url": TILE_SERVER_URL}
    except Exception:
        return {"status": "offline", "url": TILE_SERVER_URL}
