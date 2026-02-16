"""
Eurus Web Application
======================
FastAPI application factory and main entry point.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add parent and src directory to path for eurus package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# IMPORT FROM EURUS PACKAGE
from eurus.config import CONFIG, PLOTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting Eurus Web Interface...")
    logger.info(f"Templates: {TEMPLATES_DIR}")
    logger.info(f"Static files: {STATIC_DIR}")
    logger.info(f"Plots directory: {PLOTS_DIR}")

    # Sessions are created per-connection in websocket.py
    logger.info("Ready to accept connections")

    yield

    # Shutdown
    logger.info("Shutting down Eurus Web Interface...")
    from web.agent_wrapper import shutdown_agent_session
    shutdown_agent_session()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Eurus Climate Agent",
        description="Interactive web interface for ERA5 climate data analysis",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Mount static files
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Mount plots directory for serving generated plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")

    # Include routers
    from web.routes import api_router, websocket_router, pages_router

    app.include_router(api_router, prefix="/api", tags=["api"])
    app.include_router(websocket_router, tags=["websocket"])
    app.include_router(pages_router, tags=["pages"])

    return app


# Create the app instance
app = create_app()


def main():
    """Main entry point for running the web server."""
    import uvicorn

    host = getattr(CONFIG, 'web_host', '127.0.0.1')
    port = getattr(CONFIG, 'web_port', 8000)

    print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ██╗   ██╗ ██████╗ ███████╗████████╗ ██████╗ ██╗  ██╗                   ║
║    ██║   ██║██╔═══██╗██╔════╝╚══██╔══╝██╔═══██╗██║ ██╔╝                   ║
║    ██║   ██║██║   ██║███████╗   ██║   ██║   ██║█████╔╝                    ║
║    ╚██╗ ██╔╝██║   ██║╚════██║   ██║   ██║   ██║██╔═██╗                    ║
║     ╚████╔╝ ╚██████╔╝███████║   ██║   ╚██████╔╝██║  ██╗                   ║
║      ╚═══╝   ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝                   ║
║                                                                           ║
║                      Eurus Web Interface v1.0                            ║
║                 ─────────────────────────────────────                     ║
║                                                                           ║
║   Starting server at: http://{host}:{port}                              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

    uvicorn.run(
        "web.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
