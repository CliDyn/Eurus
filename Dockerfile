# ============================================================================
# Eurus ERA5 Agent — Docker Image
# ============================================================================
# Multi-target build:
#   docker build --target agent -t eurus-agent .
#   docker build --target web   -t eurus-web   .
#
# Or use docker-compose (preferred):
#   docker compose run --rm agent     # interactive CLI
#   docker compose up web             # FastAPI on :8000
# ============================================================================

# ---------- base ----------
FROM python:3.12-slim AS base

# System deps for scientific stack (numpy/scipy wheels, geopandas/shapely, matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
        libgeos-dev \
        libproj-dev \
        libffi-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY pyproject.toml .
COPY src/ src/
COPY main.py .
COPY web/ web/
COPY tests/ tests/
COPY scripts/ scripts/
COPY assets/ assets/
COPY README.md LICENSE ./

# Install eurus package in editable mode
RUN pip install --no-cache-dir -e ".[agent,web]"

# Create dirs the agent expects
RUN mkdir -p /app/data/plots /app/.memory /app/logs

# Signal to the REPL that we're inside Docker → security checks disabled
ENV EURUS_DOCKER=1
# Matplotlib: no GUI backend
ENV MPLBACKEND=Agg
# Ensure Python output is unbuffered (for docker logs)
ENV PYTHONUNBUFFERED=1

# ---------- agent (CLI mode) ----------
FROM base AS agent
ENTRYPOINT ["python", "main.py"]

# ---------- web (FastAPI mode) ----------
FROM base AS web
EXPOSE 8000
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
