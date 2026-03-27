# ============================================================================
# Eurus ERA5 Agent — Docker Image
# ============================================================================
# Single-stage build for HuggingFace Spaces + local docker-compose.
#
# Local usage:
#   docker build -t eurus-web .
#   docker run -p 7860:7860 --env-file .env eurus-web
#
# Or use docker-compose:
#   docker compose up web
# ============================================================================

FROM python:3.12-slim

# System deps for scientific stack (numpy/scipy wheels, geopandas/shapely, matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libgeos-dev \
    libproj-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 for React frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build React frontend (separate layer for caching)
COPY frontend/package.json frontend/package-lock.json* frontend/
RUN cd frontend && npm ci
COPY frontend/ frontend/
RUN cd frontend && npm run build

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

# Pre-download Natural Earth data for Cartopy coastlines
RUN python -c "import cartopy; cartopy.io.shapereader.natural_earth(resolution='110m', category='physical', name='coastline')" \
    && python -c "import cartopy; cartopy.io.shapereader.natural_earth(resolution='50m', category='physical', name='coastline')" \
    && python -c "import cartopy; cartopy.io.shapereader.natural_earth(resolution='110m', category='physical', name='land')" \
    && python -c "import cartopy; cartopy.io.shapereader.natural_earth(resolution='50m', category='physical', name='land')"

# Pre-import heavy modules at BUILD time to force compilation / bytecode caching.
# This avoids a 30+ minute first-import delay on HF Spaces' constrained free tier.
RUN python -c "\
import langchain; \
import langchain_openai; \
import arraylake; \
import icechunk; \
import xarray; \
import scipy; \
import matplotlib; \
import scgraph; \
print('All heavy modules pre-imported successfully')"

# Matplotlib: no GUI backend
ENV MPLBACKEND=Agg
# Ensure Python output is unbuffered (for docker logs)
ENV PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
