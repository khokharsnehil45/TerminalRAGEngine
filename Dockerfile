# ==========================================
# TRAG Production Containerfile (Multi-Stage)
# ==========================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8450

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure data and db directories exist
RUN mkdir -p /app/data /root/.local/share

# Expose Web GUI & REST API port
EXPOSE 8450

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8450/healthz || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8450", "--workers", "2"]
