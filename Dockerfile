# Multi-stage build for OpenSesame Predictor FastAPI application
# Stage 1: Build stage with all dependencies
FROM python:3.12.3-slim AS builder
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building (including ML dependencies)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    cmake \
    libgomp1 \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    # Pre-download sentence-transformers model to reduce startup time
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2: Runtime stage with minimal footprint
FROM python:3.12.3-slim AS runtime

# Set environment variables (including ML Layer variables)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    OPENSESAME_ENVIRONMENT=production \
    OPENSESAME_API_HOST=0.0.0.0 \
    OPENSESAME_API_PORT=8000 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Install minimal runtime system dependencies (including ML runtime deps)
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    libgomp1 \
    libopenblas-dev \
    libomp5 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment and ML model cache from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache /app/.cache

# Create application directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY .env ./

# Create necessary directories and set permissions (including ML cache)
RUN mkdir -p /app/data/training_data /app/tests /app/logs /app/.cache/huggingface /app/.cache/sentence_transformers && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Create database directory with proper permissions (Phase 3 uses /app/data/)
RUN mkdir -p /app/data && \
    chown appuser:appuser /app/data && \
    chmod 755 /app/data

# Switch to non-root user
USER appuser

# Health check endpoint (extended timeout for ML processing)
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command with production-optimized settings (reduced workers for ML)
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--access-log", \
     "--log-level", "info", \
     "--no-server-header", \
     "--timeout-keep-alive", "30"]

# Labels for container metadata (updated for Phase 3)
LABEL maintainer="OpenSesame Team" \
      version="3.0.0" \
      description="OpenSesame Predictor - Phase 3 ML Layer with LightGBM ranking" \
      python.version="3.12" \
      framework="FastAPI" \
      ml.libraries="LightGBM,sentence-transformers" \
      phase="3-ml-layer"

# Optional: Multi-architecture build support
# Add --platform flag when building for different architectures:
# docker buildx build --platform linux/amd64,linux/arm64 -t opensesame-predictor .
