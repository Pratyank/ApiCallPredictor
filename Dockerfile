# Multi-stage build for OpenSesame Predictor FastAPI application
# Stage 1: Build stage with all dependencies
FROM python:3.12.3-slim AS builder
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 2: Runtime stage with minimal footprint
FROM python:3.12.3-slim AS runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    OPENSESAME_ENVIRONMENT=production \
    OPENSESAME_API_HOST=0.0.0.0 \
    OPENSESAME_API_PORT=8000

# Install minimal runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY .env ./

# Create necessary directories and set permissions
RUN mkdir -p /app/data/training_data /app/tests /app/logs && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Create SQLite database directory with proper permissions
RUN mkdir -p /app/database && \
    chown appuser:appuser /app/database && \
    chmod 755 /app/database

# Switch to non-root user
USER appuser

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command with production-optimized settings
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--access-log", \
     "--log-level", "info", \
     "--no-server-header"]

# Labels for container metadata
LABEL maintainer="OpenSesame Team" \
      version="1.0.0" \
      description="OpenSesame Predictor - AI-powered API call prediction service" \
      python.version="3.9" \
      framework="FastAPI"

# Optional: Multi-architecture build support
# Add --platform flag when building for different architectures:
# docker buildx build --platform linux/amd64,linux/arm64 -t opensesame-predictor .
