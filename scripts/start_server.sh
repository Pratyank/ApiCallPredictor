#!/bin/bash
# Phase 2 AI Layer Server Startup Script
# Ensures environment variables are loaded properly

set -e

# Change to project directory
cd "$(dirname "$0")/.."

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: No .env file found!"
fi

# Verify critical environment variables
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set in environment"
    exit 1
fi

echo "Starting OpenSesame Predictor with Phase 2 AI Layer..."
echo "Anthropic API Key: ${ANTHROPIC_API_KEY:0:20}..."
echo "Environment loaded successfully"

# Kill any existing processes on port 8000
echo "Checking for existing processes on port 8000..."
if lsof -i :8000 > /dev/null 2>&1; then
    echo "Killing existing processes on port 8000..."
    pkill -f "uvicorn.*8000" || true
    sleep 2
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the server with explicit environment variable passing
echo "Starting FastAPI server..."
ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload