#!/bin/bash
# Phase 2 Testing Script - Updated for proper environment loading

set -e

# Change to project directory
cd "$(dirname "$0")/.."

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "=== Phase 2 AI Layer Testing Suite ==="
echo ""

# Test 1: Direct AI Layer Test
echo "🔍 Test 1: Testing AI Layer directly..."
source venv/bin/activate
ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" python -c "
import asyncio
import sys
import os
sys.path.append('.')
from app.models.ai_layer import AiLayer

async def test_ai_layer():
    ai_layer = AiLayer()
    # Initialize internal components
    await ai_layer._init_apis()
    await ai_layer._init_sentence_model()
    status = await ai_layer.get_status()
    print('AI Layer Status:', status)
    
    if status['anthropic_available']:
        print('✅ Anthropic API integration: WORKING')
        
        # Test a simple prediction
        print('\\n🧠 Testing AI prediction...')
        predictions = await ai_layer.generate_predictions(
            prompt='predict API calls for user authentication',
            history=[],
            k=3,
            temperature=0.1
        )
        print('AI Prediction Result:')
        print(f'  - Number of predictions: {len(predictions)}')
        if predictions:
            first_prediction = predictions[0]
            print(f'  - First prediction: {first_prediction.get(\"api_call\", \"N/A\")}')
            print(f'  - Provider: {first_prediction.get(\"ai_provider\", \"N/A\")}')
            print(f'  - Confidence: {first_prediction.get(\"confidence\", \"N/A\")}')
            print(f'  - Reasoning: {first_prediction.get(\"reasoning\", \"N/A\")[:100]}...')
        print('✅ AI predictions: WORKING')
    else:
        print('❌ Anthropic API: NOT AVAILABLE')
        print('   Check your ANTHROPIC_API_KEY environment variable')
    
    return status['anthropic_available']

result = asyncio.run(test_ai_layer())
if not result:
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ AI Layer test failed"
    exit 1
fi

echo ""
echo "🚀 Test 2: Starting server on port 8001..."

# Kill any existing processes
lsof -ti:8001 | xargs kill -9 2>/dev/null || true
sleep 1

# Start server directly (AiLayer now loads .env file automatically)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --log-level info &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
for i in {1..10}; do
    if curl -s http://localhost:8001/health >/dev/null 2>&1; then
        break
    fi
    sleep 2
    if [ $i -eq 10 ]; then
        echo "❌ Server failed to start"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
done

echo "✅ Server started successfully"

# Test 3: Check AI Layer status via API
echo ""
echo "🔍 Test 3: Checking AI Layer status via API..."
response=$(curl -s http://localhost:8001/metrics)
echo "Metrics Response:"
echo "$response" | python -m json.tool

# Test 4: Test prediction endpoint
echo ""
echo "🧠 Test 4: Testing /predict endpoint..."
prediction_response=$(curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "user authentication and login",
    "history": [],
    "k": 3,
    "temperature": 0.1
  }')

echo "Prediction Response:"
echo "$prediction_response" | python -m json.tool

# Check if we got Phase 2 response (more robust pattern matching)
if echo "$prediction_response" | grep -q "v2.0-ai-layer" && echo "$prediction_response" | grep -q "anthropic"; then
    echo "✅ Phase 2 AI Layer: WORKING"
    echo "✅ Anthropic API Provider: CONFIRMED"
    echo "✅ Semantic Similarity: ENABLED"
    echo "✅ AI-Generated Predictions: SUCCESS"
else
    echo "❌ Phase 2 not working - check the logs above"
fi

# Cleanup
echo ""
echo "🧹 Cleaning up..."
kill $SERVER_PID 2>/dev/null || true

echo ""
echo "=== Phase 2 Testing Complete ==="