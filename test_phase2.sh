#!/bin/bash

echo "🚀 Testing OpenSesame Predictor Phase 2 AI Layer"
echo "================================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if server is running
echo -e "\n${YELLOW}1. Testing server health...${NC}"
curl -s http://localhost:8000/health | jq '.' || echo -e "${RED}❌ Server not running. Start with: uvicorn app.main:app --reload --port 8000${NC}"

# Test metrics endpoint
echo -e "\n${YELLOW}2. Checking AI Layer status...${NC}"
curl -s http://localhost:8000/metrics | jq '.ai_layer_metrics' || echo -e "${RED}❌ Metrics endpoint failed${NC}"

# Test basic prediction
echo -e "\n${YELLOW}3. Testing basic AI prediction...${NC}"
RESPONSE1=$(curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I need to get user information",
    "max_predictions": 2,
    "temperature": 0.7
  }')

echo "$RESPONSE1" | jq '.'

# Check if response has AI Layer features
AI_PROVIDER=$(echo "$RESPONSE1" | jq -r '.metadata.ai_provider // empty')
MODEL_VERSION=$(echo "$RESPONSE1" | jq -r '.metadata.model_version // empty')

if [[ "$AI_PROVIDER" != "" && "$MODEL_VERSION" == "v2.0-ai-layer" ]]; then
    echo -e "${GREEN}✅ AI Layer detected: $AI_PROVIDER${NC}"
else
    echo -e "${RED}❌ AI Layer not working properly${NC}"
fi

# Test with history (context awareness)
echo -e "\n${YELLOW}4. Testing history-based predictions...${NC}"
RESPONSE2=$(curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "update user profile",
    "history": [
      {"api_call": "/api/auth/login", "method": "POST", "timestamp": "2024-01-01T10:00:00Z"},
      {"api_call": "/api/users/123", "method": "GET", "timestamp": "2024-01-01T10:01:00Z"}
    ],
    "max_predictions": 3,
    "temperature": 0.7
  }')

echo "$RESPONSE2" | jq '.'

# Check semantic similarity
SEMANTIC_ENABLED=$(echo "$RESPONSE2" | jq -r '.predictions[0].semantic_similarity_enabled // false')
if [[ "$SEMANTIC_ENABLED" == "true" ]]; then
    echo -e "${GREEN}✅ Semantic similarity enabled${NC}"
else
    echo -e "${YELLOW}⚠️  Semantic similarity not enabled (sentence-transformers may not be loaded)${NC}"
fi

# Test different prompts
echo -e "\n${YELLOW}5. Testing semantic understanding...${NC}"
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "create a new user account",
    "max_predictions": 2
  }' | jq '.predictions[0] | {api_call, confidence, reasoning}'

echo -e "\n${YELLOW}6. Testing database caching...${NC}"
if [ -f "data/cache.db" ]; then
    echo -e "${GREEN}✅ Cache database exists: data/cache.db${NC}"
    # Check if database has tables (requires sqlite3)
    if command -v sqlite3 &> /dev/null; then
        TABLES=$(sqlite3 data/cache.db ".tables" 2>/dev/null || echo "")
        if [[ "$TABLES" == *"parsed_endpoints"* ]]; then
            echo -e "${GREEN}✅ Parsed endpoints table exists${NC}"
            ENDPOINT_COUNT=$(sqlite3 data/cache.db "SELECT COUNT(*) FROM parsed_endpoints;" 2>/dev/null || echo "0")
            echo -e "${GREEN}📊 Cached endpoints: $ENDPOINT_COUNT${NC}"
        else
            echo -e "${YELLOW}⚠️  Parsed endpoints table not found${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  sqlite3 not available to check database contents${NC}"
    fi
else
    echo -e "${RED}❌ Cache database not found${NC}"
fi

echo -e "\n${YELLOW}7. Performance test...${NC}"
START_TIME=$(date +%s%3N)
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "delete user data", "max_predictions": 1}' > /dev/null
END_TIME=$(date +%s%3N)
DURATION=$((END_TIME - START_TIME))
echo -e "${GREEN}⏱️  Response time: ${DURATION}ms${NC}"

if [ $DURATION -lt 2000 ]; then
    echo -e "${GREEN}✅ Performance target met (<2000ms)${NC}"
else
    echo -e "${YELLOW}⚠️  Response time slower than expected${NC}"
fi

echo -e "\n${GREEN}🎉 Phase 2 testing complete!${NC}"
echo -e "\n${YELLOW}📋 What to verify:${NC}"
echo "✅ Server responds to /health and /predict endpoints"
echo "✅ Responses include 'v2.0-ai-layer' model_version"
echo "✅ AI provider shows 'anthropic', 'openai', or 'rule_based'"
echo "✅ Predictions include 'reasoning' field"
echo "✅ Semantic similarity features enabled"
echo "✅ data/cache.db database exists"
echo "✅ Response times under 2 seconds"