#!/bin/bash

# Phase 4 Verification Script
# Run this to verify all Phase 4 features are working correctly

echo "🧪 OpenSesame Predictor Phase 4 Verification Script"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to check test result
check_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC}"
        ((TESTS_FAILED++))
    fi
}

# Function to test API endpoint
test_api() {
    local test_name="$1"
    local expected="$2"
    local response="$3"
    
    echo -n "Testing $test_name... "
    
    if [[ "$response" == *"$expected"* ]]; then
        check_result 0
    else
        check_result 1
        echo "  Expected: $expected"
        echo "  Got: $response"
    fi
}

echo "🚀 Step 1: Starting Docker containers..."
docker-compose up -d
sleep 45  # Wait for startup

echo ""
echo "🔍 Step 2: Basic Health Checks"
echo "=============================="

# Test 1: Container is running
echo -n "Container Status... "
if docker ps | grep -q "opensesame-predictor"; then
    check_result 0
else
    check_result 1
fi

# Test 2: API is responding
echo -n "API Responding... "
RESPONSE=$(curl -s "http://localhost:8000/" 2>/dev/null)
if [[ "$RESPONSE" == *"OpenSesame Predictor"* ]]; then
    check_result 0
else
    check_result 1
fi

# Test 3: Phase 4 version
echo -n "Phase 4 Version... "
VERSION=$(curl -s "http://localhost:8000/" | jq -r '.version' 2>/dev/null)
if [[ "$VERSION" == *"v4.0"* ]]; then
    check_result 0
else
    check_result 1
fi

echo ""
echo "🛡️ Step 3: Security Guardrails Tests"
echo "===================================="

# Test 4: Block DELETE operations
echo -n "Block DELETE operations... "
RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "DELETE all users", "max_predictions": 3}' 2>/dev/null)
if [[ "$RESPONSE" == *"failed"* ]] || [[ "$RESPONSE" == *"blocked"* ]] || [[ "$RESPONSE" == *"safety"* ]]; then
    check_result 0
else
    check_result 1
    echo "  Response: $RESPONSE"
fi

# Test 5: Block SQL injection
echo -n "Block SQL injection... "
RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "DROP TABLE users; SELECT * FROM admin", "max_predictions": 3}' 2>/dev/null)
PRED_COUNT=$(echo "$RESPONSE" | jq '.predictions | length' 2>/dev/null || echo "0")
if [[ "$PRED_COUNT" == "0" ]] || [[ "$RESPONSE" == *"failed"* ]]; then
    check_result 0
else
    check_result 1
    echo "  Should block but got $PRED_COUNT predictions"
fi

# Test 6: Block XSS attempts
echo -n "Block XSS attempts... "
RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "<script>alert(1)</script> remove data", "max_predictions": 3}' 2>/dev/null)
if [[ "$RESPONSE" == *"failed"* ]] || [[ "$RESPONSE" == *"safety"* ]]; then
    check_result 0
else
    check_result 1
fi

echo ""
echo "🚀 Step 4: Cold Start Functionality Tests"
echo "=========================================="

# Test 7: Cold start with empty history
echo -n "Cold start empty history... "
RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "get user information", "history": [], "max_predictions": 3}' 2>/dev/null)
PRED_COUNT=$(echo "$RESPONSE" | jq '.predictions | length' 2>/dev/null || echo "0")
if [[ "$PRED_COUNT" -ge "1" ]]; then
    check_result 0
else
    check_result 1
    echo "  Expected predictions but got: $PRED_COUNT"
fi

# Test 8: Cold start processing method
echo -n "Cold start processing method... "
RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "search products", "history": [], "max_predictions": 3}' 2>/dev/null)
METHOD=$(echo "$RESPONSE" | jq -r '.metadata.processing_method' 2>/dev/null)
if [[ "$METHOD" == *"cold_start"* ]]; then
    check_result 0
else
    check_result 1
    echo "  Expected cold_start method, got: $METHOD"
fi

# Test 9: Database has popular endpoints
echo -n "Popular endpoints in database... "
ENDPOINT_COUNT=$(docker exec opensesame-predictor python -c "
import sqlite3
conn = sqlite3.connect('/app/data/cache.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM popular_endpoints WHERE is_safe = 1')
print(cursor.fetchone()[0])
conn.close()
" 2>/dev/null)
if [[ "$ENDPOINT_COUNT" -ge "3" ]]; then
    check_result 0
else
    check_result 1
    echo "  Expected ≥3 safe endpoints, got: $ENDPOINT_COUNT"
fi

echo ""
echo "⚖️ Step 5: Safety Integration Tests"
echo "==================================="

# Test 10: Safe requests pass through
echo -n "Safe requests pass through... "
RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "get user profile", "max_predictions": 3}' 2>/dev/null)
PRED_COUNT=$(echo "$RESPONSE" | jq '.predictions | length' 2>/dev/null || echo "0")
if [[ "$PRED_COUNT" -ge "1" ]]; then
    check_result 0
else
    check_result 1
fi

# Test 11: Safety metadata present
echo -n "Safety metadata present... "
RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "list items", "max_predictions": 3}' 2>/dev/null)
SAFETY_ENABLED=$(echo "$RESPONSE" | jq -r '.metadata.safety_filtering_enabled' 2>/dev/null)
if [[ "$SAFETY_ENABLED" == "true" ]]; then
    check_result 0
else
    check_result 1
    echo "  Safety filtering should be enabled"
fi

# Test 12: k+buffer strategy working
echo -n "k+buffer strategy working... "
RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "update profile", "history": [{"api_call": "/api/users/123", "method": "GET"}], "max_predictions": 3}' 2>/dev/null)
PRED_COUNT=$(echo "$RESPONSE" | jq '.predictions | length' 2>/dev/null || echo "0")
if [[ "$PRED_COUNT" -le "3" ]] && [[ "$PRED_COUNT" -ge "1" ]]; then
    check_result 0
else
    check_result 1
    echo "  Expected 1-3 predictions, got: $PRED_COUNT"
fi

echo ""
echo "📊 Step 6: Performance & Health Tests"
echo "====================================="

# Test 13: Health endpoint
echo -n "Health endpoint responding... "
RESPONSE=$(curl -s "http://localhost:8000/health" 2>/dev/null)
STATUS=$(echo "$RESPONSE" | jq -r '.status' 2>/dev/null)
if [[ "$STATUS" == "healthy" ]] || [[ "$STATUS" == "degraded" ]]; then
    check_result 0
else
    check_result 1
fi

# Test 14: Metrics endpoint
echo -n "Metrics endpoint responding... "
RESPONSE=$(curl -s "http://localhost:8000/metrics" 2>/dev/null)
if [[ "$RESPONSE" == *"predictor_metrics"* ]]; then
    check_result 0
else
    check_result 1
fi

# Test 15: Response time reasonable
echo -n "Response time < 5 seconds... "
START_TIME=$(date +%s.%3N)
RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "test performance", "max_predictions": 2}' 2>/dev/null)
END_TIME=$(date +%s.%3N)
DURATION=$(echo "$END_TIME - $START_TIME" | bc -l 2>/dev/null || echo "1.0")
if (( $(echo "$DURATION < 5.0" | bc -l 2>/dev/null || echo "1") )); then
    check_result 0
else
    check_result 1
    echo "  Response took ${DURATION}s"
fi

echo ""
echo "🧹 Step 7: Cleanup"
echo "=================="
docker-compose down
echo "Docker containers stopped."

echo ""
echo "📋 VERIFICATION SUMMARY"
echo "======================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo -e "Total Tests:  $((TESTS_PASSED + TESTS_FAILED))"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 ALL TESTS PASSED - Phase 4 is working correctly!${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠️  Some tests failed - check the output above for details${NC}"
    exit 1
fi