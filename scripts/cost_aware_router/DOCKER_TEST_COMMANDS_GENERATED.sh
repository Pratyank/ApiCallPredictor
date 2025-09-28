#!/bin/bash
# Docker Test Commands for Cost-Aware Router


# 1. Build Docker image
docker build -t opensesame-predictor .

# 2. Run container with cost router enabled
docker run -d \
  --name opensesame-test \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-test \
  -e COST_ROUTER_ENABLED=true \
  -e DAILY_BUDGET_LIMIT=100.0 \
  opensesame-predictor

# 3. Wait for startup (30 seconds)
sleep 30

# 4. Test health endpoint
curl -f http://localhost:8000/health

# 5. Test prediction with cost-aware routing
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Get user information",
    "history": [],
    "max_predictions": 3
  }'

# 6. Test cost router directly in container
docker exec opensesame-test python -c "
from app.models.cost_aware_router import CostAwareRouter
router = CostAwareRouter()
print('Models:', list(router.models.keys()))
result = router.route(0.3, 2.0)
print('Simple routing:', result['selected_model'])
result = router.route(0.8, 20.0)
print('Complex routing:', result['selected_model'])
"

# 7. Check container logs for cost router activity
docker logs opensesame-test | grep -i cost

# 8. Clean up
docker stop opensesame-test && docker rm opensesame-test

echo "✅ Docker tests completed"
