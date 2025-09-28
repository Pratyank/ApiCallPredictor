# COST-AWARE ROUTER DOCKER TESTING INSTRUCTIONS

## Overview
This document provides step-by-step instructions for testing the **Cost-Aware Model Router** implementation in Docker. This is part of the OpenSesame Predictor bonus feature that intelligently routes queries between Claude 3 Haiku (cheap) and Claude 3 Opus (premium) based on complexity and budget constraints.

## Feature Description
The Cost-Aware Router automatically selects the optimal Anthropic model:
- **Simple queries** → Claude 3 Haiku ($0.00025/1K tokens, 70% accuracy)
- **Complex queries** → Claude 3 Opus ($0.015/1K tokens, 90% accuracy)  
- **Budget constraints** → Forces cheaper model when budget limits reached

## Docker Testing Instructions


To test the Cost-Aware Router in Docker:

1. Install Docker Desktop:
   - Download from: https://docs.docker.com/desktop/
   - Enable WSL integration in Docker Desktop settings

2. Build the image:
   docker build -t opensesame-predictor .

3. Run the container:
   docker run -d \
     --name opensesame-test \
     -p 8000:8000 \
     -e ANTHROPIC_API_KEY=your_api_key \
     -e COST_ROUTER_ENABLED=true \
     -e DAILY_BUDGET_LIMIT=100.0 \
     opensesame-predictor

4. Test the API:
   # Health check
   curl http://localhost:8000/health
   
   # Test prediction with cost-aware routing
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "I need to get user information",
       "history": [],
       "max_predictions": 3
     }'

5. Check cost router status:
   curl http://localhost:8000/metrics

6. View container logs:
   docker logs opensesame-test

7. Test in container:
   docker exec -it opensesame-test python -c "
   from app.models.cost_aware_router import CostAwareRouter
   router = CostAwareRouter()
   result = router.route(0.5, 5.0)
   print(f'Routing result: {result}')
   "

8. Clean up:
   docker stop opensesame-test
   docker rm opensesame-test

Expected Results:
- Health endpoint returns 200 OK
- Prediction endpoint returns JSON with cost metadata
- Metrics show cost router information
- Logs show cost-aware routing decisions
- Container test shows proper model selection
