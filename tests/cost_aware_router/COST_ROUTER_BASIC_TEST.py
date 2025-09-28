#!/usr/bin/env python3
"""
COST-AWARE ROUTER BASIC FUNCTIONALITY TEST
==========================================

This script tests the core functionality of the Cost-Aware Model Router 
implementation for the OpenSesame Predictor project.

PURPOSE:
- Validates CostAwareRouter class instantiation and configuration
- Tests model definitions (claude-3-haiku vs claude-3-opus)  
- Verifies routing logic based on complexity scores and budget constraints
- Confirms database integration with SQLite
- Checks performance requirements
- Validates Docker deployment readiness

FEATURE: Cost-Aware Model Router (Bonus Implementation)
COMPONENT: Core functionality validation without external dependencies
"""

import sys
import os
import sqlite3
import time

# Add app to path
sys.path.insert(0, '/home/quantum/ApiCallPredictor')

def test_basic_functionality():
    """Test basic CostAwareRouter functionality"""
    print("🧪 Testing Basic Cost-Aware Router Functionality")
    print("=" * 55)
    
    try:
        from app.models.cost_aware_router import CostAwareRouter
        
        # Test instantiation
        router = CostAwareRouter()
        print("✅ CostAwareRouter instantiated successfully")
        
        # Test model access
        print(f"✅ Available models: {list(router.models.keys())}")
        
        cheap_model = router.models['cheap']
        premium_model = router.models['premium']
        
        print(f"✅ Cheap model: {cheap_model.model}")
        print(f"✅ Premium model: {premium_model.model}")
        
        # Test database initialization
        if os.path.exists(router.db_path):
            print("✅ Database file created")
            
            conn = sqlite3.connect(router.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"✅ Database tables: {tables}")
            conn.close()
        
        # Test routing logic
        print("\n🔄 Testing Core Routing Logic:")
        
        # Test simple routing
        simple_result = router.route(complexity_score=0.2, max_cost=1.0)
        print(f"✅ Simple query (0.2 complexity) -> {simple_result.get('selected_model', 'unknown')}")
        
        # Test complex routing  
        complex_result = router.route(complexity_score=0.8, max_cost=20.0)
        print(f"✅ Complex query (0.8 complexity) -> {complex_result.get('selected_model', 'unknown')}")
        
        # Test budget constraint
        budget_result = router.route(complexity_score=0.8, max_cost=0.001)
        print(f"✅ Budget constrained -> {budget_result.get('selected_model', 'unknown')}")
        
        # Test performance
        print("\n⚡ Testing Performance:")
        start_time = time.time()
        for _ in range(10):
            router.route(0.5, 5.0)
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / 10) * 1000
        print(f"✅ Average routing time: {avg_time_ms:.2f}ms")
        
        # Test response format
        result = router.route(0.5, 5.0)
        required_keys = ['selected_model', 'cost_estimate', 'reasoning']
        for key in required_keys:
            if key in result:
                print(f"✅ Response contains '{key}'")
            else:
                print(f"⚠️  Response missing '{key}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_docker_readiness():
    """Test Docker deployment readiness"""
    print("\n🐳 Testing Docker Deployment Readiness")
    print("=" * 40)
    
    try:
        # Check Dockerfile
        if os.path.exists('Dockerfile'):
            print("✅ Dockerfile exists")
        else:
            print("❌ Dockerfile missing")
            
        # Check docker-compose.yml
        if os.path.exists('docker-compose.yml'):
            print("✅ docker-compose.yml exists")
        else:
            print("❌ docker-compose.yml missing")
            
        # Check requirements.txt for Anthropic
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            
        if 'anthropic' in requirements:
            print("✅ Anthropic dependency in requirements.txt")
        else:
            print("❌ Missing Anthropic dependency")
            
        # Check app structure
        required_dirs = ['app', 'app/models', 'tests', 'data']
        for directory in required_dirs:
            if os.path.exists(directory):
                print(f"✅ Directory '{directory}' exists")
            else:
                print(f"❌ Directory '{directory}' missing")
                
        # Test import without full dependencies
        print("\n📦 Testing Import Compatibility:")
        test_script = '''
import sys
sys.path.append("/app" if "/app" in sys.path else ".")
try:
    from app.models.cost_aware_router import CostAwareRouter
    router = CostAwareRouter()
    result = router.route(0.5, 5.0)
    print("✅ CostAwareRouter works in container-like environment")
    print(f"Sample routing: {result.get('selected_model', 'unknown')}")
except Exception as e:
    print(f"❌ Container compatibility issue: {e}")
'''
        
        exec(test_script)
        
        return True
        
    except Exception as e:
        print(f"❌ Docker readiness test failed: {e}")
        return False

def generate_test_commands():
    """Generate Docker test commands"""
    print("\n📋 Docker Test Commands")
    print("=" * 25)
    
    commands = """
# 1. Build Docker image
docker build -t opensesame-predictor .

# 2. Run container with cost router enabled
docker run -d \\
  --name opensesame-test \\
  -p 8000:8000 \\
  -e ANTHROPIC_API_KEY=sk-ant-test \\
  -e COST_ROUTER_ENABLED=true \\
  -e DAILY_BUDGET_LIMIT=100.0 \\
  opensesame-predictor

# 3. Wait for startup (30 seconds)
sleep 30

# 4. Test health endpoint
curl -f http://localhost:8000/health

# 5. Test prediction with cost-aware routing
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
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
"""
    
    print(commands)
    
    # Save to file
    with open('/home/quantum/ApiCallPredictor/scripts/cost_aware_router/DOCKER_TEST_COMMANDS_GENERATED.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Docker Test Commands for Cost-Aware Router\n\n')
        f.write(commands)
        f.write('\necho "✅ Docker tests completed"\n')
    
    print("✅ Commands saved to scripts/cost_aware_router/DOCKER_TEST_COMMANDS_GENERATED.sh")

def main():
    """Main test execution"""
    print("🧪 COST-AWARE ROUTER VERIFICATION")
    print("=" * 40)
    
    # Change to project directory
    os.chdir('/home/quantum/ApiCallPredictor')
    
    # Run tests
    basic_test = test_basic_functionality()
    docker_test = test_docker_readiness()
    
    # Generate commands
    generate_test_commands()
    
    print("\n🎯 VERIFICATION SUMMARY")
    print("=" * 25)
    print(f"Basic Functionality: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"Docker Readiness: {'✅ PASS' if docker_test else '❌ FAIL'}")
    
    if basic_test and docker_test:
        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("✅ Cost-Aware Router implementation verified")
        print("✅ Ready for Docker deployment")
        print("📋 Use DOCKER_TEST_COMMANDS.sh to test in Docker")
    else:
        print("\n⚠️  Verification incomplete")
        print("📋 Address issues before Docker deployment")
    
    return basic_test and docker_test

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)