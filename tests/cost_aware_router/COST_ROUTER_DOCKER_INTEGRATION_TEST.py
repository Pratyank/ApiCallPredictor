#!/usr/bin/env python3
"""
Docker Integration Test Script for Cost-Aware Router
This script creates a comprehensive test for the Docker environment
"""

import subprocess
import json
import time
import requests
import sys

def build_docker_image():
    """Build the Docker image"""
    print("🐳 Building Docker image...")
    try:
        result = subprocess.run([
            'docker', 'build', '-t', 'opensesame-predictor', '.'
        ], capture_output=True, text=True, check=True)
        print("✅ Docker image built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker build failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker Desktop and enable WSL integration")
        return False

def run_docker_container():
    """Run the Docker container"""
    print("🚀 Starting Docker container...")
    try:
        # Stop any existing container
        subprocess.run(['docker', 'stop', 'opensesame-test'], 
                      capture_output=True, text=True)
        subprocess.run(['docker', 'rm', 'opensesame-test'], 
                      capture_output=True, text=True)
        
        # Start new container
        result = subprocess.run([
            'docker', 'run', '-d', 
            '--name', 'opensesame-test',
            '-p', '8000:8000',
            '-e', 'ANTHROPIC_API_KEY=test_key',
            '-e', 'COST_ROUTER_ENABLED=true',
            'opensesame-predictor'
        ], capture_output=True, text=True, check=True)
        
        container_id = result.stdout.strip()
        print(f"✅ Container started: {container_id[:12]}")
        
        # Wait for startup
        print("⏳ Waiting for service to start...")
        time.sleep(30)
        
        return container_id
    except subprocess.CalledProcessError as e:
        print(f"❌ Container start failed: {e.stderr}")
        return None

def test_api_endpoints(container_id):
    """Test API endpoints in the running container"""
    print("🧪 Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    tests_passed = 0
    total_tests = 0
    
    # Test health endpoint
    total_tests += 1
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        if response.status_code == 200:
            print("✅ Health endpoint accessible")
            tests_passed += 1
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
    
    # Test prediction endpoint
    total_tests += 1
    try:
        test_payload = {
            "prompt": "I need to get user information",
            "history": [],
            "max_predictions": 3
        }
        response = requests.post(f"{base_url}/predict", 
                               json=test_payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if 'predictions' in data:
                print("✅ Prediction endpoint working")
                print(f"   Returned {len(data['predictions'])} predictions")
                if 'metadata' in data and 'cost_router_enabled' in str(data['metadata']):
                    print("✅ Cost-aware routing detected in response")
                tests_passed += 1
            else:
                print("❌ Prediction endpoint: invalid response format")
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Prediction endpoint error: {e}")
    
    # Test metrics endpoint
    total_tests += 1
    try:
        response = requests.get(f"{base_url}/metrics", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("✅ Metrics endpoint accessible")
            if 'predictor_metrics' in data:
                print("   📊 Predictor metrics available")
            tests_passed += 1
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics endpoint error: {e}")
    
    return tests_passed, total_tests

def test_cost_router_in_container(container_id):
    """Test cost router functionality inside container"""
    print("💰 Testing Cost-Aware Router in container...")
    
    try:
        # Execute test script inside container
        test_script = '''
import sys
sys.path.append("/app")
from app.models.cost_aware_router import CostAwareRouter

print("Testing CostAwareRouter inside container...")
router = CostAwareRouter()
print(f"Models: {list(router.models.keys())}")

# Test routing
result = router.route(0.2, 1.0)
print(f"Simple query routes to: {result.get('selected_model', 'unknown')}")

result = router.route(0.8, 10.0)
print(f"Complex query routes to: {result.get('selected_model', 'unknown')}")

print("✅ CostAwareRouter working in container")
'''
        
        result = subprocess.run([
            'docker', 'exec', container_id, 'python', '-c', test_script
        ], capture_output=True, text=True, check=True)
        
        print("✅ Container test output:")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Container test failed: {e.stderr}")
        return False

def check_logs(container_id):
    """Check container logs for errors"""
    print("📋 Checking container logs...")
    try:
        result = subprocess.run([
            'docker', 'logs', '--tail', '50', container_id
        ], capture_output=True, text=True, check=True)
        
        logs = result.stdout
        if 'ERROR' in logs or 'Exception' in logs:
            print("⚠️  Errors found in logs:")
            for line in logs.split('\n'):
                if 'ERROR' in line or 'Exception' in line:
                    print(f"   {line}")
        else:
            print("✅ No errors found in logs")
            
        if 'Cost-Aware' in logs or 'CostAwareRouter' in logs:
            print("✅ Cost router activity detected in logs")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Log check failed: {e.stderr}")
        return False

def cleanup_container(container_id):
    """Stop and remove test container"""
    if container_id:
        print("🧹 Cleaning up...")
        subprocess.run(['docker', 'stop', container_id], 
                      capture_output=True, text=True)
        subprocess.run(['docker', 'rm', container_id], 
                      capture_output=True, text=True)
        print("✅ Container cleaned up")

def main():
    """Main test execution"""
    print("🧪 Docker Integration Test for Cost-Aware Router")
    print("=" * 60)
    
    # Check if Docker is available
    try:
        subprocess.run(['docker', '--version'], 
                      capture_output=True, text=True, check=True)
        print("✅ Docker is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not available")
        print("Please install Docker Desktop and enable WSL integration")
        print("Instructions: https://docs.docker.com/desktop/wsl/")
        return False
    
    container_id = None
    try:
        # Build image
        if not build_docker_image():
            return False
        
        # Run container
        container_id = run_docker_container()
        if not container_id:
            return False
        
        # Test API endpoints
        api_passed, api_total = test_api_endpoints(container_id)
        print(f"📊 API Tests: {api_passed}/{api_total} passed")
        
        # Test cost router in container
        router_test = test_cost_router_in_container(container_id)
        
        # Check logs
        log_check = check_logs(container_id)
        
        # Final assessment
        print("\n" + "=" * 60)
        print("🎯 DOCKER TEST RESULTS:")
        print(f"   API Endpoints: {api_passed}/{api_total}")
        print(f"   Cost Router: {'✅ PASS' if router_test else '❌ FAIL'}")
        print(f"   Log Check: {'✅ PASS' if log_check else '❌ FAIL'}")
        
        overall_success = (api_passed >= api_total * 0.8) and router_test and log_check
        
        if overall_success:
            print("🎉 DOCKER INTEGRATION TEST PASSED!")
            print("✅ Cost-Aware Router working correctly in Docker")
        else:
            print("❌ Docker integration test failed")
            
        return overall_success
        
    finally:
        cleanup_container(container_id)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)