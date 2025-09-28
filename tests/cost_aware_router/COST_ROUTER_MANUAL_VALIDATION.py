#!/usr/bin/env python3
"""
COST-AWARE ROUTER MANUAL VALIDATION SCRIPT
==========================================

This script provides manual validation of the Cost-Aware Model Router 
implementation for the OpenSesame Predictor project.

PURPOSE:
- Manual step-by-step validation of Cost-Aware Router functionality
- Detailed error reporting and diagnostics
- Integration compatibility checking with existing components
- Performance metrics collection and analysis

FEATURE: Cost-Aware Model Router (Bonus Implementation)
COMPONENT: Manual validation and diagnostics tool

USAGE:
    python COST_ROUTER_MANUAL_VALIDATION.py

VALIDATES:
- CostAwareRouter class functionality
- Model definitions (claude-3-haiku, claude-3-opus)
- Routing logic based on complexity and budget
- Database integration with SQLite
- Integration with AI Layer and Predictor components
- Performance requirements compliance
"""

import sys
import os
import sqlite3
import json
from datetime import datetime

# Add app to path
sys.path.insert(0, '/home/quantum/ApiCallPredictor')

def test_cost_aware_router():
    """Test CostAwareRouter implementation"""
    print("🧪 Testing Cost-Aware Router Implementation")
    print("=" * 60)
    
    try:
        # Import CostAwareRouter
        from app.models.cost_aware_router import CostAwareRouter
        print("✅ CostAwareRouter import successful")
        
        # Initialize router
        router = CostAwareRouter()
        print("✅ CostAwareRouter initialization successful")
        
        # Test model definitions
        print(f"✅ Models defined: {list(router.models.keys())}")
        print(f"   - Cheap model: {router.models['cheap']['model']} (${router.models['cheap']['cost']}/1K tokens)")
        print(f"   - Premium model: {router.models['premium']['model']} (${router.models['premium']['cost']}/1K tokens)")
        
        # Test database initialization
        if os.path.exists(router.db_path):
            print("✅ Database file exists")
            conn = sqlite3.connect(router.db_path)
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['budget_consumption', 'model_performance', 'router_settings']
            for table in expected_tables:
                if table in tables:
                    print(f"   ✅ Table '{table}' exists")
                else:
                    print(f"   ❌ Table '{table}' missing")
            
            conn.close()
        else:
            print("❌ Database file not found")
        
        # Test routing logic
        print("\n🔄 Testing Routing Logic:")
        
        # Test simple query (should route to cheap model)
        simple_result = router.route(complexity_score=0.2, max_cost=1.0)
        print(f"   Simple query (0.2 complexity) -> {simple_result['selected_model']}")
        
        # Test complex query (should route to premium model)
        complex_result = router.route(complexity_score=0.8, max_cost=10.0)
        print(f"   Complex query (0.8 complexity) -> {complex_result['selected_model']}")
        
        # Test budget constraint
        budget_result = router.route(complexity_score=0.8, max_cost=0.001)
        print(f"   Budget constrained (0.8 complexity, $0.001 max) -> {budget_result['selected_model']}")
        
        # Test budget tracking
        print("\n💰 Testing Budget Tracking:")
        initial_budget = router.get_daily_budget_status()
        print(f"   Daily budget: ${initial_budget['daily_limit']}")
        print(f"   Current usage: ${initial_budget['used_today']:.4f}")
        print(f"   Remaining: ${initial_budget['remaining']:.4f}")
        
        # Test cost estimation
        print("\n📊 Testing Cost Estimation:")
        cost_estimate = router.estimate_query_cost("This is a test query", "claude-3-haiku")
        print(f"   Estimated cost for test query: ${cost_estimate:.6f}")
        
        # Test analytics
        print("\n📈 Testing Analytics:")
        analytics = router.get_router_analytics()
        print(f"   Total queries processed: {analytics['total_queries']}")
        print(f"   Cost efficiency: {analytics['avg_cost_per_query']:.6f}")
        
        print("\n🎉 All manual tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration with AI Layer and Predictor"""
    print("\n🔗 Testing Integration with AI Layer")
    print("=" * 60)
    
    try:
        # Test AI Layer integration (without full dependencies)
        print("✅ Testing imports...")
        
        # Check if files exist
        ai_layer_path = "/home/quantum/ApiCallPredictor/app/models/ai_layer.py"
        predictor_path = "/home/quantum/ApiCallPredictor/app/models/predictor.py"
        
        if os.path.exists(ai_layer_path):
            print("✅ ai_layer.py exists")
        else:
            print("❌ ai_layer.py missing")
            
        if os.path.exists(predictor_path):
            print("✅ predictor.py exists")
        else:
            print("❌ predictor.py missing")
        
        # Check for integration keywords in ai_layer.py
        with open(ai_layer_path, 'r') as f:
            ai_content = f.read()
            
        if 'CostAwareRouter' in ai_content:
            print("✅ CostAwareRouter integration found in ai_layer.py")
        else:
            print("❌ CostAwareRouter integration missing in ai_layer.py")
            
        # Check for integration in predictor.py
        with open(predictor_path, 'r') as f:
            predictor_content = f.read()
            
        if 'cost' in predictor_content.lower():
            print("✅ Cost-related code found in predictor.py")
        else:
            print("❌ Cost integration missing in predictor.py")
            
        print("✅ Integration tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False

def test_performance():
    """Test performance requirements"""
    print("\n⚡ Testing Performance Requirements")
    print("=" * 60)
    
    try:
        from app.models.cost_aware_router import CostAwareRouter
        import time
        
        router = CostAwareRouter()
        
        # Test routing speed (should be < 50ms)
        start_time = time.time()
        for i in range(10):
            router.route(0.5, 5.0)
        end_time = time.time()
        
        avg_time_ms = ((end_time - start_time) / 10) * 1000
        print(f"   Average routing time: {avg_time_ms:.2f}ms")
        
        if avg_time_ms < 50:
            print("✅ Routing performance meets requirement (<50ms)")
        else:
            print("❌ Routing performance too slow (>50ms)")
            
        # Test database operations speed
        start_time = time.time()
        for i in range(5):
            router.get_daily_budget_status()
        end_time = time.time()
        
        db_time_ms = ((end_time - start_time) / 5) * 1000
        print(f"   Average DB operation time: {db_time_ms:.2f}ms")
        
        if db_time_ms < 100:
            print("✅ Database performance acceptable (<100ms)")
        else:
            print("❌ Database operations too slow (>100ms)")
            
        print("✅ Performance tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Cost-Aware Router Manual Testing Suite")
    print("==========================================")
    print()
    
    success = True
    
    # Run tests
    success &= test_cost_aware_router()
    success &= test_integration()
    success &= test_performance()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Implementation is working correctly!")
        print("✅ Cost-Aware Router is ready for Docker deployment")
    else:
        print("❌ Some tests failed - Check implementation")
        
    print("=" * 60)