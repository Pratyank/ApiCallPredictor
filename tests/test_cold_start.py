#!/usr/bin/env python3
"""
Test script for Phase 4 Cold Start functionality
Tests the cold_start_predict implementation and integration
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.predictor import Predictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_cold_start_functionality():
    """Test cold start functionality comprehensively"""
    
    print("🚀 Testing Phase 4 Cold Start Implementation")
    print("=" * 50)
    
    predictor = Predictor()
    
    # Test 1: Cold start without prompt (should return popular safe endpoints)
    print("\n📋 Test 1: Cold start without prompt")
    results = await predictor.cold_start_predict(k=3)
    print(f"✅ Got {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result.get('api_call', 'N/A')} - confidence: {result.get('confidence', 0):.2f} - type: {result.get('cold_start_type', 'unknown')}")
    
    # Test 2: Cold start with prompt (should use semantic search)
    print("\n📋 Test 2: Cold start with prompt (semantic search)")
    results = await predictor.cold_start_predict(prompt='search for users and profiles', k=3)
    print(f"✅ Got {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result.get('api_call', 'N/A')} - confidence: {result.get('confidence', 0):.2f} - type: {result.get('cold_start_type', 'unknown')}")
        if 'semantic_score' in result:
            print(f"      semantic_score: {result['semantic_score']:.3f}")
    
    # Test 3: Main predict method with no history (should trigger cold start)
    print("\n📋 Test 3: Main predict method with no history")
    results = await predictor.predict('find user authentication endpoints', history=None)
    print(f"✅ Got {len(results['predictions'])} predictions")
    print(f"   Processing method: {results['metadata'].get('processing_method', 'unknown')}")
    print(f"   Model version: {results['metadata'].get('model_version', 'unknown')}")
    for i, result in enumerate(results['predictions'], 1):
        print(f"   {i}. {result.get('api_call', 'N/A')} - confidence: {result.get('confidence', 0):.2f}")
    
    # Test 4: Main predict method with empty history (should trigger cold start)
    print("\n📋 Test 4: Main predict method with empty history")
    results = await predictor.predict('get system information', history=[])
    print(f"✅ Got {len(results['predictions'])} predictions")
    print(f"   Processing method: {results['metadata'].get('processing_method', 'unknown')}")
    
    # Test 5: Test metrics include cold start data
    print("\n📋 Test 5: Metrics include cold start data")
    metrics = await predictor.get_metrics()
    cold_start_metrics = metrics.get('cold_start_metrics', {})
    print(f"✅ Cold start metrics:")
    print(f"   Total endpoints: {cold_start_metrics.get('total_endpoints', 0)}")
    print(f"   Safe endpoints: {cold_start_metrics.get('safe_endpoints', 0)}")
    print(f"   Semantic model loaded: {cold_start_metrics.get('semantic_model_loaded', False)}")
    print(f"   Semantic model type: {cold_start_metrics.get('semantic_model_type', 'N/A')}")
    
    # Test 6: Health check includes cold start status
    print("\n📋 Test 6: Health check includes cold start status")
    health = await predictor.health_check()
    cold_start_health = health.get('components', {}).get('cold_start', {})
    print(f"✅ Cold start health:")
    print(f"   Status: {cold_start_health.get('status', 'unknown')}")
    print(f"   Semantic model: {cold_start_health.get('semantic_model', 'unknown')}")
    print(f"   Popular endpoints count: {cold_start_health.get('popular_endpoints_count', 0)}")
    
    # Test 7: Update endpoint popularity
    print("\n📋 Test 7: Update endpoint popularity")
    await predictor.update_endpoint_popularity('GET', '/api/test', was_clicked=True)
    print("✅ Updated endpoint popularity")
    
    print("\n🎉 All cold start tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_cold_start_functionality())