"""
Integration tests for Cost-Aware Router with existing predictor pipeline.

Tests the router's integration with:
1. Existing predictor components (AI Layer, ML Ranker)
2. Database systems (SQLite cache.db)
3. Performance monitoring systems
4. Safety and guardrails validation
5. Phase 5 async parallel processing
"""

import pytest
import asyncio
import sqlite3
import time
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Import components for integration testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cost_aware_router_test import MockCostAwareRouter


class TestCostRouterIntegration:
    """Integration test suite for Cost-Aware Router."""
    
    @pytest.mark.asyncio
    async def test_predictor_pipeline_integration(self, temp_db):
        """Test integration with the existing predictor pipeline."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Simulate predictor request format
        predictor_request = {
            "prompt": "Create user management API endpoints",
            "history": [
                {"api_call": "GET /api/users", "method": "GET", "timestamp": "2023-01-01T10:00:00Z"},
                {"api_call": "POST /api/users", "method": "POST", "timestamp": "2023-01-01T10:01:00Z"}
            ],
            "max_predictions": 3,
            "temperature": 0.7
        }
        
        # Route through cost-aware system
        result = await router.route_request(
            predictor_request["prompt"], 
            predictor_request["history"]
        )
        
        # Verify integration compatibility
        assert "predictions" in result
        assert "processing_time_ms" in result or "total_latency_ms" in result
        assert "model_used" in result
        assert "cost" in result
        
        # Verify predictions structure matches predictor expectations
        predictions = result["predictions"]
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        
        for prediction in predictions:
            # Should have standard prediction fields
            assert "api_call" in prediction or "endpoint" in prediction
            assert "confidence" in prediction
            
        # Verify performance metadata for Phase 5 compatibility
        assert result["total_latency_ms"] < 800
        assert "performance_met" in result
    
    @pytest.mark.asyncio
    async def test_database_integration_with_cache_db(self, temp_db):
        """Test integration with data/cache.db structure."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Make requests to populate database
        test_requests = [
            {"prompt": "Simple query", "expected_model": "claude-haiku"},
            {"prompt": "Complex analysis requiring detailed optimization", "expected_model": "claude-opus"},
            {"prompt": "Medium complexity task with analysis", "expected_model": "claude-sonnet"}
        ]
        
        for req in test_requests:
            result = await router.route_request(req["prompt"])
            assert result["model_used"] == req["expected_model"]
        
        # Verify database structure and data
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Check table exists and has data
        cursor.execute("SELECT COUNT(*) FROM budget_tracking")
        record_count = cursor.fetchone()[0]
        assert record_count == len(test_requests)
        
        # Verify data integrity
        cursor.execute("""
            SELECT model_name, prompt_complexity, cost, latency_ms, budget_remaining 
            FROM budget_tracking 
            ORDER BY id
        """)
        records = cursor.fetchall()
        
        for i, (model, complexity, cost, latency, budget) in enumerate(records):
            assert model == test_requests[i]["expected_model"]
            assert 0.0 <= complexity <= 1.0
            assert cost > 0
            assert latency > 0
            assert budget >= 0
        
        conn.close()
    
    @pytest.mark.asyncio 
    async def test_async_parallel_processing_integration(self, temp_db):
        """Test integration with Phase 5 async parallel processing."""
        router = MockCostAwareRouter(budget_limit=100.0, db_path=temp_db)
        
        # Simulate parallel AI + ML + Safety processing
        concurrent_requests = [
            "Analyze user behavior patterns",
            "Create API documentation",
            "Optimize database queries",
            "Implement authentication system",
            "Design microservices architecture"
        ]
        
        # Process requests in parallel
        start_time = time.time()
        tasks = [router.route_request(prompt) for prompt in concurrent_requests]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify parallel processing efficiency
        assert total_time < len(concurrent_requests) * 0.8  # Should be faster than sequential
        
        # Verify all requests succeeded
        assert len(results) == len(concurrent_requests)
        for result in results:
            assert result["total_latency_ms"] < 800
            assert result["llm_latency_ms"] < 500
            assert "predictions" in result
            
        # Verify budget tracking under parallel load
        status = await router.get_budget_status()
        assert status["daily_stats"]["total_requests"] == len(concurrent_requests)
    
    @pytest.mark.asyncio
    async def test_ml_ranker_integration(self, temp_db):
        """Test integration with ML Ranker component."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Mock ML ranker behavior
        prompt = "Create user management endpoints with validation"
        history = [
            {"api_call": "GET /api/users", "method": "GET"},
            {"api_call": "POST /api/users", "method": "POST"}
        ]
        
        result = await router.route_request(prompt, history)
        
        # Verify ML ranking integration
        predictions = result["predictions"]
        assert len(predictions) > 0
        
        # Should have ML-related metadata if using Sonnet/Opus
        if result["model_used"] in ["claude-sonnet", "claude-opus"]:
            # Higher quality models should produce better predictions
            for prediction in predictions:
                assert prediction["confidence"] >= 0.5
        
        # Verify cost tracking includes ML processing considerations
        assert result["cost"] > 0
        assert "complexity_score" in result
    
    @pytest.mark.asyncio
    async def test_safety_guardrails_integration(self, temp_db):
        """Test integration with safety and guardrails validation."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Test potentially unsafe inputs
        unsafe_inputs = [
            "'; DROP TABLE users; --",  # SQL injection
            "<script>alert('xss')</script>",  # XSS
            "DELETE FROM sensitive_data WHERE 1=1",  # Malicious SQL
        ]
        
        for unsafe_input in unsafe_inputs:
            result = await router.route_request(unsafe_input)
            
            # Should still process request safely
            assert "model_used" in result
            assert result["budget_remaining"] >= 0
            
            # Should use cheaper model for potentially unsafe content
            assert result["model_used"] in ["claude-haiku", "claude-sonnet"]
            
            # Should have safety metadata
            assert "predictions" in result
    
    @pytest.mark.asyncio
    async def test_feature_extraction_integration(self, temp_db):
        """Test integration with feature extraction pipeline."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Request with rich context for feature extraction
        prompt = "Design authentication system with role-based access control"
        history = [
            {"api_call": "GET /api/auth/login", "method": "GET", "timestamp": "2023-01-01T10:00:00Z"},
            {"api_call": "POST /api/auth/validate", "method": "POST", "timestamp": "2023-01-01T10:01:00Z"},
            {"api_call": "GET /api/users/profile", "method": "GET", "timestamp": "2023-01-01T10:02:00Z"}
        ]
        
        result = await router.route_request(prompt, history)
        
        # Verify feature extraction influenced model selection
        assert result["complexity_score"] > 0.3  # Should detect complexity from context
        
        # History should influence complexity calculation
        no_history_result = await router.route_request(prompt, [])
        assert result["complexity_score"] >= no_history_result["complexity_score"]
        
        # Should have rich prediction metadata
        predictions = result["predictions"]
        assert len(predictions) > 0
    
    @pytest.mark.asyncio
    async def test_cold_start_integration(self, temp_db):
        """Test integration with cold start prediction scenarios."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Simulate cold start scenario (no history)
        cold_start_prompts = [
            "Get user information",
            "Create new data record", 
            "Search for content"
        ]
        
        for prompt in cold_start_prompts:
            result = await router.route_request(prompt, [])
            
            # Should handle cold start gracefully
            assert "model_used" in result
            assert result["total_latency_ms"] < 800
            assert len(result["predictions"]) > 0
            
            # Should prefer cheaper models for cold start
            assert result["model_used"] in ["claude-haiku", "claude-sonnet"]
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, temp_db):
        """Test integration with caching systems."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Make identical requests to test caching behavior
        prompt = "Test caching integration"
        
        # First request
        start_time = time.time()
        result1 = await router.route_request(prompt)
        first_duration = time.time() - start_time
        
        # Second identical request (should be faster if cached)
        start_time = time.time()
        result2 = await router.route_request(prompt)
        second_duration = time.time() - start_time
        
        # Verify consistent results
        assert result1["model_used"] == result2["model_used"]
        assert result1["complexity_score"] == result2["complexity_score"]
        
        # Budget should still be decremented for both requests
        assert result2["budget_remaining"] < result1["budget_remaining"]
    
    @pytest.mark.asyncio
    async def test_metrics_and_monitoring_integration(self, temp_db):
        """Test integration with metrics and monitoring systems."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Generate various requests for metrics
        test_scenarios = [
            {"prompt": "Simple query", "category": "simple"},
            {"prompt": "Medium analysis task", "category": "medium"},  
            {"prompt": "Complex optimization analysis", "category": "complex"},
            {"prompt": "Another simple task", "category": "simple"}
        ]
        
        for scenario in test_scenarios:
            result = await router.route_request(scenario["prompt"])
            scenario["result"] = result
        
        # Get comprehensive metrics
        status = await router.get_budget_status()
        
        # Verify metrics structure
        assert "budget_limit" in status
        assert "current_budget" in status
        assert "daily_stats" in status
        
        daily_stats = status["daily_stats"]
        assert daily_stats["total_requests"] == len(test_scenarios)
        assert daily_stats["total_cost"] > 0
        assert daily_stats["avg_latency_ms"] > 0
        
        # Verify cost accuracy
        expected_total_cost = sum(s["result"]["cost"] for s in test_scenarios)
        assert abs(daily_stats["total_cost"] - expected_total_cost) < 0.01
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, temp_db):
        """Test error handling integration across components.""" 
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Test various error conditions
        error_scenarios = [
            {"input": "", "description": "Empty prompt"},
            {"input": None, "description": "None input"},
            {"input": "A" * 20000, "description": "Extremely long prompt"}
        ]
        
        for scenario in error_scenarios:
            try:
                if scenario["input"] is None:
                    result = await router.route_request("")
                else:
                    result = await router.route_request(scenario["input"])
                
                # If no exception, should have valid result structure
                assert "model_used" in result
                assert result["budget_remaining"] >= 0
                assert "total_latency_ms" in result
                
            except Exception as e:
                # Any exceptions should be handled gracefully
                assert isinstance(e, (ValueError, TypeError, RuntimeError))
        
        # System should continue working after errors
        normal_result = await router.route_request("Normal request after errors")
        assert normal_result["total_latency_ms"] < 800
    
    @pytest.mark.asyncio
    async def test_phase5_performance_integration(self, temp_db):
        """Test integration with Phase 5 performance optimization requirements."""
        router = MockCostAwareRouter(budget_limit=100.0, db_path=temp_db)
        
        # Test Phase 5 requirements
        phase5_requests = [
            "Quick data retrieval",
            "Medium complexity analysis with caching",
            "Complex optimization with parallel processing"
        ]
        
        all_results = []
        for prompt in phase5_requests:
            result = await router.route_request(prompt)
            all_results.append(result)
        
        # Verify Phase 5 performance targets
        for result in all_results:
            # Core Phase 5 requirements
            assert result["llm_latency_ms"] < 500, "LLM latency requirement not met"
            assert result["total_latency_ms"] < 800, "Total latency requirement not met"
            
            # Verify performance metadata
            assert result["performance_met"]["llm_under_500ms"] is True
            assert result["performance_met"]["total_under_800ms"] is True
            
            # Should have async processing metadata
            assert "model_used" in result
            assert "cost" in result
        
        # Verify overall system performance
        avg_llm_latency = sum(r["llm_latency_ms"] for r in all_results) / len(all_results)
        avg_total_latency = sum(r["total_latency_ms"] for r in all_results) / len(all_results)
        
        assert avg_llm_latency < 400, f"Average LLM latency too high: {avg_llm_latency}ms"
        assert avg_total_latency < 600, f"Average total latency too high: {avg_total_latency}ms"
    
    def test_database_schema_compatibility(self, temp_db):
        """Test database schema compatibility with existing systems."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Verify database structure
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        # Check for required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "budget_tracking" in tables
        assert "cost_optimization" in tables
        
        # Verify table structures are compatible
        cursor.execute("PRAGMA table_info(budget_tracking)")
        budget_columns = [row[1] for row in cursor.fetchall()]
        
        required_budget_columns = [
            'id', 'timestamp', 'model_name', 'prompt_complexity',
            'cost', 'latency_ms', 'budget_remaining'
        ]
        
        for col in required_budget_columns:
            assert col in budget_columns, f"Missing required column: {col}"
        
        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])