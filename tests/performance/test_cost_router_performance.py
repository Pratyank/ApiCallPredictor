"""
Performance tests for Cost-Aware Router.

Validates that the router meets the Phase 5 performance requirements:
- LLM latency < 500ms
- Total response time < 800ms
- Handles concurrent requests efficiently
- Maintains performance under budget constraints
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Import the test router from the main test file
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cost_aware_router_test import MockCostAwareRouter


class PerformanceTestRunner:
    """Utility class for running performance tests."""
    
    def __init__(self):
        self.results = []
        self.errors = []
    
    async def run_concurrent_requests(self, router: MockCostAwareRouter, 
                                    prompts: List[str], concurrency: int = 10) -> List[Dict[str, Any]]:
        """Run multiple requests concurrently and measure performance."""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(prompt: str, request_id: int) -> Dict[str, Any]:
            async with semaphore:
                start_time = time.time()
                try:
                    result = await router.route_request(prompt)
                    end_time = time.time()
                    
                    return {
                        "request_id": request_id,
                        "prompt": prompt,
                        "success": True,
                        "external_latency_ms": (end_time - start_time) * 1000,
                        "reported_latency_ms": result["total_latency_ms"],
                        "llm_latency_ms": result["llm_latency_ms"],
                        "model_used": result["model_used"],
                        "cost": result["cost"],
                        "performance_met": result["performance_met"]
                    }
                except Exception as e:
                    end_time = time.time()
                    return {
                        "request_id": request_id,
                        "prompt": prompt,
                        "success": False,
                        "error": str(e),
                        "external_latency_ms": (end_time - start_time) * 1000
                    }
        
        tasks = [limited_request(prompt, i) for i, prompt in enumerate(prompts)]
        return await asyncio.gather(*tasks)
    
    def calculate_performance_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics from test results."""
        successful_results = [r for r in results if r.get("success", False)]
        
        if not successful_results:
            return {"error": "No successful requests"}
        
        latencies = [r["external_latency_ms"] for r in successful_results]
        llm_latencies = [r["llm_latency_ms"] for r in successful_results]
        reported_latencies = [r["reported_latency_ms"] for r in successful_results]
        
        # Performance requirement compliance
        llm_under_500ms = [r["performance_met"]["llm_under_500ms"] for r in successful_results]
        total_under_800ms = [r["performance_met"]["total_under_800ms"] for r in successful_results]
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "success_rate": len(successful_results) / len(results),
            "latency_stats": {
                "external_ms": {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": statistics.mean(latencies),
                    "median": statistics.median(latencies),
                    "p95": self._percentile(latencies, 95),
                    "p99": self._percentile(latencies, 99)
                },
                "llm_ms": {
                    "min": min(llm_latencies),
                    "max": max(llm_latencies),
                    "mean": statistics.mean(llm_latencies),
                    "median": statistics.median(llm_latencies),
                    "p95": self._percentile(llm_latencies, 95),
                    "p99": self._percentile(llm_latencies, 99)
                },
                "reported_ms": {
                    "min": min(reported_latencies),
                    "max": max(reported_latencies),
                    "mean": statistics.mean(reported_latencies),
                    "median": statistics.median(reported_latencies),
                    "p95": self._percentile(reported_latencies, 95),
                    "p99": self._percentile(reported_latencies, 99)
                }
            },
            "performance_compliance": {
                "llm_under_500ms_rate": sum(llm_under_500ms) / len(llm_under_500ms),
                "total_under_800ms_rate": sum(total_under_800ms) / len(total_under_800ms)
            },
            "cost_stats": {
                "total_cost": sum(r["cost"] for r in successful_results),
                "avg_cost": statistics.mean([r["cost"] for r in successful_results]),
                "cost_range": (
                    min(r["cost"] for r in successful_results),
                    max(r["cost"] for r in successful_results)
                )
            },
            "model_usage": {
                model: sum(1 for r in successful_results if r["model_used"] == model)
                for model in set(r["model_used"] for r in successful_results)
            }
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_data) - 1)
        weight = index - lower_index
        return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight


@pytest.fixture
def performance_runner():
    """Performance test runner fixture."""
    return PerformanceTestRunner()


@pytest.fixture
def performance_prompts():
    """Standard prompts for performance testing."""
    return {
        "simple": [
            "Get user data",
            "List items",
            "Show status",
            "Quick lookup",
            "Basic query"
        ],
        "medium": [
            "Analyze user behavior patterns with detailed metrics",
            "Create comprehensive API documentation for the user system",
            "Design efficient data processing pipeline with error handling",
            "Implement authentication system with role-based access control",
            "Optimize database queries for better performance"
        ],
        "complex": [
            "Design and implement a distributed microservices architecture for enterprise e-commerce platform with comprehensive analytics and real-time optimization",
            "Analyze complex strategic integration patterns for multi-tenant SaaS platform with advanced security and scalability considerations",
            "Create comprehensive performance optimization framework with detailed monitoring, alerting, and automated remediation capabilities",
            "Implement advanced machine learning pipeline for real-time recommendation system with A/B testing and continuous learning",
            "Design fault-tolerant distributed system architecture with eventual consistency and comprehensive disaster recovery"
        ]
    }


class TestCostRouterPerformance:
    """Performance test suite for Cost-Aware Router."""
    
    @pytest.mark.asyncio
    async def test_single_request_baseline_performance(self, temp_db, performance_runner):
        """Test baseline performance for single requests."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        test_cases = [
            ("Simple query", "claude-haiku", 300),
            ("Medium complexity analysis task", "claude-sonnet", 450),
            ("Complex optimization requiring detailed analysis", "claude-opus", 550)
        ]
        
        for prompt, expected_model, max_expected_latency in test_cases:
            start_time = time.time()
            result = await router.route_request(prompt)
            end_time = time.time()
            
            external_latency = (end_time - start_time) * 1000
            
            # Verify performance requirements
            assert result["llm_latency_ms"] < 500, f"LLM latency exceeded: {result['llm_latency_ms']}ms"
            assert result["total_latency_ms"] < 800, f"Total latency exceeded: {result['total_latency_ms']}ms"
            assert external_latency < 1000, f"External measurement exceeded: {external_latency}ms"
            
            # Verify model selection
            assert result["model_used"] == expected_model
            
            # Verify timing accuracy (within 20% tolerance)
            timing_diff = abs(external_latency - result["total_latency_ms"])
            assert timing_diff < result["total_latency_ms"] * 0.2
    
    @pytest.mark.asyncio
    async def test_concurrent_load_performance(self, temp_db, performance_runner, performance_prompts):
        """Test performance under concurrent load."""
        router = MockCostAwareRouter(budget_limit=100.0, db_path=temp_db)
        
        # Mix of different complexity requests
        mixed_prompts = (
            performance_prompts["simple"] * 4 +
            performance_prompts["medium"] * 2 +
            performance_prompts["complex"] * 1
        )
        
        # Test different concurrency levels
        concurrency_levels = [5, 10, 15]
        
        for concurrency in concurrency_levels:
            results = await performance_runner.run_concurrent_requests(
                router, mixed_prompts[:concurrency], concurrency
            )
            
            metrics = performance_runner.calculate_performance_metrics(results)
            
            # Verify high success rate
            assert metrics["success_rate"] >= 0.95, f"Low success rate at concurrency {concurrency}: {metrics['success_rate']}"
            
            # Verify performance compliance
            assert metrics["performance_compliance"]["llm_under_500ms_rate"] >= 0.90
            assert metrics["performance_compliance"]["total_under_800ms_rate"] >= 0.90
            
            # Verify reasonable latency increases under load
            assert metrics["latency_stats"]["external_ms"]["p95"] < 1000
            assert metrics["latency_stats"]["llm_ms"]["p95"] < 600
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, temp_db, performance_runner, performance_prompts):
        """Test performance under sustained load over time."""
        router = MockCostAwareRouter(budget_limit=200.0, db_path=temp_db)
        
        # Run sustained load for 30 seconds
        duration_seconds = 30
        requests_per_second = 3
        
        all_results = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            batch_start = time.time()
            
            # Send batch of requests
            batch_prompts = performance_prompts["simple"][:requests_per_second]
            batch_results = await performance_runner.run_concurrent_requests(
                router, batch_prompts, requests_per_second
            )
            all_results.extend(batch_results)
            
            # Wait for next second
            batch_duration = time.time() - batch_start
            if batch_duration < 1.0:
                await asyncio.sleep(1.0 - batch_duration)
        
        # Analyze sustained performance
        metrics = performance_runner.calculate_performance_metrics(all_results)
        
        # Verify sustained performance
        assert metrics["success_rate"] >= 0.95
        assert metrics["performance_compliance"]["llm_under_500ms_rate"] >= 0.85
        assert metrics["performance_compliance"]["total_under_800ms_rate"] >= 0.85
        
        # Verify no significant performance degradation over time
        first_half = all_results[:len(all_results)//2]
        second_half = all_results[len(all_results)//2:]
        
        first_half_metrics = performance_runner.calculate_performance_metrics(first_half)
        second_half_metrics = performance_runner.calculate_performance_metrics(second_half)
        
        # Latency shouldn't increase by more than 50% in second half
        latency_increase = (
            second_half_metrics["latency_stats"]["external_ms"]["mean"] /
            first_half_metrics["latency_stats"]["external_ms"]["mean"]
        )
        assert latency_increase < 1.5, f"Significant performance degradation: {latency_increase}x"
    
    @pytest.mark.asyncio
    async def test_performance_under_budget_constraints(self, temp_db, performance_runner, performance_prompts):
        """Test performance when operating under budget constraints."""
        # Start with very limited budget
        router = MockCostAwareRouter(budget_limit=2.0, db_path=temp_db)
        
        # Mix of requests that would normally use expensive models
        constrained_prompts = performance_prompts["complex"][:5]
        
        results = await performance_runner.run_concurrent_requests(
            router, constrained_prompts, 3
        )
        
        metrics = performance_runner.calculate_performance_metrics(results)
        
        # Should still meet performance requirements even with budget constraints
        assert metrics["success_rate"] >= 0.95
        assert metrics["performance_compliance"]["llm_under_500ms_rate"] >= 0.80
        assert metrics["performance_compliance"]["total_under_800ms_rate"] >= 0.80
        
        # Should primarily use cheaper models due to budget constraints
        model_usage = metrics["model_usage"]
        total_requests = sum(model_usage.values())
        haiku_percentage = model_usage.get("claude-haiku", 0) / total_requests
        
        assert haiku_percentage >= 0.6, f"Expected more Haiku usage under budget constraints: {haiku_percentage}"
    
    @pytest.mark.asyncio
    async def test_performance_scalability_patterns(self, temp_db, performance_runner, performance_prompts):
        """Test how performance scales with different load patterns."""
        router = MockCostAwareRouter(budget_limit=100.0, db_path=temp_db)
        
        # Test scalability patterns
        patterns = [
            {"name": "burst", "prompts": performance_prompts["simple"] * 10, "concurrency": 10},
            {"name": "gradual", "prompts": performance_prompts["medium"] * 5, "concurrency": 5},
            {"name": "mixed_load", "prompts": (
                performance_prompts["simple"] * 3 +
                performance_prompts["medium"] * 2 +
                performance_prompts["complex"] * 1
            ), "concurrency": 6}
        ]
        
        scalability_results = {}
        
        for pattern in patterns:
            results = await performance_runner.run_concurrent_requests(
                router, pattern["prompts"], pattern["concurrency"]
            )
            
            metrics = performance_runner.calculate_performance_metrics(results)
            scalability_results[pattern["name"]] = metrics
            
            # Each pattern should maintain acceptable performance
            assert metrics["success_rate"] >= 0.90
            assert metrics["performance_compliance"]["total_under_800ms_rate"] >= 0.85
        
        # Compare relative performance across patterns
        burst_latency = scalability_results["burst"]["latency_stats"]["external_ms"]["mean"]
        gradual_latency = scalability_results["gradual"]["latency_stats"]["external_ms"]["mean"]
        mixed_latency = scalability_results["mixed_load"]["latency_stats"]["external_ms"]["mean"]
        
        # Burst load should have higher latency but still reasonable
        assert burst_latency < 800, f"Burst load latency too high: {burst_latency}ms"
        assert gradual_latency < 600, f"Gradual load latency too high: {gradual_latency}ms"
        assert mixed_latency < 700, f"Mixed load latency too high: {mixed_latency}ms"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_during_load(self, temp_db, performance_runner):
        """Test memory efficiency during sustained load."""
        import psutil
        import os
        
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run sustained load
        load_prompts = ["Test memory efficiency"] * 50
        results = await performance_runner.run_concurrent_requests(
            router, load_prompts, 10
        )
        
        # Check memory after load
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase}MB increase"
        
        # Performance should still be good
        metrics = performance_runner.calculate_performance_metrics(results)
        assert metrics["success_rate"] >= 0.95
        assert metrics["performance_compliance"]["total_under_800ms_rate"] >= 0.85
    
    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, temp_db, performance_runner):
        """Test performance recovery after error conditions."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Simulate error conditions with edge case inputs
        error_prompts = [
            "",  # Empty prompt
            "x" * 10000,  # Very long prompt
            None,  # Invalid input
            "Normal prompt after errors"
        ]
        
        # Test that system recovers gracefully
        for prompt in error_prompts:
            try:
                if prompt is None:
                    result = await router.route_request("")
                else:
                    result = await router.route_request(prompt)
                
                # If no exception, should still meet performance requirements
                if "total_latency_ms" in result:
                    assert result["total_latency_ms"] < 800
                    assert result["llm_latency_ms"] < 500
                    
            except Exception:
                # Errors should be handled quickly (not hang the system)
                pass
        
        # System should continue to work normally after errors
        normal_result = await router.route_request("Normal request after error recovery")
        assert normal_result["total_latency_ms"] < 800
        assert normal_result["llm_latency_ms"] < 500
        assert "model_used" in normal_result
    
    @pytest.mark.asyncio
    async def test_cold_start_performance(self, temp_db):
        """Test performance during cold start conditions."""
        # Create fresh router instance (cold start)
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # First request (cold start)
        start_time = time.time()
        first_result = await router.route_request("Cold start test request")
        first_duration = (time.time() - start_time) * 1000
        
        # Subsequent requests (warm)
        warm_durations = []
        for i in range(5):
            start_time = time.time()
            await router.route_request(f"Warm request {i}")
            warm_durations.append((time.time() - start_time) * 1000)
        
        avg_warm_duration = sum(warm_durations) / len(warm_durations)
        
        # Cold start should still meet performance requirements
        assert first_result["total_latency_ms"] < 800
        assert first_result["llm_latency_ms"] < 500
        
        # Warm requests should be faster or similar
        assert avg_warm_duration <= first_duration * 1.2  # Allow 20% tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])