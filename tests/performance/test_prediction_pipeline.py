"""
Performance tests for the prediction pipeline.

Tests performance characteristics including:
- Response time under load
- Memory usage and leaks
- Concurrent request handling
- Throughput measurements
- Resource utilization
- Bottleneck identification
"""

import pytest
import asyncio
import time
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
from typing import List, Dict, Any

# Mock imports - replace with actual imports when modules exist
# from app.models.predictor import Predictor
# from app.main import app


class TestPredictionPerformance:
    """Performance tests for prediction functionality."""
    
    @pytest.fixture
    def performance_config(self):
        """Performance test configuration."""
        return {
            "max_response_time_ms": 800,
            "target_throughput_rps": 10,
            "memory_limit_mb": 512,
            "concurrent_requests": 20,
            "load_test_duration": 30
        }
    
    @pytest.fixture
    def sample_prediction_request(self, sample_openapi_spec, sample_user_events):
        """Standard prediction request for performance testing."""
        return {
            "prompt": "Create a new user account",
            "recent_events": sample_user_events,
            "openapi_spec": sample_openapi_spec,
            "k": 5
        }
    
    @pytest.mark.performance
    def test_single_prediction_response_time(self, sample_prediction_request, performance_config):
        """Test single prediction response time meets requirements."""
        # Mock predictor
        predictor = MockPredictor()
        
        response_times = []
        
        # Run 100 predictions to get statistical data
        for _ in range(100):
            start_time = time.time()
            
            result = predictor.predict(
                prompt=sample_prediction_request["prompt"],
                recent_events=sample_prediction_request["recent_events"],
                openapi_spec=sample_prediction_request["openapi_spec"],
                k=sample_prediction_request["k"]
            )
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
        
        # Statistical analysis
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        
        # Assertions
        assert avg_response_time < performance_config["max_response_time_ms"]
        assert p95_response_time < performance_config["max_response_time_ms"] * 1.2
        assert p99_response_time < performance_config["max_response_time_ms"] * 1.5
        
        print(f"Performance Results:")
        print(f"  Average: {avg_response_time:.2f}ms")
        print(f"  95th percentile: {p95_response_time:.2f}ms") 
        print(f"  99th percentile: {p99_response_time:.2f}ms")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_prediction_performance(self, sample_prediction_request, performance_config):
        """Test performance under concurrent load."""
        predictor = MockAsyncPredictor()
        concurrent_requests = performance_config["concurrent_requests"]
        
        async def single_prediction():
            start_time = time.time()
            
            result = await predictor.predict(
                prompt=sample_prediction_request["prompt"],
                recent_events=sample_prediction_request["recent_events"],
                openapi_spec=sample_prediction_request["openapi_spec"],
                k=sample_prediction_request["k"]
            )
            
            end_time = time.time()
            return (end_time - start_time) * 1000
        
        # Launch concurrent requests
        start_time = time.time()
        tasks = [single_prediction() for _ in range(concurrent_requests)]
        response_times = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_duration = end_time - start_time
        throughput = concurrent_requests / total_duration
        
        # Performance assertions
        avg_response_time = statistics.mean(response_times)
        assert avg_response_time < performance_config["max_response_time_ms"] * 1.5  # Allow 50% degradation
        assert throughput >= performance_config["target_throughput_rps"] * 0.8  # Allow 20% degradation
        
        print(f"Concurrent Performance Results:")
        print(f"  Requests: {concurrent_requests}")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} RPS")
        print(f"  Average response time: {avg_response_time:.2f}ms")
    
    @pytest.mark.performance
    def test_memory_usage_under_load(self, sample_prediction_request, performance_config):
        """Test memory usage during sustained load."""
        predictor = MockPredictor()
        
        # Get baseline memory usage
        gc.collect()  # Force garbage collection
        process = psutil.Process()
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024
        
        memory_samples = []
        
        # Run predictions and monitor memory
        for i in range(1000):
            result = predictor.predict(
                prompt=sample_prediction_request["prompt"],
                recent_events=sample_prediction_request["recent_events"],
                openapi_spec=sample_prediction_request["openapi_spec"],
                k=sample_prediction_request["k"]
            )
            
            if i % 100 == 0:  # Sample memory every 100 requests
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory_mb)
        
        # Final memory check
        gc.collect()
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Memory usage analysis
        max_memory_mb = max(memory_samples)
        memory_growth_mb = final_memory_mb - baseline_memory_mb
        
        # Assertions
        assert max_memory_mb < performance_config["memory_limit_mb"]
        assert memory_growth_mb < 50  # Should not grow more than 50MB
        assert final_memory_mb < baseline_memory_mb * 1.2  # Should return close to baseline
        
        print(f"Memory Usage Results:")
        print(f"  Baseline: {baseline_memory_mb:.2f}MB")
        print(f"  Maximum: {max_memory_mb:.2f}MB")
        print(f"  Final: {final_memory_mb:.2f}MB")
        print(f"  Growth: {memory_growth_mb:.2f}MB")
    
    @pytest.mark.performance
    def test_ai_layer_performance(self, sample_openapi_spec, sample_user_events):
        """Test AI layer performance specifically."""
        ai_layer = MockAILayer()
        
        response_times = []
        
        for _ in range(50):
            start_time = time.time()
            
            candidates = ai_layer.generate_candidates(
                prompt="Create a new user",
                recent_events=sample_user_events,
                openapi_spec=sample_openapi_spec,
                k=10
            )
            
            end_time = time.time()
            response_times.append((end_time - start_time) * 1000)
        
        avg_ai_response_time = statistics.mean(response_times)
        
        # AI layer should respond within 500ms (per requirements)
        assert avg_ai_response_time < 500
        
        print(f"AI Layer Performance:")
        print(f"  Average response time: {avg_ai_response_time:.2f}ms")
    
    @pytest.mark.performance
    def test_ml_ranker_performance(self):
        """Test ML ranker performance."""
        ml_ranker = MockMLRanker()
        
        # Generate test candidates
        candidates = [
            {"endpoint": f"/endpoint_{i}", "method": "GET", "confidence": 0.8}
            for i in range(100)
        ]
        
        response_times = []
        
        for _ in range(100):
            start_time = time.time()
            
            ranked_candidates = ml_ranker.rank_candidates(candidates)
            
            end_time = time.time()
            response_times.append((end_time - start_time) * 1000)
        
        avg_ml_response_time = statistics.mean(response_times)
        
        # ML ranker should respond within 100ms (per requirements)
        assert avg_ml_response_time < 100
        
        print(f"ML Ranker Performance:")
        print(f"  Average response time: {avg_ml_response_time:.2f}ms")
        print(f"  Candidates processed: {len(candidates)}")
    
    @pytest.mark.performance
    def test_spec_parser_performance(self, sample_openapi_spec):
        """Test OpenAPI spec parsing performance."""
        spec_parser = MockSpecParser()
        
        # Test with various spec sizes
        spec_sizes = [10, 50, 100, 500, 1000]  # Number of endpoints
        
        for size in spec_sizes:
            # Generate spec with specified number of endpoints
            large_spec = generate_large_openapi_spec(size)
            
            start_time = time.time()
            endpoints = spec_parser.parse_spec(large_spec)
            end_time = time.time()
            
            parse_time_ms = (end_time - start_time) * 1000
            
            # Should parse 1000 endpoints in under 1 second
            max_time_ms = size * 1  # 1ms per endpoint
            assert parse_time_ms < max_time_ms
            
            print(f"Spec Parser - {size} endpoints: {parse_time_ms:.2f}ms")
    
    @pytest.mark.performance
    def test_cache_performance(self):
        """Test caching system performance."""
        cache = MockCache()
        
        # Test cache writes
        write_times = []
        for i in range(1000):
            start_time = time.time()
            cache.set(f"key_{i}", f"value_{i}")
            end_time = time.time()
            write_times.append((end_time - start_time) * 1000000)  # microseconds
        
        avg_write_time_us = statistics.mean(write_times)
        
        # Test cache reads
        read_times = []
        for i in range(1000):
            start_time = time.time()
            value = cache.get(f"key_{i}")
            end_time = time.time()
            read_times.append((end_time - start_time) * 1000000)  # microseconds
        
        avg_read_time_us = statistics.mean(read_times)
        
        # Cache operations should be very fast
        assert avg_write_time_us < 100  # < 0.1ms
        assert avg_read_time_us < 50   # < 0.05ms
        
        print(f"Cache Performance:")
        print(f"  Average write time: {avg_write_time_us:.2f}μs")
        print(f"  Average read time: {avg_read_time_us:.2f}μs")


class TestLoadTesting:
    """Load testing scenarios."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_sustained_load(self, sample_prediction_request, performance_config):
        """Test performance under sustained load."""
        predictor = MockPredictor()
        
        duration = performance_config["load_test_duration"]
        target_rps = performance_config["target_throughput_rps"]
        
        start_time = time.time()
        end_time = start_time + duration
        request_count = 0
        response_times = []
        errors = 0
        
        while time.time() < end_time:
            try:
                request_start = time.time()
                
                result = predictor.predict(
                    prompt=sample_prediction_request["prompt"],
                    recent_events=sample_prediction_request["recent_events"],
                    openapi_spec=sample_prediction_request["openapi_spec"],
                    k=sample_prediction_request["k"]
                )
                
                request_end = time.time()
                response_times.append((request_end - request_start) * 1000)
                request_count += 1
                
            except Exception as e:
                errors += 1
            
            # Rate limiting to achieve target RPS
            time.sleep(max(0, (1.0 / target_rps) - (time.time() - request_start)))
        
        total_duration = time.time() - start_time
        actual_rps = request_count / total_duration
        error_rate = errors / (request_count + errors) if (request_count + errors) > 0 else 0
        
        # Performance assertions
        assert actual_rps >= target_rps * 0.9  # Within 10% of target
        assert error_rate < 0.01  # Less than 1% error rate
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            assert avg_response_time < 1000  # Should not degrade significantly
            
            print(f"Load Test Results ({duration}s):")
            print(f"  Target RPS: {target_rps}")
            print(f"  Actual RPS: {actual_rps:.2f}")
            print(f"  Total requests: {request_count}")
            print(f"  Error rate: {error_rate:.2%}")
            print(f"  Average response time: {avg_response_time:.2f}ms")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_spike_load(self, sample_prediction_request):
        """Test performance under sudden load spikes."""
        predictor = MockPredictor()
        
        # Baseline load
        baseline_rps = 5
        spike_rps = 50
        spike_duration = 10  # seconds
        
        def run_requests_at_rate(rps, duration):
            start_time = time.time()
            end_time = start_time + duration
            response_times = []
            errors = 0
            
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    
                    result = predictor.predict(
                        prompt=sample_prediction_request["prompt"],
                        recent_events=sample_prediction_request["recent_events"],
                        openapi_spec=sample_prediction_request["openapi_spec"],
                        k=sample_prediction_request["k"]
                    )
                    
                    request_end = time.time()
                    response_times.append((request_end - request_start) * 1000)
                    
                except Exception:
                    errors += 1
                
                time.sleep(max(0, (1.0 / rps) - (time.time() - request_start)))
            
            return response_times, errors
        
        # Run baseline load
        print("Running baseline load...")
        baseline_times, baseline_errors = run_requests_at_rate(baseline_rps, 10)
        
        # Run spike load
        print("Running spike load...")
        spike_times, spike_errors = run_requests_at_rate(spike_rps, spike_duration)
        
        # Run recovery period
        print("Running recovery period...")
        recovery_times, recovery_errors = run_requests_at_rate(baseline_rps, 10)
        
        # Analysis
        baseline_avg = statistics.mean(baseline_times) if baseline_times else 0
        spike_avg = statistics.mean(spike_times) if spike_times else 0
        recovery_avg = statistics.mean(recovery_times) if recovery_times else 0
        
        # Assertions
        assert spike_avg < baseline_avg * 3  # Should not degrade more than 3x
        assert recovery_avg < baseline_avg * 1.2  # Should recover to within 20%
        assert spike_errors / (len(spike_times) + spike_errors) < 0.05 if spike_times else True  # < 5% error rate
        
        print(f"Spike Test Results:")
        print(f"  Baseline avg: {baseline_avg:.2f}ms")
        print(f"  Spike avg: {spike_avg:.2f}ms")
        print(f"  Recovery avg: {recovery_avg:.2f}ms")


# Mock classes for testing
class MockPredictor:
    """Mock predictor for performance testing."""
    
    def predict(self, prompt, recent_events, openapi_spec, k):
        # Simulate processing time
        time.sleep(0.1 + (k * 0.01))  # Base time + time per candidate
        
        return {
            "predictions": [
                {"endpoint": f"/test_{i}", "method": "GET", "confidence": 0.8}
                for i in range(k)
            ],
            "metadata": {"processing_time_ms": 100}
        }


class MockAsyncPredictor:
    """Mock async predictor for performance testing."""
    
    async def predict(self, prompt, recent_events, openapi_spec, k):
        # Simulate async processing time
        await asyncio.sleep(0.1 + (k * 0.01))
        
        return {
            "predictions": [
                {"endpoint": f"/test_{i}", "method": "GET", "confidence": 0.8}
                for i in range(k)
            ],
            "metadata": {"processing_time_ms": 100}
        }


class MockAILayer:
    """Mock AI layer for performance testing."""
    
    def generate_candidates(self, prompt, recent_events, openapi_spec, k):
        # Simulate AI processing time
        time.sleep(0.3 + (len(openapi_spec.get("paths", {})) * 0.001))
        
        return [
            {"endpoint": f"/ai_candidate_{i}", "method": "GET", "confidence": 0.8}
            for i in range(k)
        ]


class MockMLRanker:
    """Mock ML ranker for performance testing."""
    
    def rank_candidates(self, candidates):
        # Simulate ML processing time
        time.sleep(0.05 + (len(candidates) * 0.0001))
        
        return sorted(candidates, key=lambda x: x.get("confidence", 0), reverse=True)


class MockSpecParser:
    """Mock spec parser for performance testing."""
    
    def parse_spec(self, spec):
        # Simulate parsing time
        paths = spec.get("paths", {})
        time.sleep(len(paths) * 0.0001)
        
        return [{"path": path, "method": "GET"} for path in paths.keys()]


class MockCache:
    """Mock cache for performance testing."""
    
    def __init__(self):
        self._cache = {}
    
    def set(self, key, value):
        # Simulate cache write time
        time.sleep(0.00001)  # 10 microseconds
        self._cache[key] = value
    
    def get(self, key):
        # Simulate cache read time
        time.sleep(0.000005)  # 5 microseconds
        return self._cache.get(key)


def generate_large_openapi_spec(num_endpoints):
    """Generate large OpenAPI spec for testing."""
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Large API", "version": "1.0.0"},
        "paths": {}
    }
    
    for i in range(num_endpoints):
        spec["paths"][f"/endpoint_{i}"] = {
            "get": {
                "operationId": f"getEndpoint{i}",
                "description": f"Get endpoint {i} data"
            }
        }
    
    return spec