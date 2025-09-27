"""
Phase 5 Performance Testing Suite - OpenSesame Predictor
Comprehensive performance testing and benchmarking for optimized prediction pipeline.

This test suite validates Phase 5 optimization targets:
- LLM latency < 500ms
- ML scoring latency < 100ms  
- Total response latency < 800ms median
- Caching effectiveness and hit rates
- Async operation performance
- 100 iterations for statistical significance
"""

import pytest
import asyncio
import time
import statistics
import json
import psutil
import gc
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import requests
import threading
from dataclasses import dataclass
from pathlib import Path

# Import application components for direct testing
import sys
sys.path.append('/home/quantum/ApiCallPredictor')

from app.models.predictor import get_predictor
from app.models.ai_layer import AiLayer
from app.models.ml_ranker import MLRanker
from app.utils.feature_eng import FeatureExtractor
from app.utils.guardrails import SafetyValidator
from app.config import get_settings


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    latencies: List[float]
    avg_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float
    error_rate: float
    cache_hit_rate: float
    memory_usage_mb: float


class PerformanceTester:
    """
    Comprehensive performance testing framework for Phase 5 optimization validation
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.test_data_file = Path(__file__).parent / "fixtures" / "perf_test_data.json"
        self.results_dir = Path(__file__).parent / "performance_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Performance targets for Phase 5
        self.targets = {
            "llm_latency_ms": 500,
            "ml_scoring_latency_ms": 100,
            "total_response_latency_ms": 800,
            "cache_hit_rate": 0.8,
            "error_rate": 0.01,
            "throughput_rps": 10,
            "memory_limit_mb": 1024
        }
        
        # Test configuration
        self.test_iterations = 100
        self.concurrent_users = 20
        self.load_test_duration = 60  # seconds
        
        # Performance tracking
        self.test_results = {}
        self.start_time = None
        
    def setup_test_data(self):
        """Generate or load test data for performance testing"""
        test_prompts = [
            "Create a new user account with email validation",
            "Search for products by category and price range", 
            "Update user profile information and preferences",
            "Delete user account and all associated data",
            "Get user authentication status and permissions",
            "List all available API endpoints with documentation",
            "Process payment for order with credit card verification",
            "Send email notification to user with custom template",
            "Generate report of user activity for the last month",
            "Backup user data to external storage service",
            "Validate user input for security vulnerabilities",
            "Cache frequently accessed data for performance",
            "Log user actions for audit and compliance",
            "Synchronize data across multiple database replicas",
            "Monitor system health and resource utilization"
        ]
        
        test_histories = [
            [],  # Cold start scenario
            [{"api_call": "/api/auth/login", "method": "POST", "timestamp": "2024-01-01T10:00:00Z"}],
            [
                {"api_call": "/api/users", "method": "GET", "timestamp": "2024-01-01T10:00:00Z"},
                {"api_call": "/api/users/123", "method": "GET", "timestamp": "2024-01-01T10:01:00Z"}
            ],
            [
                {"api_call": "/api/products", "method": "GET", "timestamp": "2024-01-01T10:00:00Z"},
                {"api_call": "/api/products/search", "method": "POST", "timestamp": "2024-01-01T10:01:00Z"},
                {"api_call": "/api/products/456", "method": "GET", "timestamp": "2024-01-01T10:02:00Z"}
            ]
        ]
        
        test_data = {
            "prompts": test_prompts,
            "histories": test_histories,
            "test_configurations": [
                {"max_predictions": 3, "temperature": 0.7, "use_ml_ranking": True},
                {"max_predictions": 5, "temperature": 0.5, "use_ml_ranking": True},
                {"max_predictions": 3, "temperature": 0.9, "use_ml_ranking": False}
            ]
        }
        
        with open(self.test_data_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        return test_data
    
    def measure_component_latency(self, component_name: str, func, *args, **kwargs):
        """Measure latency of individual component operations"""
        start_time = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(func):
                result = asyncio.run(func(*args, **kwargs))
            else:
                result = func(*args, **kwargs)
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            return latency_ms, result, None
        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            return latency_ms, None, str(e)
    
    async def test_llm_latency_target(self) -> PerformanceMetrics:
        """
        Test LLM call latency to meet < 500ms target
        Measures direct AI layer performance over 100 iterations
        """
        print(f"\n🧠 Testing LLM Latency (Target: < {self.targets['llm_latency_ms']}ms)")
        
        ai_layer = AiLayer()
        latencies = []
        errors = 0
        
        test_data = self.setup_test_data()
        
        for i in range(self.test_iterations):
            prompt = test_data["prompts"][i % len(test_data["prompts"])]
            history = test_data["histories"][i % len(test_data["histories"])]
            
            start_time = time.perf_counter()
            try:
                predictions = await ai_layer.generate_predictions(
                    prompt=prompt,
                    history=history,
                    k=5,
                    temperature=0.7
                )
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                if i % 20 == 0:
                    print(f"  Iteration {i+1}: {latency_ms:.2f}ms")
                    
            except Exception as e:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                errors += 1
                print(f"  Error at iteration {i+1}: {e}")
        
        return self._calculate_metrics(latencies, errors, "LLM Latency")
    
    async def test_ml_scoring_latency_target(self) -> PerformanceMetrics:
        """
        Test ML scoring latency to meet < 100ms target
        Measures ML ranker performance with various candidate sizes
        """
        print(f"\n🤖 Testing ML Scoring Latency (Target: < {self.targets['ml_scoring_latency_ms']}ms)")
        
        ml_ranker = MLRanker()
        feature_extractor = FeatureExtractor()
        latencies = []
        errors = 0
        
        # Load or train ML model
        try:
            await ml_ranker.load_model()
        except:
            print("  Training ML model for testing...")
            await ml_ranker.train_ranker()
        
        test_data = self.setup_test_data()
        
        for i in range(self.test_iterations):
            prompt = test_data["prompts"][i % len(test_data["prompts"])]
            history = test_data["histories"][i % len(test_data["histories"])]
            
            # Generate mock AI predictions for ML ranking
            mock_predictions = [
                {
                    "api_call": f"GET /api/test_{j}",
                    "method": "GET",
                    "description": f"Test endpoint {j}",
                    "confidence": 0.8 - (j * 0.1),
                    "parameters": {}
                }
                for j in range(5)  # k=3, buffer=2
            ]
            
            start_time = time.perf_counter()
            try:
                ranked_predictions = await ml_ranker.rank_predictions(
                    predictions=mock_predictions,
                    prompt=prompt,
                    history=history,
                    k=3,
                    buffer=2
                )
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                if i % 20 == 0:
                    print(f"  Iteration {i+1}: {latency_ms:.2f}ms")
                    
            except Exception as e:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                errors += 1
                print(f"  Error at iteration {i+1}: {e}")
        
        return self._calculate_metrics(latencies, errors, "ML Scoring Latency")
    
    async def test_total_response_latency_target(self) -> PerformanceMetrics:
        """
        Test total end-to-end response latency to meet < 800ms median target
        Measures complete prediction pipeline over 100 iterations
        """
        print(f"\n⚡ Testing Total Response Latency (Target: < {self.targets['total_response_latency_ms']}ms median)")
        
        predictor = await get_predictor()
        latencies = []
        errors = 0
        cache_hits = 0
        
        test_data = self.setup_test_data()
        
        for i in range(self.test_iterations):
            prompt = test_data["prompts"][i % len(test_data["prompts"])]
            history = test_data["histories"][i % len(test_data["histories"])]
            config = test_data["test_configurations"][i % len(test_data["test_configurations"])]
            
            start_time = time.perf_counter()
            try:
                result = await predictor.predict(
                    prompt=prompt,
                    history=history,
                    max_predictions=config["max_predictions"],
                    temperature=config["temperature"],
                    use_ml_ranking=config["use_ml_ranking"]
                )
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                # Check if this was a cache hit
                if latency_ms < 50:  # Likely cache hit
                    cache_hits += 1
                
                if i % 20 == 0:
                    print(f"  Iteration {i+1}: {latency_ms:.2f}ms (predictions: {len(result.get('predictions', []))})")
                    
            except Exception as e:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                errors += 1
                print(f"  Error at iteration {i+1}: {e}")
        
        cache_hit_rate = cache_hits / len(latencies) if latencies else 0
        print(f"  Cache hit rate: {cache_hit_rate:.1%}")
        
        metrics = self._calculate_metrics(latencies, errors, "Total Response Latency")
        metrics.cache_hit_rate = cache_hit_rate
        return metrics
    
    async def test_caching_effectiveness(self) -> Dict[str, Any]:
        """
        Test caching effectiveness and hit rates
        Validates cache performance with repeated requests
        """
        print(f"\n💾 Testing Caching Effectiveness (Target: > {self.targets['cache_hit_rate']:.0%} hit rate)")
        
        predictor = await get_predictor()
        
        # Test with identical requests to trigger cache hits
        test_prompt = "Get user information for dashboard display"
        test_history = [{"api_call": "/api/auth/login", "method": "POST"}]
        
        cache_test_results = {
            "first_request_latencies": [],
            "cached_request_latencies": [],
            "cache_hit_count": 0,
            "total_requests": 0
        }
        
        # First request (cache miss)
        start_time = time.perf_counter()
        result1 = await predictor.predict(
            prompt=test_prompt,
            history=test_history,
            max_predictions=3,
            use_ml_ranking=True
        )
        first_latency = (time.perf_counter() - start_time) * 1000
        cache_test_results["first_request_latencies"].append(first_latency)
        cache_test_results["total_requests"] += 1
        
        print(f"  First request: {first_latency:.2f}ms")
        
        # Subsequent identical requests (should be cache hits)
        for i in range(10):
            start_time = time.perf_counter()
            result = await predictor.predict(
                prompt=test_prompt,
                history=test_history,
                max_predictions=3,
                use_ml_ranking=True
            )
            latency = (time.perf_counter() - start_time) * 1000
            cache_test_results["cached_request_latencies"].append(latency)
            cache_test_results["total_requests"] += 1
            
            # Detect cache hits by low latency
            if latency < 50:  # Cache hit threshold
                cache_test_results["cache_hit_count"] += 1
            
            if i % 3 == 0:
                print(f"  Cached request {i+1}: {latency:.2f}ms")
        
        # Test cache with variations
        variations = [
            {"max_predictions": 5},  # Different parameter
            {"temperature": 0.5},    # Different parameter
            {"use_ml_ranking": False}  # Different parameter
        ]
        
        for i, variation in enumerate(variations):
            params = {
                "prompt": test_prompt,
                "history": test_history,
                "max_predictions": 3,
                "temperature": 0.7,
                "use_ml_ranking": True,
                **variation
            }
            
            start_time = time.perf_counter()
            result = await predictor.predict(**params)
            latency = (time.perf_counter() - start_time) * 1000
            cache_test_results["total_requests"] += 1
            
            print(f"  Variation {i+1}: {latency:.2f}ms")
        
        cache_hit_rate = cache_test_results["cache_hit_count"] / cache_test_results["total_requests"]
        avg_cached_latency = statistics.mean(cache_test_results["cached_request_latencies"])
        
        print(f"  Cache hit rate: {cache_hit_rate:.1%}")
        print(f"  Average cached response time: {avg_cached_latency:.2f}ms")
        
        cache_test_results["cache_hit_rate"] = cache_hit_rate
        cache_test_results["avg_cached_latency"] = avg_cached_latency
        cache_test_results["cache_effectiveness"] = cache_hit_rate >= self.targets["cache_hit_rate"]
        
        return cache_test_results
    
    async def test_concurrent_performance(self) -> PerformanceMetrics:
        """
        Test performance under concurrent load
        Validates async operation performance with multiple concurrent requests
        """
        print(f"\n🚀 Testing Concurrent Performance ({self.concurrent_users} concurrent users)")
        
        predictor = await get_predictor()
        test_data = self.setup_test_data()
        
        async def single_request(request_id: int):
            """Single request for concurrent testing"""
            prompt = test_data["prompts"][request_id % len(test_data["prompts"])]
            history = test_data["histories"][request_id % len(test_data["histories"])]
            
            start_time = time.perf_counter()
            try:
                result = await predictor.predict(
                    prompt=prompt,
                    history=history,
                    max_predictions=3,
                    temperature=0.7,
                    use_ml_ranking=True
                )
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                return latency_ms, len(result.get("predictions", [])), None
            except Exception as e:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                return latency_ms, 0, str(e)
        
        # Launch concurrent requests
        start_time = time.perf_counter()
        tasks = [single_request(i) for i in range(self.concurrent_users)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.perf_counter() - start_time
        
        # Process results
        latencies = []
        errors = 0
        successful_predictions = 0
        
        for result in results:
            if isinstance(result, Exception):
                errors += 1
                latencies.append(5000)  # Penalty for exceptions
            else:
                latency, prediction_count, error = result
                latencies.append(latency)
                if error:
                    errors += 1
                else:
                    successful_predictions += prediction_count
        
        throughput = len(results) / total_duration
        
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} RPS")
        print(f"  Successful predictions: {successful_predictions}")
        
        metrics = self._calculate_metrics(latencies, errors, "Concurrent Performance")
        metrics.throughput = throughput
        return metrics
    
    async def test_memory_usage(self) -> Dict[str, Any]:
        """
        Test memory usage during sustained operation
        Validates memory limits and leak detection
        """
        print(f"\n🧠 Testing Memory Usage (Limit: < {self.targets['memory_limit_mb']}MB)")
        
        predictor = await get_predictor()
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        print(f"  Baseline memory: {baseline_memory:.2f}MB")
        
        memory_samples = []
        test_data = self.setup_test_data()
        
        # Run sustained predictions and monitor memory
        for i in range(50):  # Reduced for performance
            prompt = test_data["prompts"][i % len(test_data["prompts"])]
            history = test_data["histories"][i % len(test_data["histories"])]
            
            await predictor.predict(
                prompt=prompt,
                history=history,
                max_predictions=3,
                temperature=0.7,
                use_ml_ranking=True
            )
            
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                print(f"  Memory at iteration {i+1}: {current_memory:.2f}MB")
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_samples.append(final_memory)
        
        memory_stats = {
            "baseline_memory_mb": baseline_memory,
            "final_memory_mb": final_memory,
            "max_memory_mb": max(memory_samples),
            "avg_memory_mb": statistics.mean(memory_samples),
            "memory_growth_mb": final_memory - baseline_memory,
            "memory_within_limit": max(memory_samples) < self.targets["memory_limit_mb"],
            "memory_samples": memory_samples
        }
        
        print(f"  Final memory: {final_memory:.2f}MB")
        print(f"  Memory growth: {memory_stats['memory_growth_mb']:.2f}MB")
        print(f"  Max memory: {memory_stats['max_memory_mb']:.2f}MB")
        
        return memory_stats
    
    async def test_k_buffer_filtering_performance(self) -> Dict[str, Any]:
        """
        Test k+buffer candidate filtering performance (k=3, buffer=2)
        Validates the efficiency of the candidate generation and filtering strategy
        """
        print(f"\n🎯 Testing k+buffer Filtering Performance (k=3, buffer=2)")
        
        predictor = await get_predictor()
        test_data = self.setup_test_data()
        
        filtering_stats = {
            "total_tests": 0,
            "candidates_generated": [],
            "candidates_returned": [],
            "filtering_latencies": [],
            "ml_ranking_latencies": [],
            "safety_filtering_counts": []
        }
        
        for i in range(50):  # Focused test on filtering
            prompt = test_data["prompts"][i % len(test_data["prompts"])]
            history = test_data["histories"][i % len(test_data["histories"])]
            
            start_time = time.perf_counter()
            result = await predictor.predict(
                prompt=prompt,
                history=history,
                max_predictions=3,
                temperature=0.7,
                use_ml_ranking=True
            )
            filtering_latency = (time.perf_counter() - start_time) * 1000
            
            # Extract metrics from result metadata
            metadata = result.get("metadata", {})
            candidates_generated = metadata.get("candidates_generated", 0)
            candidates_returned = len(result.get("predictions", []))
            safety_filtered = metadata.get("unsafe_candidates_removed", 0)
            
            filtering_stats["total_tests"] += 1
            filtering_stats["candidates_generated"].append(candidates_generated)
            filtering_stats["candidates_returned"].append(candidates_returned)
            filtering_stats["filtering_latencies"].append(filtering_latency)
            filtering_stats["safety_filtering_counts"].append(safety_filtered)
            
            if i % 10 == 0:
                print(f"  Test {i+1}: {candidates_generated}→{candidates_returned} candidates, {filtering_latency:.2f}ms")
        
        # Calculate statistics
        filtering_stats["avg_candidates_generated"] = statistics.mean(filtering_stats["candidates_generated"])
        filtering_stats["avg_candidates_returned"] = statistics.mean(filtering_stats["candidates_returned"])
        filtering_stats["avg_filtering_latency"] = statistics.mean(filtering_stats["filtering_latencies"])
        filtering_stats["total_safety_filtered"] = sum(filtering_stats["safety_filtering_counts"])
        filtering_stats["avg_safety_filtered"] = statistics.mean(filtering_stats["safety_filtering_counts"])
        
        print(f"  Average candidates generated: {filtering_stats['avg_candidates_generated']:.1f}")
        print(f"  Average candidates returned: {filtering_stats['avg_candidates_returned']:.1f}")
        print(f"  Average filtering latency: {filtering_stats['avg_filtering_latency']:.2f}ms")
        print(f"  Total safety filtered: {filtering_stats['total_safety_filtered']}")
        
        return filtering_stats
    
    def _calculate_metrics(self, latencies: List[float], errors: int, test_name: str) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics from latency data"""
        if not latencies:
            return PerformanceMetrics([], 0, 0, 0, 0, 0, 1.0, 0, 0)
        
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        error_rate = errors / len(latencies) if latencies else 1.0
        
        # Get current memory usage
        try:
            process = psutil.Process()
            memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except:
            memory_usage_mb = 0
        
        metrics = PerformanceMetrics(
            latencies=latencies,
            avg_latency=avg_latency,
            median_latency=median_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            throughput=0,  # Will be calculated separately if needed
            error_rate=error_rate,
            cache_hit_rate=0,  # Will be calculated separately if needed
            memory_usage_mb=memory_usage_mb
        )
        
        print(f"  {test_name} Results:")
        print(f"    Average: {avg_latency:.2f}ms")
        print(f"    Median: {median_latency:.2f}ms")
        print(f"    P95: {p95_latency:.2f}ms")
        print(f"    P99: {p99_latency:.2f}ms")
        print(f"    Error rate: {error_rate:.1%}")
        
        return metrics
    
    def generate_performance_report(self, all_results: Dict[str, Any]) -> str:
        """Generate comprehensive performance testing report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# OpenSesame Predictor - Phase 5 Performance Testing Report
Generated: {timestamp}

## Executive Summary
This report validates Phase 5 optimization targets through comprehensive performance testing.

## Performance Targets vs Results

### LLM Latency Target: < {self.targets['llm_latency_ms']}ms
"""
        
        if "llm_latency" in all_results:
            llm_metrics = all_results["llm_latency"]
            target_met = llm_metrics.avg_latency < self.targets['llm_latency_ms']
            report += f"""
- **Average Latency**: {llm_metrics.avg_latency:.2f}ms {'✅' if target_met else '❌'}
- **Median Latency**: {llm_metrics.median_latency:.2f}ms
- **P95 Latency**: {llm_metrics.p95_latency:.2f}ms
- **P99 Latency**: {llm_metrics.p99_latency:.2f}ms
- **Error Rate**: {llm_metrics.error_rate:.1%}
- **Target Met**: {'Yes' if target_met else 'No'}
"""
        
        report += f"""
### ML Scoring Latency Target: < {self.targets['ml_scoring_latency_ms']}ms
"""
        
        if "ml_scoring" in all_results:
            ml_metrics = all_results["ml_scoring"]
            target_met = ml_metrics.avg_latency < self.targets['ml_scoring_latency_ms']
            report += f"""
- **Average Latency**: {ml_metrics.avg_latency:.2f}ms {'✅' if target_met else '❌'}
- **Median Latency**: {ml_metrics.median_latency:.2f}ms
- **P95 Latency**: {ml_metrics.p95_latency:.2f}ms
- **Error Rate**: {ml_metrics.error_rate:.1%}
- **Target Met**: {'Yes' if target_met else 'No'}
"""
        
        report += f"""
### Total Response Latency Target: < {self.targets['total_response_latency_ms']}ms (Median)
"""
        
        if "total_response" in all_results:
            total_metrics = all_results["total_response"]
            target_met = total_metrics.median_latency < self.targets['total_response_latency_ms']
            report += f"""
- **Average Latency**: {total_metrics.avg_latency:.2f}ms
- **Median Latency**: {total_metrics.median_latency:.2f}ms {'✅' if target_met else '❌'}
- **P95 Latency**: {total_metrics.p95_latency:.2f}ms
- **P99 Latency**: {total_metrics.p99_latency:.2f}ms
- **Cache Hit Rate**: {total_metrics.cache_hit_rate:.1%}
- **Target Met**: {'Yes' if target_met else 'No'}
"""
        
        report += f"""
### Caching Effectiveness Target: > {self.targets['cache_hit_rate']:.0%} Hit Rate
"""
        
        if "caching" in all_results:
            cache_results = all_results["caching"]
            target_met = cache_results["cache_hit_rate"] >= self.targets["cache_hit_rate"]
            report += f"""
- **Cache Hit Rate**: {cache_results['cache_hit_rate']:.1%} {'✅' if target_met else '❌'}
- **Average Cached Response**: {cache_results['avg_cached_latency']:.2f}ms
- **First Request Latency**: {statistics.mean(cache_results['first_request_latencies']):.2f}ms
- **Cache Effectiveness**: {'Yes' if cache_results['cache_effectiveness'] else 'No'}
- **Target Met**: {'Yes' if target_met else 'No'}
"""
        
        report += """
## Concurrent Performance Analysis
"""
        
        if "concurrent" in all_results:
            concurrent_metrics = all_results["concurrent"]
            throughput_met = concurrent_metrics.throughput >= self.targets["throughput_rps"]
            report += f"""
- **Throughput**: {concurrent_metrics.throughput:.2f} RPS {'✅' if throughput_met else '❌'}
- **Average Latency**: {concurrent_metrics.avg_latency:.2f}ms
- **P95 Latency**: {concurrent_metrics.p95_latency:.2f}ms
- **Error Rate**: {concurrent_metrics.error_rate:.1%}
- **Concurrent Users**: {self.concurrent_users}
"""
        
        report += """
## Memory Usage Analysis
"""
        
        if "memory" in all_results:
            memory_results = all_results["memory"]
            memory_met = memory_results["memory_within_limit"]
            report += f"""
- **Baseline Memory**: {memory_results['baseline_memory_mb']:.2f}MB
- **Maximum Memory**: {memory_results['max_memory_mb']:.2f}MB {'✅' if memory_met else '❌'}
- **Final Memory**: {memory_results['final_memory_mb']:.2f}MB
- **Memory Growth**: {memory_results['memory_growth_mb']:.2f}MB
- **Within Limit**: {'Yes' if memory_met else 'No'}
"""
        
        report += """
## k+buffer Filtering Analysis
"""
        
        if "k_buffer" in all_results:
            filter_results = all_results["k_buffer"]
            report += f"""
- **Average Candidates Generated**: {filter_results['avg_candidates_generated']:.1f}
- **Average Candidates Returned**: {filter_results['avg_candidates_returned']:.1f}
- **Average Filtering Latency**: {filter_results['avg_filtering_latency']:.2f}ms
- **Total Safety Filtered**: {filter_results['total_safety_filtered']}
- **Average Safety Filtered**: {filter_results['avg_safety_filtered']:.1f}
"""
        
        report += f"""
## Test Configuration
- **Test Iterations**: {self.test_iterations}
- **Concurrent Users**: {self.concurrent_users}
- **Test Duration**: {self.load_test_duration}s
- **k+buffer Strategy**: k=3, buffer=2

## Recommendations
"""
        
        # Generate recommendations based on results
        recommendations = []
        
        if "llm_latency" in all_results and all_results["llm_latency"].avg_latency >= self.targets['llm_latency_ms']:
            recommendations.append("- Consider caching LLM responses for common prompts")
            recommendations.append("- Evaluate switching to faster LLM models")
        
        if "total_response" in all_results and all_results["total_response"].median_latency >= self.targets['total_response_latency_ms']:
            recommendations.append("- Optimize ML ranking algorithm for faster processing")
            recommendations.append("- Implement request batching for better throughput")
        
        if "caching" in all_results and all_results["caching"]["cache_hit_rate"] < self.targets["cache_hit_rate"]:
            recommendations.append("- Tune cache key generation for better hit rates")
            recommendations.append("- Increase cache TTL for stable predictions")
        
        if not recommendations:
            recommendations.append("- All performance targets met! Consider further optimizations for even better performance.")
        
        report += "\n".join(recommendations)
        
        report += """

## Conclusion
This performance test validates the Phase 5 optimization targets and provides insights for further improvements.
"""
        
        return report
    
    async def run_full_performance_test_suite(self) -> Dict[str, Any]:
        """
        Run the complete performance testing suite
        Returns comprehensive results for Phase 5 validation
        """
        print("🚀 Starting Phase 5 Performance Testing Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        all_results = {}
        
        # Run all performance tests
        try:
            all_results["llm_latency"] = await self.test_llm_latency_target()
        except Exception as e:
            print(f"❌ LLM latency test failed: {e}")
            all_results["llm_latency"] = None
        
        try:
            all_results["ml_scoring"] = await self.test_ml_scoring_latency_target()
        except Exception as e:
            print(f"❌ ML scoring test failed: {e}")
            all_results["ml_scoring"] = None
        
        try:
            all_results["total_response"] = await self.test_total_response_latency_target()
        except Exception as e:
            print(f"❌ Total response test failed: {e}")
            all_results["total_response"] = None
        
        try:
            all_results["caching"] = await self.test_caching_effectiveness()
        except Exception as e:
            print(f"❌ Caching test failed: {e}")
            all_results["caching"] = None
        
        try:
            all_results["concurrent"] = await self.test_concurrent_performance()
        except Exception as e:
            print(f"❌ Concurrent test failed: {e}")
            all_results["concurrent"] = None
        
        try:
            all_results["memory"] = await self.test_memory_usage()
        except Exception as e:
            print(f"❌ Memory test failed: {e}")
            all_results["memory"] = None
        
        try:
            all_results["k_buffer"] = await self.test_k_buffer_filtering_performance()
        except Exception as e:
            print(f"❌ k+buffer test failed: {e}")
            all_results["k_buffer"] = None
        
        # Generate and save report
        report = self.generate_performance_report(all_results)
        report_file = self.results_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save raw results
        results_file = self.results_dir / f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for key, value in all_results.items():
            if isinstance(value, PerformanceMetrics):
                serializable_results[key] = {
                    "avg_latency": value.avg_latency,
                    "median_latency": value.median_latency,
                    "p95_latency": value.p95_latency,
                    "p99_latency": value.p99_latency,
                    "throughput": value.throughput,
                    "error_rate": value.error_rate,
                    "cache_hit_rate": value.cache_hit_rate,
                    "memory_usage_mb": value.memory_usage_mb,
                    "latency_samples": value.latencies[:10]  # Save first 10 samples
                }
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        total_duration = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print(f"✅ Performance Testing Suite Completed in {total_duration:.1f}s")
        print(f"📊 Report saved to: {report_file}")
        print(f"💾 Raw results saved to: {results_file}")
        
        return all_results


# Pytest test functions for integration with test suite
class TestPhase5Performance:
    """Phase 5 Performance Test Cases for pytest integration"""
    
    @pytest.fixture(scope="class")
    def performance_tester(self):
        """Initialize performance tester"""
        return PerformanceTester()
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_llm_latency_target(self, performance_tester):
        """Test LLM latency meets < 500ms target"""
        metrics = await performance_tester.test_llm_latency_target()
        assert metrics.avg_latency < performance_tester.targets['llm_latency_ms'], \
            f"LLM latency {metrics.avg_latency:.2f}ms exceeds target {performance_tester.targets['llm_latency_ms']}ms"
        assert metrics.error_rate < 0.05, f"LLM error rate {metrics.error_rate:.1%} too high"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_ml_scoring_latency_target(self, performance_tester):
        """Test ML scoring latency meets < 100ms target"""
        metrics = await performance_tester.test_ml_scoring_latency_target()
        assert metrics.avg_latency < performance_tester.targets['ml_scoring_latency_ms'], \
            f"ML scoring latency {metrics.avg_latency:.2f}ms exceeds target {performance_tester.targets['ml_scoring_latency_ms']}ms"
        assert metrics.error_rate < 0.05, f"ML scoring error rate {metrics.error_rate:.1%} too high"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_total_response_latency_target(self, performance_tester):
        """Test total response latency meets < 800ms median target"""
        metrics = await performance_tester.test_total_response_latency_target()
        assert metrics.median_latency < performance_tester.targets['total_response_latency_ms'], \
            f"Total response median latency {metrics.median_latency:.2f}ms exceeds target {performance_tester.targets['total_response_latency_ms']}ms"
        assert metrics.error_rate < 0.02, f"Total response error rate {metrics.error_rate:.1%} too high"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_caching_effectiveness(self, performance_tester):
        """Test caching effectiveness meets > 80% hit rate target"""
        results = await performance_tester.test_caching_effectiveness()
        assert results["cache_hit_rate"] >= performance_tester.targets["cache_hit_rate"], \
            f"Cache hit rate {results['cache_hit_rate']:.1%} below target {performance_tester.targets['cache_hit_rate']:.0%}"
        assert results["avg_cached_latency"] < 50, \
            f"Cached response latency {results['avg_cached_latency']:.2f}ms too high"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_performance(self, performance_tester):
        """Test concurrent performance meets throughput targets"""
        metrics = await performance_tester.test_concurrent_performance()
        assert metrics.throughput >= performance_tester.targets["throughput_rps"], \
            f"Throughput {metrics.throughput:.2f} RPS below target {performance_tester.targets['throughput_rps']} RPS"
        assert metrics.error_rate < 0.05, f"Concurrent error rate {metrics.error_rate:.1%} too high"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_limits(self, performance_tester):
        """Test memory usage stays within limits"""
        results = await performance_tester.test_memory_usage()
        assert results["memory_within_limit"], \
            f"Memory usage {results['max_memory_mb']:.2f}MB exceeds limit {performance_tester.targets['memory_limit_mb']}MB"
        assert results["memory_growth_mb"] < 200, \
            f"Memory growth {results['memory_growth_mb']:.2f}MB too high"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_k_buffer_filtering_efficiency(self, performance_tester):
        """Test k+buffer filtering strategy efficiency"""
        results = await performance_tester.test_k_buffer_filtering_performance()
        assert results["avg_candidates_generated"] >= 3, "Not enough candidates generated"
        assert results["avg_candidates_returned"] <= 3, "Too many candidates returned"
        assert results["avg_filtering_latency"] < 100, \
            f"Filtering latency {results['avg_filtering_latency']:.2f}ms too high"
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_full_performance_suite(self, performance_tester):
        """Run the complete performance testing suite"""
        results = await performance_tester.run_full_performance_test_suite()
        
        # Validate key performance targets
        if results.get("llm_latency"):
            assert results["llm_latency"].avg_latency < 500, "LLM latency target not met"
        
        if results.get("ml_scoring"):
            assert results["ml_scoring"].avg_latency < 100, "ML scoring latency target not met"
        
        if results.get("total_response"):
            assert results["total_response"].median_latency < 800, "Total response latency target not met"
        
        if results.get("caching"):
            assert results["caching"]["cache_hit_rate"] >= 0.8, "Cache hit rate target not met"


# Command-line execution
if __name__ == "__main__":
    async def main():
        """Run performance tests from command line"""
        tester = PerformanceTester()
        await tester.run_full_performance_test_suite()
    
    asyncio.run(main())