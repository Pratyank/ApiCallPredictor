#!/usr/bin/env python3
"""
Performance Test Scenarios for Feature Engineering Optimization
Provides benchmarking tools and test scenarios for validating
the enhanced feature engineering system performance.
"""

import asyncio
import time
import json
import statistics
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineeringBenchmark:
    """
    Comprehensive benchmarking suite for feature engineering performance
    """
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.results = []
        self.cache_hit_counts = 0
        self.cache_miss_counts = 0
        
    async def load_test_scenarios(self, scenario_files: List[str]) -> List[Dict[str, Any]]:
        """Load test scenarios from JSON files"""
        all_scenarios = []
        
        for file_path in scenario_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    scenarios = data.get('scenarios', [])
                    all_scenarios.extend(scenarios)
                    logger.info(f"Loaded {len(scenarios)} scenarios from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load scenarios from {file_path}: {e}")
                
        return all_scenarios
    
    async def benchmark_extraction_speed(
        self, 
        scenarios: List[Dict[str, Any]], 
        iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Benchmark feature extraction speed across different scenarios
        """
        logger.info(f"Starting speed benchmark with {iterations} iterations")
        
        extraction_times = []
        start_time = time.time()
        
        # Warm up the cache with a few requests
        for i in range(min(10, len(scenarios))):
            scenario = scenarios[i % len(scenarios)]
            await self._extract_features_for_scenario(scenario)
        
        cache_start_hits = getattr(self.feature_extractor, 'cache_hits', 0)
        cache_start_misses = getattr(self.feature_extractor, 'cache_misses', 0)
        
        # Main benchmark loop
        for i in range(iterations):
            scenario = scenarios[i % len(scenarios)]
            
            iteration_start = time.time()
            features = await self._extract_features_for_scenario(scenario)
            iteration_time = (time.time() - iteration_start) * 1000  # Convert to ms
            
            extraction_times.append(iteration_time)
            
            if i % 100 == 0:
                logger.info(f"Completed {i}/{iterations} iterations")
                
        total_time = time.time() - start_time
        cache_end_hits = getattr(self.feature_extractor, 'cache_hits', 0)
        cache_end_misses = getattr(self.feature_extractor, 'cache_misses', 0)
        
        # Calculate statistics
        return {
            'total_iterations': iterations,
            'total_time_seconds': total_time,
            'avg_extraction_time_ms': statistics.mean(extraction_times),
            'median_extraction_time_ms': statistics.median(extraction_times),
            'p95_extraction_time_ms': np.percentile(extraction_times, 95),
            'p99_extraction_time_ms': np.percentile(extraction_times, 99),
            'max_extraction_time_ms': max(extraction_times),
            'min_extraction_time_ms': min(extraction_times),
            'std_dev_ms': statistics.stdev(extraction_times),
            'cache_hits': cache_end_hits - cache_start_hits,
            'cache_misses': cache_end_misses - cache_start_misses,
            'cache_hit_rate': (cache_end_hits - cache_start_hits) / iterations if iterations > 0 else 0,
            'throughput_req_per_sec': iterations / total_time
        }
    
    async def benchmark_workflow_patterns(
        self, 
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Benchmark workflow pattern recognition accuracy and performance
        """
        logger.info("Starting workflow pattern benchmark")
        
        pattern_results = {
            'browse_edit_save': {'correct': 0, 'total': 0, 'times': []},
            'browse_confirm_save': {'correct': 0, 'total': 0, 'times': []},
            'cold_start_create': {'correct': 0, 'total': 0, 'times': []},
            'cold_start_browse': {'correct': 0, 'total': 0, 'times': []}
        }
        
        for scenario in scenarios:
            expected_pattern = None
            for prediction in scenario.get('expected_predictions', []):
                expected_pattern = prediction.get('workflow_pattern')
                break
                
            if not expected_pattern:
                continue
                
            start_time = time.time()
            features = await self._extract_features_for_scenario(scenario)
            pattern_time = (time.time() - start_time) * 1000
            
            # Check if the workflow distance feature indicates correct pattern
            workflow_distance = features.get('workflow_distance', 1.0)
            detected_pattern = self._classify_pattern_from_distance(workflow_distance, scenario)
            
            if expected_pattern in pattern_results:
                pattern_results[expected_pattern]['total'] += 1
                pattern_results[expected_pattern]['times'].append(pattern_time)
                
                if detected_pattern == expected_pattern:
                    pattern_results[expected_pattern]['correct'] += 1
        
        # Calculate accuracy for each pattern
        for pattern in pattern_results:
            stats = pattern_results[pattern]
            if stats['total'] > 0:
                stats['accuracy'] = stats['correct'] / stats['total']
                stats['avg_time_ms'] = statistics.mean(stats['times']) if stats['times'] else 0
            else:
                stats['accuracy'] = 0
                stats['avg_time_ms'] = 0
                
        return pattern_results
    
    async def benchmark_caching_effectiveness(
        self, 
        scenarios: List[Dict[str, Any]], 
        cache_test_iterations: int = 500
    ) -> Dict[str, Any]:
        """
        Test caching effectiveness by running scenarios multiple times
        """
        logger.info("Starting cache effectiveness benchmark")
        
        # First pass - populate cache
        for i, scenario in enumerate(scenarios[:100]):  # Use first 100 scenarios
            await self._extract_features_for_scenario(scenario)
            
        # Clear cache statistics
        if hasattr(self.feature_extractor, 'cache_hits'):
            self.feature_extractor.cache_hits = 0
        if hasattr(self.feature_extractor, 'cache_misses'):
            self.feature_extractor.cache_misses = 0
            
        # Second pass - test cache hit rates
        cache_test_times = []
        start_time = time.time()
        
        for i in range(cache_test_iterations):
            scenario = scenarios[i % min(100, len(scenarios))]  # Cycle through first 100
            
            iteration_start = time.time()
            await self._extract_features_for_scenario(scenario)
            iteration_time = (time.time() - iteration_start) * 1000
            
            cache_test_times.append(iteration_time)
            
        total_time = time.time() - start_time
        cache_hits = getattr(self.feature_extractor, 'cache_hits', 0)
        cache_misses = getattr(self.feature_extractor, 'cache_misses', 0)
        
        return {
            'cache_test_iterations': cache_test_iterations,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0,
            'avg_cached_time_ms': statistics.mean(cache_test_times),
            'total_cache_test_time': total_time
        }
    
    async def benchmark_concurrent_load(
        self, 
        scenarios: List[Dict[str, Any]], 
        concurrent_requests: int = 10,
        requests_per_worker: int = 50
    ) -> Dict[str, Any]:
        """
        Test performance under concurrent load
        """
        logger.info(f"Starting concurrent load test: {concurrent_requests} workers, {requests_per_worker} requests each")
        
        async def worker(worker_id: int, scenarios: List[Dict[str, Any]]) -> List[float]:
            """Individual worker for concurrent testing"""
            worker_times = []
            
            for i in range(requests_per_worker):
                scenario = scenarios[(worker_id * requests_per_worker + i) % len(scenarios)]
                
                start_time = time.time()
                await self._extract_features_for_scenario(scenario)
                worker_times.append((time.time() - start_time) * 1000)
                
            return worker_times
        
        # Run concurrent workers
        start_time = time.time()
        tasks = [
            worker(i, scenarios) for i in range(concurrent_requests)
        ]
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Aggregate results
        all_times = []
        for worker_times in results:
            all_times.extend(worker_times)
            
        total_requests = concurrent_requests * requests_per_worker
        
        return {
            'concurrent_workers': concurrent_requests,
            'requests_per_worker': requests_per_worker,
            'total_requests': total_requests,
            'total_time_seconds': total_time,
            'avg_response_time_ms': statistics.mean(all_times),
            'p95_response_time_ms': np.percentile(all_times, 95),
            'p99_response_time_ms': np.percentile(all_times, 99),
            'throughput_req_per_sec': total_requests / total_time,
            'requests_per_worker_per_sec': requests_per_worker / (total_time / concurrent_requests)
        }
    
    async def _extract_features_for_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for a single test scenario"""
        prompt = scenario.get('prompt', '')
        history = scenario.get('history', [])
        
        # Create candidate API from expected predictions
        expected_predictions = scenario.get('expected_predictions', [])
        if expected_predictions:
            first_prediction = expected_predictions[0]
            candidate_api = {
                'api_call': first_prediction.get('api_call', ''),
                'method': first_prediction.get('method', 'GET'),
                'description': first_prediction.get('description', '')
            }
        else:
            candidate_api = {
                'api_call': 'GET /api/default',
                'method': 'GET',
                'description': 'Default API call'
            }
        
        return await self.feature_extractor.extract_ml_features(
            prompt=prompt,
            history=history,
            candidate_api=candidate_api
        )
    
    def _classify_pattern_from_distance(self, workflow_distance: float, scenario: Dict[str, Any]) -> str:
        """Classify workflow pattern based on distance and scenario context"""
        # Simple heuristic - in real implementation, this would be more sophisticated
        history_length = len(scenario.get('history', []))
        
        if history_length == 0:
            return 'cold_start_create' if 'create' in scenario.get('prompt', '').lower() else 'cold_start_browse'
        elif workflow_distance < 0.2:
            return 'browse_confirm_save' if 'confirm' in scenario.get('prompt', '').lower() else 'browse_edit_save'
        else:
            return 'browse_edit_save'
    
    async def run_comprehensive_benchmark(
        self, 
        scenario_files: List[str]
    ) -> Dict[str, Any]:
        """
        Run all benchmarks and return comprehensive results
        """
        logger.info("Starting comprehensive feature engineering benchmark")
        
        # Load all test scenarios
        scenarios = await self.load_test_scenarios(scenario_files)
        
        if not scenarios:
            logger.error("No scenarios loaded for benchmarking")
            return {}
        
        logger.info(f"Loaded {len(scenarios)} total test scenarios")
        
        # Run all benchmarks
        results = {
            'benchmark_timestamp': datetime.utcnow().isoformat(),
            'total_scenarios': len(scenarios),
            'scenario_files': scenario_files
        }
        
        # Speed benchmark
        logger.info("Running speed benchmark...")
        speed_results = await self.benchmark_extraction_speed(scenarios, iterations=min(1000, len(scenarios) * 3))
        results['speed_benchmark'] = speed_results
        
        # Workflow pattern benchmark
        logger.info("Running workflow pattern benchmark...")
        pattern_results = await self.benchmark_workflow_patterns(scenarios)
        results['workflow_pattern_benchmark'] = pattern_results
        
        # Cache effectiveness benchmark
        logger.info("Running cache effectiveness benchmark...")
        cache_results = await self.benchmark_caching_effectiveness(scenarios, cache_test_iterations=500)
        results['cache_effectiveness_benchmark'] = cache_results
        
        # Concurrent load benchmark
        logger.info("Running concurrent load benchmark...")
        concurrent_results = await self.benchmark_concurrent_load(scenarios, concurrent_requests=5, requests_per_worker=20)
        results['concurrent_load_benchmark'] = concurrent_results
        
        logger.info("Comprehensive benchmark completed")
        return results

# Example usage and test runner
async def main():
    """
    Example benchmark runner
    """
    # This would be replaced with actual feature extractor import
    # from app.utils.feature_eng import FeatureExtractor
    # feature_extractor = FeatureExtractor()
    
    # Mock feature extractor for demonstration
    class MockFeatureExtractor:
        def __init__(self):
            self.cache_hits = 0
            self.cache_misses = 0
            
        async def extract_ml_features(self, prompt, history=None, candidate_api=None):
            # Simulate feature extraction time
            await asyncio.sleep(0.02)  # 20ms simulation
            
            # Simulate cache behavior
            if hash(prompt) % 3 == 0:  # 33% cache hit rate
                self.cache_hits += 1
            else:
                self.cache_misses += 1
                
            return {
                'workflow_distance': 0.1 if history else 0.5,
                'prompt_similarity': 0.8,
                'resource_match': 1 if history else 0,
                'extraction_timestamp': datetime.utcnow().isoformat()
            }
    
    # Initialize benchmark
    mock_extractor = MockFeatureExtractor()
    benchmark = FeatureEngineeringBenchmark(mock_extractor)
    
    # Run comprehensive benchmark
    scenario_files = [
        '/home/quantum/ApiCallPredictor/examples/stripe_billing.json',
        '/home/quantum/ApiCallPredictor/examples/github_pr_workflow.json'
    ]
    
    results = await benchmark.run_comprehensive_benchmark(scenario_files)
    
    # Print results
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())