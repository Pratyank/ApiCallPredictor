#!/usr/bin/env python3
"""
Quick Performance Test Runner for Phase 5 Validation
Simplified test runner to verify the performance testing framework is working
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

async def quick_performance_validation():
    """Run a quick performance validation to ensure the testing framework works"""
    
    print("🚀 Quick Performance Validation for Phase 5")
    print("=" * 50)
    
    # Test 1: Basic import validation
    try:
        from tests.perf_test import PerformanceTester
        print("✅ Performance testing framework imports successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Initialize performance tester
    try:
        tester = PerformanceTester()
        print("✅ Performance tester initialized")
        print(f"   - Test iterations: {tester.test_iterations}")
        print(f"   - Concurrent users: {tester.concurrent_users}")
        print(f"   - LLM latency target: {tester.targets['llm_latency_ms']}ms")
        print(f"   - ML scoring target: {tester.targets['ml_scoring_latency_ms']}ms")
        print(f"   - Total response target: {tester.targets['total_response_latency_ms']}ms")
    except Exception as e:
        print(f"❌ Tester initialization failed: {e}")
        return False
    
    # Test 3: Generate test data
    try:
        test_data = tester.setup_test_data()
        print("✅ Test data generation successful")
        print(f"   - Test prompts: {len(test_data['prompts'])}")
        print(f"   - Test histories: {len(test_data['histories'])}")
        print(f"   - Test configurations: {len(test_data['test_configurations'])}")
    except Exception as e:
        print(f"❌ Test data generation failed: {e}")
        return False
    
    # Test 4: Quick component latency measurement
    try:
        def sample_function():
            """Sample function for latency testing"""
            time.sleep(0.01)  # 10ms simulated work
            return "test_result"
        
        latency, result, error = tester.measure_component_latency("test_component", sample_function)
        print("✅ Component latency measurement working")
        print(f"   - Measured latency: {latency:.2f}ms")
        print(f"   - Result: {result}")
        print(f"   - Error: {error}")
    except Exception as e:
        print(f"❌ Component latency measurement failed: {e}")
        return False
    
    # Test 5: Metrics calculation
    try:
        sample_latencies = [100, 120, 110, 95, 105, 130, 115, 98, 125, 108]
        metrics = tester._calculate_metrics(sample_latencies, 0, "Sample Test")
        print("✅ Metrics calculation working")
        print(f"   - Average latency: {metrics.avg_latency:.2f}ms")
        print(f"   - Median latency: {metrics.median_latency:.2f}ms")
        print(f"   - P95 latency: {metrics.p95_latency:.2f}ms")
    except Exception as e:
        print(f"❌ Metrics calculation failed: {e}")
        return False
    
    # Test 6: Check if application components are available
    try:
        from app.models.predictor import get_predictor
        predictor = await get_predictor()
        print("✅ Application predictor accessible")
        print(f"   - Predictor initialized with k={predictor.k}, buffer={predictor.buffer}")
    except Exception as e:
        print(f"⚠️  Application predictor not accessible (expected in test environment): {e}")
    
    # Test 7: Report generation preview
    try:
        sample_results = {
            "llm_latency": tester._calculate_metrics([450, 480, 420, 510, 475], 0, "LLM Test"),
            "ml_scoring": tester._calculate_metrics([85, 92, 78, 95, 88], 0, "ML Test"),
            "total_response": tester._calculate_metrics([720, 760, 680, 800, 740], 0, "Total Test")
        }
        
        # Add cache simulation
        sample_results["caching"] = {
            "cache_hit_rate": 0.85,
            "avg_cached_latency": 25.5,
            "first_request_latencies": [720],
            "cache_effectiveness": True
        }
        
        report = tester.generate_performance_report(sample_results)
        print("✅ Performance report generation working")
        print(f"   - Report length: {len(report)} characters")
        print(f"   - Contains targets: {'Target Met' in report}")
        
        # Save sample report
        sample_report_file = tester.results_dir / "sample_performance_report.md"
        with open(sample_report_file, 'w') as f:
            f.write(report)
        print(f"   - Sample report saved to: {sample_report_file}")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Quick Performance Validation PASSED")
    print("🎯 Performance testing framework is ready for Phase 5 validation")
    print("\nNext steps:")
    print("1. Run full performance suite: python tests/perf_test.py")
    print("2. Run pytest tests: pytest tests/perf_test.py -m performance -v")
    print("3. Monitor results in tests/performance_results/")
    
    return True

async def main():
    """Main entry point"""
    success = await quick_performance_validation()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())