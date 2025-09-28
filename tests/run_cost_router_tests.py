#!/usr/bin/env python3
"""
Test runner for Cost-Aware Router test suite.

Runs comprehensive tests including:
- Unit tests for routing logic
- Performance validation tests
- Integration tests with predictor pipeline
- Edge case and robustness tests
- Budget tracking and database tests
"""

import sys
import os
import subprocess
import time
import sqlite3
from pathlib import Path


def setup_test_environment():
    """Set up the test environment."""
    print("🔧 Setting up test environment...")
    
    # Ensure test directories exist
    test_dirs = [
        "tests/fixtures",
        "tests/performance", 
        "tests/integration",
        "tests/edge_cases",
        "data"
    ]
    
    for test_dir in test_dirs:
        Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    # Create test database if needed
    test_db_path = "data/test_cache.db"
    if not os.path.exists(test_db_path):
        conn = sqlite3.connect(test_db_path)
        conn.close()
    
    print("✅ Test environment ready")


def run_test_suite(test_category="all", verbose=True, coverage=True):
    """Run the specified test suite."""
    
    test_files = {
        "unit": ["tests/cost_aware_router_test.py"],
        "performance": ["tests/performance/test_cost_router_performance.py"],
        "integration": ["tests/integration/test_cost_router_integration.py"],
        "edge_cases": ["tests/edge_cases/test_cost_router_edge_cases.py"],
        "all": [
            "tests/cost_aware_router_test.py",
            "tests/performance/test_cost_router_performance.py", 
            "tests/integration/test_cost_router_integration.py",
            "tests/edge_cases/test_cost_router_edge_cases.py"
        ]
    }
    
    if test_category not in test_files:
        print(f"❌ Unknown test category: {test_category}")
        print(f"Available categories: {list(test_files.keys())}")
        return False
    
    files_to_test = test_files[test_category]
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.extend(["-v", "--tb=short"])
    
    if coverage:
        cmd.extend([
            "--cov=cost_aware_router",
            "--cov-report=term-missing",
            "--cov-report=html:tests/coverage_html"
        ])
    
    # Add performance markers for performance tests
    if test_category in ["performance", "all"]:
        cmd.extend(["-m", "not slow"])  # Skip slow tests by default
    
    cmd.extend(files_to_test)
    
    print(f"🧪 Running {test_category} tests...")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_category} tests passed in {duration:.2f}s")
            return True
        else:
            print(f"❌ {test_category} tests failed after {duration:.2f}s")
            return False
            
    except FileNotFoundError:
        print("❌ pytest not found. Please install with: pip install pytest pytest-asyncio pytest-cov")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def run_specific_performance_tests():
    """Run specific performance validation tests."""
    print("🚀 Running performance validation tests...")
    
    performance_cmd = [
        "python", "-m", "pytest",
        "tests/performance/test_cost_router_performance.py",
        "-v", "--tb=short",
        "-k", "test_performance_requirements",  # Focus on core performance tests
        "--durations=10"  # Show slowest 10 tests
    ]
    
    try:
        result = subprocess.run(performance_cmd)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
        return False


def validate_requirements():
    """Validate that performance and functional requirements are met."""
    print("📋 Validating requirements...")
    
    requirements = {
        "LLM latency < 500ms": False,
        "Total latency < 800ms": False,
        "Budget tracking accuracy": False,
        "Model selection logic": False,
        "Edge case handling": False,
        "Integration compatibility": False
    }
    
    # Run minimal validation tests
    validation_cmd = [
        "python", "-m", "pytest",
        "tests/cost_aware_router_test.py::TestCostAwareRouter::test_performance_requirements_llm_latency",
        "tests/cost_aware_router_test.py::TestCostAwareRouter::test_performance_requirements_total_latency",
        "tests/cost_aware_router_test.py::TestCostAwareRouter::test_budget_tracking_accuracy",
        "tests/cost_aware_router_test.py::TestCostAwareRouter::test_simple_prompt_uses_haiku",
        "tests/edge_cases/test_cost_router_edge_cases.py::TestCostRouterEdgeCases::test_empty_and_null_inputs",
        "tests/integration/test_cost_router_integration.py::TestCostRouterIntegration::test_predictor_pipeline_integration",
        "-v", "--tb=line"
    ]
    
    try:
        result = subprocess.run(validation_cmd, capture_output=True, text=True)
        
        # Parse results to update requirements
        if "test_performance_requirements_llm_latency PASSED" in result.stdout:
            requirements["LLM latency < 500ms"] = True
        if "test_performance_requirements_total_latency PASSED" in result.stdout:
            requirements["Total latency < 800ms"] = True
        if "test_budget_tracking_accuracy PASSED" in result.stdout:
            requirements["Budget tracking accuracy"] = True
        if "test_simple_prompt_uses_haiku PASSED" in result.stdout:
            requirements["Model selection logic"] = True
        if "test_empty_and_null_inputs PASSED" in result.stdout:
            requirements["Edge case handling"] = True
        if "test_predictor_pipeline_integration PASSED" in result.stdout:
            requirements["Integration compatibility"] = True
        
        print("\n📊 Requirements Validation Results:")
        for req, passed in requirements.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} {req}")
        
        return all(requirements.values())
        
    except Exception as e:
        print(f"❌ Requirements validation failed: {e}")
        return False


def generate_test_report():
    """Generate a comprehensive test report."""
    print("📄 Generating test report...")
    
    report_cmd = [
        "python", "-m", "pytest",
        "tests/",
        "--tb=no",
        "--quiet",
        "--json-report",
        "--json-report-file=tests/test_report.json"
    ]
    
    try:
        subprocess.run(report_cmd, capture_output=True)
        print("✅ Test report generated at tests/test_report.json")
        
        # Generate summary
        summary_cmd = [
            "python", "-m", "pytest",
            "tests/",
            "--collect-only",
            "--quiet"
        ]
        
        result = subprocess.run(summary_cmd, capture_output=True, text=True)
        test_count = result.stdout.count("::test_")
        
        print(f"📊 Test Summary:")
        print(f"  Total test cases: {test_count}")
        print(f"  Test files: 4 (unit, performance, integration, edge cases)")
        print(f"  Coverage report: tests/coverage_html/index.html")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")


def main():
    """Main test runner function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cost-Aware Router Test Runner")
    parser.add_argument(
        "category",
        nargs="?",
        default="all",
        choices=["unit", "performance", "integration", "edge_cases", "all"],
        help="Test category to run"
    )
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--validate-only", action="store_true", help="Only run requirement validation")
    parser.add_argument("--report", action="store_true", help="Generate test report")
    
    args = parser.parse_args()
    
    print("🧪 Cost-Aware Router Test Suite")
    print("=" * 50)
    
    # Setup environment
    setup_test_environment()
    
    if args.validate_only:
        success = validate_requirements()
        sys.exit(0 if success else 1)
    
    if args.report:
        generate_test_report()
        return
    
    # Run tests
    success = run_test_suite(
        test_category=args.category,
        verbose=not args.quiet,
        coverage=not args.no_coverage
    )
    
    if success and args.category == "all":
        # Run requirements validation
        print("\n" + "=" * 50)
        validate_requirements()
        
        # Generate report
        generate_test_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()