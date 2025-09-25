#!/usr/bin/env python3
"""
Test runner script for opensesame-predictor.

Provides programmatic test execution with reporting and CI integration.
Can be used as an alternative to pytest command line for complex test scenarios.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import signal
from dataclasses import dataclass


@dataclass
class TestResult:
    """Test execution result."""
    suite_name: str
    passed: int
    failed: int
    skipped: int
    duration: float
    coverage: Optional[float] = None
    exit_code: int = 0
    output: str = ""


class TestRunner:
    """Orchestrates test execution with reporting."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def run_command(self, cmd: List[str], timeout: int = 300) -> TestResult:
        """Run a test command and capture results."""
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            # Set up timeout
            def timeout_handler(signum, frame):
                process.kill()
                raise TimeoutError(f"Command timed out after {timeout} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            output, _ = process.communicate()
            signal.alarm(0)  # Cancel timeout
            
            duration = time.time() - start_time
            
            # Parse pytest output for statistics
            passed, failed, skipped = self._parse_pytest_output(output)
            coverage = self._extract_coverage(output)
            
            result = TestResult(
                suite_name=' '.join(cmd[1:3]) if len(cmd) > 2 else cmd[1],
                passed=passed,
                failed=failed,
                skipped=skipped,
                duration=duration,
                coverage=coverage,
                exit_code=process.returncode,
                output=output
            )
            
            self.results.append(result)
            return result
            
        except TimeoutError as e:
            duration = time.time() - start_time
            result = TestResult(
                suite_name=' '.join(cmd[1:3]) if len(cmd) > 2 else cmd[1],
                passed=0,
                failed=1,
                skipped=0,
                duration=duration,
                exit_code=124,  # Timeout exit code
                output=str(e)
            )
            self.results.append(result)
            return result
        
        except Exception as e:
            duration = time.time() - start_time
            result = TestResult(
                suite_name=' '.join(cmd[1:3]) if len(cmd) > 2 else cmd[1],
                passed=0,
                failed=1,
                skipped=0,
                duration=duration,
                exit_code=1,
                output=str(e)
            )
            self.results.append(result)
            return result
    
    def _parse_pytest_output(self, output: str) -> tuple:
        """Parse pytest output to extract test statistics."""
        lines = output.split('\n')
        
        # Look for summary line like "= 25 passed, 3 failed, 2 skipped in 15.23s ="
        summary_patterns = [
            "passed", "failed", "skipped", "error"
        ]
        
        passed = failed = skipped = 0
        
        for line in reversed(lines):  # Start from end to find final summary
            if any(pattern in line for pattern in summary_patterns):
                # Extract numbers using simple parsing
                words = line.split()
                for i, word in enumerate(words):
                    if word.endswith("passed") and i > 0:
                        try:
                            passed = int(words[i-1])
                        except (ValueError, IndexError):
                            pass
                    elif word.endswith("failed") and i > 0:
                        try:
                            failed = int(words[i-1])
                        except (ValueError, IndexError):
                            pass
                    elif word.endswith("skipped") and i > 0:
                        try:
                            skipped = int(words[i-1])
                        except (ValueError, IndexError):
                            pass
                break
        
        return passed, failed, skipped
    
    def _extract_coverage(self, output: str) -> Optional[float]:
        """Extract coverage percentage from output."""
        lines = output.split('\n')
        
        for line in lines:
            if "TOTAL" in line and "%" in line:
                # Look for percentage in TOTAL line
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            return float(part.rstrip('%'))
                        except ValueError:
                            pass
        
        return None
    
    def run_unit_tests(self) -> TestResult:
        """Run unit tests."""
        cmd = ["python", "-m", "pytest", "tests/unit/", "-v", "--tb=short"]
        return self.run_command(cmd)
    
    def run_integration_tests(self) -> TestResult:
        """Run integration tests."""
        cmd = ["python", "-m", "pytest", "tests/integration/", "-v", "--asyncio-mode=auto"]
        return self.run_command(cmd)
    
    def run_security_tests(self) -> TestResult:
        """Run security tests."""
        cmd = ["python", "-m", "pytest", "tests/security/", "-v", "--tb=short"]
        return self.run_command(cmd)
    
    def run_performance_tests(self) -> TestResult:
        """Run performance tests."""
        cmd = ["python", "-m", "pytest", "tests/performance/", "-v", "-m", "performance"]
        return self.run_command(cmd, timeout=600)  # Longer timeout for performance tests
    
    def run_docker_tests(self) -> TestResult:
        """Run docker tests."""
        cmd = ["python", "-m", "pytest", "tests/docker/", "-v"]
        return self.run_command(cmd, timeout=600)  # Longer timeout for docker tests
    
    def run_coverage_tests(self) -> TestResult:
        """Run tests with coverage."""
        cmd = [
            "python", "-m", "pytest", 
            "tests/unit/", "tests/integration/",
            "--cov=app", "--cov-report=term-missing", "--cov-report=xml"
        ]
        return self.run_command(cmd)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = time.time() - self.start_time
        
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_tests = total_passed + total_failed + total_skipped
        
        # Calculate overall success rate
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Find average coverage
        coverage_results = [r.coverage for r in self.results if r.coverage is not None]
        avg_coverage = sum(coverage_results) / len(coverage_results) if coverage_results else None
        
        report = {
            "timestamp": time.time(),
            "total_duration": total_duration,
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "success_rate": round(success_rate, 2),
                "average_coverage": round(avg_coverage, 2) if avg_coverage else None
            },
            "test_suites": [
                {
                    "name": result.suite_name,
                    "passed": result.passed,
                    "failed": result.failed,
                    "skipped": result.skipped,
                    "duration": round(result.duration, 2),
                    "coverage": result.coverage,
                    "exit_code": result.exit_code,
                    "status": "PASS" if result.exit_code == 0 else "FAIL"
                }
                for result in self.results
            ],
            "quality_gates": self._evaluate_quality_gates()
        }
        
        return report
    
    def _evaluate_quality_gates(self) -> Dict[str, Any]:
        """Evaluate quality gates against results."""
        gates = {
            "coverage_threshold": {"threshold": 85, "status": "UNKNOWN", "actual": None},
            "test_pass_rate": {"threshold": 95, "status": "UNKNOWN", "actual": None},
            "performance_sla": {"threshold": 800, "status": "UNKNOWN", "actual": None},
            "security_issues": {"threshold": 0, "status": "UNKNOWN", "actual": None}
        }
        
        # Coverage gate
        coverage_results = [r.coverage for r in self.results if r.coverage is not None]
        if coverage_results:
            avg_coverage = sum(coverage_results) / len(coverage_results)
            gates["coverage_threshold"]["actual"] = round(avg_coverage, 2)
            gates["coverage_threshold"]["status"] = "PASS" if avg_coverage >= 85 else "FAIL"
        
        # Pass rate gate
        total_passed = sum(r.passed for r in self.results)
        total_tests = sum(r.passed + r.failed for r in self.results)
        if total_tests > 0:
            pass_rate = (total_passed / total_tests) * 100
            gates["test_pass_rate"]["actual"] = round(pass_rate, 2)
            gates["test_pass_rate"]["status"] = "PASS" if pass_rate >= 95 else "FAIL"
        
        # Security gate (no failures in security tests)
        security_results = [r for r in self.results if "security" in r.suite_name.lower()]
        if security_results:
            total_security_failures = sum(r.failed for r in security_results)
            gates["security_issues"]["actual"] = total_security_failures
            gates["security_issues"]["status"] = "PASS" if total_security_failures == 0 else "FAIL"
        
        return gates
    
    def print_summary(self):
        """Print test execution summary."""
        print("\n" + "="*60)
        print("TEST EXECUTION SUMMARY")
        print("="*60)
        
        for result in self.results:
            status = "✅ PASS" if result.exit_code == 0 else "❌ FAIL"
            print(f"{status} {result.suite_name:<30} "
                  f"({result.passed}P/{result.failed}F/{result.skipped}S) "
                  f"{result.duration:.1f}s")
            
            if result.coverage:
                print(f"     Coverage: {result.coverage:.1f}%")
        
        # Overall summary
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_duration = time.time() - self.start_time
        
        print("\n" + "-"*60)
        print(f"TOTAL: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
        print(f"Duration: {total_duration:.1f}s")
        
        if total_failed > 0:
            print("\n❌ SOME TESTS FAILED")
            return False
        else:
            print("\n✅ ALL TESTS PASSED")
            return True


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="OpenSesame Predictor Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--docker", action="store_true", help="Run docker tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--report", help="Output JSON report to file")
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    
    print("Starting OpenSesame Predictor Test Suite...")
    print(f"Working directory: {Path.cwd()}")
    
    # Determine which tests to run
    run_all = args.all or not any([
        args.unit, args.integration, args.performance, 
        args.security, args.docker, args.coverage
    ])
    
    try:
        # Run selected test suites
        if args.unit or run_all:
            print("\n📋 Running unit tests...")
            runner.run_unit_tests()
        
        if args.integration or run_all:
            print("\n🔗 Running integration tests...")
            runner.run_integration_tests()
        
        if args.security or run_all:
            print("\n🔒 Running security tests...")
            runner.run_security_tests()
        
        if args.coverage and not run_all:
            print("\n📊 Running coverage tests...")
            runner.run_coverage_tests()
        
        if args.performance:
            print("\n⚡ Running performance tests...")
            runner.run_performance_tests()
        
        if args.docker:
            print("\n🐳 Running docker tests...")
            runner.run_docker_tests()
        
        # Generate report
        report = runner.generate_report()
        
        # Save JSON report if requested
        if args.report:
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Report saved to {args.report}")
        
        # Print summary
        success = runner.print_summary()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()