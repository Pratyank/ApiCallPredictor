# Makefile for opensesame-predictor testing
# Provides convenient test execution commands

.PHONY: help test test-unit test-integration test-performance test-security test-docker test-all coverage clean install

# Default target
help:
	@echo "OpenSesame Predictor Testing Commands"
	@echo "====================================="
	@echo ""
	@echo "Installation:"
	@echo "  install          Install test dependencies"
	@echo "  install-dev      Install dev and test dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run unit and integration tests (fast)"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests (slow)"
	@echo "  test-security   Run security tests"
	@echo "  test-docker     Run docker container tests"
	@echo "  test-all        Run all tests including slow ones"
	@echo ""
	@echo "Coverage:"
	@echo "  coverage        Run tests with coverage report"
	@echo "  coverage-html   Generate HTML coverage report"
	@echo "  coverage-xml    Generate XML coverage report"
	@echo ""
	@echo "Utilities:"
	@echo "  lint            Run code linting"
	@echo "  format          Format code with black and isort"
	@echo "  clean           Clean test artifacts and cache files"
	@echo "  docker-build    Build test docker image"
	@echo ""

# Installation targets
install:
	pip install -r requirements-test.txt

install-dev:
	pip install -r requirements-test.txt
	pip install -e .

# Main testing targets
test:
	pytest tests/unit/ tests/integration/ -v

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --asyncio-mode=auto

test-performance:
	pytest tests/performance/ -v -m performance --durations=0

test-security:
	pytest tests/security/ -v --tb=short

test-docker:
	pytest tests/docker/ -v

test-all:
	pytest tests/ -v --runslow

# Coverage targets
coverage:
	pytest tests/unit/ tests/integration/ --cov=app --cov-report=term-missing

coverage-html:
	pytest tests/unit/ tests/integration/ --cov=app --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

coverage-xml:
	pytest tests/unit/ tests/integration/ --cov=app --cov-report=xml
	@echo "Coverage report generated in coverage.xml"

# Performance benchmarking
benchmark:
	pytest tests/performance/ --benchmark-only --benchmark-json=benchmark-results.json
	@echo "Benchmark results saved to benchmark-results.json"

# Load testing
load-test:
	pytest tests/performance/test_prediction_pipeline.py::TestLoadTesting -v
	@echo "Load test completed"

# Security scanning
security-scan:
	bandit -r app/ -f json -o security-report.json
	safety check --json --output security-deps.json
	pytest tests/security/ --html=security-test-report.html
	@echo "Security scan completed - check security-*.json and security-test-report.html"

# Code quality
lint:
	flake8 app/ tests/
	mypy app/
	black --check app/ tests/
	isort --check-only app/ tests/

format:
	black app/ tests/
	isort app/ tests/
	@echo "Code formatting completed"

# Docker targets
docker-build:
	docker build -t opensesame-predictor:test .
	@echo "Test Docker image built: opensesame-predictor:test"

docker-test: docker-build
	pytest tests/docker/ -v
	@echo "Docker tests completed"

# Utility targets
clean:
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf test-results.xml
	rm -rf .mypy_cache/
	rm -rf benchmark-results.json
	rm -rf security-*.json
	rm -rf security-test-report.html
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned test artifacts and cache files"

# Development workflow targets
dev-test: format lint test-unit
	@echo "Development testing workflow completed"

ci-test: test coverage-xml
	@echo "CI testing workflow completed"

pre-commit: format lint test coverage
	@echo "Pre-commit checks completed"

# Parallel testing
test-parallel:
	pytest tests/unit/ tests/integration/ -n auto -v
	@echo "Parallel tests completed"

# Test specific components
test-predictor:
	pytest tests/unit/models/test_predictor.py -v

test-ai-layer:
	pytest tests/unit/models/test_ai_layer.py -v

test-ml-ranker:
	pytest tests/unit/models/test_ml_ranker.py -v

test-guardrails:
	pytest tests/unit/utils/test_guardrails.py tests/security/ -v

test-api:
	pytest tests/integration/test_api_endpoints.py -v

# Performance monitoring
perf-monitor:
	pytest tests/performance/ --benchmark-only --benchmark-autosave
	@echo "Performance monitoring completed - results saved"

# Memory profiling
memory-profile:
	pytest tests/performance/test_prediction_pipeline.py::TestPredictionPerformance::test_memory_usage_under_load -v --memprof
	@echo "Memory profiling completed"

# Stress testing
stress-test:
	pytest tests/performance/test_prediction_pipeline.py::TestLoadTesting::test_sustained_load -v
	pytest tests/performance/test_prediction_pipeline.py::TestLoadTesting::test_spike_load -v
	@echo "Stress testing completed"