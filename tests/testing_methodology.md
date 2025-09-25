# OpenSesame Predictor Testing Methodology

## Overview
This document outlines the comprehensive testing strategy for the opensesame-predictor API prediction system, covering all aspects from unit tests to security validation.

## Testing Architecture

### Test Organization
```
tests/
├── conftest.py              # Shared fixtures and configuration
├── __init__.py
├── unit/                    # Unit tests (isolated components)
│   ├── models/              # Core prediction logic tests
│   │   ├── test_predictor.py
│   │   ├── test_ai_layer.py
│   │   └── test_ml_ranker.py
│   └── utils/               # Utility function tests
│       ├── test_spec_parser.py
│       ├── test_guardrails.py
│       └── test_feature_eng.py
├── integration/             # Integration tests (component interactions)
│   └── test_api_endpoints.py
├── performance/             # Performance and load tests
│   └── test_prediction_pipeline.py
├── security/                # Security and penetration tests
│   └── test_guardrails_security.py
├── docker/                  # Container and deployment tests
│   └── test_container_deployment.py
├── fixtures/                # Test data and mock objects
│   └── sample_data.py
└── mocks/                   # Mock implementations
```

## Testing Levels

### 1. Unit Tests (85% Coverage Target)
**Purpose**: Test individual components in isolation

**Components Tested**:
- `app/models/predictor.py` - Core prediction orchestration
- `app/models/ai_layer.py` - LLM integration and candidate generation
- `app/models/ml_ranker.py` - ML ranking and feature engineering
- `app/utils/spec_parser.py` - OpenAPI specification parsing
- `app/utils/guardrails.py` - Safety and security validation
- `app/utils/feature_eng.py` - Feature extraction and engineering

**Key Test Scenarios**:
- Valid input processing
- Error handling and edge cases
- Performance within component-level SLAs
- Mock external dependencies (OpenAI, Anthropic, file system)
- Parameter validation and sanitization

**Execution**:
```bash
# Run unit tests only
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=app --cov-report=html --cov-report=term
```

### 2. Integration Tests (End-to-End Workflows)
**Purpose**: Test component interactions and API contracts

**Areas Covered**:
- FastAPI endpoint functionality (`/predict`, `/health`, `/metrics`)
- Request/response validation
- Error handling across the stack
- Authentication and authorization flows
- Rate limiting implementation
- CORS and security headers
- OpenAPI specification compliance

**Key Test Scenarios**:
- Complete prediction workflows with various input combinations
- Guardrails integration with prediction pipeline
- AI provider failover scenarios
- Concurrent request handling
- Large payload processing
- Malformed input handling

**Execution**:
```bash
# Run integration tests
pytest tests/integration/ -v --asyncio-mode=auto

# Run against local container
pytest tests/integration/ --base-url=http://localhost:8000
```

### 3. Performance Tests (SLA Validation)
**Purpose**: Validate performance requirements and identify bottlenecks

**Performance Requirements**:
- Single prediction: < 800ms median response time
- AI layer: < 500ms response time
- ML ranker: < 100ms processing time
- Throughput: 10+ RPS sustained
- Memory usage: < 512MB under normal load
- Concurrent requests: 20+ simultaneous users

**Test Types**:
- **Load Testing**: Sustained traffic at target RPS
- **Stress Testing**: Beyond normal capacity limits
- **Spike Testing**: Sudden traffic increases
- **Endurance Testing**: Long-duration sustained load
- **Volume Testing**: Large OpenAPI specs and event histories

**Tools and Frameworks**:
- Python `asyncio` for concurrent testing
- `psutil` for resource monitoring
- Custom performance monitoring utilities
- Memory leak detection

**Execution**:
```bash
# Run performance tests (marked as slow)
pytest tests/performance/ -v -m performance

# Run specific performance scenarios
pytest tests/performance/ -k "test_concurrent" -v

# Include slow tests
pytest tests/performance/ --runslow
```

### 4. Security Tests (Penetration Testing)
**Purpose**: Validate security measures and identify vulnerabilities

**Security Areas**:
- **Input Validation**: SQL injection, XSS, command injection prevention
- **Authentication**: Bypass attempts and session security
- **Authorization**: Access control validation
- **Rate Limiting**: Abuse prevention and bypass attempts
- **Data Sanitization**: Malicious payload handling
- **Container Security**: Non-root execution, minimal attack surface

**Attack Vectors Tested**:
- SQL injection in all input fields
- Cross-site scripting (XSS) attempts
- Command injection via prompts and events
- Path traversal attacks
- Unicode normalization attacks
- Null byte injection
- Oversized payload attacks
- Rate limiting bypass techniques

**Execution**:
```bash
# Run security tests
pytest tests/security/ -v

# Run specific attack vector tests
pytest tests/security/ -k "sql_injection" -v
```

### 5. Docker Container Tests
**Purpose**: Validate containerized deployment and production readiness

**Container Validation**:
- **Build Process**: Multi-stage optimization, layer efficiency
- **Runtime Security**: Non-root user, file permissions, minimal packages
- **Resource Management**: Memory limits, CPU constraints
- **Networking**: Port mapping, DNS resolution, isolation
- **Health Checks**: Startup verification, graceful shutdown
- **Performance**: Startup time, resource efficiency

**Execution**:
```bash
# Build and test container
docker build -t opensesame-predictor:test .
pytest tests/docker/ -v

# Test with resource constraints
docker run --cpus=2 --memory=512m opensesame-predictor:test
```

## Test Data Strategy

### Fixtures and Sample Data
**Real-world API Specifications**:
- Stripe API (payment processing workflows)
- GitHub API (development workflows) 
- Generic REST API patterns
- Large-scale APIs (1000+ endpoints)

**User Event Sequences**:
- E-commerce shopping flows
- User management workflows
- Payment processing sequences
- GitHub PR workflows
- Administrative operations

**Edge Cases**:
- Empty prompts (cold start scenarios)
- Malformed OpenAPI specs
- Extremely large payloads
- Unicode and special characters
- Boundary value testing

### Mock Strategy
**External Dependencies**:
- OpenAI/Anthropic API calls → Mock responses
- File system operations → Temporary directories
- Database connections → In-memory stores
- Network requests → Controlled responses

**Performance Isolation**:
- Deterministic response times for components
- Controlled latency injection
- Resource usage simulation

## Test Execution Strategy

### Development Workflow
```bash
# Quick feedback loop (unit tests only)
pytest tests/unit/ --tb=short -q

# Pre-commit validation
pytest tests/unit/ tests/integration/ --cov=app

# Full test suite (CI/CD)
pytest tests/ --cov=app --cov-report=xml --junitxml=test-results.xml
```

### Continuous Integration
```yaml
# GitHub Actions / CI Pipeline
test_matrix:
  python_version: [3.9, 3.10, 3.11]
  test_type: [unit, integration, security]
  
stages:
  1. Linting and static analysis
  2. Unit tests with coverage
  3. Integration tests
  4. Security scanning
  5. Performance validation
  6. Container testing
```

### Performance Monitoring
```bash
# Benchmark tracking
pytest tests/performance/ --benchmark-only --benchmark-json=benchmarks.json

# Memory profiling
pytest tests/performance/ --memprof

# Load testing with reporting
pytest tests/performance/ --load-test --html=performance-report.html
```

## Quality Gates

### Coverage Requirements
- **Unit Tests**: 85% minimum line coverage
- **Critical Paths**: 95% coverage (prediction pipeline, guardrails)
- **Integration**: All API endpoints covered
- **Security**: All input validation paths tested

### Performance Thresholds
- **Response Time**: 95th percentile < 800ms
- **Throughput**: Sustained 10 RPS minimum
- **Memory**: No memory leaks over 1000 requests
- **Startup**: Container ready in < 30 seconds

### Security Validation
- **Zero High/Critical**: No high or critical security vulnerabilities
- **Input Validation**: 100% of attack vectors blocked
- **Rate Limiting**: Effective against abuse patterns
- **Container Security**: Passes security benchmarks

## Test Environment Management

### Local Development
```bash
# Setup test environment
python -m venv test-env
source test-env/bin/activate
pip install -r requirements-test.txt

# Run test database/services
docker-compose -f docker-compose.test.yml up -d
```

### CI/CD Environment
- Isolated test containers
- Ephemeral test databases  
- Controlled network conditions
- Resource-constrained testing

### Load Testing Environment
- Production-like hardware specs
- Multiple geographic regions
- Real network latency simulation
- Scaled infrastructure testing

## Reporting and Metrics

### Test Reports
- **Coverage Reports**: HTML and XML formats
- **Performance Reports**: Response time distributions, throughput graphs
- **Security Reports**: Vulnerability scan results, penetration test findings
- **Trend Analysis**: Historical performance and quality metrics

### Key Metrics
- Test execution time trends
- Flaky test identification
- Code coverage evolution
- Performance regression detection
- Security posture tracking

## Maintenance and Updates

### Test Data Refresh
- Monthly update of sample API specifications
- Quarterly review of attack vector coverage
- Annual security testing methodology review

### Performance Baseline Updates
- Recalibrate performance thresholds quarterly
- Update load testing scenarios based on usage patterns
- Refresh container optimization benchmarks

### Security Testing Evolution
- Monthly security vulnerability database updates
- Quarterly penetration testing methodology review
- Annual third-party security audit integration

This comprehensive testing methodology ensures the opensesame-predictor system is robust, secure, performant, and reliable across all deployment scenarios.