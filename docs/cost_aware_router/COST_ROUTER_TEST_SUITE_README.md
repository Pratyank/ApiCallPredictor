# Cost-Aware Router Test Suite Documentation

## Overview
This directory contains all test files and validation scripts specifically for the **Cost-Aware Model Router** feature implementation in the OpenSesame Predictor project.

## Feature Description
The Cost-Aware Router is a bonus implementation that provides intelligent model selection between Anthropic Claude models based on query complexity and budget constraints:

- **Claude 3 Haiku**: $0.00025/1K tokens, 70% accuracy (cheap option)
- **Claude 3 Opus**: $0.015/1K tokens, 90% accuracy (premium option)
- **Intelligent Routing**: Complexity-based model selection with budget enforcement
- **Budget Tracking**: Real-time cost monitoring with SQLite persistence

## Test Files Overview

### 🧪 Core Test Files

#### `tests/cost_aware_router_test.py`
**Purpose**: Comprehensive unit and integration test suite (29KB)
- Unit tests for routing logic
- Budget tracking validation
- Performance testing (LLM <500ms, total <800ms)
- Database integration tests
- Edge case handling

#### `tests/cost_aware_router/COST_ROUTER_BASIC_TEST.py`
**Purpose**: Basic functionality validation without external dependencies
- Model configuration verification
- Core routing logic testing
- Database initialization checking
- Performance benchmarking
- Docker readiness validation

#### `tests/cost_aware_router/COST_ROUTER_MANUAL_VALIDATION.py`
**Purpose**: Manual validation script for step-by-step testing
- Interactive testing interface
- Detailed error reporting
- Integration compatibility checks
- Performance metrics collection

### 🐳 Docker Testing Files

#### `scripts/cost_aware_router/COST_ROUTER_DOCKER_TEST.sh`
**Purpose**: Automated Docker testing script
- Builds Docker image with Cost-Aware Router
- Runs container with cost router enabled
- Tests API endpoints for cost-aware routing
- Validates routing decisions in container environment
- Checks container logs for cost router activity

#### `docs/cost_aware_router/COST_ROUTER_DOCKER_INSTRUCTIONS.md`
**Purpose**: Step-by-step Docker testing guide
- Complete Docker deployment instructions
- Environment variable configuration
- API testing examples
- Expected results documentation

#### `tests/cost_aware_router/COST_ROUTER_DOCKER_INTEGRATION_TEST.py`
**Purpose**: Comprehensive Docker integration testing
- Container compatibility validation
- API endpoint testing
- Real-time monitoring setup
- Performance validation in Docker

### 📋 Validation and Analysis Files

#### `tests/cost_aware_router/COST_ROUTER_IMPLEMENTATION_VALIDATOR.py`
**Purpose**: Complete implementation validation suite
- File structure verification
- Code syntax validation
- Integration compatibility checking
- Documentation completeness review
- Docker deployment readiness assessment

#### `VERIFICATION_REPORT.md`
**Purpose**: Comprehensive verification report
- Complete test results summary
- Performance metrics documentation
- Implementation status overview
- Production readiness assessment

### 📚 Documentation Files

#### `docs/cost_aware_router_implementation.md`
**Purpose**: Complete implementation documentation
- Technical architecture overview
- Hive Mind development methodology explanation
- Implementation details and design decisions
- Integration guidelines
- Future enhancement roadmap

#### `Assumptions.md` (Updated)
**Purpose**: Project assumptions including Cost-Aware Router
- Anthropic model pricing stability assumptions
- Performance requirement assumptions
- Integration compatibility assumptions

## Test Execution Guide

### Quick Validation
```bash
# Test basic functionality
python tests/cost_aware_router/COST_ROUTER_BASIC_TEST.py

# Validate complete implementation
python tests/cost_aware_router/COST_ROUTER_IMPLEMENTATION_VALIDATOR.py
```

### Docker Testing
```bash
# Quick Docker test
./scripts/cost_aware_router/COST_ROUTER_DOCKER_TEST.sh

# Manual Docker testing
# Follow instructions in docs/cost_aware_router/COST_ROUTER_DOCKER_INSTRUCTIONS.md
```

### Comprehensive Testing
```bash
# Run full test suite
cd tests/
python -m pytest cost_aware_router_test.py -v

# Run performance validation
python run_cost_router_tests.py performance
```

## Expected Test Results

### ✅ Successful Test Indicators
- **Model Configuration**: Haiku and Opus models properly defined
- **Routing Logic**: Simple→Haiku, Complex→Opus, Budget→Haiku
- **Database Integration**: SQLite tables created and functional
- **Performance**: Routing <50ms, Total <800ms
- **Docker Compatibility**: Container builds and runs successfully
- **API Integration**: Endpoints respond with cost metadata

### ❌ Failure Indicators
- Import errors for CostAwareRouter
- Incorrect model routing decisions
- Missing database tables
- Performance above thresholds
- Docker build or runtime failures
- Missing cost information in API responses

## Test Data and Scenarios

### Complexity Test Cases
- **Simple (0.0-0.4)**: Short prompts, basic queries
- **Medium (0.4-0.7)**: Moderate complexity, multi-step requests
- **Complex (0.7-1.0)**: Long prompts, sophisticated analysis

### Budget Test Cases
- **High Budget**: $20+ allows premium model selection
- **Medium Budget**: $1-5 selective premium usage
- **Low Budget**: <$1 forces cheap model only

### Performance Test Cases
- **Latency**: Individual routing decisions <50ms
- **Throughput**: 100 concurrent requests <800ms total
- **Memory**: Efficient memory usage patterns

## Integration Points

### With Existing OpenSesame Components
- **AI Layer**: Enhanced with cost-aware model selection
- **Predictor**: Integrated with k+buffer (k=3, buffer=2) strategy
- **ML Ranker**: Compatible with existing ML ranking pipeline
- **Safety Guardrails**: All safety filtering preserved

### External Dependencies
- **Anthropic API**: Claude 3 Haiku and Opus models
- **SQLite**: Budget tracking and performance metrics
- **Docker**: Containerized deployment environment

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure cost_aware_router.py is in app/models/
2. **Database Errors**: Check data/ directory permissions
3. **Docker Issues**: Verify Docker Desktop WSL integration
4. **API Errors**: Validate ANTHROPIC_API_KEY environment variable

### Debug Commands
```bash
# Check file structure
ls -la app/models/cost_aware_router.py

# Test imports
python -c "from app.models.cost_aware_router import CostAwareRouter; print('OK')"

# Verify Docker
docker --version && docker images | grep opensesame

# Check container logs
docker logs opensesame-predictor | grep -i cost
```

## Contributing to Tests

When adding new Cost-Aware Router tests:
1. Follow the naming convention: `COST_ROUTER_*`
2. Include clear documentation headers
3. Test both success and failure scenarios
4. Validate performance requirements
5. Ensure Docker compatibility
6. Update this README with new test descriptions

---

**Note**: All files prefixed with `COST_ROUTER_` are specifically for testing the Cost-Aware Model Router feature and can be safely ignored for other OpenSesame Predictor functionality testing.