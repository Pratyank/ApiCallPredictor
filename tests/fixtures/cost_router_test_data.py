"""
Test data fixtures for Cost-Aware Router testing.

Provides realistic test scenarios, edge cases, and performance benchmarks
for comprehensive validation of the cost-aware routing system.
"""

from typing import Dict, List, Any
import pytest


@pytest.fixture
def complexity_test_cases() -> List[Dict[str, Any]]:
    """Test cases for complexity scoring validation."""
    return [
        {
            "prompt": "get users",
            "expected_complexity": 0.0,
            "expected_model": "claude-haiku",
            "description": "Simple CRUD operation"
        },
        {
            "prompt": "list all active users with pagination",
            "expected_complexity": 0.1,
            "expected_model": "claude-haiku",
            "description": "Simple query with basic parameters"
        },
        {
            "prompt": "create a comprehensive user management system with role-based access control",
            "expected_complexity": 0.4,
            "expected_model": "claude-sonnet",
            "description": "Medium complexity system design"
        },
        {
            "prompt": """
            Design and implement a distributed microservices architecture 
            for a high-performance e-commerce platform with comprehensive 
            analytics, real-time inventory management, advanced optimization 
            algorithms, and strategic integration patterns for enterprise deployment
            """,
            "expected_complexity": 0.8,
            "expected_model": "claude-opus",
            "description": "Complex architectural design"
        },
        {
            "prompt": "analyze optimize integrate strategic architecture comprehensive detailed enterprise",
            "expected_complexity": 0.7,
            "expected_model": "claude-opus",
            "description": "Multiple complex keywords"
        },
        {
            "prompt": "quick simple basic get list show easy",
            "expected_complexity": 0.0,
            "expected_model": "claude-haiku",
            "description": "Multiple simple keywords"
        }
    ]


@pytest.fixture
def budget_constraint_scenarios() -> List[Dict[str, Any]]:
    """Budget constraint test scenarios."""
    return [
        {
            "name": "abundant_budget",
            "budget_limit": 100.0,
            "prompts": [
                "Complex architectural analysis requiring detailed strategic optimization",
                "Comprehensive system integration with enterprise patterns",
                "Advanced algorithm optimization with performance analysis"
            ],
            "expected_behavior": "Should use optimal models (Opus/Sonnet) without constraint"
        },
        {
            "name": "moderate_budget",
            "budget_limit": 5.0,
            "prompts": [
                "Complex analysis requiring detailed optimization",
                "Medium complexity system design",
                "Simple data retrieval query"
            ],
            "expected_behavior": "Should balance model quality with cost constraints"
        },
        {
            "name": "tight_budget",
            "budget_limit": 0.1,
            "prompts": [
                "Complex architectural analysis",
                "Simple user query",
                "Basic data lookup"
            ],
            "expected_behavior": "Should primarily use Haiku regardless of complexity"
        },
        {
            "name": "depleted_budget",
            "budget_limit": 0.005,
            "prompts": [
                "Any request should still work"
            ],
            "expected_behavior": "Should fallback to cheapest model"
        }
    ]


@pytest.fixture
def performance_test_scenarios() -> List[Dict[str, Any]]:
    """Performance testing scenarios with various load patterns."""
    return [
        {
            "name": "baseline_performance",
            "requests": [
                {"prompt": "Simple query", "expected_llm_ms": 250, "expected_total_ms": 350}
            ],
            "description": "Single request baseline performance"
        },
        {
            "name": "mixed_complexity_load",
            "requests": [
                {"prompt": "Quick data lookup", "expected_llm_ms": 200, "expected_total_ms": 300},
                {"prompt": "Medium analysis task with details", "expected_llm_ms": 350, "expected_total_ms": 450},
                {"prompt": "Complex optimization analysis", "expected_llm_ms": 450, "expected_total_ms": 550}
            ],
            "description": "Mixed complexity requests"
        },
        {
            "name": "high_volume_simple",
            "requests": [
                {"prompt": f"Simple query {i}", "expected_llm_ms": 250, "expected_total_ms": 350}
                for i in range(20)
            ],
            "description": "High volume simple requests"
        },
        {
            "name": "concurrent_complex",
            "requests": [
                {"prompt": f"Complex analysis requiring detailed optimization {i}", 
                 "expected_llm_ms": 450, "expected_total_ms": 550}
                for i in range(5)
            ],
            "description": "Concurrent complex requests",
            "concurrent": True
        }
    ]


@pytest.fixture
def edge_case_scenarios() -> List[Dict[str, Any]]:
    """Edge case scenarios for robustness testing."""
    return [
        {
            "name": "empty_inputs",
            "test_cases": [
                {"prompt": "", "history": [], "should_handle": True},
                {"prompt": None, "history": None, "should_handle": True},
                {"prompt": "   ", "history": [], "should_handle": True}
            ]
        },
        {
            "name": "extreme_inputs",
            "test_cases": [
                {"prompt": "A" * 10000, "history": [], "should_handle": True},
                {"prompt": "Short", "history": [{"endpoint": f"/test/{i}"} for i in range(1000)], "should_handle": True}
            ]
        },
        {
            "name": "special_characters",
            "test_cases": [
                {"prompt": "Test with unicode: 你好世界 🚀 ñáéíóú", "should_handle": True},
                {"prompt": "SQL injection: '; DROP TABLE users; --", "should_handle": True},
                {"prompt": "XSS attempt: <script>alert('test')</script>", "should_handle": True}
            ]
        },
        {
            "name": "boundary_conditions",
            "test_cases": [
                {"prompt": "x" * 999, "description": "Just under 1000 chars"},
                {"prompt": "x" * 1000, "description": "Exactly 1000 chars"},
                {"prompt": "x" * 1001, "description": "Just over 1000 chars"}
            ]
        }
    ]


@pytest.fixture
def integration_test_data() -> Dict[str, Any]:
    """Integration test data for predictor pipeline compatibility."""
    return {
        "predictor_request_format": {
            "prompt": "Create user management API",
            "history": [
                {"api_call": "GET /api/users", "method": "GET", "timestamp": "2023-01-01T10:00:00Z"},
                {"api_call": "POST /api/users", "method": "POST", "timestamp": "2023-01-01T10:01:00Z"}
            ],
            "max_predictions": 3,
            "temperature": 0.7
        },
        "expected_response_format": {
            "required_fields": [
                "model_used", "complexity_score", "cost", "budget_remaining",
                "llm_latency_ms", "total_latency_ms", "predictions", "performance_met"
            ],
            "prediction_structure": {
                "api_call": "string",
                "confidence": "float",
                "model_used": "string"
            }
        },
        "phase5_compatibility": {
            "performance_targets": {
                "llm_latency_ms": 500,
                "total_latency_ms": 800
            },
            "async_processing": True,
            "caching_enabled": True
        }
    }


@pytest.fixture
def model_cost_matrix() -> Dict[str, Dict[str, float]]:
    """Model cost and performance characteristics."""
    return {
        "claude-haiku": {
            "cost_per_1k_tokens": 0.01,
            "avg_latency_ms": 200,
            "quality_score": 0.7,
            "max_tokens": 4000
        },
        "claude-sonnet": {
            "cost_per_1k_tokens": 0.05,
            "avg_latency_ms": 350,
            "quality_score": 0.85,
            "max_tokens": 8000
        },
        "claude-opus": {
            "cost_per_1k_tokens": 0.15,
            "avg_latency_ms": 450,
            "quality_score": 0.95,
            "max_tokens": 8000
        },
        "gpt-3.5-turbo": {
            "cost_per_1k_tokens": 0.02,
            "avg_latency_ms": 250,
            "quality_score": 0.75,
            "max_tokens": 4000
        },
        "gpt-4": {
            "cost_per_1k_tokens": 0.10,
            "avg_latency_ms": 400,
            "quality_score": 0.90,
            "max_tokens": 8000
        }
    }


@pytest.fixture
def database_test_queries() -> List[Dict[str, Any]]:
    """SQL queries for testing database operations."""
    return [
        {
            "name": "budget_tracking_insert",
            "query": """
                INSERT INTO budget_tracking 
                (model_name, prompt_complexity, cost, latency_ms, budget_remaining, tokens_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
            "test_data": ("claude-haiku", 0.3, 0.02, 250, 98.0, 100)
        },
        {
            "name": "cost_optimization_tracking",
            "query": """
                INSERT INTO cost_optimization 
                (original_model, selected_model, complexity_score, cost_savings, performance_impact)
                VALUES (?, ?, ?, ?, ?)
            """,
            "test_data": ("claude-opus", "claude-sonnet", 0.6, 0.10, 0.1)
        },
        {
            "name": "budget_usage_analysis",
            "query": """
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(cost) as total_cost,
                    AVG(latency_ms) as avg_latency,
                    AVG(prompt_complexity) as avg_complexity,
                    model_name
                FROM budget_tracking 
                WHERE timestamp >= datetime('now', '-1 day')
                GROUP BY model_name
            """,
            "expected_columns": ["total_requests", "total_cost", "avg_latency", "avg_complexity", "model_name"]
        },
        {
            "name": "cost_savings_report",
            "query": """
                SELECT 
                    SUM(cost_savings) as total_savings,
                    AVG(performance_impact) as avg_performance_impact,
                    COUNT(*) as optimization_count
                FROM cost_optimization 
                WHERE timestamp >= datetime('now', '-1 day')
            """,
            "expected_columns": ["total_savings", "avg_performance_impact", "optimization_count"]
        }
    ]


@pytest.fixture
def stress_test_config() -> Dict[str, Any]:
    """Configuration for stress testing the cost-aware router."""
    return {
        "load_patterns": {
            "light_load": {
                "concurrent_requests": 5,
                "duration_seconds": 10,
                "requests_per_second": 2
            },
            "moderate_load": {
                "concurrent_requests": 10,
                "duration_seconds": 30,
                "requests_per_second": 5
            },
            "heavy_load": {
                "concurrent_requests": 20,
                "duration_seconds": 60,
                "requests_per_second": 10
            }
        },
        "performance_thresholds": {
            "max_latency_ms": 800,
            "max_llm_latency_ms": 500,
            "min_success_rate": 0.95,
            "max_error_rate": 0.05
        },
        "budget_constraints": {
            "tight": 1.0,
            "moderate": 10.0,
            "abundant": 100.0
        }
    }


@pytest.fixture
def mock_external_apis() -> Dict[str, Any]:
    """Mock configurations for external API dependencies."""
    return {
        "anthropic_api": {
            "models": ["claude-haiku", "claude-sonnet", "claude-opus"],
            "mock_responses": {
                "claude-haiku": {
                    "latency_ms": 200,
                    "cost_multiplier": 1.0,
                    "quality_score": 0.7
                },
                "claude-sonnet": {
                    "latency_ms": 350,
                    "cost_multiplier": 5.0,
                    "quality_score": 0.85
                },
                "claude-opus": {
                    "latency_ms": 450,
                    "cost_multiplier": 15.0,
                    "quality_score": 0.95
                }
            }
        },
        "openai_api": {
            "models": ["gpt-3.5-turbo", "gpt-4"],
            "mock_responses": {
                "gpt-3.5-turbo": {
                    "latency_ms": 250,
                    "cost_multiplier": 2.0,
                    "quality_score": 0.75
                },
                "gpt-4": {
                    "latency_ms": 400,
                    "cost_multiplier": 10.0,
                    "quality_score": 0.90
                }
            }
        }
    }