"""
Test configuration and shared fixtures for opensesame-predictor testing framework.

This module provides:
- Test client fixtures for FastAPI app
- Database fixtures for ML model testing
- Mock fixtures for external API calls
- Performance testing utilities
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator, Dict, Any
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient
import tempfile
import json
import os

# Mock imports - will be replaced when actual modules exist
# from app.main import app
# from app.config import settings
# from app.models.predictor import Predictor
# from app.models.ai_layer import AILayer
# from app.models.ml_ranker import MLRanker


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI application."""
    # Mock implementation - replace with actual app import
    from fastapi import FastAPI
    mock_app = FastAPI()
    
    @mock_app.get("/health")
    def health_check():
        return {"status": "ok"}
    
    with TestClient(mock_app) as client:
        yield client


@pytest.fixture
def sample_openapi_spec() -> Dict[str, Any]:
    """Sample OpenAPI specification for testing."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/users": {
                "get": {
                    "operationId": "listUsers",
                    "description": "List all users",
                    "responses": {"200": {"description": "Success"}}
                },
                "post": {
                    "operationId": "createUser",
                    "description": "Create a new user",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "email": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/users/{id}": {
                "get": {
                    "operationId": "getUser",
                    "description": "Get user by ID",
                    "parameters": [{
                        "name": "id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"}
                    }]
                },
                "put": {
                    "operationId": "updateUser",
                    "description": "Update user by ID"
                },
                "delete": {
                    "operationId": "deleteUser",
                    "description": "Delete user by ID"
                }
            },
            "/invoices": {
                "get": {
                    "operationId": "listInvoices",
                    "description": "List invoices"
                },
                "post": {
                    "operationId": "createInvoice",
                    "description": "Create new invoice"
                }
            }
        }
    }


@pytest.fixture
def sample_user_events() -> list:
    """Sample user event history for testing."""
    return [
        {"endpoint": "/users", "method": "GET", "timestamp": "2023-01-01T10:00:00Z"},
        {"endpoint": "/users/123", "method": "GET", "timestamp": "2023-01-01T10:01:00Z"},
        {"endpoint": "/users/123", "method": "PUT", "timestamp": "2023-01-01T10:02:00Z"},
    ]


@pytest.fixture
def mock_ai_layer() -> Mock:
    """Mock AI layer for testing without external API calls."""
    mock_ai = Mock()
    mock_ai.generate_candidates = AsyncMock(return_value=[
        {"endpoint": "/users", "method": "POST", "confidence": 0.9},
        {"endpoint": "/users/{id}", "method": "GET", "confidence": 0.8},
        {"endpoint": "/invoices", "method": "GET", "confidence": 0.7},
    ])
    return mock_ai


@pytest.fixture
def mock_ml_ranker() -> Mock:
    """Mock ML ranker for testing without trained model."""
    mock_ranker = Mock()
    mock_ranker.rank_candidates = Mock(return_value=[
        {"endpoint": "/users", "method": "POST", "score": 0.95},
        {"endpoint": "/invoices", "method": "GET", "score": 0.85},
        {"endpoint": "/users/{id}", "method": "GET", "score": 0.75},
    ])
    mock_ranker.is_trained = Mock(return_value=True)
    return mock_ranker


@pytest.fixture
def temp_model_dir() -> Generator[str, None, None]:
    """Temporary directory for model storage during tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def performance_config() -> Dict[str, Any]:
    """Configuration for performance testing."""
    return {
        "max_response_time_ms": 800,
        "llm_timeout_ms": 500,
        "ml_timeout_ms": 100,
        "concurrent_requests": 10,
        "load_test_duration": 30,
        "memory_limit_mb": 512,
        "cpu_limit": 2.0
    }


@pytest.fixture
def security_test_payloads() -> Dict[str, Any]:
    """Malicious payloads for security testing."""
    return {
        "sql_injection": {
            "prompt": "'; DROP TABLE users; --",
            "events": [{"endpoint": "'; DELETE FROM events; --"}]
        },
        "xss_payload": {
            "prompt": "<script>alert('xss')</script>",
            "events": [{"endpoint": "<img src=x onerror=alert(1)>"}]
        },
        "large_payload": {
            "prompt": "A" * 10000,
            "events": [{"endpoint": "/test"} for _ in range(1000)]
        },
        "null_injection": {
            "prompt": None,
            "events": [{"endpoint": None, "method": None}]
        }
    }


@pytest.fixture
def docker_test_config() -> Dict[str, str]:
    """Docker container testing configuration."""
    return {
        "image_name": "opensesame-predictor:test",
        "container_port": "8000",
        "host_port": "8001",
        "memory_limit": "512m",
        "cpu_limit": "2.0",
        "health_endpoint": "/health"
    }


# Performance monitoring utilities
class PerformanceMonitor:
    """Utility class for monitoring test performance."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        import time
        self.metrics[name] = {"start": time.time()}
    
    def end_timer(self, name: str):
        import time
        if name in self.metrics:
            self.metrics[name]["end"] = time.time()
            self.metrics[name]["duration"] = self.metrics[name]["end"] - self.metrics[name]["start"]
    
    def get_duration(self, name: str) -> float:
        return self.metrics.get(name, {}).get("duration", 0)


@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Performance monitoring fixture."""
    return PerformanceMonitor()