"""
Integration tests for FastAPI endpoints.

Tests the complete API functionality including:
- /predict endpoint with various scenarios
- Request/response validation
- Error handling and status codes
- Authentication and authorization
- Rate limiting
- Health checks
"""

import pytest
import asyncio
import json
from httpx import AsyncClient
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from typing import Dict, Any

# Mock imports - replace with actual imports when modules exist
# from app.main import app
# from app.models.predictor import Predictor
# from app.config import settings


class TestPredictEndpoint:
    """Integration tests for the /predict endpoint."""
    
    @pytest.fixture
    def async_client(self, test_client):
        """Create async client for testing."""
        return AsyncClient(app=test_client.app, base_url="http://testserver")
    
    @pytest.mark.asyncio
    async def test_predict_success(self, async_client, sample_openapi_spec, sample_user_events):
        """Test successful prediction request."""
        request_payload = {
            "prompt": "Create a new user account",
            "recent_events": sample_user_events,
            "openapi_spec": sample_openapi_spec,
            "k": 3
        }
        
        expected_response = {
            "predictions": [
                {
                    "endpoint": "/users",
                    "method": "POST",
                    "confidence": 0.95,
                    "parameters": {
                        "name": "string",
                        "email": "string"
                    },
                    "description": "Create a new user",
                    "reasoning": "User explicitly requested to create account"
                }
            ],
            "metadata": {
                "processing_time_ms": 234,
                "ai_provider": "openai",
                "ml_model_version": "1.0.0",
                "guardrails_applied": True,
                "total_candidates": 5,
                "filtered_candidates": 3
            }
        }
        
        # Mock the endpoint response
        with patch("app.main.predict_endpoint", return_value=expected_response):
            response = await async_client.post("/predict", json=request_payload)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "predictions" in response_data
        assert len(response_data["predictions"]) >= 1
        assert response_data["predictions"][0]["endpoint"] == "/users"
        assert response_data["predictions"][0]["method"] == "POST"
        assert "metadata" in response_data
        assert response_data["metadata"]["processing_time_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_predict_with_empty_prompt(self, async_client, sample_openapi_spec):
        """Test prediction with empty prompt (cold start)."""
        request_payload = {
            "prompt": "",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 3
        }
        
        expected_response = {
            "predictions": [
                {"endpoint": "/users", "method": "GET", "confidence": 0.7, "description": "List users"}
            ],
            "metadata": {
                "cold_start": True,
                "processing_time_ms": 89
            }
        }
        
        with patch("app.main.predict_endpoint", return_value=expected_response):
            response = await async_client.post("/predict", json=request_payload)
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["metadata"]["cold_start"] is True
    
    @pytest.mark.asyncio
    async def test_predict_invalid_request_body(self, async_client):
        """Test prediction with invalid request body."""
        invalid_payloads = [
            {},  # Empty payload
            {"prompt": "test"},  # Missing required fields
            {"prompt": "test", "k": -1},  # Invalid k value
            {"prompt": "test", "openapi_spec": "invalid"},  # Invalid spec format
            {"prompt": None, "openapi_spec": {}}  # Null prompt
        ]
        
        for payload in invalid_payloads:
            response = await async_client.post("/predict", json=payload)
            
            assert response.status_code == 422  # Unprocessable Entity
            error_data = response.json()
            assert "detail" in error_data
    
    @pytest.mark.asyncio
    async def test_predict_guardrails_block(self, async_client, sample_openapi_spec):
        """Test that guardrails properly block dangerous requests."""
        malicious_request = {
            "prompt": "Delete all users permanently",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 3
        }
        
        expected_response = {
            "predictions": [],
            "metadata": {
                "blocked_by_guardrails": True,
                "block_reason": "Destructive operation detected",
                "processing_time_ms": 12
            }
        }
        
        with patch("app.main.predict_endpoint", return_value=expected_response):
            response = await async_client.post("/predict", json=malicious_request)
        
        assert response.status_code == 200  # Request succeeds but returns empty predictions
        response_data = response.json()
        assert len(response_data["predictions"]) == 0
        assert response_data["metadata"]["blocked_by_guardrails"] is True
    
    @pytest.mark.asyncio
    async def test_predict_rate_limiting(self, async_client, sample_openapi_spec):
        """Test API rate limiting functionality."""
        request_payload = {
            "prompt": "test request",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 1
        }
        
        # Make multiple requests rapidly
        responses = []
        for i in range(15):  # Assume limit is 10 requests per minute
            response = await async_client.post("/predict", json=request_payload)
            responses.append(response)
        
        # First 10 should succeed, rest should be rate limited
        success_responses = [r for r in responses if r.status_code == 200]
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        
        assert len(success_responses) <= 10
        assert len(rate_limited_responses) >= 5
        
        # Check rate limit headers
        if rate_limited_responses:
            rate_limit_response = rate_limited_responses[0]
            assert "X-RateLimit-Limit" in rate_limit_response.headers
            assert "X-RateLimit-Remaining" in rate_limit_response.headers
    
    @pytest.mark.asyncio
    async def test_predict_performance_requirements(self, async_client, sample_openapi_spec, performance_monitor):
        """Test that prediction responses meet performance requirements."""
        request_payload = {
            "prompt": "Update user profile information",
            "recent_events": [{"endpoint": "/users/123", "method": "GET"}],
            "openapi_spec": sample_openapi_spec,
            "k": 5
        }
        
        performance_monitor.start_timer("api_request")
        
        response = await async_client.post("/predict", json=request_payload)
        
        performance_monitor.end_timer("api_request")
        
        assert response.status_code == 200
        
        # Check response time requirement (< 800ms)
        response_time_ms = performance_monitor.get_duration("api_request") * 1000
        assert response_time_ms < 800
        
        # Check that metadata includes timing information
        response_data = response.json()
        if "metadata" in response_data:
            assert "processing_time_ms" in response_data["metadata"]
            assert response_data["metadata"]["processing_time_ms"] < 800
    
    @pytest.mark.asyncio
    async def test_predict_large_payload_handling(self, async_client):
        """Test handling of large request payloads."""
        # Create large OpenAPI spec
        large_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Large API", "version": "1.0.0"},
            "paths": {}
        }
        
        # Add 1000 endpoints
        for i in range(1000):
            large_spec["paths"][f"/endpoint_{i}"] = {
                "get": {
                    "operationId": f"getEndpoint{i}",
                    "description": f"Get endpoint {i} data"
                }
            }
        
        # Create large event history
        large_events = [
            {"endpoint": f"/endpoint_{i}", "method": "GET", "timestamp": "2023-01-01T10:00:00Z"}
            for i in range(100)
        ]
        
        request_payload = {
            "prompt": "Get some data",
            "recent_events": large_events,
            "openapi_spec": large_spec,
            "k": 10
        }
        
        response = await async_client.post("/predict", json=request_payload)
        
        # Should either succeed or return 413 (Payload Too Large)
        assert response.status_code in [200, 413]
        
        if response.status_code == 200:
            response_data = response.json()
            assert "predictions" in response_data
        else:
            # Check that appropriate error message is returned
            error_data = response.json()
            assert "payload" in error_data["detail"].lower()


class TestHealthEndpoint:
    """Integration tests for health check endpoint."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, async_client):
        """Test successful health check."""
        response = await async_client.get("/health")
        
        assert response.status_code == 200
        health_data = response.json()
        
        assert health_data["status"] == "ok"
        assert "timestamp" in health_data
        assert "version" in health_data
        assert "uptime" in health_data
    
    @pytest.mark.asyncio
    async def test_health_check_with_dependencies(self, async_client):
        """Test health check including dependency status."""
        response = await async_client.get("/health?include_dependencies=true")
        
        assert response.status_code == 200
        health_data = response.json()
        
        assert "dependencies" in health_data
        assert "ai_provider" in health_data["dependencies"]
        assert "ml_model" in health_data["dependencies"]
        assert "database" in health_data["dependencies"]
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self, async_client, performance_monitor):
        """Test health check response time."""
        performance_monitor.start_timer("health_check")
        
        response = await async_client.get("/health")
        
        performance_monitor.end_timer("health_check")
        
        assert response.status_code == 200
        
        # Health check should be very fast (< 100ms)
        response_time_ms = performance_monitor.get_duration("health_check") * 1000
        assert response_time_ms < 100


class TestMetricsEndpoint:
    """Integration tests for metrics endpoint."""
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, async_client):
        """Test metrics collection endpoint."""
        response = await async_client.get("/metrics")
        
        assert response.status_code == 200
        metrics_data = response.json()
        
        assert "request_count" in metrics_data
        assert "response_time_avg" in metrics_data
        assert "error_rate" in metrics_data
        assert "active_predictions" in metrics_data
    
    @pytest.mark.asyncio
    async def test_metrics_prometheus_format(self, async_client):
        """Test Prometheus format metrics."""
        response = await async_client.get("/metrics", headers={"Accept": "text/plain"})
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        
        metrics_text = response.text
        assert "# HELP" in metrics_text
        assert "# TYPE" in metrics_text
        assert "http_requests_total" in metrics_text


class TestAPISecurityIntegration:
    """Integration tests for API security features."""
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, async_client):
        """Test CORS headers in responses."""
        response = await async_client.options("/predict")
        
        assert response.status_code in [200, 204]
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers
    
    @pytest.mark.asyncio
    async def test_security_headers(self, async_client):
        """Test security headers in responses."""
        response = await async_client.get("/health")
        
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security"
        ]
        
        for header in security_headers:
            assert header in response.headers
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self, async_client, security_test_payloads):
        """Test that malicious inputs are properly sanitized."""
        for payload_type, malicious_payload in security_test_payloads.items():
            response = await async_client.post("/predict", json=malicious_payload)
            
            # Should either be sanitized and processed (200) or rejected (400/422)
            assert response.status_code in [200, 400, 422]
            
            if response.status_code == 200:
                # If processed, should not contain malicious content
                response_text = response.text.lower()
                malicious_indicators = ["script", "drop table", "union select", "{{", "${"]
                
                for indicator in malicious_indicators:
                    assert indicator not in response_text


class TestAPIErrorHandling:
    """Integration tests for API error handling."""
    
    @pytest.mark.asyncio
    async def test_500_error_handling(self, async_client, sample_openapi_spec):
        """Test handling of internal server errors."""
        request_payload = {
            "prompt": "test",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 3
        }
        
        # Mock an internal error
        with patch("app.models.predictor.Predictor.predict", side_effect=Exception("Internal error")):
            response = await async_client.post("/predict", json=request_payload)
        
        assert response.status_code == 500
        error_data = response.json()
        
        assert "detail" in error_data
        assert "error_id" in error_data  # For error tracking
        # Should not expose internal error details in production
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, async_client, sample_openapi_spec):
        """Test handling of request timeouts."""
        request_payload = {
            "prompt": "test",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 3
        }
        
        # Mock a timeout
        with patch("app.models.predictor.Predictor.predict", side_effect=asyncio.TimeoutError("Request timeout")):
            response = await async_client.post("/predict", json=request_payload, timeout=1.0)
        
        assert response.status_code == 408  # Request Timeout
        error_data = response.json()
        assert "timeout" in error_data["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_validation_error_details(self, async_client):
        """Test that validation errors provide helpful details."""
        invalid_request = {
            "prompt": 123,  # Should be string
            "k": "invalid",  # Should be integer
            "recent_events": "not_a_list"  # Should be list
        }
        
        response = await async_client.post("/predict", json=invalid_request)
        
        assert response.status_code == 422
        error_data = response.json()
        
        assert "detail" in error_data
        assert isinstance(error_data["detail"], list)
        
        # Should have specific field-level errors
        field_errors = {error["loc"][-1] for error in error_data["detail"]}
        assert "prompt" in field_errors
        assert "k" in field_errors


class TestAPIContractCompliance:
    """Integration tests for API contract compliance."""
    
    @pytest.mark.asyncio
    async def test_openapi_spec_compliance(self, async_client):
        """Test that API responses match OpenAPI specification."""
        # Get the OpenAPI spec
        response = await async_client.get("/openapi.json")
        
        assert response.status_code == 200
        openapi_spec = response.json()
        
        assert openapi_spec["openapi"].startswith("3.")
        assert "paths" in openapi_spec
        assert "/predict" in openapi_spec["paths"]
        assert "/health" in openapi_spec["paths"]
    
    @pytest.mark.asyncio
    async def test_response_schema_validation(self, async_client, sample_openapi_spec, sample_user_events):
        """Test that response schemas match specification."""
        request_payload = {
            "prompt": "test",
            "recent_events": sample_user_events,
            "openapi_spec": sample_openapi_spec,
            "k": 3
        }
        
        response = await async_client.post("/predict", json=request_payload)
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Validate response structure
            assert isinstance(response_data, dict)
            assert "predictions" in response_data
            assert "metadata" in response_data
            assert isinstance(response_data["predictions"], list)
            assert isinstance(response_data["metadata"], dict)
            
            # Validate prediction structure
            if response_data["predictions"]:
                prediction = response_data["predictions"][0]
                required_fields = ["endpoint", "method", "confidence"]
                for field in required_fields:
                    assert field in prediction
    
    @pytest.mark.asyncio 
    async def test_content_type_handling(self, async_client):
        """Test proper content type handling."""
        # Test JSON content type
        response = await async_client.post("/predict", 
                                         json={"prompt": "test"},
                                         headers={"Content-Type": "application/json"})
        
        # Should accept JSON
        assert response.status_code in [200, 422]  # 422 for incomplete request
        
        # Test unsupported content type
        response = await async_client.post("/predict",
                                         data="test data",
                                         headers={"Content-Type": "text/plain"})
        
        assert response.status_code == 415  # Unsupported Media Type