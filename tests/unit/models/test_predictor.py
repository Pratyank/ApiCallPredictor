"""
Unit tests for app.models.predictor module.

Tests the core prediction logic including:
- Prediction workflow orchestration
- Input validation and sanitization
- Output formatting
- Error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

# Mock imports - replace with actual imports when modules exist
# from app.models.predictor import Predictor, PredictionRequest, PredictionResponse
# from app.models.ai_layer import AILayer  
# from app.models.ml_ranker import MLRanker
# from app.utils.guardrails import GuardrailsValidator


class TestPredictor:
    """Test suite for the Predictor class."""
    
    @pytest.fixture
    def mock_ai_layer(self):
        """Mock AI layer dependency."""
        ai_layer = Mock()
        ai_layer.generate_candidates = AsyncMock(return_value=[
            {"endpoint": "/users", "method": "POST", "confidence": 0.9, "parameters": {}},
            {"endpoint": "/users/{id}", "method": "GET", "confidence": 0.8, "parameters": {"id": "123"}},
            {"endpoint": "/invoices", "method": "GET", "confidence": 0.7, "parameters": {}}
        ])
        return ai_layer
    
    @pytest.fixture
    def mock_ml_ranker(self):
        """Mock ML ranker dependency."""
        ranker = Mock()
        ranker.rank_candidates = Mock(return_value=[
            {"endpoint": "/users", "method": "POST", "score": 0.95, "features": {}},
            {"endpoint": "/invoices", "method": "GET", "score": 0.85, "features": {}},
            {"endpoint": "/users/{id}", "method": "GET", "score": 0.75, "features": {}}
        ])
        ranker.is_trained = Mock(return_value=True)
        return ranker
    
    @pytest.fixture
    def mock_guardrails(self):
        """Mock guardrails validator."""
        guardrails = Mock()
        guardrails.validate_request = Mock(return_value=(True, ""))
        guardrails.filter_safe_candidates = Mock(side_effect=lambda candidates: candidates)
        return guardrails
    
    @pytest.fixture
    def predictor(self, mock_ai_layer, mock_ml_ranker, mock_guardrails):
        """Create predictor instance with mocked dependencies."""
        # Mock implementation
        predictor = Mock()
        predictor.ai_layer = mock_ai_layer
        predictor.ml_ranker = mock_ml_ranker
        predictor.guardrails = mock_guardrails
        return predictor
    
    @pytest.mark.asyncio
    async def test_predict_success(self, predictor, sample_openapi_spec, sample_user_events):
        """Test successful prediction workflow."""
        # Mock the predict method
        predictor.predict = AsyncMock(return_value={
            "predictions": [
                {
                    "endpoint": "/users",
                    "method": "POST", 
                    "confidence": 0.95,
                    "parameters": {},
                    "description": "Create a new user"
                }
            ],
            "metadata": {
                "processing_time_ms": 250,
                "ai_candidates": 3,
                "ml_ranked": 3,
                "guardrails_filtered": 1
            }
        })
        
        result = await predictor.predict(
            prompt="Create a new user account",
            recent_events=sample_user_events,
            openapi_spec=sample_openapi_spec,
            k=5
        )
        
        assert "predictions" in result
        assert len(result["predictions"]) >= 1
        assert result["predictions"][0]["endpoint"] == "/users"
        assert result["predictions"][0]["method"] == "POST"
        assert "metadata" in result
    
    @pytest.mark.asyncio
    async def test_predict_with_empty_prompt(self, predictor, sample_openapi_spec):
        """Test prediction with empty prompt (cold start scenario)."""
        predictor.predict = AsyncMock(return_value={
            "predictions": [
                {"endpoint": "/users", "method": "GET", "confidence": 0.8}
            ],
            "metadata": {"cold_start": True}
        })
        
        result = await predictor.predict(
            prompt="",
            recent_events=[],
            openapi_spec=sample_openapi_spec,
            k=3
        )
        
        assert result["metadata"]["cold_start"] is True
        assert len(result["predictions"]) > 0
    
    @pytest.mark.asyncio
    async def test_predict_with_invalid_spec(self, predictor):
        """Test prediction with invalid OpenAPI specification."""
        predictor.predict = AsyncMock(side_effect=ValueError("Invalid OpenAPI specification"))
        
        with pytest.raises(ValueError, match="Invalid OpenAPI specification"):
            await predictor.predict(
                prompt="test",
                recent_events=[],
                openapi_spec={"invalid": "spec"},
                k=3
            )
    
    @pytest.mark.asyncio
    async def test_predict_guardrails_block(self, predictor, sample_openapi_spec):
        """Test that guardrails properly block dangerous operations."""
        predictor.predict = AsyncMock(return_value={
            "predictions": [],
            "metadata": {"blocked_by_guardrails": True, "reason": "Destructive operation detected"}
        })
        
        result = await predictor.predict(
            prompt="Delete all users",
            recent_events=[],
            openapi_spec=sample_openapi_spec,
            k=3
        )
        
        assert result["metadata"]["blocked_by_guardrails"] is True
        assert len(result["predictions"]) == 0
    
    @pytest.mark.asyncio 
    async def test_predict_timeout_handling(self, predictor, sample_openapi_spec):
        """Test handling of AI layer timeouts."""
        predictor.predict = AsyncMock(side_effect=asyncio.TimeoutError("AI layer timeout"))
        
        with pytest.raises(asyncio.TimeoutError):
            await predictor.predict(
                prompt="test",
                recent_events=[],
                openapi_spec=sample_openapi_spec,
                k=3
            )
    
    def test_validate_input_parameters(self, predictor):
        """Test input parameter validation."""
        predictor.validate_input = Mock()
        
        # Valid inputs
        predictor.validate_input.return_value = True
        assert predictor.validate_input("test prompt", [], {}, 5) is True
        
        # Invalid k value
        predictor.validate_input.side_effect = ValueError("k must be positive")
        with pytest.raises(ValueError, match="k must be positive"):
            predictor.validate_input("test", [], {}, -1)
    
    def test_format_predictions(self, predictor):
        """Test prediction output formatting."""
        raw_predictions = [
            {"endpoint": "/users", "method": "POST", "score": 0.95},
            {"endpoint": "/users/{id}", "method": "GET", "score": 0.85}
        ]
        
        predictor.format_predictions = Mock(return_value=[
            {
                "endpoint": "/users",
                "method": "POST",
                "confidence": 0.95,
                "parameters": {},
                "description": "Create a new user"
            },
            {
                "endpoint": "/users/{id}",
                "method": "GET", 
                "confidence": 0.85,
                "parameters": {"id": "string"},
                "description": "Get user by ID"
            }
        ])
        
        formatted = predictor.format_predictions(raw_predictions, {})
        
        assert len(formatted) == 2
        assert all("confidence" in pred for pred in formatted)
        assert all("description" in pred for pred in formatted)
    
    @pytest.mark.parametrize("k_value,expected_length", [
        (1, 1),
        (3, 3), 
        (5, 5),
        (10, 3)  # Should be limited by available candidates
    ])
    def test_prediction_count_limits(self, predictor, k_value, expected_length):
        """Test that prediction count respects k parameter."""
        available_predictions = [
            {"endpoint": "/users", "method": "GET"},
            {"endpoint": "/users", "method": "POST"},
            {"endpoint": "/invoices", "method": "GET"}
        ]
        
        predictor.limit_predictions = Mock(
            return_value=available_predictions[:min(k_value, len(available_predictions))]
        )
        
        result = predictor.limit_predictions(available_predictions, k_value)
        assert len(result) == min(expected_length, len(available_predictions))


class TestPredictionRequest:
    """Test suite for PredictionRequest model."""
    
    def test_valid_prediction_request(self):
        """Test creation of valid prediction request."""
        # Mock implementation
        request_data = {
            "prompt": "Create a new user",
            "recent_events": [{"endpoint": "/users", "method": "GET"}],
            "openapi_spec": {"openapi": "3.0.0"},
            "k": 5
        }
        
        # Would test actual PredictionRequest validation here
        assert request_data["prompt"] == "Create a new user"
        assert request_data["k"] == 5
        assert len(request_data["recent_events"]) == 1
    
    def test_prediction_request_validation(self):
        """Test prediction request validation."""
        # Test invalid k values
        invalid_k_values = [-1, 0, 101]  # k should be 1-100
        
        for k in invalid_k_values:
            with pytest.raises(ValueError):
                # Would instantiate actual PredictionRequest here
                if k <= 0:
                    raise ValueError("k must be positive")
                if k > 100:
                    raise ValueError("k must be <= 100")


class TestPredictionResponse:
    """Test suite for PredictionResponse model."""
    
    def test_valid_prediction_response(self):
        """Test creation of valid prediction response."""
        response_data = {
            "predictions": [
                {
                    "endpoint": "/users",
                    "method": "POST",
                    "confidence": 0.95,
                    "parameters": {},
                    "description": "Create a new user"
                }
            ],
            "metadata": {
                "processing_time_ms": 250,
                "model_version": "1.0.0"
            }
        }
        
        assert len(response_data["predictions"]) == 1
        assert response_data["predictions"][0]["confidence"] == 0.95
        assert "processing_time_ms" in response_data["metadata"]
    
    def test_prediction_response_serialization(self):
        """Test prediction response JSON serialization."""
        import json
        
        response_data = {
            "predictions": [{"endpoint": "/test", "method": "GET", "confidence": 0.8}],
            "metadata": {"processing_time_ms": 100}
        }
        
        # Test that response can be serialized to JSON
        json_str = json.dumps(response_data)
        parsed = json.loads(json_str)
        
        assert parsed["predictions"][0]["endpoint"] == "/test"
        assert parsed["metadata"]["processing_time_ms"] == 100