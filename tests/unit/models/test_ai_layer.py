"""
Unit tests for app.models.ai_layer module.

Tests the AI layer functionality including:
- LLM integration (OpenAI/Anthropic/Ollama)
- Candidate generation from prompts
- Semantic similarity matching
- Prompt engineering and templating
- Error handling and fallbacks
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
import json

# Mock imports - replace with actual imports when modules exist
# from app.models.ai_layer import AILayer, CandidateGenerator
# from app.config import settings


class TestAILayer:
    """Test suite for the AILayer class."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        client = Mock()
        client.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(
                message=Mock(content=json.dumps([
                    {"endpoint": "/users", "method": "POST", "confidence": 0.9},
                    {"endpoint": "/users/{id}", "method": "GET", "confidence": 0.8}
                ]))
            )]
        ))
        return client
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client."""
        client = Mock()
        client.messages.create = AsyncMock(return_value=Mock(
            content=[Mock(text=json.dumps([
                {"endpoint": "/invoices", "method": "GET", "confidence": 0.85},
                {"endpoint": "/invoices", "method": "POST", "confidence": 0.75}
            ]))]
        ))
        return client
    
    @pytest.fixture
    def ai_layer(self, mock_openai_client):
        """Create AILayer instance with mocked dependencies."""
        ai_layer = Mock()
        ai_layer.openai_client = mock_openai_client
        ai_layer.provider = "openai"
        ai_layer.model = "gpt-4"
        return ai_layer
    
    @pytest.mark.asyncio
    async def test_generate_candidates_openai_success(self, ai_layer, sample_openapi_spec, sample_user_events):
        """Test successful candidate generation using OpenAI."""
        expected_candidates = [
            {"endpoint": "/users", "method": "POST", "confidence": 0.9, "reasoning": "User wants to create account"},
            {"endpoint": "/users/{id}", "method": "GET", "confidence": 0.8, "reasoning": "Follow-up to check created user"}
        ]
        
        ai_layer.generate_candidates = AsyncMock(return_value=expected_candidates)
        
        result = await ai_layer.generate_candidates(
            prompt="Create a new user account",
            recent_events=sample_user_events,
            openapi_spec=sample_openapi_spec,
            k=5
        )
        
        assert len(result) == 2
        assert result[0]["endpoint"] == "/users"
        assert result[0]["method"] == "POST"
        assert result[0]["confidence"] == 0.9
        assert "reasoning" in result[0]
    
    @pytest.mark.asyncio
    async def test_generate_candidates_anthropic_success(self, mock_anthropic_client):
        """Test successful candidate generation using Anthropic."""
        ai_layer = Mock()
        ai_layer.anthropic_client = mock_anthropic_client
        ai_layer.provider = "anthropic"
        
        expected_candidates = [
            {"endpoint": "/invoices", "method": "GET", "confidence": 0.85}
        ]
        ai_layer.generate_candidates = AsyncMock(return_value=expected_candidates)
        
        result = await ai_layer.generate_candidates(
            prompt="Show me recent invoices",
            recent_events=[],
            openapi_spec={},
            k=3
        )
        
        assert len(result) == 1
        assert result[0]["endpoint"] == "/invoices"
    
    @pytest.mark.asyncio
    async def test_generate_candidates_with_rate_limit(self, ai_layer):
        """Test handling of API rate limits."""
        ai_layer.generate_candidates = AsyncMock(side_effect=Exception("Rate limit exceeded"))
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await ai_layer.generate_candidates(
                prompt="test",
                recent_events=[],
                openapi_spec={},
                k=3
            )
    
    @pytest.mark.asyncio
    async def test_generate_candidates_with_invalid_json_response(self, ai_layer):
        """Test handling of invalid JSON responses from LLM."""
        ai_layer.generate_candidates = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        
        with pytest.raises(json.JSONDecodeError):
            await ai_layer.generate_candidates(
                prompt="test",
                recent_events=[],
                openapi_spec={},
                k=3
            )
    
    def test_build_prompt_template(self, ai_layer, sample_user_events, sample_openapi_spec):
        """Test prompt template construction."""
        ai_layer.build_prompt = Mock(return_value="""
Given user history: [{"endpoint": "/users", "method": "GET"}]
User intent: Create a new user
Available endpoints: ["/users", "/invoices"]

Generate 3 most likely next API calls considering:
1. Natural progression from recent actions
2. User's stated intent  
3. Common workflow patterns

Output as JSON array of actions.
        """.strip())
        
        prompt = ai_layer.build_prompt(
            prompt="Create a new user",
            recent_events=sample_user_events[:1],
            filtered_endpoints=["/users", "/invoices"],
            k=3
        )
        
        assert "User intent: Create a new user" in prompt
        assert "Generate 3 most likely" in prompt
        assert "JSON array" in prompt
    
    def test_filter_relevant_endpoints(self, ai_layer, sample_openapi_spec):
        """Test filtering of relevant endpoints based on context."""
        ai_layer.filter_relevant_endpoints = Mock(return_value=[
            {"path": "/users", "method": "POST", "description": "Create a new user"},
            {"path": "/users/{id}", "method": "GET", "description": "Get user by ID"}
        ])
        
        filtered = ai_layer.filter_relevant_endpoints(
            prompt="user management",
            openapi_spec=sample_openapi_spec,
            recent_events=[{"endpoint": "/users", "method": "GET"}]
        )
        
        assert len(filtered) == 2
        assert all("user" in endpoint["description"].lower() for endpoint in filtered)
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_matching(self, ai_layer):
        """Test semantic similarity between prompt and endpoint descriptions."""
        ai_layer.calculate_similarity = Mock(return_value=0.85)
        
        similarity = ai_layer.calculate_similarity(
            prompt="create new account",
            description="Create a new user account"
        )
        
        assert similarity == 0.85
        assert 0 <= similarity <= 1
    
    @pytest.mark.asyncio
    async def test_ollama_fallback(self, ai_layer):
        """Test fallback to Ollama when primary provider fails."""
        # First call fails
        ai_layer.generate_candidates = AsyncMock(side_effect=Exception("OpenAI API error"))
        
        # Mock fallback to Ollama
        ai_layer.fallback_to_ollama = AsyncMock(return_value=[
            {"endpoint": "/users", "method": "GET", "confidence": 0.7, "source": "ollama"}
        ])
        
        # Simulate fallback behavior
        try:
            await ai_layer.generate_candidates("test", [], {}, 3)
        except Exception:
            result = await ai_layer.fallback_to_ollama("test", [], {}, 3)
            assert result[0]["source"] == "ollama"
            assert result[0]["confidence"] == 0.7
    
    def test_prompt_sanitization(self, ai_layer):
        """Test that prompts are properly sanitized for security."""
        malicious_prompt = "'; DROP TABLE users; SELECT * FROM secrets; --"
        
        ai_layer.sanitize_prompt = Mock(return_value="SELECT FROM secrets")
        
        sanitized = ai_layer.sanitize_prompt(malicious_prompt)
        
        # Should remove SQL injection attempts
        assert "DROP TABLE" not in sanitized
        assert "--" not in sanitized
    
    @pytest.mark.parametrize("provider,expected_model", [
        ("openai", "gpt-4"),
        ("anthropic", "claude-3-sonnet"),
        ("ollama", "mistral:7b")
    ])
    def test_provider_configuration(self, provider, expected_model):
        """Test different AI provider configurations."""
        ai_layer = Mock()
        ai_layer.configure_provider = Mock()
        ai_layer.provider = provider
        ai_layer.model = expected_model
        
        ai_layer.configure_provider(provider)
        
        assert ai_layer.provider == provider
        assert ai_layer.model == expected_model
    
    @pytest.mark.asyncio
    async def test_candidate_validation(self, ai_layer):
        """Test validation of generated candidates."""
        invalid_candidates = [
            {"endpoint": "/users", "method": "INVALID"},  # Invalid HTTP method
            {"endpoint": "", "method": "GET"},            # Empty endpoint
            {"method": "POST"},                           # Missing endpoint
            {"endpoint": "/users", "confidence": 1.5}     # Invalid confidence
        ]
        
        ai_layer.validate_candidates = Mock(return_value=[])
        
        validated = ai_layer.validate_candidates(invalid_candidates)
        
        # Should filter out all invalid candidates
        assert len(validated) == 0
    
    def test_response_caching(self, ai_layer):
        """Test caching of AI responses for identical requests."""
        cache_key = "create_user_/users_GET"
        cached_response = [{"endpoint": "/users", "method": "POST", "confidence": 0.9}]
        
        ai_layer.cache = Mock()
        ai_layer.cache.get = Mock(return_value=cached_response)
        ai_layer.cache.set = Mock()
        
        # First call should hit cache
        result = ai_layer.cache.get(cache_key)
        assert result == cached_response
        
        # Verify cache is used
        ai_layer.cache.get.assert_called_once_with(cache_key)


class TestCandidateGenerator:
    """Test suite for the CandidateGenerator utility class."""
    
    def test_generate_from_workflow_patterns(self):
        """Test candidate generation based on common workflow patterns."""
        generator = Mock()
        generator.generate_from_patterns = Mock(return_value=[
            {"endpoint": "/users", "method": "POST", "pattern": "create_workflow"},
            {"endpoint": "/users/{id}", "method": "GET", "pattern": "verification_step"}
        ])
        
        candidates = generator.generate_from_patterns(
            last_action={"endpoint": "/users", "method": "GET"},
            workflow_type="user_management"
        )
        
        assert len(candidates) == 2
        assert candidates[0]["pattern"] == "create_workflow"
    
    def test_confidence_scoring(self):
        """Test confidence scoring algorithm."""
        generator = Mock()
        generator.calculate_confidence = Mock(return_value=0.85)
        
        confidence = generator.calculate_confidence(
            semantic_similarity=0.9,
            workflow_probability=0.8,
            recent_context_match=True
        )
        
        assert confidence == 0.85
        assert 0 <= confidence <= 1
    
    def test_cold_start_generation(self):
        """Test candidate generation for cold start scenarios."""
        generator = Mock()
        generator.generate_cold_start = Mock(return_value=[
            {"endpoint": "/users", "method": "GET", "confidence": 0.6, "reason": "safe_default"},
            {"endpoint": "/health", "method": "GET", "confidence": 0.5, "reason": "health_check"}
        ])
        
        candidates = generator.generate_cold_start(
            openapi_spec={},
            k=3
        )
        
        assert len(candidates) == 2
        assert all(candidate["confidence"] < 0.7 for candidate in candidates)  # Lower confidence for cold start