import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from app.config import get_settings

logger = logging.getLogger(__name__)

class LLMInterface:
    """
    Large Language Model interface for generating API call predictions
    This class handles integration with various LLM providers (OpenAI, Anthropic, etc.)
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.provider = self.settings.llm_provider
        self.model = self.settings.llm_model
        self.max_tokens = self.settings.llm_max_tokens
        self.temperature = self.settings.llm_temperature
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens_used = 0
        self.total_response_time = 0.0
        
        logger.info(f"Initialized LLM interface with provider: {self.provider}, model: {self.model}")
    
    async def generate_api_predictions(
        self,
        prompt: str,
        history: List[Dict[str, Any]] = None,
        max_predictions: int = 5,
        temperature: float = None
    ) -> List[Dict[str, Any]]:
        """
        Generate API call predictions using LLM based on user prompt and conversation history
        
        Args:
            prompt: User input describing desired API functionality
            history: Previous conversation/API call history
            max_predictions: Maximum number of API calls to predict
            temperature: Randomness factor (overrides default if provided)
            
        Returns:
            List of predicted API calls with metadata
        """
        
        try:
            # Use provided temperature or default
            use_temperature = temperature if temperature is not None else self.temperature
            
            # Construct system prompt for API prediction
            system_prompt = self._build_system_prompt()
            
            # Build user prompt with context
            user_prompt = self._build_user_prompt(prompt, history, max_predictions)
            
            # Generate predictions using the configured LLM provider
            predictions = await self._call_llm_provider(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=use_temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse and validate predictions
            parsed_predictions = self._parse_predictions(predictions)
            
            logger.info(f"Generated {len(parsed_predictions)} API predictions using {self.provider}")
            return parsed_predictions
            
        except Exception as e:
            logger.error(f"LLM API prediction error: {str(e)}")
            return await self._get_fallback_predictions(prompt, max_predictions)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for API prediction task"""
        return """You are an expert API architect and developer. Your task is to predict the most likely API calls that would be needed to fulfill a user's request based on their natural language prompt.

For each API call prediction, provide:
1. HTTP method (GET, POST, PUT, DELETE, etc.)
2. Endpoint path (e.g., /api/users/{id})
3. Brief description of what the endpoint does
4. Expected parameters (query, path, or body)
5. Expected response format
6. Confidence level (0.0 to 1.0)

Consider:
- REST API best practices
- Common API patterns and conventions
- Resource hierarchy and relationships
- Authentication and authorization needs
- Error handling requirements

Return predictions as valid JSON array format."""

    def _build_user_prompt(
        self, 
        prompt: str, 
        history: List[Dict[str, Any]], 
        max_predictions: int
    ) -> str:
        """Build user prompt with context and examples"""
        
        context_parts = [
            f"User Request: {prompt}",
            f"Max Predictions: {max_predictions}"
        ]
        
        # Add conversation history if available
        if history:
            history_text = self._format_history(history)
            context_parts.append(f"Previous Context:\n{history_text}")
        
        # Add examples for better predictions
        context_parts.append(self._get_prediction_examples())
        
        context_parts.append("Based on the above request and context, predict the most likely API calls needed:")
        
        return "\n\n".join(context_parts)
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for context"""
        formatted_history = []
        
        for i, item in enumerate(history[-5:]):  # Use last 5 items to avoid token limits
            if isinstance(item, dict):
                formatted_history.append(f"{i+1}. {json.dumps(item, indent=2)}")
            else:
                formatted_history.append(f"{i+1}. {str(item)}")
        
        return "\n".join(formatted_history)
    
    def _get_prediction_examples(self) -> str:
        """Provide examples to guide LLM predictions"""
        return """Example API Predictions Format:
[
    {
        "api_call": "GET /api/users",
        "method": "GET",
        "description": "Retrieve list of users",
        "parameters": {
            "limit": 10,
            "offset": 0,
            "sort": "created_at"
        },
        "response_format": "Array of user objects",
        "confidence": 0.85
    },
    {
        "api_call": "POST /api/users/{id}/preferences",
        "method": "POST", 
        "description": "Update user preferences",
        "parameters": {
            "path": {"id": "user_id"},
            "body": {"theme": "dark", "notifications": true}
        },
        "response_format": "Updated preferences object",
        "confidence": 0.72
    }
]"""
    
    async def _call_llm_provider(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Call the configured LLM provider to generate predictions
        This is a placeholder implementation - in production this would integrate
        with actual LLM APIs like OpenAI, Anthropic, etc.
        """
        
        start_time = asyncio.get_event_loop().time()
        
        # PLACEHOLDER IMPLEMENTATION
        # In production, this would make actual API calls to LLM providers
        await asyncio.sleep(0.1)  # Simulate API call delay
        
        # Generate mock predictions based on prompt analysis
        mock_predictions = self._generate_mock_predictions(user_prompt)
        
        # Update metrics
        end_time = asyncio.get_event_loop().time()
        self.total_requests += 1
        self.total_response_time += (end_time - start_time)
        self.total_tokens_used += len(user_prompt.split()) + len(mock_predictions.split())
        
        logger.debug(f"LLM call completed in {(end_time - start_time)*1000:.2f}ms")
        
        return mock_predictions
    
    def _generate_mock_predictions(self, user_prompt: str) -> str:
        """Generate mock API predictions for placeholder implementation"""
        
        # Simple keyword-based mock predictions
        predictions = []
        
        if "user" in user_prompt.lower():
            predictions.append({
                "api_call": "GET /api/users",
                "method": "GET",
                "description": "Retrieve user information",
                "parameters": {"include": "profile,preferences"},
                "response_format": "User object with profile data",
                "confidence": 0.8
            })
        
        if "search" in user_prompt.lower() or "find" in user_prompt.lower():
            predictions.append({
                "api_call": "POST /api/search",
                "method": "POST", 
                "description": "Search functionality",
                "parameters": {"query": "search_term", "filters": {}},
                "response_format": "Array of search results",
                "confidence": 0.75
            })
        
        if "create" in user_prompt.lower() or "add" in user_prompt.lower():
            predictions.append({
                "api_call": "POST /api/resources",
                "method": "POST",
                "description": "Create new resource",
                "parameters": {"data": "resource_object"},
                "response_format": "Created resource with ID",
                "confidence": 0.7
            })
        
        # Default predictions if no keywords match
        if not predictions:
            predictions = [
                {
                    "api_call": "GET /api/data",
                    "method": "GET",
                    "description": "Generic data retrieval",
                    "parameters": {"limit": 10},
                    "response_format": "Array of data objects",
                    "confidence": 0.5
                }
            ]
        
        return json.dumps(predictions, indent=2)
    
    def _parse_predictions(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse and validate LLM response into structured predictions"""
        try:
            # Attempt to parse JSON response
            parsed = json.loads(llm_response)
            
            if not isinstance(parsed, list):
                logger.warning("LLM response is not a list, wrapping in array")
                parsed = [parsed] if isinstance(parsed, dict) else []
            
            # Validate and clean predictions
            validated_predictions = []
            for pred in parsed:
                if self._validate_prediction(pred):
                    validated_predictions.append(pred)
                else:
                    logger.warning(f"Invalid prediction format: {pred}")
            
            return validated_predictions
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Prediction parsing error: {str(e)}")
            return []
    
    def _validate_prediction(self, prediction: Dict[str, Any]) -> bool:
        """Validate that a prediction has required fields"""
        required_fields = ["api_call", "method", "description"]
        
        for field in required_fields:
            if field not in prediction:
                return False
        
        # Validate HTTP method
        valid_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        if prediction["method"].upper() not in valid_methods:
            return False
        
        return True
    
    async def _get_fallback_predictions(
        self, 
        prompt: str, 
        max_predictions: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback predictions when LLM call fails"""
        return [
            {
                "api_call": "GET /api/fallback",
                "method": "GET",
                "description": "Fallback API endpoint",
                "parameters": {},
                "response_format": "Generic response",
                "confidence": 0.1,
                "is_fallback": True
            }
        ][:max_predictions]
    
    async def get_status(self) -> Dict[str, Any]:
        """Get LLM interface status and metrics"""
        avg_response_time = (
            self.total_response_time / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            "provider": self.provider,
            "model": self.model,
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "average_response_time_s": avg_response_time,
            "status": "operational"  # In production, this would check actual API availability
        }