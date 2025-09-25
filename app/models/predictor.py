import asyncio
import time
import hashlib
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.config import get_settings, db_manager
from app.models.ai_layer import LLMInterface
from app.models.ml_ranker import MLRanker
from app.utils.feature_eng import FeatureExtractor

logger = logging.getLogger(__name__)

class PredictionEngine:
    """
    Core prediction engine that orchestrates ML models and LLM integration
    for API call prediction based on user prompts and conversation history
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.llm_interface = LLMInterface()
        self.ml_ranker = MLRanker()
        self.feature_extractor = FeatureExtractor()
        self.prediction_cache = {}
        
        # Performance metrics
        self.total_predictions = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        
    async def predict(
        self, 
        prompt: str, 
        history: List[Dict[str, Any]] = None,
        max_predictions: int = 5,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate API call predictions based on user prompt and conversation history
        
        Args:
            prompt: User input prompt describing desired API functionality
            history: List of previous interactions/API calls
            max_predictions: Maximum number of predictions to return
            temperature: Randomness factor for prediction generation
            
        Returns:
            Dictionary containing predictions, confidence scores, and metadata
        """
        start_time = time.time()
        
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(prompt, history, max_predictions, temperature)
            
            # Check cache first
            cached_result = await self._get_cached_prediction(cache_key)
            if cached_result:
                self.cache_hits += 1
                logger.info("Returning cached prediction")
                return cached_result
            
            # Extract features from prompt and history
            features = await self.feature_extractor.extract_features(prompt, history)
            
            # Generate LLM-based predictions
            llm_predictions = await self.llm_interface.generate_api_predictions(
                prompt=prompt,
                history=history,
                max_predictions=max_predictions,
                temperature=temperature
            )
            
            # Rank and score predictions using ML model
            ranked_predictions = await self.ml_ranker.rank_predictions(
                predictions=llm_predictions,
                features=features,
                user_context={"prompt": prompt, "history": history}
            )
            
            # Format final results
            processing_time_ms = (time.time() - start_time) * 1000
            result = {
                "predictions": ranked_predictions["predictions"][:max_predictions],
                "confidence_scores": ranked_predictions["confidence_scores"][:max_predictions],
                "processing_time_ms": processing_time_ms,
                "model_version": "v1.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "features_used": features.get("feature_names", []),
                "llm_provider": self.settings.llm_provider
            }
            
            # Cache the result
            await self._cache_prediction(cache_key, result)
            
            # Update metrics
            self.total_predictions += 1
            self.total_processing_time += processing_time_ms
            
            # Log to database
            await self._log_prediction(prompt, len(result["predictions"]), processing_time_ms)
            
            logger.info(f"Generated {len(result['predictions'])} predictions in {processing_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            # Return fallback predictions
            return await self._get_fallback_predictions(prompt, max_predictions)
    
    async def _generate_cache_key(
        self, 
        prompt: str, 
        history: List[Dict[str, Any]], 
        max_predictions: int, 
        temperature: float
    ) -> str:
        """Generate unique cache key for request parameters"""
        content = {
            "prompt": prompt,
            "history": history or [],
            "max_predictions": max_predictions,
            "temperature": temperature
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    async def _get_cached_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached prediction if available and not expired"""
        try:
            # Check in-memory cache first
            if cache_key in self.prediction_cache:
                cached_data = self.prediction_cache[cache_key]
                if time.time() - cached_data["timestamp"] < self.settings.cache_ttl_seconds:
                    return cached_data["result"]
                else:
                    del self.prediction_cache[cache_key]
            
            # TODO: Check SQLite cache for persistence across restarts
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval error: {str(e)}")
            return None
    
    async def _cache_prediction(self, cache_key: str, result: Dict[str, Any]):
        """Cache prediction result with TTL"""
        try:
            # Store in memory cache
            self.prediction_cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
            
            # Limit cache size
            if len(self.prediction_cache) > self.settings.cache_max_size:
                # Remove oldest entries (simple LRU approximation)
                oldest_key = min(self.prediction_cache.keys(), 
                               key=lambda k: self.prediction_cache[k]["timestamp"])
                del self.prediction_cache[oldest_key]
            
            # TODO: Store in SQLite for persistence
            
        except Exception as e:
            logger.warning(f"Cache storage error: {str(e)}")
    
    async def _get_fallback_predictions(
        self, 
        prompt: str, 
        max_predictions: int
    ) -> Dict[str, Any]:
        """Generate fallback predictions when main prediction fails"""
        
        # Simple rule-based fallback predictions
        fallback_predictions = [
            {
                "api_call": "GET /api/data",
                "description": "Generic data retrieval endpoint",
                "parameters": {"limit": 10},
                "method": "GET"
            },
            {
                "api_call": "POST /api/search", 
                "description": "Search functionality",
                "parameters": {"query": prompt[:100]},
                "method": "POST"
            },
            {
                "api_call": "GET /api/status",
                "description": "System status check",
                "parameters": {},
                "method": "GET"
            }
        ]
        
        return {
            "predictions": fallback_predictions[:max_predictions],
            "confidence_scores": [0.3] * min(len(fallback_predictions), max_predictions),
            "processing_time_ms": 1.0,
            "model_version": "fallback-v1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "is_fallback": True
        }
    
    async def _log_prediction(self, prompt: str, prediction_count: int, processing_time: float):
        """Log prediction metrics to database"""
        try:
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            # TODO: Implement database logging
            logger.debug(f"Logged prediction: {prompt_hash}, count: {prediction_count}, time: {processing_time}ms")
        except Exception as e:
            logger.warning(f"Prediction logging error: {str(e)}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get prediction engine performance metrics"""
        avg_processing_time = (
            self.total_processing_time / self.total_predictions 
            if self.total_predictions > 0 else 0
        )
        
        cache_hit_rate = (
            self.cache_hits / self.total_predictions 
            if self.total_predictions > 0 else 0
        )
        
        return {
            "total_predictions": self.total_predictions,
            "average_processing_time_ms": avg_processing_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.prediction_cache),
            "ml_model_status": await self.ml_ranker.get_status(),
            "llm_interface_status": await self.llm_interface.get_status()
        }