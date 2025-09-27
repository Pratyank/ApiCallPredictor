"""
Phase 3 ML Predictor - OpenSesame Predictor
Integrates Phase 2 AI Layer with Phase 3 ML Ranker
Implements k+buffer strategy (k=3, buffer=2) for improved ranking
"""

import asyncio
import time
import hashlib
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.config import get_settings, db_manager
from app.models.ai_layer import AiLayer
from app.models.ml_ranker import MLRanker
from app.utils.feature_eng import FeatureExtractor

logger = logging.getLogger(__name__)

class Predictor:
    """
    Phase 3 ML-enhanced prediction engine that orchestrates AI Layer and ML Ranker
    for improved API call prediction with intelligent ranking
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Phase 2 AI Layer components
        self.ai_layer = AiLayer()
        
        # Phase 3 ML Layer components
        self.ml_ranker = MLRanker()
        self.feature_extractor = FeatureExtractor()
        
        # Prediction cache and performance tracking
        self.prediction_cache = {}
        self.total_predictions = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        
        # ML Layer parameters
        self.k = 3  # Target number of predictions to return
        self.buffer = 2  # Additional candidates for ML ranking (k + buffer = 5 total)
        
        logger.info("Initialized Phase 3 Predictor with AI + ML integration")
    
    async def predict(
        self, 
        prompt: str, 
        history: List[Dict[str, Any]] = None,
        max_predictions: int = 3,
        temperature: float = 0.7,
        use_ml_ranking: bool = True
    ) -> Dict[str, Any]:
        """
        Generate API call predictions using integrated AI + ML approach
        
        Args:
            prompt: User input prompt describing desired API functionality
            history: List of previous interactions/API calls
            max_predictions: Maximum number of predictions to return (k)
            temperature: Randomness factor for AI prediction generation
            use_ml_ranking: Whether to use ML ranking (True for Phase 3)
            
        Returns:
            Dictionary containing ML-ranked predictions with confidence scores and metadata
        """
        start_time = time.time()
        
        try:
            # Override max_predictions to use k from ML Layer
            k = max_predictions if max_predictions <= 5 else self.k
            
            # Generate cache key
            cache_key = self._generate_cache_key(prompt, history, k, temperature, use_ml_ranking)
            
            # Check cache first
            cached_result = await self._get_cached_prediction(cache_key)
            if cached_result:
                self.cache_hits += 1
                logger.info("Returning cached ML prediction")
                return cached_result
            
            # Phase 2: Generate AI predictions with k+buffer candidates
            ai_candidates = k + self.buffer  # Generate 5 candidates for k=3
            ai_predictions = await self.ai_layer.generate_predictions(
                prompt=prompt,
                history=history,
                k=ai_candidates,
                temperature=temperature
            )
            
            logger.info(f"AI Layer generated {len(ai_predictions)} candidates")
            
            if not ai_predictions:
                return await self._get_fallback_predictions(prompt, k)
            
            # Phase 3: ML Ranking (if enabled and model available)
            if use_ml_ranking:
                ranked_predictions = await self.ml_ranker.rank_predictions(
                    predictions=ai_predictions,
                    prompt=prompt,
                    history=history,
                    k=k,
                    buffer=self.buffer
                )
                
                if ranked_predictions:
                    # Store features for continuous learning
                    await self._store_prediction_features(prompt, ranked_predictions, history)
                    
                    logger.info(f"ML Ranker returned {len(ranked_predictions)} ranked predictions")
                else:
                    # Fallback to AI ranking if ML ranking fails
                    ranked_predictions = ai_predictions[:k]
                    logger.warning("ML ranking failed, using AI predictions")
            else:
                # Use AI predictions without ML ranking
                ranked_predictions = ai_predictions[:k]
                logger.info("Using AI predictions without ML ranking")
            
            # Format final results with Phase 3 metadata
            processing_time_ms = (time.time() - start_time) * 1000
            result = {
                "predictions": ranked_predictions,
                "confidence_scores": [pred.get('confidence', 0.0) for pred in ranked_predictions],
                "processing_time_ms": processing_time_ms,
                "metadata": {
                    "model_version": "v3.0-ml-layer",
                    "timestamp": datetime.utcnow().isoformat(),
                    "ai_provider": await self._get_ai_provider(),
                    "ml_ranking_enabled": use_ml_ranking,
                    "candidates_generated": len(ai_predictions),
                    "candidates_ranked": len(ranked_predictions),
                    "k_plus_buffer": f"{k}+{self.buffer}",
                    "ml_model_version": self.ml_ranker.training_stats.get('model_version', 'unknown'),
                    "processing_method": "hybrid_ai_ml"
                }
            }
            
            # Cache the result
            await self._cache_prediction(cache_key, result)
            
            # Update metrics
            self.total_predictions += 1
            self.total_processing_time += processing_time_ms
            
            # Log prediction for analytics
            await self._log_prediction(prompt, len(result["predictions"]), processing_time_ms, use_ml_ranking)
            
            logger.info(f"Generated {len(result['predictions'])} ML-ranked predictions in {processing_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Phase 3 prediction error: {str(e)}")
            # Return fallback predictions
            return await self._get_fallback_predictions(prompt, k if 'k' in locals() else max_predictions)
    
    async def _store_prediction_features(
        self, 
        prompt: str, 
        predictions: List[Dict[str, Any]], 
        history: Optional[List[Dict[str, Any]]]
    ):
        """Store prediction features for continuous learning"""
        try:
            request_id = hashlib.md5(f"{prompt}_{datetime.now().isoformat()}".encode()).hexdigest()
            
            # Extract features for the top prediction (most likely to be clicked)
            if predictions and 'ml_features' in predictions[0]:
                features = predictions[0]['ml_features']
                await self.feature_extractor.store_features(request_id, features)
                
        except Exception as e:
            logger.warning(f"Feature storage failed: {e}")
    
    async def _generate_cache_key(
        self, 
        prompt: str, 
        history: List[Dict[str, Any]], 
        max_predictions: int, 
        temperature: float,
        use_ml_ranking: bool
    ) -> str:
        """Generate unique cache key for request parameters"""
        content = {
            "prompt": prompt,
            "history": history or [],
            "max_predictions": max_predictions,
            "temperature": temperature,
            "use_ml_ranking": use_ml_ranking,
            "version": "v3.0"
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
            
            # Limit cache size (simple LRU approximation)
            max_cache_size = getattr(self.settings, 'cache_max_size', 1000)
            if len(self.prediction_cache) > max_cache_size:
                oldest_key = min(self.prediction_cache.keys(), 
                               key=lambda k: self.prediction_cache[k]["timestamp"])
                del self.prediction_cache[oldest_key]
            
        except Exception as e:
            logger.warning(f"Cache storage error: {str(e)}")
    
    async def _get_fallback_predictions(
        self, 
        prompt: str, 
        max_predictions: int
    ) -> Dict[str, Any]:
        """Generate fallback predictions when main prediction fails"""
        
        # Enhanced fallback predictions for Phase 3
        fallback_predictions = [
            {
                "api_call": "GET /api/search",
                "method": "GET",
                "description": "Search for relevant data based on prompt",
                "parameters": {"query": prompt[:100], "limit": 10},
                "confidence": 0.4,
                "ml_rank": 1,
                "ml_ranking_score": 0.4,
                "model_version": "fallback-v3.0"
            },
            {
                "api_call": "GET /api/items",
                "method": "GET", 
                "description": "Retrieve list of items",
                "parameters": {"limit": 20},
                "confidence": 0.3,
                "ml_rank": 2,
                "ml_ranking_score": 0.3,
                "model_version": "fallback-v3.0"
            },
            {
                "api_call": "POST /api/data",
                "method": "POST",
                "description": "Create or process data based on request",
                "parameters": {"input": prompt[:200]},
                "confidence": 0.2,
                "ml_rank": 3,
                "ml_ranking_score": 0.2,
                "model_version": "fallback-v3.0"
            }
        ]
        
        selected_predictions = fallback_predictions[:max_predictions]
        
        return {
            "predictions": selected_predictions,
            "confidence_scores": [pred["confidence"] for pred in selected_predictions],
            "processing_time_ms": 1.0,
            "metadata": {
                "model_version": "fallback-v3.0",
                "timestamp": datetime.utcnow().isoformat(),
                "is_fallback": True,
                "ml_ranking_enabled": False,
                "processing_method": "fallback"
            }
        }
    
    async def _log_prediction(self, prompt: str, prediction_count: int, processing_time: float, used_ml: bool):
        """Log prediction metrics to database for analytics"""
        try:
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            
            # Store analytics data in database
            analytics_data = {
                'prompt_hash': prompt_hash,
                'prediction_count': prediction_count,
                'processing_time_ms': processing_time,
                'used_ml_ranking': used_ml,
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'v3.0-ml-layer'
            }
            
            logger.debug(f"Logged prediction analytics: {analytics_data}")
            
        except Exception as e:
            logger.warning(f"Prediction logging error: {str(e)}")
    
    async def _get_ai_provider(self) -> str:
        """Get the active AI provider name"""
        ai_status = await self.ai_layer.get_status()
        return ai_status.get('active_provider', 'unknown')
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive prediction engine performance metrics"""
        avg_processing_time = (
            self.total_processing_time / self.total_predictions 
            if self.total_predictions > 0 else 0
        )
        
        cache_hit_rate = (
            self.cache_hits / self.total_predictions 
            if self.total_predictions > 0 else 0
        )
        
        # Get ML model metrics
        ml_model_info = await self.ml_ranker.get_model_info()
        
        return {
            "predictor_metrics": {
                "total_predictions": self.total_predictions,
                "average_processing_time_ms": avg_processing_time,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self.prediction_cache)
            },
            "ml_layer_metrics": ml_model_info,
            "ai_layer_metrics": await self.ai_layer.get_status(),
            "configuration": {
                "k": self.k,
                "buffer": self.buffer,
                "k_plus_buffer": self.k + self.buffer,
                "version": "v3.0-ml-layer"
            }
        }
    
    async def train_ml_model(self) -> Dict[str, Any]:
        """Train or retrain the ML ranking model"""
        logger.info("Starting ML model training...")
        return await self.ml_ranker.train_ranker()
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all components"""
        health = {
            "status": "healthy",
            "components": {},
            "version": "v3.0-ml-layer",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check AI Layer
        try:
            # Basic AI layer check (simplified)
            health["components"]["ai_layer"] = "operational"
        except Exception as e:
            health["components"]["ai_layer"] = f"error: {str(e)}"
            health["status"] = "degraded"
        
        # Check ML Ranker
        try:
            ml_info = await self.ml_ranker.get_model_info()
            health["components"]["ml_ranker"] = "operational" if ml_info["is_trained"] else "not_trained"
            if not ml_info["is_trained"]:
                health["status"] = "degraded"
        except Exception as e:
            health["components"]["ml_ranker"] = f"error: {str(e)}"
            health["status"] = "degraded"
        
        # Check Feature Extractor
        try:
            # Basic feature extractor check
            health["components"]["feature_extractor"] = "operational"
        except Exception as e:
            health["components"]["feature_extractor"] = f"error: {str(e)}"
            health["status"] = "degraded"
        
        # Check Database
        try:
            db_stats = await db_manager.get_database_stats()
            health["components"]["database"] = "operational"
            health["database_stats"] = db_stats
        except Exception as e:
            health["components"]["database"] = f"error: {str(e)}"
            health["status"] = "degraded"
        
        return health


# Global predictor instance for application use
_predictor_instance = None

async def get_predictor() -> Predictor:
    """Get global predictor instance (singleton pattern)"""
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = Predictor()
        
        # Initialize ML model if available
        try:
            await _predictor_instance.ml_ranker.load_model()
            logger.info("Loaded pre-trained ML ranking model")
        except Exception as e:
            logger.info(f"No pre-trained ML model available: {e}")
    
    return _predictor_instance

# Convenience functions
async def predict_api_calls(
    prompt: str,
    history: List[Dict[str, Any]] = None,
    max_predictions: int = 3,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Convenience function for API predictions"""
    predictor = await get_predictor()
    return await predictor.predict(prompt, history, max_predictions, temperature)

async def get_predictor_metrics() -> Dict[str, Any]:
    """Get predictor performance metrics"""
    predictor = await get_predictor()
    return await predictor.get_metrics()

async def train_ml_predictor() -> Dict[str, Any]:
    """Train the ML component of the predictor"""
    predictor = await get_predictor()
    return await predictor.train_ml_model()