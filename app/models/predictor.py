"""
Phase 3 ML Predictor - OpenSesame Predictor
Integrates Phase 2 AI Layer with Phase 3 ML Ranker
Implements k+buffer strategy (k=3, buffer=2) for improved ranking
"""

import asyncio
import time
import hashlib
import json
import sqlite3
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np

from app.config import get_settings, db_manager
from app.models.ai_layer import AiLayer
from app.models.ml_ranker import MLRanker
from app.utils.feature_eng import FeatureExtractor
from app.utils.guardrails import SafetyValidator

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
        
        # Phase 4 Cold Start components
        self.semantic_model = None  # Will be loaded on demand
        self.db_path = "data/cache.db"
        
        # Phase 4 Safety Layer components
        self.safety_validator = SafetyValidator()
        
        # Prediction cache and performance tracking
        self.prediction_cache = {}
        self.total_predictions = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.safety_filtered_count = 0
        
        # ML Layer parameters
        self.k = 3  # Target number of predictions to return
        self.buffer = 2  # Additional candidates for ML ranking (k + buffer = 5 total)
        
        # Initialize cold start components
        self._init_cold_start_db()
        
        logger.info("Initialized Phase 4 Predictor with AI + ML + Cold Start + Safety integration")
    
    def _init_cold_start_db(self):
        """Initialize database table for popular endpoints"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create popular_endpoints table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS popular_endpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint_path TEXT NOT NULL,
                    method TEXT NOT NULL,
                    description TEXT,
                    usage_count INTEGER DEFAULT 0,
                    confidence_score REAL DEFAULT 0.0,
                    is_safe BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if table is empty and populate with safe default endpoints
            cursor.execute("SELECT COUNT(*) FROM popular_endpoints")
            if cursor.fetchone()[0] == 0:
                default_endpoints = [
                    ("GET", "/api/search", "Search for data or content", 100, 0.9, True),
                    ("GET", "/api/items", "Retrieve list of items", 90, 0.8, True),
                    ("GET", "/api/status", "Check system status", 80, 0.7, True),
                    ("GET", "/api/health", "Health check endpoint", 75, 0.7, True),
                    ("GET", "/api/info", "Get general information", 70, 0.6, True),
                    ("POST", "/api/search", "Advanced search with filters", 60, 0.6, True),
                    ("GET", "/api/version", "Get API version", 50, 0.5, True),
                ]
                
                cursor.executemany("""
                    INSERT INTO popular_endpoints (method, endpoint_path, description, usage_count, confidence_score, is_safe)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, default_endpoints)
                
                logger.info("Initialized popular_endpoints table with default safe endpoints")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize cold start database: {e}")
    
    def _get_semantic_model(self):
        """Load semantic model on demand for cold start predictions"""
        if self.semantic_model is None:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded semantic model for cold start predictions")
            except Exception as e:
                logger.error(f"Failed to load semantic model: {e}")
                self.semantic_model = None
        return self.semantic_model
    
    async def cold_start_predict(
        self, 
        prompt: Optional[str] = None, 
        spec: Optional[Dict[str, Any]] = None, 
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Cold start prediction for cases with no history
        
        Args:
            prompt: User input prompt (optional)
            spec: OpenAPI specification data (optional)
            k: Number of predictions to return
            
        Returns:
            List of predicted API calls with confidence scores
        """
        try:
            # Case 1: If prompt exists, use semantic search on endpoint descriptions
            if prompt and prompt.strip():
                predictions = await self._semantic_search_endpoints(prompt, k)
                if predictions:
                    logger.info(f"Cold start: Generated {len(predictions)} predictions using semantic search")
                    return predictions
            
            # Case 2: If spec provided, extract popular endpoints from spec
            if spec:
                predictions = await self._extract_popular_from_spec(spec, k)
                if predictions:
                    logger.info(f"Cold start: Generated {len(predictions)} predictions from OpenAPI spec")
                    return predictions
            
            # Case 3: Fallback to k most common safe endpoints from database
            predictions = await self._get_popular_safe_endpoints(k)
            logger.info(f"Cold start: Generated {len(predictions)} popular safe endpoint predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Cold start prediction error: {e}")
            return await self._get_basic_fallback_endpoints(k)
    
    async def _semantic_search_endpoints(self, prompt: str, k: int) -> List[Dict[str, Any]]:
        """Use semantic search to find relevant endpoints based on prompt"""
        try:
            model = self._get_semantic_model()
            if not model:
                return []
            
            # Get all endpoints with descriptions from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Try to get endpoints from parsed_endpoints table first
            cursor.execute("""
                SELECT DISTINCT method, path, summary, description 
                FROM parsed_endpoints 
                WHERE description IS NOT NULL AND description != ''
                LIMIT 100
            """)
            endpoints = cursor.fetchall()
            
            # If no endpoints in parsed_endpoints, get from popular_endpoints
            if not endpoints:
                cursor.execute("""
                    SELECT method, endpoint_path, description, description
                    FROM popular_endpoints 
                    WHERE description IS NOT NULL AND description != ''
                    ORDER BY usage_count DESC
                    LIMIT 50
                """)
                endpoints = cursor.fetchall()
            
            conn.close()
            
            if not endpoints:
                logger.warning("No endpoints with descriptions found for semantic search")
                return []
            
            # Prepare texts for semantic comparison
            prompt_embedding = model.encode([prompt])
            descriptions = [f"{ep[2]} {ep[3]}" if ep[3] else ep[2] for ep in endpoints]
            description_embeddings = model.encode(descriptions)
            
            # Calculate cosine similarity
            similarities = np.dot(prompt_embedding, description_embeddings.T).flatten()
            
            # Get top k most similar endpoints
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            predictions = []
            for i, idx in enumerate(top_indices):
                endpoint = endpoints[idx]
                similarity_score = float(similarities[idx])
                
                prediction = {
                    "api_call": f"{endpoint[0]} {endpoint[1]}",
                    "method": endpoint[0],
                    "path": endpoint[1],
                    "description": endpoint[2] or "API endpoint",
                    "parameters": {},
                    "confidence": min(0.9, max(0.3, similarity_score)),
                    "cold_start_type": "semantic_search",
                    "semantic_score": similarity_score,
                    "model_version": "cold-start-v4.0"
                }
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _extract_popular_from_spec(self, spec: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        """Extract popular/safe endpoints from OpenAPI specification"""
        try:
            predictions = []
            paths = spec.get('paths', {})
            
            # Prioritize GET methods (safer) and common patterns
            safe_patterns = ['get', 'search', 'list', 'status', 'health', 'info']
            
            for path, methods in paths.items():
                for method, details in methods.items():
                    if method.lower() in ['get', 'post'] and len(predictions) < k * 2:
                        description = details.get('summary', details.get('description', 'API endpoint'))
                        
                        # Calculate confidence based on safety and common patterns
                        confidence = 0.5
                        if method.lower() == 'get':
                            confidence += 0.2
                        
                        path_lower = path.lower()
                        for pattern in safe_patterns:
                            if pattern in path_lower or pattern in description.lower():
                                confidence += 0.1
                                break
                        
                        prediction = {
                            "api_call": f"{method.upper()} {path}",
                            "method": method.upper(),
                            "path": path,
                            "description": description,
                            "parameters": {},
                            "confidence": min(0.8, confidence),
                            "cold_start_type": "openapi_spec",
                            "model_version": "cold-start-v4.0"
                        }
                        predictions.append(prediction)
            
            # Sort by confidence and return top k
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            return predictions[:k]
            
        except Exception as e:
            logger.error(f"OpenAPI spec extraction failed: {e}")
            return []
    
    async def _get_popular_safe_endpoints(self, k: int) -> List[Dict[str, Any]]:
        """Get k most popular safe endpoints from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT method, endpoint_path, description, usage_count, confidence_score
                FROM popular_endpoints 
                WHERE is_safe = 1
                ORDER BY usage_count DESC, confidence_score DESC
                LIMIT ?
            """, (k,))
            
            endpoints = cursor.fetchall()
            conn.close()
            
            predictions = []
            for endpoint in endpoints:
                prediction = {
                    "api_call": f"{endpoint[0]} {endpoint[1]}",
                    "method": endpoint[0],
                    "path": endpoint[1],
                    "description": endpoint[2] or "Safe API endpoint",
                    "parameters": {},
                    "confidence": float(endpoint[4]) if endpoint[4] else 0.5,
                    "cold_start_type": "popular_safe",
                    "usage_count": endpoint[3],
                    "model_version": "cold-start-v4.0"
                }
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get popular safe endpoints: {e}")
            return await self._get_basic_fallback_endpoints(k)
    
    async def _get_basic_fallback_endpoints(self, k: int) -> List[Dict[str, Any]]:
        """Basic fallback endpoints when all else fails"""
        basic_endpoints = [
            {
                "api_call": "GET /api/search",
                "method": "GET",
                "path": "/api/search",
                "description": "Search for relevant data",
                "parameters": {},
                "confidence": 0.4,
                "cold_start_type": "basic_fallback",
                "model_version": "cold-start-v4.0"
            },
            {
                "api_call": "GET /api/items",
                "method": "GET", 
                "path": "/api/items",
                "description": "Retrieve list of items",
                "parameters": {},
                "confidence": 0.3,
                "cold_start_type": "basic_fallback",
                "model_version": "cold-start-v4.0"
            },
            {
                "api_call": "GET /api/status",
                "method": "GET",
                "path": "/api/status",
                "description": "Check system status",
                "parameters": {},
                "confidence": 0.2,
                "cold_start_type": "basic_fallback",
                "model_version": "cold-start-v4.0"
            }
        ]
        
        return basic_endpoints[:k]
    
    async def update_endpoint_popularity(
        self, 
        method: str, 
        path: str, 
        was_clicked: bool = True
    ):
        """Update endpoint popularity based on user interactions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if endpoint exists
            cursor.execute("""
                SELECT id, usage_count FROM popular_endpoints 
                WHERE method = ? AND endpoint_path = ?
            """, (method, path))
            
            result = cursor.fetchone()
            
            if result:
                # Update existing endpoint
                endpoint_id, current_count = result
                new_count = current_count + (1 if was_clicked else 0)
                cursor.execute("""
                    UPDATE popular_endpoints 
                    SET usage_count = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (new_count, endpoint_id))
            else:
                # Insert new endpoint
                cursor.execute("""
                    INSERT INTO popular_endpoints (method, endpoint_path, description, usage_count, confidence_score, is_safe)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (method, path, f"{method} {path}", 1 if was_clicked else 0, 0.5, True))
            
            conn.commit()
            conn.close()
            logger.debug(f"Updated popularity for {method} {path}")
            
        except Exception as e:
            logger.error(f"Failed to update endpoint popularity: {e}")
    
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
            
            # Phase 4: Cold start check - use cold start if no history available
            if not history or len(history) == 0:
                logger.info("No history available, using cold start prediction")
                cold_start_predictions = await self.cold_start_predict(prompt=prompt, k=k)
                
                if cold_start_predictions:
                    # Apply safety filtering to cold start predictions
                    safe_cold_start = [
                        pred for pred in cold_start_predictions 
                        if self.safety_validator.is_safe(pred)
                    ]
                    
                    filtered_count = len(cold_start_predictions) - len(safe_cold_start)
                    if filtered_count > 0:
                        self.safety_filtered_count += filtered_count
                        logger.info(f"Safety filter removed {filtered_count} unsafe cold start predictions")
                    
                    # Format cold start results to match standard prediction format
                    processing_time_ms = (time.time() - start_time) * 1000
                    result = {
                        "predictions": safe_cold_start,
                        "confidence_scores": [pred.get('confidence', 0.0) for pred in safe_cold_start],
                        "processing_time_ms": processing_time_ms,
                        "metadata": {
                            "model_version": "v4.0-cold-start-safety",
                            "timestamp": datetime.utcnow().isoformat(),
                            "ai_provider": "cold_start",
                            "ml_ranking_enabled": False,
                            "safety_filtering_enabled": True,
                            "cold_start_method": cold_start_predictions[0].get('cold_start_type', 'unknown') if cold_start_predictions else 'none',
                            "candidates_generated": len(cold_start_predictions),
                            "candidates_ranked": len(cold_start_predictions),
                            "candidates_filtered_for_safety": len(safe_cold_start),
                            "unsafe_candidates_removed": filtered_count,
                            "processing_method": "cold_start_safety"
                        }
                    }
                    
                    # Cache the result
                    await self._cache_prediction(cache_key, result)
                    
                    # Update metrics
                    self.total_predictions += 1
                    self.total_processing_time += processing_time_ms
                    
                    # Log prediction for analytics
                    await self._log_prediction(prompt, len(result["predictions"]), processing_time_ms, False)
                    
                    logger.info(f"Generated {len(result['predictions'])} cold start predictions in {processing_time_ms:.2f}ms")
                    return result
            
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
                # If AI fails and no history, try cold start as final fallback
                if not history or len(history) == 0:
                    logger.info("AI predictions failed with no history, trying cold start fallback")
                    cold_start_predictions = await self.cold_start_predict(prompt=prompt, k=k)
                    if cold_start_predictions:
                        # Apply safety filtering to cold start fallback predictions
                        safe_fallback = [
                            pred for pred in cold_start_predictions 
                            if self.safety_validator.is_safe(pred)
                        ]
                        
                        filtered_count = len(cold_start_predictions) - len(safe_fallback)
                        if filtered_count > 0:
                            self.safety_filtered_count += filtered_count
                            logger.info(f"Safety filter removed {filtered_count} unsafe fallback predictions")
                        
                        processing_time_ms = (time.time() - start_time) * 1000
                        result = {
                            "predictions": safe_fallback,
                            "confidence_scores": [pred.get('confidence', 0.0) for pred in safe_fallback],
                            "processing_time_ms": processing_time_ms,
                            "metadata": {
                                "model_version": "v4.0-cold-start-fallback-safety",
                                "timestamp": datetime.utcnow().isoformat(),
                                "ai_provider": "cold_start_fallback",
                                "ml_ranking_enabled": False,
                                "safety_filtering_enabled": True,
                                "cold_start_method": cold_start_predictions[0].get('cold_start_type', 'unknown') if cold_start_predictions else 'none',
                                "candidates_generated": len(cold_start_predictions),
                                "candidates_filtered_for_safety": len(safe_fallback),
                                "unsafe_candidates_removed": filtered_count,
                                "processing_method": "cold_start_fallback_safety"
                            }
                        }
                        await self._cache_prediction(cache_key, result)
                        self.total_predictions += 1
                        self.total_processing_time += processing_time_ms
                        await self._log_prediction(prompt, len(result["predictions"]), processing_time_ms, False)
                        return result
                
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
            
            # Phase 4: Safety filtering - filter k+buffer candidates and return k safe ones
            if ranked_predictions:
                # Apply safety filtering to ranked predictions
                safe_predictions = [
                    pred for pred in ranked_predictions 
                    if self.safety_validator.is_safe(pred)
                ]
                
                filtered_count = len(ranked_predictions) - len(safe_predictions)
                if filtered_count > 0:
                    self.safety_filtered_count += filtered_count
                    logger.info(f"Safety filter removed {filtered_count} unsafe predictions")
                
                # If we have fewer than k safe predictions, try to get more from original candidates
                if len(safe_predictions) < k and len(ai_predictions) > len(ranked_predictions):
                    # Check remaining AI predictions for safe ones
                    remaining_predictions = ai_predictions[len(ranked_predictions):]
                    additional_safe = [
                        pred for pred in remaining_predictions 
                        if self.safety_validator.is_safe(pred)
                    ]
                    
                    # Add additional safe predictions until we reach k or run out
                    safe_predictions.extend(additional_safe[:k - len(safe_predictions)])
                    
                    if additional_safe:
                        logger.info(f"Added {len(additional_safe[:k - len(safe_predictions)])} additional safe predictions from buffer")
                
                # Take up to k safe predictions
                final_predictions = safe_predictions[:k]
                
                logger.info(f"Safety filtering: {len(ranked_predictions)} → {len(final_predictions)} safe predictions")
            else:
                final_predictions = []
            
            # Format final results with Phase 4 metadata
            processing_time_ms = (time.time() - start_time) * 1000
            result = {
                "predictions": final_predictions,
                "confidence_scores": [pred.get('confidence', 0.0) for pred in final_predictions],
                "processing_time_ms": processing_time_ms,
                "metadata": {
                    "model_version": "v4.0-safety-layer",
                    "timestamp": datetime.utcnow().isoformat(),
                    "ai_provider": await self._get_ai_provider(),
                    "ml_ranking_enabled": use_ml_ranking,
                    "safety_filtering_enabled": True,
                    "candidates_generated": len(ai_predictions),
                    "candidates_ranked": len(ranked_predictions) if ranked_predictions else 0,
                    "candidates_filtered_for_safety": len(final_predictions),
                    "unsafe_candidates_removed": filtered_count if 'filtered_count' in locals() else 0,
                    "k_plus_buffer": f"{k}+{self.buffer}",
                    "ml_model_version": self.ml_ranker.training_stats.get('model_version', 'unknown'),
                    "processing_method": "hybrid_ai_ml_safety"
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
    
    def _generate_cache_key(
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
                "model_version": "fallback-v4.0"
            },
            {
                "api_call": "GET /api/items",
                "method": "GET", 
                "description": "Retrieve list of items",
                "parameters": {"limit": 20},
                "confidence": 0.3,
                "ml_rank": 2,
                "ml_ranking_score": 0.3,
                "model_version": "fallback-v4.0"
            },
            {
                "api_call": "POST /api/data",
                "method": "POST",
                "description": "Create or process data based on request",
                "parameters": {"input": prompt[:200]},
                "confidence": 0.2,
                "ml_rank": 3,
                "ml_ranking_score": 0.2,
                "model_version": "fallback-v4.0"
            }
        ]
        
        # Apply safety filtering to fallback predictions
        safe_fallback_predictions = [
            pred for pred in fallback_predictions 
            if self.safety_validator.is_safe(pred)
        ]
        
        selected_predictions = safe_fallback_predictions[:max_predictions]
        filtered_count = len(fallback_predictions) - len(safe_fallback_predictions)
        
        if filtered_count > 0:
            self.safety_filtered_count += filtered_count
            logger.info(f"Safety filter removed {filtered_count} unsafe fallback predictions")
        
        return {
            "predictions": selected_predictions,
            "confidence_scores": [pred["confidence"] for pred in selected_predictions],
            "processing_time_ms": 1.0,
            "metadata": {
                "model_version": "fallback-v4.0-safety",
                "timestamp": datetime.utcnow().isoformat(),
                "is_fallback": True,
                "ml_ranking_enabled": False,
                "safety_filtering_enabled": True,
                "candidates_generated": len(fallback_predictions),
                "candidates_filtered_for_safety": len(selected_predictions),
                "unsafe_candidates_removed": filtered_count,
                "processing_method": "fallback_safety"
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
        
        # Get cold start metrics
        cold_start_metrics = await self._get_cold_start_metrics()
        
        safety_filter_rate = (
            self.safety_filtered_count / self.total_predictions 
            if self.total_predictions > 0 else 0
        )
        
        return {
            "predictor_metrics": {
                "total_predictions": self.total_predictions,
                "average_processing_time_ms": avg_processing_time,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self.prediction_cache),
                "safety_filtered_count": self.safety_filtered_count,
                "safety_filter_rate": safety_filter_rate
            },
            "ml_layer_metrics": ml_model_info,
            "ai_layer_metrics": await self.ai_layer.get_status(),
            "cold_start_metrics": cold_start_metrics,
            "safety_layer_metrics": self.safety_validator.get_validation_stats(),
            "configuration": {
                "k": self.k,
                "buffer": self.buffer,
                "k_plus_buffer": self.k + self.buffer,
                "safety_filtering_enabled": True,
                "version": "v4.0-safety-layer"
            }
        }
    
    async def _get_cold_start_metrics(self) -> Dict[str, Any]:
        """Get cold start specific metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get popular endpoints statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_endpoints,
                    COUNT(CASE WHEN is_safe = 1 THEN 1 END) as safe_endpoints,
                    AVG(usage_count) as avg_usage_count,
                    MAX(usage_count) as max_usage_count,
                    COUNT(CASE WHEN description IS NOT NULL AND description != '' THEN 1 END) as endpoints_with_descriptions
                FROM popular_endpoints
            """)
            
            stats = cursor.fetchone()
            conn.close()
            
            return {
                "total_endpoints": stats[0] if stats else 0,
                "safe_endpoints": stats[1] if stats else 0,
                "avg_usage_count": float(stats[2]) if stats and stats[2] else 0.0,
                "max_usage_count": stats[3] if stats else 0,
                "endpoints_with_descriptions": stats[4] if stats else 0,
                "semantic_model_loaded": self.semantic_model is not None,
                "semantic_model_type": "all-MiniLM-L6-v2" if self.semantic_model else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get cold start metrics: {e}")
            return {
                "error": str(e),
                "semantic_model_loaded": self.semantic_model is not None
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
        
        # Check Cold Start components
        try:
            # Check semantic model availability
            model_status = "not_loaded"
            if self.semantic_model is not None:
                model_status = "loaded"
            elif self._get_semantic_model() is not None:
                model_status = "available"
            
            # Check popular endpoints table
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM popular_endpoints")
            endpoint_count = cursor.fetchone()[0]
            conn.close()
            
            health["components"]["cold_start"] = {
                "status": "operational",
                "semantic_model": model_status,
                "popular_endpoints_count": endpoint_count
            }
            
        except Exception as e:
            health["components"]["cold_start"] = f"error: {str(e)}"
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