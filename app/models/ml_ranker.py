import numpy as np
import logging
from typing import List, Dict, Any, Optional
import json
import asyncio
from datetime import datetime

from app.config import get_settings

logger = logging.getLogger(__name__)

class MLRanker:
    """
    Machine Learning-based ranking system for scoring and ordering API predictions
    This class implements various ML algorithms to rank API calls based on relevance and confidence
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_weights = self._initialize_feature_weights()
        self.ranking_models = self._initialize_ranking_models()
        
        # Performance tracking
        self.total_rankings = 0
        self.total_ranking_time = 0.0
        
        logger.info("Initialized ML Ranker with feature-based scoring")
    
    async def rank_predictions(
        self,
        predictions: List[Dict[str, Any]],
        features: Dict[str, Any],
        user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Rank and score API predictions using ML algorithms
        
        Args:
            predictions: List of API call predictions from LLM
            features: Extracted features from user prompt and history
            user_context: Additional context about the user and request
            
        Returns:
            Dictionary with ranked predictions and confidence scores
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not predictions:
                logger.warning("No predictions to rank")
                return {"predictions": [], "confidence_scores": []}
            
            # Calculate relevance scores for each prediction
            scored_predictions = []
            
            for prediction in predictions:
                score = await self._calculate_prediction_score(
                    prediction=prediction,
                    features=features,
                    user_context=user_context
                )
                
                scored_predictions.append({
                    "prediction": prediction,
                    "score": score,
                    "ranking_factors": self._get_ranking_factors(prediction, features)
                })
            
            # Sort predictions by score (descending)
            scored_predictions.sort(key=lambda x: x["score"], reverse=True)
            
            # Extract ranked predictions and scores
            ranked_predictions = [item["prediction"] for item in scored_predictions]
            confidence_scores = [item["score"] for item in scored_predictions]
            
            # Add ranking metadata
            for i, pred in enumerate(ranked_predictions):
                pred["rank"] = i + 1
                pred["ml_confidence"] = confidence_scores[i]
                pred["ranking_factors"] = scored_predictions[i]["ranking_factors"]
            
            # Update performance metrics
            end_time = asyncio.get_event_loop().time()
            self.total_rankings += 1
            self.total_ranking_time += (end_time - start_time)
            
            logger.info(f"Ranked {len(predictions)} predictions in {(end_time - start_time)*1000:.2f}ms")
            
            return {
                "predictions": ranked_predictions,
                "confidence_scores": confidence_scores,
                "ranking_metadata": {
                    "model_version": "ml-ranker-v1.0",
                    "ranking_time_ms": (end_time - start_time) * 1000,
                    "features_used": list(features.keys()) if features else []
                }
            }
            
        except Exception as e:
            logger.error(f"ML ranking error: {str(e)}")
            # Return original predictions with basic confidence scores
            return {
                "predictions": predictions,
                "confidence_scores": [0.5] * len(predictions),
                "error": str(e)
            }
    
    async def _calculate_prediction_score(
        self,
        prediction: Dict[str, Any],
        features: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> float:
        """Calculate ML-based score for a single prediction"""
        
        try:
            # Start with base confidence from LLM
            base_score = prediction.get("confidence", 0.5)
            
            # Feature-based scoring
            feature_score = self._calculate_feature_score(prediction, features)
            
            # Context-based scoring
            context_score = self._calculate_context_score(prediction, user_context)
            
            # Semantic similarity scoring
            semantic_score = await self._calculate_semantic_score(prediction, features)
            
            # API pattern scoring
            pattern_score = self._calculate_pattern_score(prediction)
            
            # Combine scores using weighted average
            final_score = (
                base_score * self.feature_weights["base_confidence"] +
                feature_score * self.feature_weights["feature_match"] +
                context_score * self.feature_weights["context_relevance"] +
                semantic_score * self.feature_weights["semantic_similarity"] +
                pattern_score * self.feature_weights["api_patterns"]
            )
            
            # Normalize to 0-1 range
            final_score = max(0.0, min(1.0, final_score))
            
            return final_score
            
        except Exception as e:
            logger.warning(f"Score calculation error: {str(e)}")
            return prediction.get("confidence", 0.5)
    
    def _calculate_feature_score(
        self,
        prediction: Dict[str, Any],
        features: Dict[str, Any]
    ) -> float:
        """Calculate score based on extracted features"""
        
        if not features:
            return 0.5
        
        score = 0.0
        
        # Intent matching
        if "intent" in features:
            intent = features["intent"].lower()
            api_call = prediction.get("api_call", "").lower()
            
            if "create" in intent and "post" in prediction.get("method", "").lower():
                score += 0.3
            elif "get" in intent and "get" in prediction.get("method", "").lower():
                score += 0.3
            elif "search" in intent and "search" in api_call:
                score += 0.4
        
        # Entity matching
        if "entities" in features:
            entities = features["entities"]
            api_call = prediction.get("api_call", "").lower()
            
            for entity in entities:
                if entity.lower() in api_call:
                    score += 0.2
        
        # Keyword relevance
        if "keywords" in features:
            keywords = features["keywords"]
            prediction_text = json.dumps(prediction).lower()
            
            matching_keywords = sum(1 for keyword in keywords if keyword.lower() in prediction_text)
            keyword_ratio = matching_keywords / len(keywords) if keywords else 0
            score += keyword_ratio * 0.3
        
        return min(1.0, score)
    
    def _calculate_context_score(
        self,
        prediction: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> float:
        """Calculate score based on user context"""
        
        if not user_context:
            return 0.5
        
        score = 0.5
        
        # Prompt relevance
        prompt = user_context.get("prompt", "").lower()
        api_call = prediction.get("api_call", "").lower()
        
        # Simple word overlap scoring
        prompt_words = set(prompt.split())
        api_words = set(api_call.split('/'))
        
        if prompt_words and api_words:
            overlap = len(prompt_words.intersection(api_words))
            score += (overlap / len(prompt_words)) * 0.3
        
        # History relevance
        history = user_context.get("history", [])
        if history:
            # Boost score if similar API patterns were used before
            for hist_item in history[-3:]:  # Check last 3 interactions
                if isinstance(hist_item, dict) and "api_call" in hist_item:
                    if prediction.get("method") == hist_item.get("method"):
                        score += 0.1
                    if any(word in prediction.get("api_call", "") 
                          for word in hist_item.get("api_call", "").split('/')):
                        score += 0.1
        
        return min(1.0, score)
    
    async def _calculate_semantic_score(
        self,
        prediction: Dict[str, Any],
        features: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity score (placeholder for vector similarity)"""
        
        # PLACEHOLDER: In production, this would use embeddings/vectors
        # to calculate semantic similarity between user intent and API prediction
        
        await asyncio.sleep(0.001)  # Simulate embedding calculation
        
        # Mock semantic scoring based on description similarity
        description = prediction.get("description", "").lower()
        
        if features and "intent" in features:
            intent_words = features["intent"].lower().split()
            description_words = description.split()
            
            # Simple word overlap as semantic proxy
            if intent_words and description_words:
                overlap = len(set(intent_words).intersection(set(description_words)))
                return min(1.0, overlap / len(intent_words))
        
        return 0.5
    
    def _calculate_pattern_score(self, prediction: Dict[str, Any]) -> float:
        """Score based on API design patterns and best practices"""
        
        score = 0.5
        api_call = prediction.get("api_call", "")
        method = prediction.get("method", "").upper()
        
        # RESTful pattern scoring
        if "/api/" in api_call:
            score += 0.1
        
        # HTTP method appropriateness
        if method == "GET" and not any(word in api_call.lower() 
                                      for word in ["create", "update", "delete"]):
            score += 0.1
        elif method == "POST" and any(word in api_call.lower() 
                                     for word in ["create", "add", "new"]):
            score += 0.1
        
        # Resource naming conventions
        path_parts = api_call.split('/')
        if len(path_parts) > 2:  # Has proper resource hierarchy
            score += 0.1
        
        # Parameter structure
        params = prediction.get("parameters", {})
        if isinstance(params, dict) and params:
            score += 0.1
        
        return min(1.0, score)
    
    def _get_ranking_factors(
        self,
        prediction: Dict[str, Any],
        features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get detailed breakdown of ranking factors for explainability"""
        
        return {
            "base_confidence": prediction.get("confidence", 0.5),
            "method_match": self._check_method_appropriateness(prediction),
            "keyword_relevance": self._check_keyword_relevance(prediction, features),
            "api_pattern_score": self._calculate_pattern_score(prediction),
            "description_quality": len(prediction.get("description", "")) > 10
        }
    
    def _check_method_appropriateness(self, prediction: Dict[str, Any]) -> bool:
        """Check if HTTP method is appropriate for the API call"""
        method = prediction.get("method", "").upper()
        api_call = prediction.get("api_call", "").lower()
        
        if method == "GET":
            return not any(word in api_call for word in ["create", "update", "delete"])
        elif method in ["POST", "PUT", "PATCH"]:
            return any(word in api_call for word in ["create", "update", "modify", "add"])
        elif method == "DELETE":
            return "delete" in api_call or "remove" in api_call
        
        return True
    
    def _check_keyword_relevance(
        self,
        prediction: Dict[str, Any],
        features: Dict[str, Any]
    ) -> float:
        """Check keyword relevance between features and prediction"""
        
        if not features or "keywords" not in features:
            return 0.5
        
        keywords = features["keywords"]
        prediction_text = json.dumps(prediction).lower()
        
        relevant_count = sum(1 for keyword in keywords if keyword.lower() in prediction_text)
        return relevant_count / len(keywords) if keywords else 0.0
    
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Initialize weights for different ranking factors"""
        return {
            "base_confidence": 0.3,
            "feature_match": 0.25,
            "context_relevance": 0.2,
            "semantic_similarity": 0.15,
            "api_patterns": 0.1
        }
    
    def _initialize_ranking_models(self) -> Dict[str, Any]:
        """Initialize ML models for ranking (placeholder)"""
        # PLACEHOLDER: In production, this would load actual trained ML models
        return {
            "feature_ranker": "placeholder_model",
            "semantic_ranker": "placeholder_embeddings",
            "pattern_classifier": "placeholder_classifier"
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get ML ranker status and performance metrics"""
        
        avg_ranking_time = (
            self.total_ranking_time / self.total_rankings 
            if self.total_rankings > 0 else 0
        )
        
        return {
            "model_version": "ml-ranker-v1.0",
            "total_rankings": self.total_rankings,
            "average_ranking_time_s": avg_ranking_time,
            "feature_weights": self.feature_weights,
            "models_loaded": list(self.ranking_models.keys()),
            "status": "operational"
        }
    
    async def retrain_models(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retrain ML models with new data (placeholder for future implementation)"""
        
        # PLACEHOLDER: Implement actual model retraining
        logger.info(f"Retraining models with {len(training_data)} samples")
        await asyncio.sleep(0.1)  # Simulate training time
        
        return {
            "status": "completed",
            "training_samples": len(training_data),
            "model_version": "ml-ranker-v1.1",
            "performance_improvement": 0.05  # Mock improvement
        }