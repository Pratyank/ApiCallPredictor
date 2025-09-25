import re
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from collections import Counter
import hashlib
from datetime import datetime

from app.config import get_settings

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Feature extraction utility for converting user prompts and conversation history
    into structured features for ML model training and prediction
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.max_history_length = self.settings.max_history_length
        self.feature_vector_size = self.settings.feature_vector_size
        
        # Feature extraction patterns
        self.http_methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
        self.intent_keywords = self._load_intent_keywords()
        self.api_patterns = self._load_api_patterns()
        
        # Performance tracking
        self.total_extractions = 0
        self.total_extraction_time = 0.0
        
        logger.info("Initialized Feature Extractor for ML model input preparation")
    
    async def extract_features(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive features from user prompt and conversation history
        
        Args:
            prompt: User input prompt
            history: Previous conversation/API call history
            additional_context: Additional contextual information
            
        Returns:
            Dictionary containing extracted features for ML processing
        """
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Initialize feature dictionary
            features = {
                "timestamp": datetime.utcnow().isoformat(),
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
                "feature_names": []
            }
            
            # Text-based features from prompt
            text_features = await self._extract_text_features(prompt)
            features.update(text_features)
            features["feature_names"].extend(text_features.keys())
            
            # Intent and semantic features
            intent_features = await self._extract_intent_features(prompt)
            features.update(intent_features)
            features["feature_names"].extend(intent_features.keys())
            
            # Historical pattern features
            if history:
                history_features = await self._extract_history_features(history)
                features.update(history_features)
                features["feature_names"].extend(history_features.keys())
            
            # Context-based features
            if additional_context:
                context_features = await self._extract_context_features(additional_context)
                features.update(context_features)
                features["feature_names"].extend(context_features.keys())
            
            # API pattern recognition features
            api_features = await self._extract_api_pattern_features(prompt, history)
            features.update(api_features)
            features["feature_names"].extend(api_features.keys())
            
            # Statistical features
            stats_features = self._extract_statistical_features(prompt, history)
            features.update(stats_features)
            features["feature_names"].extend(stats_features.keys())
            
            # Update performance metrics
            end_time = asyncio.get_event_loop().time()
            self.total_extractions += 1
            self.total_extraction_time += (end_time - start_time)
            
            features["extraction_time_ms"] = (end_time - start_time) * 1000
            features["total_features"] = len(features["feature_names"])
            
            logger.debug(f"Extracted {len(features['feature_names'])} features in {(end_time - start_time)*1000:.2f}ms")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "feature_names": []
            }
    
    async def _extract_text_features(self, prompt: str) -> Dict[str, Any]:
        """Extract basic text-based features from the prompt"""
        
        features = {}
        
        # Basic text statistics
        features["prompt_length"] = len(prompt)
        features["word_count"] = len(prompt.split())
        features["sentence_count"] = len(re.split(r'[.!?]+', prompt))
        features["character_count"] = len(prompt.replace(' ', ''))
        
        # Text complexity metrics
        words = prompt.split()
        features["avg_word_length"] = sum(len(word) for word in words) / len(words) if words else 0
        features["unique_word_ratio"] = len(set(words)) / len(words) if words else 0
        
        # Punctuation and formatting
        features["question_marks"] = prompt.count('?')
        features["exclamation_marks"] = prompt.count('!')
        features["uppercase_ratio"] = sum(1 for c in prompt if c.isupper()) / len(prompt) if prompt else 0
        
        # Technical content indicators
        features["contains_json"] = 1 if '{' in prompt and '}' in prompt else 0
        features["contains_url"] = 1 if re.search(r'https?://', prompt) else 0
        features["contains_email"] = 1 if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', prompt) else 0
        features["contains_code"] = 1 if any(word in prompt.lower() for word in ['function', 'class', 'import', 'return']) else 0
        
        return features
    
    async def _extract_intent_features(self, prompt: str) -> Dict[str, Any]:
        """Extract user intent and semantic features"""
        
        features = {}
        prompt_lower = prompt.lower()
        
        # Intent classification based on keywords
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            intent_scores[f"intent_{intent}"] = score / len(keywords) if keywords else 0
        
        features.update(intent_scores)
        
        # Determine primary intent
        primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k]) if intent_scores else "unknown"
        features["primary_intent"] = primary_intent
        features["intent_confidence"] = max(intent_scores.values()) if intent_scores.values() else 0.0
        
        # Action words detection
        action_words = ["create", "get", "update", "delete", "list", "search", "find", "add", "remove", "modify"]
        features["action_word_count"] = sum(1 for word in action_words if word in prompt_lower)
        
        # Entity extraction (simplified)
        entities = self._extract_entities(prompt)
        features["entity_count"] = len(entities)
        features["entities"] = entities[:10]  # Limit to first 10 entities
        
        # Sentiment indicators (basic)
        positive_words = ["good", "great", "excellent", "perfect", "awesome", "love"]
        negative_words = ["bad", "terrible", "awful", "hate", "wrong", "error"]
        
        features["positive_sentiment"] = sum(1 for word in positive_words if word in prompt_lower)
        features["negative_sentiment"] = sum(1 for word in negative_words if word in prompt_lower)
        
        return features
    
    async def _extract_history_features(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from conversation/API call history"""
        
        features = {}
        
        # Basic history statistics
        features["history_length"] = min(len(history), self.max_history_length)
        features["has_history"] = 1 if history else 0
        
        if not history:
            return features
        
        # Limit history to prevent feature explosion
        recent_history = history[-self.max_history_length:]
        
        # HTTP method patterns
        methods = []
        api_calls = []
        for item in recent_history:
            if isinstance(item, dict):
                if "method" in item:
                    methods.append(item["method"].upper())
                if "api_call" in item:
                    api_calls.append(item["api_call"])
        
        # Method distribution
        method_counts = Counter(methods)
        for method in self.http_methods:
            features[f"history_{method.lower()}_count"] = method_counts.get(method, 0)
        
        # API call patterns
        features["unique_api_calls"] = len(set(api_calls))
        features["api_call_repetition"] = len(api_calls) - len(set(api_calls)) if api_calls else 0
        
        # Temporal patterns
        if len(recent_history) > 1:
            features["history_recency"] = 1.0  # Recent items weight more
        
        # Success/failure patterns
        success_count = sum(1 for item in recent_history 
                          if isinstance(item, dict) and item.get("status") == "success")
        features["history_success_rate"] = success_count / len(recent_history) if recent_history else 0
        
        return features
    
    async def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from additional context"""
        
        features = {}
        
        # User context
        if "user_id" in context:
            features["has_user_id"] = 1
        
        if "session_id" in context:
            features["has_session_id"] = 1
        
        # Request metadata
        if "timestamp" in context:
            features["request_with_timestamp"] = 1
        
        if "user_agent" in context:
            features["has_user_agent"] = 1
        
        # Custom parameters
        if "parameters" in context and isinstance(context["parameters"], dict):
            features["custom_params_count"] = len(context["parameters"])
        
        return features
    
    async def _extract_api_pattern_features(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Extract API-specific pattern features"""
        
        features = {}
        prompt_lower = prompt.lower()
        
        # REST pattern recognition
        rest_indicators = ["api", "endpoint", "rest", "resource", "service"]
        features["rest_pattern_score"] = sum(1 for word in rest_indicators if word in prompt_lower)
        
        # CRUD operation detection
        crud_patterns = {
            "create": ["create", "new", "add", "insert", "post"],
            "read": ["get", "fetch", "retrieve", "read", "list", "show"],
            "update": ["update", "modify", "edit", "change", "put", "patch"],
            "delete": ["delete", "remove", "destroy", "drop"]
        }
        
        for operation, keywords in crud_patterns.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            features[f"crud_{operation}_score"] = score
        
        # Authentication patterns
        auth_keywords = ["auth", "login", "token", "key", "credential", "permission"]
        features["auth_pattern_score"] = sum(1 for word in auth_keywords if word in prompt_lower)
        
        # Data format patterns
        format_keywords = {
            "json": ["json", "javascript", "object"],
            "xml": ["xml", "soap"],
            "csv": ["csv", "comma", "separated"],
            "binary": ["binary", "file", "upload", "download"]
        }
        
        for format_type, keywords in format_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            features[f"format_{format_type}_score"] = score
        
        # URL/path patterns
        if re.search(r'/\w+(/\w+)*', prompt):
            features["contains_url_path"] = 1
        else:
            features["contains_url_path"] = 0
        
        # Parameter patterns
        if re.search(r'\{[\w]+\}', prompt):  # {id} patterns
            features["contains_path_params"] = 1
        else:
            features["contains_path_params"] = 0
        
        return features
    
    def _extract_statistical_features(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Extract statistical features for ML processing"""
        
        features = {}
        
        # Prompt complexity scores
        words = prompt.split()
        if words:
            word_lengths = [len(word) for word in words]
            features["word_length_std"] = (sum((l - sum(word_lengths)/len(word_lengths))**2 for l in word_lengths) / len(word_lengths))**0.5
            features["word_length_max"] = max(word_lengths)
            features["word_length_min"] = min(word_lengths)
        
        # Lexical diversity
        if words:
            features["lexical_diversity"] = len(set(words)) / len(words)
        
        # Character distribution
        char_counts = Counter(prompt.lower())
        features["most_common_char_freq"] = max(char_counts.values()) / len(prompt) if prompt else 0
        
        # Numerical content
        numbers = re.findall(r'\d+', prompt)
        features["number_count"] = len(numbers)
        features["avg_number_value"] = sum(int(n) for n in numbers) / len(numbers) if numbers else 0
        
        return features
    
    def _extract_entities(self, prompt: str) -> List[str]:
        """Simple entity extraction (placeholder for NER)"""
        
        # PLACEHOLDER: In production, this would use proper NER models
        
        entities = []
        
        # Simple pattern-based entity extraction
        # URLs
        urls = re.findall(r'https?://[^\s]+', prompt)
        entities.extend(urls)
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', prompt)
        entities.extend(emails)
        
        # Numbers
        numbers = re.findall(r'\b\d+\b', prompt)
        entities.extend(numbers)
        
        # Capitalized words (potential proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', prompt)
        entities.extend(proper_nouns[:5])  # Limit to first 5
        
        return list(set(entities))  # Remove duplicates
    
    def _load_intent_keywords(self) -> Dict[str, List[str]]:
        """Load keyword patterns for intent classification"""
        
        return {
            "create": ["create", "new", "add", "make", "build", "generate", "insert", "post"],
            "retrieve": ["get", "fetch", "retrieve", "find", "search", "show", "list", "read"],
            "update": ["update", "modify", "change", "edit", "alter", "put", "patch"],
            "delete": ["delete", "remove", "destroy", "drop", "clear", "erase"],
            "authentication": ["login", "auth", "authenticate", "signin", "token", "credential"],
            "search": ["search", "find", "query", "lookup", "filter", "browse"],
            "upload": ["upload", "send", "transfer", "submit", "post", "attach"],
            "download": ["download", "get", "fetch", "retrieve", "export", "save"],
            "status": ["status", "health", "check", "ping", "test", "monitor"],
            "configuration": ["config", "setting", "configure", "setup", "preference"]
        }
    
    def _load_api_patterns(self) -> Dict[str, List[str]]:
        """Load common API patterns and conventions"""
        
        return {
            "rest_resources": ["users", "posts", "comments", "orders", "products", "files"],
            "rest_actions": ["list", "create", "show", "update", "destroy"],
            "auth_endpoints": ["login", "logout", "register", "refresh", "verify"],
            "admin_endpoints": ["admin", "dashboard", "settings", "config"],
            "api_versions": ["v1", "v2", "api/v1", "api/v2"]
        }
    
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (placeholder for ML model feedback)"""
        
        # PLACEHOLDER: In production, this would be derived from trained models
        return {
            "prompt_length": 0.15,
            "word_count": 0.12,
            "intent_confidence": 0.18,
            "crud_read_score": 0.14,
            "crud_create_score": 0.13,
            "history_length": 0.10,
            "rest_pattern_score": 0.08,
            "entity_count": 0.06,
            "action_word_count": 0.04
        }
    
    async def get_extraction_stats(self) -> Dict[str, Any]:
        """Get feature extraction performance statistics"""
        
        avg_extraction_time = (
            self.total_extraction_time / self.total_extractions 
            if self.total_extractions > 0 else 0
        )
        
        return {
            "total_extractions": self.total_extractions,
            "average_extraction_time_s": avg_extraction_time,
            "max_history_length": self.max_history_length,
            "feature_vector_size": self.feature_vector_size,
            "intent_categories": len(self.intent_keywords),
            "api_pattern_categories": len(self.api_patterns)
        }

# Convenience function for quick feature extraction
async def extract_features(
    prompt: str,
    history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Convenience function for feature extraction"""
    extractor = FeatureExtractor()
    return await extractor.extract_features(prompt, history)