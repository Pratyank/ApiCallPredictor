"""
Feature Engineering for Phase 3 ML Layer - OpenSesame Predictor
Implements specific ML features for LightGBM ranking model:
- last_endpoint_type, last_resource, time_since_last, session_length
- endpoint_type, resource_match, workflow_distance, prompt_similarity
- action_verb_match, bigram_prob, trigram_prob
Stores features in data/cache.db using sqlite3
"""

import re
import json
import asyncio
import logging
import hashlib
import pickle
import sqlite3
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import numpy as np
from difflib import SequenceMatcher

from app.config import get_settings
from app.utils.db_manager import db_manager

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Phase 3 ML Feature extractor implementing specific features for LightGBM ranking model
    """

    def __init__(self):
        self.settings = get_settings()

        # Cache database path
        self.cache_db_path = os.path.join(os.getcwd(), 'data', 'cache.db')
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)

        # Initialize sentence transformer for prompt_similarity with caching
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformers: {e}")
            self.sentence_model = None

        # N-gram language models for bigram/trigram probabilities
        self.bigram_model = defaultdict(Counter)
        self.trigram_model = defaultdict(Counter)
        self._build_ngram_models()

        # Common SaaS resources and actions
        self.resources = {
            'users',
            'items',
            'documents',
            'products',
            'orders',
            'files',
            'invoices',
            'projects',
            'tasks',
            'comments',
            'settings',
            'auth'}
        self.action_verbs = {
            'get',
            'post',
            'put',
            'delete',
            'patch',
            'create',
            'update',
            'retrieve',
            'fetch',
            'save',
            'remove',
            'edit',
            'view',
            'browse',
            'search',
            'filter',
            'sort',
            'export',
            'import',
            'upload',
            'download'}

        # Pre-computed workflow pattern vectors for performance
        self._init_precomputed_features()

        logger.info("Initialized Phase 5 Optimized ML Feature Extractor")

    def _init_precomputed_features(self):
        """Initialize pre-computed feature vectors for common SaaS workflow patterns"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            # Create table for pre-computed feature vectors
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS precomputed_features (
                    pattern_hash TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    feature_vector BLOB NOT NULL,
                    pattern_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
            ''')

            conn.commit()

            # Pre-compute workflow pattern features
            workflow_patterns = [
                {'type': 'auth_flow', 'sequence': ['POST /auth/login', 'GET /user/profile', 'GET /dashboard']},
                {'type': 'crud_flow', 'sequence': ['GET /items', 'POST /items', 'PUT /items/{id}', 'DELETE /items/{id}']},
                {'type': 'search_flow', 'sequence': ['GET /search', 'GET /items', 'GET /items/{id}']},
                {'type': 'file_flow', 'sequence': ['POST /files/upload', 'GET /files', 'GET /files/{id}']},
                {'type': 'user_mgmt', 'sequence': ['GET /users', 'POST /users', 'PUT /users/{id}']},
            ]

            for pattern in workflow_patterns:
                pattern_hash = hashlib.md5(str(pattern).encode()).hexdigest()

                # Check if already exists
                cursor.execute(
                    'SELECT pattern_hash FROM precomputed_features WHERE pattern_hash = ?',
                    (pattern_hash,
                     ))
                if not cursor.fetchone():
                    # Compute feature vector
                    feature_vector = self._compute_workflow_vector(
                        pattern['sequence'])

                    cursor.execute('''
                        INSERT INTO precomputed_features
                        (pattern_hash, pattern_type, feature_vector, pattern_data)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        pattern_hash,
                        pattern['type'],
                        pickle.dumps(feature_vector),
                        json.dumps(pattern)
                    ))

            conn.commit()
            conn.close()

            logger.debug("Initialized pre-computed SaaS workflow features")

        except Exception as e:
            logger.error(
                f"Pre-computed features initialization error: {str(e)}")

    def _compute_workflow_vector(self, sequence: List[str]) -> np.ndarray:
        """Compute feature vector for a workflow sequence"""
        # Extract resource transitions, method patterns, etc.
        resources = []
        methods = []

        for api_call in sequence:
            parts = api_call.split(' ', 1)
            if len(parts) == 2:
                method, path = parts
                methods.append(method.upper())

                # Extract resource from path
                path_parts = path.strip('/').split('/')
                for part in path_parts:
                    if part in self.resources:
                        resources.append(part)
                        break

        # Create feature vector (simplified)
        vector = np.zeros(20)  # 20-dimensional vector

        # Method distribution
        for i, method in enumerate(['GET', 'POST', 'PUT', 'DELETE', 'PATCH']):
            vector[i] = methods.count(method) / len(methods) if methods else 0

        # Resource diversity
        vector[5] = len(set(resources)) / len(resources) if resources else 0

        # Sequence length
        vector[6] = len(sequence) / 10.0  # Normalized

        # Add more features as needed...

        return vector

    async def extract_ml_features(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        candidate_api: Dict[str, Any] = None,
        session_start_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Extract ML features for LightGBM ranking model

        Args:
            prompt: User input prompt
            history: Previous conversation/API call history
            candidate_api: Candidate API call being evaluated
            session_start_time: When the session started

        Returns:
            Dictionary with ML features for ranking
        """

        try:
            features = {}
            current_time = datetime.now()

            # 1. last_endpoint_type: categorical (GET, POST, PUT)
            features['last_endpoint_type'] = self._get_last_endpoint_type(
                history)

            # 2. last_resource: categorical (invoices, users, etc.)
            features['last_resource'] = self._get_last_resource(history)

            # 3. time_since_last: numeric (seconds since last API call)
            features['time_since_last'] = self._get_time_since_last(
                history, current_time)

            # 4. session_length: numeric (total session duration in minutes)
            features['session_length'] = self._get_session_length(
                session_start_time, current_time)

            # 5. endpoint_type: categorical (current candidate's method)
            features['endpoint_type'] = self._get_endpoint_type(candidate_api)

            # 6. resource_match: boolean (does candidate match recent
            # resources)
            features['resource_match'] = self._get_resource_match(
                candidate_api, history)

            # 7. workflow_distance: numeric (similarity to typical workflows)
            features['workflow_distance'] = await self._get_workflow_distance(candidate_api, history)

            # 8. prompt_similarity: numeric (semantic similarity using
            # sentence-transformers)
            features['prompt_similarity'] = await self._get_prompt_similarity(prompt, candidate_api)

            # 9. action_verb_match: boolean (does prompt contain action verbs)
            features['action_verb_match'] = self._get_action_verb_match(
                prompt, candidate_api)

            # 10. bigram_prob: numeric (bigram probability from language model)
            features['bigram_prob'] = self._get_bigram_probability(prompt)

            # 11. trigram_prob: numeric (trigram probability from language
            # model)
            features['trigram_prob'] = self._get_trigram_probability(prompt)

            # Additional metadata
            features['extraction_timestamp'] = current_time.isoformat()
            features['prompt_hash'] = hashlib.md5(prompt.encode()).hexdigest()

            return features

        except Exception as e:
            logger.error(f"ML feature extraction error: {str(e)}")
            return self._get_fallback_features()

    def _get_last_endpoint_type(
            self, history: Optional[List[Dict[str, Any]]]) -> str:
        """Extract the HTTP method of the last API call"""
        if not history:
            return 'NONE'

        for item in reversed(history):
            if isinstance(item, dict) and 'method' in item:
                method = item['method'].upper()
                if method in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                    return method

        return 'UNKNOWN'

    def _get_last_resource(
            self, history: Optional[List[Dict[str, Any]]]) -> str:
        """Extract the resource type from the last API call"""
        if not history:
            return 'none'

        for item in reversed(history):
            if isinstance(item, dict) and 'api_call' in item:
                api_call = item['api_call'].lower()

                # Extract resource from API path
                for resource in self.resources:
                    if resource in api_call:
                        return resource

                # Try to extract from path segments
                path_parts = api_call.strip('/').split('/')
                for part in path_parts:
                    if part in self.resources:
                        return part
                    # Check for plural forms
                    singular = part.rstrip('s')
                    if singular in self.resources:
                        return singular

        return 'unknown'

    def _get_time_since_last(
            self, history: Optional[List[Dict[str, Any]]], current_time: datetime) -> float:
        """Calculate seconds since last API call"""
        if not history:
            return -1.0

        for item in reversed(history):
            if isinstance(item, dict) and 'timestamp' in item:
                try:
                    last_time = datetime.fromisoformat(
                        item['timestamp'].replace('Z', '+00:00'))
                    time_diff = (
                        current_time -
                        last_time.replace(
                            tzinfo=None)).total_seconds()
                    return max(0.0, time_diff)
                except (ValueError, TypeError):
                    continue

        return -1.0

    def _get_session_length(
            self,
            session_start: Optional[datetime],
            current_time: datetime) -> float:
        """Calculate session length in minutes"""
        if not session_start:
            return 0.0

        try:
            duration = (current_time - session_start).total_seconds() / 60.0
            return max(0.0, duration)
        except (TypeError, ValueError):
            return 0.0

    def _get_endpoint_type(
            self, candidate_api: Optional[Dict[str, Any]]) -> str:
        """Get the HTTP method of the candidate API"""
        if not candidate_api or 'method' not in candidate_api:
            return 'UNKNOWN'

        method = candidate_api['method'].upper()
        return method if method in ['GET', 'POST',
                                    'PUT', 'DELETE', 'PATCH'] else 'UNKNOWN'

    def _get_resource_match(self,
                            candidate_api: Optional[Dict[str,
                                                         Any]],
                            history: Optional[List[Dict[str,
                                                        Any]]]) -> int:
        """Check if candidate API matches recent resources (boolean as int)"""
        if not candidate_api or not history:
            return 0

        candidate_resource = self._extract_resource_from_api(
            candidate_api.get('api_call', ''))
        recent_resources = set()

        # Check last 5 API calls for resource patterns
        for item in list(reversed(history))[:5]:
            if isinstance(item, dict) and 'api_call' in item:
                resource = self._extract_resource_from_api(item['api_call'])
                if resource != 'unknown':
                    recent_resources.add(resource)

        return 1 if candidate_resource in recent_resources else 0

    def _extract_resource_from_api(self, api_call: str) -> str:
        """Extract resource type from API call path"""
        api_call = api_call.lower()

        for resource in self.resources:
            if resource in api_call:
                return resource

        # Try path segments
        path_parts = api_call.strip('/').split('/')
        for part in path_parts:
            if part in self.resources:
                return part
            singular = part.rstrip('s')
            if singular in self.resources:
                return singular

        return 'unknown'

    async def _get_workflow_distance(self,
                                     candidate_api: Optional[Dict[str,
                                                                  Any]],
                                     history: Optional[List[Dict[str,
                                                                 Any]]]) -> float:
        """Calculate workflow distance using pre-computed SaaS patterns for performance"""
        if not candidate_api:
            return 1.0

        try:
            candidate_method = candidate_api.get('method', '').upper()
            candidate_resource = self._extract_resource_from_api(
                candidate_api.get('api_call', ''))

            # Fast lookup for common patterns
            common_patterns = {
                ('auth', 'POST'): 0.1,
                ('users', 'GET'): 0.2,
                ('items', 'GET'): 0.15,
                ('items', 'POST'): 0.25,
                ('items', 'PUT'): 0.3,
                ('files', 'POST'): 0.35,
            }

            # Get recent context with caching
            if not history:
                return 0.5

            # Build current sequence for pattern matching
            current_sequence = []
            for item in list(reversed(history))[:5]:  # Last 5 items
                if isinstance(
                        item,
                        dict) and 'api_call' in item and 'method' in item:
                    api_call = f"{item['method'].upper()} {item['api_call']}"
                    current_sequence.append(api_call)

            # Add candidate to sequence
            candidate_call = f"{candidate_method} {
                candidate_api.get(
                    'api_call', '')}"
            current_sequence.append(candidate_call)

            # Check against pre-computed workflow patterns
            min_distance = await self._match_precomputed_patterns(current_sequence)

            # Fallback to quick pattern matching
            if min_distance == 1.0:
                for context_item in [(self._extract_resource_from_api(item.get('api_call', '')), item.get(
                        'method', '').upper()) for item in history[-3:] if isinstance(item, dict)]:
                    if context_item in common_patterns:
                        candidate_pair = (candidate_resource, candidate_method)
                        if candidate_pair in common_patterns:
                            pattern_distance = abs(
                                common_patterns[context_item] -
                                common_patterns[candidate_pair])
                            min_distance = min(min_distance, pattern_distance)

            return min_distance

        except Exception as e:
            logger.warning(f"Workflow distance calculation error: {e}")
            return 0.5

    async def _match_precomputed_patterns(self, sequence: List[str]) -> float:
        """Match current sequence against pre-computed workflow patterns"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            # Get all pre-computed patterns
            cursor.execute('''
                SELECT pattern_type, feature_vector, pattern_data
                FROM precomputed_features
            ''')

            patterns = cursor.fetchall()
            conn.close()

            if not patterns:
                return 1.0

            # Compute current sequence vector
            current_vector = self._compute_workflow_vector(sequence)

            min_distance = 1.0
            for pattern_type, feature_blob, pattern_data in patterns:
                pattern_vector = pickle.loads(feature_blob)

                # Calculate cosine distance
                if np.linalg.norm(current_vector) > 0 and np.linalg.norm(
                        pattern_vector) > 0:
                    similarity = np.dot( current_vector, pattern_vector) / (
                        np.linalg.norm(current_vector) * np.linalg.norm(pattern_vector))
                    distance = 1.0 - similarity
                    min_distance = min(min_distance, distance)

            return min_distance

        except Exception as e:
            logger.warning(f"Pre-computed pattern matching error: {e}")
            return 1.0

    async def _get_prompt_similarity(
            self, prompt: str, candidate_api: Optional[Dict[str, Any]]) -> float:
        """Calculate semantic similarity with embedding caching for performance"""
        if not self.sentence_model or not candidate_api:
            return 0.0

        try:
            # Create text representations
            api_text = self._api_to_text(candidate_api)

            # Check cache for embeddings
            prompt_embedding = await self._get_cached_embedding(prompt)
            api_embedding = await self._get_cached_embedding(api_text)

            # Compute missing embeddings
            if prompt_embedding is None:
                prompt_embedding = self.sentence_model.encode([prompt])[0]
                await self._cache_embedding(prompt, prompt_embedding)

            if api_embedding is None:
                api_embedding = self.sentence_model.encode([api_text])[0]
                await self._cache_embedding(api_text, api_embedding)

            # Calculate cosine similarity
            similarity = np.dot(prompt_embedding, api_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(api_embedding)
            )

            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logger.warning(f"Prompt similarity calculation failed: {e}")
            return self._fallback_text_similarity(prompt, candidate_api)

    async def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding or return None"""
        try:
            text_hash = hashlib.md5(text.encode()).hexdigest()

            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT embedding_data FROM embedding_cache
                WHERE text_hash = ? AND model_name = ?
            ''', (text_hash, 'all-MiniLM-L6-v2'))

            result = cursor.fetchone()
            conn.close()

            if result:
                return pickle.loads(result[0])

            return None

        except Exception as e:
            logger.warning(f"Embedding cache retrieval error: {str(e)}")
            return None

    async def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding for future use"""
        try:
            text_hash = hashlib.md5(text.encode()).hexdigest()

            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO embedding_cache
                (text_hash, embedding_data, model_name, access_count)
                VALUES (?, ?, ?, 1)
            ''', (text_hash, pickle.dumps(embedding), 'all-MiniLM-L6-v2'))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Embedding cache storage error: {str(e)}")

    def _api_to_text(self, api: Dict[str, Any]) -> str:
        """Convert API call to natural language text"""
        method = api.get('method', '').lower()
        api_call = api.get('api_call', '')
        description = api.get('description', '')

        # Extract action and resource
        action_map = {
            'get': 'retrieve',
            'post': 'create',
            'put': 'update',
            'delete': 'remove',
            'patch': 'modify'
        }

        action = action_map.get(method, method)
        resource = self._extract_resource_from_api(api_call)

        if description:
            return f"{action} {resource} {description}"
        else:
            return f"{action} {resource} from {api_call}"

    def _fallback_text_similarity(
            self, prompt: str, candidate_api: Optional[Dict[str, Any]]) -> float:
        """Fallback text similarity when sentence-transformers unavailable"""
        if not candidate_api:
            return 0.0

        api_text = self._api_to_text(candidate_api).lower()
        prompt_lower = prompt.lower()

        # Simple word overlap similarity
        prompt_words = set(prompt_lower.split())
        api_words = set(api_text.split())

        if not prompt_words or not api_words:
            return 0.0

        intersection = len(prompt_words.intersection(api_words))
        union = len(prompt_words.union(api_words))

        return intersection / union if union > 0 else 0.0

    def _get_action_verb_match(
            self, prompt: str, candidate_api: Optional[Dict[str, Any]]) -> int:
        """Check if prompt contains action verbs matching the API method (boolean as int)"""
        if not candidate_api:
            return 0

        prompt_lower = prompt.lower()
        method = candidate_api.get('method', '').lower()

        # Map HTTP methods to action verbs
        method_verbs = {
            'get': [
                'get', 'retrieve', 'fetch', 'show', 'view', 'browse', 'list', 'find'], 'post': [
                'create', 'add', 'new', 'insert', 'post', 'submit', 'make'], 'put': [
                'update', 'modify', 'change', 'edit', 'replace', 'put'], 'delete': [
                    'delete', 'remove', 'destroy', 'erase', 'drop'], 'patch': [
                        'update', 'modify', 'patch', 'change', 'edit']}

        verbs = method_verbs.get(method, [])

        for verb in verbs:
            if verb in prompt_lower:
                return 1

        return 0

    def _get_bigram_probability(self, prompt: str) -> float:
        """Calculate bigram probability from language model"""
        words = prompt.lower().split()
        if len(words) < 2:
            return 0.0

        total_prob = 0.0
        count = 0

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]

            if w1 in self.bigram_model:
                total_occurrences = sum(self.bigram_model[w1].values())
                w2_count = self.bigram_model[w1][w2]
                prob = w2_count / total_occurrences if total_occurrences > 0 else 0.0
                total_prob += prob
                count += 1

        return total_prob / count if count > 0 else 0.0

    def _get_trigram_probability(self, prompt: str) -> float:
        """Calculate trigram probability from language model"""
        words = prompt.lower().split()
        if len(words) < 3:
            return 0.0

        total_prob = 0.0
        count = 0

        for i in range(len(words) - 2):
            w1, w2, w3 = words[i], words[i + 1], words[i + 2]
            bigram_key = (w1, w2)

            if bigram_key in self.trigram_model:
                total_occurrences = sum(
                    self.trigram_model[bigram_key].values())
                w3_count = self.trigram_model[bigram_key][w3]
                prob = w3_count / total_occurrences if total_occurrences > 0 else 0.0
                total_prob += prob
                count += 1

        return total_prob / count if count > 0 else 0.0

    def _build_ngram_models(self):
        """Build n-gram language models from common SaaS prompts"""
        # Common SaaS prompts for training n-gram models
        training_prompts = [
            "get user information from the database",
            "create new user account with email",
            "update user profile settings",
            "delete user from the system",
            "retrieve list of all items",
            "search for products by category",
            "add new product to inventory",
            "modify existing order details",
            "remove item from shopping cart",
            "browse available documents",
            "upload file to storage system",
            "download document from server",
            "share document with other users",
            "save changes to draft",
            "confirm transaction payment",
            "cancel pending order",
            "view order history details",
            "export data to csv file",
            "import users from spreadsheet",
            "authenticate user credentials",
            "logout from current session",
            "reset password for account",
            "verify email address",
            "send notification message",
            "schedule automated task",
            "generate monthly report",
            "analyze sales performance",
            "filter results by date",
            "sort items by priority",
            "search items by keyword",
            "update settings configuration"
        ]

        # Build bigram model
        for prompt in training_prompts:
            words = prompt.lower().split()
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                self.bigram_model[w1][w2] += 1

        # Build trigram model
        for prompt in training_prompts:
            words = prompt.lower().split()
            for i in range(len(words) - 2):
                w1, w2, w3 = words[i], words[i + 1], words[i + 2]
                bigram_key = (w1, w2)
                self.trigram_model[bigram_key][w3] += 1

    def _get_fallback_features(self) -> Dict[str, Any]:
        """Return fallback features when extraction fails"""
        return {
            'last_endpoint_type': 'UNKNOWN',
            'last_resource': 'unknown',
            'time_since_last': -1.0,
            'session_length': 0.0,
            'endpoint_type': 'UNKNOWN',
            'resource_match': 0,
            'workflow_distance': 1.0,
            'prompt_similarity': 0.0,
            'action_verb_match': 0,
            'bigram_prob': 0.0,
            'trigram_prob': 0.0,
            'extraction_error': True
        }

    async def store_features(self, request_id: str,
                             features: Dict[str, Any]) -> bool:
        """Store extracted features in database"""
        return await db_manager.store_features(request_id, features)

    async def batch_extract_features(
        self,
        training_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract features for multiple training samples"""
        results = []

        for i, sample in enumerate(training_data):
            try:
                # Create candidate API from sample
                candidate_api = {
                    'api_call': sample.get('api_call', ''),
                    'method': sample.get('method', ''),
                    'description': sample.get('description', '')
                }

                # Extract features
                features = await self.extract_ml_features(
                    prompt=sample.get('prompt', ''),
                    history=[],  # Empty history for training data
                    candidate_api=candidate_api
                )

                # Add sample metadata
                features.update({
                    'sequence_id': sample.get('sequence_id'),
                    'is_positive': sample.get('is_positive', False),
                    'rank': sample.get('rank', 0)
                })

                results.append(features)

                if (i + 1) % 1000 == 0:
                    logger.info(
                        f"Extracted features for {i + 1}/{len(training_data)} samples")

            except Exception as e:
                logger.error(f"Feature extraction failed for sample {i}: {e}")
                results.append(self._get_fallback_features())

        logger.info(f"Completed feature extraction for {len(results)} samples")
        return results


# Convenience functions
async def extract_features_for_sample(
    prompt: str,
    candidate_api: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Extract ML features for a single sample"""
    extractor = FeatureExtractor()
    return await extractor.extract_ml_features(prompt, history, candidate_api)


async def extract_features_for_training(
        training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract features for training data batch"""
    extractor = FeatureExtractor()
    return await extractor.batch_extract_features(training_data)
