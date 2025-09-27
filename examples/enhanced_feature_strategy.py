#!/usr/bin/env python3
"""
Enhanced Feature Engineering Strategy Implementation
Optimized for SaaS workflow patterns with pre-computed vectors and intelligent caching.
"""

import asyncio
import aiosqlite
import numpy as np
import pickle
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, LRU
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class WorkflowPatternOptimizer:
    """
    Optimized workflow pattern recognition using pre-computed transition matrices
    """
    
    def __init__(self):
        # Pre-computed workflow transition matrices based on SaaS patterns
        self.transition_matrices = {
            'browse_edit_save': np.array([
                [0.1, 0.8, 0.1],  # Browse → [Browse, Edit, Save]
                [0.2, 0.3, 0.5],  # Edit → [Browse, Edit, Save]  
                [0.7, 0.2, 0.1]   # Save → [Browse, Edit, Save]
            ]),
            'browse_confirm_save': np.array([
                [0.1, 0.7, 0.2],  # Browse → [Browse, Confirm, Save]
                [0.3, 0.2, 0.5],  # Confirm → [Browse, Confirm, Save]
                [0.8, 0.1, 0.1]   # Save → [Browse, Confirm, Save]
            ]),
            'cold_start_create': np.array([
                [0.0, 0.9, 0.1],  # None → [Browse, Create, Save]
                [0.3, 0.4, 0.3],  # Create → [Browse, Create, Save]
                [0.6, 0.2, 0.2]   # Save → [Browse, Create, Save]
            ])
        }
        
        # State mappings for O(1) lookup
        self.method_to_state = {
            'GET': 0,      # Browse
            'POST': 1,     # Edit/Create/Confirm
            'PUT': 1,      # Edit/Update
            'DELETE': 2,   # Save/Remove
            'PATCH': 1     # Edit/Modify
        }
        
        # Resource continuity patterns
        self.resource_relationships = {
            'customers': ['subscriptions', 'invoices', 'payment_methods'],
            'subscriptions': ['invoices', 'plans', 'customers'],
            'invoices': ['customers', 'payments', 'refunds'],
            'repos': ['pulls', 'issues', 'commits', 'branches'],
            'pulls': ['reviews', 'comments', 'files'],
            'issues': ['comments', 'labels', 'milestones']
        }
        
    def calculate_workflow_probability(
        self, 
        history: Optional[List[Dict[str, Any]]], 
        candidate: Dict[str, Any]
    ) -> Tuple[float, str]:
        """
        Calculate workflow probability using pre-computed transition matrices
        Returns: (probability, detected_pattern)
        """
        if not history:
            pattern = 'cold_start_create'
            candidate_state = self._get_state(candidate)
            return self.transition_matrices[pattern][0][candidate_state], pattern
        
        # Detect workflow pattern from history
        pattern = self._detect_workflow_pattern(history)
        
        # Get current and candidate states
        current_state = self._get_state(history[-1])
        candidate_state = self._get_state(candidate)
        
        # O(1) matrix lookup
        probability = self.transition_matrices[pattern][current_state][candidate_state]
        
        # Adjust probability based on resource continuity
        continuity_bonus = self._calculate_resource_continuity(history, candidate)
        adjusted_probability = min(1.0, probability + continuity_bonus)
        
        return adjusted_probability, pattern
    
    def _get_state(self, api_call: Dict[str, Any]) -> int:
        """Map API call to workflow state"""
        method = api_call.get('method', 'GET').upper()
        return self.method_to_state.get(method, 0)
    
    def _detect_workflow_pattern(self, history: List[Dict[str, Any]]) -> str:
        """Detect workflow pattern from API call history"""
        if len(history) < 2:
            return 'browse_edit_save'
        
        # Analyze method patterns in recent history
        recent_methods = [call.get('method', 'GET').upper() for call in history[-3:]]
        
        # Pattern detection heuristics
        if 'POST' in recent_methods and recent_methods.count('GET') >= 2:
            if any(term in str(history).lower() for term in ['confirm', 'approve', 'merge', 'pay']):
                return 'browse_confirm_save'
        
        return 'browse_edit_save'
    
    def _calculate_resource_continuity(
        self, 
        history: List[Dict[str, Any]], 
        candidate: Dict[str, Any]
    ) -> float:
        """Calculate resource continuity bonus"""
        if not history:
            return 0.0
        
        # Extract resource from candidate
        candidate_resource = self._extract_resource(candidate.get('api_call', ''))
        
        # Check recent history for related resources
        for call in reversed(history[-3:]):  # Check last 3 calls
            history_resource = self._extract_resource(call.get('api_call', ''))
            
            if history_resource == candidate_resource:
                return 0.2  # Same resource bonus
            
            if (history_resource in self.resource_relationships and 
                candidate_resource in self.resource_relationships[history_resource]):
                return 0.1  # Related resource bonus
        
        return 0.0
    
    def _extract_resource(self, api_call: str) -> str:
        """Extract resource type from API call path"""
        # Simple resource extraction - could be enhanced with regex
        path_parts = api_call.lower().strip('/').split('/')
        
        for part in path_parts:
            if part in self.resource_relationships:
                return part
            # Check for plural forms
            singular = part.rstrip('s')
            if singular in self.resource_relationships:
                return singular
        
        return 'unknown'

class FeatureVectorCache:
    """
    High-performance caching system for feature vectors and embeddings
    """
    
    def __init__(self, db_path: str, cache_size: int = 10000):
        self.db_path = db_path
        self.cache_size = cache_size
        
        # In-memory LRU caches
        self.feature_cache = {}  # LRU cache for feature vectors
        self.embedding_cache = {}  # LRU cache for embeddings
        
        # Performance counters
        self.cache_hits = 0
        self.cache_misses = 0
        
    async def initialize_database(self):
        """Initialize SQLite schema for caching"""
        async with aiosqlite.connect(self.db_path) as db:
            # Feature vector cache table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS feature_vector_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT NOT NULL,
                    candidate_hash TEXT NOT NULL,
                    feature_vector BLOB NOT NULL,
                    workflow_pattern TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Embedding cache table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_hash TEXT UNIQUE NOT NULL,
                    embedding BLOB NOT NULL,
                    model_version TEXT DEFAULT 'all-MiniLM-L6-v2',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            # Create indexes for performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_prompt_candidate ON feature_vector_cache(prompt_hash, candidate_hash)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON feature_vector_cache(last_accessed)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_text_hash ON embedding_cache(text_hash)")
            
            await db.commit()
    
    async def get_cached_features(
        self, 
        prompt_hash: str, 
        candidate_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached feature vector"""
        cache_key = f"{prompt_hash}_{candidate_hash}"
        
        # Check in-memory cache first
        if cache_key in self.feature_cache:
            self.cache_hits += 1
            return self.feature_cache[cache_key]
        
        # Check database cache
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT feature_vector, workflow_pattern, confidence_score 
                FROM feature_vector_cache 
                WHERE prompt_hash = ? AND candidate_hash = ?
                ORDER BY last_accessed DESC
                LIMIT 1
            """, (prompt_hash, candidate_hash))
            
            row = await cursor.fetchone()
            
            if row:
                # Update access statistics
                await db.execute("""
                    UPDATE feature_vector_cache 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE prompt_hash = ? AND candidate_hash = ?
                """, (prompt_hash, candidate_hash))
                await db.commit()
                
                # Deserialize and cache in memory
                features = pickle.loads(row[0])
                features['workflow_pattern'] = row[1]
                features['confidence_score'] = row[2]
                
                self._add_to_memory_cache(cache_key, features)
                self.cache_hits += 1
                return features
        
        self.cache_misses += 1
        return None
    
    async def store_features(
        self, 
        prompt_hash: str, 
        candidate_hash: str, 
        features: Dict[str, Any]
    ):
        """Store feature vector in cache"""
        cache_key = f"{prompt_hash}_{candidate_hash}"
        
        # Store in memory cache
        self._add_to_memory_cache(cache_key, features)
        
        # Store in database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO feature_vector_cache 
                (prompt_hash, candidate_hash, feature_vector, workflow_pattern, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            """, (
                prompt_hash, 
                candidate_hash, 
                pickle.dumps(features),
                features.get('workflow_pattern'),
                features.get('confidence_score', 0.0)
            ))
            await db.commit()
    
    async def get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding"""
        # Check in-memory cache
        if text_hash in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[text_hash]
        
        # Check database
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT embedding FROM embedding_cache WHERE text_hash = ?
            """, (text_hash,))
            
            row = await cursor.fetchone()
            
            if row:
                # Update access count
                await db.execute("""
                    UPDATE embedding_cache 
                    SET access_count = access_count + 1
                    WHERE text_hash = ?
                """, (text_hash,))
                await db.commit()
                
                embedding = pickle.loads(row[0])
                self.embedding_cache[text_hash] = embedding
                self.cache_hits += 1
                return embedding
        
        self.cache_misses += 1
        return None
    
    async def store_embedding(self, text_hash: str, embedding: np.ndarray):
        """Store embedding in cache"""
        # Store in memory
        self.embedding_cache[text_hash] = embedding
        
        # Store in database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO embedding_cache (text_hash, embedding)
                VALUES (?, ?)
            """, (text_hash, pickle.dumps(embedding)))
            await db.commit()
    
    def _add_to_memory_cache(self, key: str, value: Any):
        """Add item to memory cache with LRU eviction"""
        if len(self.feature_cache) >= self.cache_size:
            # Simple LRU: remove oldest item (in practice, use proper LRU)
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
        
        self.feature_cache[key] = value
    
    async def cleanup_old_entries(self, days_old: int = 30):
        """Clean up old cache entries"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                DELETE FROM feature_vector_cache 
                WHERE last_accessed < ? AND access_count < 5
            """, (cutoff_date.isoformat(),))
            
            await db.execute("""
                DELETE FROM embedding_cache 
                WHERE created_at < ? AND access_count < 10
            """, (cutoff_date.isoformat(),))
            
            await db.commit()

class BatchFeatureProcessor:
    """
    Batch processing for efficient feature computation and embedding generation
    """
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.sentence_model = None
        
    def _get_sentence_model(self):
        """Lazy load sentence transformer model"""
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.sentence_model
    
    async def batch_compute_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Compute embeddings in batches for efficiency"""
        model = self._get_sentence_model()
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = model.encode(batch)
            all_embeddings.extend(embeddings)
            
            # Allow other coroutines to run
            await asyncio.sleep(0)
        
        return all_embeddings
    
    async def precompute_resource_embeddings(self, cache: FeatureVectorCache):
        """Pre-compute embeddings for all known resources"""
        resources = [
            'users', 'customers', 'items', 'products', 'invoices', 'payments',
            'subscriptions', 'plans', 'orders', 'files', 'documents',
            'projects', 'tasks', 'comments', 'settings', 'auth',
            'repos', 'pulls', 'issues', 'commits', 'branches', 'reviews'
        ]
        
        # Create rich text descriptions for better embeddings
        resource_texts = [
            f"manage {resource} data and perform operations on {resource} in SaaS application"
            for resource in resources
        ]
        
        logger.info(f"Pre-computing embeddings for {len(resources)} resources")
        embeddings = await self.batch_compute_embeddings(resource_texts)
        
        # Store in cache
        for resource, embedding in zip(resources, embeddings):
            text_hash = hashlib.md5(resource.encode()).hexdigest()
            await cache.store_embedding(text_hash, embedding)
        
        logger.info("Completed pre-computing resource embeddings")

class EnhancedFeatureExtractor:
    """
    Enhanced feature extractor with optimized caching and workflow pattern recognition
    """
    
    def __init__(self, db_path: str = "data/cache.db"):
        self.db_path = db_path
        
        # Initialize components
        self.workflow_optimizer = WorkflowPatternOptimizer()
        self.cache = FeatureVectorCache(db_path)
        self.batch_processor = BatchFeatureProcessor()
        
        # Performance metrics
        self.extraction_times = []
        self.cache_hit_rate = 0.0
        
    async def initialize(self):
        """Initialize the enhanced feature extractor"""
        await self.cache.initialize_database()
        await self.batch_processor.precompute_resource_embeddings(self.cache)
    
    async def extract_ml_features(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        candidate_api: Dict[str, Any] = None,
        session_start_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Extract ML features with enhanced caching and optimization
        """
        start_time = time.time()
        
        try:
            # Generate cache keys
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            candidate_hash = hashlib.md5(
                f"{candidate_api.get('method', '')}_{candidate_api.get('api_call', '')}".encode()
            ).hexdigest()
            
            # Check cache first
            cached_features = await self.cache.get_cached_features(prompt_hash, candidate_hash)
            if cached_features:
                extraction_time = (time.time() - start_time) * 1000
                self.extraction_times.append(extraction_time)
                return cached_features
            
            # Compute features
            features = await self._compute_fresh_features(
                prompt, history, candidate_api, session_start_time
            )
            
            # Store in cache
            await self.cache.store_features(prompt_hash, candidate_hash, features)
            
            # Update metrics
            extraction_time = (time.time() - start_time) * 1000
            self.extraction_times.append(extraction_time)
            
            return features
            
        except Exception as e:
            logger.error(f"Enhanced feature extraction error: {e}")
            return self._get_fallback_features()
    
    async def _compute_fresh_features(
        self,
        prompt: str,
        history: Optional[List[Dict[str, Any]]],
        candidate_api: Dict[str, Any],
        session_start_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Compute fresh feature vector"""
        current_time = datetime.now()
        features = {}
        
        # Enhanced workflow distance using transition matrices
        workflow_prob, workflow_pattern = self.workflow_optimizer.calculate_workflow_probability(
            history, candidate_api
        )
        features['workflow_distance'] = 1.0 - workflow_prob
        features['workflow_pattern'] = workflow_pattern
        features['workflow_probability'] = workflow_prob
        
        # Basic features (optimized versions of original features)
        features['last_endpoint_type'] = self._get_last_endpoint_type(history)
        features['last_resource'] = self._get_last_resource(history)
        features['time_since_last'] = self._get_time_since_last(history, current_time)
        features['session_length'] = self._get_session_length(session_start_time, current_time)
        features['endpoint_type'] = candidate_api.get('method', 'UNKNOWN').upper()
        features['resource_match'] = self._get_resource_match(candidate_api, history)
        
        # Enhanced semantic similarity with caching
        features['prompt_similarity'] = await self._get_cached_prompt_similarity(prompt, candidate_api)
        
        # Enhanced action verb matching
        features['action_verb_match'] = self._get_enhanced_action_verb_match(prompt, candidate_api)
        
        # N-gram probabilities (simplified for demo)
        features['bigram_prob'] = self._get_simple_ngram_prob(prompt, 2)
        features['trigram_prob'] = self._get_simple_ngram_prob(prompt, 3)
        
        # Additional metadata
        features['extraction_timestamp'] = current_time.isoformat()
        features['prompt_hash'] = hashlib.md5(prompt.encode()).hexdigest()
        features['confidence_score'] = self._calculate_confidence_score(features)
        
        return features
    
    async def _get_cached_prompt_similarity(
        self, 
        prompt: str, 
        candidate_api: Dict[str, Any]
    ) -> float:
        """Get prompt similarity with embedding caching"""
        try:
            # Check for cached embeddings
            prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
            api_text = self._api_to_text(candidate_api)
            api_hash = hashlib.md5(api_text.encode()).hexdigest()
            
            prompt_embedding = await self.cache.get_cached_embedding(prompt_hash)
            api_embedding = await self.cache.get_cached_embedding(api_hash)
            
            # Compute missing embeddings
            texts_to_compute = []
            hashes_to_store = []
            
            if prompt_embedding is None:
                texts_to_compute.append(prompt)
                hashes_to_store.append(prompt_hash)
            
            if api_embedding is None:
                texts_to_compute.append(api_text)
                hashes_to_store.append(api_hash)
            
            if texts_to_compute:
                new_embeddings = await self.batch_processor.batch_compute_embeddings(texts_to_compute)
                
                for hash_val, embedding in zip(hashes_to_store, new_embeddings):
                    await self.cache.store_embedding(hash_val, embedding)
                    
                    if hash_val == prompt_hash:
                        prompt_embedding = embedding
                    else:
                        api_embedding = embedding
            
            # Calculate cosine similarity
            if prompt_embedding is not None and api_embedding is not None:
                similarity = np.dot(prompt_embedding, api_embedding) / (
                    np.linalg.norm(prompt_embedding) * np.linalg.norm(api_embedding)
                )
                return float(max(0.0, min(1.0, similarity)))
            
        except Exception as e:
            logger.warning(f"Cached similarity calculation failed: {e}")
        
        return 0.0
    
    def _get_enhanced_action_verb_match(self, prompt: str, candidate_api: Dict[str, Any]) -> int:
        """Enhanced action verb matching with more comprehensive mapping"""
        prompt_lower = prompt.lower()
        method = candidate_api.get('method', '').lower()
        
        # Enhanced verb mappings
        enhanced_method_verbs = {
            'get': ['get', 'retrieve', 'fetch', 'show', 'view', 'browse', 'list', 'find', 'search', 'explore'],
            'post': ['create', 'add', 'new', 'insert', 'post', 'submit', 'make', 'build', 'generate'],
            'put': ['update', 'modify', 'change', 'edit', 'replace', 'put', 'set', 'configure'],
            'delete': ['delete', 'remove', 'destroy', 'erase', 'drop', 'clear', 'cancel'],
            'patch': ['update', 'modify', 'patch', 'change', 'edit', 'adjust', 'fix']
        }
        
        verbs = enhanced_method_verbs.get(method, [])
        
        # Check for verb matches
        for verb in verbs:
            if verb in prompt_lower:
                return 1
        
        # Check for contextual matches
        if method == 'post' and any(word in prompt_lower for word in ['issue', 'problem', 'bug']):
            return 1
        
        return 0
    
    def _calculate_confidence_score(self, features: Dict[str, Any]) -> float:
        """Calculate overall confidence score for feature vector"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on feature quality
        if features.get('workflow_probability', 0) > 0.7:
            confidence += 0.2
        
        if features.get('prompt_similarity', 0) > 0.8:
            confidence += 0.2
        
        if features.get('resource_match', 0) == 1:
            confidence += 0.1
        
        if features.get('action_verb_match', 0) == 1:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    # Simplified implementations of basic feature extraction methods
    def _get_last_endpoint_type(self, history: Optional[List[Dict[str, Any]]]) -> str:
        """Extract the HTTP method of the last API call"""
        if not history:
            return 'NONE'
        
        for item in reversed(history):
            if isinstance(item, dict) and 'method' in item:
                return item['method'].upper()
        
        return 'UNKNOWN'
    
    def _get_last_resource(self, history: Optional[List[Dict[str, Any]]]) -> str:
        """Extract the resource type from the last API call"""
        if not history:
            return 'none'
        
        for item in reversed(history):
            if isinstance(item, dict) and 'api_call' in item:
                return self.workflow_optimizer._extract_resource(item['api_call'])
        
        return 'unknown'
    
    def _get_time_since_last(self, history: Optional[List[Dict[str, Any]]], current_time: datetime) -> float:
        """Calculate seconds since last API call"""
        if not history:
            return -1.0
        
        for item in reversed(history):
            if isinstance(item, dict) and 'timestamp' in item:
                try:
                    last_time = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                    return (current_time - last_time.replace(tzinfo=None)).total_seconds()
                except (ValueError, TypeError):
                    continue
        
        return -1.0
    
    def _get_session_length(self, session_start: Optional[datetime], current_time: datetime) -> float:
        """Calculate session length in minutes"""
        if not session_start:
            return 0.0
        
        try:
            return (current_time - session_start).total_seconds() / 60.0
        except (TypeError, ValueError):
            return 0.0
    
    def _get_resource_match(self, candidate_api: Optional[Dict[str, Any]], history: Optional[List[Dict[str, Any]]]) -> int:
        """Check if candidate API matches recent resources"""
        if not candidate_api or not history:
            return 0
        
        candidate_resource = self.workflow_optimizer._extract_resource(candidate_api.get('api_call', ''))
        
        for item in reversed(history[-5:]):  # Check last 5 calls
            if isinstance(item, dict) and 'api_call' in item:
                resource = self.workflow_optimizer._extract_resource(item['api_call'])
                if resource == candidate_resource:
                    return 1
        
        return 0
    
    def _api_to_text(self, api: Dict[str, Any]) -> str:
        """Convert API call to natural language text"""
        method = api.get('method', '').lower()
        api_call = api.get('api_call', '')
        description = api.get('description', '')
        
        action_map = {'get': 'retrieve', 'post': 'create', 'put': 'update', 'delete': 'remove', 'patch': 'modify'}
        action = action_map.get(method, method)
        resource = self.workflow_optimizer._extract_resource(api_call)
        
        if description:
            return f"{action} {resource} {description}"
        else:
            return f"{action} {resource} from {api_call}"
    
    def _get_simple_ngram_prob(self, prompt: str, n: int) -> float:
        """Simplified n-gram probability calculation"""
        words = prompt.lower().split()
        if len(words) < n:
            return 0.0
        
        # Simple frequency-based probability (in practice, use trained models)
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        return 1.0 / (len(ngrams) + 1)  # Simplified scoring
    
    def _get_fallback_features(self) -> Dict[str, Any]:
        """Return fallback features when extraction fails"""
        return {
            'workflow_distance': 1.0,
            'workflow_pattern': 'unknown',
            'workflow_probability': 0.0,
            'last_endpoint_type': 'UNKNOWN',
            'last_resource': 'unknown',
            'time_since_last': -1.0,
            'session_length': 0.0,
            'endpoint_type': 'UNKNOWN',
            'resource_match': 0,
            'prompt_similarity': 0.0,
            'action_verb_match': 0,
            'bigram_prob': 0.0,
            'trigram_prob': 0.0,
            'confidence_score': 0.1,
            'extraction_error': True
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the enhanced extractor"""
        avg_extraction_time = (
            sum(self.extraction_times) / len(self.extraction_times)
            if self.extraction_times else 0
        )
        
        cache_hit_rate = (
            self.cache.cache_hits / (self.cache.cache_hits + self.cache.cache_misses)
            if (self.cache.cache_hits + self.cache.cache_misses) > 0 else 0
        )
        
        return {
            'avg_extraction_time_ms': avg_extraction_time,
            'cache_hit_rate': cache_hit_rate,
            'cache_hits': self.cache.cache_hits,
            'cache_misses': self.cache.cache_misses,
            'total_extractions': len(self.extraction_times),
            'in_memory_cache_size': len(self.cache.feature_cache),
            'embedding_cache_size': len(self.cache.embedding_cache)
        }

# Example usage
async def example_usage():
    """Example of using the enhanced feature extractor"""
    
    # Initialize
    extractor = EnhancedFeatureExtractor()
    await extractor.initialize()
    
    # Example scenario
    prompt = "create invoice for customer subscription"
    history = [
        {
            "api_call": "GET /v1/customers/cus_123",
            "method": "GET",
            "timestamp": "2024-09-27T10:15:30Z"
        },
        {
            "api_call": "GET /v1/subscriptions?customer=cus_123",
            "method": "GET",
            "timestamp": "2024-09-27T10:16:45Z"
        }
    ]
    candidate_api = {
        "api_call": "POST /v1/invoices",
        "method": "POST",
        "description": "Create invoice for customer"
    }
    
    # Extract features
    features = await extractor.extract_ml_features(prompt, history, candidate_api)
    
    print("Enhanced Feature Vector:")
    print(json.dumps(features, indent=2, default=str))
    
    # Get performance metrics
    metrics = await extractor.get_performance_metrics()
    print("\nPerformance Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    asyncio.run(example_usage())