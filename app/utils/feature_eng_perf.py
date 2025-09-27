"""
Phase 5 Performance Optimizations for Feature Engineering
Additional methods for the FeatureExtractor class to support caching and pre-computed features
"""

import sqlite3
import pickle
import hashlib
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class FeatureExtractorPerformanceMixin:
    """Performance optimization methods for FeatureExtractor"""
    
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
                cursor.execute('SELECT pattern_hash FROM precomputed_features WHERE pattern_hash = ?', (pattern_hash,))
                if not cursor.fetchone():
                    # Compute feature vector
                    feature_vector = self._compute_workflow_vector(pattern['sequence'])
                    
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
            logger.error(f"Pre-computed features initialization error: {str(e)}")
    
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
                if np.linalg.norm(current_vector) > 0 and np.linalg.norm(pattern_vector) > 0:
                    similarity = np.dot(current_vector, pattern_vector) / (
                        np.linalg.norm(current_vector) * np.linalg.norm(pattern_vector)
                    )
                    distance = 1.0 - similarity
                    min_distance = min(min_distance, distance)
            
            return min_distance
            
        except Exception as e:
            logger.warning(f"Pre-computed pattern matching error: {e}")
            return 1.0
    
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