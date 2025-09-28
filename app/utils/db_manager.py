"""
Database manager for Phase 3 ML Layer data storage
Handles SQLite operations for synthetic sequences, features, and ML training data
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database operations for ML Layer data storage"""

    def __init__(self, db_path: str = "data/cache.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with required tables for Phase 3"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Synthetic sequences table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS synthetic_sequences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sequence_id TEXT UNIQUE NOT NULL,
                        workflow_type TEXT NOT NULL,
                        sequence_data TEXT NOT NULL,
                        sequence_length INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Features table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL,
                        last_endpoint_type TEXT,
                        last_resource TEXT,
                        time_since_last REAL,
                        session_length INTEGER,
                        endpoint_type TEXT,
                        resource_match INTEGER,
                        workflow_distance REAL,
                        prompt_similarity REAL,
                        action_verb_match INTEGER,
                        bigram_prob REAL,
                        trigram_prob REAL,
                        additional_features TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Training data table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sequence_id TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        api_call TEXT NOT NULL,
                        method TEXT NOT NULL,
                        is_positive INTEGER NOT NULL,
                        rank INTEGER,
                        features TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # ML models table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ml_models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT UNIQUE NOT NULL,
                        model_version TEXT NOT NULL,
                        model_data BLOB,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise

    async def store_synthetic_sequences(
            self, sequences: List[Dict[str, Any]]) -> int:
        """Store synthetic workflow sequences in database"""
        stored_count = 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for seq in sequences:
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO synthetic_sequences
                            (sequence_id, workflow_type, sequence_data, sequence_length)
                            VALUES (?, ?, ?, ?)
                        """, (
                            seq['sequence_id'],
                            seq['workflow_type'],
                            json.dumps(seq['sequence_data']),
                            seq['sequence_length']
                        ))
                        stored_count += 1
                    except sqlite3.IntegrityError:
                        logger.warning(
                            f"Sequence {
                                seq.get('sequence_id')} already exists")

                conn.commit()
                logger.info(f"Stored {stored_count} synthetic sequences")

        except Exception as e:
            logger.error(f"Error storing synthetic sequences: {str(e)}")
            raise

        return stored_count

    async def store_features(self, request_id: str,
                             features: Dict[str, Any]) -> bool:
        """Store extracted features for ML training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Extract specific ML features
                additional_features = {
                    k: v for k,
                    v in features.items() if k not in [
                        'last_endpoint_type',
                        'last_resource',
                        'time_since_last',
                        'session_length',
                        'endpoint_type',
                        'resource_match',
                        'workflow_distance',
                        'prompt_similarity',
                        'action_verb_match',
                        'bigram_prob',
                        'trigram_prob']}

                cursor.execute("""
                    INSERT OR REPLACE INTO features
                    (request_id, last_endpoint_type, last_resource, time_since_last,
                     session_length, endpoint_type, resource_match, workflow_distance,
                     prompt_similarity, action_verb_match, bigram_prob, trigram_prob,
                     additional_features)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request_id,
                    features.get('last_endpoint_type'),
                    features.get('last_resource'),
                    features.get('time_since_last'),
                    features.get('session_length'),
                    features.get('endpoint_type'),
                    features.get('resource_match', 0),
                    features.get('workflow_distance'),
                    features.get('prompt_similarity'),
                    features.get('action_verb_match', 0),
                    features.get('bigram_prob'),
                    features.get('trigram_prob'),
                    json.dumps(additional_features)
                ))

                conn.commit()
                logger.debug(f"Stored features for request {request_id}")
                return True

        except Exception as e:
            logger.error(f"Error storing features: {str(e)}")
            return False

    async def store_training_data(
            self, training_samples: List[Dict[str, Any]]) -> int:
        """Store training data samples for ML model training"""
        stored_count = 0

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for sample in training_samples:
                    cursor.execute("""
                        INSERT INTO training_data
                        (sequence_id, prompt, api_call, method, is_positive, rank, features)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sample['sequence_id'],
                        sample['prompt'],
                        sample['api_call'],
                        sample['method'],
                        sample['is_positive'],
                        sample.get('rank'),
                        json.dumps(sample.get('features', {}))
                    ))
                    stored_count += 1

                conn.commit()
                logger.info(f"Stored {stored_count} training samples")

        except Exception as e:
            logger.error(f"Error storing training data: {str(e)}")
            raise

        return stored_count

    async def get_synthetic_sequences(
            self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve synthetic sequences from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = "SELECT sequence_id, workflow_type, sequence_data, sequence_length FROM synthetic_sequences"
                if limit:
                    query += f" LIMIT {limit}"

                cursor.execute(query)
                results = cursor.fetchall()

                sequences = []
                for row in results:
                    sequences.append({
                        'sequence_id': row[0],
                        'workflow_type': row[1],
                        'sequence_data': json.loads(row[2]),
                        'sequence_length': row[3]
                    })

                return sequences

        except Exception as e:
            logger.error(f"Error retrieving synthetic sequences: {str(e)}")
            return []

    async def get_training_data(
            self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve training data for ML model training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                query = """
                    SELECT sequence_id, prompt, api_call, method, is_positive, rank, features
                    FROM training_data
                """
                if limit:
                    query += f" LIMIT {limit}"

                cursor.execute(query)
                results = cursor.fetchall()

                training_data = []
                for row in results:
                    training_data.append({
                        'sequence_id': row[0],
                        'prompt': row[1],
                        'api_call': row[2],
                        'method': row[3],
                        'is_positive': bool(row[4]),
                        'rank': row[5],
                        'features': json.loads(row[6]) if row[6] else {}
                    })

                return training_data

        except Exception as e:
            logger.error(f"Error retrieving training data: {str(e)}")
            return []

    async def store_ml_model(self,
                             model_name: str,
                             model_version: str,
                             model_data: bytes,
                             metadata: Dict[str,
                                            Any]) -> bool:
        """Store trained ML model in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO ml_models
                    (model_name, model_version, model_data, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    model_name,
                    model_version,
                    model_data,
                    json.dumps(metadata),
                    datetime.now().isoformat()
                ))

                conn.commit()
                logger.info(f"Stored ML model {model_name} v{model_version}")
                return True

        except Exception as e:
            logger.error(f"Error storing ML model: {str(e)}")
            return False

    async def get_ml_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve trained ML model from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT model_version, model_data, metadata, updated_at
                    FROM ml_models
                    WHERE model_name = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (model_name,))

                result = cursor.fetchone()
                if result:
                    return {
                        'model_name': model_name,
                        'model_version': result[0],
                        'model_data': result[1],
                        'metadata': json.loads(result[2]),
                        'updated_at': result[3]
                    }

                return None

        except Exception as e:
            logger.error(f"Error retrieving ML model: {str(e)}")
            return None

    async def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                stats = {}
                tables = [
                    'synthetic_sequences',
                    'features',
                    'training_data',
                    'ml_models']

                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    stats[f"{table}_count"] = cursor.fetchone()[0]

                return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}


# Global database manager instance
db_manager = DatabaseManager()
