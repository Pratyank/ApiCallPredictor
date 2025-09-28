"""
ML Ranker for Phase 3 ML Layer - OpenSesame Predictor
Implements LightGBM-based ranking model with learning-to-rank
Uses objective='lambdarank', metric='ndcg', n_estimators=100, num_leaves=31
"""

import numpy as np
import pandas as pd
import logging
import pickle
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

from app.config import get_settings
from app.utils.db_manager import db_manager
from app.utils.feature_eng import FeatureExtractor

logger = logging.getLogger(__name__)


class MLRanker:
    """
    LightGBM-based ranking system for API call prediction
    Implements learning-to-rank with NDCG optimization
    """

    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.feature_names = None
        self.label_encoders = {}
        self.is_trained = False
        self.training_stats = {}

        # LightGBM parameters as specified in requirements
        self.lgb_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }

        self.feature_extractor = FeatureExtractor()

        logger.info("Initialized MLRanker with LightGBM lambdarank objective")

    async def train_ranker(
            self, training_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Train LightGBM ranker using synthetic sequences and features

        Args:
            training_data: Optional training data, if None loads from database

        Returns:
            Training statistics and model performance metrics
        """

        try:
            logger.info("Starting MLRanker training with LightGBM")
            start_time = datetime.now()

            # Load training data
            if training_data is None:
                training_data = await self._load_training_data()

            if not training_data:
                raise ValueError("No training data available")

            logger.info(f"Loaded {len(training_data)} training samples")

            # Extract features for training data
            features_data = await self._prepare_training_features(training_data)

            # Prepare training dataset for LightGBM ranking
            X_train, y_train, groups_train, X_val, y_val, groups_val = await self._prepare_ranking_dataset(features_data)

            logger.info(
                f"Training set: {
                    len(X_train)} samples, {
                    len(groups_train)} groups")
            logger.info(
                f"Validation set: {
                    len(X_val)} samples, {
                    len(groups_val)} groups")

            # Create LightGBM datasets
            train_data = lgb.Dataset(
                X_train, label=y_train, group=groups_train)
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                group=groups_val,
                reference=train_data)

            # Train LightGBM ranker
            logger.info("Training LightGBM ranker...")
            self.model = lgb.train(
                params=self.lgb_params,
                train_set=train_data,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10)],
                num_boost_round=self.lgb_params['n_estimators']
            )

            # Store model and metadata
            await self._save_model()

            # Calculate training statistics
            self.is_trained = True
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()

            # Evaluate model performance
            val_predictions = self.model.predict(X_val)
            ndcg_score = self._calculate_ndcg(
                y_val, val_predictions, groups_val)

            self.training_stats = {
                'training_samples': len(training_data),
                'features_extracted': len(
                    X_train[0]) if len(X_train) > 0 else 0,
                'training_time_seconds': training_time,
                'validation_ndcg': ndcg_score,
                'model_version': 'v1.0-lightgbm',
                'training_timestamp': end_time.isoformat(),
                'lgb_params': self.lgb_params.copy()}

            logger.info(
                f"MLRanker training completed - NDCG: {
                    ndcg_score:.4f}, Time: {
                    training_time:.2f}s")
            return self.training_stats

        except Exception as e:
            logger.error(f"MLRanker training failed: {str(e)}")
            raise

    async def rank_predictions(
        self,
        predictions: List[Dict[str, Any]],
        prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        k: int = 3,
        buffer: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Rank API predictions using trained LightGBM model

        Args:
            predictions: List of API predictions to rank
            prompt: User prompt for context
            history: Conversation history
            k: Number of top results to return
            buffer: Additional candidates to consider (k + buffer total)

        Returns:
            Ranked predictions with ML confidence scores
        """

        if not self.is_trained or self.model is None:
            logger.warning("MLRanker not trained, using fallback ranking")
            return await self._fallback_ranking(predictions, k)

        try:
            # Extract features for each prediction candidate
            candidates_with_features = []

            for prediction in predictions[:k +
                                          buffer]:  # Only process k+buffer candidates
                # Extract ML features
                features = await self.feature_extractor.extract_ml_features(
                    prompt=prompt,
                    history=history,
                    candidate_api=prediction
                )

                candidates_with_features.append({
                    'prediction': prediction,
                    'features': features
                })

            if not candidates_with_features:
                return []

            # Prepare features for prediction
            X_pred = self._prepare_prediction_features(
                candidates_with_features)

            # Get ML ranking scores
            ranking_scores = self.model.predict(X_pred)

            # Combine predictions with scores and rank
            ranked_candidates = []
            for i, candidate in enumerate(candidates_with_features):
                prediction = candidate['prediction'].copy()
                prediction['ml_ranking_score'] = float(ranking_scores[i])
                prediction['ml_features'] = candidate['features']
                ranked_candidates.append(prediction)

            # Sort by ML ranking score
            ranked_candidates.sort(
                key=lambda x: x['ml_ranking_score'], reverse=True)

            # Add ranking metadata
            for i, prediction in enumerate(ranked_candidates):
                prediction['ml_rank'] = i + 1
                prediction['model_version'] = self.training_stats.get(
                    'model_version', 'unknown')

            logger.info(
                f"Ranked {
                    len(ranked_candidates)} predictions using ML model")
            return ranked_candidates[:k]  # Return top k results

        except Exception as e:
            logger.error(f"ML ranking failed: {str(e)}")
            return await self._fallback_ranking(predictions, k)

    async def _load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from database"""
        return await db_manager.get_training_data()

    async def _prepare_training_features(
            self, training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and prepare features for training data"""
        logger.info("Extracting features for training data...")

        # Use batch feature extraction from FeatureExtractor
        return await self.feature_extractor.batch_extract_features(training_data)

    async def _prepare_ranking_dataset(
            self, features_data: List[Dict[str, Any]]) -> Tuple:
        """Prepare dataset for LightGBM ranking"""

        # Group data by sequence_id for ranking
        sequence_groups = defaultdict(list)
        for item in features_data:
            sequence_id = item.get('sequence_id', 'unknown')
            sequence_groups[sequence_id].append(item)

        # Prepare features and labels
        all_features = []
        all_labels = []
        all_groups = []

        for sequence_id, group_items in sequence_groups.items():
            if len(group_items) < 2:  # Skip groups with less than 2 items
                continue

            group_features = []
            group_labels = []

            for item in group_items:
                # Extract feature vector
                feature_vector = self._extract_feature_vector(item)
                if feature_vector is not None:
                    group_features.append(feature_vector)

                    # Convert is_positive to ranking label (1 for positive, 0
                    # for negative)
                    label = 1 if item.get('is_positive', False) else 0
                    group_labels.append(label)

            if group_features:
                all_features.extend(group_features)
                all_labels.extend(group_labels)
                all_groups.append(len(group_features))

        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        groups = np.array(all_groups)

        # Split into train/validation
        # For ranking, we need to split by groups, not individual samples
        group_indices = np.arange(len(groups))
        train_group_idx, val_group_idx = train_test_split(
            group_indices, test_size=0.2, random_state=42)

        # Calculate sample indices for each group
        train_start = 0
        val_start = 0
        train_indices = []
        val_indices = []

        for i, group_size in enumerate(groups):
            if i in train_group_idx:
                train_indices.extend(
                    range(
                        train_start,
                        train_start +
                        group_size))
                train_start += group_size
            else:
                val_indices.extend(range(val_start, val_start + group_size))
                val_start += group_size

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        groups_train = groups[train_group_idx]
        groups_val = groups[val_group_idx]

        return X_train, y_train, groups_train, X_val, y_val, groups_val

    def _extract_feature_vector(
            self, feature_item: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract numerical feature vector from feature dictionary"""
        try:
            feature_vector = []

            # Define expected features in order
            expected_features = [
                'time_since_last', 'session_length', 'resource_match',
                'workflow_distance', 'prompt_similarity', 'action_verb_match',
                'bigram_prob', 'trigram_prob'
            ]

            # Extract numerical features
            for feature_name in expected_features:
                value = feature_item.get(feature_name, 0.0)
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                else:
                    feature_vector.append(0.0)

            # Encode categorical features
            categorical_features = {
                'last_endpoint_type': [
                    'GET',
                    'POST',
                    'PUT',
                    'DELETE',
                    'PATCH',
                    'NONE',
                    'UNKNOWN'],
                'last_resource': [
                    'users',
                    'items',
                    'documents',
                    'products',
                    'orders',
                    'files',
                    'auth',
                    'none',
                    'unknown'],
                'endpoint_type': [
                    'GET',
                    'POST',
                    'PUT',
                    'DELETE',
                    'PATCH',
                    'UNKNOWN']}

            for cat_feature, categories in categorical_features.items():
                value = feature_item.get(cat_feature, 'UNKNOWN')
                if value in categories:
                    # One-hot encode
                    for cat in categories:
                        feature_vector.append(1.0 if value == cat else 0.0)
                else:
                    # Unknown category
                    feature_vector.extend([0.0] * len(categories))

            return np.array(feature_vector)

        except Exception as e:
            logger.error(f"Feature vector extraction failed: {e}")
            return None

    def _prepare_prediction_features(
            self, candidates_with_features: List[Dict[str, Any]]) -> np.ndarray:
        """Prepare features for prediction"""
        feature_vectors = []

        for candidate in candidates_with_features:
            feature_vector = self._extract_feature_vector(
                candidate['features'])
            if feature_vector is not None:
                feature_vectors.append(feature_vector)

        return np.array(feature_vectors)

    def _calculate_ndcg(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray,
            groups: np.ndarray) -> float:
        """Calculate NDCG score for ranking evaluation"""
        try:
            from sklearn.metrics import ndcg_score

            # For group-wise NDCG calculation
            ndcg_scores = []
            start_idx = 0

            for group_size in groups:
                end_idx = start_idx + group_size
                if group_size > 1:  # Only calculate NDCG for groups with multiple items
                    group_true = y_true[start_idx:end_idx].reshape(1, -1)
                    group_pred = y_pred[start_idx:end_idx].reshape(1, -1)
                    ndcg = ndcg_score(group_true, group_pred)
                    ndcg_scores.append(ndcg)
                start_idx = end_idx

            return np.mean(ndcg_scores) if ndcg_scores else 0.0

        except ImportError:
            logger.warning("sklearn NDCG not available, using simple accuracy")
            return self._simple_accuracy(y_true, y_pred)

    def _simple_accuracy(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray) -> float:
        """Simple accuracy calculation as fallback"""
        # Convert predictions to binary
        y_pred_binary = (y_pred > 0.5).astype(int)
        return np.mean(y_true == y_pred_binary)

    async def _save_model(self):
        """Save trained model to database"""
        if self.model is None:
            return

        try:
            # Serialize model
            model_data = pickle.dumps(self.model)

            # Save to database
            metadata = {
                'lgb_params': self.lgb_params,
                'feature_names': self.feature_names,
                'training_timestamp': datetime.now().isoformat(),
                'model_type': 'lightgbm_ranker'
            }

            await db_manager.store_ml_model(
                model_name='ml_ranker',
                model_version='v1.0-lightgbm',
                model_data=model_data,
                metadata=metadata
            )

            logger.info("Saved MLRanker model to database")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    async def load_model(self) -> bool:
        """Load trained model from database"""
        try:
            model_data = await db_manager.get_ml_model('ml_ranker')

            if model_data:
                self.model = pickle.loads(model_data['model_data'])
                metadata = model_data['metadata']
                self.lgb_params = metadata.get('lgb_params', self.lgb_params)
                self.feature_names = metadata.get('feature_names')
                self.is_trained = True

                logger.info(
                    f"Loaded MLRanker model v{
                        model_data['model_version']}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    async def _fallback_ranking(
            self, predictions: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Fallback ranking when ML model not available"""
        logger.info("Using fallback ranking (confidence-based)")

        # Sort by confidence score
        ranked = sorted(
            predictions,
            key=lambda x: x.get(
                'confidence',
                0.0),
            reverse=True)

        # Add fallback ranking metadata
        for i, prediction in enumerate(ranked[:k]):
            prediction['ml_rank'] = i + 1
            prediction['ml_ranking_score'] = prediction.get('confidence', 0.0)
            prediction['model_version'] = 'fallback'

        return ranked[:k]

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        return {
            'is_trained': self.is_trained,
            'model_type': 'LightGBM Ranker',
            'objective': self.lgb_params['objective'],
            'metric': self.lgb_params['metric'],
            'n_estimators': self.lgb_params['n_estimators'],
            'num_leaves': self.lgb_params['num_leaves'],
            'training_stats': self.training_stats,
            'model_version': 'v1.0-lightgbm'
        }

    async def retrain_with_feedback(
            self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Retrain model with user feedback data"""
        logger.info(
            f"Retraining MLRanker with {
                len(feedback_data)} feedback samples")

        # Combine existing training data with feedback
        existing_data = await self._load_training_data()
        combined_data = existing_data + feedback_data

        # Retrain model
        return await self.train_ranker(combined_data)


# Convenience functions
async def train_ml_ranker() -> Dict[str, Any]:
    """Train the ML ranker with available data"""
    ranker = MLRanker()
    return await ranker.train_ranker()


async def get_trained_ranker() -> MLRanker:
    """Get a trained ML ranker instance"""
    ranker = MLRanker()
    if not await ranker.load_model():
        logger.info("No trained model found, training new MLRanker...")
        await ranker.train_ranker()
    return ranker
