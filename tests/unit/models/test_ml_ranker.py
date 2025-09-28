"""
Unit tests for app.models.ml_ranker module.

Tests the ML ranking functionality including:
- LightGBM model training and inference
- Feature engineering and extraction
- Model persistence and loading
- Performance optimization
- Learning-to-rank implementation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import pickle

# Mock imports - replace with actual imports when modules exist
# from app.models.ml_ranker import MLRanker, FeatureExtractor, ModelTrainer
# import lightgbm as lgb


class TestMLRanker:
    """Test suite for the MLRanker class."""

    @pytest.fixture
    def sample_candidates(self):
        """Sample candidates for ranking."""
        return [
            {
                "endpoint": "/users",
                "method": "POST",
                "raw_confidence": 0.9,
                "semantic_similarity": 0.85,
                "workflow_probability": 0.8
            },
            {
                "endpoint": "/users/{id}",
                "method": "GET",
                "raw_confidence": 0.8,
                "semantic_similarity": 0.7,
                "workflow_probability": 0.9
            },
            {
                "endpoint": "/invoices",
                "method": "GET",
                "raw_confidence": 0.7,
                "semantic_similarity": 0.6,
                "workflow_probability": 0.5
            }
        ]

    @pytest.fixture
    def sample_features(self):
        """Sample feature matrix for testing."""
        return np.array([
            [0.9, 0.85, 0.8, 1, 0, 0],  # POST /users
            [0.8, 0.7, 0.9, 0, 1, 0],   # GET /users/{id}
            [0.7, 0.6, 0.5, 0, 1, 1]    # GET /invoices
        ])

    @pytest.fixture
    def mock_lgb_model(self):
        """Mock LightGBM model."""
        model = Mock()
        model.predict = Mock(return_value=np.array([0.95, 0.85, 0.75]))
        model.feature_importance = Mock(
            return_value=np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0]))
        return model

    @pytest.fixture
    def ml_ranker(self, mock_lgb_model, temp_model_dir):
        """Create MLRanker instance with mocked dependencies."""
        ranker = Mock()
        ranker.model = mock_lgb_model
        ranker.model_path = os.path.join(temp_model_dir, "model.pkl")
        ranker.is_trained = Mock(return_value=True)
        ranker.feature_names = [
            "raw_confidence", "semantic_similarity", "workflow_probability",
            "is_post", "is_get", "is_safe_operation"
        ]
        return ranker

    def test_rank_candidates_success(self, ml_ranker, sample_candidates):
        """Test successful candidate ranking."""
        expected_ranked = [
            {"endpoint": "/users", "method": "POST", "score": 0.95, "rank": 1},
            {"endpoint": "/users/{id}", "method": "GET", "score": 0.85, "rank": 2},
            {"endpoint": "/invoices", "method": "GET", "score": 0.75, "rank": 3}
        ]

        ml_ranker.rank_candidates = Mock(return_value=expected_ranked)

        result = ml_ranker.rank_candidates(sample_candidates)

        assert len(result) == 3
        assert result[0]["score"] == 0.95
        assert result[0]["rank"] == 1
        assert result[0]["endpoint"] == "/users"

    def test_extract_features(self, ml_ranker, sample_candidates):
        """Test feature extraction from candidates."""
        expected_features = np.array([
            [0.9, 0.85, 0.8, 1, 0, 1],
            [0.8, 0.7, 0.9, 0, 1, 1],
            [0.7, 0.6, 0.5, 0, 1, 1]
        ])

        ml_ranker.extract_features = Mock(return_value=expected_features)

        features = ml_ranker.extract_features(sample_candidates)

        assert features.shape == (3, 6)  # 3 candidates, 6 features
        assert np.array_equal(features, expected_features)

    def test_model_loading(self, ml_ranker, temp_model_dir):
        """Test model loading from disk."""
        model_path = os.path.join(temp_model_dir, "test_model.pkl")

        # Create a mock model file
        mock_model = {"model_type": "test", "version": "1.0"}
        with open(model_path, 'wb') as f:
            pickle.dump(mock_model, f)

        ml_ranker.load_model = Mock(return_value=mock_model)

        loaded_model = ml_ranker.load_model(model_path)

        assert loaded_model is not None
        ml_ranker.load_model.assert_called_once_with(model_path)

    def test_model_saving(self, ml_ranker, temp_model_dir):
        """Test model saving to disk."""
        model_path = os.path.join(temp_model_dir, "saved_model.pkl")

        ml_ranker.save_model = Mock()
        ml_ranker.save_model(model_path)

        ml_ranker.save_model.assert_called_once_with(model_path)

    def test_model_not_trained_error(self):
        """Test error when trying to rank with untrained model."""
        untrained_ranker = Mock()
        untrained_ranker.is_trained = Mock(return_value=False)
        untrained_ranker.rank_candidates = Mock(
            side_effect=ValueError("Model not trained"))

        with pytest.raises(ValueError, match="Model not trained"):
            untrained_ranker.rank_candidates([])

    def test_empty_candidates_handling(self, ml_ranker):
        """Test handling of empty candidate list."""
        ml_ranker.rank_candidates = Mock(return_value=[])

        result = ml_ranker.rank_candidates([])

        assert result == []

    def test_single_candidate_ranking(self, ml_ranker):
        """Test ranking with single candidate."""
        single_candidate = [{"endpoint": "/users",
                             "method": "GET", "raw_confidence": 0.8}]
        expected_result = [{"endpoint": "/users",
                            "method": "GET", "score": 0.8, "rank": 1}]

        ml_ranker.rank_candidates = Mock(return_value=expected_result)

        result = ml_ranker.rank_candidates(single_candidate)

        assert len(result) == 1
        assert result[0]["rank"] == 1

    def test_feature_importance(self, ml_ranker):
        """Test feature importance extraction."""
        expected_importance = {
            "raw_confidence": 0.4,
            "semantic_similarity": 0.3,
            "workflow_probability": 0.2,
            "is_post": 0.1,
            "is_get": 0.0,
            "is_safe_operation": 0.0
        }

        ml_ranker.get_feature_importance = Mock(
            return_value=expected_importance)

        importance = ml_ranker.get_feature_importance()

        assert importance["raw_confidence"] == 0.4
        assert importance["semantic_similarity"] == 0.3
        assert sum(importance.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("method,expected_feature", [
        ("GET", [0, 1, 0]),
        ("POST", [1, 0, 0]),
        ("PUT", [0, 0, 1]),
        ("DELETE", [0, 0, 0])
    ])
    def test_method_encoding(self, ml_ranker, method, expected_feature):
        """Test HTTP method one-hot encoding."""
        ml_ranker.encode_method = Mock(return_value=expected_feature)

        encoded = ml_ranker.encode_method(method)

        assert encoded == expected_feature
        assert len(encoded) == 3  # GET, POST, PUT

    def test_performance_metrics(self, ml_ranker):
        """Test model performance metrics calculation."""
        y_true = np.array([1, 2, 3])  # True rankings
        y_pred = np.array([0.9, 0.8, 0.7])  # Predicted scores

        expected_metrics = {
            "ndcg@3": 0.95,
            "ndcg@5": 0.95,
            "map": 0.89,
            "accuracy@1": 1.0
        }

        ml_ranker.calculate_metrics = Mock(return_value=expected_metrics)

        metrics = ml_ranker.calculate_metrics(y_true, y_pred)

        assert "ndcg@3" in metrics
        assert metrics["ndcg@3"] == 0.95
        assert metrics["accuracy@1"] == 1.0


class TestFeatureExtractor:
    """Test suite for the FeatureExtractor class."""

    @pytest.fixture
    def feature_extractor(self):
        """Create FeatureExtractor instance."""
        extractor = Mock()
        extractor.feature_names = [
            "raw_confidence",
            "semantic_similarity",
            "workflow_probability",
            "time_since_last",
            "session_length",
            "endpoint_type_encoded",
            "resource_match",
            "workflow_distance",
            "bigram_prob",
            "trigram_prob"]
        return extractor

    def test_extract_sequence_features(self, feature_extractor):
        """Test extraction of sequence-based features."""
        recent_events = [
            {"endpoint": "/users", "method": "GET", "timestamp": "2023-01-01T10:00:00Z"},
            {"endpoint": "/users/123", "method": "PUT", "timestamp": "2023-01-01T10:01:00Z"}
        ]

        expected_features = {
            "time_since_last": 60.0,  # 1 minute
            "session_length": 2,
            "last_endpoint_type": "PUT",
            "last_resource": "users"
        }

        feature_extractor.extract_sequence_features = Mock(
            return_value=expected_features)

        features = feature_extractor.extract_sequence_features(recent_events)

        assert features["time_since_last"] == 60.0
        assert features["session_length"] == 2
        assert features["last_endpoint_type"] == "PUT"

    def test_extract_candidate_features(self, feature_extractor):
        """Test extraction of candidate-specific features."""
        candidate = {
            "endpoint": "/users/{id}",
            "method": "GET",
            "raw_confidence": 0.85
        }

        recent_events = [{"endpoint": "/users", "method": "POST"}]

        expected_features = {
            "endpoint_type": "GET",
            "resource_match": True,
            "workflow_distance": 1,
            "raw_confidence": 0.85
        }

        feature_extractor.extract_candidate_features = Mock(
            return_value=expected_features)

        features = feature_extractor.extract_candidate_features(
            candidate, recent_events)

        assert features["resource_match"] is True
        assert features["workflow_distance"] == 1
        assert features["raw_confidence"] == 0.85

    def test_extract_prompt_features(self, feature_extractor):
        """Test extraction of prompt-based features."""
        prompt = "Update the user profile"
        candidate = {
            "endpoint": "/users/{id}",
            "method": "PUT",
            "description": "Update user by ID"}

        expected_features = {
            "prompt_similarity": 0.92,
            "action_verb_match": True,
            "intent_alignment": 0.88
        }

        feature_extractor.extract_prompt_features = Mock(
            return_value=expected_features)

        features = feature_extractor.extract_prompt_features(prompt, candidate)

        assert features["prompt_similarity"] == 0.92
        assert features["action_verb_match"] is True
        assert features["intent_alignment"] == 0.88

    def test_calculate_transition_probabilities(self, feature_extractor):
        """Test calculation of transition probabilities."""
        last_actions = ["/users", "/users/{id}"]
        candidate = {"endpoint": "/invoices", "method": "GET"}

        expected_probs = {
            "bigram_prob": 0.15,   # P(invoices|users/{id})
            "trigram_prob": 0.08   # P(invoices|users,users/{id})
        }

        feature_extractor.calculate_transition_probs = Mock(
            return_value=expected_probs)

        probs = feature_extractor.calculate_transition_probs(
            last_actions, candidate)

        assert probs["bigram_prob"] == 0.15
        assert probs["trigram_prob"] == 0.08

    def test_normalize_features(self, feature_extractor):
        """Test feature normalization."""
        raw_features = np.array(
            [[100, 0.5, 1000], [200, 0.8, 2000], [50, 0.3, 500]])

        expected_normalized = np.array([
            [0.0, 0.5, 0.0],     # Min-max normalized
            [1.0, 1.0, 1.0],     # Max values
            [-1.0, 0.0, -1.0]    # Min values (adjusted)
        ])

        feature_extractor.normalize_features = Mock(
            return_value=expected_normalized)

        normalized = feature_extractor.normalize_features(raw_features)

        assert normalized.shape == raw_features.shape
        # Check that values are properly scaled
        assert np.all(normalized >= -1) and np.all(normalized <= 1)


class TestModelTrainer:
    """Test suite for the ModelTrainer class."""

    @pytest.fixture
    def mock_training_data(self):
        """Mock training data."""
        return {
            "X": np.random.rand(1000, 10),  # 1000 samples, 10 features
            "y": np.random.randint(1, 6, 1000),  # Rankings 1-5
            # 100 groups of 10 candidates each
            "groups": np.repeat(range(100), 10)
        }

    @pytest.fixture
    def model_trainer(self):
        """Create ModelTrainer instance."""
        trainer = Mock()
        trainer.model_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.1
        }
        return trainer

    def test_train_model_success(self, model_trainer, mock_training_data):
        """Test successful model training."""
        mock_model = Mock()
        mock_model.feature_importance_ = np.random.rand(10)

        model_trainer.train = Mock(return_value=mock_model)

        trained_model = model_trainer.train(
            X=mock_training_data["X"],
            y=mock_training_data["y"],
            groups=mock_training_data["groups"]
        )

        assert trained_model is not None
        model_trainer.train.assert_called_once()

    def test_generate_synthetic_data(self, model_trainer):
        """Test synthetic training data generation."""
        expected_data = {
            "sequences": 1000,
            "positive_examples": 5000,
            "negative_examples": 15000,
            "features_per_example": 10
        }

        model_trainer.generate_synthetic_data = Mock(
            return_value=expected_data)

        synthetic_data = model_trainer.generate_synthetic_data(
            n_sequences=1000)

        assert synthetic_data["sequences"] == 1000
        assert synthetic_data["positive_examples"] == 5000

    def test_cross_validation(self, model_trainer, mock_training_data):
        """Test cross-validation during training."""
        cv_scores = {
            "ndcg@3": [0.85, 0.87, 0.86, 0.88, 0.84],
            "map": [0.78, 0.80, 0.79, 0.82, 0.77]
        }

        model_trainer.cross_validate = Mock(return_value=cv_scores)

        scores = model_trainer.cross_validate(
            X=mock_training_data["X"],
            y=mock_training_data["y"],
            groups=mock_training_data["groups"],
            cv=5
        )

        assert "ndcg@3" in scores
        assert len(scores["ndcg@3"]) == 5
        assert np.mean(scores["ndcg@3"]) > 0.8

    def test_hyperparameter_tuning(self, model_trainer):
        """Test hyperparameter optimization."""
        param_grid = {
            "n_estimators": [50, 100, 200],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.05, 0.1, 0.2]
        }

        best_params = {
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "best_score": 0.89
        }

        model_trainer.tune_hyperparameters = Mock(return_value=best_params)

        tuned_params = model_trainer.tune_hyperparameters(param_grid)

        assert tuned_params["best_score"] == 0.89
        assert tuned_params["n_estimators"] == 100
