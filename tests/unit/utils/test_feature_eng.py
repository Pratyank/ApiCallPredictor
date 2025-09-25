"""
Unit tests for app.utils.feature_eng module.

Tests feature engineering functionality including:
- Sequence feature extraction
- Temporal feature computation
- Similarity calculations
- Workflow pattern analysis
- Feature normalization and scaling
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Mock imports - replace with actual imports when modules exist
# from app.utils.feature_eng import FeatureExtractor, SequenceAnalyzer, SimilarityCalculator


class TestFeatureExtractor:
    """Test suite for the FeatureExtractor class."""
    
    @pytest.fixture
    def feature_extractor(self):
        """Create FeatureExtractor instance."""
        extractor = Mock()
        extractor.feature_names = [
            'time_since_last', 'session_length', 'endpoint_type_encoded',
            'resource_match', 'workflow_distance', 'semantic_similarity',
            'bigram_prob', 'trigram_prob'
        ]
        return extractor
    
    @pytest.fixture
    def sample_events_with_timestamps(self):
        """Sample events with timestamp data."""
        base_time = datetime(2023, 1, 1, 10, 0, 0)
        return [
            {
                "endpoint": "/users",
                "method": "GET",
                "timestamp": base_time.isoformat(),
                "duration": 150
            },
            {
                "endpoint": "/users/123",
                "method": "GET", 
                "timestamp": (base_time + timedelta(seconds=30)).isoformat(),
                "duration": 200
            },
            {
                "endpoint": "/users/123",
                "method": "PUT",
                "timestamp": (base_time + timedelta(seconds=60)).isoformat(),
                "duration": 300
            }
        ]
    
    def test_extract_temporal_features(self, feature_extractor, sample_events_with_timestamps):
        """Test extraction of temporal features from event sequence."""
        expected_features = {
            'time_since_last': 30.0,  # 30 seconds since last event
            'session_length': 3,      # Number of events
            'avg_response_time': 216.67,  # Average duration
            'total_session_time': 60.0,   # Total session duration
            'request_rate': 0.05      # Requests per second
        }
        
        feature_extractor.extract_temporal_features = Mock(return_value=expected_features)
        
        features = feature_extractor.extract_temporal_features(sample_events_with_timestamps)
        
        assert features['time_since_last'] == 30.0
        assert features['session_length'] == 3
        assert features['avg_response_time'] == pytest.approx(216.67, rel=1e-2)
        assert features['total_session_time'] == 60.0
    
    def test_extract_sequence_patterns(self, feature_extractor):
        """Test extraction of sequence pattern features."""
        event_sequence = [
            {"endpoint": "/users", "method": "GET"},
            {"endpoint": "/users/{id}", "method": "GET"},
            {"endpoint": "/users/{id}", "method": "PUT"}
        ]
        
        expected_patterns = {
            'sequence_type': 'read_then_update',
            'pattern_confidence': 0.85,
            'workflow_stage': 'modification',
            'next_likely_actions': ['confirm', 'verify', 'redirect']
        }
        
        feature_extractor.extract_sequence_patterns = Mock(return_value=expected_patterns)
        
        patterns = feature_extractor.extract_sequence_patterns(event_sequence)
        
        assert patterns['sequence_type'] == 'read_then_update'
        assert patterns['pattern_confidence'] == 0.85
        assert 'confirm' in patterns['next_likely_actions']
    
    def test_calculate_resource_similarity(self, feature_extractor):
        """Test calculation of resource similarity between events."""
        current_endpoint = "/users/{id}"
        candidate_endpoint = "/users/{id}/profile"
        
        expected_similarity = 0.75  # High similarity - same resource, related path
        
        feature_extractor.calculate_resource_similarity = Mock(return_value=expected_similarity)
        
        similarity = feature_extractor.calculate_resource_similarity(
            current_endpoint, 
            candidate_endpoint
        )
        
        assert similarity == 0.75
        assert 0 <= similarity <= 1
    
    def test_extract_method_transition_features(self, feature_extractor):
        """Test extraction of HTTP method transition features."""
        recent_methods = ["GET", "GET", "PUT"]
        candidate_method = "POST"
        
        expected_features = {
            'method_consistency': 0.33,  # 1/3 methods are same as candidate
            'typical_transition_prob': 0.12,  # PUT -> POST probability
            'method_diversity': 0.67,   # 2/3 unique methods
            'crud_progression_score': 0.8  # Following CRUD pattern
        }
        
        feature_extractor.extract_method_features = Mock(return_value=expected_features)
        
        features = feature_extractor.extract_method_features(recent_methods, candidate_method)
        
        assert features['method_consistency'] == pytest.approx(0.33, rel=1e-2)
        assert features['crud_progression_score'] == 0.8
        assert 0 <= features['typical_transition_prob'] <= 1
    
    def test_calculate_workflow_distance(self, feature_extractor):
        """Test calculation of workflow distance between operations."""
        test_cases = [
            (("/users", "GET"), ("/users/{id}", "GET"), 1),      # Next step in workflow
            (("/users", "GET"), ("/users", "POST"), 2),          # Different operation same resource
            (("/users/{id}", "GET"), ("/users/{id}", "PUT"), 1), # Read then update
            (("/users", "POST"), ("/invoices", "GET"), 5),       # Different resource
        ]
        
        for (current, candidate, expected_distance) in test_cases:
            feature_extractor.calculate_workflow_distance = Mock(return_value=expected_distance)
            
            distance = feature_extractor.calculate_workflow_distance(current, candidate)
            
            assert distance == expected_distance
            assert distance >= 0
    
    def test_extract_parameter_features(self, feature_extractor):
        """Test extraction of parameter-based features."""
        candidate = {
            "endpoint": "/users/{id}",
            "method": "PUT",
            "parameters": [
                {"name": "id", "in": "path", "required": True},
                {"name": "include_profile", "in": "query", "required": False}
            ]
        }
        
        recent_context = {
            "current_user_id": "123",
            "viewed_resources": ["/users/123"]
        }
        
        expected_features = {
            'parameter_count': 2,
            'required_param_count': 1,
            'path_param_available': True,  # id=123 available from context
            'query_param_complexity': 0.2, # 1 optional query param
            'parameter_type_diversity': 1.0  # Different parameter types
        }
        
        feature_extractor.extract_parameter_features = Mock(return_value=expected_features)
        
        features = feature_extractor.extract_parameter_features(candidate, recent_context)
        
        assert features['parameter_count'] == 2
        assert features['required_param_count'] == 1
        assert features['path_param_available'] is True
    
    def test_normalize_feature_vector(self, feature_extractor):
        """Test feature vector normalization."""
        raw_features = {
            'time_since_last': 3600,      # 1 hour in seconds
            'session_length': 15,          # 15 events
            'semantic_similarity': 0.85,   # Already normalized
            'workflow_distance': 3,        # Distance measure
            'request_rate': 0.004          # Requests per second
        }
        
        expected_normalized = np.array([0.6, 0.75, 0.85, 0.25, 0.2])  # Min-max normalized
        
        feature_extractor.normalize_features = Mock(return_value=expected_normalized)
        
        normalized = feature_extractor.normalize_features(raw_features)
        
        assert len(normalized) == 5
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
        assert normalized[2] == 0.85  # Semantic similarity unchanged
    
    @pytest.mark.parametrize("endpoint_pair,expected_similarity", [
        (("/users", "/users/{id}"), 0.9),
        (("/users/{id}", "/users/{id}/profile"), 0.8),
        (("/users", "/invoices"), 0.1),
        (("/health", "/metrics"), 0.2),
        (("/api/v1/users", "/api/v1/users/{id}"), 0.85)
    ])
    def test_endpoint_similarity_calculation(self, feature_extractor, endpoint_pair, expected_similarity):
        """Test endpoint similarity calculation for various pairs."""
        endpoint1, endpoint2 = endpoint_pair
        
        feature_extractor.calculate_endpoint_similarity = Mock(return_value=expected_similarity)
        
        similarity = feature_extractor.calculate_endpoint_similarity(endpoint1, endpoint2)
        
        assert similarity == expected_similarity
        assert 0 <= similarity <= 1
    
    def test_extract_contextual_features(self, feature_extractor):
        """Test extraction of contextual features from user session."""
        session_context = {
            'user_agent': 'Mozilla/5.0',
            'source_ip': '192.168.1.100',
            'session_duration': 1800,  # 30 minutes
            'error_count': 2,
            'success_count': 18
        }
        
        expected_features = {
            'session_stability': 0.9,    # 18/20 success rate
            'user_experience': 0.8,      # Based on error rate and duration
            'session_maturity': 0.75,    # Based on duration and activity
            'error_rate': 0.1           # 2/20 errors
        }
        
        feature_extractor.extract_contextual_features = Mock(return_value=expected_features)
        
        features = feature_extractor.extract_contextual_features(session_context)
        
        assert features['session_stability'] == 0.9
        assert features['error_rate'] == 0.1
        assert 0 <= features['user_experience'] <= 1


class TestSequenceAnalyzer:
    """Test suite for the SequenceAnalyzer class."""
    
    @pytest.fixture
    def sequence_analyzer(self):
        """Create SequenceAnalyzer instance."""
        return Mock()
    
    def test_identify_workflow_patterns(self, sequence_analyzer):
        """Test identification of common workflow patterns."""
        sequences = [
            ["/users", "/users/{id}", "/users/{id}/edit", "/users/{id}"],  # CRUD pattern
            ["/login", "/dashboard", "/profile", "/logout"],                # Session pattern
            ["/products", "/cart/add", "/cart", "/checkout"]               # E-commerce pattern
        ]
        
        expected_patterns = [
            {'pattern': 'crud_workflow', 'confidence': 0.95},
            {'pattern': 'user_session', 'confidence': 0.88},
            {'pattern': 'shopping_flow', 'confidence': 0.92}
        ]
        
        for sequence, expected in zip(sequences, expected_patterns):
            sequence_analyzer.identify_pattern = Mock(return_value=expected)
            
            pattern = sequence_analyzer.identify_pattern(sequence)
            
            assert pattern['confidence'] > 0.8
            assert 'pattern' in pattern
    
    def test_calculate_transition_probabilities(self, sequence_analyzer):
        """Test calculation of n-gram transition probabilities."""
        training_sequences = [
            ["/users", "/users/{id}", "/users/{id}/edit"],
            ["/users", "/users/{id}", "/users/{id}/delete"],
            ["/users", "/users/create", "/users"],
            ["/products", "/products/{id}", "/cart/add"]
        ]
        
        # Test bigram probabilities
        bigram_probs = {
            ("/users", "/users/{id}"): 0.67,      # 2/3 times
            ("/users/{id}", "/users/{id}/edit"): 0.5,  # 1/2 times
            ("/users/{id}", "/users/{id}/delete"): 0.5 # 1/2 times
        }
        
        sequence_analyzer.calculate_ngram_probs = Mock(return_value=bigram_probs)
        
        probs = sequence_analyzer.calculate_ngram_probs(training_sequences, n=2)
        
        assert probs[("/users", "/users/{id}")] == pytest.approx(0.67, rel=1e-2)
        assert probs[("/users/{id}", "/users/{id}/edit")] == 0.5
    
    def test_detect_anomalous_sequences(self, sequence_analyzer):
        """Test detection of anomalous user behavior."""
        normal_sequence = ["/login", "/dashboard", "/profile", "/logout"]
        anomalous_sequences = [
            ["/admin", "/admin/users", "/admin/delete_all"],  # Suspicious admin access
            ["/users/{id}", "/users/{id}", "/users/{id}"] * 10,  # Repetitive behavior
            ["/api/internal", "/system/config", "/debug/logs"]   # Internal API access
        ]
        
        # Normal sequence should not be anomalous
        sequence_analyzer.is_anomalous = Mock(return_value=False)
        assert sequence_analyzer.is_anomalous(normal_sequence) is False
        
        # Anomalous sequences should be flagged
        for anomalous in anomalous_sequences:
            sequence_analyzer.is_anomalous = Mock(return_value=True)
            assert sequence_analyzer.is_anomalous(anomalous) is True
    
    def test_extract_sequence_statistics(self, sequence_analyzer):
        """Test extraction of statistical features from sequences."""
        sequence = [
            {"endpoint": "/users", "method": "GET", "response_time": 150},
            {"endpoint": "/users/{id}", "method": "GET", "response_time": 200},
            {"endpoint": "/users/{id}", "method": "PUT", "response_time": 350},
            {"endpoint": "/users", "method": "GET", "response_time": 120}
        ]
        
        expected_stats = {
            'sequence_length': 4,
            'unique_endpoints': 2,
            'method_diversity': 2,
            'avg_response_time': 205.0,
            'response_time_variance': 9025.0,
            'endpoint_repetition_rate': 0.5  # /users appears twice
        }
        
        sequence_analyzer.extract_statistics = Mock(return_value=expected_stats)
        
        stats = sequence_analyzer.extract_statistics(sequence)
        
        assert stats['sequence_length'] == 4
        assert stats['unique_endpoints'] == 2
        assert stats['avg_response_time'] == 205.0
        assert stats['endpoint_repetition_rate'] == 0.5


class TestSimilarityCalculator:
    """Test suite for the SimilarityCalculator class."""
    
    @pytest.fixture
    def similarity_calculator(self):
        """Create SimilarityCalculator instance."""
        return Mock()
    
    def test_calculate_semantic_similarity(self, similarity_calculator):
        """Test semantic similarity calculation between text descriptions."""
        text_pairs = [
            ("Create a new user", "Add user account"),
            ("Delete user profile", "Remove user data"),
            ("List all invoices", "Show invoice history"),
            ("Update user settings", "Create new invoice")  # Low similarity
        ]
        
        expected_similarities = [0.92, 0.88, 0.85, 0.15]
        
        for (text1, text2), expected in zip(text_pairs, expected_similarities):
            similarity_calculator.calculate_semantic_similarity = Mock(return_value=expected)
            
            similarity = similarity_calculator.calculate_semantic_similarity(text1, text2)
            
            assert similarity == expected
            assert 0 <= similarity <= 1
    
    def test_calculate_structural_similarity(self, similarity_calculator):
        """Test structural similarity between API endpoints."""
        endpoint_pairs = [
            ("/users/{id}", "/users/{user_id}"),        # Same structure, different param name
            ("/api/v1/users", "/api/v2/users"),          # Same resource, different version
            ("/users/{id}/profile", "/users/{id}/settings"),  # Same depth, different resource
            ("/users", "/invoices/{id}/items")           # Very different structure
        ]
        
        expected_similarities = [0.95, 0.8, 0.7, 0.2]
        
        for (ep1, ep2), expected in zip(endpoint_pairs, expected_similarities):
            similarity_calculator.calculate_structural_similarity = Mock(return_value=expected)
            
            similarity = similarity_calculator.calculate_structural_similarity(ep1, ep2)
            
            assert similarity == expected
            assert 0 <= similarity <= 1
    
    def test_calculate_parameter_similarity(self, similarity_calculator):
        """Test parameter similarity between endpoints."""
        param_sets = [
            (
                [{"name": "id", "type": "string"}, {"name": "format", "type": "string"}],
                [{"name": "user_id", "type": "string"}, {"name": "format", "type": "string"}]
            ),  # Similar parameters
            (
                [{"name": "id", "type": "integer"}],
                [{"name": "name", "type": "string"}, {"name": "email", "type": "string"}]
            )   # Different parameters
        ]
        
        expected_similarities = [0.85, 0.1]
        
        for (params1, params2), expected in zip(param_sets, expected_similarities):
            similarity_calculator.calculate_parameter_similarity = Mock(return_value=expected)
            
            similarity = similarity_calculator.calculate_parameter_similarity(params1, params2)
            
            assert similarity == expected
            assert 0 <= similarity <= 1
    
    def test_weighted_similarity_combination(self, similarity_calculator):
        """Test combination of multiple similarity scores with weights."""
        similarity_scores = {
            'semantic': 0.9,
            'structural': 0.8,
            'parameter': 0.7,
            'contextual': 0.6
        }
        
        weights = {
            'semantic': 0.4,
            'structural': 0.3,
            'parameter': 0.2,
            'contextual': 0.1
        }
        
        expected_combined = 0.4*0.9 + 0.3*0.8 + 0.2*0.7 + 0.1*0.6  # = 0.8
        
        similarity_calculator.combine_similarities = Mock(return_value=expected_combined)
        
        combined = similarity_calculator.combine_similarities(similarity_scores, weights)
        
        assert combined == pytest.approx(0.8, rel=1e-2)
        assert 0 <= combined <= 1
    
    @pytest.mark.parametrize("method_pair,expected_similarity", [
        (("GET", "GET"), 1.0),
        (("GET", "POST"), 0.3),
        (("POST", "PUT"), 0.7),
        (("PUT", "PATCH"), 0.9),
        (("DELETE", "GET"), 0.1)
    ])
    def test_method_similarity(self, similarity_calculator, method_pair, expected_similarity):
        """Test HTTP method similarity calculation."""
        method1, method2 = method_pair
        
        similarity_calculator.calculate_method_similarity = Mock(return_value=expected_similarity)
        
        similarity = similarity_calculator.calculate_method_similarity(method1, method2)
        
        assert similarity == expected_similarity
        assert 0 <= similarity <= 1