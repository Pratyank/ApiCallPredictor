"""
Unit tests for Cost-Aware Model Router

Test coverage:
- Model selection based on complexity and budget
- Budget tracking and consumption monitoring  
- Database operations and schema initialization
- Edge cases and error handling
"""

import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from app.models.cost_aware_router import CostAwareRouter, ModelTier


class TestCostAwareRouter:
    """Test suite for CostAwareRouter class"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)

    @pytest.fixture
    def router(self, temp_db):
        """Create router instance with temporary database"""
        return CostAwareRouter(db_path=temp_db, daily_budget=5.0)

    def test_initialization(self, router):
        """Test router initialization"""
        assert router.daily_budget == 5.0
        assert ModelTier.CHEAP.value in router.models
        assert ModelTier.PREMIUM.value in router.models
        
        # Check model configurations
        cheap_model = router.models[ModelTier.CHEAP.value]
        assert cheap_model.model == 'claude-3-haiku-20240307'
        assert cheap_model.cost == 0.00025
        assert cheap_model.accuracy == 0.7
        
        premium_model = router.models[ModelTier.PREMIUM.value]
        assert premium_model.model == 'claude-3-opus-20240229'
        assert premium_model.cost == 0.015
        assert premium_model.accuracy == 0.9

    def test_database_initialization(self, router):
        """Test database schema creation"""
        with sqlite3.connect(router.db_path) as conn:
            cursor = conn.cursor()
            
            # Check budget_consumption table
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='budget_consumption'
            """)
            assert cursor.fetchone() is not None
            
            # Check model_performance table
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='model_performance'
            """)
            assert cursor.fetchone() is not None
            
            # Check router_settings table
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='router_settings'
            """)
            assert cursor.fetchone() is not None

    def test_route_low_complexity(self, router):
        """Test routing decision for low complexity query"""
        complexity_score = 0.3
        decision = router.route(complexity_score)
        
        assert decision['model_tier'] == ModelTier.CHEAP.value
        assert decision['selected_model'] == 'claude-3-haiku-20240307'
        assert decision['complexity_score'] == complexity_score
        assert 'estimated_cost' in decision
        assert 'routing_reason' in decision

    def test_route_high_complexity(self, router):
        """Test routing decision for high complexity query"""
        complexity_score = 0.8
        decision = router.route(complexity_score)
        
        assert decision['model_tier'] == ModelTier.PREMIUM.value
        assert decision['selected_model'] == 'claude-3-opus-20240229'
        assert decision['complexity_score'] == complexity_score
        assert decision['estimated_cost'] > 0

    def test_route_with_budget_constraint(self, router):
        """Test routing with strict budget constraints"""
        complexity_score = 0.9  # High complexity
        max_cost = 0.001  # Very low budget
        
        decision = router.route(complexity_score, max_cost)
        
        # Should fallback to cheap model due to budget constraint
        assert decision['model_tier'] == ModelTier.CHEAP.value
        assert 'budget constraint' in decision['routing_reason'].lower()

    def test_estimate_tokens(self, router):
        """Test token estimation based on complexity"""
        # Low complexity
        tokens_low = router._estimate_tokens(0.1)
        assert 200 <= tokens_low <= 350
        
        # High complexity
        tokens_high = router._estimate_tokens(0.9)
        assert tokens_high > tokens_low
        assert tokens_high <= 1700  # 200 base + 1500 max complexity

    def test_track_usage(self, router):
        """Test usage tracking functionality"""
        model_used = 'claude-3-haiku-20240307'
        tokens_consumed = 500
        query_hash = 'test_hash_123'
        complexity_score = 0.4
        
        router.track_usage(model_used, tokens_consumed, query_hash, complexity_score)
        
        # Verify data was stored
        with sqlite3.connect(router.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_used, tokens_consumed, complexity_score 
                FROM budget_consumption 
                WHERE query_hash = ?
            """, (query_hash,))
            
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == model_used
            assert result[1] == tokens_consumed
            assert result[2] == complexity_score

    def test_get_daily_consumption(self, router):
        """Test daily consumption calculation"""
        # Initial consumption should be 0
        consumption = router._get_daily_consumption()
        assert consumption == 0.0
        
        # Add some usage data
        router.track_usage('claude-3-haiku-20240307', 1000, 'hash1', 0.5)
        router.track_usage('claude-3-opus-20240229', 500, 'hash2', 0.8)
        
        # Clear cache to force recalculation
        router._daily_consumption_cache = None
        consumption = router._get_daily_consumption()
        
        # Should be greater than 0 now
        assert consumption > 0.0

    def test_get_budget_status(self, router):
        """Test budget status reporting"""
        status = router.get_budget_status()
        
        assert 'daily_budget' in status
        assert 'daily_consumption' in status
        assert 'remaining_budget' in status
        assert 'budget_utilization' in status
        assert 'available_models' in status
        
        assert status['daily_budget'] == 5.0
        assert status['remaining_budget'] <= status['daily_budget']

    def test_complexity_threshold_management(self, router):
        """Test complexity threshold storage and retrieval"""
        # Test default threshold
        threshold = router._get_complexity_threshold()
        assert 0.0 <= threshold <= 1.0
        
        # Test setting custom threshold
        with sqlite3.connect(router.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO router_settings 
                (setting_key, setting_value)
                VALUES ('complexity_threshold', '0.75')
            """)
            conn.commit()
        
        new_threshold = router._get_complexity_threshold()
        assert new_threshold == 0.75

    def test_fallback_routing(self, router):
        """Test fallback routing when errors occur"""
        fallback = router._get_fallback_routing()
        
        assert fallback['model_tier'] == ModelTier.CHEAP.value
        assert fallback['is_fallback'] is True
        assert 'Fallback' in fallback['routing_reason']

    def test_select_optimal_model_logic(self, router):
        """Test model selection logic with various scenarios"""
        # Low complexity, sufficient budget
        tier = router._select_optimal_model(0.3, 1.0)
        assert tier == ModelTier.CHEAP.value
        
        # High complexity, sufficient budget
        tier = router._select_optimal_model(0.8, 1.0)
        assert tier == ModelTier.PREMIUM.value
        
        # High complexity, insufficient budget for premium
        tier = router._select_optimal_model(0.8, 0.001)
        assert tier == ModelTier.CHEAP.value

    def test_global_router_instance(self, temp_db):
        """Test global router instance management"""
        # Import the function here
        from app.models.cost_aware_router import get_cost_aware_router
        
        # Clear any existing global instance
        import app.models.cost_aware_router
        app.models.cost_aware_router._router_instance = None
        
        router1 = get_cost_aware_router(db_path=temp_db)
        router2 = get_cost_aware_router(db_path=temp_db)
        
        # Should return the same instance
        assert router1 is router2

    def test_error_handling(self, router):
        """Test error handling in various scenarios"""
        # Test with invalid database path
        invalid_router = CostAwareRouter(db_path="/invalid/path/db.sqlite")
        
        # Should still work with fallback behavior
        decision = invalid_router.route(0.5)
        assert 'selected_model' in decision

    def test_routing_reason_generation(self, router):
        """Test routing reason explanations"""
        # Test different scenarios and ensure reasons are descriptive
        decision_low = router.route(0.2)
        assert 'low complexity' in decision_low['routing_reason']
        
        decision_high = router.route(0.9)
        assert 'high complexity' in decision_high['routing_reason'] or \
               'Premium model selected' in decision_high['routing_reason']

    def test_cache_behavior(self, router):
        """Test consumption cache behavior"""
        # First call - should cache result
        consumption1 = router._get_daily_consumption()
        timestamp1 = router._cache_timestamp
        
        # Immediate second call - should use cache
        consumption2 = router._get_daily_consumption()
        timestamp2 = router._cache_timestamp
        
        assert consumption1 == consumption2
        assert timestamp1 == timestamp2
        
        # Invalidate cache and verify
        router._daily_consumption_cache = None
        consumption3 = router._get_daily_consumption()
        # Should recalculate but likely same value
        assert isinstance(consumption3, float)

    def test_optimization_placeholder(self, router):
        """Test optimization method placeholder"""
        # Test that optimization method runs without errors
        performance_data = [
            {'model_tier': 'cheap', 'complexity': 0.3, 'accuracy': 0.7},
            {'model_tier': 'premium', 'complexity': 0.8, 'accuracy': 0.9}
        ]
        
        # Should not raise exceptions
        router.optimize_threshold(performance_data)
        
        # Test with empty data
        router.optimize_threshold([])


class TestIntegrationScenarios:
    """Integration test scenarios for realistic usage"""

    @pytest.fixture
    def router(self):
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        router = CostAwareRouter(db_path=path, daily_budget=10.0)
        yield router
        os.unlink(path)

    def test_daily_usage_simulation(self, router):
        """Simulate a day of API usage"""
        queries = [
            (0.2, "simple query"),
            (0.7, "complex analysis"),
            (0.9, "advanced processing"),
            (0.3, "basic lookup"),
            (0.8, "detailed report")
        ]
        
        total_cost = 0.0
        
        for complexity, description in queries:
            decision = router.route(complexity)
            estimated_cost = decision['estimated_cost']
            total_cost += estimated_cost
            
            # Simulate usage tracking
            router.track_usage(
                model_used=decision['selected_model'],
                tokens_consumed=decision['estimated_tokens'],
                query_hash=f"hash_{description}",
                complexity_score=complexity
            )
        
        # Check final budget status
        status = router.get_budget_status()
        assert status['daily_consumption'] > 0
        assert status['remaining_budget'] < status['daily_budget']

    def test_budget_exhaustion_scenario(self, router):
        """Test behavior when budget is nearly exhausted"""
        # Simulate high usage to exhaust budget
        for i in range(20):
            router.track_usage(
                model_used='claude-3-opus-20240229',
                tokens_consumed=1000,  # High token usage
                query_hash=f'exhaust_hash_{i}',
                complexity_score=0.8
            )
        
        # Try routing with exhausted budget
        decision = router.route(0.9)  # High complexity
        
        # Should adapt to budget constraints
        assert 'selected_model' in decision
        assert decision['remaining_budget'] <= decision['daily_budget_used']


if __name__ == "__main__":
    pytest.main([__file__])