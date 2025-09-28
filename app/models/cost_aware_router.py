"""
Cost-Aware Model Router for OpenSesame Predictor

This module implements intelligent model selection based on query complexity
and budget constraints, optimizing for both cost and accuracy.
"""

import sqlite3
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import os


logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tiers with different cost and capability profiles"""
    CHEAP = "cheap"
    PREMIUM = "premium"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model: str
    cost: float  # Cost per 1K tokens
    accuracy: float  # Expected accuracy score (0-1)
    max_tokens: int = 4000
    temperature: float = 0.7


@dataclass
class BudgetEntry:
    """Budget consumption tracking entry"""
    timestamp: datetime
    model_used: str
    tokens_consumed: int
    cost_incurred: float
    query_hash: str
    complexity_score: float


class CostAwareRouter:
    """
    Intelligent model router that selects optimal Anthropic models based on
    complexity analysis and budget constraints.
    
    Features:
    - Dynamic model selection based on query complexity
    - Budget tracking and consumption monitoring
    - Cost optimization with accuracy trade-offs
    - Historical usage analysis
    """
    
    def __init__(self, db_path: str = "data/cache.db", daily_budget: float = 10.0):
        """
        Initialize the cost-aware router.
        
        Args:
            db_path: Path to SQLite database for budget tracking
            daily_budget: Maximum daily budget in USD
        """
        self.db_path = db_path
        self.daily_budget = daily_budget
        
        # Define available models with their characteristics
        self.models = {
            ModelTier.CHEAP.value: ModelConfig(
                model='claude-3-haiku-20240307',
                cost=0.00025,  # $0.25 per 1K tokens (input)
                accuracy=0.7,
                max_tokens=4000,
                temperature=0.7
            ),
            ModelTier.PREMIUM.value: ModelConfig(
                model='claude-3-opus-20240229',
                cost=0.015,  # $15 per 1K tokens (input)
                accuracy=0.9,
                max_tokens=4000,
                temperature=0.7
            )
        }
        
        self._init_database()
        
        # Cache for performance optimization
        self._daily_consumption_cache = None
        self._cache_timestamp = None
        
        logger.info(f"CostAwareRouter initialized with daily budget: ${daily_budget}")
    
    def _init_database(self) -> None:
        """Initialize database tables for budget tracking"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Budget consumption tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS budget_consumption (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        model_used TEXT NOT NULL,
                        tokens_consumed INTEGER NOT NULL,
                        cost_incurred REAL NOT NULL,
                        query_hash TEXT NOT NULL,
                        complexity_score REAL NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Model performance tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        complexity_range TEXT NOT NULL,
                        accuracy_score REAL,
                        avg_response_time REAL,
                        total_requests INTEGER DEFAULT 1,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(model_name, complexity_range) ON CONFLICT REPLACE
                    )
                """)
                
                # Cost optimization settings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS router_settings (
                        setting_key TEXT PRIMARY KEY,
                        setting_value TEXT NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_budget_timestamp 
                    ON budget_consumption(timestamp)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_budget_model 
                    ON budget_consumption(model_used)
                """)
                
                conn.commit()
                logger.debug("Budget tracking database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize budget tracking database: {str(e)}")
            raise
    
    def route(self, complexity_score: float, max_cost: Optional[float] = None) -> Dict[str, Any]:
        """
        Select optimal model based on complexity and budget constraints.
        
        Args:
            complexity_score: Query complexity score (0-1)
            max_cost: Maximum acceptable cost for this query (USD)
            
        Returns:
            Dict containing selected model configuration and routing decision
        """
        try:
            # Get current daily consumption
            daily_consumption = self._get_daily_consumption()
            remaining_budget = self.daily_budget - daily_consumption
            
            # Apply max_cost constraint if provided
            effective_max_cost = max_cost if max_cost is not None else remaining_budget
            effective_max_cost = min(effective_max_cost, remaining_budget)
            
            # Determine optimal model based on complexity and budget
            selected_tier = self._select_optimal_model(
                complexity_score, 
                effective_max_cost
            )
            
            model_config = self.models[selected_tier]
            
            # Estimate token consumption (rough estimate based on complexity)
            estimated_tokens = self._estimate_tokens(complexity_score)
            estimated_cost = (estimated_tokens / 1000) * model_config.cost
            
            routing_decision = {
                'selected_model': model_config.model,
                'model_tier': selected_tier,
                'cost_per_1k_tokens': model_config.cost,
                'expected_accuracy': model_config.accuracy,
                'estimated_tokens': estimated_tokens,
                'estimated_cost': estimated_cost,
                'max_tokens': model_config.max_tokens,
                'temperature': model_config.temperature,
                'complexity_score': complexity_score,
                'daily_budget_used': daily_consumption,
                'remaining_budget': remaining_budget,
                'routing_reason': self._get_routing_reason(
                    complexity_score, effective_max_cost, selected_tier
                )
            }
            
            logger.info(
                f"Selected {selected_tier} model ({model_config.model}) for "
                f"complexity {complexity_score:.3f}, estimated cost: ${estimated_cost:.4f}"
            )
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Model routing failed: {str(e)}")
            # Fallback to cheap model
            return self._get_fallback_routing()
    
    def _select_optimal_model(self, complexity_score: float, max_cost: float) -> str:
        """
        Select the optimal model tier based on complexity and cost constraints.
        
        Args:
            complexity_score: Query complexity (0-1)
            max_cost: Maximum acceptable cost
            
        Returns:
            Selected model tier
        """
        # Calculate cost-benefit ratio for each model
        cheap_config = self.models[ModelTier.CHEAP.value]
        premium_config = self.models[ModelTier.PREMIUM.value]
        
        # Estimate tokens for cost calculation
        estimated_tokens = self._estimate_tokens(complexity_score)
        
        cheap_cost = (estimated_tokens / 1000) * cheap_config.cost
        premium_cost = (estimated_tokens / 1000) * premium_config.cost
        
        # Check budget constraints
        if premium_cost > max_cost:
            if cheap_cost <= max_cost:
                return ModelTier.CHEAP.value
            else:
                # Even cheap model exceeds budget - return cheap with warning
                logger.warning(
                    f"Query cost (${cheap_cost:.4f}) exceeds max budget (${max_cost:.4f})"
                )
                return ModelTier.CHEAP.value
        
        # Both models are within budget - select based on complexity and value
        complexity_threshold = self._get_complexity_threshold()
        
        if complexity_score >= complexity_threshold:
            # High complexity - use premium model for better accuracy
            return ModelTier.PREMIUM.value
        else:
            # Low complexity - cheap model is sufficient
            return ModelTier.CHEAP.value
    
    def _get_complexity_threshold(self) -> float:
        """
        Get the complexity threshold for model selection.
        This can be dynamically adjusted based on performance data.
        
        Returns:
            Complexity threshold (0-1)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT setting_value FROM router_settings 
                    WHERE setting_key = 'complexity_threshold'
                """)
                
                result = cursor.fetchone()
                if result:
                    return float(result[0])
                
        except Exception as e:
            logger.warning(f"Failed to get complexity threshold from DB: {str(e)}")
        
        # Default threshold
        return 0.6
    
    def _estimate_tokens(self, complexity_score: float) -> int:
        """
        Estimate token consumption based on complexity score.
        
        Args:
            complexity_score: Query complexity (0-1)
            
        Returns:
            Estimated token count
        """
        # Base token count + complexity-based scaling
        base_tokens = 200
        complexity_tokens = int(complexity_score * 1500)  # Max 1500 additional tokens
        
        return base_tokens + complexity_tokens
    
    def _get_routing_reason(self, complexity_score: float, max_cost: float, selected_tier: str) -> str:
        """Generate human-readable explanation for routing decision"""
        cheap_config = self.models[ModelTier.CHEAP.value]
        premium_config = self.models[ModelTier.PREMIUM.value]
        
        estimated_tokens = self._estimate_tokens(complexity_score)
        cheap_cost = (estimated_tokens / 1000) * cheap_config.cost
        premium_cost = (estimated_tokens / 1000) * premium_config.cost
        
        if selected_tier == ModelTier.PREMIUM.value:
            if premium_cost > max_cost:
                return f"Premium model selected despite budget constraint (${premium_cost:.4f} > ${max_cost:.4f})"
            else:
                return f"Premium model selected for high complexity ({complexity_score:.3f}) and sufficient budget"
        else:
            if premium_cost > max_cost:
                return f"Cheap model selected due to budget constraint (premium would cost ${premium_cost:.4f})"
            else:
                return f"Cheap model selected for low complexity ({complexity_score:.3f})"
    
    def _get_daily_consumption(self) -> float:
        """
        Get total daily budget consumption with caching.
        
        Returns:
            Total consumption in USD for current day
        """
        # Check cache validity (5 minute cache)
        current_time = datetime.now()
        if (self._daily_consumption_cache is not None and 
            self._cache_timestamp is not None and
            (current_time - self._cache_timestamp).total_seconds() < 300):
            return self._daily_consumption_cache
        
        try:
            today = datetime.now().date()
            today_start = datetime.combine(today, datetime.min.time())
            today_end = datetime.combine(today, datetime.max.time())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT COALESCE(SUM(cost_incurred), 0.0) 
                    FROM budget_consumption 
                    WHERE timestamp BETWEEN ? AND ?
                """, (today_start, today_end))
                
                consumption = cursor.fetchone()[0]
                
                # Update cache
                self._daily_consumption_cache = float(consumption)
                self._cache_timestamp = current_time
                
                return self._daily_consumption_cache
                
        except Exception as e:
            logger.error(f"Failed to get daily consumption: {str(e)}")
            return 0.0
    
    def track_usage(self, model_used: str, tokens_consumed: int, 
                   query_hash: str, complexity_score: float) -> None:
        """
        Track model usage for budget monitoring.
        
        Args:
            model_used: Name of the model that was used
            tokens_consumed: Actual tokens consumed
            query_hash: Hash of the query for deduplication
            complexity_score: Complexity score of the query
        """
        try:
            # Find model config to calculate cost
            model_config = None
            for tier, config in self.models.items():
                if config.model == model_used:
                    model_config = config
                    break
            
            if not model_config:
                logger.warning(f"Unknown model used: {model_used}")
                return
            
            cost_incurred = (tokens_consumed / 1000) * model_config.cost
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO budget_consumption 
                    (timestamp, model_used, tokens_consumed, cost_incurred, 
                     query_hash, complexity_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    model_used,
                    tokens_consumed,
                    cost_incurred,
                    query_hash,
                    complexity_score
                ))
                
                conn.commit()
                
                # Invalidate cache
                self._daily_consumption_cache = None
                self._cache_timestamp = None
                
                logger.debug(
                    f"Tracked usage: {model_used}, {tokens_consumed} tokens, "
                    f"${cost_incurred:.4f} cost"
                )
                
        except Exception as e:
            logger.error(f"Failed to track usage: {str(e)}")
    
    def get_budget_status(self) -> Dict[str, Any]:
        """
        Get current budget status and consumption analytics.
        
        Returns:
            Dictionary with budget status and analytics
        """
        try:
            daily_consumption = self._get_daily_consumption()
            remaining_budget = self.daily_budget - daily_consumption
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get today's usage by model
                today = datetime.now().date()
                today_start = datetime.combine(today, datetime.min.time())
                today_end = datetime.combine(today, datetime.max.time())
                
                cursor.execute("""
                    SELECT model_used, COUNT(*) as requests, SUM(cost_incurred) as total_cost,
                           SUM(tokens_consumed) as total_tokens
                    FROM budget_consumption 
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY model_used
                """, (today_start, today_end))
                
                model_usage = {}
                for row in cursor.fetchall():
                    model_usage[row[0]] = {
                        'requests': row[1],
                        'total_cost': row[2],
                        'total_tokens': row[3]
                    }
                
                # Get weekly stats
                week_start = datetime.now() - timedelta(days=7)
                cursor.execute("""
                    SELECT COUNT(*) as total_requests, SUM(cost_incurred) as total_cost
                    FROM budget_consumption 
                    WHERE timestamp >= ?
                """, (week_start,))
                
                weekly_stats = cursor.fetchone()
                
                return {
                    'daily_budget': self.daily_budget,
                    'daily_consumption': daily_consumption,
                    'remaining_budget': remaining_budget,
                    'budget_utilization': (daily_consumption / self.daily_budget) * 100,
                    'model_usage_today': model_usage,
                    'weekly_stats': {
                        'total_requests': weekly_stats[0] if weekly_stats else 0,
                        'total_cost': weekly_stats[1] if weekly_stats else 0.0
                    },
                    'available_models': {
                        tier: {
                            'model': config.model,
                            'cost_per_1k_tokens': config.cost,
                            'accuracy': config.accuracy
                        }
                        for tier, config in self.models.items()
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get budget status: {str(e)}")
            return {
                'daily_budget': self.daily_budget,
                'daily_consumption': 0.0,
                'remaining_budget': self.daily_budget,
                'error': str(e)
            }
    
    def _get_fallback_routing(self) -> Dict[str, Any]:
        """Get fallback routing decision when primary routing fails"""
        cheap_config = self.models[ModelTier.CHEAP.value]
        
        return {
            'selected_model': cheap_config.model,
            'model_tier': ModelTier.CHEAP.value,
            'cost_per_1k_tokens': cheap_config.cost,
            'expected_accuracy': cheap_config.accuracy,
            'estimated_tokens': 500,  # Conservative estimate
            'estimated_cost': 0.000125,  # 500 tokens * $0.25/1K
            'max_tokens': cheap_config.max_tokens,
            'temperature': cheap_config.temperature,
            'complexity_score': 0.5,  # Default complexity
            'daily_budget_used': 0.0,
            'remaining_budget': self.daily_budget,
            'routing_reason': 'Fallback to cheap model due to routing error',
            'is_fallback': True
        }
    
    def optimize_threshold(self, performance_data: List[Dict[str, Any]]) -> None:
        """
        Optimize complexity threshold based on performance data.
        
        Args:
            performance_data: List of performance records with complexity, 
                            accuracy, and cost data
        """
        try:
            if not performance_data:
                return
            
            # Analyze performance data to find optimal threshold
            # This is a simplified implementation - could be enhanced with ML
            
            cheap_performance = []
            premium_performance = []
            
            for record in performance_data:
                if record.get('model_tier') == ModelTier.CHEAP.value:
                    cheap_performance.append(record)
                elif record.get('model_tier') == ModelTier.PREMIUM.value:
                    premium_performance.append(record)
            
            if not cheap_performance or not premium_performance:
                logger.warning("Insufficient performance data for threshold optimization")
                return
            
            # Find threshold where premium model starts providing significant value
            # Simple heuristic: find complexity where premium accuracy improvement 
            # justifies the cost increase
            
            # This is a placeholder for more sophisticated optimization
            new_threshold = 0.6  # Default threshold
            
            # Store updated threshold
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO router_settings 
                    (setting_key, setting_value, updated_at)
                    VALUES ('complexity_threshold', ?, ?)
                """, (str(new_threshold), datetime.now()))
                
                conn.commit()
                
            logger.info(f"Updated complexity threshold to {new_threshold}")
            
        except Exception as e:
            logger.error(f"Failed to optimize threshold: {str(e)}")


# Global router instance for reuse
_router_instance = None


def get_cost_aware_router(db_path: str = "data/cache.db", 
                         daily_budget: float = 10.0) -> CostAwareRouter:
    """
    Get global cost-aware router instance.
    
    Args:
        db_path: Database path for budget tracking
        daily_budget: Daily budget limit in USD
        
    Returns:
        CostAwareRouter instance
    """
    global _router_instance
    
    if _router_instance is None:
        _router_instance = CostAwareRouter(db_path=db_path, daily_budget=daily_budget)
    
    return _router_instance