import os
from functools import lru_cache
from pydantic_settings import BaseSettings
import sqlite3
from typing import Dict, Any
import json

class Settings(BaseSettings):
    """Configuration settings for OpenSesame Predictor"""
    
    # Application settings
    app_name: str = "OpenSesame Predictor"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    # Database settings
    database_url: str = "sqlite:///./opensesame.db"
    database_pool_size: int = 5
    
    # Cache settings
    cache_ttl_seconds: int = 3600  # 1 hour TTL
    cache_max_size: int = 1000
    openapi_cache_ttl: int = 3600  # 1 hour for OpenAPI spec caching
    
    # ML Model settings
    model_path: str = "./models"
    max_model_memory_mb: int = 512
    prediction_timeout_seconds: int = 30
    
    # LLM settings
    llm_provider: str = "placeholder"
    llm_model: str = "gpt-3.5-turbo"
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.7
    
    # Feature engineering settings
    max_history_length: int = 100
    feature_vector_size: int = 256
    
    # Safety settings
    enable_guardrails: bool = True
    max_prompt_length: int = 4096
    blocked_patterns: list = []
    
    # Performance settings
    worker_count: int = 2
    max_concurrent_requests: int = 100
    request_timeout_seconds: int = 60
    
    class Config:
        env_file = ".env"
        env_prefix = "OPENSESAME_"

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

class DatabaseManager:
    """SQLite database manager for caching"""
    
    def __init__(self, db_path: str = "./opensesame.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ttl INTEGER DEFAULT 3600
            )
        ''')
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create predictions log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash TEXT NOT NULL,
                prediction_count INTEGER,
                processing_time_ms REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM api_cache')
        total_entries = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM api_cache WHERE datetime(timestamp, "+" || ttl || " seconds") > datetime("now")')
        valid_entries = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": total_entries - valid_entries,
            "hit_rate": 0.0  # Placeholder for actual hit rate calculation
        }

# Global database manager instance
db_manager = DatabaseManager()

# Environment-specific configurations
DEVELOPMENT_CONFIG = {
    "debug": True,
    "log_level": "DEBUG",
    "reload": True
}

PRODUCTION_CONFIG = {
    "debug": False,
    "log_level": "INFO",
    "reload": False,
    "workers": 4
}

def get_environment_config() -> Dict[str, Any]:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return PRODUCTION_CONFIG
    else:
        return DEVELOPMENT_CONFIG
