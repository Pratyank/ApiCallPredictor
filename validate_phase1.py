#!/usr/bin/env python3
"""
Phase 1 Validation Script for OpenSesame Predictor
Comprehensive testing to ensure all components work before Phase 2
"""

import asyncio
import json
import sys
import os
import time
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🔍 Phase 1 Validation Starting...")
print(f"Project root: {project_root}")

class Phase1Validator:
    def __init__(self):
        self.results = {}
        self.total_checks = 0
        self.passed_checks = 0
        
    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        self.total_checks += 1
        if passed:
            self.passed_checks += 1
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name}: {details}")
        
        self.results[test_name] = {
            "passed": passed,
            "details": details
        }
    
    def test_imports(self):
        """Test that all core modules can be imported"""
        print("\n📦 Testing Module Imports...")
        
        try:
            from app.config import get_settings, Settings
            self.log_result("Import app.config", True)
        except Exception as e:
            self.log_result("Import app.config", False, str(e))
        
        try:
            from app.models.predictor import PredictionEngine
            self.log_result("Import PredictionEngine", True)
        except Exception as e:
            self.log_result("Import PredictionEngine", False, str(e))
        
        try:
            from app.models.ai_layer import LLMInterface
            self.log_result("Import LLMInterface", True)
        except Exception as e:
            self.log_result("Import LLMInterface", False, str(e))
        
        try:
            from app.models.ml_ranker import MLRanker
            self.log_result("Import MLRanker", True)
        except Exception as e:
            self.log_result("Import MLRanker", False, str(e))
        
        try:
            from app.utils.spec_parser import OpenAPISpecParser
            self.log_result("Import OpenAPISpecParser", True)
        except Exception as e:
            self.log_result("Import OpenAPISpecParser", False, str(e))
        
        try:
            from app.utils.guardrails import SafetyValidator
            self.log_result("Import SafetyValidator", True)
        except Exception as e:
            self.log_result("Import SafetyValidator", False, str(e))
        
        try:
            from app.utils.feature_eng import FeatureExtractor
            self.log_result("Import FeatureExtractor", True)
        except Exception as e:
            self.log_result("Import FeatureExtractor", False, str(e))
    
    def test_configuration(self):
        """Test configuration system"""
        print("\n⚙️ Testing Configuration...")
        
        try:
            from app.config import get_settings
            settings = get_settings()
            
            # Test basic settings
            assert hasattr(settings, 'app_name')
            assert hasattr(settings, 'api_port')
            assert hasattr(settings, 'database_url')
            assert settings.api_port == 8000
            
            self.log_result("Configuration loading", True)
        except Exception as e:
            self.log_result("Configuration loading", False, str(e))
        
        try:
            from app.config import DatabaseManager
            db_manager = DatabaseManager(":memory:")  # In-memory for testing
            self.log_result("Database manager initialization", True)
        except Exception as e:
            self.log_result("Database manager initialization", False, str(e))
    
    async def test_prediction_pipeline(self):
        """Test the core prediction pipeline"""
        print("\n🧠 Testing Prediction Pipeline...")
        
        try:
            from app.models.predictor import PredictionEngine
            
            engine = PredictionEngine()
            
            # Test basic prediction
            test_prompt = "I need to get user information"
            result = await engine.predict(
                prompt=test_prompt,
                max_predictions=3,
                temperature=0.7
            )
            
            # Validate result structure
            assert "predictions" in result
            assert "confidence_scores" in result
            assert "processing_time_ms" in result
            assert isinstance(result["predictions"], list)
            assert isinstance(result["confidence_scores"], list)
            assert len(result["predictions"]) <= 3
            
            self.log_result("Prediction pipeline", True)
        except Exception as e:
            self.log_result("Prediction pipeline", False, str(e))
    
    async def test_llm_interface(self):
        """Test LLM interface (mock implementation)"""
        print("\n🤖 Testing LLM Interface...")
        
        try:
            from app.models.ai_layer import LLMInterface
            
            llm = LLMInterface()
            
            # Test prediction generation
            predictions = await llm.generate_api_predictions(
                prompt="Get user data",
                max_predictions=2
            )
            
            assert isinstance(predictions, list)
            assert len(predictions) <= 2
            
            if predictions:
                pred = predictions[0]
                assert "api_call" in pred
                assert "method" in pred
                assert "description" in pred
            
            self.log_result("LLM interface", True)
        except Exception as e:
            self.log_result("LLM interface", False, str(e))
    
    async def test_ml_ranker(self):
        """Test ML ranking system"""
        print("\n📊 Testing ML Ranker...")
        
        try:
            from app.models.ml_ranker import MLRanker
            
            ranker = MLRanker()
            
            # Test ranking
            mock_predictions = [
                {
                    "api_call": "/api/users",
                    "method": "GET",
                    "description": "Get users",
                    "confidence": 0.8
                },
                {
                    "api_call": "/api/posts",
                    "method": "GET", 
                    "description": "Get posts",
                    "confidence": 0.6
                }
            ]
            
            mock_features = {"intent_create": 0.2, "intent_retrieve": 0.8}
            
            ranked = await ranker.rank_predictions(
                predictions=mock_predictions,
                features=mock_features,
                user_context={"prompt": "get users"}
            )
            
            assert "predictions" in ranked
            assert "confidence_scores" in ranked
            assert isinstance(ranked["predictions"], list)
            
            self.log_result("ML ranker", True)
        except Exception as e:
            self.log_result("ML ranker", False, str(e))
    
    def test_guardrails(self):
        """Test security guardrails"""
        print("\n🛡️ Testing Security Guardrails...")
        
        try:
            from app.utils.guardrails import SafetyValidator
            
            validator = SafetyValidator()
            
            # Test safe input
            safe_prompt = "I need to get user information"
            assert validator.validate_input(safe_prompt) == True
            
            # Test malicious input
            malicious_prompt = "'; DROP TABLE users; --"
            assert validator.validate_input(malicious_prompt) == False
            
            # Test output validation
            test_predictions = [
                {
                    "api_call": "/api/users",
                    "method": "GET",
                    "description": "Get users"
                }
            ]
            
            filtered, warnings = validator.validate_output(test_predictions)
            assert isinstance(filtered, list)
            assert isinstance(warnings, list)
            
            self.log_result("Security guardrails", True)
        except Exception as e:
            self.log_result("Security guardrails", False, str(e))
    
    async def test_feature_extractor(self):
        """Test feature extraction"""
        print("\n🔧 Testing Feature Extraction...")
        
        try:
            from app.utils.feature_eng import FeatureExtractor
            
            extractor = FeatureExtractor()
            
            features = await extractor.extract_features(
                prompt="I want to create a new user account",
                history=[{"api_call": "/api/login", "method": "POST"}]
            )
            
            assert isinstance(features, dict)
            assert "prompt_length" in features
            assert "word_count" in features
            assert "feature_names" in features
            
            self.log_result("Feature extraction", True)
        except Exception as e:
            self.log_result("Feature extraction", False, str(e))
    
    async def test_spec_parser(self):
        """Test OpenAPI spec parser"""
        print("\n📋 Testing OpenAPI Spec Parser...")
        
        try:
            from app.utils.spec_parser import OpenAPISpecParser
            
            async with OpenAPISpecParser() as parser:
                # Test cache stats
                stats = await parser.get_cache_stats()
                assert isinstance(stats, dict)
                
                # Test cleanup
                cleanup_result = await parser.cleanup_expired_cache()
                assert isinstance(cleanup_result, dict)
            
            self.log_result("OpenAPI spec parser", True)
        except Exception as e:
            self.log_result("OpenAPI spec parser", False, str(e))
    
    def test_synthetic_data_generator(self):
        """Test synthetic data generation"""
        print("\n🎲 Testing Synthetic Data Generator...")
        
        try:
            from data.synthetic_generator import SyntheticDataGenerator
            
            generator = SyntheticDataGenerator()
            
            # Generate small test dataset
            dataset = generator.generate_training_dataset(num_samples=5)
            
            assert isinstance(dataset, list)
            assert len(dataset) == 5
            
            if dataset:
                sample = dataset[0]
                assert "prompt" in sample
                assert "expected_prediction" in sample
                assert "quality_label" in sample
            
            self.log_result("Synthetic data generator", True)
        except Exception as e:
            self.log_result("Synthetic data generator", False, str(e))
    
    def test_fastapi_app(self):
        """Test FastAPI application creation"""
        print("\n🚀 Testing FastAPI Application...")
        
        try:
            from app.main import app
            
            # Test app creation
            assert app is not None
            assert hasattr(app, 'routes')
            
            # Check for required routes
            route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
            
            assert "/" in route_paths
            assert "/health" in route_paths
            assert "/predict" in route_paths
            assert "/metrics" in route_paths
            
            self.log_result("FastAPI application", True)
        except Exception as e:
            self.log_result("FastAPI application", False, str(e))
    
    def test_docker_files(self):
        """Test Docker configuration files"""
        print("\n🐳 Testing Docker Configuration...")
        
        try:
            # Check Dockerfile exists and has key components
            dockerfile_path = project_root / "Dockerfile"
            assert dockerfile_path.exists()
            
            dockerfile_content = dockerfile_path.read_text()
            assert "FROM python:" in dockerfile_content
            assert "COPY requirements.txt" in dockerfile_content
            assert "pip install" in dockerfile_content
            assert "uvicorn" in dockerfile_content
            
            self.log_result("Dockerfile validation", True)
        except Exception as e:
            self.log_result("Dockerfile validation", False, str(e))
        
        try:
            # Check docker-compose.yml
            compose_path = project_root / "docker-compose.yml"
            assert compose_path.exists()
            
            compose_content = compose_path.read_text()
            assert "opensesame-predictor" in compose_content
            assert "8000:8000" in compose_content
            assert "cpus: '2.0'" in compose_content
            assert "memory: 4G" in compose_content
            
            self.log_result("Docker Compose validation", True)
        except Exception as e:
            self.log_result("Docker Compose validation", False, str(e))
    
    def test_requirements(self):
        """Test requirements.txt"""
        print("\n📋 Testing Requirements...")
        
        try:
            requirements_path = project_root / "requirements.txt"
            assert requirements_path.exists()
            
            requirements_content = requirements_path.read_text()
            assert "fastapi" in requirements_content.lower()
            assert "uvicorn" in requirements_content.lower()
            assert "pydantic" in requirements_content.lower()
            assert "aiohttp" in requirements_content.lower()
            
            self.log_result("Requirements validation", True)
        except Exception as e:
            self.log_result("Requirements validation", False, str(e))
    
    async def run_all_tests(self):
        """Run all validation tests"""
        print("🧪 Running Phase 1 Validation Tests...\n")
        
        # Import and basic structure tests
        self.test_imports()
        self.test_configuration()
        
        # Core functionality tests
        await self.test_prediction_pipeline()
        await self.test_llm_interface()
        await self.test_ml_ranker()
        self.test_guardrails()
        await self.test_feature_extractor()
        await self.test_spec_parser()
        
        # Data and application tests
        self.test_synthetic_data_generator()
        self.test_fastapi_app()
        
        # Deployment tests
        self.test_docker_files()
        self.test_requirements()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("📊 PHASE 1 VALIDATION SUMMARY")
        print("="*60)
        
        print(f"Total Tests: {self.total_checks}")
        print(f"Passed: {self.passed_checks}")
        print(f"Failed: {self.total_checks - self.passed_checks}")
        
        success_rate = (self.passed_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("\n🎉 PHASE 1 READY FOR PHASE 2!")
            print("✅ All critical components working")
        elif success_rate >= 75:
            print("\n⚠️  PHASE 1 MOSTLY READY")
            print("🔧 Some components need attention")
        else:
            print("\n❌ PHASE 1 NEEDS WORK") 
            print("🚨 Critical issues must be fixed")
        
        # Print failed tests
        failed_tests = [name for name, result in self.results.items() if not result["passed"]]
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for test in failed_tests:
                details = self.results[test]["details"]
                print(f"  • {test}: {details}")
        
        print("\n" + "="*60)

async def main():
    """Main validation function"""
    validator = Phase1Validator()
    await validator.run_all_tests()
    
    return validator.passed_checks / validator.total_checks >= 0.9

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)