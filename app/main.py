from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from app.config import get_settings
from app.models.predictor import get_predictor
from app.utils.guardrails import SafetyValidator
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OpenSesame Predictor",
    description="AI-powered API call prediction service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Get configuration
settings = get_settings()

# Initialize components
safety_validator = SafetyValidator()

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    prompt: str
    history: Optional[List[Dict[str, Any]]] = []
    max_predictions: Optional[int] = 5
    temperature: Optional[float] = 0.7

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    metadata: Dict[str, Any]
    processing_time_ms: float

@app.get("/")
async def root():
    """Root endpoint with Phase 5 Performance Optimization information"""
    return {
        "message": "OpenSesame Predictor API is running - Phase 5 Performance Optimization", 
        "version": "v5.0-performance-optimized",
        "phase": "Phase 5 - Performance Optimization with async parallel processing",
        "features": [
            "Async parallel AI + ML + Safety predictions",
            "Performance targets: LLM <500ms, ML <100ms, Total <800ms", 
            "Embedding caching with sqlite3",
            "Pre-computed feature vectors",
            "Batch sentence-transformers operations",
            "1-hour TTL caching in data/cache.db"
        ],
        "performance_targets": {
            "llm_latency_ms": "<500",
            "ml_latency_ms": "<100", 
            "total_response_time_ms": "<800",
            "caching_enabled": True,
            "async_processing": True
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive Phase 4 Safety Layer health check endpoint"""
    try:
        predictor = await get_predictor()
        health = await predictor.health_check()
        return health
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "opensesame-predictor-safety",
            "version": "v4.0-safety-layer",
            "error": str(e)
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_api_calls(request: PredictionRequest):
    """
    Phase 5 Performance-Optimized prediction endpoint with async parallel processing
    Returns safety-filtered, ML-ranked API calls with <800ms total latency
    Target: LLM <500ms, ML <100ms, Total <800ms
    """
    try:
        # Start timing for performance monitoring
        request_start_time = time.time()
        
        # Validate input safety (fast validation)
        if not safety_validator.validate_input(request.prompt):
            raise HTTPException(status_code=400, detail="Input failed safety validation")
        
        # Log request (without sensitive data)
        logger.info(f"Processing Phase 5 Performance prediction request with prompt length: {len(request.prompt)}")
        
        # Generate predictions using the optimized Predictor (Async AI + ML + Safety)
        predictor = await get_predictor()
        
        # Generate performance-optimized, safety-filtered, ML-ranked predictions
        result = await predictor.predict(
            prompt=request.prompt,
            history=request.history,
            max_predictions=request.max_predictions,
            temperature=request.temperature,
            use_ml_ranking=True
        )
        
        # Calculate total request processing time
        total_request_time = (time.time() - request_start_time) * 1000
        
        # Add performance metrics to metadata
        if "metadata" in result:
            result["metadata"]["request_processing_time_ms"] = total_request_time
            result["metadata"]["performance_optimized"] = True
            result["metadata"]["phase"] = "Phase 5 - Performance Optimization"
        
        # Log performance metrics
        logger.info(f"Phase 5 request completed in {total_request_time:.2f}ms (target: <800ms)")
        
        return PredictionResponse(
            predictions=result["predictions"],
            confidence_scores=result["confidence_scores"],
            metadata=result.get("metadata", {}),
            processing_time_ms=result["processing_time_ms"]
        )
        
    except Exception as e:
        logger.error(f"Phase 5 Performance prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance-optimized prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive Phase 4 Safety Layer system metrics and performance stats"""
    try:
        predictor = await get_predictor()
        metrics = await predictor.get_metrics()
        
        return {
            **metrics,
            "uptime": "placeholder_uptime",
            "version": "v4.0-safety-layer",
            "phase": "Phase 4 - Safety Layer"
        }
    except Exception as e:
        logger.error(f"Phase 4 metrics retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Safety metrics unavailable")

@app.post("/train")
async def train_ml_model():
    """Train or retrain the ML ranking model with available data"""
    try:
        logger.info("Received request to train ML model")
        
        predictor = await get_predictor()
        training_stats = await predictor.train_ml_model()
        
        return {
            "status": "training_completed",
            "training_stats": training_stats,
            "timestamp": datetime.now().isoformat(),
            "version": "v4.0-safety-layer"
        }
    except Exception as e:
        logger.error(f"ML model training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML training failed: {str(e)}")

@app.post("/generate-data")
async def generate_training_data():
    """Generate synthetic training data for ML model"""
    try:
        from data.synthetic_generator import generate_ml_training_data
        
        logger.info("Generating synthetic training data...")
        stats = await generate_ml_training_data(num_sequences=10000)
        
        return {
            "status": "data_generated",
            "generation_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Data generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )