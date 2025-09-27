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
    """Root endpoint with Phase 3 ML Layer information"""
    return {
        "message": "OpenSesame Predictor API is running - Phase 3 ML Layer", 
        "version": "v3.0-ml-layer",
        "phase": "Phase 3 - ML Layer with LightGBM ranking",
        "features": ["AI + ML hybrid predictions", "LightGBM ranking", "Feature engineering", "Markov chain training data"]
    }

@app.get("/health")
async def health_check():
    """Comprehensive Phase 3 ML health check endpoint"""
    try:
        predictor = await get_predictor()
        health = await predictor.health_check()
        return health
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "service": "opensesame-predictor-ml",
            "version": "v3.0-ml-layer",
            "error": str(e)
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_api_calls(request: PredictionRequest):
    """
    Phase 3 ML prediction endpoint that uses integrated AI + ML ranking
    Returns ML-ranked API calls with confidence scores and enhanced metadata
    """
    try:
        # Validate input safety
        if not safety_validator.validate_input(request.prompt):
            raise HTTPException(status_code=400, detail="Input failed safety validation")
        
        # Log request (without sensitive data)
        logger.info(f"Processing Phase 3 ML prediction request with prompt length: {len(request.prompt)}")
        
        # Generate predictions using the integrated Predictor (AI + ML)
        predictor = await get_predictor()
        
        # Generate ML-ranked predictions
        result = await predictor.predict(
            prompt=request.prompt,
            history=request.history,
            max_predictions=request.max_predictions,
            temperature=request.temperature,
            use_ml_ranking=True
        )
        
        return PredictionResponse(
            predictions=result["predictions"],
            confidence_scores=result["confidence_scores"],
            metadata=result.get("metadata", {}),
            processing_time_ms=result["processing_time_ms"]
        )
        
    except Exception as e:
        logger.error(f"Phase 3 ML prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get comprehensive Phase 3 ML system metrics and performance stats"""
    try:
        predictor = await get_predictor()
        metrics = await predictor.get_metrics()
        
        return {
            **metrics,
            "uptime": "placeholder_uptime",
            "version": "v3.0-ml-layer",
            "phase": "Phase 3 - ML Layer"
        }
    except Exception as e:
        logger.error(f"Phase 3 metrics retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="ML metrics unavailable")

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
            "timestamp": datetime.now().isoformat()
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