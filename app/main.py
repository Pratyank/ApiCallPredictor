from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from app.config import get_settings
from app.models.ai_layer import get_ai_layer
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
    """Health check endpoint"""
    return {"message": "OpenSesame Predictor API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "service": "opensesame-predictor",
        "version": "1.0.0",
        "components": {
            "prediction_engine": "operational",
            "safety_validator": "operational",
            "cache": "operational"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_api_calls(request: PredictionRequest):
    """
    Main prediction endpoint that accepts JSON payload with user prompt and history
    Returns predicted API calls with confidence scores
    """
    try:
        # Validate input safety
        if not safety_validator.validate_input(request.prompt):
            raise HTTPException(status_code=400, detail="Input failed safety validation")
        
        # Log request (without sensitive data)
        logger.info(f"Processing prediction request with prompt length: {len(request.prompt)}")
        
        # Generate predictions using the new AI layer
        ai_layer = await get_ai_layer()
        
        # Generate predictions with enhanced AI capabilities
        predictions = await ai_layer.generate_predictions(
            prompt=request.prompt,
            history=request.history,
            k=request.max_predictions,
            temperature=request.temperature
        )
        
        # Extract confidence scores
        confidence_scores = [pred.get('confidence', 0.0) for pred in predictions]
        
        # Calculate processing time (approximated from predictions)
        processing_time_ms = predictions[0].get('processing_time_ms', 0.0) if predictions else 0.0
        
        result = {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'processing_time_ms': processing_time_ms,
            'model_version': 'v2.0-ai-layer',
            'timestamp': datetime.now().isoformat(),
            'ai_provider': predictions[0].get('ai_provider', 'unknown') if predictions else 'unknown'
        }
        
        return PredictionResponse(
            predictions=result["predictions"],
            confidence_scores=result["confidence_scores"],
            metadata={
                "model_version": result.get("model_version", "v1.0"),
                "timestamp": result.get("timestamp"),
                "processing_method": "hybrid_ml_llm"
            },
            processing_time_ms=result["processing_time_ms"]
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get system metrics and performance stats"""
    try:
        ai_layer = await get_ai_layer()
        ai_metrics = await ai_layer.get_status()
        
        # Get cache stats from spec parser
        cache_stats = await ai_layer.spec_parser.get_cache_stats()
        
        return {
            "ai_layer_metrics": ai_metrics,
            "cache_stats": cache_stats,
            "uptime": "placeholder_uptime",
            "version": "v2.0-ai-layer"
        }
    except Exception as e:
        logger.error(f"Metrics retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )