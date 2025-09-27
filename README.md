# OpenSesame Predictor

**AI-Powered API Call Prediction Service**

OpenSesame Predictor is a sophisticated FastAPI-based service that uses advanced machine learning and large language models to predict the most relevant API calls based on natural language user prompts and conversation history.

## 🚀 Features

### Core Capabilities
- **Advanced AI Prediction**: Phase 2 AI Layer with Anthropic Claude integration and OpenAI fallback
- **Semantic Understanding**: Sentence-transformers for intelligent endpoint matching
- **Context-Aware Generation**: History-based filtering with k+buffer candidate selection
- **Multi-Modal Architecture**: Hybrid AI/ML approach with confidence scoring
- **Real-time Processing**: Sub-second response times with intelligent caching
- **Safety & Security**: Comprehensive input validation and guardrails
- **Scalable Design**: Docker containerization with resource optimization

### Technical Highlights (Phase 2)
- **AI Layer**: Anthropic Claude 3 Haiku integration with OpenAI GPT-3.5 fallback
- **Semantic Similarity**: sentence-transformers (all-MiniLM-L6-v2) for endpoint matching
- **Smart Caching**: Enhanced SQLite caching with parsed endpoint storage in data/cache.db
- **Context Processing**: Recent event analysis with workflow pattern recognition
- **Candidate Generation**: k+buffer logic for improved prediction quality
- **FastAPI Framework**: High-performance async API with automatic documentation
- **Feature Engineering**: Advanced text analysis and pattern recognition
- **OpenAPI Integration**: Automatic spec parsing and endpoint extraction with semantic indexing

## 📋 Project Structure

```
opensesame-predictor/
├── app/                          # Main application code
│   ├── main.py                   # FastAPI app with /predict endpoint
│   ├── config.py                 # Configuration and settings
│   ├── models/                   # Core prediction models
│   │   ├── predictor.py          # Main prediction orchestrator
│   │   ├── ai_layer.py           # LLM integration layer
│   │   └── ml_ranker.py          # ML-based prediction ranking
│   └── utils/                    # Utility modules
│       ├── spec_parser.py        # OpenAPI spec fetching (1-hour TTL)
│       ├── guardrails.py         # Safety validation system
│       └── feature_eng.py        # Feature extraction for ML
├── data/                         # Training and synthetic data
│   ├── synthetic_generator.py    # Training data generation
│   └── training_data/            # Generated training datasets
├── tests/                        # Comprehensive test suite
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Multi-stage container build
├── docker-compose.yml           # Production deployment
└── README.md                     # This file
```

## 🛠 Installation & Setup

### Local Development

1. **Clone and Setup**:
```bash
git clone <repository-url>
cd opensesame-predictor
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure AI API Keys** (Phase 2):
```bash
# Required: Set Anthropic API key for primary AI functionality
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"

# Optional: Set OpenAI API key for enhanced fallback
export OPENAI_API_KEY="your-openai-api-key-here"
```

**Getting API Keys:**
- **Anthropic**: Visit [console.anthropic.com](https://console.anthropic.com) to get your Claude API key
- **OpenAI**: Visit [platform.openai.com](https://platform.openai.com/api-keys) to get your OpenAI API key

**Note**: The system will work with just the Anthropic key, or even without any keys (using rule-based fallback), but Anthropic key is recommended for best AI performance.

4. **Run Development Server**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **Build and Run** (with AI API keys):
```bash
# Create .env file with your API keys
echo "ANTHROPIC_API_KEY=your-anthropic-api-key-here" > .env
echo "OPENAI_API_KEY=your-openai-api-key-here" >> .env

docker-compose up --build
```

2. **Production with Monitoring**:
```bash
docker-compose --profile production --profile monitoring up
```

## 📡 API Endpoints

### Core Prediction Endpoint
```http
POST /predict
Content-Type: application/json

{
  "prompt": "I need to get user information for authentication",
  "history": [
    {"api_call": "/api/auth/login", "method": "POST", "timestamp": "2024-01-01T10:00:00Z"}
  ],
  "max_predictions": 5,
  "temperature": 0.7
}
```

**Response**:
```json
{
  "predictions": [
    {
      "api_call": "GET /api/users/{id}",
      "method": "GET",
      "description": "Retrieve user information by ID",
      "parameters": {"id": "user_id", "include": "profile,permissions"},
      "confidence": 0.89,
      "rank": 1
    }
  ],
  "confidence_scores": [0.89, 0.76, 0.65],
  "metadata": {
    "model_version": "v1.0.0",
    "processing_time_ms": 245,
    "processing_method": "hybrid_ml_llm"
  },
  "processing_time_ms": 245
}
```

### Additional Endpoints
- `GET /` - Service health check
- `GET /health` - Detailed system health
- `GET /metrics` - Performance metrics
- `GET /docs` - Interactive API documentation

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENSESAME_API_HOST` | `0.0.0.0` | API server host |
| `OPENSESAME_API_PORT` | `8000` | API server port |
| `OPENSESAME_DATABASE_URL` | `sqlite:///./opensesame.db` | Database connection |
| `OPENSESAME_CACHE_TTL_SECONDS` | `3600` | Cache TTL (1 hour) |
| `OPENSESAME_MAX_PROMPT_LENGTH` | `4096` | Maximum prompt length |
| `OPENSESAME_ENABLE_GUARDRAILS` | `true` | Enable safety validation |

### AI Layer Configuration (Phase 2)

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic Claude API key for primary AI predictions |
| `OPENAI_API_KEY` | *(optional)* | OpenAI API key for fallback predictions |

### ML Model Settings
- **Feature Vector Size**: 256 dimensions
- **Max History Length**: 100 events
- **Prediction Timeout**: 30 seconds
- **Cache TTL**: 1 hour (3600 seconds)

## 🧠 Architecture Overview

### Phase 2 AI Layer Pipeline
1. **Input Validation**: Safety checks and guardrails
2. **Context Analysis**: Recent event extraction and normalization
3. **Endpoint Retrieval**: Cached endpoint access from data/cache.db
4. **Smart Filtering**: 
   - Pattern-based filtering using recent API call history
   - Semantic similarity filtering using sentence-transformers
5. **AI Generation**: 
   - Anthropic Claude 3 Haiku for primary predictions
   - OpenAI GPT-3.5 Turbo fallback with function calling
   - Rule-based fallback for offline operation
6. **Semantic Enhancement**: Confidence score enhancement using embedding similarity
7. **Candidate Selection**: Top-k selection with uniqueness guarantees
8. **Result Synthesis**: Formatted predictions with comprehensive metadata

### Enhanced Caching Strategy
- **Endpoint Cache**: Parsed OpenAPI endpoints stored in data/cache.db with semantic indexing
- **Specification Cache**: 1-hour TTL caching for raw OpenAPI specs
- **Embedding Cache**: Computed embeddings for frequent endpoints (future enhancement)
- **Cache Invalidation**: Intelligent TTL-based expiration with cleanup utilities

### AI Layer Features
- **Anthropic Integration**: Claude 3 Haiku for cost-effective, fast predictions
- **OpenAI Fallback**: GPT-3.5 Turbo with structured function calling
- **Semantic Matching**: Context-aware endpoint filtering using sentence similarity
- **History Analysis**: Recent event pattern recognition and workflow understanding
- **Multi-Provider Support**: Automatic fallback between AI providers
- **Offline Resilience**: Rule-based predictions when AI services unavailable

### Security Features
- **Input Sanitization**: SQL injection and XSS prevention
- **Rate Limiting**: Per-user request throttling
- **Content Filtering**: Inappropriate content detection
- **Parameter Validation**: Type checking and bounds enforcement
- **API Key Security**: Environment-based credential management

## 📊 Performance Targets

### Response Times (Phase 2)
- **P50 Response Time**: < 800ms (including AI processing)
- **P95 Response Time**: < 1500ms 
- **AI Layer**: < 600ms per request (Anthropic Claude)
- **Semantic Processing**: < 200ms per similarity calculation
- **Endpoint Filtering**: < 100ms per request

### Throughput
- **Single Instance**: 10+ RPS sustained
- **Scaled Deployment**: 100+ RPS with load balancing
- **Cache Hit Rate**: > 80% for common patterns

### Resource Usage
- **Memory**: < 512MB under normal load
- **CPU**: 2 cores recommended
- **Storage**: Minimal (SQLite database)

## 🔒 Security Considerations

### Input Validation
- Maximum prompt length enforcement
- Regular expression pattern blocking
- Command injection prevention
- Path traversal protection

### Rate Limiting
- 60 requests per minute per user
- Temporary blocking for violations
- Distributed rate limiting support

### Data Privacy
- No persistent storage of user prompts
- Hashed user identifiers for rate limiting
- Configurable data retention policies

## 📈 Monitoring & Observability

### Health Checks
- Application health endpoint
- Component status monitoring
- Database connectivity checks
- Cache performance metrics

### Metrics Collection
- Request/response times
- Prediction accuracy scores
- Cache hit/miss ratios
- Error rates and patterns

### Optional Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Redis**: Advanced caching layer

## 🧪 Testing

### Test Coverage
- **Unit Tests**: 85%+ coverage target
- **Integration Tests**: Full API endpoint validation
- **Performance Tests**: Load and stress testing
- **Security Tests**: Penetration testing for vulnerabilities

### Running Tests
```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests  
pytest tests/integration/ -v

# Full test suite with coverage
pytest --cov=app --cov-report=html
```

## 🚀 Deployment Options

### Development
```bash
# Local development server
uvicorn app.main:app --reload

# Development with Docker
docker-compose up
```

### Production
```bash
# Production with reverse proxy
docker-compose --profile production up

# Production with monitoring
docker-compose --profile production --profile monitoring up
```

### Scaling
- **Horizontal**: Multiple container instances behind load balancer
- **Vertical**: Increased CPU/memory allocation
- **Caching**: Redis cluster for distributed caching

## 📝 Development Notes

### Key Assumptions
1. **Stateless Predictions**: No user session persistence required
2. **English Prompts**: Primarily English language input
3. **Max 100 Events**: Limited conversation history context
4. **Speed over Accuracy**: Optimized for response time
5. **Synthetic Training**: Initial training on generated data

### Extension Points
- **AI Providers**: Easy switching between Anthropic, OpenAI, and local models
- **Semantic Models**: Pluggable sentence-transformer models for different domains
- **Cache Backends**: Redis/Memcached integration for distributed caching
- **Embedding Stores**: Vector databases for large-scale semantic search
- **Auth Systems**: JWT/OAuth2 integration
- **Monitoring**: OpenTelemetry instrumentation

### Performance Optimizations
- **Async Processing**: Non-blocking I/O throughout
- **Connection Pooling**: Efficient database connections
- **Request Batching**: Multiple predictions per request
- **Model Preloading**: Reduced cold start times

## 🤝 Contributing

### Development Setup
1. Fork and clone the repository
2. Create feature branch from `main`
3. Install development dependencies
4. Run tests before committing
5. Submit pull request with description

### Code Quality
- **Linting**: `flake8` and `black` formatting
- **Type Checking**: `mypy` static analysis
- **Testing**: Minimum 85% coverage
- **Documentation**: Docstring requirements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♀️ Support

### Documentation
- **API Docs**: `/docs` endpoint (Swagger UI)
- **ReDoc**: `/redoc` endpoint (alternative docs)

### Issues & Questions
- GitHub Issues for bug reports
- Discussions for feature requests
- Wiki for detailed guides

### Performance Tuning
- Monitor `/metrics` endpoint
- Adjust cache TTL settings
- Scale based on usage patterns
- Optimize ML model parameters

---

## 📝 Phase 2 AI Layer Implementation

### Key Enhancements

**🤖 Advanced AI Integration**
- **Primary AI**: Anthropic Claude 3 Haiku for fast, accurate predictions
- **Fallback AI**: OpenAI GPT-3.5 Turbo with structured function calling
- **Offline Mode**: Rule-based predictions when AI services unavailable

**🎯 Semantic Understanding**
- **Sentence Transformers**: all-MiniLM-L6-v2 model for endpoint similarity
- **Context Filtering**: History-aware endpoint selection
- **Embedding Enhancement**: Confidence scores improved with semantic similarity

**💾 Enhanced Caching**
- **Endpoint Storage**: Parsed endpoints cached in data/cache.db with full metadata
- **Search Optimization**: Semantic search capabilities for endpoint discovery
- **Performance**: Sub-second response times with intelligent caching

**🔄 Intelligent Candidate Generation**
- **k+buffer Logic**: Generate k+2 candidates for better selection quality
- **Context Awareness**: Recent API call patterns influence predictions
- **Workflow Recognition**: Common API usage patterns automatically detected

### API Response Enhancements

Enhanced `/predict` endpoint now returns:
```json
{
  "predictions": [
    {
      "api_call": "GET /api/users/{id}",
      "method": "GET", 
      "description": "Retrieve user information by ID",
      "parameters": {"id": "user_id"},
      "confidence": 0.89,
      "reasoning": "Natural progression from login to user data retrieval",
      "rank": 1,
      "ai_provider": "anthropic",
      "semantic_similarity": 0.85,
      "processing_time_ms": 245
    }
  ],
  "confidence_scores": [0.89],
  "metadata": {
    "model_version": "v2.0-ai-layer",
    "ai_provider": "anthropic",
    "semantic_similarity_enabled": true,
    "context_events_used": 3,
    "total_endpoints_considered": 150,
    "filtered_endpoints_used": 12
  }
}
```

## 🧪 Testing Phase 2 AI Layer

### Manual Testing Steps

1. **Start the Server**:
```bash
# With API keys configured
export ANTHROPIC_API_KEY="your-key-here"
uvicorn app.main:app --reload --port 8000
```

2. **Test Basic Functionality**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I need to get user information for authentication",
    "history": [
      {"api_call": "/api/auth/login", "method": "POST", "timestamp": "2024-01-01T10:00:00Z"}
    ],
    "max_predictions": 3,
    "temperature": 0.7
  }'
```

3. **Expected Response** (with AI enabled):
```json
{
  "predictions": [
    {
      "api_call": "GET /api/users/{id}",
      "method": "GET",
      "description": "Retrieve user information by ID", 
      "confidence": 0.85,
      "reasoning": "Natural progression from login to user data",
      "rank": 1,
      "ai_provider": "anthropic",
      "semantic_similarity_enabled": true
    }
  ],
  "confidence_scores": [0.85],
  "metadata": {
    "model_version": "v2.0-ai-layer",
    "ai_provider": "anthropic"
  }
}
```

4. **Test Semantic Similarity**:
```bash
# First, add some OpenAPI specs to the cache
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "create a new user account",
    "max_predictions": 3
  }'
```

5. **Test History-Based Filtering**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "update user profile",
    "history": [
      {"api_call": "/api/users/123", "method": "GET"},
      {"api_call": "/api/users/123/profile", "method": "GET"}
    ],
    "max_predictions": 3
  }'
```

6. **Check System Status**:
```bash
curl "http://localhost:8000/metrics"
```

Expected metrics should show:
- `ai_layer_status: "operational"`
- `anthropic_available: true`
- `semantic_model_loaded: true`
- Cache and endpoint statistics

### Testing Without API Keys

The system gracefully falls back to rule-based predictions:

```bash
# Unset API keys to test fallback
unset ANTHROPIC_API_KEY
unset OPENAI_API_KEY

# Restart server and test - should still work with rule-based predictions
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "get user data", "max_predictions": 2}'
```

### Validation Checklist

✅ **AI Integration**: Responses include `ai_provider` field  
✅ **Semantic Similarity**: `semantic_similarity_enabled: true` in metadata  
✅ **Enhanced Caching**: Endpoints stored in `data/cache.db`  
✅ **Context Awareness**: History influences predictions  
✅ **Fallback Handling**: Works without API keys  
✅ **Performance**: Response times under 1 second  

---

## 📝 Phase 3 ML Layer Implementation

### Key ML Enhancements

**🤖 LightGBM ML Ranking**
- **ML Ranker**: LightGBM-based ranking model with `objective='lambdarank'`, `metric='ndcg'`
- **Model Parameters**: `n_estimators=100`, `num_leaves=31` for optimal ranking performance
- **Learning-to-Rank**: Implements NDCG optimization for improved candidate ordering
- **k+buffer Strategy**: Generates k+2 candidates (5 total) and returns top k (3) results

**🔢 Advanced Feature Engineering**
- **11 ML Features**: Comprehensive feature set including temporal, semantic, and contextual features
- **Sentence Transformers**: `all-MiniLM-L6-v2` model for semantic similarity calculations
- **N-gram Language Models**: Bigram and trigram probability calculations for prompt analysis
- **Feature Categories**: 
  - Temporal: `time_since_last`, `session_length`
  - Categorical: `last_endpoint_type`, `last_resource`, `endpoint_type`
  - Boolean: `resource_match`, `action_verb_match`
  - Numeric: `workflow_distance`, `prompt_similarity`, `bigram_prob`, `trigram_prob`

**⚙️ Synthetic Training Data**
- **Markov Chain Generator**: 10,000 synthetic SaaS workflow sequences
- **Workflow Patterns**: Realistic SaaS workflows (Browse → Edit → Save → Confirm)
- **Training Examples**: Positive (actual next calls) and negative (random sampling) examples
- **Database Storage**: All synthetic data and features stored in `data/cache.db`

**🎯 Integrated AI + ML Pipeline**
- **Phase 2 + Phase 3**: Seamless integration of AI Layer with ML Ranking
- **Hybrid Approach**: AI generates candidates, ML ranks them optimally
- **Fallback Support**: Graceful degradation when ML model unavailable
- **Real-time Features**: Feature extraction and ranking in <1 second

### API Enhancements

**Enhanced `/predict` Endpoint**
```json
{
  "predictions": [
    {
      "api_call": "GET /api/users/{id}",
      "method": "GET",
      "description": "Retrieve user information by ID",
      "parameters": {"id": "user_id"},
      "confidence": 0.89,
      "ml_ranking_score": 0.91,
      "ml_rank": 1,
      "ml_features": {...},
      "model_version": "v1.0-lightgbm"
    }
  ],
  "confidence_scores": [0.89],
  "metadata": {
    "model_version": "v3.0-ml-layer",
    "ai_provider": "anthropic",
    "ml_ranking_enabled": true,
    "candidates_generated": 5,
    "candidates_ranked": 3,
    "k_plus_buffer": "3+2",
    "processing_method": "hybrid_ai_ml"
  },
  "processing_time_ms": 185
}
```

**New ML Endpoints**
- **`POST /train`**: Train or retrain the LightGBM ranking model
- **`POST /generate-data`**: Generate 10,000 synthetic workflow sequences
- **`GET /metrics`**: Comprehensive ML layer performance metrics
- **`GET /health`**: Multi-component health check including ML status

### ML Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Prompt   │ -> │   AI Layer       │ -> │  ML Ranker      │
│   + History     │    │  (Phase 2)       │    │  (Phase 3)      │
└─────────────────┘    │ • Anthropic      │    │ • LightGBM      │
                       │ • OpenAI         │    │ • Feature Eng   │
                       │ • Candidates: 5  │    │ • Returns: 3    │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Synthetic Data   │    │ Ranked Results  │
                       │ • Markov Chains  │    │ • ML Confidence │
                       │ • 10K Sequences  │    │ • NDCG Optimized│
                       │ • SQLite Storage │    │ • k+buffer Logic│
                       └──────────────────┘    └─────────────────┘
```

### Performance Improvements

**ML Layer Benefits**
- **Improved Ranking**: NDCG-optimized ranking vs simple confidence scoring
- **Context Awareness**: 11 contextual features vs basic text analysis
- **Workflow Understanding**: Markov chain patterns for realistic sequences
- **Semantic Similarity**: Transformer-based semantic matching
- **Continuous Learning**: Feature storage for model retraining

**Performance Metrics**
- **ML Ranking Time**: <200ms for 5 candidates
- **Feature Extraction**: <50ms for 11 features
- **Total Processing**: <800ms end-to-end (AI + ML)
- **Cache Hit Rate**: >80% for common patterns
- **Training Time**: ~30 seconds for 10K sequences

### 🧪 Testing Phase 3 ML Layer

#### 1. Generate Training Data
```bash
curl -X POST "http://localhost:8000/generate-data" \
  -H "Content-Type: application/json"
```

#### 2. Train ML Model
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json"
```

#### 3. Test ML-Ranked Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I need to update user profile information",
    "history": [
      {"api_call": "/api/users/123", "method": "GET", "timestamp": "2024-01-01T10:00:00Z"}
    ],
    "max_predictions": 3,
    "temperature": 0.7
  }'
```

#### 4. Expected ML Response
```json
{
  "predictions": [
    {
      "api_call": "PUT /api/users/{id}",
      "method": "PUT",
      "description": "Update user profile information",
      "confidence": 0.87,
      "ml_ranking_score": 0.92,
      "ml_rank": 1,
      "model_version": "v1.0-lightgbm"
    }
  ],
  "metadata": {
    "model_version": "v3.0-ml-layer",
    "ml_ranking_enabled": true,
    "processing_method": "hybrid_ai_ml"
  }
}
```

#### 5. Check ML Metrics
```bash
curl "http://localhost:8000/metrics"
```

Expected metrics include:
- `ml_layer_metrics.is_trained: true`
- `ml_layer_metrics.validation_ndcg: >0.8`
- `predictor_metrics.average_processing_time_ms: <800`
- Database statistics for synthetic sequences and features

### Phase 3 Validation Checklist

✅ **ML Model Training**: LightGBM trains with lambdarank + NDCG  
✅ **Feature Engineering**: All 11 features extracted correctly  
✅ **Synthetic Data**: 10K Markov chain sequences generated  
✅ **Database Storage**: All data stored in cache.db with SQLite  
✅ **AI + ML Integration**: Seamless k+buffer pipeline  
✅ **Performance**: <800ms end-to-end processing  
✅ **Semantic Similarity**: Sentence-transformers working  
✅ **N-gram Models**: Bigram/trigram probabilities calculated  

### Dependencies Added

```requirements.txt
# ML Layer Dependencies (Phase 3)
lightgbm==4.1.0
sentence-transformers==3.2.0  # Already present from Phase 2
```

### Database Schema

**Phase 3 Tables in `data/cache.db`:**
- `synthetic_sequences`: Markov chain workflow sequences
- `features`: Extracted ML features for training and inference
- `training_data`: Positive/negative examples for ranking
- `ml_models`: Serialized LightGBM models with metadata

---

**OpenSesame Predictor v3.0** - Unlocking API Intelligence with AI + ML Hybrid System 🗝️🤖🧠✨