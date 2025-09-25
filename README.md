# OpenSesame Predictor

**AI-Powered API Call Prediction Service**

OpenSesame Predictor is a sophisticated FastAPI-based service that uses advanced machine learning and large language models to predict the most relevant API calls based on natural language user prompts and conversation history.

## 🚀 Features

### Core Capabilities
- **Intelligent API Prediction**: Combines LLM reasoning with ML ranking for accurate predictions
- **Multi-Modal Architecture**: Hybrid AI/ML approach with confidence scoring
- **Real-time Processing**: Sub-second response times with intelligent caching
- **Safety & Security**: Comprehensive input validation and guardrails
- **Scalable Design**: Docker containerization with resource optimization

### Technical Highlights
- **FastAPI Framework**: High-performance async API with automatic documentation
- **SQLite Caching**: 1-hour TTL caching system for OpenAPI specifications
- **Feature Engineering**: Advanced text analysis and pattern recognition
- **ML Ranking System**: Context-aware prediction scoring and ranking
- **OpenAPI Integration**: Automatic spec parsing and endpoint extraction

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

3. **Run Development Server**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **Build and Run**:
```bash
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
| `OPENSESAME_LLM_PROVIDER` | `placeholder` | LLM provider (OpenAI, Anthropic) |
| `OPENSESAME_MAX_PROMPT_LENGTH` | `4096` | Maximum prompt length |
| `OPENSESAME_ENABLE_GUARDRAILS` | `true` | Enable safety validation |

### ML Model Settings
- **Feature Vector Size**: 256 dimensions
- **Max History Length**: 100 events
- **Prediction Timeout**: 30 seconds
- **Cache TTL**: 1 hour (3600 seconds)

## 🧠 Architecture Overview

### Prediction Pipeline
1. **Input Validation**: Safety checks and guardrails
2. **Feature Extraction**: NLP processing and pattern recognition
3. **LLM Generation**: Context-aware API call candidates
4. **ML Ranking**: Confidence scoring and relevance ranking
5. **Result Synthesis**: Formatted predictions with metadata

### Caching Strategy
- **L1 Cache**: In-memory TTL cache for frequent requests
- **L2 Cache**: SQLite persistent cache for OpenAPI specs
- **Cache Invalidation**: Intelligent TTL-based expiration

### Security Features
- **Input Sanitization**: SQL injection and XSS prevention
- **Rate Limiting**: Per-user request throttling
- **Content Filtering**: Inappropriate content detection
- **Parameter Validation**: Type checking and bounds enforcement

## 📊 Performance Targets

### Response Times
- **P50 Response Time**: < 500ms
- **P95 Response Time**: < 1000ms
- **AI Layer**: < 500ms per request
- **ML Ranking**: < 100ms per prediction set

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
- **LLM Integration**: Easy provider switching (OpenAI/Anthropic/Local)
- **ML Models**: Pluggable ranking algorithms
- **Cache Backends**: Redis/Memcached integration
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

**OpenSesame Predictor** - Unlocking API Intelligence with AI 🗝️✨