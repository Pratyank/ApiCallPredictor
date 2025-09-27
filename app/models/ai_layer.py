import asyncio
import logging
import json
import os
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from app.config import get_settings
from app.utils.spec_parser import OpenAPISpecParser

logger = logging.getLogger(__name__)

class AiLayer:
    """
    Advanced AI Layer for API call prediction using Anthropic's API with semantic similarity
    
    Features:
    - Anthropic Claude integration with OpenAI fallback
    - Semantic similarity using sentence-transformers
    - Context-aware candidate generation with k+buffer logic
    - History-based filtering and workflow pattern recognition
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize API clients
        self.anthropic_client = None
        self.openai_client = None
        
        # Initialize sentence transformer model with caching
        self.sentence_model = None
        self.model_loaded = False
        
        # Embedding cache for performance optimization
        self.cache_db_path = os.path.join(os.getcwd(), 'data', 'cache.db')
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)
        self._init_embedding_cache()
        
        # Initialize spec parser for endpoint retrieval
        self.spec_parser = OpenAPISpecParser()
        
        # Prediction parameters
        self.k = 3  # Number of predictions requested
        self.buffer = 2  # Additional candidates for better filtering
        self.max_context_events = 10  # Maximum recent events to consider
        
        # Performance tracking
        self.total_predictions = 0
        self.anthropic_calls = 0
        self.openai_calls = 0
        self.semantic_searches = 0
        
        logger.info("Initialized AiLayer with Anthropic API and semantic similarity capabilities")
    
    def _load_env_file(self):
        """Load environment variables from .env file"""
        import os
        from pathlib import Path
        
        # Look for .env file in current directory or project root
        env_paths = [
            Path(".env"),
            Path("./.env"),
            Path("../.env"),
            Path(os.getcwd()) / ".env"
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                logger.info(f"Loading .env file from {env_path}")
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            if not os.getenv(key):  # Don't override existing env vars
                                os.environ[key] = value
                                logger.debug(f"Set {key} from .env file")
                return
        logger.warning("No .env file found in expected locations")
    
    async def _init_apis(self):
        """Initialize API clients lazily"""
        # Try to load .env file manually if environment variable not found
        if not os.getenv('ANTHROPIC_API_KEY'):
            self._load_env_file()
        
        if not self.anthropic_client:
            try:
                import anthropic
                api_key = os.getenv('ANTHROPIC_API_KEY')
                logger.info(f"Anthropic API key from environment: {'Found' if api_key else 'NOT FOUND'}")
                if api_key:
                    logger.info(f"Anthropic API key length: {len(api_key)} characters")
                    self.anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
                    logger.info("Anthropic API client initialized successfully")
                else:
                    logger.warning("ANTHROPIC_API_KEY environment variable not set")
            except ImportError:
                logger.warning("Anthropic library not installed, will use fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {str(e)}")
        
        if not self.openai_client:
            try:
                import openai
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    self.openai_client = openai.AsyncOpenAI(api_key=api_key)
                    logger.info("OpenAI API client initialized")
            except ImportError:
                logger.warning("OpenAI library not installed")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _init_embedding_cache(self):
        """Initialize embedding cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT PRIMARY KEY,
                    embedding_data BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create index for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_embedding_model ON embedding_cache(model_name)')
            
            conn.commit()
            conn.close()
            logger.debug("Initialized embedding cache table")
            
        except Exception as e:
            logger.error(f"Embedding cache initialization error: {str(e)}")

    async def _init_sentence_model(self):
        """Initialize sentence transformer model lazily with async support"""
        if not self.model_loaded:
            try:
                from sentence_transformers import SentenceTransformer
                
                # Use a lightweight model for better performance
                model_name = 'all-MiniLM-L6-v2'
                
                # Load model in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.sentence_model = await loop.run_in_executor(
                    None, lambda: SentenceTransformer(model_name)
                )
                self.model_loaded = True
                
                logger.info(f"Sentence transformer model '{model_name}' loaded successfully")
                
            except ImportError:
                logger.error("sentence-transformers not installed. Please install it for semantic similarity features")
                self.model_loaded = False
            except Exception as e:
                logger.error(f"Failed to load sentence transformer model: {str(e)}")
                self.model_loaded = False
    
    async def generate_predictions(
        self, 
        prompt: str, 
        history: List[Dict[str, Any]] = None,
        k: int = None,
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate API call predictions using AI with semantic similarity
        
        Args:
            prompt: User intent description
            history: List of recent API call events
            k: Number of predictions to return (default: 3)
            temperature: Randomness factor for AI generation
            
        Returns:
            List of predicted API calls with confidence scores and metadata
        """
        
        try:
            await self._init_apis()
            await self._init_sentence_model()
            
            self.total_predictions += 1
            start_time = asyncio.get_event_loop().time()
            
            # Use provided k or default
            k_predictions = k if k is not None else self.k
            
            # Get recent events for context
            recent_events = self._extract_recent_events(history or [])
            
            # Get available endpoints from cache
            available_endpoints = await self._get_available_endpoints()
            
            # Filter endpoints based on recent events and semantic similarity
            filtered_endpoints = await self._filter_endpoints_by_context(
                available_endpoints, recent_events, prompt
            )
            
            # Generate candidates using AI (k + buffer for better selection)
            candidates = await self._generate_ai_candidates(
                prompt, recent_events, filtered_endpoints, k_predictions + self.buffer, temperature
            )
            
            # Apply semantic similarity scoring
            if self.model_loaded and candidates:
                candidates = await self._apply_semantic_scoring(prompt, candidates)
            
            # Select top k predictions
            final_predictions = self._select_top_predictions(candidates, k_predictions)
            
            # Add metadata
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            for i, prediction in enumerate(final_predictions):
                prediction.update({
                    'rank': i + 1,
                    'processing_time_ms': processing_time,
                    'ai_provider': self._get_active_provider(),
                    'semantic_similarity_enabled': self.model_loaded,
                    'context_events_used': len(recent_events),
                    'total_endpoints_considered': len(available_endpoints),
                    'filtered_endpoints_used': len(filtered_endpoints)
                })
            
            logger.info(f"Generated {len(final_predictions)} AI predictions in {processing_time:.2f}ms")
            return final_predictions
            
        except Exception as e:
            logger.error(f"AI prediction generation error: {str(e)}")
            return await self._get_fallback_predictions(prompt, k_predictions or self.k)
    
    def _extract_recent_events(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and format recent events from history"""
        if not history:
            return []
        
        # Take the most recent events (up to max_context_events)
        recent_events = history[-self.max_context_events:] if len(history) > self.max_context_events else history
        
        # Normalize event format
        normalized_events = []
        for event in recent_events:
            if isinstance(event, dict):
                normalized_event = {
                    'api_call': event.get('api_call', ''),
                    'method': event.get('method', ''),
                    'timestamp': event.get('timestamp', ''),
                    'parameters': event.get('parameters', {}),
                    'status': event.get('status', 'unknown')
                }
                normalized_events.append(normalized_event)
        
        logger.debug(f"Extracted {len(normalized_events)} recent events from history")
        return normalized_events
    
    async def _get_available_endpoints(self) -> List[Dict[str, Any]]:
        """Get available API endpoints from cached data"""
        try:
            # Get all cached endpoints
            endpoints = await self.spec_parser.get_cached_endpoints(limit=1000)
            
            logger.debug(f"Retrieved {len(endpoints)} available endpoints from cache")
            return endpoints
            
        except Exception as e:
            logger.error(f"Error retrieving available endpoints: {str(e)}")
            return []
    
    async def _filter_endpoints_by_context(
        self, 
        endpoints: List[Dict[str, Any]], 
        recent_events: List[Dict[str, Any]],
        prompt: str
    ) -> List[Dict[str, Any]]:
        """Filter endpoints based on recent events and semantic similarity"""
        
        if not endpoints:
            return []
        
        try:
            # If we have recent events, prioritize related endpoints
            if recent_events:
                filtered_endpoints = self._filter_by_recent_patterns(endpoints, recent_events)
            else:
                filtered_endpoints = endpoints
            
            # Apply semantic filtering if model is loaded
            if self.model_loaded and filtered_endpoints:
                filtered_endpoints = await self._filter_by_semantic_similarity(
                    filtered_endpoints, prompt, top_k=min(50, len(filtered_endpoints))
                )
            
            # Limit to reasonable number for AI processing
            max_endpoints = 30
            final_filtered = filtered_endpoints[:max_endpoints] if len(filtered_endpoints) > max_endpoints else filtered_endpoints
            
            logger.debug(f"Filtered to {len(final_filtered)} relevant endpoints")
            return final_filtered
            
        except Exception as e:
            logger.error(f"Error filtering endpoints: {str(e)}")
            return endpoints[:30]  # Fallback to first 30 endpoints
    
    def _filter_by_recent_patterns(
        self, 
        endpoints: List[Dict[str, Any]], 
        recent_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter endpoints based on patterns from recent API calls"""
        
        if not recent_events:
            return endpoints
        
        # Extract patterns from recent events
        recent_methods = [event.get('method', '').upper() for event in recent_events]
        recent_paths = [event.get('api_call', '') for event in recent_events]
        recent_resources = set()
        
        for path in recent_paths:
            # Extract resource names from paths (e.g., '/api/users/123' -> 'users')
            parts = path.strip('/').split('/')
            for part in parts:
                if part and not part.isdigit() and part not in ['api', 'v1', 'v2']:
                    recent_resources.add(part.lower())
        
        # Score endpoints based on similarity to recent patterns
        scored_endpoints = []
        for endpoint in endpoints:
            score = 0
            
            # Method similarity
            if endpoint.get('method', '').upper() in recent_methods:
                score += 2
            
            # Path/resource similarity
            endpoint_path = endpoint.get('path', '').lower()
            for resource in recent_resources:
                if resource in endpoint_path:
                    score += 3
            
            # Tag similarity (if available)
            endpoint_tags = endpoint.get('tags', [])
            if isinstance(endpoint_tags, list):
                for tag in endpoint_tags:
                    if isinstance(tag, str) and tag.lower() in recent_resources:
                        score += 1
            
            scored_endpoints.append((score, endpoint))
        
        # Sort by score (descending) and return endpoints
        scored_endpoints.sort(key=lambda x: x[0], reverse=True)
        filtered = [endpoint for score, endpoint in scored_endpoints if score > 0]
        
        # If no matches found, return all endpoints
        if not filtered:
            filtered = endpoints
        
        logger.debug(f"Pattern-based filtering: {len(filtered)} endpoints retained")
        return filtered
    
    async def _get_cached_embedding(self, text: str, model_name: str = 'all-MiniLM-L6-v2') -> Optional[np.ndarray]:
        """Get cached embedding or return None"""
        try:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT embedding_data FROM embedding_cache 
                WHERE text_hash = ? AND model_name = ?
            ''', (text_hash, model_name))
            
            result = cursor.fetchone()
            
            if result:
                # Update access count
                cursor.execute('''
                    UPDATE embedding_cache 
                    SET access_count = access_count + 1 
                    WHERE text_hash = ? AND model_name = ?
                ''', (text_hash, model_name))
                conn.commit()
                
                # Deserialize embedding
                import pickle
                embedding = pickle.loads(result[0])
                conn.close()
                return embedding
            
            conn.close()
            return None
            
        except Exception as e:
            logger.warning(f"Embedding cache retrieval error: {str(e)}")
            return None
    
    async def _cache_embedding(self, text: str, embedding: np.ndarray, model_name: str = 'all-MiniLM-L6-v2'):
        """Cache embedding for future use"""
        try:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Serialize embedding
            import pickle
            embedding_data = pickle.dumps(embedding)
            
            cursor.execute('''
                INSERT OR REPLACE INTO embedding_cache 
                (text_hash, embedding_data, model_name, access_count)
                VALUES (?, ?, ?, 1)
            ''', (text_hash, embedding_data, model_name))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Embedding cache storage error: {str(e)}")
    
    async def _batch_encode_with_cache(self, texts: List[str]) -> List[np.ndarray]:
        """Encode texts with caching support"""
        embeddings = []
        texts_to_encode = []
        text_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = await self._get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)
                texts_to_encode.append(text)
                text_indices.append(i)
        
        # Batch encode uncached texts
        if texts_to_encode:
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(
                None, lambda: self.sentence_model.encode(texts_to_encode)
            )
            
            # Cache new embeddings and update results
            for j, embedding in enumerate(new_embeddings):
                text_idx = text_indices[j]
                embeddings[text_idx] = embedding
                
                # Cache asynchronously without blocking
                asyncio.create_task(
                    self._cache_embedding(texts_to_encode[j], embedding)
                )
        
        return embeddings

    async def _filter_by_semantic_similarity(
        self, 
        endpoints: List[Dict[str, Any]], 
        prompt: str, 
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """Filter endpoints using semantic similarity to user prompt with caching"""
        
        if not self.model_loaded or not endpoints:
            return endpoints
        
        try:
            self.semantic_searches += 1
            
            # Create endpoint descriptions for similarity matching
            endpoint_texts = []
            for endpoint in endpoints:
                # Combine path, summary, description for rich context
                text_parts = [
                    endpoint.get('path', ''),
                    endpoint.get('method', ''),
                    endpoint.get('summary', ''),
                    endpoint.get('description', ''),
                ]
                
                # Add tag information
                tags = endpoint.get('tags', [])
                if isinstance(tags, list):
                    text_parts.extend([str(tag) for tag in tags])
                
                # Create comprehensive description
                endpoint_text = ' '.join([part for part in text_parts if part]).strip()
                endpoint_texts.append(endpoint_text)
            
            if not endpoint_texts:
                return endpoints
            
            # Compute embeddings with caching
            prompt_embedding = await self._get_cached_embedding(prompt)
            if prompt_embedding is None:
                loop = asyncio.get_event_loop()
                prompt_embedding = await loop.run_in_executor(
                    None, lambda: self.sentence_model.encode([prompt])[0]
                )
                await self._cache_embedding(prompt, prompt_embedding)
            
            endpoint_embeddings = await self._batch_encode_with_cache(endpoint_texts)
            
            # Calculate cosine similarities
            similarities = np.array([
                np.dot(embedding, prompt_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(prompt_embedding)
                ) for embedding in endpoint_embeddings
            ])
            
            # Sort by similarity and take top_k
            similarity_scores = list(zip(similarities, endpoints))
            similarity_scores.sort(key=lambda x: x[0], reverse=True)
            
            filtered_endpoints = [endpoint for score, endpoint in similarity_scores[:top_k]]
            
            logger.debug(f"Semantic filtering: selected {len(filtered_endpoints)} most similar endpoints (cached: {len([e for e in endpoint_embeddings if e is not None])})")
            return filtered_endpoints
            
        except Exception as e:
            logger.error(f"Semantic filtering error: {str(e)}")
            return endpoints
    
    async def _generate_ai_candidates(
        self,
        prompt: str,
        recent_events: List[Dict[str, Any]],
        filtered_endpoints: List[Dict[str, Any]],
        num_candidates: int,
        temperature: float
    ) -> List[Dict[str, Any]]:
        """Generate API call candidates using AI models"""
        
        # Build AI prompt
        ai_prompt = self._build_ai_prompt(prompt, recent_events, filtered_endpoints, num_candidates)
        
        # Try Anthropic first, then OpenAI fallback
        try:
            if self.anthropic_client:
                return await self._call_anthropic_api(ai_prompt, temperature)
            elif self.openai_client:
                return await self._call_openai_api(ai_prompt, temperature)
            else:
                logger.warning("No AI providers available, using rule-based fallback")
                return await self._generate_rule_based_predictions(prompt, filtered_endpoints, num_candidates)
                
        except Exception as e:
            logger.error(f"AI candidate generation failed: {str(e)}")
            return await self._generate_rule_based_predictions(prompt, filtered_endpoints, num_candidates)
    
    def _build_ai_prompt(
        self,
        prompt: str,
        recent_events: List[Dict[str, Any]],
        filtered_endpoints: List[Dict[str, Any]],
        num_candidates: int
    ) -> str:
        """Build structured prompt for AI models"""
        
        # Format recent events
        events_text = ""
        if recent_events:
            events_text = "\n".join([
                f"- {event.get('method', '')} {event.get('api_call', '')} (status: {event.get('status', 'unknown')})"
                for event in recent_events[-5:]  # Last 5 events
            ])
        
        # Format available endpoints (sample to avoid token limits)
        endpoints_text = ""
        if filtered_endpoints:
            sample_endpoints = filtered_endpoints[:15]  # Limit to avoid token overflow
            endpoints_text = "\n".join([
                f"- {ep.get('method', '')} {ep.get('path', '')} - {ep.get('summary', '') or ep.get('description', '')[:100]}"
                for ep in sample_endpoints
            ])
        
        # Build the prompt using the specified template
        ai_prompt = f"""Given user history: {events_text}

User intent: {prompt}

Available endpoints: {endpoints_text}

Generate {num_candidates} most likely next API calls considering:
1. Natural progression from recent actions
2. User's stated intent  
3. Common workflow patterns

Output as JSON array of actions with the following structure:
[
  {{
    "api_call": "HTTP_METHOD /path/to/endpoint",
    "method": "GET|POST|PUT|DELETE|PATCH",
    "description": "Brief description of what this endpoint does",
    "parameters": {{"param1": "value1", "param2": "value2"}},
    "confidence": 0.85,
    "reasoning": "Why this API call is likely next"
  }}
]

Focus on realistic API calls that would logically follow the user's history and fulfill their stated intent."""
        
        return ai_prompt
    
    async def _call_anthropic_api(self, prompt: str, temperature: float) -> List[Dict[str, Any]]:
        """Call Anthropic Claude API for predictions"""
        
        try:
            self.anthropic_calls += 1
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",  # Fast, cost-effective model
                max_tokens=1000,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse JSON response
            response_text = response.content[0].text.strip()
            
            # Extract JSON from response (handle potential markdown formatting)
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            else:
                # Look for JSON array in the response
                start = response_text.find("[")
                if start != -1:
                    # Find the matching closing bracket
                    bracket_count = 0
                    end = start
                    for i, char in enumerate(response_text[start:], start):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end = i + 1
                                break
                    response_text = response_text[start:end]
            
            predictions = json.loads(response_text)
            
            if not isinstance(predictions, list):
                predictions = [predictions] if isinstance(predictions, dict) else []
            
            logger.debug(f"Anthropic API returned {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Anthropic API call failed: {str(e)}")
            raise
    
    async def _call_openai_api(self, prompt: str, temperature: float) -> List[Dict[str, Any]]:
        """Call OpenAI API for predictions with function calling"""
        
        try:
            self.openai_calls += 1
            
            # Define function schema for structured output
            function_schema = {
                "name": "generate_api_predictions",
                "description": "Generate API call predictions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "predictions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "api_call": {"type": "string"},
                                    "method": {"type": "string"},
                                    "description": {"type": "string"},
                                    "parameters": {"type": "object"},
                                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                    "reasoning": {"type": "string"}
                                },
                                "required": ["api_call", "method", "description", "confidence"]
                            }
                        }
                    },
                    "required": ["predictions"]
                }
            }
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                functions=[function_schema],
                function_call={"name": "generate_api_predictions"},
                temperature=temperature,
                max_tokens=1000
            )
            
            # Extract function call result
            function_call = response.choices[0].message.function_call
            result = json.loads(function_call.arguments)
            predictions = result.get("predictions", [])
            
            logger.debug(f"OpenAI API returned {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    async def _generate_rule_based_predictions(
        self, 
        prompt: str, 
        endpoints: List[Dict[str, Any]], 
        num_candidates: int
    ) -> List[Dict[str, Any]]:
        """Generate predictions using rule-based approach when AI is unavailable"""
        
        predictions = []
        prompt_lower = prompt.lower()
        
        # Simple keyword matching
        keywords_methods = {
            'get': ['GET'],
            'fetch': ['GET'],
            'retrieve': ['GET'],
            'list': ['GET'],
            'show': ['GET'],
            'create': ['POST'],
            'add': ['POST'],
            'insert': ['POST'],
            'update': ['PUT', 'PATCH'],
            'modify': ['PUT', 'PATCH'],
            'edit': ['PUT', 'PATCH'],
            'delete': ['DELETE'],
            'remove': ['DELETE']
        }
        
        # Score endpoints based on keyword matching
        for endpoint in endpoints[:num_candidates * 2]:  # Consider more than needed
            score = 0.5  # Base score
            
            # Method matching
            endpoint_method = endpoint.get('method', '').upper()
            for keyword, methods in keywords_methods.items():
                if keyword in prompt_lower and endpoint_method in methods:
                    score += 0.3
            
            # Path/description matching
            endpoint_desc = (endpoint.get('description', '') + ' ' + endpoint.get('summary', '')).lower()
            path_words = endpoint.get('path', '').lower().split('/')
            
            for word in prompt_lower.split():
                if len(word) > 3:  # Skip short words
                    if word in endpoint_desc:
                        score += 0.2
                    if any(word in path_word for path_word in path_words):
                        score += 0.1
            
            prediction = {
                'api_call': f"{endpoint_method} {endpoint.get('path', '')}",
                'method': endpoint_method,
                'description': endpoint.get('description', '') or endpoint.get('summary', ''),
                'parameters': {},
                'confidence': min(score, 0.9),  # Cap confidence for rule-based
                'reasoning': 'Rule-based prediction using keyword matching',
                'is_rule_based': True
            }
            predictions.append(prediction)
        
        # Sort by confidence and take top predictions
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        return predictions[:num_candidates]
    
    async def _apply_semantic_scoring(
        self, 
        prompt: str, 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply semantic similarity scoring to enhance confidence scores with caching"""
        
        if not self.model_loaded or not candidates:
            return candidates
        
        try:
            # Create candidate descriptions
            candidate_texts = []
            for candidate in candidates:
                text = f"{candidate.get('api_call', '')} {candidate.get('description', '')} {candidate.get('reasoning', '')}"
                candidate_texts.append(text)
            
            # Compute similarities with caching
            prompt_embedding = await self._get_cached_embedding(prompt)
            if prompt_embedding is None:
                loop = asyncio.get_event_loop()
                prompt_embedding = await loop.run_in_executor(
                    None, lambda: self.sentence_model.encode([prompt])[0]
                )
                await self._cache_embedding(prompt, prompt_embedding)
            
            candidate_embeddings = await self._batch_encode_with_cache(candidate_texts)
            
            similarities = np.array([
                np.dot(embedding, prompt_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(prompt_embedding)
                ) for embedding in candidate_embeddings
            ])
            
            # Enhance confidence scores with semantic similarity
            for i, candidate in enumerate(candidates):
                semantic_score = float(similarities[i])
                original_confidence = candidate.get('confidence', 0.5)
                
                # Weighted combination of AI confidence and semantic similarity
                enhanced_confidence = (0.7 * original_confidence) + (0.3 * semantic_score)
                candidate['confidence'] = min(enhanced_confidence, 1.0)
                candidate['semantic_similarity'] = semantic_score
            
            logger.debug("Applied cached semantic similarity scoring to candidates")
            return candidates
            
        except Exception as e:
            logger.error(f"Semantic scoring error: {str(e)}")
            return candidates
    
    def _select_top_predictions(
        self, 
        candidates: List[Dict[str, Any]], 
        k: int
    ) -> List[Dict[str, Any]]:
        """Select top k predictions based on confidence scores"""
        
        # Sort by confidence score (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Take top k and ensure uniqueness
        selected = []
        seen_calls = set()
        
        for candidate in sorted_candidates:
            api_call = candidate.get('api_call', '')
            if api_call not in seen_calls and len(selected) < k:
                seen_calls.add(api_call)
                selected.append(candidate)
        
        return selected
    
    def _get_active_provider(self) -> str:
        """Get the name of the active AI provider"""
        if self.anthropic_client:
            return "anthropic"
        elif self.openai_client:
            return "openai"
        else:
            return "rule_based"
    
    async def _get_fallback_predictions(
        self, 
        prompt: str, 
        k: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback predictions when all else fails"""
        
        fallback_predictions = [
            {
                'api_call': 'GET /api/data',
                'method': 'GET',
                'description': 'Retrieve general data',
                'parameters': {'limit': 10},
                'confidence': 0.3,
                'reasoning': 'Fallback generic data retrieval',
                'is_fallback': True,
                'rank': 1
            }
        ]
        
        return fallback_predictions[:k]
    
    async def get_status(self) -> Dict[str, Any]:
        """Get AI layer status and performance metrics"""
        
        return {
            'ai_layer_status': 'operational',
            'anthropic_available': self.anthropic_client is not None,
            'openai_available': self.openai_client is not None,
            'semantic_model_loaded': self.model_loaded,
            'total_predictions': self.total_predictions,
            'anthropic_calls': self.anthropic_calls,
            'openai_calls': self.openai_calls,
            'semantic_searches': self.semantic_searches,
            'active_provider': self._get_active_provider(),
            'prediction_parameters': {
                'default_k': self.k,
                'buffer_size': self.buffer,
                'max_context_events': self.max_context_events
            }
        }

# Global instance for reuse
ai_layer = None

async def get_ai_layer() -> AiLayer:
    """Get global AI layer instance"""
    global ai_layer
    if ai_layer is None:
        ai_layer = AiLayer()
        # Initialize APIs immediately to ensure environment variables are loaded
        await ai_layer._init_apis()
        await ai_layer._init_sentence_model()
    return ai_layer