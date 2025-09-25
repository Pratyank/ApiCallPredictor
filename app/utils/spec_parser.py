import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import logging
import aiohttp

from app.config import get_settings, db_manager

logger = logging.getLogger(__name__)

class OpenAPISpecParser:
    """
    OpenAPI specification fetching and parsing utility with caching capabilities
    Implements 1-hour TTL caching for API specifications using SQLite
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_ttl = self.settings.openapi_cache_ttl  # 1 hour default
        self.session = None
        self._init_cache_table()
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_fetches = 0
        
        logger.info("Initialized OpenAPI Spec Parser with 1-hour TTL caching")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'OpenSesame-Predictor/1.0.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _init_cache_table(self):
        """Initialize SQLite table for caching OpenAPI specs"""
        try:
            conn = sqlite3.connect(self.settings.database_url.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS openapi_cache (
                    spec_url TEXT PRIMARY KEY,
                    spec_hash TEXT NOT NULL,
                    spec_content TEXT NOT NULL,
                    cached_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    ttl_seconds INTEGER DEFAULT 3600,
                    access_count INTEGER DEFAULT 0,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.debug("OpenAPI cache table initialized")
            
        except Exception as e:
            logger.error(f"Cache table initialization error: {str(e)}")
    
    async def fetch_openapi_spec(
        self, 
        spec_url: str, 
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch OpenAPI specification with intelligent caching
        
        Args:
            spec_url: URL to the OpenAPI specification (JSON or YAML)
            force_refresh: Bypass cache and fetch fresh spec
            
        Returns:
            Parsed OpenAPI specification as dictionary, or None if failed
        """
        
        try:
            self.total_fetches += 1
            
            # Check cache first unless force refresh is requested
            if not force_refresh:
                cached_spec = await self._get_cached_spec(spec_url)
                if cached_spec:
                    self.cache_hits += 1
                    logger.debug(f"Cache hit for OpenAPI spec: {spec_url}")
                    return cached_spec
            
            self.cache_misses += 1
            logger.info(f"Fetching OpenAPI spec from: {spec_url}")
            
            # Fetch fresh specification
            if not self.session:
                async with aiohttp.ClientSession() as session:
                    spec_content = await self._fetch_spec_content(session, spec_url)
            else:
                spec_content = await self._fetch_spec_content(self.session, spec_url)
            
            if not spec_content:
                return None
            
            # Parse the specification
            parsed_spec = await self._parse_spec_content(spec_content, spec_url)
            
            # Cache the parsed specification
            if parsed_spec:
                await self._cache_spec(spec_url, parsed_spec, spec_content)
                logger.info(f"Successfully cached OpenAPI spec: {spec_url}")
            
            return parsed_spec
            
        except Exception as e:
            logger.error(f"OpenAPI spec fetch error for {spec_url}: {str(e)}")
            return None
    
    async def _fetch_spec_content(
        self, 
        session: aiohttp.ClientSession, 
        spec_url: str
    ) -> Optional[str]:
        """Fetch raw specification content from URL"""
        
        try:
            async with session.get(spec_url) as response:
                if response.status == 200:
                    content = await response.text()
                    logger.debug(f"Fetched {len(content)} bytes from {spec_url}")
                    return content
                else:
                    logger.warning(f"HTTP {response.status} when fetching {spec_url}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching OpenAPI spec: {spec_url}")
            return None
        except Exception as e:
            logger.error(f"Network error fetching {spec_url}: {str(e)}")
            return None
    
    async def _parse_spec_content(
        self, 
        content: str, 
        spec_url: str
    ) -> Optional[Dict[str, Any]]:
        """Parse OpenAPI specification content (JSON or YAML)"""
        
        try:
            # Try JSON first
            if content.strip().startswith('{'):
                parsed = json.loads(content)
                logger.debug("Parsed OpenAPI spec as JSON")
                return parsed
            
            # Try YAML if JSON fails
            try:
                import yaml
                parsed = yaml.safe_load(content)
                logger.debug("Parsed OpenAPI spec as YAML")
                return parsed
            except ImportError:
                logger.warning("YAML parsing not available, install PyYAML for YAML support")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {spec_url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Spec parsing error for {spec_url}: {str(e)}")
            return None
    
    async def _get_cached_spec(self, spec_url: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached OpenAPI specification if valid"""
        
        try:
            conn = sqlite3.connect(self.settings.database_url.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            # Check for valid cached spec
            cursor.execute('''
                SELECT spec_content, cached_at, ttl_seconds, access_count
                FROM openapi_cache 
                WHERE spec_url = ? 
                AND datetime(cached_at, "+" || ttl_seconds || " seconds") > datetime("now")
            ''', (spec_url,))
            
            result = cursor.fetchone()
            
            if result:
                spec_content, cached_at, ttl_seconds, access_count = result
                
                # Update access statistics
                cursor.execute('''
                    UPDATE openapi_cache 
                    SET access_count = access_count + 1, last_accessed = datetime("now")
                    WHERE spec_url = ?
                ''', (spec_url,))
                
                conn.commit()
                conn.close()
                
                # Parse cached content
                cached_spec = json.loads(spec_content)
                logger.debug(f"Retrieved cached spec for {spec_url} (accessed {access_count + 1} times)")
                return cached_spec
            
            conn.close()
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval error: {str(e)}")
            return None
    
    async def _cache_spec(
        self, 
        spec_url: str, 
        parsed_spec: Dict[str, Any], 
        raw_content: str
    ):
        """Cache OpenAPI specification with TTL"""
        
        try:
            # Generate content hash for integrity checking
            content_hash = hashlib.md5(raw_content.encode()).hexdigest()
            spec_json = json.dumps(parsed_spec)
            
            conn = sqlite3.connect(self.settings.database_url.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            # Insert or replace cached spec
            cursor.execute('''
                INSERT OR REPLACE INTO openapi_cache 
                (spec_url, spec_hash, spec_content, cached_at, ttl_seconds, access_count, last_accessed)
                VALUES (?, ?, ?, datetime("now"), ?, 0, datetime("now"))
            ''', (spec_url, content_hash, spec_json, self.cache_ttl))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Cached OpenAPI spec for {spec_url} with {self.cache_ttl}s TTL")
            
        except Exception as e:
            logger.warning(f"Spec caching error: {str(e)}")
    
    async def extract_api_endpoints(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract API endpoints from OpenAPI specification"""
        
        try:
            endpoints = []
            paths = spec.get("paths", {})
            
            for path, path_info in paths.items():
                for method, method_info in path_info.items():
                    if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        endpoint = {
                            "path": path,
                            "method": method.upper(),
                            "summary": method_info.get("summary", ""),
                            "description": method_info.get("description", ""),
                            "tags": method_info.get("tags", []),
                            "parameters": self._extract_parameters(method_info),
                            "responses": list(method_info.get("responses", {}).keys()),
                            "operationId": method_info.get("operationId", "")
                        }
                        endpoints.append(endpoint)
            
            logger.info(f"Extracted {len(endpoints)} endpoints from OpenAPI spec")
            return endpoints
            
        except Exception as e:
            logger.error(f"Endpoint extraction error: {str(e)}")
            return []
    
    def _extract_parameters(self, method_info: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract parameter information from method definition"""
        
        parameters = {
            "query": [],
            "path": [],
            "header": [],
            "body": []
        }
        
        # Extract parameters
        for param in method_info.get("parameters", []):
            param_info = {
                "name": param.get("name", ""),
                "type": param.get("type", param.get("schema", {}).get("type", "string")),
                "required": param.get("required", False),
                "description": param.get("description", "")
            }
            
            param_location = param.get("in", "query")
            if param_location in parameters:
                parameters[param_location].append(param_info)
        
        # Extract request body schema
        request_body = method_info.get("requestBody", {})
        if request_body:
            content = request_body.get("content", {})
            for content_type, content_info in content.items():
                schema = content_info.get("schema", {})
                parameters["body"].append({
                    "content_type": content_type,
                    "schema": schema,
                    "required": request_body.get("required", False)
                })
        
        return parameters
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching performance statistics"""
        
        try:
            conn = sqlite3.connect(self.settings.database_url.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            # Get cache statistics
            cursor.execute('SELECT COUNT(*) FROM openapi_cache')
            total_cached = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT COUNT(*) FROM openapi_cache 
                WHERE datetime(cached_at, "+" || ttl_seconds || " seconds") > datetime("now")
            ''')
            valid_cached = cursor.fetchone()[0]
            
            cursor.execute('SELECT SUM(access_count) FROM openapi_cache')
            total_accesses = cursor.fetchone()[0] or 0
            
            conn.close()
            
            hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses)) if (self.cache_hits + self.cache_misses) > 0 else 0
            
            return {
                "total_specs_cached": total_cached,
                "valid_specs_cached": valid_cached,
                "expired_specs": total_cached - valid_cached,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "total_accesses": total_accesses,
                "ttl_seconds": self.cache_ttl
            }
            
        except Exception as e:
            logger.error(f"Cache stats error: {str(e)}")
            return {}
    
    async def clear_cache(self, spec_url: Optional[str] = None) -> Dict[str, Any]:
        """Clear cached specifications"""
        
        try:
            conn = sqlite3.connect(self.settings.database_url.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            if spec_url:
                # Clear specific spec
                cursor.execute('DELETE FROM openapi_cache WHERE spec_url = ?', (spec_url,))
                cleared = cursor.rowcount
                logger.info(f"Cleared cache for {spec_url}")
            else:
                # Clear all cached specs
                cursor.execute('DELETE FROM openapi_cache')
                cleared = cursor.rowcount
                logger.info("Cleared all cached OpenAPI specs")
            
            conn.commit()
            conn.close()
            
            return {
                "cleared_count": cleared,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup_expired_cache(self) -> Dict[str, Any]:
        """Clean up expired cache entries"""
        
        try:
            conn = sqlite3.connect(self.settings.database_url.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            # Remove expired entries
            cursor.execute('''
                DELETE FROM openapi_cache 
                WHERE datetime(cached_at, "+" || ttl_seconds || " seconds") <= datetime("now")
            ''')
            
            cleaned = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {cleaned} expired cache entries")
            
            return {
                "cleaned_entries": cleaned,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")
            return {"error": str(e)}

# Convenience function for quick spec fetching
async def fetch_spec(spec_url: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """Convenience function to fetch OpenAPI specification with caching"""
    async with OpenAPISpecParser() as parser:
        return await parser.fetch_openapi_spec(spec_url, force_refresh)