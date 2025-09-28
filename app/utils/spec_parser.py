import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import hashlib
import logging
import aiohttp
import os

from app.config import get_settings, db_manager

logger = logging.getLogger(__name__)


class OpenAPISpecParser:
    """
    OpenAPI specification fetching and parsing utility with caching capabilities
    Implements 1-hour TTL caching for API specifications using SQLite
    Enhanced with endpoint extraction and storage in data/cache.db
    """

    def __init__(self):
        self.settings = get_settings()
        self.cache_ttl = self.settings.openapi_cache_ttl  # 1 hour default
        self.session = None

        # Use data/cache.db for all caching operations
        self.cache_db_path = os.path.join(os.getcwd(), 'data', 'cache.db')
        os.makedirs(os.path.dirname(self.cache_db_path), exist_ok=True)

        self._init_cache_table()
        self._init_endpoints_table()

        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_fetches = 0

        logger.info(
            "Initialized OpenAPI Spec Parser with 1-hour TTL caching and endpoint storage")

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
            conn = sqlite3.connect(self.cache_db_path)
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

    def _init_endpoints_table(self):
        """Initialize SQLite table for storing parsed endpoints"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS parsed_endpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint_id TEXT UNIQUE,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    summary TEXT,
                    description TEXT,
                    parameters TEXT,  -- JSON string of parameters
                    tags TEXT,        -- JSON string of tags
                    responses TEXT,   -- JSON string of response codes
                    operation_id TEXT,
                    spec_url TEXT,
                    spec_hash TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create indexes for faster lookups
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_endpoints_spec_url
                ON parsed_endpoints(spec_url)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_endpoints_method_path
                ON parsed_endpoints(method, path)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_endpoints_description
                ON parsed_endpoints(description)
            ''')

            conn.commit()
            conn.close()
            logger.debug("Parsed endpoints table initialized")

        except Exception as e:
            logger.error(f"Endpoints table initialization error: {str(e)}")

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

            # Cache the parsed specification and extract endpoints
            if parsed_spec:
                await self._cache_spec(spec_url, parsed_spec, spec_content)

                # Extract and store endpoints
                endpoints = await self.extract_api_endpoints(parsed_spec)
                await self._store_parsed_endpoints(endpoints, spec_url, spec_content)

                logger.info(
                    f"Successfully cached OpenAPI spec and extracted {
                        len(endpoints)} endpoints: {spec_url}")

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
                    logger.debug(
                        f"Fetched {
                            len(content)} bytes from {spec_url}")
                    return content
                else:
                    logger.warning(
                        f"HTTP {
                            response.status} when fetching {spec_url}")
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
                logger.warning(
                    "YAML parsing not available, install PyYAML for YAML support")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {spec_url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Spec parsing error for {spec_url}: {str(e)}")
            return None

    async def _get_cached_spec(
            self, spec_url: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached OpenAPI specification if valid"""

        try:
            conn = sqlite3.connect(self.cache_db_path)
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
                logger.debug(
                    f"Retrieved cached spec for {spec_url} (accessed {
                        access_count + 1} times)")
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

            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            # Insert or replace cached spec
            cursor.execute('''
                INSERT OR REPLACE INTO openapi_cache
                (spec_url, spec_hash, spec_content, cached_at, ttl_seconds, access_count, last_accessed)
                VALUES (?, ?, ?, datetime("now"), ?, 0, datetime("now"))
            ''', (spec_url, content_hash, spec_json, self.cache_ttl))

            conn.commit()
            conn.close()

            logger.debug(
                f"Cached OpenAPI spec for {spec_url} with {
                    self.cache_ttl}s TTL")

        except Exception as e:
            logger.warning(f"Spec caching error: {str(e)}")

    async def _store_parsed_endpoints(
        self,
        endpoints: List[Dict[str, Any]],
        spec_url: str,
        spec_content: str
    ):
        """Store parsed endpoints in the cache database"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            # Generate spec hash for consistency
            spec_hash = hashlib.md5(spec_content.encode()).hexdigest()

            for endpoint in endpoints:
                endpoint_id = f"{endpoint['method']}_{endpoint['path']}_{spec_hash[:8]}"

                cursor.execute('''
                    INSERT OR REPLACE INTO parsed_endpoints
                    (endpoint_id, method, path, summary, description, parameters, tags, responses, operation_id, spec_url, spec_hash, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime("now"))
                ''', (
                    endpoint_id,
                    endpoint['method'],
                    endpoint['path'],
                    endpoint.get('summary', ''),
                    endpoint.get('description', ''),
                    json.dumps(endpoint.get('parameters', {})),
                    json.dumps(endpoint.get('tags', [])),
                    json.dumps(endpoint.get('responses', [])),
                    endpoint.get('operationId', ''),
                    spec_url,
                    spec_hash
                ))

            conn.commit()
            conn.close()

            logger.info(
                f"Stored {
                    len(endpoints)} parsed endpoints from {spec_url}")

        except Exception as e:
            logger.error(f"Error storing parsed endpoints: {str(e)}")

    async def get_cached_endpoints(
        self,
        spec_url: str = None,
        method: str = None,
        path_contains: str = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Retrieve cached parsed endpoints with optional filtering"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            query = 'SELECT * FROM parsed_endpoints WHERE 1=1'
            params = []

            if spec_url:
                query += ' AND spec_url = ?'
                params.append(spec_url)

            if method:
                query += ' AND method = ?'
                params.append(method.upper())

            if path_contains:
                query += ' AND path LIKE ?'
                params.append(f'%{path_contains}%')

            query += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to list of dictionaries
            columns = [description[0] for description in cursor.description]
            endpoints = []

            for row in rows:
                endpoint = dict(zip(columns, row))
                # Parse JSON fields
                endpoint['parameters'] = json.loads(
                    endpoint['parameters']) if endpoint['parameters'] else {}
                endpoint['tags'] = json.loads(
                    endpoint['tags']) if endpoint['tags'] else []
                endpoint['responses'] = json.loads(
                    endpoint['responses']) if endpoint['responses'] else []
                endpoints.append(endpoint)

            conn.close()

            logger.debug(f"Retrieved {len(endpoints)} cached endpoints")
            return endpoints

        except Exception as e:
            logger.error(f"Error retrieving cached endpoints: {str(e)}")
            return []

    async def search_endpoints_by_description(
        self,
        search_terms: List[str],
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search endpoints by description content for semantic matching"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            # Build search query for description matching
            search_conditions = []
            params = []

            for term in search_terms:
                search_conditions.append(
                    '(description LIKE ? OR summary LIKE ? OR path LIKE ?)')
                term_pattern = f'%{term}%'
                params.extend([term_pattern, term_pattern, term_pattern])

            if not search_conditions:
                # Return all endpoints if no search terms
                query = 'SELECT * FROM parsed_endpoints ORDER BY created_at DESC LIMIT ?'
                params = [limit]
            else:
                query = f'''
                    SELECT * FROM parsed_endpoints
                    WHERE {' OR '.join(search_conditions)}
                    ORDER BY created_at DESC
                    LIMIT ?
                '''
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to list of dictionaries
            columns = [description[0] for description in cursor.description]
            endpoints = []

            for row in rows:
                endpoint = dict(zip(columns, row))
                # Parse JSON fields
                endpoint['parameters'] = json.loads(
                    endpoint['parameters']) if endpoint['parameters'] else {}
                endpoint['tags'] = json.loads(
                    endpoint['tags']) if endpoint['tags'] else []
                endpoint['responses'] = json.loads(
                    endpoint['responses']) if endpoint['responses'] else []
                endpoints.append(endpoint)

            conn.close()

            logger.debug(
                f"Found {
                    len(endpoints)} endpoints matching search terms")
            return endpoints

        except Exception as e:
            logger.error(f"Error searching endpoints: {str(e)}")
            return []

    async def extract_api_endpoints(
            self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract API endpoints from OpenAPI specification"""

        try:
            endpoints = []
            paths = spec.get("paths", {})

            for path, path_info in paths.items():
                for method, method_info in path_info.items():
                    if method.upper() in [
                            "GET", "POST", "PUT", "DELETE", "PATCH"]:
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

            logger.info(
                f"Extracted {
                    len(endpoints)} endpoints from OpenAPI spec")
            return endpoints

        except Exception as e:
            logger.error(f"Endpoint extraction error: {str(e)}")
            return []

    def _extract_parameters(
            self, method_info: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
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
                "name": param.get(
                    "name", ""), "type": param.get(
                    "type", param.get(
                        "schema", {}).get(
                        "type", "string")), "required": param.get(
                        "required", False), "description": param.get(
                            "description", "")}

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
            conn = sqlite3.connect(self.cache_db_path)
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

            # Get endpoint statistics
            cursor.execute('SELECT COUNT(*) FROM parsed_endpoints')
            total_endpoints = cursor.fetchone()[0]

            conn.close()

            hit_rate = (self.cache_hits /
                        (self.cache_hits +
                         self.cache_misses)) if (self.cache_hits +
                                                 self.cache_misses) > 0 else 0

            return {
                "total_specs_cached": total_cached,
                "valid_specs_cached": valid_cached,
                "expired_specs": total_cached - valid_cached,
                "total_endpoints_cached": total_endpoints,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": hit_rate,
                "total_accesses": total_accesses,
                "ttl_seconds": self.cache_ttl,
                "cache_db_path": self.cache_db_path
            }

        except Exception as e:
            logger.error(f"Cache stats error: {str(e)}")
            return {}

    async def clear_cache(
            self, spec_url: Optional[str] = None) -> Dict[str, Any]:
        """Clear cached specifications"""

        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            if spec_url:
                # Clear specific spec and its endpoints
                cursor.execute(
                    'DELETE FROM openapi_cache WHERE spec_url = ?', (spec_url,))
                cache_cleared = cursor.rowcount
                cursor.execute(
                    'DELETE FROM parsed_endpoints WHERE spec_url = ?', (spec_url,))
                endpoints_cleared = cursor.rowcount
                logger.info(f"Cleared cache for {spec_url}")
            else:
                # Clear all cached specs and endpoints
                cursor.execute('DELETE FROM openapi_cache')
                cache_cleared = cursor.rowcount
                cursor.execute('DELETE FROM parsed_endpoints')
                endpoints_cleared = cursor.rowcount
                logger.info("Cleared all cached OpenAPI specs and endpoints")

            conn.commit()
            conn.close()

            return {
                "cache_cleared_count": cache_cleared,
                "endpoints_cleared_count": endpoints_cleared,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            return {"error": str(e)}

    async def cleanup_expired_cache(self) -> Dict[str, Any]:
        """Clean up expired cache entries"""

        try:
            conn = sqlite3.connect(self.cache_db_path)
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


async def fetch_spec(
        spec_url: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """Convenience function to fetch OpenAPI specification with caching"""
    async with OpenAPISpecParser() as parser:
        return await parser.fetch_openapi_spec(spec_url, force_refresh)
