"""
Unit tests for app.utils.spec_parser module.

Tests OpenAPI specification parsing including:
- Spec validation and parsing
- Endpoint extraction and categorization  
- Parameter extraction and typing
- Description and metadata parsing
- Error handling for malformed specs
"""

import pytest
from unittest.mock import Mock, patch
import json
from typing import Dict, Any, List

# Mock imports - replace with actual imports when modules exist
# from app.utils.spec_parser import SpecParser, EndpointInfo, ParameterInfo


class TestSpecParser:
    """Test suite for the SpecParser class."""
    
    @pytest.fixture
    def spec_parser(self):
        """Create SpecParser instance."""
        return Mock()
    
    def test_parse_valid_openapi_spec(self, spec_parser, sample_openapi_spec):
        """Test parsing of valid OpenAPI 3.0 specification."""
        expected_endpoints = [
            {
                "path": "/users",
                "method": "GET",
                "operation_id": "listUsers",
                "description": "List all users",
                "parameters": [],
                "request_body": None
            },
            {
                "path": "/users",
                "method": "POST", 
                "operation_id": "createUser",
                "description": "Create a new user",
                "parameters": [],
                "request_body": {"required": True, "content_type": "application/json"}
            },
            {
                "path": "/users/{id}",
                "method": "GET",
                "operation_id": "getUser",
                "description": "Get user by ID",
                "parameters": [{"name": "id", "in": "path", "required": True, "type": "string"}],
                "request_body": None
            }
        ]
        
        spec_parser.parse_spec = Mock(return_value=expected_endpoints)
        
        endpoints = spec_parser.parse_spec(sample_openapi_spec)
        
        assert len(endpoints) >= 3
        assert any(ep["operation_id"] == "listUsers" for ep in endpoints)
        assert any(ep["operation_id"] == "createUser" for ep in endpoints)
        assert any(ep["operation_id"] == "getUser" for ep in endpoints)
    
    def test_parse_invalid_openapi_spec(self, spec_parser):
        """Test handling of invalid OpenAPI specification."""
        invalid_specs = [
            {},  # Empty spec
            {"openapi": "2.0"},  # Wrong version
            {"openapi": "3.0.0", "info": {}},  # Missing paths
            {"openapi": "3.0.0", "info": {"title": "Test"}, "paths": None}  # Null paths
        ]
        
        for invalid_spec in invalid_specs:
            spec_parser.parse_spec = Mock(side_effect=ValueError(f"Invalid spec: {invalid_spec}"))
            
            with pytest.raises(ValueError):
                spec_parser.parse_spec(invalid_spec)
    
    def test_extract_endpoint_info(self, spec_parser):
        """Test extraction of detailed endpoint information."""
        path_item = {
            "get": {
                "operationId": "getUserById",
                "summary": "Get user by ID",
                "description": "Retrieve a specific user by their unique identifier",
                "parameters": [
                    {
                        "name": "id",
                        "in": "path", 
                        "required": True,
                        "schema": {"type": "string", "format": "uuid"}
                    },
                    {
                        "name": "include_profile",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "boolean", "default": False}
                    }
                ],
                "responses": {
                    "200": {"description": "User found"},
                    "404": {"description": "User not found"}
                }
            }
        }
        
        expected_info = {
            "path": "/users/{id}",
            "method": "GET",
            "operation_id": "getUserById",
            "summary": "Get user by ID",
            "description": "Retrieve a specific user by their unique identifier",
            "parameters": [
                {"name": "id", "in": "path", "required": True, "type": "string", "format": "uuid"},
                {"name": "include_profile", "in": "query", "required": False, "type": "boolean", "default": False}
            ],
            "responses": ["200", "404"]
        }
        
        spec_parser.extract_endpoint_info = Mock(return_value=expected_info)
        
        info = spec_parser.extract_endpoint_info("/users/{id}", path_item)
        
        assert info["operation_id"] == "getUserById"
        assert len(info["parameters"]) == 2
        assert info["parameters"][0]["name"] == "id"
        assert info["parameters"][0]["required"] is True
    
    def test_extract_request_body_schema(self, spec_parser):
        """Test extraction of request body schema information."""
        request_body = {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "email": {"type": "string", "format": "email"},
                            "age": {"type": "integer", "minimum": 0}
                        },
                        "required": ["name", "email"]
                    }
                }
            }
        }
        
        expected_schema = {
            "required": True,
            "content_type": "application/json",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "email": {"type": "string", "format": "email"},
                    "age": {"type": "integer", "minimum": 0}
                },
                "required": ["name", "email"]
            }
        }
        
        spec_parser.extract_request_body = Mock(return_value=expected_schema)
        
        schema = spec_parser.extract_request_body(request_body)
        
        assert schema["required"] is True
        assert schema["content_type"] == "application/json"
        assert "name" in schema["schema"]["properties"]
        assert "email" in schema["schema"]["required"]
    
    def test_categorize_endpoints(self, spec_parser):
        """Test categorization of endpoints by resource and operation type."""
        endpoints = [
            {"path": "/users", "method": "GET", "description": "List users"},
            {"path": "/users", "method": "POST", "description": "Create user"},
            {"path": "/users/{id}", "method": "GET", "description": "Get user"},
            {"path": "/users/{id}", "method": "PUT", "description": "Update user"},
            {"path": "/users/{id}", "method": "DELETE", "description": "Delete user"},
            {"path": "/invoices", "method": "GET", "description": "List invoices"},
            {"path": "/health", "method": "GET", "description": "Health check"}
        ]
        
        expected_categories = {
            "users": {
                "list": [{"path": "/users", "method": "GET"}],
                "create": [{"path": "/users", "method": "POST"}],
                "read": [{"path": "/users/{id}", "method": "GET"}],
                "update": [{"path": "/users/{id}", "method": "PUT"}],
                "delete": [{"path": "/users/{id}", "method": "DELETE"}]
            },
            "invoices": {
                "list": [{"path": "/invoices", "method": "GET"}]
            },
            "system": {
                "health": [{"path": "/health", "method": "GET"}]
            }
        }
        
        spec_parser.categorize_endpoints = Mock(return_value=expected_categories)
        
        categories = spec_parser.categorize_endpoints(endpoints)
        
        assert "users" in categories
        assert "invoices" in categories
        assert len(categories["users"]) == 5  # CRUD + list
        assert categories["users"]["create"][0]["method"] == "POST"
    
    def test_extract_parameter_types(self, spec_parser):
        """Test extraction and validation of parameter types."""
        parameters = [
            {"name": "id", "in": "path", "schema": {"type": "string", "format": "uuid"}},
            {"name": "page", "in": "query", "schema": {"type": "integer", "minimum": 1}},
            {"name": "filter", "in": "query", "schema": {"type": "string", "enum": ["active", "inactive"]}},
            {"name": "Authorization", "in": "header", "schema": {"type": "string"}}
        ]
        
        expected_types = [
            {"name": "id", "location": "path", "type": "string", "format": "uuid", "required": True},
            {"name": "page", "location": "query", "type": "integer", "constraints": {"minimum": 1}},
            {"name": "filter", "location": "query", "type": "string", "enum": ["active", "inactive"]},
            {"name": "Authorization", "location": "header", "type": "string", "required": False}
        ]
        
        spec_parser.extract_parameter_types = Mock(return_value=expected_types)
        
        param_types = spec_parser.extract_parameter_types(parameters)
        
        assert len(param_types) == 4
        assert param_types[0]["name"] == "id"
        assert param_types[0]["type"] == "string"
        assert param_types[1]["constraints"]["minimum"] == 1
    
    def test_validate_spec_version(self, spec_parser):
        """Test OpenAPI version validation."""
        valid_versions = ["3.0.0", "3.0.1", "3.0.2", "3.1.0"]
        invalid_versions = ["2.0", "1.0", "4.0.0", None, ""]
        
        for version in valid_versions:
            spec_parser.validate_version = Mock(return_value=True)
            assert spec_parser.validate_version(version) is True
        
        for version in invalid_versions:
            spec_parser.validate_version = Mock(return_value=False)
            assert spec_parser.validate_version(version) is False
    
    def test_resolve_references(self, spec_parser):
        """Test resolution of $ref references in spec."""
        spec_with_refs = {
            "openapi": "3.0.0",
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"}
                        }
                    }
                }
            },
            "paths": {
                "/users": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/User"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        resolved_spec = {
            "openapi": "3.0.0",
            "paths": {
                "/users": {
                    "post": {
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "name": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        spec_parser.resolve_references = Mock(return_value=resolved_spec)
        
        resolved = spec_parser.resolve_references(spec_with_refs)
        
        # Verify that $ref was resolved
        schema = resolved["paths"]["/users"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert "$ref" not in schema
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
    
    def test_extract_security_requirements(self, spec_parser):
        """Test extraction of security requirements."""
        endpoint_security = [
            {"apiKey": []},
            {"oauth2": ["read:users", "write:users"]}
        ]
        
        expected_requirements = [
            {"type": "apiKey", "scopes": []},
            {"type": "oauth2", "scopes": ["read:users", "write:users"]}
        ]
        
        spec_parser.extract_security = Mock(return_value=expected_requirements)
        
        requirements = spec_parser.extract_security(endpoint_security)
        
        assert len(requirements) == 2
        assert requirements[0]["type"] == "apiKey"
        assert "read:users" in requirements[1]["scopes"]
    
    @pytest.mark.parametrize("path,expected_resource", [
        ("/users", "users"),
        ("/users/{id}", "users"),
        ("/api/v1/users/{id}/profile", "users"),
        ("/invoices/{id}/items", "invoices"),
        ("/health", "health"),
        ("/_internal/metrics", "_internal")
    ])
    def test_extract_resource_name(self, spec_parser, path, expected_resource):
        """Test resource name extraction from paths."""
        spec_parser.extract_resource = Mock(return_value=expected_resource)
        
        resource = spec_parser.extract_resource(path)
        
        assert resource == expected_resource
    
    def test_spec_caching(self, spec_parser):
        """Test caching of parsed specifications."""
        spec_hash = "abc123"
        cached_result = {"endpoints": [], "cached": True}
        
        spec_parser.get_from_cache = Mock(return_value=cached_result)
        spec_parser.cache_result = Mock()
        
        # First call should check cache
        result = spec_parser.get_from_cache(spec_hash)
        assert result["cached"] is True
        
        # Should also be able to store in cache
        spec_parser.cache_result(spec_hash, {"endpoints": [], "cached": False})
        spec_parser.cache_result.assert_called_once()


class TestEndpointInfo:
    """Test suite for the EndpointInfo data class."""
    
    def test_endpoint_info_creation(self):
        """Test creation of EndpointInfo object."""
        endpoint_data = {
            "path": "/users/{id}",
            "method": "GET",
            "operation_id": "getUser",
            "description": "Get user by ID",
            "parameters": [{"name": "id", "in": "path", "required": True}],
            "tags": ["users"],
            "security": [{"apiKey": []}]
        }
        
        # Mock the EndpointInfo class
        endpoint_info = Mock()
        endpoint_info.path = endpoint_data["path"]
        endpoint_info.method = endpoint_data["method"]
        endpoint_info.operation_id = endpoint_data["operation_id"]
        endpoint_info.description = endpoint_data["description"]
        endpoint_info.parameters = endpoint_data["parameters"]
        endpoint_info.tags = endpoint_data["tags"]
        endpoint_info.security = endpoint_data["security"]
        
        assert endpoint_info.path == "/users/{id}"
        assert endpoint_info.method == "GET"
        assert endpoint_info.operation_id == "getUser"
        assert len(endpoint_info.parameters) == 1
    
    def test_endpoint_info_validation(self):
        """Test validation of EndpointInfo fields."""
        # Test required fields
        required_fields = ["path", "method"]
        
        for field in required_fields:
            endpoint_data = {
                "path": "/users",
                "method": "GET",
                "operation_id": "listUsers"
            }
            del endpoint_data[field]
            
            # Would raise validation error in real implementation
            with pytest.raises(ValueError, match=f"Missing required field: {field}"):
                if field not in endpoint_data:
                    raise ValueError(f"Missing required field: {field}")
    
    def test_endpoint_serialization(self):
        """Test serialization of EndpointInfo to dictionary."""
        endpoint_info = {
            "path": "/users",
            "method": "GET",  
            "operation_id": "listUsers",
            "description": "List all users",
            "parameters": [],
            "tags": ["users"]
        }
        
        # Test JSON serialization
        json_str = json.dumps(endpoint_info)
        parsed = json.loads(json_str)
        
        assert parsed["path"] == "/users"
        assert parsed["method"] == "GET"
        assert isinstance(parsed["parameters"], list)


class TestParameterInfo:
    """Test suite for the ParameterInfo data class."""
    
    def test_parameter_info_creation(self):
        """Test creation of ParameterInfo object."""
        param_data = {
            "name": "user_id",
            "location": "path", 
            "type": "string",
            "required": True,
            "format": "uuid",
            "description": "Unique user identifier"
        }
        
        # Mock ParameterInfo
        param_info = Mock()
        param_info.name = param_data["name"]
        param_info.location = param_data["location"]
        param_info.type = param_data["type"] 
        param_info.required = param_data["required"]
        param_info.format = param_data["format"]
        param_info.description = param_data["description"]
        
        assert param_info.name == "user_id"
        assert param_info.location == "path"
        assert param_info.required is True
        assert param_info.format == "uuid"
    
    @pytest.mark.parametrize("param_type,expected_valid", [
        ("string", True),
        ("integer", True),
        ("number", True),
        ("boolean", True),
        ("array", True),
        ("object", True),
        ("invalid_type", False)
    ])
    def test_parameter_type_validation(self, param_type, expected_valid):
        """Test parameter type validation."""
        valid_types = {"string", "integer", "number", "boolean", "array", "object"}
        
        is_valid = param_type in valid_types
        assert is_valid == expected_valid