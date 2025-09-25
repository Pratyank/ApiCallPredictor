"""
Sample data fixtures for testing opensesame-predictor.

Provides realistic test data including:
- OpenAPI specifications for various APIs
- User event sequences  
- Prediction scenarios
- Mock responses
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta


def get_stripe_api_spec() -> Dict[str, Any]:
    """Sample Stripe API specification."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Stripe API", "version": "2023-10-16"},
        "paths": {
            "/customers": {
                "get": {
                    "operationId": "listCustomers",
                    "description": "Returns a list of your customers"
                },
                "post": {
                    "operationId": "createCustomer", 
                    "description": "Creates a new customer object"
                }
            },
            "/customers/{customer_id}": {
                "get": {
                    "operationId": "retrieveCustomer",
                    "description": "Retrieves a customer object"
                },
                "put": {
                    "operationId": "updateCustomer",
                    "description": "Updates the specified customer"
                },
                "delete": {
                    "operationId": "deleteCustomer",
                    "description": "Permanently deletes a customer"
                }
            },
            "/payment_intents": {
                "get": {
                    "operationId": "listPaymentIntents",
                    "description": "Returns a list of PaymentIntents"
                },
                "post": {
                    "operationId": "createPaymentIntent",
                    "description": "Creates a PaymentIntent object"
                }
            },
            "/payment_intents/{payment_intent_id}": {
                "get": {
                    "operationId": "retrievePaymentIntent",
                    "description": "Retrieves the details of a PaymentIntent"
                },
                "put": {
                    "operationId": "updatePaymentIntent", 
                    "description": "Updates properties on a PaymentIntent object"
                }
            },
            "/payment_intents/{payment_intent_id}/confirm": {
                "post": {
                    "operationId": "confirmPaymentIntent",
                    "description": "Confirm that your customer intends to pay"
                }
            },
            "/invoices": {
                "get": {
                    "operationId": "listInvoices",
                    "description": "List all invoices"
                },
                "post": {
                    "operationId": "createInvoice",
                    "description": "Create an invoice"
                }
            }
        }
    }


def get_github_api_spec() -> Dict[str, Any]:
    """Sample GitHub API specification."""
    return {
        "openapi": "3.0.0", 
        "info": {"title": "GitHub REST API", "version": "v3"},
        "paths": {
            "/user": {
                "get": {
                    "operationId": "getAuthenticatedUser",
                    "description": "Get the authenticated user"
                }
            },
            "/user/repos": {
                "get": {
                    "operationId": "listReposForAuthenticatedUser", 
                    "description": "List repositories for the authenticated user"
                },
                "post": {
                    "operationId": "createRepoForAuthenticatedUser",
                    "description": "Create a repository for the authenticated user"
                }
            },
            "/repos/{owner}/{repo}": {
                "get": {
                    "operationId": "getRepo",
                    "description": "Get a repository"
                },
                "patch": {
                    "operationId": "updateRepo",
                    "description": "Update a repository" 
                },
                "delete": {
                    "operationId": "deleteRepo",
                    "description": "Delete a repository"
                }
            },
            "/repos/{owner}/{repo}/pulls": {
                "get": {
                    "operationId": "listPullRequests",
                    "description": "List pull requests"
                },
                "post": {
                    "operationId": "createPullRequest",
                    "description": "Create a pull request"
                }
            },
            "/repos/{owner}/{repo}/pulls/{pull_number}": {
                "get": {
                    "operationId": "getPullRequest",
                    "description": "Get a pull request"
                },
                "patch": {
                    "operationId": "updatePullRequest",
                    "description": "Update a pull request"
                }
            },
            "/repos/{owner}/{repo}/pulls/{pull_number}/merge": {
                "put": {
                    "operationId": "mergePullRequest",
                    "description": "Merge a pull request"
                }
            },
            "/repos/{owner}/{repo}/issues": {
                "get": {
                    "operationId": "listIssues", 
                    "description": "List repository issues"
                },
                "post": {
                    "operationId": "createIssue",
                    "description": "Create an issue"
                }
            }
        }
    }


def get_user_management_workflow() -> List[Dict[str, Any]]:
    """Sample user management workflow sequence."""
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    
    return [
        {
            "endpoint": "/users",
            "method": "GET",
            "timestamp": base_time.isoformat(),
            "response_time": 150,
            "status_code": 200
        },
        {
            "endpoint": "/users/search", 
            "method": "GET",
            "timestamp": (base_time + timedelta(seconds=15)).isoformat(),
            "response_time": 200,
            "status_code": 200,
            "query_params": {"q": "john@example.com"}
        },
        {
            "endpoint": "/users/123",
            "method": "GET", 
            "timestamp": (base_time + timedelta(seconds=30)).isoformat(),
            "response_time": 100,
            "status_code": 200
        },
        {
            "endpoint": "/users/123",
            "method": "PUT",
            "timestamp": (base_time + timedelta(seconds=45)).isoformat(),
            "response_time": 250,
            "status_code": 200,
            "request_body": {"name": "John Updated", "email": "john.new@example.com"}
        },
        {
            "endpoint": "/users/123",
            "method": "GET",
            "timestamp": (base_time + timedelta(seconds=60)).isoformat(), 
            "response_time": 120,
            "status_code": 200
        }
    ]


def get_ecommerce_workflow() -> List[Dict[str, Any]]:
    """Sample e-commerce shopping workflow."""
    base_time = datetime(2023, 1, 1, 14, 0, 0)
    
    return [
        {
            "endpoint": "/products",
            "method": "GET",
            "timestamp": base_time.isoformat(),
            "response_time": 180
        },
        {
            "endpoint": "/products/search",
            "method": "GET", 
            "timestamp": (base_time + timedelta(seconds=10)).isoformat(),
            "response_time": 220,
            "query_params": {"q": "laptop", "category": "electronics"}
        },
        {
            "endpoint": "/products/laptop-123",
            "method": "GET",
            "timestamp": (base_time + timedelta(seconds=25)).isoformat(),
            "response_time": 150
        },
        {
            "endpoint": "/cart/items",
            "method": "POST",
            "timestamp": (base_time + timedelta(seconds=40)).isoformat(),
            "response_time": 300,
            "request_body": {"product_id": "laptop-123", "quantity": 1}
        },
        {
            "endpoint": "/cart",
            "method": "GET",
            "timestamp": (base_time + timedelta(seconds=45)).isoformat(),
            "response_time": 100
        },
        {
            "endpoint": "/checkout",
            "method": "GET",
            "timestamp": (base_time + timedelta(seconds=60)).isoformat(),
            "response_time": 200
        }
    ]


def get_payment_workflow() -> List[Dict[str, Any]]:
    """Sample payment processing workflow using Stripe."""
    base_time = datetime(2023, 1, 1, 16, 0, 0)
    
    return [
        {
            "endpoint": "/customers",
            "method": "GET",
            "timestamp": base_time.isoformat(),
            "response_time": 120
        },
        {
            "endpoint": "/customers",
            "method": "POST", 
            "timestamp": (base_time + timedelta(seconds=10)).isoformat(),
            "response_time": 400,
            "request_body": {"email": "customer@example.com", "name": "John Doe"}
        },
        {
            "endpoint": "/payment_intents",
            "method": "POST",
            "timestamp": (base_time + timedelta(seconds=20)).isoformat(),
            "response_time": 300,
            "request_body": {"amount": 2000, "currency": "usd", "customer": "cus_123"}
        },
        {
            "endpoint": "/payment_intents/pi_123",
            "method": "GET",
            "timestamp": (base_time + timedelta(seconds=30)).isoformat(),
            "response_time": 150
        },
        {
            "endpoint": "/payment_intents/pi_123/confirm",
            "method": "POST",
            "timestamp": (base_time + timedelta(seconds=45)).isoformat(),
            "response_time": 800,
            "request_body": {"payment_method": "pm_card_visa"}
        }
    ]


def get_github_pr_workflow() -> List[Dict[str, Any]]:
    """Sample GitHub pull request workflow."""
    base_time = datetime(2023, 1, 1, 9, 0, 0)
    
    return [
        {
            "endpoint": "/user/repos",
            "method": "GET", 
            "timestamp": base_time.isoformat(),
            "response_time": 200
        },
        {
            "endpoint": "/repos/owner/repo",
            "method": "GET",
            "timestamp": (base_time + timedelta(seconds=5)).isoformat(),
            "response_time": 150
        },
        {
            "endpoint": "/repos/owner/repo/pulls",
            "method": "POST",
            "timestamp": (base_time + timedelta(seconds=20)).isoformat(),
            "response_time": 400,
            "request_body": {
                "title": "Add new feature",
                "head": "feature-branch",
                "base": "main"
            }
        },
        {
            "endpoint": "/repos/owner/repo/pulls/1",
            "method": "GET",
            "timestamp": (base_time + timedelta(seconds=30)).isoformat(),
            "response_time": 180
        },
        {
            "endpoint": "/repos/owner/repo/pulls/1",
            "method": "PATCH",
            "timestamp": (base_time + timedelta(seconds=45)).isoformat(),
            "response_time": 250,
            "request_body": {"body": "Updated PR description"}
        }
    ]


def get_prediction_test_cases() -> List[Dict[str, Any]]:
    """Test cases for prediction scenarios."""
    return [
        {
            "name": "user_creation_intent",
            "prompt": "Create a new user account",
            "context": "user_management",
            "expected_endpoints": ["/users", "/user/register", "/accounts"],
            "expected_methods": ["POST"],
            "confidence_threshold": 0.8
        },
        {
            "name": "user_update_following_read",
            "prompt": "Update the user profile",
            "recent_events": [
                {"endpoint": "/users/123", "method": "GET"}
            ],
            "expected_endpoints": ["/users/123", "/users/{id}"],
            "expected_methods": ["PUT", "PATCH"],
            "confidence_threshold": 0.9
        },
        {
            "name": "payment_processing_flow",
            "prompt": "Process the payment",
            "recent_events": [
                {"endpoint": "/cart", "method": "GET"},
                {"endpoint": "/checkout", "method": "GET"}
            ],
            "expected_endpoints": ["/payment_intents", "/charges", "/payments"],
            "expected_methods": ["POST"],
            "confidence_threshold": 0.85
        },
        {
            "name": "cold_start_scenario",
            "prompt": "",
            "recent_events": [],
            "expected_safe_endpoints": ["/health", "/users", "/products"],
            "expected_methods": ["GET"],
            "confidence_threshold": 0.6
        },
        {
            "name": "destructive_operation_blocked",
            "prompt": "Delete all user data permanently",
            "expected_predictions": [],
            "should_be_blocked": True,
            "block_reason": "destructive_operation"
        },
        {
            "name": "list_after_search",
            "prompt": "Show me the results", 
            "recent_events": [
                {"endpoint": "/users/search", "method": "GET", "query_params": {"q": "john"}}
            ],
            "expected_endpoints": ["/users", "/search/results"],
            "expected_methods": ["GET"],
            "confidence_threshold": 0.7
        },
        {
            "name": "confirm_after_create",
            "prompt": "Confirm the action",
            "recent_events": [
                {"endpoint": "/payment_intents", "method": "POST"}
            ],
            "expected_endpoints": ["/payment_intents/{id}/confirm", "/confirm"],
            "expected_methods": ["POST"],
            "confidence_threshold": 0.8
        }
    ]


def get_mock_ai_responses() -> Dict[str, List[Dict[str, Any]]]:
    """Mock AI layer responses for different scenarios."""
    return {
        "create_user": [
            {
                "endpoint": "/users",
                "method": "POST", 
                "confidence": 0.95,
                "reasoning": "User explicitly requested to create a new user account"
            },
            {
                "endpoint": "/accounts",
                "method": "POST",
                "confidence": 0.85,
                "reasoning": "Alternative endpoint for account creation"
            },
            {
                "endpoint": "/register",
                "method": "POST",
                "confidence": 0.80,
                "reasoning": "Common registration endpoint pattern"
            }
        ],
        "update_user": [
            {
                "endpoint": "/users/{id}",
                "method": "PUT",
                "confidence": 0.90, 
                "reasoning": "Standard RESTful update operation"
            },
            {
                "endpoint": "/users/{id}",
                "method": "PATCH",
                "confidence": 0.85,
                "reasoning": "Partial update operation"
            },
            {
                "endpoint": "/profile/update",
                "method": "POST", 
                "confidence": 0.75,
                "reasoning": "Alternative update endpoint pattern"
            }
        ],
        "process_payment": [
            {
                "endpoint": "/payment_intents",
                "method": "POST",
                "confidence": 0.90,
                "reasoning": "Stripe-style payment intent creation"
            },
            {
                "endpoint": "/charges",
                "method": "POST", 
                "confidence": 0.85,
                "reasoning": "Direct charge creation"
            },
            {
                "endpoint": "/payments",
                "method": "POST",
                "confidence": 0.80,
                "reasoning": "Generic payment processing endpoint"
            }
        ]
    }


def get_feature_test_data() -> Dict[str, Any]:
    """Test data for feature engineering."""
    return {
        "sequence_features": {
            "recent_events": get_user_management_workflow(),
            "expected_features": {
                "session_length": 5,
                "time_since_last": 0,
                "avg_response_time": 164.0,
                "method_diversity": 2,  # GET and PUT
                "endpoint_diversity": 2,  # /users and /users/{id}
                "workflow_stage": "modification"
            }
        },
        "similarity_features": {
            "prompt": "Create a new user account",
            "candidate_descriptions": [
                "Creates a new user object",
                "Updates an existing user", 
                "Deletes a user account",
                "Lists all invoices"
            ],
            "expected_similarities": [0.95, 0.3, 0.1, 0.05]
        },
        "transition_probabilities": {
            "sequence": ["/users", "/users/{id}", "/users/{id}"],
            "candidate": "/users/{id}/profile",
            "expected_bigram_prob": 0.4,
            "expected_trigram_prob": 0.7
        }
    }


def get_performance_test_scenarios() -> List[Dict[str, Any]]:
    """Performance testing scenarios.""" 
    return [
        {
            "name": "baseline_single_request",
            "concurrent_requests": 1,
            "duration_seconds": 10,
            "expected_avg_response_ms": 200,
            "expected_p95_response_ms": 400,
            "expected_throughput_rps": 5
        },
        {
            "name": "moderate_concurrent_load",
            "concurrent_requests": 10,
            "duration_seconds": 30,
            "expected_avg_response_ms": 400,
            "expected_p95_response_ms": 800,
            "expected_throughput_rps": 20
        },
        {
            "name": "high_concurrent_load", 
            "concurrent_requests": 50,
            "duration_seconds": 60,
            "expected_avg_response_ms": 800,
            "expected_p95_response_ms": 1500,
            "expected_throughput_rps": 50
        },
        {
            "name": "spike_load_test",
            "phases": [
                {"rps": 5, "duration": 30},   # Baseline
                {"rps": 100, "duration": 10}, # Spike  
                {"rps": 5, "duration": 30}    # Recovery
            ],
            "spike_degradation_threshold": 3.0,  # Max 3x response time increase
            "recovery_threshold": 1.2  # Should recover to within 20%
        }
    ]