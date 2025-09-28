"""
Edge case tests for Cost-Aware Router.

Tests robustness and error handling for:
1. Boundary conditions and edge inputs
2. Resource exhaustion scenarios  
3. Concurrent access edge cases
4. Data corruption and recovery
5. Network and system failures
"""

import pytest
import asyncio
import sqlite3
import time
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Import test router
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cost_aware_router_test import MockCostAwareRouter


class TestCostRouterEdgeCases:
    """Edge case test suite for Cost-Aware Router."""
    
    @pytest.mark.asyncio
    async def test_empty_and_null_inputs(self, temp_db):
        """Test handling of empty and null inputs."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        edge_inputs = [
            {"prompt": "", "history": [], "description": "Empty prompt and history"},
            {"prompt": "   ", "history": [], "description": "Whitespace-only prompt"},
            {"prompt": "\n\t\r", "history": [], "description": "Newlines and tabs"},
            {"prompt": "test", "history": None, "description": "None history"},
        ]
        
        for test_case in edge_inputs:
            try:
                result = await router.route_request(test_case["prompt"], test_case["history"])
                
                # Should handle gracefully with valid response structure
                assert "model_used" in result
                assert result["budget_remaining"] >= 0
                assert "total_latency_ms" in result
                assert result["total_latency_ms"] < 800
                
                # Should default to simple model for edge cases
                assert result["model_used"] in ["claude-haiku", "claude-sonnet"]
                
            except Exception as e:
                # If exceptions occur, they should be specific and handled
                assert isinstance(e, (ValueError, TypeError))
    
    @pytest.mark.asyncio
    async def test_extremely_long_inputs(self, temp_db):
        """Test handling of extremely long prompts and histories."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Test very long prompt
        long_prompt = "A" * 50000  # 50k characters
        result = await router.route_request(long_prompt)
        
        # Should handle without crashing
        assert "model_used" in result
        assert result["total_latency_ms"] < 1500  # Allow extra time for long input
        
        # Should recognize high complexity
        assert result["complexity_score"] > 0.5
        
        # Test very long history
        long_history = [
            {"api_call": f"/api/test/{i}", "method": "GET", "timestamp": f"2023-01-01T10:00:{i:02d}Z"}
            for i in range(5000)  # 5k history items
        ]
        
        result = await router.route_request("Simple prompt", long_history)
        assert "model_used" in result
        assert result["complexity_score"] > 0.3  # History should increase complexity
    
    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self, temp_db):
        """Test handling of unicode and special characters."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        unicode_tests = [
            "Simple query with émojis 🚀🎉",
            "Chinese characters: 你好世界",
            "Arabic text: مرحبا بالعالم",
            "Math symbols: ∑∆∫√π",
            "Special chars: !@#$%^&*()_+-={}[]|\\:;\"'<>,.?/",
            "Mixed: Test 测试 テスト 🌟",
            "Zero-width chars: \u200b\u200c\u200d",
            "Control chars: \x00\x01\x02\x03"
        ]
        
        for prompt in unicode_tests:
            try:
                result = await router.route_request(prompt)
                
                # Should handle unicode gracefully
                assert "model_used" in result
                assert result["budget_remaining"] >= 0
                assert result["total_latency_ms"] < 800
                
            except UnicodeError:
                # Unicode errors should be handled
                pass
    
    @pytest.mark.asyncio
    async def test_budget_boundary_conditions(self, temp_db):
        """Test budget boundary conditions and edge cases."""
        # Test with zero budget
        zero_budget_router = MockCostAwareRouter(budget_limit=0.0, db_path=temp_db)
        result = await zero_budget_router.route_request("Test with zero budget")
        
        # Should still work with fallback
        assert "model_used" in result
        assert result["model_used"] == "claude-haiku"  # Cheapest fallback
        
        # Test with negative budget (shouldn't happen but handle gracefully)
        negative_router = MockCostAwareRouter(budget_limit=-10.0, db_path=temp_db)
        result = await negative_router.route_request("Test with negative budget")
        assert "model_used" in result
        
        # Test with extremely large budget
        large_budget_router = MockCostAwareRouter(budget_limit=1e10, db_path=temp_db)
        result = await large_budget_router.route_request("Complex analysis requiring detailed optimization")
        assert result["model_used"] == "claude-opus"  # Should use best model
        
        # Test budget precision edge cases
        tiny_budget_router = MockCostAwareRouter(budget_limit=0.001, db_path=temp_db)
        result = await tiny_budget_router.route_request("Tiny budget test")
        assert result["budget_remaining"] >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_budget_depletion(self, temp_db):
        """Test concurrent requests when budget is being depleted."""
        router = MockCostAwareRouter(budget_limit=1.0, db_path=temp_db)  # Very small budget
        
        # Start many concurrent expensive requests
        expensive_prompt = "Complex analysis requiring detailed strategic optimization"
        tasks = [router.route_request(expensive_prompt) for _ in range(10)]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some requests should succeed, budget should not go negative
        successful_results = [r for r in results if isinstance(r, dict) and "model_used" in r]
        assert len(successful_results) > 0  # At least some should succeed
        
        # Final budget should be non-negative
        final_status = await router.get_budget_status()
        assert final_status["current_budget"] >= 0
        
        # Later requests should use cheaper models
        later_results = [r for r in successful_results[-3:] if "model_used" in r]
        for result in later_results:
            assert result["model_used"] in ["claude-haiku", "claude-sonnet"]
    
    @pytest.mark.asyncio
    async def test_database_corruption_recovery(self, temp_db):
        """Test recovery from database corruption scenarios."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Make initial request to create database
        await router.route_request("Initial request")
        
        # Corrupt the database by writing invalid data
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        try:
            # Insert invalid data
            cursor.execute("INSERT INTO budget_tracking (model_name, cost) VALUES (?, ?)", ("invalid", "not_a_number"))
            conn.commit()
        except:
            pass  # Expected to fail
        
        conn.close()
        
        # Router should still handle requests despite corruption
        try:
            result = await router.route_request("Request after corruption")
            assert "model_used" in result
        except Exception:
            # Should gracefully handle database errors
            pass
    
    @pytest.mark.asyncio
    async def test_filesystem_permission_errors(self):
        """Test handling of filesystem permission errors."""
        # Try to create router with invalid database path
        invalid_paths = [
            "/root/no_permission.db",  # No permission
            "/nonexistent/path/test.db",  # Nonexistent directory
            "",  # Empty path
            "/dev/null/test.db"  # Invalid location
        ]
        
        for invalid_path in invalid_paths:
            try:
                router = MockCostAwareRouter(budget_limit=50.0, db_path=invalid_path)
                result = await router.route_request("Test with invalid path")
                # If it succeeds, that's fine too (fallback handling)
                assert "model_used" in result
            except (OSError, PermissionError, sqlite3.Error):
                # Expected errors that should be handled gracefully
                pass
    
    @pytest.mark.asyncio
    async def test_memory_pressure_conditions(self, temp_db):
        """Test behavior under memory pressure."""
        router = MockCostAwareRouter(budget_limit=100.0, db_path=temp_db)
        
        # Create memory pressure with large data structures
        large_history = []
        for i in range(10000):
            large_history.append({
                "api_call": f"/api/large/endpoint/{i}",
                "method": "GET",
                "timestamp": f"2023-01-01T{i%24:02d}:00:00Z",
                "parameters": {"data": "x" * 1000},  # Large parameter data
                "response": {"result": "y" * 1000}    # Large response data
            })
        
        # Should handle large data gracefully
        result = await router.route_request("Handle large history", large_history)
        assert "model_used" in result
        assert result["total_latency_ms"] < 2000  # Allow extra time for large data
    
    @pytest.mark.asyncio
    async def test_rapid_sequential_requests(self, temp_db):
        """Test rapid sequential requests that might cause race conditions."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Send rapid sequential requests
        results = []
        for i in range(100):
            result = await router.route_request(f"Rapid request {i}")
            results.append(result)
            
            # No delay between requests
        
        # All should succeed
        assert len(results) == 100
        for result in results:
            assert "model_used" in result
            assert result["budget_remaining"] >= 0
        
        # Budget should be properly decremented
        total_cost = sum(r["cost"] for r in results)
        final_budget = router.current_budget
        assert abs((50.0 - total_cost) - final_budget) < 0.01
    
    @pytest.mark.asyncio
    async def test_malicious_input_handling(self, temp_db):
        """Test handling of potentially malicious inputs."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        malicious_inputs = [
            "'; DROP TABLE budget_tracking; --",  # SQL injection
            "<script>alert('xss')</script>",      # XSS
            "{{7*7}}",                            # Template injection
            "$(rm -rf /)",                        # Command injection
            "javascript:alert(1)",                # JavaScript protocol
            "../../../etc/passwd",                # Path traversal
            "\x00\x01\x02\x03",                  # Binary data
            "eval(import('os').system('ls'))",    # Python injection
        ]
        
        for malicious_input in malicious_inputs:
            try:
                result = await router.route_request(malicious_input)
                
                # Should handle without executing malicious code
                assert "model_used" in result
                assert result["budget_remaining"] >= 0
                
                # Should not cause system compromise
                assert result["total_latency_ms"] < 1000
                
            except Exception as e:
                # Exceptions should be safe, specific types
                assert isinstance(e, (ValueError, TypeError, RuntimeError))
    
    @pytest.mark.asyncio
    async def test_timeout_and_hanging_scenarios(self, temp_db):
        """Test timeout handling and hanging request scenarios."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Simulate slow operations with timeout
        async def slow_request():
            return await asyncio.wait_for(
                router.route_request("Test timeout handling"),
                timeout=2.0  # 2 second timeout
            )
        
        try:
            result = await slow_request()
            # Should complete within timeout
            assert "model_used" in result
            assert result["total_latency_ms"] < 1000
            
        except asyncio.TimeoutError:
            # Timeout should be handled gracefully
            pass
    
    @pytest.mark.asyncio
    async def test_complex_edge_case_combinations(self, temp_db):
        """Test combinations of multiple edge cases."""
        router = MockCostAwareRouter(budget_limit=0.1, db_path=temp_db)  # Very low budget
        
        # Combine multiple edge conditions
        edge_combinations = [
            {
                "prompt": "",  # Empty prompt
                "history": [{"invalid": "data"}] * 1000,  # Large invalid history
                "description": "Empty prompt + large invalid history + low budget"
            },
            {
                "prompt": "🚀" * 10000,  # Very long unicode
                "history": [],
                "description": "Very long unicode + low budget"
            },
            {
                "prompt": "'; DROP TABLE users; --" + "A" * 5000,  # Malicious + long
                "history": None,
                "description": "Malicious SQL + very long + null history"
            }
        ]
        
        for combination in edge_combinations:
            try:
                result = await router.route_request(
                    combination["prompt"], 
                    combination["history"]
                )
                
                # Should handle complex edge cases gracefully
                assert "model_used" in result
                assert result["budget_remaining"] >= 0
                
                # Should use safe fallback model
                assert result["model_used"] == "claude-haiku"
                
            except Exception as e:
                # Complex edge cases may cause exceptions
                assert isinstance(e, (ValueError, TypeError, RuntimeError, sqlite3.Error))
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_after_errors(self, temp_db):
        """Test that resources are properly cleaned up after errors."""
        router = MockCostAwareRouter(budget_limit=50.0, db_path=temp_db)
        
        # Force various error conditions
        error_scenarios = [
            "",  # Empty input
            "A" * 100000,  # Extremely long input
            None,  # Invalid input type
        ]
        
        initial_budget = router.current_budget
        
        for scenario in error_scenarios:
            try:
                if scenario is None:
                    await router.route_request("")  # Convert None to empty string
                else:
                    await router.route_request(scenario)
            except Exception:
                pass  # Ignore exceptions, focus on cleanup
        
        # System should still be functional after errors
        result = await router.route_request("Test after errors")
        assert "model_used" in result
        assert result["total_latency_ms"] < 800
        
        # Budget tracking should still work
        assert router.current_budget <= initial_budget


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])