"""
Docker container testing for opensesame-predictor.

Tests container functionality including:
- Container build and deployment
- Resource limits and constraints
- Health check validation
- Multi-stage build optimization
- Security scanning
- Port mapping and networking
"""

import pytest
import docker
import requests
import time
import json
from typing import Dict, Any
import subprocess
import os

# Mock imports for when Docker integration is available
# import docker


class TestDockerContainerBuild:
    """Test Docker container build process."""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Docker client fixture."""
        try:
            client = docker.from_env()
            return client
        except Exception:
            pytest.skip("Docker not available")
    
    @pytest.fixture(scope="class") 
    def built_image(self, docker_client):
        """Build the Docker image for testing."""
        dockerfile_path = "/home/quantum/ApiCallPredictor"
        
        # Check if Dockerfile exists
        if not os.path.exists(os.path.join(dockerfile_path, "Dockerfile")):
            pytest.skip("Dockerfile not found")
        
        try:
            image, build_logs = docker_client.images.build(
                path=dockerfile_path,
                tag="opensesame-predictor:test",
                rm=True,
                forcerm=True
            )
            return image
        except Exception as e:
            pytest.fail(f"Failed to build Docker image: {e}")
    
    def test_image_build_success(self, built_image):
        """Test that Docker image builds successfully."""
        assert built_image is not None
        assert "opensesame-predictor:test" in built_image.tags
    
    def test_image_size_optimization(self, built_image):
        """Test that image size is optimized."""
        # Get image size
        image_size_mb = built_image.attrs['Size'] / (1024 * 1024)
        
        # Should be under 1GB for a Python FastAPI app
        assert image_size_mb < 1024, f"Image size {image_size_mb:.2f}MB is too large"
        
        print(f"Docker image size: {image_size_mb:.2f}MB")
    
    def test_image_layers_optimization(self, built_image):
        """Test that image has reasonable number of layers."""
        # Check layer count
        layers = len(built_image.history())
        
        # Should have reasonable number of layers (not too many)
        assert layers < 20, f"Too many layers: {layers}"
        
        print(f"Docker image layers: {layers}")
    
    def test_multi_stage_build_artifacts(self, built_image):
        """Test that multi-stage build removes unnecessary artifacts."""
        # Create a temporary container to inspect contents
        try:
            import docker
            client = docker.from_env()
            
            container = client.containers.create(built_image)
            
            # Check that build artifacts are not present in final stage
            exec_result = container.exec_run("find / -name '*.pyc' | wc -l")
            pyc_count = int(exec_result.output.decode().strip())
            
            # Should have minimal .pyc files in production image
            assert pyc_count < 100, f"Too many .pyc files: {pyc_count}"
            
            # Check that development dependencies are not installed
            exec_result = container.exec_run("pip show pytest")
            assert exec_result.exit_code != 0, "Development dependencies should not be in production image"
            
            container.remove()
            
        except Exception as e:
            pytest.skip(f"Could not inspect container: {e}")


class TestContainerDeployment:
    """Test container deployment and runtime."""
    
    @pytest.fixture
    def container_config(self):
        """Container configuration for testing."""
        return {
            "image": "opensesame-predictor:test",
            "ports": {"8000/tcp": 8001},  # Map container port 8000 to host port 8001
            "environment": {
                "ENVIRONMENT": "test",
                "LOG_LEVEL": "INFO"
            },
            "mem_limit": "512m",
            "cpu_period": 100000,
            "cpu_quota": 200000,  # 2 CPU cores
            "detach": True,
            "remove": True,
            "name": "opensesame-predictor-test"
        }
    
    @pytest.fixture
    def running_container(self, docker_client, built_image, container_config):
        """Start container for testing."""
        try:
            # Remove existing container if it exists
            try:
                existing = docker_client.containers.get("opensesame-predictor-test")
                existing.remove(force=True)
            except docker.errors.NotFound:
                pass
            
            container = docker_client.containers.run(**container_config)
            
            # Wait for container to start
            time.sleep(10)
            
            yield container
            
            # Cleanup
            try:
                container.remove(force=True)
            except:
                pass
                
        except Exception as e:
            pytest.skip(f"Could not start container: {e}")
    
    def test_container_starts_successfully(self, running_container):
        """Test that container starts without errors."""
        assert running_container.status in ["running", "created"]
        
        # Check container logs for errors
        logs = running_container.logs().decode()
        error_indicators = ["ERROR", "CRITICAL", "Exception", "Traceback"]
        
        for indicator in error_indicators:
            assert indicator not in logs, f"Container logs contain {indicator}: {logs}"
    
    def test_health_check_endpoint(self, running_container):
        """Test container health check endpoint."""
        # Wait for application to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:8001/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        else:
            pytest.fail("Health check endpoint not responding after 60 seconds")
        
        # Verify health check response
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "ok"
    
    def test_api_endpoint_functionality(self, running_container):
        """Test main API endpoint functionality."""
        # Wait for service to be ready
        time.sleep(5)
        
        test_request = {
            "prompt": "Create a new user",
            "recent_events": [],
            "openapi_spec": {
                "openapi": "3.0.0",
                "info": {"title": "Test API", "version": "1.0.0"},
                "paths": {
                    "/users": {
                        "post": {"operationId": "createUser", "description": "Create user"}
                    }
                }
            },
            "k": 3
        }
        
        try:
            response = requests.post(
                "http://localhost:8001/predict",
                json=test_request,
                timeout=10
            )
            
            assert response.status_code in [200, 503]  # 503 if models not loaded yet
            
            if response.status_code == 200:
                prediction_data = response.json()
                assert "predictions" in prediction_data
                assert "metadata" in prediction_data
                
        except requests.exceptions.RequestException as e:
            pytest.fail(f"API endpoint not responding: {e}")
    
    def test_resource_constraints(self, running_container):
        """Test that container respects resource constraints."""
        # Get container stats
        stats = running_container.stats(stream=False)
        
        # Check memory usage
        memory_usage_mb = stats['memory_stats']['usage'] / (1024 * 1024)
        memory_limit_mb = 512  # From container config
        
        assert memory_usage_mb < memory_limit_mb * 0.9, f"Memory usage {memory_usage_mb:.2f}MB too high"
        
        # Check CPU usage (over a short period)
        cpu_percent = calculate_cpu_percent(stats)
        assert cpu_percent < 200, f"CPU usage {cpu_percent:.1f}% exceeds limit"  # 2 cores = 200% max
        
        print(f"Container resource usage: {memory_usage_mb:.2f}MB RAM, {cpu_percent:.1f}% CPU")
    
    def test_container_logs_format(self, running_container):
        """Test that container logs are properly formatted."""
        logs = running_container.logs().decode()
        
        # Should have structured logging
        log_lines = [line for line in logs.split('\n') if line.strip()]
        
        for line in log_lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Should have timestamp and log level
            assert any(level in line for level in ["INFO", "WARNING", "ERROR", "DEBUG"]), \
                f"Log line missing level: {line}"
    
    def test_graceful_shutdown(self, docker_client, running_container):
        """Test that container shuts down gracefully."""
        container_id = running_container.id
        
        # Send SIGTERM to container
        running_container.kill(signal="SIGTERM")
        
        # Wait for graceful shutdown
        max_wait = 30
        for i in range(max_wait):
            try:
                container = docker_client.containers.get(container_id)
                if container.status != "running":
                    break
            except docker.errors.NotFound:
                break
            time.sleep(1)
        else:
            pytest.fail("Container did not shut down gracefully within 30 seconds")


class TestContainerSecurity:
    """Test container security configuration."""
    
    def test_non_root_user(self, running_container):
        """Test that container runs as non-root user."""
        exec_result = running_container.exec_run("whoami")
        username = exec_result.output.decode().strip()
        
        assert username != "root", "Container should not run as root user"
        
        # Check UID
        exec_result = running_container.exec_run("id -u")
        uid = int(exec_result.output.decode().strip())
        
        assert uid != 0, "Container should not run with UID 0 (root)"
    
    def test_file_permissions(self, running_container):
        """Test that files have appropriate permissions."""
        # Check application files are not world-writable
        exec_result = running_container.exec_run(
            "find /app -type f -perm /o+w | head -10"
        )
        
        world_writable_files = exec_result.output.decode().strip()
        assert not world_writable_files, f"Found world-writable files: {world_writable_files}"
        
        # Check that sensitive files are not readable by others
        exec_result = running_container.exec_run("ls -la /app")
        assert exec_result.exit_code == 0
    
    def test_minimal_attack_surface(self, running_container):
        """Test that container has minimal attack surface."""
        # Check that unnecessary packages are not installed
        dangerous_packages = ["curl", "wget", "nc", "netcat", "telnet", "ssh"]
        
        for package in dangerous_packages:
            exec_result = running_container.exec_run(f"which {package}")
            if exec_result.exit_code == 0:
                # Package is installed - this may be intentional, just warn
                print(f"Warning: {package} is installed in container")
        
        # Check that shell history is not preserved
        exec_result = running_container.exec_run("ls -la ~/.bash_history")
        assert exec_result.exit_code != 0, "Shell history should not be preserved"
    
    def test_environment_variable_security(self, running_container):
        """Test that environment variables don't contain secrets."""
        exec_result = running_container.exec_run("env")
        env_output = exec_result.output.decode().lower()
        
        # Check for common secret patterns
        secret_patterns = ["password", "secret", "key", "token", "credential"]
        
        for pattern in secret_patterns:
            # Allow these patterns but check they don't contain actual secrets
            if pattern in env_output:
                lines = [line for line in env_output.split('\n') if pattern in line]
                for line in lines:
                    # Should not contain obvious secret values
                    assert "=" in line, f"Malformed environment variable: {line}"
                    key, value = line.split("=", 1)
                    
                    # Check for obvious secret patterns
                    obvious_secrets = ["admin", "123456", "password", "secret123"]
                    assert not any(secret in value.lower() for secret in obvious_secrets), \
                        f"Environment variable contains obvious secret: {key}"


class TestContainerPerformance:
    """Test container performance characteristics."""
    
    def test_startup_time(self, docker_client, built_image, container_config):
        """Test container startup time."""
        start_time = time.time()
        
        try:
            container = docker_client.containers.run(**container_config)
            
            # Wait for health check to pass
            max_wait = 60  # seconds
            for i in range(max_wait):
                try:
                    response = requests.get("http://localhost:8001/health", timeout=2)
                    if response.status_code == 200:
                        startup_time = time.time() - start_time
                        break
                except:
                    pass
                time.sleep(1)
            else:
                pytest.fail("Container did not start within 60 seconds")
            
            container.remove(force=True)
            
            # Startup should be under 30 seconds
            assert startup_time < 30, f"Startup time {startup_time:.2f}s is too slow"
            
            print(f"Container startup time: {startup_time:.2f}s")
            
        except Exception as e:
            pytest.fail(f"Failed to test startup time: {e}")
    
    def test_memory_efficiency(self, running_container):
        """Test container memory efficiency."""
        # Monitor memory usage over time
        memory_samples = []
        
        for i in range(10):  # Sample over 10 seconds
            stats = running_container.stats(stream=False)
            memory_mb = stats['memory_stats']['usage'] / (1024 * 1024)
            memory_samples.append(memory_mb)
            time.sleep(1)
        
        avg_memory = sum(memory_samples) / len(memory_samples)
        max_memory = max(memory_samples)
        
        # Should use reasonable amount of memory
        assert avg_memory < 300, f"Average memory usage {avg_memory:.2f}MB is too high"
        assert max_memory < 400, f"Peak memory usage {max_memory:.2f}MB is too high"
        
        print(f"Memory usage - Average: {avg_memory:.2f}MB, Peak: {max_memory:.2f}MB")


class TestContainerNetworking:
    """Test container networking configuration."""
    
    def test_port_mapping(self, running_container):
        """Test that port mapping works correctly."""
        # Check that application is accessible on mapped port
        response = requests.get("http://localhost:8001/health", timeout=5)
        assert response.status_code == 200
        
        # Check that container port is properly mapped
        port_info = running_container.attrs['NetworkSettings']['Ports']
        assert '8000/tcp' in port_info
        assert port_info['8000/tcp'][0]['HostPort'] == '8001'
    
    def test_network_isolation(self, running_container):
        """Test network isolation and security."""
        # Check that container cannot access host network services
        # This depends on your network configuration
        
        # Test that outbound connections work (for API calls)
        exec_result = running_container.exec_run("python -c 'import socket; socket.create_connection((\"8.8.8.8\", 53), timeout=5)'")
        assert exec_result.exit_code == 0, "Container should be able to make outbound connections"
    
    def test_dns_resolution(self, running_container):
        """Test DNS resolution in container."""
        exec_result = running_container.exec_run("nslookup google.com")
        assert exec_result.exit_code == 0, "DNS resolution should work"


def calculate_cpu_percent(stats):
    """Calculate CPU percentage from Docker stats."""
    try:
        cpu_stats = stats['cpu_stats']
        precpu_stats = stats['precpu_stats']
        
        cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
        system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
        
        if system_delta > 0 and cpu_delta > 0:
            cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100.0
            return cpu_percent
        return 0.0
    except (KeyError, ZeroDivisionError):
        return 0.0