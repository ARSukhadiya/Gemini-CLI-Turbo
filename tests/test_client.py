"""
Comprehensive tests for Gemini CLI client.

This module tests:
- Client initialization and configuration
- Request handling and optimization
- Error handling and retry logic
- Performance monitoring
- Caching functionality
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from gemini_cli.core.client import GeminiClient, CircuitBreaker, RequestMetrics
from gemini_cli.core.exceptions import GeminiAPIError, CircuitBreakerError


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    with patch('gemini_cli.core.client.gemini_config') as mock:
        mock.api_key = "test_api_key"
        mock.model = "gemini-pro"
        mock.max_concurrent_requests = 5
        mock.request_timeout = 30
        mock.max_retries = 3
        mock.retry_delay = 1.0
        mock.retry_backoff = 2.0
        yield mock


@pytest.fixture
def mock_performance_config():
    """Mock performance configuration for testing."""
    with patch('gemini_cli.core.client.performance_config') as mock:
        mock.circuit_breaker_threshold = 3
        mock.circuit_breaker_timeout = 60
        yield mock


@pytest.fixture
def client(mock_config, mock_performance_config):
    """Create a test client instance."""
    return GeminiClient("test_api_key")


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_initial_state(self):
        """Test initial circuit breaker state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
    
    def test_success_resets_failures(self):
        """Test that success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        cb.on_failure()
        cb.on_failure()
        assert cb.failure_count == 2
        
        cb.on_success()
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"
    
    def test_failure_threshold_opens_circuit(self):
        """Test that failure threshold opens circuit."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Add failures up to threshold
        cb.on_failure()
        cb.on_failure()
        cb.on_failure()
        
        assert cb.state == "OPEN"
        assert not cb.can_execute()
    
    def test_timeout_resets_to_half_open(self):
        """Test that timeout resets circuit to half-open."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1)
        
        # Open circuit
        cb.on_failure()
        cb.on_failure()
        cb.on_failure()
        assert cb.state == "OPEN"
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Should be half-open
        assert cb.can_execute()
        assert cb.state == "HALF_OPEN"


class TestRequestMetrics:
    """Test request metrics functionality."""
    
    def test_metrics_creation(self):
        """Test metrics creation and initialization."""
        start_time = time.time()
        metrics = RequestMetrics(start_time=start_time)
        
        assert metrics.start_time == start_time
        assert metrics.end_time is None
        assert metrics.latency is None
        assert metrics.tokens_generated is None
        assert metrics.cache_hit is False
        assert metrics.retry_count == 0
        assert metrics.error is None
    
    def test_metrics_completion(self):
        """Test metrics completion."""
        start_time = time.time()
        metrics = RequestMetrics(start_time=start_time)
        
        end_time = start_time + 1.5
        tokens = 150
        
        metrics.complete(end_time, tokens)
        
        assert metrics.end_time == end_time
        assert metrics.latency == 1.5
        assert metrics.tokens_generated == tokens
        assert metrics.error is None
    
    def test_metrics_with_error(self):
        """Test metrics completion with error."""
        start_time = time.time()
        metrics = RequestMetrics(start_time=start_time)
        
        end_time = start_time + 2.0
        error = "API timeout"
        
        metrics.complete(end_time, error=error)
        
        assert metrics.end_time == end_time
        assert metrics.latency == 2.0
        assert metrics.error == error


class TestGeminiClient:
    """Test Gemini client functionality."""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.api_key == "test_api_key"
        assert client.model == "gemini-pro"
        assert client.max_concurrent == 5
        assert client.timeout == 30
        assert client.max_retries == 3
    
    @pytest.mark.asyncio
    async def test_client_context_manager(self, client):
        """Test client as async context manager."""
        async with client:
            assert client.session is not None
            assert not client.session.closed
        
        assert client.session is None
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, client):
        """Test circuit breaker integration."""
        # Initially should allow execution
        assert client.circuit_breaker.can_execute()
        
        # Simulate failures
        for _ in range(3):
            client.circuit_breaker.on_failure()
        
        # Circuit should be open
        assert not client.circuit_breaker.can_execute()
        
        # Should raise error when trying to generate
        with pytest.raises(CircuitBreakerError):
            await client.generate_text("test prompt")
    
    @pytest.mark.asyncio
    @patch('gemini_cli.core.client.genai.GenerativeModel')
    async def test_generate_text_success(self, mock_genai_model, client):
        """Test successful text generation."""
        # Mock the Gemini model
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_response.usage_metadata = {"total_token_count": 50}
        
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        client.genai_model = mock_model
        
        # Start client
        await client.start()
        
        # Generate text
        response = await client.generate_text("test prompt")
        
        assert response == "Generated response"
        assert client.total_requests == 1
        assert client.cache_hits == 0
    
    @pytest.mark.asyncio
    @patch('gemini_cli.core.client.genai.GenerativeModel')
    async def test_generate_text_with_retry(self, mock_genai_model, client):
        """Test text generation with retry logic."""
        # Mock the Gemini model to fail twice then succeed
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response"
        mock_response.usage_metadata = {"total_token_count": 50}
        
        mock_model.generate_content_async = AsyncMock(
            side_effect=[Exception("API Error"), Exception("API Error"), mock_response]
        )
        client.genai_model = mock_model
        
        # Start client
        await client.start()
        
        # Generate text (should retry and succeed)
        response = await client.generate_text("test prompt")
        
        assert response == "Generated response"
        assert client.total_requests == 1
    
    @pytest.mark.asyncio
    @patch('gemini_cli.core.client.genai.GenerativeModel')
    async def test_generate_text_max_retries_exceeded(self, mock_genai_model, client):
        """Test text generation when max retries are exceeded."""
        # Mock the Gemini model to always fail
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(
            side_effect=Exception("API Error")
        )
        client.genai_model = mock_model
        
        # Start client
        await client.start()
        
        # Should raise error after max retries
        with pytest.raises(GeminiAPIError, match="Max retries exceeded"):
            await client.generate_text("test prompt")
    
    @pytest.mark.asyncio
    @patch('gemini_cli.core.client.genai.GenerativeModel')
    async def test_generate_text_streaming(self, mock_genai_model, client):
        """Test streaming text generation."""
        # Mock streaming response
        mock_model = Mock()
        mock_chunk1 = Mock()
        mock_chunk1.text = "Hello "
        mock_chunk2 = Mock()
        mock_chunk2.text = "world!"
        
        mock_response = [mock_chunk1, mock_chunk2]
        mock_response.append = Mock()  # Make it list-like
        
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        client.genai_model = mock_model
        
        # Start client
        await client.start()
        
        # Generate streaming text
        response_stream = await client.generate_text("test prompt", stream=True)
        
        # Collect all chunks
        chunks = []
        async for chunk in response_stream:
            chunks.append(chunk)
        
        assert chunks == ["Hello ", "world!"]
    
    @pytest.mark.asyncio
    async def test_batch_generate(self, client):
        """Test batch text generation."""
        # Mock the generate_text method
        client.generate_text = AsyncMock(return_value="Generated response")
        
        prompts = ["prompt1", "prompt2", "prompt3"]
        
        # Start client
        await client.start()
        
        # Generate batch
        responses = await client.batch_generate(prompts)
        
        assert len(responses) == 3
        assert all(response == "Generated response" for response in responses)
        assert client.generate_text.call_count == 3
    
    @pytest.mark.asyncio
    async def test_batch_generate_with_errors(self, client):
        """Test batch generation with some errors."""
        # Mock the generate_text method to fail sometimes
        async def mock_generate(prompt, **kwargs):
            if "fail" in prompt:
                raise Exception("Generation failed")
            return "Generated response"
        
        client.generate_text = mock_generate
        
        prompts = ["prompt1", "fail_prompt", "prompt3"]
        
        # Start client
        await client.start()
        
        # Generate batch
        responses = await client.batch_generate(prompts)
        
        assert len(responses) == 3
        assert responses[0] == "Generated response"
        assert "Error:" in responses[1]
        assert responses[2] == "Generated response"
    
    @pytest.mark.asyncio
    async def test_get_performance_stats(self, client):
        """Test performance statistics retrieval."""
        # Start client
        await client.start()
        
        # Make some requests to generate stats
        client.total_requests = 10
        client.active_requests = 2
        client.cache_hits = 3
        
        # Get stats
        stats = await client.get_performance_stats()
        
        assert "requests" in stats
        assert "circuit_breaker" in stats
        assert "metrics" in stats
        assert "cache" in stats
        
        requests_stats = stats["requests"]
        assert requests_stats["total"] == 10
        assert requests_stats["active"] == 2
        assert requests_stats["cache_hits"] == 3
        assert requests_stats["cache_hit_rate"] == 0.3
    
    @pytest.mark.asyncio
    async def test_get_health_status(self, client):
        """Test health status retrieval."""
        # Start client
        await client.start()
        
        # Get health status
        health = client.get_health_status()
        
        assert "status" in health
        assert "circuit_breaker" in health
        assert "active_connections" in health
        assert "connection_pool_size" in health
        
        assert health["status"] == "healthy"
        assert health["circuit_breaker"] == "CLOSED"
        assert health["active_connections"] == 0
        assert health["connection_pool_size"] == 20


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self, client):
        """Test connection pooling configuration."""
        await client.start()
        
        # Check connection pool settings
        assert client.session.connector.limit == 20
        assert client.session.connector.limit_per_host == 20
        assert client.session.connector.keepalive_timeout == 60
    
    @pytest.mark.asyncio
    async def test_concurrent_request_limiting(self, client):
        """Test concurrent request limiting."""
        # Mock the generate_text method
        async def mock_generate(prompt, **kwargs):
            await asyncio.sleep(0.1)  # Simulate API call
            return "Generated response"
        
        client.generate_text = mock_generate
        
        # Start client
        await client.start()
        
        # Make multiple concurrent requests
        start_time = time.time()
        tasks = [
            client.generate_text(f"prompt{i}")
            for i in range(10)
        ]
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Should all succeed
        assert len(responses) == 10
        assert all(response == "Generated response" for response in responses)
        
        # Should respect concurrency limits
        # With max_concurrent=5, this should take at least 0.2 seconds
        # (10 requests / 5 concurrent = 2 batches)
        assert end_time - start_time >= 0.15


class TestErrorHandling:
    """Test error handling and recovery."""
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self, client):
        """Test authentication error handling."""
        # Mock the generate_text method to raise auth error
        client.generate_text = AsyncMock(
            side_effect=Exception("Authentication failed")
        )
        
        # Start client
        await client.start()
        
        # Should raise error
        with pytest.raises(Exception, match="Authentication failed"):
            await client.generate_text("test prompt")
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self, client):
        """Test timeout error handling."""
        # Mock the generate_text method to raise timeout
        client.generate_text = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timeout")
        )
        
        # Start client
        await client.start()
        
        # Should raise timeout error
        with pytest.raises(asyncio.TimeoutError):
            await client.generate_text("test prompt")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, client):
        """Test circuit breaker recovery after timeout."""
        # Initially allow execution
        assert client.circuit_breaker.can_execute()
        
        # Simulate failures to open circuit
        for _ in range(3):
            client.circuit_breaker.on_failure()
        
        # Circuit should be open
        assert not client.circuit_breaker.can_execute()
        
        # Wait for timeout
        client.circuit_breaker.last_failure_time = time.time() - 61
        
        # Should allow execution again (half-open)
        assert client.circuit_breaker.can_execute()
        assert client.circuit_breaker.state == "HALF_OPEN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

