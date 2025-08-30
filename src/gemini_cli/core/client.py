"""
High-performance Gemini API client with advanced optimization features.

This module implements:
- Async request handling with connection pooling
- Intelligent retry logic with exponential backoff
- Circuit breaker pattern for fault tolerance
- Request batching and prefetching
- Performance monitoring and metrics
"""

import asyncio
import time
import hashlib
import json
import logging
from typing import (
    Optional, Dict, List, Any, AsyncGenerator, Union, Tuple
)
from dataclasses import dataclass
from contextlib import asynccontextmanager

import aiohttp
import google.generativeai as genai
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from aiohttp.client_exceptions import ClientError, ClientTimeoutError
from asyncio_throttle import Throttler

from ..config import gemini_config, performance_config
from .cache import CacheManager
from .metrics import PerformanceMonitor
from .optimizer import RequestOptimizer
from .exceptions import (
    GeminiAPIError, RateLimitError, TimeoutError, CircuitBreakerError
)


@dataclass
class RequestMetrics:
    """Metrics for individual API requests."""
    start_time: float
    end_time: Optional[float] = None
    latency: Optional[float] = None
    tokens_generated: Optional[int] = None
    cache_hit: bool = False
    retry_count: int = 0
    error: Optional[str] = None
    
    def complete(self, end_time: float, tokens: Optional[int] = None, error: Optional[str] = None):
        """Complete the request metrics."""
        self.end_time = end_time
        self.latency = end_time - self.start_time
        self.tokens_generated = tokens
        self.error = error


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: float, timeout: int):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def can_execute(self) -> bool:
        """Check if the circuit breaker allows execution."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class GeminiClient:
    """
    High-performance Gemini API client with advanced optimization features.
    
    Features:
    - Async request handling with connection pooling
    - Intelligent caching with semantic similarity
    - Circuit breaker pattern for fault tolerance
    - Request batching and prefetching
    - Real-time performance monitoring
    - Streaming responses with chunked processing
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client with performance optimizations."""
        self.api_key = api_key or gemini_config.api_key
        self.model = gemini_config.model
        self.base_url = gemini_config.api_base_url
        
        # Performance configuration
        self.max_concurrent = gemini_config.max_concurrent_requests
        self.timeout = gemini_config.request_timeout
        self.max_retries = gemini_config.max_retries
        self.retry_delay = gemini_config.retry_delay
        self.retry_backoff = gemini_config.retry_backoff
        
        # Initialize components
        self.cache = CacheManager()
        self.monitor = PerformanceMonitor()
        self.optimizer = RequestOptimizer()
        self.circuit_breaker = CircuitBreaker(
            performance_config.circuit_breaker_threshold,
            performance_config.circuit_breaker_timeout
        )
        
        # Connection management
        self.session: Optional[ClientSession] = None
        self.throttler = Throttler(rate_limit=self.max_concurrent)
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Request tracking
        self.active_requests = 0
        self.total_requests = 0
        self.cache_hits = 0
        
        # Initialize Gemini
        genai.configure(api_key=self.api_key)
        self.genai_model = genai.GenerativeModel(self.model)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def start(self):
        """Start the client and initialize resources."""
        if self.session is None:
            connector = TCPConnector(
                limit=gemini_config.connection_pool_size,
                limit_per_host=gemini_config.connection_pool_size,
                keepalive_timeout=gemini_config.keep_alive_timeout,
                enable_cleanup_closed=True
            )
            
            timeout = ClientTimeout(total=self.timeout)
            self.session = ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "Gemini-CLI/1.0.0",
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate" if gemini_config.enable_compression else "identity"
                }
            )
            
            # Warm up connection pool
            await self._warmup_connections()
    
    async def close(self):
        """Close the client and cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
        
        # Finalize metrics
        await self.monitor.finalize()
    
    async def _warmup_connections(self):
        """Warm up the connection pool for better performance."""
        try:
            # Make a simple health check request
            async with self.session.get(f"{self.base_url}health") as response:
                if response.status == 200:
                    self.logger.info("Connection pool warmed up successfully")
        except Exception as e:
            self.logger.warning(f"Connection warmup failed: {e}")
    
    async def generate_text(
        self,
        prompt: str,
        stream: bool = False,
        use_cache: bool = True,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using Gemini API with performance optimizations.
        
        Args:
            prompt: Input prompt for text generation
            stream: Whether to stream the response
            use_cache: Whether to use caching
            **kwargs: Additional Gemini API parameters
            
        Returns:
            Generated text or streaming generator
        """
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerError("Circuit breaker is open")
        
        # Check cache first
        if use_cache and self.cache.is_enabled():
            cached_response = await self.cache.get(prompt)
            if cached_response:
                self.cache_hits += 1
                self.monitor.record_cache_hit()
                return cached_response
        
        # Create request metrics
        metrics = RequestMetrics(start_time=time.time())
        
        try:
            async with self.semaphore:
                self.active_requests += 1
                self.total_requests += 1
                
                if stream:
                    return await self._generate_stream(prompt, metrics, **kwargs)
                else:
                    return await self._generate_sync(prompt, metrics, **kwargs)
                    
        except Exception as e:
            metrics.complete(time.time(), error=str(e))
            self.monitor.record_error(str(e))
            self.circuit_breaker.on_failure()
            raise
        finally:
            self.active_requests -= 1
            self.monitor.record_request(metrics)
    
    async def _generate_sync(
        self, 
        prompt: str, 
        metrics: RequestMetrics, 
        **kwargs
    ) -> str:
        """Generate text synchronously with retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                async with self.throttler:
                    response = await self.genai_model.generate_content_async(
                        prompt, **kwargs
                    )
                    
                    # Extract text and metrics
                    text = response.text
                    tokens = getattr(response, 'usage_metadata', {}).get('total_token_count', 0)
                    
                    # Complete metrics
                    metrics.complete(time.time(), tokens)
                    
                    # Cache successful response
                    if self.cache.is_enabled():
                        await self.cache.set(prompt, text)
                    
                    # Record success
                    self.circuit_breaker.on_success()
                    self.monitor.record_success(metrics.latency, tokens)
                    
                    return text
                    
            except Exception as e:
                metrics.retry_count = attempt
                
                if attempt == self.max_retries:
                    metrics.complete(time.time(), error=str(e))
                    raise GeminiAPIError(f"Max retries exceeded: {e}")
                
                # Exponential backoff
                delay = self.retry_delay * (self.retry_backoff ** attempt)
                await asyncio.sleep(delay)
                
                self.logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {delay}s: {e}")
    
    async def _generate_stream(
        self, 
        prompt: str, 
        metrics: RequestMetrics, 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming for real-time responses."""
        try:
            async with self.throttler:
                response = await self.genai_model.generate_content_async(
                    prompt, stream=True, **kwargs
                )
                
                full_text = ""
                async for chunk in response:
                    if chunk.text:
                        full_text += chunk.text
                        yield chunk.text
                
                # Complete metrics
                tokens = getattr(response, 'usage_metadata', {}).get('total_token_count', 0)
                metrics.complete(time.time(), tokens)
                
                # Cache complete response
                if self.cache.is_enabled():
                    await self.cache.set(prompt, full_text)
                
                # Record success
                self.circuit_breaker.on_success()
                self.monitor.record_success(metrics.latency, tokens)
                
        except Exception as e:
            metrics.complete(time.time(), error=str(e))
            self.monitor.record_error(str(e))
            self.circuit_breaker.on_failure()
            raise GeminiAPIError(f"Streaming failed: {e}")
    
    async def batch_generate(
        self, 
        prompts: List[str], 
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in parallel with batching.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional Gemini API parameters
            
        Returns:
            List of generated texts
        """
        if not prompts:
            return []
        
        # Optimize batch size based on performance config
        batch_size = min(performance_config.batch_size, len(prompts))
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.generate_text(prompt, use_cache=True, **kwargs)
                for prompt in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and errors
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch generation failed for prompt {i+j}: {result}")
                    results.append(f"Error: {result}")
                else:
                    results.append(result)
        
        return results
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "requests": {
                "total": self.total_requests,
                "active": self.active_requests,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": self.cache_hits / max(self.total_requests, 1)
            },
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count,
                "last_failure": self.circuit_breaker.last_failure_time
            },
            "metrics": await self.monitor.get_stats(),
            "cache": await self.cache.get_stats()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of the client."""
        return {
            "status": "healthy" if self.session and not self.session.closed else "unhealthy",
            "circuit_breaker": self.circuit_breaker.state,
            "active_connections": self.active_requests,
            "connection_pool_size": gemini_config.connection_pool_size
        }

