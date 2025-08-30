"""
Configuration management for Gemini CLI with environment variable support
and performance tuning options.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings


class GeminiConfig(BaseSettings):
    """Configuration for Gemini API and CLI behavior."""
    
    # API Configuration
    api_key: str = Field(..., env="GEMINI_API_KEY", description="Google Gemini API key")
    model: str = Field("gemini-pro", env="GEMINI_MODEL", description="Gemini model to use")
    api_base_url: str = Field(
        "https://generativelanguage.googleapis.com/v1beta/models/",
        env="GEMINI_API_BASE_URL",
        description="Base URL for Gemini API"
    )
    
    # Performance Configuration
    max_concurrent_requests: int = Field(
        10, env="MAX_CONCURRENT_REQUESTS", description="Maximum concurrent API requests"
    )
    request_timeout: int = Field(
        30, env="REQUEST_TIMEOUT", description="Request timeout in seconds"
    )
    connection_pool_size: int = Field(
        20, env="CONNECTION_POOL_SIZE", description="HTTP connection pool size"
    )
    keep_alive_timeout: int = Field(
        60, env="KEEP_ALIVE_TIMEOUT", description="Keep-alive timeout in seconds"
    )
    
    # Caching Configuration
    cache_enabled: bool = Field(
        True, env="CACHE_ENABLED", description="Enable response caching"
    )
    cache_ttl: int = Field(
        3600, env="CACHE_TTL", description="Cache TTL in seconds"
    )
    cache_max_size: int = Field(
        1000, env="CACHE_MAX_SIZE", description="Maximum cache entries"
    )
    cache_strategy: str = Field(
        "lru", env="CACHE_STRATEGY", description="Cache eviction strategy (lru, lfu, ttl)"
    )
    
    # Retry Configuration
    max_retries: int = Field(
        3, env="MAX_RETRIES", description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        1.0, env="RETRY_DELAY", description="Initial retry delay in seconds"
    )
    retry_backoff: float = Field(
        2.0, env="RETRY_BACKOFF", description="Retry backoff multiplier"
    )
    
    # Monitoring Configuration
    metrics_enabled: bool = Field(
        True, env="METRICS_ENABLED", description="Enable performance metrics"
    )
    metrics_port: int = Field(
        9090, env="METRICS_PORT", description="Prometheus metrics port"
    )
    log_level: str = Field(
        "INFO", env="LOG_LEVEL", description="Logging level"
    )
    
    # CLI Configuration
    interactive_mode: bool = Field(
        True, env="INTERACTIVE_MODE", description="Enable interactive CLI mode"
    )
    output_format: str = Field(
        "rich", env="OUTPUT_FORMAT", description="Output format (rich, plain, json)"
    )
    show_progress: bool = Field(
        True, env="SHOW_PROGRESS", description="Show progress bars and spinners"
    )
    
    # Advanced Configuration
    enable_streaming: bool = Field(
        True, env="ENABLE_STREAMING", description="Enable response streaming"
    )
    chunk_size: int = Field(
        1024, env="CHUNK_SIZE", description="Streaming chunk size in bytes"
    )
    enable_compression: bool = Field(
        True, env="ENABLE_COMPRESSION", description="Enable HTTP compression"
    )
    
    @validator("api_key")
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format."""
        if not v or len(v) < 10:
            raise ValueError("API key must be at least 10 characters long")
        return v
    
    @validator("model")
    def validate_model(cls, v: str) -> str:
        """Validate Gemini model name."""
        valid_models = ["gemini-pro", "gemini-pro-vision", "gemini-ultra"]
        if v not in valid_models:
            raise ValueError(f"Invalid model. Must be one of: {valid_models}")
        return v
    
    @validator("cache_strategy")
    def validate_cache_strategy(cls, v: str) -> str:
        """Validate cache strategy."""
        valid_strategies = ["lru", "lfu", "ttl"]
        if v not in valid_strategies:
            raise ValueError(f"Invalid cache strategy. Must be one of: {valid_strategies}")
        return v
    
    @validator("output_format")
    def validate_output_format(cls, v: str) -> str:
        """Validate output format."""
        valid_formats = ["rich", "plain", "json"]
        if v not in valid_formats:
            raise ValueError(f"Invalid output format. Must be one of: {valid_formats}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class PerformanceConfig(BaseSettings):
    """Performance tuning configuration."""
    
    # Request Optimization
    batch_size: int = Field(
        5, env="BATCH_SIZE", description="Batch size for multiple requests"
    )
    prefetch_enabled: bool = Field(
        True, env="PREFETCH_ENABLED", description="Enable request prefetching"
    )
    circuit_breaker_threshold: float = Field(
        0.5, env="CIRCUIT_BREAKER_THRESHOLD", description="Circuit breaker error threshold"
    )
    circuit_breaker_timeout: int = Field(
        60, env="CIRCUIT_BREAKER_TIMEOUT", description="Circuit breaker timeout in seconds"
    )
    
    # Memory Management
    max_memory_usage: int = Field(
        512, env="MAX_MEMORY_USAGE", description="Maximum memory usage in MB"
    )
    gc_threshold: int = Field(
        100, env="GC_THRESHOLD", description="Garbage collection threshold"
    )
    
    # Async Configuration
    event_loop_policy: str = Field(
        "auto", env="EVENT_LOOP_POLICY", description="Event loop policy (auto, uvloop, asyncio)"
    )
    max_workers: int = Field(
        None, env="MAX_WORKERS", description="Maximum worker threads (None for auto)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class CacheConfig(BaseSettings):
    """Cache configuration for different backends."""
    
    # Redis Configuration
    redis_enabled: bool = Field(
        False, env="REDIS_ENABLED", description="Enable Redis caching"
    )
    redis_host: str = Field(
        "localhost", env="REDIS_HOST", description="Redis host"
    )
    redis_port: int = Field(
        6379, env="REDIS_PORT", description="Redis port"
    )
    redis_db: int = Field(
        0, env="REDIS_DB", description="Redis database number"
    )
    redis_password: Optional[str] = Field(
        None, env="REDIS_PASSWORD", description="Redis password"
    )
    
    # MongoDB Configuration
    mongo_enabled: bool = Field(
        False, env="MONGO_ENABLED", description="Enable MongoDB caching"
    )
    mongo_uri: str = Field(
        "mongodb://localhost:27017", env="MONGO_URI", description="MongoDB connection URI"
    )
    mongo_database: str = Field(
        "gemini_cache", env="MONGO_DATABASE", description="MongoDB database name"
    )
    mongo_collection: str = Field(
        "responses", env="MONGO_COLLECTION", description="MongoDB collection name"
    )
    
    # Local Cache Configuration
    local_cache_enabled: bool = Field(
        True, env="LOCAL_CACHE_ENABLED", description="Enable local in-memory caching"
    )
    local_cache_path: Path = Field(
        Path("./cache"), env="LOCAL_CACHE_PATH", description="Local cache directory"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global configuration instances
gemini_config = GeminiConfig()
performance_config = PerformanceConfig()
cache_config = CacheConfig()


def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary."""
    return {
        "gemini": gemini_config.dict(),
        "performance": performance_config.dict(),
        "cache": cache_config.dict(),
    }


def reload_config() -> None:
    """Reload configuration from environment variables."""
    global gemini_config, performance_config, cache_config
    gemini_config = GeminiConfig()
    performance_config = PerformanceConfig()
    cache_config = CacheConfig()

