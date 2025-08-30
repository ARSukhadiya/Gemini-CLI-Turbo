"""
Custom exception classes for Gemini CLI.

This module defines:
- Base exception classes
- API-specific exceptions
- Performance and optimization exceptions
- Cache and configuration exceptions
"""

from typing import Optional, Dict, Any


class GeminiCLIError(Exception):
    """Base exception class for Gemini CLI."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class GeminiAPIError(GeminiCLIError):
    """Exception raised for Gemini API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_data: Optional[Dict[str, Any]] = None):
        details = {}
        if status_code is not None:
            details["status_code"] = status_code
        if response_data:
            details["response_data"] = response_data
        
        super().__init__(message, details)
        self.status_code = status_code
        self.response_data = response_data


class RateLimitError(GeminiAPIError):
    """Exception raised when API rate limits are exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", 
                 retry_after: Optional[int] = None):
        details = {}
        if retry_after is not None:
            details["retry_after"] = retry_after
        
        super().__init__(message, details=details)
        self.retry_after = retry_after


class AuthenticationError(GeminiAPIError):
    """Exception raised for authentication failures."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message)


class TimeoutError(GeminiCLIError):
    """Exception raised when requests timeout."""
    
    def __init__(self, message: str = "Request timeout", timeout_value: Optional[float] = None):
        details = {}
        if timeout_value is not None:
            details["timeout_value"] = timeout_value
        
        super().__init__(message, details)
        self.timeout_value = timeout_value


class CircuitBreakerError(GeminiCLIError):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str = "Circuit breaker is open", 
                 failure_count: Optional[int] = None,
                 last_failure_time: Optional[float] = None):
        details = {}
        if failure_count is not None:
            details["failure_count"] = failure_count
        if last_failure_time is not None:
            details["last_failure_time"] = last_failure_time
        
        super().__init__(message, details)
        self.failure_count = failure_count
        self.last_failure_time = last_failure_time


class CacheError(GeminiCLIError):
    """Exception raised for cache-related errors."""
    
    def __init__(self, message: str, cache_type: Optional[str] = None):
        details = {}
        if cache_type:
            details["cache_type"] = cache_type
        
        super().__init__(message, details)
        self.cache_type = cache_type


class ConfigurationError(GeminiCLIError):
    """Exception raised for configuration errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_value: Optional[Any] = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = config_value
        
        super().__init__(message, details)
        self.config_key = config_key
        self.config_value = config_value


class PerformanceError(GeminiCLIError):
    """Exception raised for performance-related errors."""
    
    def __init__(self, message: str, metric_name: Optional[str] = None, 
                 metric_value: Optional[float] = None):
        details = {}
        if metric_name:
            details["metric_name"] = metric_name
        if metric_value is not None:
            details["metric_value"] = metric_value
        
        super().__init__(message, details)
        self.metric_name = metric_name
        self.metric_value = metric_value


class ValidationError(GeminiCLIError):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 field_value: Optional[Any] = None, validation_rule: Optional[str] = None):
        details = {}
        if field_name:
            details["field_name"] = field_name
        if field_value is not None:
            details["field_value"] = field_value
        if validation_rule:
            details["validation_rule"] = validation_rule
        
        super().__init__(message, details)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule


class ConnectionError(GeminiCLIError):
    """Exception raised for connection-related errors."""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, 
                 connection_type: Optional[str] = None):
        details = {}
        if endpoint:
            details["endpoint"] = endpoint
        if connection_type:
            details["connection_type"] = connection_type
        
        super().__init__(message, details)
        self.endpoint = endpoint
        self.connection_type = connection_type


class RetryExhaustedError(GeminiCLIError):
    """Exception raised when all retry attempts are exhausted."""
    
    def __init__(self, message: str = "All retry attempts exhausted", 
                 max_retries: Optional[int] = None,
                 last_error: Optional[Exception] = None):
        details = {}
        if max_retries is not None:
            details["max_retries"] = max_retries
        if last_error:
            details["last_error"] = str(last_error)
        
        super().__init__(message, details)
        self.max_retries = max_retries
        self.last_error = last_error


class ResourceExhaustedError(GeminiCLIError):
    """Exception raised when system resources are exhausted."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, 
                 current_usage: Optional[float] = None,
                 limit: Optional[float] = None):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if current_usage is not None:
            details["current_usage"] = current_usage
        if limit is not None:
            details["limit"] = limit
        
        super().__init__(message, details)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


# Exception mapping for common error scenarios
EXCEPTION_MAPPING = {
    "rate_limit": RateLimitError,
    "authentication": AuthenticationError,
    "timeout": TimeoutError,
    "circuit_breaker": CircuitBreakerError,
    "cache": CacheError,
    "configuration": ConfigurationError,
    "performance": PerformanceError,
    "validation": ValidationError,
    "connection": ConnectionError,
    "retry_exhausted": RetryExhaustedError,
    "resource_exhausted": ResourceExhaustedError
}


def create_exception(exception_type: str, message: str, **kwargs) -> GeminiCLIError:
    """Create an exception of the specified type with the given message and details."""
    if exception_type not in EXCEPTION_MAPPING:
        return GeminiCLIError(message, {"exception_type": exception_type, **kwargs})
    
    exception_class = EXCEPTION_MAPPING[exception_type]
    return exception_class(message, **kwargs)


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable."""
    retryable_exceptions = (
        TimeoutError,
        ConnectionError,
        RateLimitError,
        CircuitBreakerError
    )
    
    return isinstance(error, retryable_exceptions)


def is_critical_error(error: Exception) -> bool:
    """Check if an error is critical and should not be retried."""
    critical_exceptions = (
        AuthenticationError,
        ConfigurationError,
        ValidationError,
        ResourceExhaustedError
    )
    
    return isinstance(error, critical_exceptions)

