"""
Gemini CLI - High-Performance Command-Line Interface for Google's Gemini LLM

A production-ready CLI tool optimized for performance, featuring:
- Async request handling with connection pooling
- Intelligent caching and response optimization
- Real-time performance monitoring
- Advanced error handling and retry logic
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.client import GeminiClient
from .core.cache import CacheManager
from .core.metrics import PerformanceMonitor
from .core.optimizer import RequestOptimizer

__all__ = [
    "GeminiClient",
    "CacheManager", 
    "PerformanceMonitor",
    "RequestOptimizer",
    "__version__",
]

