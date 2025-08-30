"""
Performance monitoring and metrics collection for Gemini CLI.

This module implements:
- Real-time performance metrics collection
- Prometheus metrics export
- Performance analytics and reporting
- Memory and CPU profiling
- Latency distribution analysis
"""

import asyncio
import time
import statistics
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific time period."""
    timestamp: float
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    cache_hit_count: int = 0
    total_latency: float = 0.0
    total_tokens: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.cache_hit_count / self.request_count) * 100
    
    @property
    def avg_latency(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency / self.request_count
    
    @property
    def avg_tokens_per_request(self) -> float:
        """Calculate average tokens generated per request."""
        if self.request_count == 0:
            return 0.0
        return self.total_tokens / self.request_count
    
    @property
    def throughput(self) -> float:
        """Calculate requests per second."""
        return self.request_count / 60.0  # Assuming 1-minute windows


class LatencyTracker:
    """Track latency distribution and percentiles."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.latencies: deque = deque(maxlen=max_samples)
        self.buckets = defaultdict(int)
        self.bucket_boundaries = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    
    def add_latency(self, latency_ms: float):
        """Add a latency measurement."""
        self.latencies.append(latency_ms)
        
        # Update histogram buckets
        for boundary in self.bucket_boundaries:
            if latency_ms <= boundary:
                self.buckets[boundary] += 1
                break
        else:
            self.buckets[float('inf')] += 1
    
    def get_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles."""
        if not self.latencies:
            return {}
        
        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)
        
        return {
            "p50": sorted_latencies[int(0.5 * n)],
            "p90": sorted_latencies[int(0.9 * n)],
            "p95": sorted_latencies[int(0.95 * n)],
            "p99": sorted_latencies[int(0.99 * n)],
            "p99.9": sorted_latencies[int(0.999 * n)] if n >= 1000 else sorted_latencies[-1]
        }
    
    def get_histogram(self) -> Dict[str, int]:
        """Get latency histogram buckets."""
        return dict(self.buckets)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive latency statistics."""
        if not self.latencies:
            return {}
        
        return {
            "count": len(self.latencies),
            "min": min(self.latencies),
            "max": max(self.latencies),
            "mean": statistics.mean(self.latencies),
            "median": statistics.median(self.latencies),
            "stdev": statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0,
            "percentiles": self.get_percentiles(),
            "histogram": self.get_histogram()
        }


class SystemMonitor:
    """Monitor system resources and performance."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.memory_samples: deque = deque(maxlen=1000)
        self.cpu_samples: deque = deque(maxlen=1000)
        
    def sample(self):
        """Take a sample of current system metrics."""
        try:
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            self.cpu_samples.append(cpu_percent)
            
        except Exception as e:
            logging.warning(f"Failed to sample system metrics: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.memory_samples:
            return {}
        
        return {
            "current_mb": self.memory_samples[-1] if self.memory_samples else 0,
            "avg_mb": statistics.mean(self.memory_samples),
            "max_mb": max(self.memory_samples),
            "min_mb": min(self.memory_samples),
            "samples": len(self.memory_samples)
        }
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get CPU usage statistics."""
        if not self.cpu_samples:
            return {}
        
        return {
            "current_percent": self.cpu_samples[-1] if self.cpu_samples else 0,
            "avg_percent": statistics.mean(self.cpu_samples),
            "max_percent": max(self.cpu_samples),
            "min_percent": min(self.cpu_samples),
            "samples": len(self.cpu_samples)
        }
    
    def get_uptime(self) -> float:
        """Get process uptime in seconds."""
        return time.time() - self.start_time


class PrometheusExporter:
    """Export metrics to Prometheus format."""
    
    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            self.available = False
            return
        
        self.available = True
        
        # Define Prometheus metrics
        self.request_counter = Counter(
            'gemini_requests_total',
            'Total number of Gemini API requests',
            ['status', 'model']
        )
        
        self.latency_histogram = Histogram(
            'gemini_request_duration_seconds',
            'Request latency in seconds',
            ['model'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 60.0]
        )
        
        self.cache_hit_counter = Counter(
            'gemini_cache_hits_total',
            'Total number of cache hits',
            ['cache_type']
        )
        
        self.active_requests_gauge = Gauge(
            'gemini_active_requests',
            'Number of currently active requests'
        )
        
        self.memory_usage_gauge = Gauge(
            'gemini_memory_usage_bytes',
            'Current memory usage in bytes'
        )
        
        self.cpu_usage_gauge = Gauge(
            'gemini_cpu_usage_percent',
            'Current CPU usage percentage'
        )
    
    def record_request(self, status: str, model: str, latency: float):
        """Record a request metric."""
        if not self.available:
            return
        
        self.request_counter.labels(status=status, model=model).inc()
        self.latency_histogram.labels(model=model).observe(latency / 1000.0)  # Convert to seconds
    
    def record_cache_hit(self, cache_type: str):
        """Record a cache hit."""
        if not self.available:
            return
        
        self.cache_hit_counter.labels(cache_type=cache_type).inc()
    
    def update_active_requests(self, count: int):
        """Update active requests gauge."""
        if not self.available:
            return
        
        self.active_requests_gauge.set(count)
    
    def update_system_metrics(self, memory_bytes: float, cpu_percent: float):
        """Update system metrics gauges."""
        if not self.available:
            return
        
        self.memory_usage_gauge.set(memory_bytes)
        self.cpu_usage_gauge.set(cpu_percent)
    
    def generate_metrics(self) -> str:
        """Generate Prometheus metrics."""
        if not self.available:
            return "# Prometheus client not available\n"
        
        return generate_latest().decode('utf-8')


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for Gemini CLI.
    
    Features:
    - Real-time metrics collection
    - Latency distribution tracking
    - System resource monitoring
    - Prometheus metrics export
    - Performance analytics and reporting
    """
    
    def __init__(self, export_interval: int = 60):
        self.export_interval = export_interval
        self.start_time = time.time()
        
        # Metrics storage
        self.current_metrics = PerformanceMetrics(timestamp=time.time())
        self.historical_metrics: List[PerformanceMetrics] = []
        self.metrics_lock = asyncio.Lock()
        
        # Component monitors
        self.latency_tracker = LatencyTracker()
        self.system_monitor = SystemMonitor()
        self.prometheus_exporter = PrometheusExporter()
        
        # Performance tracking
        self.request_start_times: Dict[str, float] = {}
        self.active_requests = 0
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.export_task: Optional[asyncio.Task] = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the performance monitor."""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.export_task = asyncio.create_task(self._export_loop())
        self.logger.info("Performance monitor started")
    
    async def stop(self):
        """Stop the performance monitor."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.export_task:
            self.export_task.cancel()
        
        # Final metrics export
        await self._export_metrics()
        self.logger.info("Performance monitor stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop for collecting metrics."""
        while True:
            try:
                # Sample system metrics
                self.system_monitor.sample()
                
                # Export metrics periodically
                await self._export_metrics()
                
                # Wait for next cycle
                await asyncio.sleep(self.export_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _export_loop(self):
        """Background loop for exporting metrics."""
        while True:
            try:
                await asyncio.sleep(self.export_interval)
                await self._export_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Export loop error: {e}")
    
    async def _export_metrics(self):
        """Export current metrics and reset counters."""
        async with self.metrics_lock:
            if self.current_metrics.request_count > 0:
                # Store historical metrics
                self.historical_metrics.append(self.current_metrics)
                
                # Keep only last 24 hours of metrics (1440 minutes)
                cutoff_time = time.time() - (24 * 60 * 60)
                self.historical_metrics = [
                    m for m in self.historical_metrics
                    if m.timestamp > cutoff_time
                ]
                
                # Reset current metrics
                self.current_metrics = PerformanceMetrics(timestamp=time.time())
    
    def record_request_start(self, request_id: str):
        """Record the start of a request."""
        self.request_start_times[request_id] = time.time()
        self.active_requests += 1
        self.prometheus_exporter.update_active_requests(self.active_requests)
    
    def record_request_end(self, request_id: str, success: bool, latency: float, tokens: int = 0):
        """Record the completion of a request."""
        if request_id in self.request_start_times:
            del self.request_start_times[request_id]
        
        self.active_requests = max(0, self.active_requests - 1)
        self.prometheus_exporter.update_active_requests(self.active_requests)
        
        # Update metrics
        async def _update():
            async with self.metrics_lock:
                self.current_metrics.request_count += 1
                if success:
                    self.current_metrics.success_count += 1
                else:
                    self.current_metrics.error_count += 1
                
                self.current_metrics.total_latency += latency
                self.current_metrics.total_tokens += tokens
        
        # Schedule update
        asyncio.create_task(_update())
        
        # Update latency tracker
        self.latency_tracker.add_latency(latency)
        
        # Update Prometheus metrics
        status = "success" if success else "error"
        self.prometheus_exporter.record_request(status, "gemini-pro", latency)
    
    def record_cache_hit(self, cache_type: str = "local"):
        """Record a cache hit."""
        async def _update():
            async with self.metrics_lock:
                self.current_metrics.cache_hit_count += 1
        
        asyncio.create_task(_update())
        self.prometheus_exporter.record_cache_hit(cache_type)
    
    def record_error(self, error: str):
        """Record an error occurrence."""
        async def _update():
            async with self.metrics_lock:
                self.current_metrics.error_count += 1
        
        asyncio.create_task(_update())
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        # Get current system metrics
        memory_stats = self.system_monitor.get_memory_stats()
        cpu_stats = self.system_monitor.get_cpu_stats()
        
        # Update Prometheus system metrics
        if memory_stats:
            memory_bytes = memory_stats.get("current_mb", 0) * 1024 * 1024
            cpu_percent = cpu_stats.get("current_percent", 0)
            self.prometheus_exporter.update_system_metrics(memory_bytes, cpu_percent)
        
        # Calculate aggregate metrics
        total_requests = sum(m.request_count for m in self.historical_metrics)
        total_success = sum(m.success_count for m in self.historical_metrics)
        total_errors = sum(m.error_count for m in self.historical_metrics)
        total_latency = sum(m.total_latency for m in self.historical_metrics)
        total_tokens = sum(m.total_tokens for m in self.historical_metrics)
        
        avg_latency = total_latency / max(total_requests, 1)
        error_rate = (total_errors / max(total_requests, 1)) * 100
        
        return {
            "current": {
                "active_requests": self.active_requests,
                "uptime_seconds": self.system_monitor.get_uptime(),
                "memory_mb": memory_stats.get("current_mb", 0),
                "cpu_percent": cpu_stats.get("current_percent", 0)
            },
            "aggregate": {
                "total_requests": total_requests,
                "total_success": total_success,
                "total_errors": total_errors,
                "error_rate_percent": error_rate,
                "avg_latency_ms": avg_latency,
                "total_tokens": total_tokens
            },
            "latency": self.latency_tracker.get_stats(),
            "system": {
                "memory": memory_stats,
                "cpu": cpu_stats
            },
            "prometheus_available": self.prometheus_exporter.available
        }
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        stats = await self.get_stats()
        
        # Calculate throughput trends
        if len(self.historical_metrics) >= 2:
            recent_metrics = self.historical_metrics[-10:]  # Last 10 minutes
            throughput_trend = [
                m.throughput for m in recent_metrics
            ]
            stats["throughput_trend"] = {
                "current": throughput_trend[-1] if throughput_trend else 0,
                "avg": statistics.mean(throughput_trend) if throughput_trend else 0,
                "trend": "increasing" if len(throughput_trend) >= 2 and throughput_trend[-1] > throughput_trend[-2] else "decreasing"
            }
        
        # Performance recommendations
        recommendations = []
        
        if stats["latency"].get("p99", 0) > 1000:  # P99 > 1 second
            recommendations.append("Consider optimizing request processing or increasing timeout values")
        
        if stats["aggregate"]["error_rate_percent"] > 5:  # Error rate > 5%
            recommendations.append("High error rate detected. Review error logs and API configuration")
        
        if memory_stats.get("current_mb", 0) > 500:  # Memory > 500MB
            recommendations.append("High memory usage. Consider enabling garbage collection or reducing cache size")
        
        stats["recommendations"] = recommendations
        
        return stats
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return self.prometheus_exporter.generate_metrics()
    
    async def finalize(self):
        """Finalize monitoring and cleanup."""
        await self.stop()
        
        # Generate final report
        final_report = await self.get_performance_report()
        self.logger.info("Final performance report generated")
        
        return final_report

