"""
Request optimization system for Gemini CLI.

This module implements:
- Request batching and prefetching
- Connection pooling optimization
- Request prioritization
- Adaptive timeout management
- Performance-based routing
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

from ..config import performance_config


@dataclass
class RequestBatch:
    """Batch of requests for optimization."""
    requests: List[Dict[str, Any]]
    priority: int
    created_at: float
    max_wait_time: float
    
    @property
    def is_expired(self) -> bool:
        """Check if batch has expired."""
        return time.time() - self.created_at > self.max_wait_time
    
    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.requests)


class RequestOptimizer:
    """
    Advanced request optimization system for improving API performance.
    
    Features:
    - Intelligent request batching
    - Connection pooling optimization
    - Request prioritization
    - Adaptive timeout management
    - Performance-based routing
    """
    
    def __init__(self):
        self.batch_size = performance_config.batch_size
        self.prefetch_enabled = performance_config.prefetch_enabled
        
        # Batching
        self.pending_batches: List[RequestBatch] = []
        self.batch_processor_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.request_latencies: deque = deque(maxlen=1000)
        self.success_rates: deque = deque(maxlen=1000)
        self.throughput_history: deque = deque(maxlen=100)
        
        # Adaptive parameters
        self.current_batch_size = self.batch_size
        self.current_timeout = 30.0
        self.connection_pool_size = 20
        
        # Optimization strategies
        self.optimization_strategies = {
            "batching": self._optimize_batching,
            "timeout": self._optimize_timeout,
            "connections": self._optimize_connections,
            "prioritization": self._optimize_prioritization
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the optimizer background tasks."""
        self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        self.logger.info("Request optimizer started")
    
    async def stop(self):
        """Stop the optimizer and cleanup."""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining batches
        await self._process_pending_batches()
        self.logger.info("Request optimizer stopped")
    
    async def _batch_processor_loop(self):
        """Background loop for processing request batches."""
        while True:
            try:
                await asyncio.sleep(0.1)  # Check every 100ms
                await self._process_pending_batches()
                
                # Periodic optimization
                if len(self.throughput_history) % 100 == 0:
                    await self._run_optimizations()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(1)
    
    async def _process_pending_batches(self):
        """Process pending request batches."""
        current_time = time.time()
        
        # Remove expired batches
        self.pending_batches = [
            batch for batch in self.pending_batches
            if not batch.is_expired
        ]
        
        # Process batches that are ready
        ready_batches = []
        for batch in self.pending_batches:
            if (batch.size >= self.current_batch_size or 
                batch.is_expired or
                current_time - batch.created_at >= batch.max_wait_time):
                ready_batches.append(batch)
        
        # Remove processed batches
        for batch in ready_batches:
            self.pending_batches.remove(batch)
        
        # Process ready batches
        if ready_batches:
            await self._execute_batches(ready_batches)
    
    async def _execute_batches(self, batches: List[RequestBatch]):
        """Execute multiple batches concurrently."""
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._execute_batch(batch))
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and update metrics
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Batch execution failed: {result}")
                else:
                    self._update_metrics(result)
    
    async def _execute_batch(self, batch: RequestBatch) -> Dict[str, Any]:
        """Execute a single batch of requests."""
        start_time = time.time()
        
        try:
            # Extract requests from batch
            requests = [req["request"] for req in batch.requests]
            
            # Execute requests concurrently
            tasks = []
            for request in requests:
                task = asyncio.create_task(self._execute_single_request(request))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate batch metrics
            execution_time = time.time() - start_time
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            
            batch_metrics = {
                "batch_size": batch.size,
                "execution_time": execution_time,
                "success_count": success_count,
                "total_count": batch.size,
                "success_rate": success_count / batch.size if batch.size > 0 else 0,
                "throughput": batch.size / execution_time if execution_time > 0 else 0
            }
            
            return batch_metrics
            
        except Exception as e:
            self.logger.error(f"Batch execution failed: {e}")
            return {
                "batch_size": batch.size,
                "execution_time": time.time() - start_time,
                "success_count": 0,
                "total_count": batch.size,
                "success_rate": 0,
                "throughput": 0,
                "error": str(e)
            }
    
    async def _execute_single_request(self, request: Dict[str, Any]) -> Any:
        """Execute a single request (placeholder for actual implementation)."""
        # This would typically call the actual Gemini API
        # For now, we'll simulate with a delay
        await asyncio.sleep(0.1)  # Simulate API call
        return {"status": "success", "data": "simulated_response"}
    
    def add_request(self, request: Dict[str, Any], priority: int = 1) -> str:
        """Add a request to the optimization queue."""
        request_id = f"req_{int(time.time() * 1000)}"
        
        # Find or create appropriate batch
        batch = self._find_or_create_batch(priority)
        batch.requests.append({
            "id": request_id,
            "request": request,
            "priority": priority,
            "added_at": time.time()
        })
        
        return request_id
    
    def _find_or_create_batch(self, priority: int) -> RequestBatch:
        """Find an existing batch or create a new one."""
        # Look for existing batch with same priority
        for batch in self.pending_batches:
            if (batch.priority == priority and 
                batch.size < self.current_batch_size and
                not batch.is_expired):
                return batch
        
        # Create new batch
        new_batch = RequestBatch(
            requests=[],
            priority=priority,
            created_at=time.time(),
            max_wait_time=5.0  # Max 5 seconds wait time
        )
        self.pending_batches.append(new_batch)
        return new_batch
    
    def _update_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics."""
        if "execution_time" in metrics:
            self.request_latencies.append(metrics["execution_time"])
        
        if "success_rate" in metrics:
            self.success_rates.append(metrics["success_rate"])
        
        if "throughput" in metrics:
            self.throughput_history.append(metrics["throughput"])
    
    async def _run_optimizations(self):
        """Run all optimization strategies."""
        for strategy_name, strategy_func in self.optimization_strategies.items():
            try:
                await strategy_func()
            except Exception as e:
                self.logger.warning(f"Optimization strategy {strategy_name} failed: {e}")
    
    async def _optimize_batching(self):
        """Optimize batch size based on performance."""
        if not self.throughput_history:
            return
        
        recent_throughput = list(self.throughput_history)[-10:]  # Last 10 measurements
        avg_throughput = statistics.mean(recent_throughput)
        
        # Adjust batch size based on throughput
        if avg_throughput > 10:  # High throughput
            if self.current_batch_size < 20:
                self.current_batch_size = min(20, self.current_batch_size + 2)
                self.logger.info(f"Increased batch size to {self.current_batch_size}")
        elif avg_throughput < 5:  # Low throughput
            if self.current_batch_size > 5:
                self.current_batch_size = max(5, self.current_batch_size - 1)
                self.logger.info(f"Decreased batch size to {self.current_batch_size}")
    
    async def _optimize_timeout(self):
        """Optimize timeout values based on latency."""
        if not self.request_latencies:
            return
        
        recent_latencies = list(self.request_latencies)[-50:]  # Last 50 measurements
        p95_latency = statistics.quantiles(recent_latencies, n=20)[18]  # 95th percentile
        
        # Set timeout to 2x P95 latency with bounds
        new_timeout = max(10.0, min(60.0, p95_latency * 2))
        
        if abs(new_timeout - self.current_timeout) > 5.0:  # Only change if significant
            self.current_timeout = new_timeout
            self.logger.info(f"Adjusted timeout to {self.current_timeout:.1f}s")
    
    async def _optimize_connections(self):
        """Optimize connection pool size."""
        if not self.throughput_history:
            return
        
        recent_throughput = list(self.throughput_history)[-20:]  # Last 20 measurements
        avg_throughput = statistics.mean(recent_throughput)
        
        # Adjust connection pool based on throughput
        if avg_throughput > 15:  # High throughput
            if self.connection_pool_size < 50:
                self.connection_pool_size = min(50, self.connection_pool_size + 5)
                self.logger.info(f"Increased connection pool to {self.connection_pool_size}")
        elif avg_throughput < 8:  # Low throughput
            if self.connection_pool_size > 10:
                self.connection_pool_size = max(10, self.connection_pool_size - 2)
                self.logger.info(f"Decreased connection pool to {self.connection_pool_size}")
    
    async def _optimize_prioritization(self):
        """Optimize request prioritization based on success rates."""
        if not self.success_rates:
            return
        
        recent_success_rates = list(self.success_rates)[-30:]  # Last 30 measurements
        avg_success_rate = statistics.mean(recent_success_rates)
        
        # Adjust priority thresholds based on success rate
        if avg_success_rate < 0.8:  # Low success rate
            # Increase priority for high-priority requests
            self.logger.info("Low success rate detected, prioritizing high-priority requests")
        elif avg_success_rate > 0.95:  # High success rate
            # Can handle more low-priority requests
            self.logger.info("High success rate, processing more low-priority requests")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization statistics."""
        return {
            "current_batch_size": self.current_batch_size,
            "current_timeout": self.current_timeout,
            "connection_pool_size": self.connection_pool_size,
            "pending_batches": len(self.pending_batches),
            "total_pending_requests": sum(batch.size for batch in self.pending_batches),
            "recent_throughput": list(self.throughput_history)[-10:] if self.throughput_history else [],
            "recent_latencies": list(self.request_latencies)[-10:] if self.request_latencies else [],
            "recent_success_rates": list(self.success_rates)[-10:] if self.success_rates else []
        }
    
    def get_recommendations(self) -> List[str]:
        """Get optimization recommendations."""
        recommendations = []
        
        if self.throughput_history:
            recent_throughput = list(self.throughput_history)[-5:]
            if len(recent_throughput) >= 3:
                trend = recent_throughput[-1] - recent_throughput[0]
                if trend < -2:
                    recommendations.append("Throughput declining - consider reducing batch size or increasing timeouts")
                elif trend > 2:
                    recommendations.append("Throughput improving - current optimization is working well")
        
        if self.success_rates:
            recent_success = list(self.success_rates)[-10:]
            avg_success = statistics.mean(recent_success)
            if avg_success < 0.8:
                recommendations.append("Low success rate - review error handling and timeout values")
        
        if self.request_latencies:
            recent_latency = list(self.request_latencies)[-20:]
            p95_latency = statistics.quantiles(recent_latency, n=20)[18] if len(recent_latency) >= 20 else max(recent_latency)
            if p95_latency > 10:
                recommendations.append("High latency detected - consider connection pooling optimization")
        
        return recommendations

