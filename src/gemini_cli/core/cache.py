"""
Intelligent caching system for Gemini CLI with multiple backends and optimization strategies.

This module implements:
- Multi-backend caching (Redis, MongoDB, Local)
- Semantic similarity-based cache lookup
- Advanced eviction strategies (LRU, LFU, TTL)
- Cache warming and prefetching
- Performance analytics and hit rate optimization
"""

import asyncio
import hashlib
import json
import pickle
import time
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, asdict
from collections import OrderedDict
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

from ..config import cache_config, gemini_config


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent management."""
    key: str
    value: str
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int
    ttl: Optional[int] = None
    tags: List[str] = None
    similarity_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.similarity_hash is None:
            self.similarity_hash = self._compute_similarity_hash()
    
    def _compute_similarity_hash(self) -> str:
        """Compute a hash for semantic similarity matching."""
        # Simple hash based on content length and first/last characters
        content = self.value[:100] + self.value[-100:] if len(self.value) > 200 else self.value
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access time and count."""
        self.accessed_at = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class LocalCache:
    """High-performance local in-memory cache with multiple eviction strategies."""
    
    def __init__(self, max_size: int = 1000, strategy: str = "lru"):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()
        self.access_counts: Dict[str, int] = {}
        
        # Strategy-specific data structures
        if strategy == "lfu":
            self.frequency_map: Dict[int, set] = {}
            self.min_frequency = 0
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[str]:
        """Get value from cache with strategy-specific updates."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if entry.is_expired():
            self._remove(key)
            return None
        
        # Update access metadata
        entry.touch()
        
        # Strategy-specific updates
        if self.strategy == "lru":
            self._update_lru(key)
        elif self.strategy == "lfu":
            self._update_lfu(key)
        
        return entry.value
    
    def set(self, key: str, value: str, ttl: Optional[int] = None, tags: List[str] = None) -> None:
        """Set value in cache with eviction if necessary."""
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=1,
            size_bytes=len(value.encode()),
            ttl=ttl or gemini_config.cache_ttl,
            tags=tags or []
        )
        
        # Check if key already exists
        if key in self.cache:
            self._remove(key)
        
        # Evict if necessary
        if len(self.cache) >= self.max_size:
            self._evict()
        
        # Add new entry
        self.cache[key] = entry
        self._add_to_strategy(key)
    
    def _evict(self):
        """Evict entry based on strategy."""
        if self.strategy == "lru":
            # Remove least recently used
            oldest_key = next(iter(self.access_order))
            self._remove(oldest_key)
        elif self.strategy == "lfu":
            # Remove least frequently used
            if self.min_frequency in self.frequency_map:
                key_to_remove = next(iter(self.frequency_map[self.min_frequency]))
                self._remove(key_to_remove)
        elif self.strategy == "ttl":
            # Remove expired entries first, then oldest
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                self._remove(expired_keys[0])
            else:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
                self._remove(oldest_key)
    
    def _remove(self, key: str):
        """Remove entry from cache and strategy structures."""
        if key in self.cache:
            del self.cache[key]
        
        if key in self.access_order:
            del self.access_order[key]
        
        if key in self.access_counts:
            del self.access_counts[key]
    
    def _add_to_strategy(self, key: str):
        """Add key to strategy-specific data structures."""
        if self.strategy == "lru":
            self.access_order[key] = None
        elif self.strategy == "lfu":
            self.access_counts[key] = 1
            if 1 not in self.frequency_map:
                self.frequency_map[1] = set()
            self.frequency_map[1].add(key)
            self.min_frequency = 1
    
    def _update_lru(self, key: str):
        """Update LRU access order."""
        if key in self.access_order:
            del self.access_order[key]
        self.access_order[key] = None
    
    def _update_lfu(self, key: str):
        """Update LFU frequency counts."""
        old_freq = self.access_counts[key]
        new_freq = old_freq + 1
        
        # Remove from old frequency
        self.frequency_map[old_freq].remove(key)
        if not self.frequency_map[old_freq]:
            del self.frequency_map[old_freq]
            if old_freq == self.min_frequency:
                self.min_frequency = new_freq
        
        # Add to new frequency
        if new_freq not in self.frequency_map:
            self.frequency_map[new_freq] = set()
        self.frequency_map[new_freq].add(key)
        self.access_counts[key] = new_freq
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
        self.access_counts.clear()
        if self.strategy == "lfu":
            self.frequency_map.clear()
            self.min_frequency = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        expired_count = sum(1 for entry in self.cache.values() if entry.is_expired())
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "strategy": self.strategy,
            "total_size_bytes": total_size,
            "expired_count": expired_count,
            "utilization": len(self.cache) / self.max_size
        }


class SemanticCache:
    """Semantic similarity-based cache for finding similar responses."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.svd = TruncatedSVD(n_components=100)
        self.vectors = []
        self.entries: List[CacheEntry] = []
        self.is_fitted = False
        
        self.logger = logging.getLogger(__name__)
    
    def add_entry(self, entry: CacheEntry):
        """Add entry to semantic cache."""
        self.entries.append(entry)
        self.is_fitted = False  # Need to refit vectors
    
    def find_similar(self, query: str, threshold: Optional[float] = None) -> Optional[CacheEntry]:
        """Find semantically similar cache entry."""
        if not self.entries:
            return None
        
        threshold = threshold or self.similarity_threshold
        
        try:
            # Fit vectorizer and transform if needed
            if not self.is_fitted:
                self._fit_vectors()
            
            # Transform query
            query_vector = self.vectorizer.transform([query])
            query_vector = self.svd.transform(query_vector)
            
            # Find best match
            best_similarity = 0
            best_entry = None
            
            for i, entry in enumerate(self.entries):
                if entry.is_expired():
                    continue
                
                entry_vector = self.vectors[i].reshape(1, -1)
                similarity = cosine_similarity(query_vector, entry_vector)[0][0]
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_entry = entry
            
            if best_entry:
                best_entry.touch()
                self.logger.debug(f"Semantic cache hit with similarity {best_similarity:.3f}")
            
            return best_entry
            
        except Exception as e:
            self.logger.warning(f"Semantic similarity failed: {e}")
            return None
    
    def _fit_vectors(self):
        """Fit vectorizer and transform all entries."""
        try:
            # Extract text content
            texts = [entry.value for entry in self.entries]
            
            # Fit and transform
            tfidf_vectors = self.vectorizer.fit_transform(texts)
            self.vectors = self.svd.fit_transform(tfidf_vectors)
            self.is_fitted = True
            
        except Exception as e:
            self.logger.error(f"Failed to fit semantic vectors: {e}")
            self.is_fitted = False
    
    def clear(self):
        """Clear semantic cache."""
        self.entries.clear()
        self.vectors.clear()
        self.is_fitted = False


class CacheManager:
    """
    Intelligent cache manager with multiple backends and optimization strategies.
    
    Features:
    - Multi-backend support (Local, Redis, MongoDB)
    - Semantic similarity matching
    - Advanced eviction strategies
    - Cache warming and prefetching
    - Performance analytics
    """
    
    def __init__(self):
        self.local_cache = LocalCache(
            max_size=gemini_config.cache_max_size,
            strategy=gemini_config.cache_strategy
        )
        self.semantic_cache = SemanticCache()
        
        # Backend caches
        self.redis_cache = None
        self.mongo_cache = None
        
        # Initialize backends if configured
        self._init_backends()
        
        # Statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.semantic_hits = 0
        
        self.logger = logging.getLogger(__name__)
    
    def _init_backends(self):
        """Initialize configured cache backends."""
        if cache_config.redis_enabled:
            try:
                import redis.asyncio as redis
                self.redis_cache = redis.Redis(
                    host=cache_config.redis_host,
                    port=cache_config.redis_port,
                    db=cache_config.redis_db,
                    password=cache_config.redis_password,
                    decode_responses=True
                )
                self.logger.info("Redis cache backend initialized")
            except ImportError:
                self.logger.warning("Redis not available, skipping Redis backend")
        
        if cache_config.mongo_enabled:
            try:
                import pymongo
                self.mongo_cache = pymongo.MongoClient(cache_config.mongo_uri)
                self.mongo_db = self.mongo_cache[cache_config.mongo_database]
                self.mongo_collection = self.mongo_db[cache_config.mongo_collection]
                self.logger.info("MongoDB cache backend initialized")
            except ImportError:
                self.logger.warning("MongoDB not available, skipping MongoDB backend")
    
    def is_enabled(self) -> bool:
        """Check if caching is enabled."""
        return gemini_config.cache_enabled
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache with fallback strategies."""
        if not self.is_enabled():
            return None
        
        self.total_requests += 1
        
        # Try local cache first
        value = self.local_cache.get(key)
        if value:
            self.cache_hits += 1
            return value
        
        # Try semantic cache
        semantic_entry = self.semantic_cache.find_similar(key)
        if semantic_entry:
            self.semantic_hits += 1
            # Cache the semantic match locally
            self.local_cache.set(key, semantic_entry.value, semantic_entry.ttl, semantic_entry.tags)
            return semantic_entry.value
        
        # Try backend caches
        if self.redis_cache:
            try:
                value = await self.redis_cache.get(key)
                if value:
                    self.cache_hits += 1
                    # Cache locally for faster access
                    self.local_cache.set(key, value)
                    return value
            except Exception as e:
                self.logger.warning(f"Redis cache access failed: {e}")
        
        if self.mongo_cache:
            try:
                doc = self.mongo_collection.find_one({"key": key})
                if doc and not self._is_expired(doc):
                    value = doc["value"]
                    self.cache_hits += 1
                    # Cache locally
                    self.local_cache.set(key, value, doc.get("ttl"))
                    return value
            except Exception as e:
                self.logger.warning(f"MongoDB cache access failed: {e}")
        
        return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None, tags: List[str] = None) -> None:
        """Set value in all cache backends."""
        if not self.is_enabled():
            return
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=1,
            size_bytes=len(value.encode()),
            ttl=ttl or gemini_config.cache_ttl,
            tags=tags or []
        )
        
        # Set in local cache
        self.local_cache.set(key, value, ttl, tags)
        
        # Add to semantic cache
        self.semantic_cache.add_entry(entry)
        
        # Set in backend caches
        if self.redis_cache:
            try:
                await self.redis_cache.setex(key, ttl or gemini_config.cache_ttl, value)
            except Exception as e:
                self.logger.warning(f"Redis cache set failed: {e}")
        
        if self.mongo_cache:
            try:
                doc = {
                    "key": key,
                    "value": value,
                    "created_at": entry.created_at,
                    "ttl": ttl or gemini_config.cache_ttl,
                    "tags": tags or []
                }
                self.mongo_collection.replace_one(
                    {"key": key}, doc, upsert=True
                )
            except Exception as e:
                self.logger.warning(f"MongoDB cache set failed: {e}")
    
    def _is_expired(self, doc: Dict[str, Any]) -> bool:
        """Check if MongoDB document is expired."""
        created_at = doc.get("created_at", 0)
        ttl = doc.get("ttl", 0)
        return time.time() - created_at > ttl
    
    async def clear(self):
        """Clear all cache backends."""
        self.local_cache.clear()
        self.semantic_cache.clear()
        
        if self.redis_cache:
            try:
                await self.redis_cache.flushdb()
            except Exception as e:
                self.logger.warning(f"Redis cache clear failed: {e}")
        
        if self.mongo_cache:
            try:
                self.mongo_collection.delete_many({})
            except Exception as e:
                self.logger.warning(f"MongoDB cache clear failed: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        local_stats = self.local_cache.get_stats()
        
        stats = {
            "local": local_stats,
            "semantic": {
                "entries": len(self.semantic_cache.entries),
                "is_fitted": self.semantic_cache.is_fitted
            },
            "performance": {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "semantic_hits": self.semantic_hits,
                "hit_rate": self.cache_hits / max(self.total_requests, 1),
                "semantic_hit_rate": self.semantic_hits / max(self.total_hits, 1)
            }
        }
        
        # Add backend stats
        if self.redis_cache:
            try:
                redis_info = await self.redis_cache.info()
                stats["redis"] = {
                    "connected": True,
                    "keyspace": redis_info.get("db0", {}),
                    "memory": redis_info.get("used_memory_human", "N/A")
                }
            except Exception:
                stats["redis"] = {"connected": False}
        
        if self.mongo_cache:
            try:
                mongo_stats = self.mongo_db.command("dbStats")
                stats["mongodb"] = {
                    "connected": True,
                    "collections": mongo_stats.get("collections", 0),
                    "data_size": mongo_stats.get("dataSize", 0)
                }
            except Exception:
                stats["mongodb"] = {"connected": False}
        
        return stats
    
    async def warm_cache(self, prompts: List[str]):
        """Warm up cache with common prompts."""
        if not self.is_enabled():
            return
        
        self.logger.info(f"Warming cache with {len(prompts)} prompts")
        
        # This would typically involve prefetching responses
        # For now, we'll just log the warming process
        for prompt in prompts[:10]:  # Limit to first 10
            if not await self.get(prompt):
                self.logger.debug(f"Cache miss for prompt: {prompt[:50]}...")
    
    async def optimize_cache(self):
        """Optimize cache performance and cleanup."""
        if not self.is_enabled():
            return
        
        # Remove expired entries
        expired_keys = [
            key for key, entry in self.local_cache.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self.local_cache._remove(key)
        
        if expired_keys:
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        # Optimize semantic cache
        if len(self.semantic_cache.entries) > 1000:
            # Keep only most frequently accessed entries
            sorted_entries = sorted(
                self.semantic_cache.entries,
                key=lambda e: e.access_count,
                reverse=True
            )
            self.semantic_cache.entries = sorted_entries[:1000]
            self.semantic_cache.is_fitted = False
            self.logger.info("Optimized semantic cache size")

