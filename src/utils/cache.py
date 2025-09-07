"""
High-Performance Caching Layer
Redis-based caching for sub-100ms recommendation serving
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
import redis.asyncio as redis
import pickle
import hashlib
import structlog
from contextlib import asynccontextmanager

logger = structlog.get_logger()

class CacheManager:
    """High-performance Redis cache manager with async support"""
    
    def __init__(self, redis_config: Dict[str, Any]):
        self.config = redis_config
        self.redis_client = None
        self.connection_pool = None
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize Redis connection pool"""
        try:
            # Create connection pool for better performance
            self.connection_pool = redis.ConnectionPool(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 6379),
                db=self.config.get('db', 0),
                password=self.config.get('password'),
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            logger.info("Redis connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with automatic deserialization"""
        try:
            cached_value = await self.redis_client.get(key)
            
            if cached_value is None:
                return default
            
            # Try JSON deserialization first (faster)
            try:
                return json.loads(cached_value)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fallback to pickle for complex objects
                return pickle.loads(cached_value)
                
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        serialize_json: bool = True
    ) -> bool:
        """Set value in cache with automatic serialization"""
        try:
            # Choose serialization method
            if serialize_json:
                try:
                    serialized_value = json.dumps(value)
                except (TypeError, ValueError):
                    # Fallback to pickle for complex objects
                    serialized_value = pickle.dumps(value)
                    serialize_json = False
            else:
                serialized_value = pickle.dumps(value)
            
            # Set with TTL if provided
            if ttl:
                success = await self.redis_client.setex(key, ttl, serialized_value)
            else:
                success = await self.redis_client.set(key, serialized_value)
            
            return bool(success)
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            deleted_count = await self.redis_client.delete(key)
            return deleted_count > 0
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists check error for key {key}: {e}")
            return False
    
    async def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache efficiently"""
        try:
            if not keys:
                return {}
            
            values = await self.redis_client.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        result[key] = pickle.loads(value)
            
            return result
            
        except Exception as e:
            logger.error(f"Cache mget error: {e}")
            return {}
    
    async def set_multiple(
        self, 
        key_value_pairs: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> bool:
        """Set multiple values in cache efficiently"""
        try:
            if not key_value_pairs:
                return True
            
            # Prepare serialized data
            serialized_pairs = {}
            for key, value in key_value_pairs.items():
                try:
                    serialized_pairs[key] = json.dumps(value)
                except (TypeError, ValueError):
                    serialized_pairs[key] = pickle.dumps(value)
            
            # Use pipeline for atomic operations
            async with self.redis_client.pipeline() as pipe:
                await pipe.mset(serialized_pairs)
                
                if ttl:
                    for key in serialized_pairs.keys():
                        await pipe.expire(key, ttl)
                
                await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache mset error: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter in cache"""
        try:
            return await self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL of a key"""
        try:
            ttl = await self.redis_client.ttl(key)
            return ttl if ttl >= 0 else None
        except Exception as e:
            logger.error(f"Cache TTL check error for key {key}: {e}")
            return None
    
    async def flush_db(self) -> bool:
        """Flush all keys from current database"""
        try:
            await self.redis_client.flushdb()
            logger.info("Cache database flushed")
            return True
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = await self.redis_client.info()
            
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'hit_rate': self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                )
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate"""
        total = hits + misses
        return (hits / total) if total > 0 else 0.0
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()
        logger.info("Redis connection closed")

class RecommendationCache:
    """Specialized cache for recommendation data"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.default_ttl = 1800  # 30 minutes
    
    def _get_user_rec_key(self, user_id: int, algorithm: str, num_recs: int) -> str:
        """Generate cache key for user recommendations"""
        return f"rec:user:{user_id}:{algorithm}:{num_recs}"
    
    def _get_user_profile_key(self, user_id: int) -> str:
        """Generate cache key for user profile"""
        return f"profile:user:{user_id}"
    
    def _get_item_features_key(self, item_id: int) -> str:
        """Generate cache key for item features"""
        return f"features:item:{item_id}"
    
    def _get_model_key(self, model_type: str, version: str) -> str:
        """Generate cache key for model artifacts"""
        return f"model:{model_type}:{version}"
    
    async def get_recommendations(
        self, 
        user_id: int, 
        algorithm: str, 
        num_recommendations: int
    ) -> Optional[List[Dict]]:
        """Get cached recommendations for user"""
        key = self._get_user_rec_key(user_id, algorithm, num_recommendations)
        return await self.cache.get(key)
    
    async def set_recommendations(
        self, 
        user_id: int, 
        algorithm: str, 
        num_recommendations: int,
        recommendations: List[Dict],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache recommendations for user"""
        key = self._get_user_rec_key(user_id, algorithm, num_recommendations)
        return await self.cache.set(key, recommendations, ttl or self.default_ttl)
    
    async def get_user_profile(self, user_id: int) -> Optional[Dict]:
        """Get cached user profile"""
        key = self._get_user_profile_key(user_id)
        return await self.cache.get(key)
    
    async def set_user_profile(
        self, 
        user_id: int, 
        profile: Dict, 
        ttl: Optional[int] = None
    ) -> bool:
        """Cache user profile"""
        key = self._get_user_profile_key(user_id)
        return await self.cache.set(key, profile, ttl or 3600)  # 1 hour default
    
    async def get_item_features(self, item_id: int) -> Optional[Dict]:
        """Get cached item features"""
        key = self._get_item_features_key(item_id)
        return await self.cache.get(key)
    
    async def set_item_features(
        self, 
        item_id: int, 
        features: Dict, 
        ttl: Optional[int] = None
    ) -> bool:
        """Cache item features"""
        key = self._get_item_features_key(item_id)
        return await self.cache.set(key, features, ttl or 7200)  # 2 hours default
    
    async def invalidate_user_cache(self, user_id: int) -> bool:
        """Invalidate all cache entries for a user"""
        try:
            # Find all user-related keys
            pattern = f"*:user:{user_id}:*"
            keys = []
            
            # Scan for keys (use cursor to handle large datasets)
            cursor = 0
            while True:
                cursor, found_keys = await self.cache.redis_client.scan(
                    cursor, match=pattern, count=100
                )
                keys.extend([key.decode() for key in found_keys])
                if cursor == 0:
                    break
            
            # Delete all found keys
            if keys:
                deleted = await self.cache.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries for user {user_id}")
                return deleted > 0
            
            return True
            
        except Exception as e:
            logger.error(f"Error invalidating cache for user {user_id}: {e}")
            return False
    
    async def warm_cache(self, user_ids: List[int], recommendations_data: Dict):
        """Warm cache with pre-computed recommendations"""
        try:
            cache_pairs = {}
            
            for user_id in user_ids:
                if user_id in recommendations_data:
                    for algorithm, recs in recommendations_data[user_id].items():
                        key = self._get_user_rec_key(user_id, algorithm, len(recs))
                        cache_pairs[key] = recs
            
            success = await self.cache.set_multiple(cache_pairs, self.default_ttl)
            logger.info(f"Cache warmed for {len(user_ids)} users")
            return success
            
        except Exception as e:
            logger.error(f"Error warming cache: {e}")
            return False

# Performance monitoring decorator
def cache_performance_monitor(func):
    """Decorator to monitor cache performance"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = (time.time() - start_time) * 1000
        
        # Log slow cache operations
        if duration > 10:  # 10ms threshold
            logger.warning(
                f"Slow cache operation: {func.__name__} took {duration:.2f}ms"
            )
        
        return result
    return wrapper

# Example usage
async def example_usage():
    """Example of how to use the cache manager"""
    
    # Initialize cache
    redis_config = {
        'host': 'localhost',
        'port': 6379,
        'db': 0
    }
    
    cache_manager = CacheManager(redis_config)
    rec_cache = RecommendationCache(cache_manager)
    
    try:
        # Cache some recommendations
        recommendations = [
            {'item_id': 1, 'score': 0.95},
            {'item_id': 2, 'score': 0.87},
            {'item_id': 3, 'score': 0.82}
        ]
        
        await rec_cache.set_recommendations(
            user_id=123,
            algorithm='hybrid',
            num_recommendations=3,
            recommendations=recommendations
        )
        
        # Retrieve recommendations
        cached_recs = await rec_cache.get_recommendations(
            user_id=123,
            algorithm='hybrid',
            num_recommendations=3
        )
        
        print(f"Cached recommendations: {cached_recs}")
        
        # Get cache stats
        stats = await cache_manager.get_stats()
        print(f"Cache stats: {stats}")
        
    finally:
        await cache_manager.close()

if __name__ == "__main__":
    asyncio.run(example_usage())
