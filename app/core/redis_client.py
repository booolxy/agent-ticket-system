# app/core/redis_client.py
"""
Redis 客户端封装
用途：
  1. 检索结果缓存（缓存 RAG 结果，降低重复查询延迟）
  2. 令牌桶限流（分布式版）
  3. BM25 索引持久化（避免每次重建）
"""
import json
import hashlib
from typing import Optional, Any
from functools import lru_cache

import redis

from app.core.config import settings


@lru_cache()
def get_redis() -> redis.Redis:
    """Redis 单例（连接池）"""
    pool = redis.ConnectionPool.from_url(
        settings.REDIS_URL,
        max_connections=20,
        decode_responses=True,
    )
    return redis.Redis(connection_pool=pool)


class RedisCache:
    """通用缓存封装"""

    def __init__(self, prefix: str, ttl: int = 3600):
        self.r = get_redis()
        self.prefix = prefix
        self.ttl = ttl

    def _key(self, raw: str) -> str:
        h = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"{self.prefix}:{h}"

    def get(self, key: str) -> Optional[Any]:
        try:
            val = self.r.get(self._key(key))
            return json.loads(val) if val else None
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        try:
            self.r.setex(
                self._key(key),
                ttl or self.ttl,
                json.dumps(value, ensure_ascii=False),
            )
        except Exception:
            pass

    def clear(self):
        try:
            for k in self.r.scan_iter(f"{self.prefix}:*"):
                self.r.delete(k)
        except Exception:
            pass


# ==================== 全局缓存实例 ====================
# RAG 检索结果缓存（1 小时）
rag_cache = RedisCache(prefix="rag", ttl=3600)
# HyDE 改写结果缓存（30 分钟）
hyde_cache = RedisCache(prefix="hyde", ttl=1800)