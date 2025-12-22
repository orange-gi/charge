from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional


class CacheBackend:
    def get(self, key: str) -> Optional[dict]:
        raise NotImplementedError

    def set(self, key: str, value: dict, ttl_seconds: int) -> None:
        raise NotImplementedError

    def incr(self, key: str) -> int:
        raise NotImplementedError


@dataclass
class _MemItem:
    expires_at: float
    value: dict


class MemoryTTLCache(CacheBackend):
    """极简内存 TTL 缓存（线程安全）。"""

    def __init__(self, max_items: int = 2048) -> None:
        self._max_items = max_items
        self._lock = threading.Lock()
        self._data: dict[str, _MemItem] = {}

    def _purge_if_needed(self) -> None:
        if len(self._data) <= self._max_items:
            return
        # 简单随机淘汰 + 清理过期（保持实现简洁）
        now = time.time()
        expired_keys = [k for k, v in self._data.items() if v.expires_at <= now]
        for k in expired_keys[: min(len(expired_keys), 256)]:
            self._data.pop(k, None)
        if len(self._data) <= self._max_items:
            return
        keys = list(self._data.keys())
        random.shuffle(keys)
        for k in keys[: max(1, len(keys) - self._max_items)]:
            self._data.pop(k, None)

    def get(self, key: str) -> Optional[dict]:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            if item.expires_at <= now:
                self._data.pop(key, None)
                return None
            return item.value

    def set(self, key: str, value: dict, ttl_seconds: int) -> None:
        expires_at = time.time() + max(1, ttl_seconds)
        with self._lock:
            self._data[key] = _MemItem(expires_at=expires_at, value=value)
            self._purge_if_needed()

    def incr(self, key: str) -> int:
        with self._lock:
            current = self._data.get(key)
            now = time.time()
            if not current or current.expires_at <= now:
                val = 1
            else:
                try:
                    val = int(current.value.get("value", 0)) + 1
                except Exception:
                    val = 1
            self._data[key] = _MemItem(expires_at=now + 24 * 3600, value={"value": val})
            return val


class RedisJSONCache(CacheBackend):
    """Redis JSON 缓存；不可用时应由调用方降级到内存缓存。"""

    def __init__(self, redis_url: str) -> None:
        import redis  # type: ignore

        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        # 连接探测：失败则抛出，让上层降级
        self._redis.ping()

    def get(self, key: str) -> Optional[dict]:
        raw = self._redis.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set(self, key: str, value: dict, ttl_seconds: int) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        # TTL 抖动，避免同一时刻大量 key 同时过期
        ttl = max(1, int(ttl_seconds))
        ttl = int(ttl * (0.9 + random.random() * 0.2))
        self._redis.setex(key, ttl, payload)

    def incr(self, key: str) -> int:
        return int(self._redis.incr(key))


class HybridCache:
    """优先 Redis，不可用则退化到内存缓存。"""

    def __init__(self, redis_url: str) -> None:
        self._backend: CacheBackend
        try:
            self._backend = RedisJSONCache(redis_url)
        except Exception:
            self._backend = MemoryTTLCache()

    def get_json(self, key: str) -> Optional[dict]:
        return self._backend.get(key)

    def set_json(self, key: str, value: dict, ttl_seconds: int) -> None:
        self._backend.set(key, value, ttl_seconds)

    def incr(self, key: str) -> int:
        return self._backend.incr(key)

