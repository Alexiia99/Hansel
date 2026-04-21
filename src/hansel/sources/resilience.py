"""Resilience utilities for adapters: rate limiting, retries, caching."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from cachetools import TTLCache
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Simple async rate limiter: ensures minimum delay between calls.
    
    Usage:
        limiter = RateLimiter(min_interval=1.1)  # 1.1s between calls
        async with limiter:
            await do_something()
    """
    
    def __init__(self, min_interval: float = 1.0) -> None:
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_call: float = 0.0
    
    async def __aenter__(self) -> None:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = asyncio.get_event_loop().time()
    
    async def __aexit__(self, *exc) -> None:
        pass


def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient HTTP errors (429, 503, timeouts)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 502, 503, 504}
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    return False


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
) -> T:
    """Run an async callable with exponential backoff retries.
    
    Retries on 429, 502, 503, 504 or network errors. Other errors raise.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    ):
        with attempt:
            return await func()
    # Unreachable in practice
    raise RuntimeError("retry_async exited without returning")


class AsyncTTLCache:
    """Async-friendly TTL cache wrapper.
    
    Stores results of expensive async calls (like adapter.search) keyed 
    by the call arguments.
    """
    
    def __init__(self, maxsize: int = 128, ttl_seconds: float = 300.0) -> None:
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self._lock = asyncio.Lock()
    
    async def get_or_compute(
        self,
        key: str,
        compute: Callable[[], Awaitable[T]],
    ) -> T:
        """Return cached value for key, or compute and cache it."""
        async with self._lock:
            if key in self._cache:
                logger.debug("Cache HIT: %s", key)
                return self._cache[key]
        
        logger.debug("Cache MISS: %s", key)
        result = await compute()
        
        async with self._lock:
            self._cache[key] = result
        
        return result