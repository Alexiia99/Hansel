"""Adapter for the Arbeitnow job board API.

Arbeitnow (https://www.arbeitnow.com) is a free, unauthenticated job board
focused on Germany and remote positions. Its API returns the full current
feed without search parameters, so we filter client-side.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx
import re

from hansel.sources.base import JobSourceAdapter
from hansel.sources.schemas import JobListing, JobSource
from hansel.sources.resilience import AsyncTTLCache, RateLimiter

logger = logging.getLogger(__name__)

_API_URL = "https://www.arbeitnow.com/api/job-board-api"
_DEFAULT_TIMEOUT = 15.0  # seconds


def _strip_html(html: str) -> str:
    """Extract plain text from an HTML string."""
    if not html:
        return ""
    # Remove HTML tags, then collapse whitespace
    text = re.sub(r"<[^>]+>", " ", html)
    return " ".join(text.split())

def _matches_keywords(item: dict[str, Any], keywords: str) -> bool:
    """Check if a raw Arbeitnow item matches the keywords (client-side filter)."""
    if not keywords:
        return True
    
    needle = keywords.lower()
    haystack = " ".join([
        item.get("title", ""),
        item.get("company_name", ""),
        item.get("location", ""),
        " ".join(item.get("tags") or []),
    ]).lower()
    
    # All words in keywords must appear somewhere (AND semantics)
    return all(word in haystack for word in needle.split())


def _matches_location(item: dict[str, Any], location: str | None) -> bool:
    """Check if the item matches the requested location."""
    if not location:
        return True
    
    loc_lower = location.lower().strip()
    
    # Special case: user wants remote
    if loc_lower in {"remote", "anywhere"}:
        return bool(item.get("remote"))
    
    # Otherwise, substring match in location field
    item_location = (item.get("location") or "").lower()
    return loc_lower in item_location


def _parse_listing(item: dict[str, Any]) -> JobListing | None:
    """Convert a raw Arbeitnow item to our unified JobListing schema.
    
    Returns None if the item is malformed (missing required fields).
    """
    try:
        posted_at = None
        if ts := item.get("created_at"):
            posted_at = datetime.fromtimestamp(int(ts))
        
        return JobListing(
            source=JobSource.ARBEITNOW,
            source_id=item["slug"],
            url=item["url"],
            title=item["title"],
            company=item["company_name"],
            location=item.get("location"),
            description=_strip_html(item.get("description", "")),
            is_remote=bool(item.get("remote")),
            posted_at=posted_at,
            tags=item.get("tags") or [],
        )
    except (KeyError, ValueError) as e:
        logger.warning("Skipped malformed Arbeitnow item: %s", e)
        return None


class ArbeitnowAdapter(JobSourceAdapter):
    """Adapter for Arbeitnow's public job board API.
    
    No authentication required. Client-side filtering on a global feed.
    """
    
    name = "Arbeitnow"
    
    # Class-level: one rate limiter and cache shared across all instances
    _rate_limiter = RateLimiter(min_interval=0.5)
    _cache = AsyncTTLCache(maxsize=128, ttl_seconds=300)
    
    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout
    
    async def search(
        self,
        keywords: str,
        location: str | None = None,
        limit: int = 20,
    ) -> list[JobListing]:
        import hashlib
        key = hashlib.sha1(
            f"{keywords}|{location or ''}|{limit}".encode()
        ).hexdigest()[:16]
        
        async def _fetch() -> list[JobListing]:
            async with self._rate_limiter:
                return await self._fetch_feed(keywords, location, limit)
        
        return await self._cache.get_or_compute(key, _fetch)
    
    async def _fetch_feed(
        self, keywords: str, location: str | None, limit: int
    ) -> list[JobListing]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(_API_URL)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as e:
            logger.error("Arbeitnow request failed: %s", e)
            return []
        except ValueError as e:
            logger.error("Arbeitnow returned invalid JSON: %s", e)
            return []
        
        raw_items = payload.get("data", [])
        logger.info("Arbeitnow returned %d raw items", len(raw_items))
        
        matched: list[JobListing] = []
        for item in raw_items:
            if not _matches_keywords(item, keywords):
                continue
            if not _matches_location(item, location):
                continue
            listing = _parse_listing(item)
            if listing is not None:
                matched.append(listing)
                if len(matched) >= limit:
                    break
        
        logger.info(
            "Arbeitnow matched %d items for keywords=%r location=%r",
            len(matched), keywords, location,
        )
        return matched