"""Adapter for the Adzuna job search API."""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime
from typing import Any

import httpx

from hansel.sources.base import JobSourceAdapter
from hansel.sources.resilience import AsyncTTLCache, RateLimiter, retry_async
from hansel.sources.schemas import JobListing, JobSource

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
_DEFAULT_COUNTRY = "ch"
_DEFAULT_TIMEOUT = 15.0
# Adzuna free tier: stay well below their rate limit
_MIN_INTERVAL = 1.2  # seconds between consecutive calls

# Country names the `/{country}/` endpoint already filters for.
# Passing these as `where=` does strict substring matching that kills recall.
_COUNTRY_NAMES_IMPLICIT = {
    "switzerland", "schweiz", "suisse", "svizzera",
    "germany", "deutschland", "allemagne",
    "austria", "österreich",
    "france",
    "united kingdom", "uk",
}


def _parse_created(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _infer_remote(item: dict[str, Any]) -> bool | None:
    text = " ".join([item.get("title", ""), item.get("description", "")]).lower()
    remote_markers = ["remote", "home office", "work from home", "télétravail", "teletrabajo"]
    if any(marker in text for marker in remote_markers):
        return True
    return None


def _parse_listing(item: dict[str, Any]) -> JobListing | None:
    try:
        source_id = str(item["id"])
        url = item["redirect_url"]
        title = item["title"]
        company = item.get("company", {}).get("display_name") or "Unknown"
        location = item.get("location", {}).get("display_name")
        description = item.get("description", "")
        
        salary_min = item.get("salary_min")
        salary_max = item.get("salary_max")
        salary_currency = "CHF" if salary_min or salary_max else None
        
        tags: list[str] = []
        if cat := item.get("category", {}).get("label"):
            if cat.lower() != "unknown":
                tags.append(cat)
        if ct := item.get("contract_type"):
            tags.append(ct)
        if ctime := item.get("contract_time"):
            tags.append(ctime)
        
        return JobListing(
            source=JobSource.ADZUNA,
            source_id=source_id,
            url=url,
            title=title,
            company=company,
            location=location,
            description=description,
            salary_min=float(salary_min) if salary_min else None,
            salary_max=float(salary_max) if salary_max else None,
            salary_currency=salary_currency,
            is_remote=_infer_remote(item),
            posted_at=_parse_created(item.get("created")),
            tags=tags,
        )
    except (KeyError, ValueError) as e:
        logger.warning("Skipped malformed Adzuna item: %s", e)
        return None


def _cache_key(keywords: str, location: str | None, limit: int, country: str) -> str:
    """Stable cache key for (keywords, location, limit, country) tuple."""
    raw = f"{country}|{keywords}|{location or ''}|{limit}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class AdzunaAdapter(JobSourceAdapter):
    """Adzuna adapter with rate limiting, retries, and in-memory TTL cache."""
    
    name = "Adzuna"
    
    # Class-level rate limiter: one across ALL instances
    # so concurrent queries don't trigger 429s
    _rate_limiter = RateLimiter(min_interval=_MIN_INTERVAL)
    # Class-level cache: shared across instances
    _cache = AsyncTTLCache(maxsize=256, ttl_seconds=300)
    
    def __init__(
        self,
        app_id: str | None = None,
        app_key: str | None = None,
        country: str = _DEFAULT_COUNTRY,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._app_id = app_id or os.getenv("ADZUNA_APP_ID")
        self._app_key = app_key or os.getenv("ADZUNA_APP_KEY")
        self._country = country
        self._timeout = timeout
        
        if not self._app_id or not self._app_key:
            raise ValueError(
                "Adzuna credentials missing. Set ADZUNA_APP_ID and ADZUNA_APP_KEY "
                "in your .env file or pass them explicitly."
            )
    
    
    async def search(
        self,
        keywords: str,
        location: str | None = None,
        limit: int = 20,
    ) -> list[JobListing]:
        """Search Adzuna for jobs. Never raises; returns [] on failure."""
        key = _cache_key(keywords, location, limit, self._country)
    
        async def _fetch() -> list[JobListing]:
            try:
                return await self._fetch_with_retry(keywords, location, limit)
            except httpx.HTTPError as e:
                logger.error("Adzuna request failed after retries: %s", e)
                return []
            except ValueError as e:
                # JSON decode errors or Pydantic validation issues
                logger.error("Adzuna response parsing failed: %s", e)
                return []
    
        return await self._cache.get_or_compute(key, _fetch)
    
    async def _fetch_with_retry(
        self, keywords: str, location: str | None, limit: int
    ) -> list[JobListing]:
        """Inner fetch: rate-limited, with retry on 429/5xx."""
        async with self._rate_limiter:
            return await retry_async(
                lambda: self._fetch_once(keywords, location, limit),
                max_attempts=3,
                min_wait=2.0,
                max_wait=10.0,
            )
    
    def _build_where_param(self, location: str | None) -> str | None:
        """Decide whether to pass `where` to Adzuna.
        
        Rules:
          - None / 'remote' / 'anywhere' → no `where` (country endpoint handles it)
          - Country names that match the endpoint → no `where` (redundant, kills recall)
          - City/region names → pass as `where`
        """
        if not location:
            return None
        loc_lower = location.lower().strip()
        if loc_lower in {"remote", "anywhere"}:
            return None
        if loc_lower in _COUNTRY_NAMES_IMPLICIT:
            logger.debug(
                "Dropping redundant location %r (already implicit in country endpoint)",
                location,
            )
            return None
        return location
    
    async def _fetch_once(
        self, keywords: str, location: str | None, limit: int
    ) -> list[JobListing]:
        """One HTTP call to Adzuna."""
        params: dict[str, str | int] = {
            "app_id": self._app_id,
            "app_key": self._app_key,
            "results_per_page": min(limit, 50),
            "what": keywords,
            "content-type": "application/json",
        }
        
        where = self._build_where_param(location)
        if where is not None:
            params["where"] = where
        
        url = f"{_BASE_URL}/{self._country}/search/1"
        
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()  # retry layer catches 429, 5xx
            payload = response.json()
        
        raw_items = payload.get("results", [])
        total = payload.get("count", 0)
        logger.info(
            "Adzuna: %d items of %d total (keywords=%r, location=%r)",
            len(raw_items), total, keywords, location,
        )
        
        listings: list[JobListing] = [
            parsed for item in raw_items
            if (parsed := _parse_listing(item)) is not None
        ]
        
        # If user asked for remote, filter out listings explicitly NOT remote.
        # _infer_remote returns True or None (never False), so this keeps 
        # listings with unknown remote status (could still be remote).
        if location and location.strip().lower() in {"remote", "anywhere"}:
            listings = [l for l in listings if l.is_remote is not False]
        
        return listings