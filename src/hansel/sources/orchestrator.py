"""Parallel orchestrator for job source adapters.

Fans out search queries across multiple adapters concurrently and 
aggregates results with deduplication.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from hansel.search.schemas import JobQuery
from hansel.sources.base import JobSourceAdapter
from hansel.sources.schemas import JobListing

logger = logging.getLogger(__name__)


class JobSearchOrchestrator:
    """Runs search queries across multiple adapters in parallel.
    
    Failure policy:
      - Individual adapter failures DO NOT cancel other adapters.
      - Adapters that raise are logged and contribute an empty list.
      - The orchestrator always returns a list (possibly empty).
    """
    
    def __init__(
        self,
        adapters: Iterable[JobSourceAdapter],
        per_query_limit: int = 20,
    ) -> None:
        self._adapters = list(adapters)
        self._per_query_limit = per_query_limit
        
        if not self._adapters:
            raise ValueError("At least one adapter is required.")
    
    async def search_all(
        self, queries: Iterable[JobQuery]
    ) -> list[JobListing]:
        """Run each query across all adapters in parallel; deduplicate and return.
        
        Args:
            queries: The JobQuery objects from the QueryGenerator.
        
        Returns:
            A single flat list of JobListing objects, deduplicated.
        """
        queries_list = list(queries)
        if not queries_list:
            logger.warning("No queries provided to orchestrator")
            return []
        
        logger.info(
            "Orchestrating %d queries across %d adapters (%d total calls)",
            len(queries_list), len(self._adapters),
            len(queries_list) * len(self._adapters),
        )
        
        # Build tasks: one per (adapter, query) pair
        tasks = [
            self._safe_search(adapter, query)
            for adapter in self._adapters
            for query in queries_list
        ]
        
        # Run all in parallel
        results_per_task: list[list[JobListing]] = await asyncio.gather(*tasks)
        
        # Flatten
        all_listings: list[JobListing] = []
        for sublist in results_per_task:
            all_listings.extend(sublist)
        
        logger.info(
            "Collected %d raw listings (before dedup)", len(all_listings)
        )
        
        deduplicated = _deduplicate(all_listings)
        logger.info(
            "After dedup: %d unique listings", len(deduplicated)
        )
        
        return deduplicated
    
    async def _safe_search(
        self, adapter: JobSourceAdapter, query: JobQuery
    ) -> list[JobListing]:
        """Run one adapter.search() with defensive error handling."""
        try:
            return await adapter.search(
                keywords=query.keywords,
                location=query.location,
                limit=self._per_query_limit,
            )
        except Exception as e:
            # Broad except is intentional here: the orchestrator's contract 
            # is "never propagate failures of individual adapters".
            logger.exception(
                "Adapter %s failed for query %r (location=%r): %s",
                adapter.name, query.keywords, query.location, e,
            )
            return []


def _deduplicate(listings: list[JobListing]) -> list[JobListing]:
    """Remove duplicates using a two-layer strategy.
    
    Layer 1: exact match on (source, source_id) — catches provider duplicates.
    Layer 2: fuzzy match on (normalized_title, normalized_company, normalized_location)
             — catches cross-query and cross-source duplicates.
    """
    seen_exact: set[tuple[str, str]] = set()
    seen_fuzzy: set[tuple[str, str, str]] = set()
    unique: list[JobListing] = []
    
    for listing in listings:
        exact_key = (listing.source.value, listing.source_id)
        fuzzy_key = (
            _normalize_title(listing.title),
            _normalize_text(listing.company),
            _normalize_text(listing.location or ""),
        )
        
        if exact_key in seen_exact:
            continue
        if fuzzy_key in seen_fuzzy:
            continue
        
        seen_exact.add(exact_key)
        seen_fuzzy.add(fuzzy_key)
        unique.append(listing)
    
    return unique


def _normalize_title(title: str) -> str:
    """Normalize a job title for fuzzy comparison.
    
    Strips parenthetical suffixes, percentages, common gender markers,
    and whitespace. Makes 'Data Engineer' == 'Data Engineer (alle)' ==
    'Data Engineer 80-100%' == 'Data Engineer (m/w/d)'.
    """
    import re
    
    t = title.lower()
    # Remove parentheticals: (alle), (m/w/d), (f/m/d), etc.
    t = re.sub(r"\([^)]*\)", "", t)
    # Remove percentages: 80-100%, 100%
    t = re.sub(r"\d+\s*[-–]\s*\d+\s*%|\d+\s*%", "", t)
    # Remove common seniority qualifiers for dedup (NOT for display)
    t = re.sub(r"\b(senior|junior|mid[-\s]level|lead|principal|staff)\b", "", t)
    # Collapse whitespace and strip
    return " ".join(t.split()).strip()


def _normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace for generic text comparison."""
    return " ".join(text.lower().split())