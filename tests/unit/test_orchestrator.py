"""Tests for the parallel job search orchestrator."""

from __future__ import annotations

import asyncio

import pytest

from hansel.search.schemas import JobQuery
from hansel.sources.base import JobSourceAdapter
from hansel.sources.orchestrator import (
    JobSearchOrchestrator,
    _deduplicate,
    _normalize_title,
)
from hansel.sources.schemas import JobListing, JobSource


# ---------- Test doubles (fake adapters) ----------


class FakeAdapter(JobSourceAdapter):
    """Adapter double that returns pre-configured responses.
    
    Each instance tracks how many times .search() was called and with 
    what arguments — useful for asserting the orchestrator did fan-out.
    """
    
    name = "fake"
    
    def __init__(
        self,
        name: str,
        responses: list[JobListing] | None = None,
        delay: float = 0.0,
        raise_exc: Exception | None = None,
    ) -> None:
        self.name = name
        self._responses = responses or []
        self._delay = delay
        self._raise = raise_exc
        self.calls: list[tuple[str, str | None, int]] = []
    
    async def search(
        self,
        keywords: str,
        location: str | None = None,
        limit: int = 20,
    ) -> list[JobListing]:
        self.calls.append((keywords, location, limit))
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raise is not None:
            raise self._raise
        return self._responses


def _make_listing(
    source: JobSource,
    source_id: str,
    title: str = "Software Engineer",
    company: str = "Acme Inc",
    location: str = "Zurich",
) -> JobListing:
    """Factory for JobListing with sensible defaults."""
    return JobListing(
        source=source,
        source_id=source_id,
        url=f"https://example.com/{source_id}",
        title=title,
        company=company,
        location=location,
        description="Job description here.",
    )


# ---------- Tests for _deduplicate helper ----------


class TestDeduplicate:
    
    def test_removes_exact_source_id_duplicates(self):
        a = _make_listing(JobSource.ADZUNA, "1")
        b = _make_listing(JobSource.ADZUNA, "1")  # same source+id
        result = _deduplicate([a, b])
        assert len(result) == 1
    
    def test_keeps_different_sources(self):
        a = _make_listing(JobSource.ADZUNA, "1", title="Data Engineer")
        b = _make_listing(JobSource.ARBEITNOW, "1", title="Data Engineer")
        result = _deduplicate([a, b])
        # Different sources but same title+company+location → fuzzy dedup kicks in
        # First one wins
        assert len(result) == 1
    
    def test_fuzzy_dedup_parenthetical_variants(self):
        a = _make_listing(JobSource.ADZUNA, "1", title="Data Engineer")
        b = _make_listing(JobSource.ADZUNA, "2", title="Data Engineer (alle)")
        c = _make_listing(JobSource.ADZUNA, "3", title="Data Engineer 80-100%")
        result = _deduplicate([a, b, c])
        assert len(result) == 1
    
    def test_fuzzy_dedup_senior_junior_same_job(self):
        a = _make_listing(JobSource.ADZUNA, "1", title="Senior Data Engineer")
        b = _make_listing(JobSource.ADZUNA, "2", title="Data Engineer")
        result = _deduplicate([a, b])
        assert len(result) == 1
    
    def test_different_companies_not_merged(self):
        a = _make_listing(JobSource.ADZUNA, "1", title="Data Engineer", company="Acme")
        b = _make_listing(JobSource.ADZUNA, "2", title="Data Engineer", company="Widgets")
        result = _deduplicate([a, b])
        assert len(result) == 2
    
    def test_empty_input(self):
        assert _deduplicate([]) == []


class TestNormalizeTitle:
    
    def test_removes_parenthetical(self):
        assert _normalize_title("Data Engineer (alle)") == "data engineer"
        assert _normalize_title("Engineer (m/w/d)") == "engineer"
    
    def test_removes_percentages(self):
        assert _normalize_title("Developer 80-100%") == "developer"
        assert _normalize_title("Role 50%") == "role"
    
    def test_removes_seniority(self):
        assert _normalize_title("Senior Developer") == "developer"
        assert _normalize_title("Junior Engineer") == "engineer"
        assert _normalize_title("Lead Architect") == "architect"
    
    def test_preserves_core_title(self):
        assert _normalize_title("Backend Developer") == "backend developer"
    
    def test_collapses_whitespace(self):
        assert _normalize_title("  Data   Engineer  ") == "data engineer"


# ---------- Tests for the orchestrator itself ----------


class TestOrchestratorFanout:
    
    async def test_runs_each_query_on_each_adapter(self):
        a1 = FakeAdapter("a1", responses=[_make_listing(JobSource.ADZUNA, "x1")])
        a2 = FakeAdapter("a2", responses=[_make_listing(JobSource.ARBEITNOW, "x2")])
        orch = JobSearchOrchestrator(adapters=[a1, a2])
        
        queries = [
            JobQuery(keywords="Python", location="Zurich", rationale="."),
            JobQuery(keywords="Go", location="anywhere", rationale="."),
        ]
        
        results = await orch.search_all(queries)
        
        # Each adapter should see every query
        assert len(a1.calls) == 2
        assert len(a2.calls) == 2
        # And the keywords of those calls should match the queries
        assert {c[0] for c in a1.calls} == {"Python", "Go"}
    
    async def test_returns_deduplicated_listings(self):
        # Same listing returned twice → dedup should remove one
        shared = _make_listing(JobSource.ADZUNA, "same-id")
        a1 = FakeAdapter("a1", responses=[shared])
        orch = JobSearchOrchestrator(adapters=[a1])
        
        queries = [
            JobQuery(keywords="Q1", location="anywhere", rationale="."),
            JobQuery(keywords="Q2", location="anywhere", rationale="."),
        ]
        
        results = await orch.search_all(queries)
        assert len(results) == 1


class TestOrchestratorResilience:
    
    async def test_adapter_exception_does_not_break_others(self):
        """If one adapter crashes, the orchestrator still returns results from the others."""
        good = FakeAdapter("good", responses=[_make_listing(JobSource.ADZUNA, "1")])
        bad = FakeAdapter("bad", raise_exc=RuntimeError("boom"))
        orch = JobSearchOrchestrator(adapters=[good, bad])
        
        queries = [JobQuery(keywords="Q", location="anywhere", rationale=".")]
        
        results = await orch.search_all(queries)
        
        # Good adapter's result survives
        assert len(results) == 1
        assert results[0].source == JobSource.ADZUNA
    
    async def test_all_adapters_fail_returns_empty(self):
        bad1 = FakeAdapter("bad1", raise_exc=RuntimeError("boom"))
        bad2 = FakeAdapter("bad2", raise_exc=ValueError("oops"))
        orch = JobSearchOrchestrator(adapters=[bad1, bad2])
        
        queries = [JobQuery(keywords="Q", location="anywhere", rationale=".")]
        
        results = await orch.search_all(queries)
        assert results == []
    
    async def test_no_adapters_raises(self):
        with pytest.raises(ValueError, match="(?i)at least one adapter"):
            JobSearchOrchestrator(adapters=[])
    
    async def test_no_queries_returns_empty(self):
        a = FakeAdapter("a", responses=[_make_listing(JobSource.ADZUNA, "1")])
        orch = JobSearchOrchestrator(adapters=[a])
        
        results = await orch.search_all([])
        assert results == []
        assert a.calls == []  # no calls made


class TestOrchestratorParallelism:
    
    async def test_adapters_run_in_parallel(self):
        """Two slow adapters should take ~delay, not 2*delay."""
        import time
        
        delay = 0.3
        a1 = FakeAdapter("a1", responses=[_make_listing(JobSource.ADZUNA, "1")], delay=delay)
        a2 = FakeAdapter("a2", responses=[_make_listing(JobSource.ARBEITNOW, "2")], delay=delay)
        orch = JobSearchOrchestrator(adapters=[a1, a2])
        
        queries = [JobQuery(keywords="Q", location="anywhere", rationale=".")]
        
        start = time.monotonic()
        await orch.search_all(queries)
        elapsed = time.monotonic() - start
        
        # Parallel execution: should be close to `delay`, not `2*delay`.
        # Generous margin to avoid flakiness on slow CI.
        assert elapsed < delay * 1.8, f"Expected parallelism (~{delay}s), took {elapsed:.2f}s"