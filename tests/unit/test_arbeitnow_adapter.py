"""Tests for ArbeitnowAdapter — HTTP mocked with respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from hansel.sources.arbeitnow import ArbeitnowAdapter, _strip_html
from hansel.sources.schemas import JobSource


_API_URL = "https://www.arbeitnow.com/api/job-board-api"


# ---------- Unit tests for pure helpers ----------


class TestStripHtml:
    
    def test_removes_tags(self):
        html = "<p>Hello <strong>world</strong></p>"
        assert _strip_html(html) == "Hello world"
    
    def test_collapses_whitespace(self):
        html = "<p>Too    many     spaces</p>"
        assert _strip_html(html) == "Too many spaces"
    
    def test_handles_empty(self):
        assert _strip_html("") == ""
    
    def test_handles_none(self):
        assert _strip_html(None) == ""  # type: ignore[arg-type]
    
    def test_unwraps_nested_lists(self):
        html = "<ul><li>One</li><li>Two</li></ul>"
        assert _strip_html(html) == "One Two"


# ---------- Factory for fake API responses ----------


def _arbeitnow_item(
    *,
    slug: str = "junior-dev-at-acme-123",
    title: str = "Junior Developer",
    company: str = "ACME Corp",
    location: str = "Berlin",
    remote: bool = False,
    tags: list[str] | None = None,
    description: str = "<p>Exciting role</p>",
) -> dict:
    return {
        "slug": slug,
        "company_name": company,
        "title": title,
        "description": description,
        "remote": remote,
        "url": f"https://www.arbeitnow.com/jobs/{slug}",
        "tags": tags or [],
        "job_types": ["Full Time"],
        "location": location,
        "created_at": 1701234567,
    }


def _payload(items: list[dict]) -> dict:
    return {"data": items, "links": {}, "meta": {}}


# ---------- Integration-ish tests with mocked HTTP ----------


class TestArbeitnowSearchMocked:
    
    @respx.mock
    async def test_parses_basic_listing(self):
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=_payload([
                _arbeitnow_item(
                    slug="python-dev-at-acme",
                    title="Python Developer",
                    company="ACME",
                    location="Berlin",
                ),
            ]))
        )
        
        adapter = ArbeitnowAdapter()
        # Bypass the class-level cache to avoid test leakage
        adapter._cache._cache.clear()
        
        results = await adapter.search(keywords="Python", location=None, limit=10)
        
        assert len(results) == 1
        listing = results[0]
        assert listing.source == JobSource.ARBEITNOW
        assert listing.title == "Python Developer"
        assert listing.company == "ACME"
        assert listing.source_id == "python-dev-at-acme"
    
    @respx.mock
    async def test_filters_by_keyword_and(self):
        """All keywords must be present (AND semantics)."""
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=_payload([
                _arbeitnow_item(slug="a", title="Python Developer"),
                _arbeitnow_item(slug="b", title="JavaScript Developer"),
                _arbeitnow_item(slug="c", title="Python Data Engineer"),
            ]))
        )
        
        adapter = ArbeitnowAdapter()
        adapter._cache._cache.clear()
        
        results = await adapter.search(keywords="python developer", location=None, limit=10)
        
        # Only 'Python Developer' matches BOTH words
        titles = {r.title for r in results}
        assert "Python Developer" in titles
        assert "JavaScript Developer" not in titles
    
    @respx.mock
    async def test_filter_remote_only(self):
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=_payload([
                _arbeitnow_item(slug="a", title="Dev A", remote=True),
                _arbeitnow_item(slug="b", title="Dev B", remote=False),
                _arbeitnow_item(slug="c", title="Dev C", remote=True),
            ]))
        )
        
        adapter = ArbeitnowAdapter()
        adapter._cache._cache.clear()
        
        results = await adapter.search(keywords="dev", location="remote", limit=10)
        
        assert len(results) == 2
        assert all(r.is_remote for r in results)
    
    @respx.mock
    async def test_respects_limit(self):
        items = [_arbeitnow_item(slug=f"job-{i}", title=f"Python {i}") for i in range(20)]
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=_payload(items))
        )
        
        adapter = ArbeitnowAdapter()
        adapter._cache._cache.clear()
        
        results = await adapter.search(keywords="Python", location=None, limit=5)
        assert len(results) == 5
    
    @respx.mock
    async def test_handles_http_error_gracefully(self):
        respx.get(_API_URL).mock(return_value=httpx.Response(500))
        
        adapter = ArbeitnowAdapter()
        adapter._cache._cache.clear()
        
        results = await adapter.search(keywords="Python", location=None, limit=10)
        assert results == []  # Does not raise; returns empty
    
    @respx.mock
    async def test_handles_malformed_json(self):
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, text="not json at all")
        )
        
        adapter = ArbeitnowAdapter()
        adapter._cache._cache.clear()
        
        results = await adapter.search(keywords="Python", location=None, limit=10)
        assert results == []
    
    @respx.mock
    async def test_skips_malformed_items(self):
        """A broken item shouldn't kill the whole response."""
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=_payload([
                _arbeitnow_item(slug="good", title="Python Good"),
                {"slug": "broken"},  # missing required fields
                _arbeitnow_item(slug="also-good", title="Python Other"),
            ]))
        )
        
        adapter = ArbeitnowAdapter()
        adapter._cache._cache.clear()
        
        results = await adapter.search(keywords="Python", location=None, limit=10)
        # Two valid items survive
        assert len(results) == 2
    
    @respx.mock
    async def test_strips_html_from_description(self):
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=_payload([
                _arbeitnow_item(
                    description="<p>Build <strong>awesome</strong> things</p>",
                    title="Python Developer",
                ),
            ]))
        )
        
        adapter = ArbeitnowAdapter()
        adapter._cache._cache.clear()
        
        results = await adapter.search(keywords="Python", location=None, limit=10)
        assert "<p>" not in results[0].description
        assert "<strong>" not in results[0].description
        assert "awesome" in results[0].description