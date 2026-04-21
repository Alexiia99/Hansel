"""Tests for AdzunaAdapter — HTTP mocked with respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from hansel.sources.adzuna import AdzunaAdapter, _infer_remote, _parse_listing
from hansel.sources.schemas import JobSource


_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
_CH_URL = f"{_BASE_URL}/ch/search/1"


# ---------- Credential validation ----------


class TestConstructor:
    
    def test_raises_without_credentials(self, monkeypatch):
        monkeypatch.delenv("ADZUNA_APP_ID", raising=False)
        monkeypatch.delenv("ADZUNA_APP_KEY", raising=False)
        
        with pytest.raises(ValueError, match="credentials missing"):
            AdzunaAdapter()
    
    def test_accepts_explicit_credentials(self):
        # Should not raise
        adapter = AdzunaAdapter(app_id="test_id", app_key="test_key")
        assert adapter._app_id == "test_id"
        assert adapter._app_key == "test_key"
    
    def test_uses_env_vars_when_no_args(self, monkeypatch):
        monkeypatch.setenv("ADZUNA_APP_ID", "env_id")
        monkeypatch.setenv("ADZUNA_APP_KEY", "env_key")
        
        adapter = AdzunaAdapter()
        assert adapter._app_id == "env_id"
        assert adapter._app_key == "env_key"


# ---------- _build_where_param helper ----------


class TestBuildWhereParam:
    
    def setup_method(self):
        self.adapter = AdzunaAdapter(app_id="x", app_key="y", country="ch")
    
    def test_none_returns_none(self):
        assert self.adapter._build_where_param(None) is None
    
    def test_remote_returns_none(self):
        assert self.adapter._build_where_param("remote") is None
        assert self.adapter._build_where_param("anywhere") is None
        assert self.adapter._build_where_param("Remote") is None  # case-insensitive
    
    def test_country_names_dropped(self):
        """Country names kill recall on the /ch/ endpoint — drop them."""
        assert self.adapter._build_where_param("Switzerland") is None
        assert self.adapter._build_where_param("schweiz") is None
        assert self.adapter._build_where_param("Suisse") is None
    
    def test_city_names_pass_through(self):
        assert self.adapter._build_where_param("Zurich") == "Zurich"
        assert self.adapter._build_where_param("Aarau") == "Aarau"
    
    def test_strips_whitespace(self):
        assert self.adapter._build_where_param("  Switzerland  ") is None
        assert self.adapter._build_where_param(" Zurich ") == " Zurich "  # inside untouched


# ---------- _infer_remote ----------


class TestInferRemote:
    
    def test_detects_remote_in_title(self):
        item = {"title": "Senior Developer (Remote)", "description": "..."}
        assert _infer_remote(item) is True
    
    def test_detects_home_office(self):
        item = {"title": "Developer", "description": "Home office possible"}
        assert _infer_remote(item) is True
    
    def test_returns_none_when_unknown(self):
        item = {"title": "Developer", "description": "Work at our office"}
        assert _infer_remote(item) is None
    
    def test_handles_missing_fields(self):
        assert _infer_remote({}) is None


# ---------- _parse_listing ----------


class TestParseListing:
    
    def test_full_item_parses(self):
        item = {
            "id": "123456",
            "redirect_url": "https://example.com/job/123",
            "title": "Senior Python Developer",
            "company": {"display_name": "ACME Corp"},
            "location": {
                "display_name": "Zurich",
                "area": ["Schweiz", "Kanton Zürich", "Zürich"],
            },
            "description": "Build great things",
            "created": "2026-04-15T10:00:00Z",
            "salary_min": 80000,
            "salary_max": 120000,
            "category": {"label": "IT Jobs"},
            "contract_type": "permanent",
            "contract_time": "full_time",
        }
        
        listing = _parse_listing(item)
        
        assert listing is not None
        assert listing.source == JobSource.ADZUNA
        assert listing.source_id == "123456"
        assert listing.title == "Senior Python Developer"
        assert listing.company == "ACME Corp"
        assert listing.location == "Zurich"
        assert listing.salary_min == 80000.0
        assert listing.salary_max == 120000.0
        assert listing.salary_currency == "CHF"
        assert "IT Jobs" in listing.tags
        assert "permanent" in listing.tags
        assert "full_time" in listing.tags
    
    def test_missing_required_returns_none(self):
        item = {"title": "No ID"}  # missing id and redirect_url
        assert _parse_listing(item) is None
    
    def test_unknown_category_excluded_from_tags(self):
        item = {
            "id": "1",
            "redirect_url": "https://x.com",
            "title": "Developer",
            "company": {"display_name": "X"},
            "location": {"display_name": "Y"},
            "description": "",
            "category": {"label": "Unknown"},
        }
        listing = _parse_listing(item)
        assert listing is not None
        assert "Unknown" not in listing.tags
    
    def test_missing_company_defaults_to_unknown(self):
        item = {
            "id": "1",
            "redirect_url": "https://x.com",
            "title": "Developer",
            "company": {},  # no display_name
            "location": {"display_name": "Y"},
            "description": "",
        }
        listing = _parse_listing(item)
        assert listing is not None
        assert listing.company == "Unknown"
    
    def test_salary_currency_only_when_salary_present(self):
        item_no_salary = {
            "id": "1",
            "redirect_url": "https://x.com",
            "title": "Dev",
            "company": {"display_name": "X"},
            "location": {"display_name": "Y"},
            "description": "",
        }
        listing = _parse_listing(item_no_salary)
        assert listing is not None
        assert listing.salary_currency is None


# ---------- End-to-end .search() with mocked HTTP ----------


def _adzuna_payload(items: list[dict], total: int | None = None) -> dict:
    return {
        "count": total if total is not None else len(items),
        "results": items,
        "__CLASS__": "Adzuna::API::Response::JobSearchResults",
    }


def _adzuna_item(
    *,
    id: str = "1",
    title: str = "Software Engineer",
    company: str = "Acme",
    location_name: str = "Zurich",
    description: str = "Job description",
) -> dict:
    return {
        "id": id,
        "redirect_url": f"https://adzuna.ch/jobs/{id}",
        "title": title,
        "company": {"display_name": company},
        "location": {"display_name": location_name, "area": []},
        "description": description,
        "created": "2026-04-15T10:00:00Z",
    }


class TestAdzunaSearchMocked:
    
    @respx.mock
    async def test_basic_search_parses_results(self):
        respx.get(_CH_URL).mock(
            return_value=httpx.Response(200, json=_adzuna_payload([
                _adzuna_item(id="1", title="Python Developer"),
                _adzuna_item(id="2", title="Data Engineer"),
            ]))
        )
        
        adapter = AdzunaAdapter(app_id="x", app_key="y")
        adapter._cache._cache.clear()
        
        results = await adapter.search(keywords="Python", location=None, limit=10)
        
        assert len(results) == 2
        assert results[0].title == "Python Developer"
        assert results[1].source == JobSource.ADZUNA
    
    @respx.mock
    async def test_drops_country_name_from_where_param(self):
        """Switzerland should NOT appear in the URL — it's implicit in /ch/."""
        route = respx.get(_CH_URL).mock(
            return_value=httpx.Response(200, json=_adzuna_payload([]))
        )
        
        adapter = AdzunaAdapter(app_id="x", app_key="y")
        adapter._cache._cache.clear()
        
        await adapter.search(keywords="Developer", location="Switzerland", limit=10)
        
        # Inspect the actual HTTP request that was made
        assert route.called
        request_url = str(route.calls[0].request.url)
        assert "where=" not in request_url, f"URL contained 'where=': {request_url}"
    
    @respx.mock
    async def test_passes_city_name_as_where(self):
        route = respx.get(_CH_URL).mock(
            return_value=httpx.Response(200, json=_adzuna_payload([]))
        )
        
        adapter = AdzunaAdapter(app_id="x", app_key="y")
        adapter._cache._cache.clear()
        
        await adapter.search(keywords="Developer", location="Zurich", limit=10)
        
        assert route.called
        request_url = str(route.calls[0].request.url)
        assert "where=Zurich" in request_url
    
    @respx.mock
    async def test_http_error_returns_empty(self):
        respx.get(_CH_URL).mock(return_value=httpx.Response(500))
        
        adapter = AdzunaAdapter(app_id="x", app_key="y")
        adapter._cache._cache.clear()
        
        results = await adapter.search(keywords="Python", location=None, limit=10)
        assert results == []
    
    @respx.mock
    async def test_uses_cache_on_repeated_calls(self):
        """Second identical call should NOT hit the network."""
        route = respx.get(_CH_URL).mock(
            return_value=httpx.Response(200, json=_adzuna_payload([
                _adzuna_item(id="1", title="Python Dev"),
            ]))
        )
        
        adapter = AdzunaAdapter(app_id="x", app_key="y")
        adapter._cache._cache.clear()
        
        results1 = await adapter.search(keywords="Python", location=None, limit=10)
        results2 = await adapter.search(keywords="Python", location=None, limit=10)
        
        assert len(results1) == 1
        assert len(results2) == 1
        # Only one HTTP call; the second used the cache
        assert route.call_count == 1