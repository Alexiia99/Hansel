"""Tests for SwissDevJobsAdapter — HTTP mocked with respx."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from hansel.sources.swissdevjobs import SwissDevJobsAdapter, _is_relevant, _parse_job
from hansel.sources.schemas import JobSource


_API_URL = "https://swissdevjobs.ch/api/jobsLight"


# ---------- Factory for fake API responses ----------


def _sdj_item(
    *,
    id: str = "abc123",
    job_url: str = "Acme-Python-Developer",
    name: str = "Python Developer",
    company: str = "Acme AG",
    actual_city: str = "Zürich",
    workplace: str = "hybrid",
    tech_category: str = "python",
    meta_category: str = "python",
    technologies: list[str] | None = None,
    salary_from: int | None = 80000,
    salary_to: int | None = 100000,
    exp_level: str = "Junior",
) -> dict:
    return {
        "_id": id,
        "jobUrl": job_url,
        "name": name,
        "company": company,
        "actualCity": actual_city,
        "cityCategory": actual_city,
        "workplace": workplace,
        "techCategory": tech_category,
        "metaCategory": meta_category,
        "technologies": technologies or ["Python", "FastAPI"],
        "annualSalaryFrom": salary_from,
        "annualSalaryTo": salary_to,
        "expLevel": exp_level,
        "isPartner": False,
        "isPaused": False,
    }


# ---------- Unit tests for pure helpers ----------


class TestIsRelevant:

    def test_python_category_is_relevant(self):
        assert _is_relevant(_sdj_item(tech_category="python")) is True

    def test_machine_learning_is_relevant(self):
        assert _is_relevant(_sdj_item(tech_category="machine-learning")) is True

    def test_mlaidata_meta_is_relevant(self):
        assert _is_relevant(_sdj_item(meta_category="mlaidata")) is True

    def test_java_is_not_relevant(self):
        assert _is_relevant(_sdj_item(tech_category="java", meta_category="java")) is False

    def test_case_insensitive(self):
        assert _is_relevant(_sdj_item(tech_category="Python")) is True


class TestParseJob:

    def test_parses_basic_fields(self):
        item = _sdj_item()
        listing = _parse_job(item)
        assert listing is not None
        assert listing.title == "Python Developer"
        assert listing.company == "Acme AG"
        assert listing.location == "Zürich"
        assert listing.source == JobSource.SWISSDEVJOBS
        assert listing.source_id == "abc123"

    def test_url_is_correct(self):
        item = _sdj_item(job_url="Acme-Python-Developer")
        listing = _parse_job(item)
        assert str(listing.url) == "https://swissdevjobs.ch/jobs/Acme-Python-Developer"

    def test_salary_parsed(self):
        item = _sdj_item(salary_from=80000, salary_to=100000)
        listing = _parse_job(item)
        assert listing.salary_min == 80000.0
        assert listing.salary_max == 100000.0
        assert listing.salary_currency == "CHF"

    def test_salary_none_when_missing(self):
        item = _sdj_item(salary_from=None, salary_to=None)
        listing = _parse_job(item)
        assert listing.salary_min is None
        assert listing.salary_max is None
        assert listing.salary_currency is None

    def test_remote_workplace(self):
        item = _sdj_item(workplace="remote")
        listing = _parse_job(item)
        assert listing.is_remote is True
        assert "Remote" in listing.location

    def test_hybrid_not_remote(self):
        item = _sdj_item(workplace="hybrid")
        listing = _parse_job(item)
        assert listing.is_remote is False

    def test_tags_capped_at_10(self):
        item = _sdj_item(technologies=[f"Tech{i}" for i in range(20)])
        listing = _parse_job(item)
        assert len(listing.tags) <= 10

    def test_returns_none_when_name_missing(self):
        item = _sdj_item()
        item["name"] = ""
        assert _parse_job(item) is None

    def test_returns_none_when_url_missing(self):
        item = _sdj_item()
        item["jobUrl"] = ""
        assert _parse_job(item) is None

    def test_description_includes_salary(self):
        item = _sdj_item(salary_from=90000, salary_to=110000)
        listing = _parse_job(item)
        assert "90" in listing.description
        assert "110" in listing.description

    def test_description_includes_level(self):
        item = _sdj_item(exp_level="Junior")
        listing = _parse_job(item)
        assert "Junior" in listing.description


# ---------- Integration tests with mocked HTTP ----------


class TestSwissDevJobsAdapterMocked:

    @respx.mock
    async def test_returns_listings(self):
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=[_sdj_item()])
        )
        adapter = SwissDevJobsAdapter(relevant_only=False)
        results = await adapter.search("python")
        assert len(results) == 1
        assert results[0].title == "Python Developer"

    @respx.mock
    async def test_filters_irrelevant_categories(self):
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=[
                _sdj_item(id="1", tech_category="python"),
                _sdj_item(id="2", name="Java Dev", tech_category="java", meta_category="java"),
            ])
        )
        adapter = SwissDevJobsAdapter(relevant_only=True)
        results = await adapter.search("python")
        assert len(results) == 1
        assert results[0].source_id == "1"

    @respx.mock
    async def test_caches_result(self):
        """Second search() call must not fire a second HTTP request."""
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=[_sdj_item()])
        )
        adapter = SwissDevJobsAdapter(relevant_only=False)
        await adapter.search("python")
        await adapter.search("data engineer")  # different keywords, same cache
        assert respx.calls.call_count == 1

    @respx.mock
    async def test_handles_http_error_gracefully(self):
        respx.get(_API_URL).mock(return_value=httpx.Response(503))
        adapter = SwissDevJobsAdapter()
        results = await adapter.search("python")
        assert results == []

    @respx.mock
    async def test_handles_invalid_json(self):
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, text="not json")
        )
        adapter = SwissDevJobsAdapter()
        results = await adapter.search("python")
        assert results == []

    @respx.mock
    async def test_respects_limit(self):
        items = [_sdj_item(id=str(i), job_url=f"Job-{i}", name=f"Dev {i}") for i in range(20)]
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=items)
        )
        adapter = SwissDevJobsAdapter(relevant_only=False)
        results = await adapter.search("python", limit=5)
        assert len(results) == 5

    @respx.mock
    async def test_skips_malformed_items(self):
        """Items missing name or jobUrl are silently skipped."""
        respx.get(_API_URL).mock(
            return_value=httpx.Response(200, json=[
                _sdj_item(id="good"),
                {"_id": "bad"},  # missing name and jobUrl
                _sdj_item(id="also-good", job_url="Another-Job"),
            ])
        )
        adapter = SwissDevJobsAdapter(relevant_only=False)
        results = await adapter.search("python")
        assert len(results) == 2