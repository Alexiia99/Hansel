# src/hansel/sources/swissdevjobs.py

from __future__ import annotations

import logging

import httpx

from hansel.sources.base import JobSourceAdapter
from hansel.sources.schemas import JobListing, JobSource

logger = logging.getLogger(__name__)

_API_URL = "https://swissdevjobs.ch/api/jobsLight"
_JOB_BASE_URL = "https://swissdevjobs.ch/jobs"

# Categories relevant to a junior ML/Data/Backend profile.
# WHY filter by category: the API returns all 236 listings regardless of query.
# We filter client-side by techCategory/metaCategory to reduce noise before
# passing to the matcher.
_RELEVANT_CATEGORIES = {
    "python", "machine-learning", "mlaidata", "data", "devops", "javascript",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://swissdevjobs.ch/jobs/Python/all",
}


def _is_relevant(job: dict) -> bool:
    """Keep only listings in relevant tech categories."""
    tech = (job.get("techCategory") or "").lower()
    meta = (job.get("metaCategory") or "").lower()
    return tech in _RELEVANT_CATEGORIES or meta in _RELEVANT_CATEGORIES


def _parse_job(job: dict) -> JobListing | None:
    """Convert a raw API job dict to a JobListing."""
    name = job.get("name", "").strip()
    company = job.get("company", "").strip()
    job_url = job.get("jobUrl", "")

    if not name or not job_url:
        return None

    url = f"{_JOB_BASE_URL}/{job_url}"
    source_id = job.get("_id") or job_url

    location = job.get("actualCity") or job.get("cityCategory") or "Switzerland"
    workplace = job.get("workplace", "")
    if workplace == "remote":
        location = "Remote (Switzerland)"

    tags = list(job.get("technologies") or job.get("filterTags") or [])

    salary_from = job.get("annualSalaryFrom")
    salary_to = job.get("annualSalaryTo")

    exp_level = job.get("expLevel", "")
    description_parts = []
    if salary_from and salary_to:
        description_parts.append(f"Salary: CHF {salary_from:,} - {salary_to:,}")
    if exp_level:
        description_parts.append(f"Level: {exp_level}")
    if workplace:
        description_parts.append(f"Workplace: {workplace}")

    return JobListing(
        source=JobSource.SWISSDEVJOBS,
        source_id=source_id,
        url=url,
        title=name,
        company=company or "Unknown",
        location=location,
        description=" | ".join(description_parts) or f"{name} at {company}",
        salary_min=float(salary_from) if salary_from else None,
        salary_max=float(salary_to) if salary_to else None,
        salary_currency="CHF" if salary_from else None,
        is_remote=workplace == "remote",
        tags=tags[:10],
    )

class SwissDevJobsAdapter(JobSourceAdapter):
    """Fetches tech jobs from swissdevjobs.ch via their internal JSON API.

    WHY swissdevjobs over jobs.ch: tech-specialist board with transparent
    salary ranges, covers German-speaking Switzerland, and exposes all
    listings via a single JSON endpoint — no scraping, no JS rendering needed.
    See ADR 009.
    """

    def __init__(self, relevant_only: bool = True) -> None:
        # WHY relevant_only=True default: the API returns 200+ listings across
        # all tech stacks. Filtering to ML/Data/Python categories before the
        # matcher reduces noise and speeds up embedding scoring.
        self._relevant_only = relevant_only
        self._cache: list[JobListing] | None = None

    @property
    def source_name(self) -> str:
        return "swissdevjobs"

    async def search(
        self,
        keywords: str,
        location: str | None = None,
        limit: int = 50,
    ) -> list[JobListing]:
        """Return listings from swissdevjobs.ch.

        WHY we ignore keywords/location: the API returns all listings at once.
        We fetch once, cache the result, and return the same list for every
        search() call the orchestrator makes — avoiding redundant HTTP requests.
        The matcher handles relevance filtering downstream.
        """
        if self._cache is None:
            self._cache = await self._fetch_all()

        return self._cache[:limit]

    async def _fetch_all(self) -> list[JobListing]:
        """Fetch and parse all listings from the API."""
        try:
            async with httpx.AsyncClient(
                headers=_HEADERS,
                follow_redirects=True,
                timeout=15.0,
            ) as client:
                response = await client.get(_API_URL)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as e:
            logger.warning("SwissDevJobs: HTTP error fetching listings: %s", e)
            return []
        except Exception as e:
            logger.warning("SwissDevJobs: unexpected error: %s", e)
            return []

        if self._relevant_only:
            data = [j for j in data if _is_relevant(j)]

        listings = []
        for job in data:
            parsed = _parse_job(job)
            if parsed:
                listings.append(parsed)

        logger.info("SwissDevJobs: %d listings fetched (%d after filter)",
                    len(data), len(listings))
        return listings