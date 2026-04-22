"""Job sources module: adapters for various job boards + orchestrator."""

from hansel.sources.adzuna import AdzunaAdapter
from hansel.sources.arbeitnow import ArbeitnowAdapter
from hansel.sources.swissdevjobs import SwissDevJobsAdapter
from hansel.sources.base import JobSourceAdapter
from hansel.sources.orchestrator import JobSearchOrchestrator
from hansel.sources.schemas import JobListing, JobSource

__all__ = [
    "AdzunaAdapter",
    "ArbeitnowAdapter",
    "SwissDevJobsAdapter",
    "JobListing",
    "JobSearchOrchestrator",
    "JobSource",
    "JobSourceAdapter",
]