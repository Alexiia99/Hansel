"""Job sources module: adapters for various job boards."""

from hansel.sources.arbeitnow import ArbeitnowAdapter
from hansel.sources.base import JobSourceAdapter
from hansel.sources.schemas import JobListing, JobSource

__all__ = [
    "ArbeitnowAdapter",
    "JobListing",
    "JobSource",
    "JobSourceAdapter",
]