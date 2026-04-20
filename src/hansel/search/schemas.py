"""Pydantic schemas for job search."""

from pydantic import BaseModel, Field


class JobQuery(BaseModel):
    """A single job search query."""
    keywords: str = Field(
        description="Short search keywords (2-5 words). E.g., 'Junior Data Engineer'"
    )
    location: str = Field(
        description="City, country, or 'remote'. E.g., 'Zurich', 'Switzerland', 'remote'"
    )
    rationale: str = Field(
        description="One short sentence explaining why this query fits the candidate"
    )


class SearchStrategy(BaseModel):
    """A set of diverse queries to maximize coverage of job listings."""
    queries: list[JobQuery] = Field(
        description="3-5 diverse queries. Must vary in role, location, or scope."
    )