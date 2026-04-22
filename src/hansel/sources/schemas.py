"""Unified schema for job postings across all sources."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl


class JobSource(str, Enum):
    """Identifier of the source that provided the listing."""
    ADZUNA = "adzuna"
    JOOBLE = "jooble"
    ARBEITNOW = "arbeitnow"
    JOBS_CH = "jobs_ch"
    SWISSDEVJOBS = "swissdevjobs"



class JobListing(BaseModel):
    """A normalized job posting.
    
    Every adapter must convert its raw API response into this shape.
    Downstream components (agent, matcher, email generator) only ever 
    see JobListing, never raw API data.
    """
    
    # --- Core identification ---
    source: JobSource = Field(description="Which provider this came from")
    source_id: str = Field(description="Provider's internal ID for deduplication")
    url: HttpUrl = Field(description="Direct link to the posting")
    
    # --- Basic info ---
    title: str = Field(description="Job title as posted")
    company: str = Field(description="Hiring company")
    location: str | None = Field(default=None, description="Location string or 'remote'")
    
    # --- Content ---
    description: str = Field(description="Full job description (may be long)")
    
    # --- Optional structured data ---
    salary_min: float | None = Field(default=None, description="Annual salary, normalized if possible")
    salary_max: float | None = None
    salary_currency: str | None = Field(default=None, description="ISO 4217 code (EUR, CHF, USD)")
    is_remote: bool | None = None
    posted_at: datetime | None = Field(default=None, description="When posted, if provided")
    
    # --- Tags / raw ---
    tags: list[str] = Field(default_factory=list, description="Free-form tags from source")
    
    class Config:
        # Allow datetime/URL coercion from raw strings
        str_strip_whitespace = True