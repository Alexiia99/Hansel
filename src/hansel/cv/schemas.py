"""Pydantic schemas for CV data."""

from enum import Enum

from pydantic import BaseModel, Field


class Seniority(str, Enum):
    """Career seniority level, provided by the user."""
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    
    @property
    def label(self) -> str:
        """Human-readable label used in prompts and search queries."""
        return {
            Seniority.JUNIOR: "Junior",
            Seniority.MID: "Mid-level",
            Seniority.SENIOR: "Senior",
            Seniority.LEAD: "Lead/Staff",
        }[self]


class Experience(BaseModel):
    """A single professional experience."""
    company: str = Field(description="Company name")
    role: str = Field(description="Job title or role")
    start_date: str = Field(description="Format: YYYY-MM or YYYY")
    end_date: str | None = Field(
        default=None,
        description="Format: YYYY-MM, YYYY, or 'current' if ongoing",
    )
    description: str = Field(description="Summary of responsibilities and achievements")


class Education(BaseModel):
    """Academic formation."""
    institution: str
    degree: str = Field(description="Degree name")
    field: str | None = Field(default=None, description="Field of study if distinct from degree")
    year: str | None = Field(default=None, description="Year of completion or range")


class CVProfileSemantic(BaseModel):
    """CV fields the LLM extracts (semantic content only)."""
    full_name: str
    location: str | None = None
    summary: str = Field(description="2-3 sentence professional summary")
    skills: list[str] = Field(
        description=(
            "Individual technical skills, normalized. Extract EVERY technology, "
            "framework, language, tool, database, and platform mentioned anywhere "
            "in the CV (including experience descriptions)."
        )
    )
    languages: list[str] = Field(
        description="Spoken languages with level (e.g. 'English (B2)', 'German (Basic)')"
    )
    experiences: list[Experience]
    education: list[Education]
    target_roles: list[str] = Field(
        description=(
            "3-5 specific job titles this person should apply to, matching the requested "
            "seniority level. Base them on actual skills and experience."
        )
    )


class CVProfile(BaseModel):
    """Complete CV profile: regex-extracted contact + LLM-extracted semantic."""
    # Deterministic (regex)
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None
    # Semantic (LLM)
    full_name: str
    location: str | None = None
    summary: str
    skills: list[str]
    languages: list[str]
    experiences: list[Experience]
    education: list[Education]
    target_roles: list[str]
    # User-provided context (not extracted)
    seniority: Seniority | None = None