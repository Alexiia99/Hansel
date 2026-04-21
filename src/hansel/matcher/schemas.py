"""Schemas for the matcher module."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from hansel.sources.schemas import JobListing


class SeniorityMode(str, Enum):
    """How to handle seniority mismatches.
    
    - STRICT: drop listings whose title signals incompatible seniority.
    - INCLUSIVE: keep all, but penalize the score.
    """
    STRICT = "strict"
    INCLUSIVE = "inclusive"


class MatchScore(BaseModel):
    """LLM-produced evaluation of one listing against a CV."""
    overall: float = Field(
        ge=0.0, le=1.0,
        description="Overall fit score, 0.0 (no fit) to 1.0 (perfect fit)."
    )
    skills_match: float = Field(
        ge=0.0, le=1.0,
        description="How well the candidate's skills match the requirements."
    )
    seniority_fit: float = Field(
        ge=0.0, le=1.0,
        description="How well the seniority level matches. 1.0 = exact match, 0.0 = very off."
    )
    rationale: str = Field(
        description="2-3 sentence explanation of strengths and gaps."
    )


class ScoredListing(BaseModel):
    """A JobListing enriched with matching scores."""
    listing: JobListing
    embedding_score: float = Field(
        description="Cosine similarity between CV and listing (0-1)."
    )
    llm_score: MatchScore | None = Field(
        default=None,
        description="Detailed LLM evaluation. None if not reranked."
    )
    final_score: float = Field(
        description="Combined score used for ranking (0-1)."
    )
    
    @property
    def has_rationale(self) -> bool:
        return self.llm_score is not None