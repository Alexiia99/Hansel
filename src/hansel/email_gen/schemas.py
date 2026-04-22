"""Schemas for email generation."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EmailLanguage(str, Enum):
    """Supported languages for email generation."""
    ENGLISH = "english"
    GERMAN = "german"
    FRENCH = "french"
    SPANISH = "spanish"
    
    @property
    def label(self) -> str:
        return {
            EmailLanguage.ENGLISH: "English",
            EmailLanguage.GERMAN: "German",
            EmailLanguage.FRENCH: "French",
            EmailLanguage.SPANISH: "Spanish",
        }[self]


class EmailDraft(BaseModel):
    """First-pass email draft."""
    subject: str = Field(description="Concise email subject line (max 10 words)")
    body: str = Field(description="Email body, 150-200 words, professional but warm")


class EmailCritique(BaseModel):
    """Self-critique output: weaknesses identified + improved version."""
    weaknesses: list[str] = Field(
        min_length=1, max_length=5,
        description="Specific, concrete weaknesses of the draft"
    )
    improved_subject: str = Field(
        description="Refined subject line addressing subject-related issues"
    )
    improved_body: str = Field(
        description="Rewritten body addressing the identified weaknesses"
    )


class GeneratedEmail(BaseModel):
    """Final output: email with metadata for the user."""
    subject: str
    body: str
    language: EmailLanguage
    word_count: int
    # Transparency: show the user what we improved
    draft_subject: str
    draft_body: str
    critique_points: list[str]


class EmailGenerationSkipped(BaseModel):
    """Returned when we decline to generate (e.g., match too low)."""
    reason: str
    match_score: float
    threshold: float


class FactCheckResult(BaseModel):
    """LLM-based fact-checking output for generated emails."""
    is_valid: bool = Field(
        description="True if every factual claim in the email is supported by the source material."
    )
    fabrications: list[str] = Field(
        default_factory=list,
        description=(
            "List of specific claims in the email that are NOT supported by "
            "the candidate's profile or the job description."
        ),
    )
