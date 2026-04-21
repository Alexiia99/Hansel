"""Matcher module: scores and ranks job listings against a CV profile."""

from hansel.matcher.embeddings import EmbeddingScorer, cv_to_text, listing_to_text
from hansel.matcher.schemas import MatchScore, ScoredListing, SeniorityMode
from hansel.matcher.seniority_filter import (
    detect_title_seniority,
    filter_by_seniority_strict,
    is_compatible,
    seniority_score,
)

__all__ = [
    "EmbeddingScorer",
    "MatchScore",
    "ScoredListing",
    "SeniorityMode",
    "cv_to_text",
    "detect_title_seniority",
    "filter_by_seniority_strict",
    "is_compatible",
    "listing_to_text",
    "seniority_score",
]