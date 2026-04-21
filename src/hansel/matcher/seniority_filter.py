"""Heuristic seniority detection and filtering from job titles."""

from __future__ import annotations

import re

from hansel.cv.schemas import Seniority
from hansel.sources.schemas import JobListing


# Keywords per seniority level, ordered: we match the MOST senior match wins.
# (A "Senior Junior Developer" would be weird but the SENIOR wins.)
_TITLE_SENIORITY_PATTERNS: list[tuple[Seniority, re.Pattern]] = [
    (Seniority.LEAD, re.compile(r"\b(lead|principal|staff|head of|chief|architect)\b", re.IGNORECASE)),
    (Seniority.SENIOR, re.compile(r"\b(senior|sr\.?|sr\s|experienced|expert)\b", re.IGNORECASE)),
    (Seniority.MID, re.compile(r"\b(mid[-\s]?level|intermediate|middle)\b", re.IGNORECASE)),
    (Seniority.JUNIOR, re.compile(r"\b(junior|jr\.?|jr\s|entry[-\s]?level|graduate|trainee|intern|internship|werkstudent|praktikum)\b", re.IGNORECASE)),
]


def detect_title_seniority(title: str) -> Seniority | None:
    """Infer seniority level from a job title.
    
    Returns None when the title doesn't signal a specific seniority
    (very common — titles often omit it and rely on the description).
    """
    for seniority, pattern in _TITLE_SENIORITY_PATTERNS:
        if pattern.search(title):
            return seniority
    return None


# Compatibility matrix: what user seniority can apply to what detected title seniority.
# Rows = user seniority (what they are), Columns = detected (what the job wants).
# Value = True if compatible.
_COMPATIBILITY: dict[Seniority, set[Seniority]] = {
    # Junior candidates: mostly junior roles. Mid is a stretch, Senior is no.
    Seniority.JUNIOR: {Seniority.JUNIOR, Seniority.MID},
    # Mid candidates: junior to senior range.
    Seniority.MID: {Seniority.JUNIOR, Seniority.MID, Seniority.SENIOR},
    # Senior candidates: mid to lead.
    Seniority.SENIOR: {Seniority.MID, Seniority.SENIOR, Seniority.LEAD},
    # Lead candidates: senior to lead.
    Seniority.LEAD: {Seniority.SENIOR, Seniority.LEAD},
}


def is_compatible(user: Seniority, title_level: Seniority | None) -> bool:
    """Check if a user at `user` seniority can apply to a role at `title_level`.
    
    If title_level is None (title doesn't signal seniority), always compatible —
    the LLM will decide later from the description.
    """
    if title_level is None:
        return True
    return title_level in _COMPATIBILITY[user]


def seniority_score(user: Seniority, title_level: Seniority | None) -> float:
    """Continuous version of is_compatible — used for soft penalization.
    
    1.0 = exact match, 0.7 = one step off, 0.3 = two steps off, 0.0 = incompatible.
    Unknown title_level returns 0.8 (slight benefit of the doubt).
    """
    if title_level is None:
        return 0.8
    
    order = [Seniority.JUNIOR, Seniority.MID, Seniority.SENIOR, Seniority.LEAD]
    try:
        gap = abs(order.index(user) - order.index(title_level))
    except ValueError:
        return 0.5
    
    return {0: 1.0, 1: 0.7, 2: 0.3, 3: 0.0}.get(gap, 0.0)


def filter_by_seniority_strict(
    listings: list[JobListing], user_seniority: Seniority
) -> list[JobListing]:
    """Drop listings whose title signals incompatible seniority."""
    return [
        l for l in listings
        if is_compatible(user_seniority, detect_title_seniority(l.title))
    ]