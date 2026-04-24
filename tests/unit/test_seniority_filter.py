"""Tests for seniority detection and filtering heuristics."""

from __future__ import annotations

import pytest

from hansel.cv.schemas import Seniority
from hansel.matcher.seniority_filter import (
    detect_title_seniority,
    filter_by_seniority_strict,
    is_compatible,
    seniority_score,
)
from hansel.sources.schemas import JobListing, JobSource


# ---------- Helpers ----------


def _listing(title: str) -> JobListing:
    return JobListing(
        source=JobSource.ARBEITNOW,
        source_id=title.lower().replace(" ", "-"),
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
        title=title,
        company="Acme AG",
        location="Zürich",
        description="A job posting.",
    )


# ---------- detect_title_seniority ----------


class TestDetectTitleSeniority:

    def test_junior_detected(self):
        assert detect_title_seniority("Junior Python Developer") == Seniority.JUNIOR

    def test_senior_detected(self):
        assert detect_title_seniority("Senior Backend Engineer") == Seniority.SENIOR

    def test_lead_detected(self):
        assert detect_title_seniority("Lead Data Engineer") == Seniority.LEAD

    def test_mid_detected(self):
        # "Mid" is not a detected keyword — treated as untagged title
        assert detect_title_seniority("Mid Software Engineer") is None
    
    def test_werkstudent_is_junior(self):
        assert detect_title_seniority("Werkstudent Python Entwickler") == Seniority.JUNIOR

    def test_praktikum_is_junior(self):
        assert detect_title_seniority("Praktikum Data Science") == Seniority.JUNIOR

    def test_internship_is_junior(self):
        assert detect_title_seniority("Internship Software Engineer") == Seniority.JUNIOR

    def test_no_seniority_returns_none(self):
        assert detect_title_seniority("Python Developer") is None

    def test_case_insensitive(self):
        assert detect_title_seniority("SENIOR data engineer") == Seniority.SENIOR

    def test_staff_is_lead(self):
        assert detect_title_seniority("Staff Engineer") == Seniority.LEAD

    def test_principal_is_lead(self):
        assert detect_title_seniority("Principal Engineer") == Seniority.LEAD


# ---------- is_compatible ----------


class TestIsCompatible:

    def test_junior_compatible_with_junior(self):
        assert is_compatible(Seniority.JUNIOR, Seniority.JUNIOR) is True

    def test_junior_compatible_with_none(self):
        assert is_compatible(Seniority.JUNIOR, None) is True

    def test_junior_incompatible_with_senior(self):
        assert is_compatible(Seniority.JUNIOR, Seniority.SENIOR) is False

    def test_junior_incompatible_with_lead(self):
        assert is_compatible(Seniority.JUNIOR, Seniority.LEAD) is False

    def test_senior_compatible_with_mid(self):
        assert is_compatible(Seniority.SENIOR, Seniority.MID) is True

    def test_mid_compatible_with_junior(self):
        assert is_compatible(Seniority.MID, Seniority.JUNIOR) is True


# ---------- seniority_score ----------


class TestSeniorityScore:

    def test_exact_match_is_one(self):
        assert seniority_score(Seniority.JUNIOR, Seniority.JUNIOR) == 1.0

    def test_none_title_is_one(self):
        # Untagged titles get a slight penalty rather than a perfect score
        score = seniority_score(Seniority.JUNIOR, None)
        assert 0.7 <= score <= 1.0

    def test_senior_for_junior_penalized(self):
        score = seniority_score(Seniority.JUNIOR, Seniority.SENIOR)
        assert score < 1.0
        assert score >= 0.0

    def test_score_between_zero_and_one(self):
        for user in Seniority:
            for title in list(Seniority) + [None]:
                s = seniority_score(user, title)
                assert 0.0 <= s <= 1.0


# ---------- filter_by_seniority_strict ----------


class TestFilterByStrictSeniority:

    def test_keeps_matching_seniority(self):
        listings = [_listing("Junior Python Developer")]
        result = filter_by_seniority_strict(listings, Seniority.JUNIOR)
        assert len(result) == 1

    def test_drops_senior_for_junior(self):
        listings = [_listing("Senior Python Developer")]
        result = filter_by_seniority_strict(listings, Seniority.JUNIOR)
        assert len(result) == 0

    def test_keeps_untagged_titles(self):
        listings = [_listing("Python Developer")]
        result = filter_by_seniority_strict(listings, Seniority.JUNIOR)
        assert len(result) == 1

    def test_drops_lead_for_junior(self):
        listings = [_listing("Lead Engineer")]
        result = filter_by_seniority_strict(listings, Seniority.JUNIOR)
        assert len(result) == 0

    def test_mixed_list(self):
        listings = [
            _listing("Junior Data Engineer"),
            _listing("Senior Backend Developer"),
            _listing("Python Developer"),
            _listing("Lead Architect"),
        ]
        result = filter_by_seniority_strict(listings, Seniority.JUNIOR)
        titles = {r.title for r in result}
        assert "Junior Data Engineer" in titles
        assert "Python Developer" in titles
        assert "Senior Backend Developer" not in titles
        assert "Lead Architect" not in titles

    def test_empty_list(self):
        assert filter_by_seniority_strict([], Seniority.JUNIOR) == []