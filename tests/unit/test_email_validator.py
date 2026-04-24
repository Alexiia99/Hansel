"""Tests for the email hallucination validator."""

from __future__ import annotations

import pytest

from hansel.email_gen.validator import ValidationResult, validate_email


# ---------- Helpers ----------


def _valid_body() -> str:
    return (
        "I am excited to apply for the Python Developer position at your company. "
        "During my time at Empresa Real SL, I built data pipelines and backend systems "
        "using Python and FastAPI. I believe my background in machine learning and "
        "backend development makes me a strong candidate for this role."
    )


# ---------- Subject validation ----------


class TestSubjectValidation:

    def test_valid_subject_passes(self):
        result = validate_email(_valid_body(), subject="Application for Python Developer Role")
        assert result.is_valid is True

    def test_empty_subject_fails(self):
        result = validate_email(_valid_body(), subject="")
        assert result.is_valid is False
        assert any("empty" in i.lower() for i in result.issues)

    def test_whitespace_only_subject_fails(self):
        result = validate_email(_valid_body(), subject="   ")
        assert result.is_valid is False

    def test_single_word_subject_fails(self):
        result = validate_email(_valid_body(), subject="Application")
        assert result.is_valid is False

    def test_none_subject_not_checked(self):
        result = validate_email(_valid_body(), subject=None)
        assert result.is_valid is True


# ---------- Body validation ----------


class TestBodyValidation:

    def test_clean_body_passes(self):
        result = validate_email(_valid_body())
        assert result.is_valid is True
        assert result.issues == []

    def test_empty_body_fails(self):
        result = validate_email("")
        assert result.is_valid is False
        assert any("empty" in i.lower() for i in result.issues)

    def test_whitespace_body_fails(self):
        result = validate_email("   \n  ")
        assert result.is_valid is False


# ---------- Placeholder company names ----------


class TestPlaceholderCompanyNames:

    def test_xyz_corp_detected(self):
        body = "During my work at XYZ Corp I built great things."
        result = validate_email(body)
        assert result.is_valid is False
        assert any("XYZ" in i for i in result.issues)

    def test_abc_detected(self):
        body = "I worked at ABC on backend systems."
        result = validate_email(body)
        assert result.is_valid is False

    def test_acme_detected(self):
        body = "At Acme I developed Python services."
        result = validate_email(body)
        assert result.is_valid is False

    def test_real_company_allowed(self):
        body = "During my time at Acme I built pipelines."
        result = validate_email(body, candidate_companies=["Acme"])
        assert result.is_valid is True

    def test_case_insensitive_detection(self):
        body = "I worked at xyz corp last year."
        result = validate_email(body)
        assert result.is_valid is False


# ---------- Placeholder project names ----------


class TestPlaceholderProjectNames:

    def test_project_alpha_detected(self):
        body = "I led Project Alpha at my previous company."
        result = validate_email(body)
        assert result.is_valid is False

    def test_product_x_detected(self):
        body = "I worked on Product X for two years."
        result = validate_email(body)
        assert result.is_valid is False

    def test_team_beta_detected(self):
        body = "As part of Team Beta I delivered key features."
        result = validate_email(body)
        assert result.is_valid is False


# ---------- Template placeholders ----------


class TestTemplatePlaceholders:

    def test_bracket_company_detected(self):
        body = "I am applying to [company name] for the role."
        result = validate_email(body)
        assert result.is_valid is False

    def test_bracket_skill_detected(self):
        body = "I have experience with [technology] and related tools."
        result = validate_email(body)
        assert result.is_valid is False

    def test_lorem_ipsum_detected(self):
        body = "Lorem ipsum dolor sit amet."
        result = validate_email(body)
        assert result.is_valid is False

    def test_todo_detected(self):
        body = "TODO: add more details here."
        result = validate_email(body)
        assert result.is_valid is False


# ---------- Suspicious metrics ----------


class TestSuspiciousMetrics:

    def test_percentage_claim_flagged(self):
        body = "I improved system performance by 40% at my last job."
        result = validate_email(body)
        assert result.is_valid is False
        assert any("quantified" in i.lower() for i in result.issues)

    def test_reduced_by_percentage_flagged(self):
        body = "I reduced latency by 30% using caching."
        result = validate_email(body)
        assert result.is_valid is False

    def test_no_percentage_passes(self):
        body = "I significantly improved system performance at my last job."
        result = validate_email(body)
        assert result.is_valid is True


# ---------- Multiple issues ----------


class TestMultipleIssues:

    def test_collects_all_issues(self):
        body = "At XYZ Corp I worked on Project Alpha and improved speed by 50%."
        result = validate_email(body, subject="")
        assert result.is_valid is False
        assert len(result.issues) >= 3  # subject + company + project + metric

    def test_returns_validation_result_type(self):
        result = validate_email(_valid_body())
        assert isinstance(result, ValidationResult)
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.issues, list)