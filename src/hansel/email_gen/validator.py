"""Post-generation validator: detects common hallucination patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validating an email for hallucination patterns."""
    is_valid: bool
    issues: list[str]


# Patterns that strongly signal the LLM fabricated a placeholder
_HALLUCINATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"\b(?:XYZ|ABC|Acme|Company\s+X|Corp\s+X|Foo(?:bar)?)\b", re.IGNORECASE),
        "Contains placeholder company name (XYZ, ABC, Acme, Foobar, etc.)",
    ),
    (
        re.compile(r"\b(?:Project|Product|Team)\s+(?:Alpha|Beta|Gamma|X|Y|Z|One|Two)\b", re.IGNORECASE),
        "Contains placeholder project/product name (Project Alpha, Product X, etc.)",
    ),
    (
        re.compile(r"\bJob\s+Posting\s+\d{4}[-/]\d+\b", re.IGNORECASE),
        "References a fabricated job posting ID",
    ),
    (
        re.compile(r"\b(?:Lorem|ipsum|placeholder|TODO|TBD|XXXX?)\b", re.IGNORECASE),
        "Contains placeholder text (Lorem ipsum, TODO, TBD, etc.)",
    ),
    (
        re.compile(r"\[.*?(?:company|role|skill|technology|project|name).*?\]", re.IGNORECASE),
        "Contains unfilled template placeholder in brackets",
    ),
]


def validate_email(
    body: str,
    candidate_companies: list[str] | None = None,
    subject: str | None = None,
) -> ValidationResult:
    """Check for common LLM hallucination patterns + basic integrity issues.
    
    Args:
        body: The email body to validate.
        candidate_companies: Companies from the candidate's actual experience
            (used to allow references to real past employers).
        subject: Optional subject line. If provided, also validated.
    
    Returns:
        ValidationResult with is_valid=False if hallucinations or empty 
        required fields are detected.
    """
    issues: list[str] = []
    
    # --- Subject integrity checks ---
    if subject is not None:
        subject_stripped = subject.strip()
        if not subject_stripped:
            issues.append("Subject line is empty")
        elif len(subject_stripped.split()) < 2:
            issues.append(
                f"Subject line is too short to be meaningful: '{subject_stripped}'"
            )
    
    # --- Body integrity checks ---
    if not body or not body.strip():
        issues.append("Body is empty")
        return ValidationResult(is_valid=False, issues=issues)
    
    # --- Hallucination pattern checks (body) ---
    for pattern, description in _HALLUCINATION_PATTERNS:
        if match := pattern.search(body):
            # Skip if the match is actually one of the candidate's real companies
            if candidate_companies:
                matched_text = match.group(0).lower()
                if any(matched_text in real.lower() for real in candidate_companies):
                    continue
            issues.append(f"{description}: '{match.group(0)}'")
    
    # --- Suspicious metrics check ---
    # Claims like "improved X by 30%" often signal fabrication in email context.
    suspicious_metrics = re.search(
        r"(?:improved|increased|reduced|optimized|saved|boosted)\s+[\w\s]{1,30}?\s+by\s+\d+%",
        body,
        re.IGNORECASE,
    )
    if suspicious_metrics:
        issues.append(
            f"Contains specific quantified claim that may be fabricated: "
            f"'{suspicious_metrics.group(0)}'. Verify against candidate's profile."
        )
    
    return ValidationResult(is_valid=not issues, issues=issues)