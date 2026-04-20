"""Deterministic extraction of contact fields from CV text."""

from __future__ import annotations

import re
from typing import NamedTuple


class ContactInfo(NamedTuple):
    """Contact fields extracted via regex."""
    email: str | None
    phone: str | None
    linkedin: str | None
    github: str | None


_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3}[\s-]?\d{3}[\s-]?\d{3,4}"
)

_LINKEDIN_RE = re.compile(
    r"linkedin\.com/in/[\w-]+|LinkedIn:\s*([\w\s-]+?)(?=\s*\||\s*$|\s*\n)",
    re.IGNORECASE,
)

_GITHUB_RE = re.compile(r"github\.com/[\w-]+", re.IGNORECASE)


def _clean_linkedin(raw: str | None) -> str | None:
    """Strip 'LinkedIn:' prefix and whitespace."""
    if not raw:
        return None
    cleaned = re.sub(r"^LinkedIn:\s*", "", raw, flags=re.IGNORECASE).strip()
    return cleaned or None


def extract_contact(text: str) -> ContactInfo:
    """Extract email, phone, LinkedIn, GitHub from CV text.
    
    Uses regex — fast, deterministic, and avoids LLM privacy filtering on PII.
    """
    email_m = _EMAIL_RE.search(text)
    phone_m = _PHONE_RE.search(text)
    linkedin_m = _LINKEDIN_RE.search(text)
    github_m = _GITHUB_RE.search(text)
    
    return ContactInfo(
        email=email_m.group(0) if email_m else None,
        phone=phone_m.group(0).strip() if phone_m else None,
        linkedin=_clean_linkedin(linkedin_m.group(0)) if linkedin_m else None,
        github=github_m.group(0) if github_m else None,
    )