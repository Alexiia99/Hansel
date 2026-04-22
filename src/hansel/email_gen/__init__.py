"""Email generation module: personalized application emails with self-critique."""

from hansel.email_gen.generator import EmailGenerator, FactChecker
from hansel.email_gen.schemas import (
    EmailCritique,
    EmailDraft,
    EmailGenerationSkipped,
    EmailLanguage,
    FactCheckResult,
    GeneratedEmail,
)
from hansel.email_gen.validator import ValidationResult, validate_email

__all__ = [
    "EmailCritique",
    "EmailDraft",
    "EmailGenerationSkipped",
    "EmailGenerator",
    "EmailLanguage",
    "FactCheckResult",
    "FactChecker",
    "GeneratedEmail",
    "ValidationResult",
    "validate_email",
]