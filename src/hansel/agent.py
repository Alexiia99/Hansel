"""Hansel: autonomous job search agent.

Orchestrates the full pipeline: CV extraction → query generation → multi-source
search → ranking → email generation.

Designed for two usage styles:

    # One-shot: full pipeline with one call.
    agent = HanselAgent()
    result = await agent.find_jobs(
        cv_path="cv.pdf",
        seniority=Seniority.JUNIOR,
        preferred_location="Switzerland",
    )

    # Step-by-step: for notebooks or custom workflows.
    profile = await agent.extract_cv("cv.pdf", Seniority.JUNIOR)
    listings = await agent.search(profile, "Switzerland")
    ranked = await agent.rank(profile, listings)
    emails = await agent.generate_emails(profile, ranked[:5])
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import asyncio

from hansel.cv import CVExtractor, CVProfile, Seniority
from hansel.email_gen import (
    EmailGenerationSkipped,
    EmailGenerator,
    EmailLanguage,
    GeneratedEmail,
)
from hansel.matcher import JobMatcher, ScoredListing, SeniorityMode
from hansel.search import QueryGenerator
from hansel.sources import (
    AdzunaAdapter,
    ArbeitnowAdapter,
    SwissDevJobsAdapter,
    JobListing,
    JobSearchOrchestrator,
    JobSourceAdapter,
)

logger = logging.getLogger(__name__)


# ---------- Progress callback protocol ----------


ProgressCallback = Callable[[str, str], None]
"""Signature: callback(step_name, detail) -> None.

Called at each pipeline stage. Examples:
    callback("extract_cv", "Parsing PDF and extracting structured data")
    callback("search", "Found 27 unique listings from 2 sources")
    callback("rank", "Ranked 19 listings after seniority filter")
"""


def _noop_progress(step: str, detail: str) -> None:
    """Default progress handler: does nothing."""
    pass


def _print_progress(step: str, detail: str) -> None:
    """Convenience progress handler: prints to stdout."""
    step_label = step.replace("_", " ").title()
    print(f"[{step_label}] {detail}")


# ---------- Result dataclass ----------


@dataclass
class HanselResult:
    """Complete output of the agent pipeline.
    
    Fields are populated as the pipeline progresses; on errors with
    strict=False, partial results are still returned.
    """
    profile: CVProfile | None = None
    raw_listings: list[JobListing] = field(default_factory=list)
    ranked_listings: list[ScoredListing] = field(default_factory=list)
    emails: list[GeneratedEmail | EmailGenerationSkipped] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """True if the pipeline completed with at least one ranked listing."""
        return bool(self.ranked_listings) and not self.errors
    
    @property
    def top_listing(self) -> ScoredListing | None:
        """Convenience accessor for the highest-ranked listing."""
        return self.ranked_listings[0] if self.ranked_listings else None


# ---------- Custom exception ----------


class HanselError(RuntimeError):
    """Raised when the agent fails in strict mode."""


# ---------- Agent ----------


class HanselAgent:
    """Autonomous job search agent.
    
    High-level interface to the full Hansel pipeline. Components are
    lazily constructed on first use; custom components can be injected
    via the constructor for testing or customization.
    """
    
    def __init__(
        self,
        *,
        cv_extractor: CVExtractor | None = None,
        query_generator: QueryGenerator | None = None,
        adapters: Sequence[JobSourceAdapter] | None = None,
        matcher: JobMatcher | None = None,
        email_generator: EmailGenerator | None = None,
        strict: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """
        Args:
            cv_extractor: Custom CV extractor. Default: CVExtractor() with Qwen 2.5 7B.
            query_generator: Custom query generator. Default: QueryGenerator().
            adapters: Job source adapters. Default: [Adzuna(country='ch'), Arbeitnow()].
            matcher: Custom matcher. Default: JobMatcher() with retrieve+rerank.
            email_generator: Custom email generator. Default: EmailGenerator().
            strict: If True, pipeline errors raise HanselError. If False,
                errors are recorded and partial results returned.
            progress_callback: Called at each stage. Default: no-op (silent).
        """
        self._cv_extractor = cv_extractor or CVExtractor()
        self._query_generator = query_generator or QueryGenerator()
        self._adapters = list(adapters) if adapters else self._default_adapters()
        self._matcher = matcher or JobMatcher()
        self._email_generator = email_generator or EmailGenerator()
        self._strict = strict
        self._progress = progress_callback or _noop_progress
        
        # Orchestrator is derived from adapters
        self._orchestrator = JobSearchOrchestrator(adapters=self._adapters)
    
    @staticmethod
    def _default_adapters() -> list[JobSourceAdapter]:
        """Default adapter set: Adzuna (Switzerland) + Arbeitnow + SwissDevJobs."""
        return [
            AdzunaAdapter(country="ch"),
            ArbeitnowAdapter(),
            SwissDevJobsAdapter(),
        ]
        
    # ---------- Individual pipeline steps ----------
    
    async def extract_cv(
        self, cv_path: str | Path, seniority: Seniority
    ) -> CVProfile:
        """Extract a structured profile from a CV file."""
        self._progress("extract_cv", f"Parsing {cv_path}")
        profile = self._cv_extractor.extract_from_file(cv_path, seniority=seniority)
        self._progress(
            "extract_cv",
            f"Extracted {len(profile.skills)} skills, "
            f"{len(profile.experiences)} experiences, "
            f"{len(profile.target_roles)} target roles",
        )
        return profile
    
    async def search(
        self,
        profile: CVProfile,
        preferred_location: str,
        remote_ok: bool = True,
    ) -> list[JobListing]:
        """Generate queries and search all configured sources in parallel."""
        self._progress("generate_queries", "Generating diverse search queries")
        strategy = self._query_generator.generate(
            profile=profile,
            preferred_location=preferred_location,
            remote_ok=remote_ok,
        )
        self._progress(
            "generate_queries",
            f"Generated {len(strategy.queries)} queries",
        )
        
        self._progress("search", f"Querying {len(self._adapters)} sources in parallel")
        listings = await self._orchestrator.search_all(strategy.queries)
        self._progress("search", f"Found {len(listings)} unique listings")
        
        return listings
    
    async def rank(
        self,
        profile: CVProfile,
        listings: Sequence[JobListing],
    ) -> list[ScoredListing]:
        """Rank listings against the profile (embeddings + LLM rerank)."""
        self._progress(
            "rank",
            f"Ranking {len(listings)} listings (embeddings + LLM rerank)",
        )
        ranked = await self._matcher.rank(profile, listings)
        self._progress(
            "rank",
            f"Ranked {len(ranked)} listings. Top score: "
            f"{ranked[0].final_score:.2f}" if ranked else "No listings passed filters",
        )
        return ranked
    
    async def generate_emails(
        self,
        profile: CVProfile,
        listings: Sequence[ScoredListing],
        language: EmailLanguage = EmailLanguage.ENGLISH,
    ) -> list[GeneratedEmail | EmailGenerationSkipped]:
        """Generate personalized emails for a list of scored listings.
        
        Each listing gets its own email. Returns a list where each entry
        is either a GeneratedEmail (success) or EmailGenerationSkipped
        (match too low or hallucinations detected).
        """
        self._progress(
            "generate_emails",
            f"Generating emails for top {len(listings)} listings",
        )
        
        results: list[GeneratedEmail | EmailGenerationSkipped] = []
        for i, scored in enumerate(listings, 1):
            self._progress(
                "generate_emails",
                f"Email {i}/{len(listings)}: {scored.listing.title}",
            )
            try:
                result = await self._email_generator.generate(
                    profile=profile,
                    scored_listing=scored,
                    language=language,
                )
                results.append(result)
            except Exception as e:
                logger.exception(
                    "Email generation failed for %r", scored.listing.title
                )
                if self._strict:
                    raise HanselError(
                        f"Email generation failed for {scored.listing.title!r}: {e}"
                    ) from e
                results.append(EmailGenerationSkipped(
                    reason=f"Unexpected error during generation: {e}",
                    match_score=scored.final_score,
                    threshold=0.0,
                ))
            
            # Yield control to the event loop so SSE progress events are
            # flushed to the frontend between emails. Without this, the LLM
            # calls block the loop and the queue drains only at the end.
            await asyncio.sleep(0)
        
        successful = sum(1 for r in results if isinstance(r, GeneratedEmail))
        self._progress(
            "generate_emails",
            f"Generated {successful}/{len(listings)} emails successfully",
        )
        return results

    # ---------- Full pipeline ----------
    
    async def find_jobs(
        self,
        cv_path: str | Path,
        seniority: Seniority,
        preferred_location: str,
        *,
        remote_ok: bool = True,
        generate_emails: bool = True,
        emails_top_n: int = 5,
        email_language: EmailLanguage = EmailLanguage.ENGLISH,
    ) -> HanselResult:
        """Run the full pipeline end-to-end.
        
        Args:
            cv_path: Path to CV file (.pdf, .md, or .txt).
            seniority: User's seniority level (Junior, Mid, Senior, Lead).
            preferred_location: Primary target location.
            remote_ok: Include remote-friendly queries.
            generate_emails: Generate application emails for top N listings.
            emails_top_n: How many top listings get an email. Ignored if
                generate_emails is False.
            email_language: Language for generated emails.
        
        Returns:
            HanselResult with all populated fields. On errors with strict=False,
            partial results are still returned and errors are recorded.
        """
        result = HanselResult()
        
        # Step 1: CV extraction
        try:
            result.profile = await self.extract_cv(cv_path, seniority)
        except Exception as e:
            return self._handle_error(result, "CV extraction failed", e)
        
        # Step 2: Search
        try:
            result.raw_listings = await self.search(
                result.profile, preferred_location, remote_ok
            )
        except Exception as e:
            return self._handle_error(result, "Job search failed", e)
        
        if not result.raw_listings:
            msg = (
                "No listings found. Try a broader location or different "
                "target roles in the CV."
            )
            return self._handle_error(result, msg, error=None)
        
        # Step 3: Ranking
        try:
            result.ranked_listings = await self.rank(
                result.profile, result.raw_listings
            )
        except Exception as e:
            return self._handle_error(result, "Ranking failed", e)
        
        if not result.ranked_listings:
            msg = (
                "All listings were filtered out (typically by seniority). "
                "Consider SeniorityMode.INCLUSIVE or a broader seniority."
            )
            return self._handle_error(result, msg, error=None)
        
        # Step 4: Email generation (optional)
        if generate_emails:
            top_for_emails = result.ranked_listings[:emails_top_n]
            try:
                result.emails = await self.generate_emails(
                    result.profile, top_for_emails, email_language
                )
            except Exception as e:
                return self._handle_error(result, "Email generation failed", e)
        
        self._progress("complete", "Pipeline finished successfully")
        return result
    
    # ---------- Error handling ----------
    
    def _handle_error(
        self,
        result: HanselResult,
        message: str,
        error: Exception | None,
    ) -> HanselResult:
        """Central error handler: log, record, and raise or return per strict mode."""
        full_msg = f"{message}: {error}" if error else message
        logger.error(full_msg)
        result.errors.append(full_msg)
        
        if self._strict:
            raise HanselError(full_msg) from error
        
        self._progress("error", full_msg)
        return result