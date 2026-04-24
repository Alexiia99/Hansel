"""LLM-based reranker: scores listings with rich justification."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.prompts import ChatPromptTemplate
from hansel.llm import make_chat_ollama

from hansel.cv.schemas import CVProfile
from hansel.matcher.schemas import MatchScore
from hansel.sources.schemas import JobListing

logger = logging.getLogger(__name__)


_SCORING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert technical recruiter evaluating how well a candidate matches a job.

Produce three numeric scores and a brief rationale:

- `overall`: 0.0 (no fit) to 1.0 (excellent fit). Be honest. Most real matches are 0.4-0.7. Reserve 0.8+ for truly strong matches. Reserve 0.2- for clear mismatches.
- `skills_match`: how well the candidate's skills align with requirements. 1.0 = all key requirements present, 0.0 = nothing matches.
- `seniority_fit`: 1.0 = role seniority matches candidate's level, 0.5 = stretch in either direction, 0.0 = very incompatible (e.g., junior for staff role).
- `rationale`: 2-3 SHORT sentences. First: the strongest match point. Second: main gap or concern. Optionally third: overall verdict.

Calibration examples:
- Junior Python dev + 'Senior Backend Engineer (Rust)': overall≈0.15 (seniority+stack mismatch)
- Junior Python dev + 'Junior Python Developer at small company': overall≈0.85 (direct match)
- Junior Python dev + 'Data Engineer (some Python)': overall≈0.55 (adjacent role)
- Junior Python dev + 'Mechatronics Engineer': overall≈0.20 (different field entirely)

Return ONLY valid JSON. No markdown, no preamble."""),
    ("human", """CANDIDATE:
Seniority: {seniority}
Summary: {summary}
Top skills: {skills}
Target roles: {target_roles}
Languages: {languages}

JOB:
Title: {title}
Company: {company}
Location: {location}
Description: {description}

Evaluate the match."""),
])


class LLMReranker:
    """Reranks a set of listings by invoking the LLM on each."""
    
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.0,
        max_concurrency: int = 1,
    ) -> None:
        """
        Args:
            model: Ollama model identifier.
            temperature: 0 for reproducibility during development.
            max_concurrency: Parallel LLM calls. On CPU, keep at 1 — more 
                does not speed up and can starve the machine. On GPU, 2-4.
        """
        self._llm = make_chat_ollama(model=model, temperature=temperature)
        self._chain = _SCORING_PROMPT | self._llm.with_structured_output(MatchScore)
        self._semaphore = asyncio.Semaphore(max_concurrency)
    
    async def score(
        self,
        profile: CVProfile,
        listing: JobListing,
    ) -> MatchScore | None:
        """Score a single listing. Returns None on LLM error."""
        async with self._semaphore:
            try:
                result: MatchScore = await self._chain.ainvoke({
                    "seniority": profile.seniority.label if profile.seniority else "Unknown",
                    "summary": profile.summary,
                    "skills": ", ".join(profile.skills[:15]),
                    "target_roles": ", ".join(profile.target_roles),
                    "languages": ", ".join(profile.languages),
                    "title": listing.title,
                    "company": listing.company,
                    "location": listing.location or "Not specified",
                    "description": (listing.description or "")[:1500],
                })
                return result
            except Exception as e:
                logger.warning(
                    "LLM scoring failed for %r: %s", listing.title, e
                )
                return None
    
    async def score_batch(
        self,
        profile: CVProfile,
        listings: list[JobListing],
    ) -> list[MatchScore | None]:
        """Score multiple listings. Returns one result per listing (or None on failure)."""
        logger.info("LLM-reranking %d listings...", len(listings))
        tasks = [self.score(profile, l) for l in listings]
        return await asyncio.gather(*tasks)