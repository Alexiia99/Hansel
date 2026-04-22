"""Full matcher: embeddings retrieval + LLM reranking."""

from __future__ import annotations

import logging
from typing import Sequence

from hansel.cv.schemas import CVProfile, Seniority
from hansel.matcher.embeddings import EmbeddingScorer
from hansel.matcher.reranker import LLMReranker
from hansel.matcher.schemas import MatchScore, ScoredListing, SeniorityMode
from hansel.matcher.seniority_filter import (
    detect_title_seniority,
    filter_by_seniority_strict,
    seniority_score,
)
from hansel.sources.schemas import JobListing

logger = logging.getLogger(__name__)


def _combine_scores(
    embedding_score: float,
    llm_score: MatchScore | None,
    seniority_penalty: float,
) -> float:
    """Combine embedding, LLM, and seniority signals into a final score.
    
    Both paths (with or without LLM rerank) produce scores in a
    comparable range, preventing non-reranked listings from dominating
    when their embedding score happens to be high.
    
    With LLM score:
        base = 0.65 * llm.overall + 0.20 * embedding + 0.15 * llm.skills_match
    
    Without LLM score (embedding-only):
        We treat unknown LLM dimensions as "neutral" (0.5), so embedding-only
        listings don't unfairly win over reranked ones that got middling LLM
        scores.
        base = 0.65 * 0.5 + 0.20 * embedding + 0.15 * 0.5
             = 0.40 + 0.20 * embedding
    
    The final step applies a seniority multiplier to penalize role-level
    mismatches detected from the title (e.g., a Junior candidate on a
    'Staff Engineer' role).
    """
    if llm_score is not None:
        base = (
            llm_score.overall * 0.65
            + embedding_score * 0.20
            + llm_score.skills_match * 0.15
        )
    else:
        # "Neutral" stand-in for unknown LLM dimensions
        base = 0.50 * 0.65 + embedding_score * 0.20 + 0.50 * 0.15
    
    return max(0.0, min(1.0, base * seniority_penalty))

class JobMatcher:
    """Ranks job listings against a CV using a retrieve-and-rerank pipeline.
    
    Pipeline:
      1. Filter by seniority (strict mode) or score penalty (inclusive mode).
      2. Compute embedding similarity for every listing (fast).
      3. Take top N by embedding score.
      4. LLM reranks those N with full-context scoring.
      5. Combine all signals into a final score; rank by it.
    """
    
    def __init__(
        self,
        embedding_scorer: EmbeddingScorer | None = None,
        reranker: LLMReranker | None = None,
        rerank_top_n: int = 10,
        seniority_mode: SeniorityMode = SeniorityMode.STRICT,
    ) -> None:
        self._embeddings = embedding_scorer or EmbeddingScorer()
        self._reranker = reranker or LLMReranker()
        self._rerank_top_n = rerank_top_n
        self._seniority_mode = seniority_mode
    
    async def rank(
        self,
        profile: CVProfile,
        listings: Sequence[JobListing],
    ) -> list[ScoredListing]:
        """Rank listings by fit against the CV profile.
        
        Args:
            profile: Extracted CV with seniority set.
            listings: Candidates from the orchestrator.
        
        Returns:
            List of ScoredListing, sorted by final_score descending.
            Includes all listings (possibly with llm_score=None for those 
            beyond top N).
        """
        if not listings:
            logger.warning("No listings provided to matcher")
            return []
        
        if profile.seniority is None:
            raise ValueError(
                "profile.seniority must be set to use the matcher"
            )
        
        # 1. Seniority filter
        candidates = self._apply_seniority_filter(list(listings), profile.seniority)
        logger.info(
            "After seniority filter (%s): %d/%d listings remain",
            self._seniority_mode.value, len(candidates), len(listings),
        )
        
        if not candidates:
            return []
        
        # 2. Embedding scoring
        embedding_results = await self._embeddings.score_listings(profile, candidates)
        
        # 3. Select top N for LLM rerank
        sorted_by_embedding = sorted(
            embedding_results, key=lambda x: x[1], reverse=True
        )
        to_rerank = sorted_by_embedding[: self._rerank_top_n]
        not_reranked = sorted_by_embedding[self._rerank_top_n :]
        
        logger.info(
            "Reranking top %d listings with LLM (remaining %d keep embedding-only scores)",
            len(to_rerank), len(not_reranked),
        )
        
        # 4. LLM rerank
        rerank_listings = [l for l, _ in to_rerank]
        llm_scores = await self._reranker.score_batch(profile, rerank_listings)
        
        # 5. Combine all scores
        scored: list[ScoredListing] = []
        
        # Top N: embedding + LLM
        for (listing, emb_score), llm_score in zip(to_rerank, llm_scores):
            senior_score = seniority_score(
                profile.seniority, detect_title_seniority(listing.title)
            )
            final = _combine_scores(emb_score, llm_score, senior_score)
            scored.append(ScoredListing(
                listing=listing,
                embedding_score=emb_score,
                llm_score=llm_score,
                final_score=final,
            ))
        
        # Rest: embedding only
        for listing, emb_score in not_reranked:
            senior_score = seniority_score(
                profile.seniority, detect_title_seniority(listing.title)
            )
            final = _combine_scores(emb_score, None, senior_score)
            scored.append(ScoredListing(
                listing=listing,
                embedding_score=emb_score,
                llm_score=None,
                final_score=final,
            ))
        
        # 6. Final sort by combined score
        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored
    
    def _apply_seniority_filter(
        self, listings: list[JobListing], user_seniority: Seniority
    ) -> list[JobListing]:
        """Filter or keep listings based on seniority_mode."""
        if self._seniority_mode is SeniorityMode.STRICT:
            return filter_by_seniority_strict(listings, user_seniority)
        # INCLUSIVE: keep all; penalty is applied in _combine_scores.
        return listings