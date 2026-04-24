"""Semantic similarity via Ollama embeddings."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from hansel.llm import make_ollama_embeddings

from hansel.cv.schemas import CVProfile
from hansel.sources.schemas import JobListing

logger = logging.getLogger(__name__)


def cv_to_text(profile: CVProfile) -> str:
    """Render a CV as a dense query vector for semantic search.
    
    Short, dense, and bilingual-safe. We deliberately avoid long prose
    (summary, experience descriptions) because it dilutes signal when
    matched against noisy multilingual job descriptions.
    
    Format: target roles + top skills. No seniority prefix (already in roles).
    """
    # Strip 'Junior '/'Senior '/'Mid-level ' prefixes that target_roles already carry,
    # to avoid "Junior Junior Backend Developer" when combined later.
    roles = ", ".join(profile.target_roles)
    skills = ", ".join(profile.skills[:15])
    return f"Looking for: {roles}. Skills: {skills}."


def listing_to_text(listing: JobListing) -> str:
    """Render a job listing as a dense target vector.
    
    We deliberately EXCLUDE the description. Job descriptions are:
    - Often in a different language than the CV (DE/FR for Swiss jobs)
    - Full of company marketing that has nothing to do with the role
    - So long they dominate the embedding and dilute title signal
    
    Tests showed the title alone is a stronger relevance signal.
    Tags supplement when present.
    """
    parts = [listing.title, listing.company]
    if listing.tags:
        parts.append("Tags: " + ", ".join(listing.tags))
    return ". ".join(parts)



def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors, in [0, 1] for normalized vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product == 0:
        return 0.0
    
    cos = float(np.dot(a, b) / norm_product)
    # Clip to [0, 1]: negative cosine would mean opposed meanings (rare for this data)
    return max(0.0, min(1.0, cos))


class EmbeddingScorer:
    """Scores listings against a CV using local embeddings.
    
    Uses nomic-embed-text via Ollama. Produces a similarity score 
    between 0 (unrelated) and 1 (nearly identical).
    """
    
    def __init__(self, model: str = "nomic-embed-text") -> None:
        self._embeddings = make_ollama_embeddings(model=model)
    
    async def score_listings(
        self,
        profile: CVProfile,
        listings: Sequence[JobListing],
    ) -> list[tuple[JobListing, float]]:
        """Compute similarity score for every listing against the profile.
        
        Returns:
            List of (listing, score) tuples in the original order.
        """
        if not listings:
            return []
        
        # 1. Prepare texts
        cv_text = cv_to_text(profile)
        listing_texts = [listing_to_text(l) for l in listings]
        
        logger.info(
            "Embedding 1 CV and %d listings...", len(listings)
        )
        
        # 2. Get embeddings (langchain-ollama batches internally)
        cv_vector = await self._embeddings.aembed_query(cv_text)
        listing_vectors = await self._embeddings.aembed_documents(listing_texts)
        
        # 3. Compute similarity per listing
        scored: list[tuple[JobListing, float]] = []
        for listing, vec in zip(listings, listing_vectors):
            score = cosine_similarity(cv_vector, vec)
            scored.append((listing, score))
        
        logger.info("Embedding scoring complete")
        return scored