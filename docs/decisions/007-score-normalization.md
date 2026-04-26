# ADR 007: Normalize scores across rerank tiers

**Status:** Accepted  
**Date:** 2026-04-22

## Context

The matcher pipeline has two tiers:

1. **Top N listings** — scored by both embeddings + LLM reranker
2. **Remaining listings** — scored by embeddings only (LLM not called)

Without normalization, a listing in tier 2 with a high embedding score
(e.g. 0.85) would outscore a tier 1 listing with a good LLM score (e.g.
overall=0.70), simply because the LLM penalty wasn't applied. This would
make the LLM rerank pointless.

## Decision

Use a weighted combination formula that treats unknown LLM dimensions as
"neutral" (0.5) for tier 2 listings:

**With LLM score (tier 1):**
```
base = 0.65 * llm.overall + 0.20 * embedding + 0.15 * llm.skills_match
```

**Without LLM score (tier 2):**
```
base = 0.65 * 0.5 + 0.20 * embedding + 0.15 * 0.5
     = 0.40 + 0.20 * embedding
```

A seniority multiplier is applied to both tiers as a final step.

## Rationale

- A tier 2 listing with embedding=1.0 gets a maximum score of 0.60,
  which is always below a tier 1 listing with a good LLM score.
- This guarantees that reranked listings always rank above non-reranked
  ones with similar embedding scores.
- The neutral 0.5 assumption is conservative — we don't penalize listings
  we haven't evaluated, but we don't reward them either.

## Consequences

**Positive:**
- Reranked listings always dominate non-reranked ones.
- Scores are comparable across tiers — no artificial inflation.

**Negative:**
- Tier 2 listings are effectively capped at 0.60, which may hide
  genuinely good matches that didn't make the top N cutoff.
- Mitigated by setting `rerank_top_n=10`, which covers most relevant
  listings in practice.
