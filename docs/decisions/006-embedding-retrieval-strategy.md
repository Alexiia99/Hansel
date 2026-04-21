# ADR 006: Embeddings for retrieval, LLM for ranking precision

**Status:** Accepted  
**Date:** 2026-04-21

## Context

The matcher receives ~20-30 job listings from the orchestrator and must rank 
them by fit against a CV profile. Two extremes are possible:

1. **LLM-scores everything**: 30 × 30s = 15 min. Unusable.
2. **Pure embedding similarity**: fast, but low precision on our data.

We chose a **retrieve + rerank** hybrid architecture.

During embedding retrieval implementation, we tested three text renderings 
to embed CVs and listings against `nomic-embed-text` (via Ollama):

| Version | CV text | Listing text | Result |
|---------|---------|--------------|--------|
| v1 | Full summary + exp + skills | Title + company + full description | Bottom 5 included "Data Engineer" (obvious match). Score spread: 0.22 |
| v2 | Summary first sentence + roles + skills | Title + tags + 400-char description | "Junior/Mid Python Developer" in bottom. Score spread: 0.14 |
| v3 (chosen) | Roles + skills only | Title + company + tags (NO description) | Better top ranking; "Junior/Mid Python Developer" at rank 16/27. Score spread: 0.19 |

All three placed the obviously-best listing ("Junior/Mid Python Developer in 
Möhlin", 15 min from user's target location of Aarau) in the middle of the 
ranking, not the top. Embedding signal alone is not precise enough.

## Root cause

Observed issues:

- **Multilingual noise**: Swiss job descriptions mix German, French, and 
  English. `nomic-embed-text` handles this imperfectly; long descriptions in 
  German dilute the signal of English job titles.
- **Corporate boilerplate**: descriptions are dominated by marketing text 
  ("We are a leading company in...") that carries no role information.
- **Short-text sensitivity**: embedding models applied to job titles are 
  sensitive to peripheral tokens (company names, tags) that shift vectors 
  away from the semantic core.

These are known limitations of generic embeddings on noisy job data. 
Production systems typically fine-tune embeddings on job-match pairs or 
extract structured features (skills, seniority) before embedding.

## Decision

Keep embeddings but redefine their job:

**Embeddings are a retrieval filter, not a ranking oracle.**

- Compute embedding similarity for all listings from the orchestrator.
- Keep the top `rerank_top_n` (default 10, configurable).
- Pass those to the LLM reranker, which reads full context (CV + description) 
  and produces the final ranking.

The embedding is now tuned for **recall**: as long as the truly-relevant 
listings appear somewhere in the top N (even if not at #1), the LLM will 
find and elevate them.

## Rationale

- **Latency stays bounded**: the LLM only processes `rerank_top_n` listings 
  (~10 × 30s = 5 min), not all 30.
- **Quality is owned by the LLM**: which can read descriptions, detect 
  seniority nuance, and produce justified scores.
- **This is industry standard**: retrieve-and-rerank is how search engines 
  (BM25 → neural reranker) and RAG systems (vector search → LLM) work.
- **Honest about limits**: we don't pretend the embedding alone is good 
  enough. We use it for what it does well.

## Consequences

**Positive:**
- Runs in acceptable time (~5-7 min end-to-end).
- Quality responsibility is clearly assigned per component.
- Architecture matches production patterns, looks good in system design 
  conversations.

**Negative:**
- Relies on the LLM working well (future risk).
- If `rerank_top_n` is too low, relevant listings could be cut. Tunable.

## Alternatives considered

- **Fine-tune embeddings**: out of scope and impractical without a labeled 
  dataset. Future work.
- **Use a cross-encoder**: better reranker, but adds another model dependency. 
  Qwen 7B as judge is simpler.
- **Trust only LLM on everything**: 15-minute runs killed this option.

## Success criteria

The architecture is working if, after the LLM reranker:

- Known relevant listings (like "Junior/Mid Python Developer in Möhlin" for 
  the test CV) appear in the top 5.
- Known irrelevant ones (like "Mechatronics Engineer" for a software CV) 
  drop below the top 10.

This will be verified when the reranker is implemented.