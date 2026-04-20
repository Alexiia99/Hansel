# ADR 003: Keyword strategy — don't trust job-board search matching

**Status:** Accepted  
**Date:** 2026 - 04 - 20

## Context

During adapter development (week 2) we discovered that Adzuna's `what` 
parameter does loose keyword matching:

- Query `"Junior Python Developer"` in Zurich → 0 results
- Query `"Junior Developer"` in Zurich → 5 results, ALL senior positions

Job boards match keywords across title AND description, and seem to favor 
partial matches over strict AND semantics. This means the word "junior" 
appearing in a sentence like "we mentor junior developers" can surface 
senior job postings.

## Decision

Two-layer strategy:

1. **Query broadly**: remove fragile seniority qualifiers from the keywords 
   passed to job boards. Use `"Python Developer"` instead of 
   `"Junior Python Developer"`.

2. **Filter locally**: apply seniority classification in our own pipeline, 
   using:
   - Deterministic title heuristics (e.g., reject titles containing 
     "Senior", "Lead", "Principal", "Staff" for junior candidates).
   - LLM-based evaluation in the matcher component (week 4), which looks 
     at the full description and decides fit.

## Rationale

- Job-board search is optimized for recall, not precision. Fighting it 
  with narrower queries just removes the signal along with the noise.
- Our own filtering layer has access to the full job description and can 
  make better calls than a keyword search engine.
- Separating retrieval from ranking is a standard pattern in search 
  systems (e.g., BM25 retrieval → neural reranker).

## Consequences

**Positive:**
- We surface more candidate jobs (higher recall).
- Our seniority filter can be richer than a keyword match (context-aware).
- Users see fewer obvious mismatches.

**Negative:**
- More post-processing work in the agent.
- More LLM calls for evaluation.

## Alternatives considered

- **Keep strict keyword queries**: rejected — empirically returns 0 results 
  for realistic combinations.
- **Use API-specific filters** (e.g., `category=IT-Jobs` in Adzuna): 
  considered for future iterations. Not all sources expose such filters.