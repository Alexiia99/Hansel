# ADR 004: Job sources — Jooble dropped after integration testing

**Status:** Accepted  
**Date:** 2025-11-XX

## Context

Week 2 planned four job sources: Adzuna, Jooble, Arbeitnow, and jobs.ch.  
During Jooble integration, empirical testing revealed that:

- `keywords="Python", location="Zurich"` → 0 results
- `keywords="Python", location="Switzerland"` → 5 results, but all from 
  "Switzerland County, IN" (Indiana, USA)
- `keywords="Entwickler", location="Zürich"` → 0 results
- `keywords="Python"` globally → 44k+ results, dominated by USA / India

Jooble's coverage is weak for Switzerland (the target market), and its 
location matching does not handle European countries well.

## Decision

Drop Jooble from the pipeline. Keep three sources:

1. **Adzuna** — primary source for Switzerland (893 results on "Python" 
   from the `/jobs/ch` endpoint).
2. **Arbeitnow** — complementary source for remote / DACH-region tech roles.
3. **jobs.ch (scraping)** — to be added in week 3 for deep Swiss market 
   coverage.

## Rationale

- Adding a source that returns near-zero relevant results wastes agent 
  runtime with no benefit.
- Three well-chosen sources provide enough diversity without added complexity.
- The `JobSourceAdapter` abstraction remains validated by two diverse 
  implementations (authenticated GET with server-side filter vs public 
  POST-less feed with client-side filter).

## Consequences

**Positive:**
- Faster agent runs (one less HTTP call per query).
- Less noise in results.
- Honest about real coverage limits.

**Negative:**
- Non-Swiss users would have less coverage. Acceptable: the project is 
  optimized for the Swiss job market.

## Alternatives considered

- **Keep Jooble anyway**: rejected. Dead weight in the pipeline.
- **Replace with The Muse**: deferred. The Muse has a similar USA bias, 
  and adding another adapter without verifying coverage first would 
  repeat the mistake.
- **Add a German source (stepstone.de)**: deferred. Out of scope for 
  Swiss focus, but a candidate for future iteration.