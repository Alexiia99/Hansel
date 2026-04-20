# ADR 002: LLM-generated diverse search queries from CV profile

**Status:** Accepted  
**Date:** 2026 - 04 - 20

## Context

Given a CV profile, the agent needs to decide what to search for on job APIs 
(Adzuna, Jooble, jobs.ch, Arbeitnow). Naive options:

1. **Use target_roles directly**: search for each inferred role verbatim. 
   Problem: no location variation, no adjacent roles explored, duplicates 
   across sources.
2. **Single hard-coded query**: search once for "Junior Data Engineer". 
   Problem: misses adjacent matches (Backend, ML, Python Developer).
3. **LLM-generated query set**: LLM produces 3-5 diverse queries using 
   profile + user preferences as context.

## Decision

Option 3. A dedicated `QueryGenerator` component takes:
- The extracted `CVProfile`
- User's `preferred_location`
- `remote_ok` boolean

And returns a `SearchStrategy` with 3-5 `JobQuery` items varying in role, 
location, and scope.

## Rationale

- **Coverage**: 3-5 queries with different angles maximize the chance of 
  finding good matches across job boards.
- **Adaptability**: the LLM can suggest adjacent roles (e.g., 'Python 
  Developer' for a 'Data Engineer' candidate) that rigid rules would miss.
- **User control**: `preferred_location` and `remote_ok` are explicit user 
  parameters, not scraped from the CV.
- **Seniority consistency**: every query includes the user-declared seniority 
  label ('Junior', 'Mid-level', 'Senior'), enforced by the prompt.

## Consequences

**Positive:**
- Diverse query generation without hand-crafted rules.
- Seniority-aware, language-aware, location-aware.
- Easy to scale: pass any profile + preferences, get queries.

**Negative:**
- Non-deterministic: different runs may produce slightly different queries.
- Requires an LLM call (~40-60s on local 7B CPU inference). Cached in 
  production to avoid re-running for the same profile + preferences.

## Alternatives considered

- **Rule-based expansion**: mapping skills → job titles via a static dict. 
  Rejected: brittle, not reusable across domains, needs constant maintenance.
- **Web-scraping seed queries from LinkedIn/Indeed**: rejected due to ToS and 
  the project's local-first philosophy.