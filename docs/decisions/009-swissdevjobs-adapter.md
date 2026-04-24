# ADR 009: Use swissdevjobs.ch as tech-specialist job source

**Status:** Accepted  
**Date:** 2026-04-23

## Context

Hansel needed a third job source with better coverage of Swiss tech roles.
The initial candidates were:

- **jobs.ch**: largest Swiss generalist board, but aggressive anti-scraping
  measures and Terms of Service that prohibit automated access
- **jobup.ch**: sister portal of jobs.ch, but focused exclusively on
  French-speaking Switzerland (Geneva, Lausanne) — not relevant for the
  Zürich-Aarau target zone
- **swissdevjobs.ch**: tech-specialist board covering German-speaking
  Switzerland, with transparent salary ranges and 200+ active listings

## Decision

Use swissdevjobs.ch via their internal `/api/jobsLight` JSON endpoint,
filtering results client-side by `techCategory` and `metaCategory`.

## Rationale

- **Coverage**: 200+ listings across Python, ML/AI, Data, DevOps categories
  covering Zürich, Aarau, Baden, Basel — exactly the target zone
- **Quality**: tech-specialist board with low noise, relevant roles only
- **Salary transparency**: every listing includes `annualSalaryFrom/To` in
  CHF — valuable signal for the matcher and for the user
- **No scraping needed**: the SPA preloads all data via a single JSON
  endpoint (`/api/jobsLight`), no JS rendering or HTML parsing required
- **Single request**: all listings fetched in one HTTP call and cached for
  the session — no per-query requests, no rate limit risk

## Consequences

**Positive:**
- Adds ~200 tech-specific Swiss listings with salary data to the pipeline
- Single HTTP request per pipeline run — negligible latency impact
- Salary fields (`salary_min`, `salary_max`, `salary_currency`) populated
  for every listing, unlike Adzuna or Arbeitnow

**Negative:**
- Endpoint is undocumented and internal — may change without notice
- All listings are fetched regardless of query; relevance filtering is
  done client-side, so some noise from non-Python categories may pass through
- `limit=50` default caps results per `search()` call; remaining listings
  are discarded before reaching the matcher

## Alternatives considered

- **jobs.ch scraping**: rejected — Terms of Service prohibit automated
  access and the site uses aggressive bot detection
- **jobup.ch**: rejected — covers French-speaking Switzerland only, not
  the Zürich-Aarau zone where the user is relocating
- **LinkedIn Jobs**: official API requires corporate approval and takes
  weeks; unofficial scraping is aggressively blocked