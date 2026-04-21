# ADR 005: Location handling in Adzuna adapter

**Status:** Accepted  
**Date:** 2026 - 04 - 21

## Context

Adzuna has country-specific endpoints (`/jobs/ch/`, `/jobs/de/`, etc.) AND 
a `where` parameter. Empirical testing showed:

- `/jobs/ch/?what=Backend+Developer` → 31 results
- `/jobs/ch/?what=Backend+Developer&where=Switzerland` → 0 results

Passing the country name as `where` makes Adzuna do strict substring 
matching against the location field of each job, which rarely contains 
"Switzerland" verbatim (jobs say "Zürich", "Schweiz", "Bern", etc.).

## Decision

In `AdzunaAdapter._fetch_once`, skip the `where` parameter when `location` 
is a country name that matches the adapter's country endpoint.

Only pass `where` for sub-country locations (cities, regions) or non-matching 
country names.

## Rationale

- The endpoint already filters by country.
- Users and the QueryGenerator use "Switzerland" as a natural specifier; 
  they shouldn't have to know about this API quirk.
- Keeps recall high for country-level queries.

## Consequences

**Positive:**
- Restores high recall for queries like "Backend Developer Switzerland".
- Users don't need to leave location blank to get country-wide results.

**Negative:**
- Tiny inconsistency: the adapter silently drops a parameter. Logged at 
  DEBUG level so it's visible during debugging.
- Hardcoded list of country names. Acceptable for MVP; a future enhancement 
  could use a proper country code lookup.