# ADR 001: Hybrid regex + LLM approach for CV extraction

**Status:** Accepted  
**Date:** 2026 - 04 - 20

## Context

The agent needs to extract structured data from CVs (PDF, Markdown). Two naive approaches:

1. **Pure regex/rules**: brittle, can't handle semantic fields (summary, inferred roles)
2. **Pure LLM**: slow, unreliable on structured fields (emails get filtered for privacy, phone formats vary)

## Decision

Hybrid pipeline:

- **Regex** for deterministic fields: email, phone, LinkedIn, GitHub URLs.
- **LLM (Qwen 2.5 7B via Ollama)** for semantic fields: summary, skills, experiences, inferred target roles.
- **Post-processing** deterministic layer to patch common LLM mistakes (e.g., missing `end_date` for ongoing roles).

## Rationale

- Testing showed Qwen 2.5 7B consistently returns `null` for emails — likely PII-filtering behavior baked into training. Regex is 100% reliable here.
- LLMs excel at semantic tasks (summarizing, inferring target roles) where rules would be fragile.
- The post-processing layer catches a specific recurring bug without requiring LLM re-runs.
- This pattern (LLM for semantics, rules for structure) is common in production NLP pipelines.

## Consequences

**Positive:**
- Faster: regex is instant; LLM only handles fields it's good at.
- Cheaper: less LLM output means lower latency.
- More reliable: email/phone extraction never fails.

**Negative:**
- Two extraction layers to maintain.
- Regex patterns assume Latin-alphabet names and standard phone formats (not i18n-proof).

## Alternatives considered

- **Larger model (70B+):** prohibitive latency on CPU; contradicts local-first goal.
- **Fine-tuning a model:** overkill for the scope; LoRA on top of Qwen could improve if CV extraction became the main product.
- **Using a commercial CV parser (Affinda, HireAbility):** defeats the privacy-first premise.