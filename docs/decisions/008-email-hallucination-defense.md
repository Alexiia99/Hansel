# ADR 008: Three-layer defense against LLM hallucinations in email generation

**Status:** Accepted  
**Date:** 2026 - 04 - 22

## Context

The email generator must produce personalized application emails without 
fabricating facts (invented company names, fake metrics, imaginary projects). 
A single email with "At XYZ Corp I increased revenue by 30%" — where neither 
XYZ Corp nor the metric exists in the candidate's profile — burns the 
candidate's credibility with the recruiter.

Single-pass LLM generation with anti-hallucination prompt rules was 
insufficient. During development we observed five categories of hallucination:

1. **Placeholder names**: "XYZ Corp", "Company X", "Project Alpha"
2. **Generic template leakage**: "[your innovative company]", Lorem ipsum
3. **Fake quantified claims**: "improved performance by 30%" (no source)
4. **Plausible invented brand names**: "Dial.Plus" (sounds real, isn't)
5. **Subject integrity failures**: empty subject lines from critique step

## Decision

A three-layer defense:

1. **Layer 1 (prompt engineering)**: explicit anti-hallucination rules in 
   both the drafting and critiquing prompts. Both explicit "FORBIDDEN" 
   sections with concrete examples.

2. **Layer 2 (regex validator)**: pattern-based detection of obvious 
   placeholder names (XYZ, ABC, Project Alpha), unfilled template brackets, 
   Lorem ipsum, suspicious percentage-based claims, and empty subject lines.

3. **Layer 3 (LLM fact-checker)**: a dedicated fact-checking LLM pass that 
   receives the generated email alongside the candidate profile and job 
   description, and flags any claim not supported by either source. 
   Runs at temperature 0 for strict, deterministic output.

Fallback chain:
- If the critique step fails layer 2 or 3 → use the draft and re-check.
- If both draft and critique fail → return `EmailGenerationSkipped` with 
  explanation rather than delivering a fabricated email.

## Rationale

- **Defense in depth**: no single layer catches everything. Prompt rules 
  are the cheapest but most violated; regex catches obvious placeholders; 
  LLM fact-check catches subtle plausible fabrications.
- **Prefer skipping to lying**: an empty result with a clear reason is 
  strictly better for the user than a fabricated email. The user can retry 
  or edit manually.
- **Graceful degradation**: the fallback chain gives the system multiple 
  chances to produce a valid output before giving up.

## Consequences

**Positive:**
- No observed hallucinations in validated output across the test set.
- Users can trust the generated emails are factually consistent with 
  their inputs.
- Clear skip reasons provide actionable feedback when generation fails.

**Negative:**
- Adds ~30s per email from the fact-check pass. Total generation time is 
  now ~2 minutes per email on CPU with Qwen 2.5 7B.
- Occasional false positives in the fact-checker: generic phrases like 
  "data pipelines" were initially flagged as fabrications. Mitigated with 
  an explicit "DO NOT flag" list in the fact-check prompt.

## Alternatives considered

- **Single-pass with strict prompt only**: rejected — hallucinations were 
  frequent even with detailed prompt rules.
- **Entity linking against a knowledge base**: overkill for the scope; 
  would require external dependencies.
- **Post-hoc manual review UI**: still valuable, but doesn't replace 
  automated quality control for users who copy-paste without reading 
  carefully.
- **Larger model (70B+)**: would reduce hallucination rate but contradicts 
  the local-first design and the CPU-only constraint.