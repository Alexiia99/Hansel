"""Two-pass email generator with self-critique and hallucination guard."""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from hansel.cv.schemas import CVProfile
from hansel.email_gen.schemas import (
    EmailCritique,
    EmailDraft,
    EmailGenerationSkipped,
    EmailLanguage,
    FactCheckResult,
    GeneratedEmail,
)
from hansel.email_gen.validator import validate_email
from hansel.matcher.schemas import ScoredListing

logger = logging.getLogger(__name__)


# ---------- Prompts ----------

_DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You write personalized job application emails that recruiters actually read.

═══════════════════════════════════════════════════════════════════════
ANTI-HALLUCINATION RULES (ABSOLUTE, NO EXCEPTIONS)
═══════════════════════════════════════════════════════════════════════

You MUST NOT invent facts. The email must contain ONLY information that is 
EXPLICITLY present in the candidate's profile or the job description.

FORBIDDEN — these are fabrications, NEVER include them:
- Company names not in the candidate's experience section 
  (e.g., "XYZ Corp", "Company X", made-up employers)
- Project names not in the candidate's experience
  (e.g., "Project Alpha", "the payment system")
- Specific metrics/numbers not in the candidate's profile
  (e.g., "improved performance by 30%", "served 10k users", "saved $50k")
  EXCEPTION: you MAY quote metrics that ARE in the candidate's experience 
  descriptions, verbatim.
- Product names not mentioned in the job description
  (e.g., don't call the company's product "Dial.Plus" if the description 
  doesn't mention it)
- Job posting IDs or reference numbers not given
  (e.g., "Job Posting 2023-12" — if you don't have an ID, don't invent one)
- Claims about the candidate's motivation or feelings you can't verify
  (e.g., "passionate about VoIP since childhood" — no)

If you would benefit from a specific detail but it's not in the profile, 
WRITE A GENERAL STATEMENT instead. Example:
- DON'T: "At XYZ Corp I built a payment gateway"
- DO: "My backend experience includes building production APIs"

═══════════════════════════════════════════════════════════════════════
EMAIL STRUCTURE
═══════════════════════════════════════════════════════════════════════

Target: {word_count_target}-word email in {language}. Structure:

1. **Subject line**: specific and direct. NOT "Application" or "Interested in your role".
   Good: "Junior Python Developer application — 1.5 years REST API experience"
   Bad: "Job Application"

2. **Opening hook** (1 sentence): reference something you can verify from the 
   job description (the role focus, the tech stack listed, the team's mission 
   as stated). NO generic flattery.
   Good: "Your posting emphasizes building data pipelines for production ML — 
         an area I've been focused on for the past year."
   Bad: "I am writing to express my interest in your innovative company."

3. **Match points** (2-3 sentences): concrete skills from the candidate's 
   profile that match requirements from the job description.
   - Use the exact technologies the candidate has listed.
   - Reference the candidate's experience only in general terms UNLESS the 
     CV explicitly states a specific project.

4. **Honest gap** (1 sentence): acknowledge one clear gap with learning attitude.
   Reference a real gap, e.g., a required skill not in the candidate's list.

5. **Call to action** (1 sentence): low-friction next step.
   Good: "Would a 20-minute call next week be possible?"

═══════════════════════════════════════════════════════════════════════
FORMATTING RULES
═══════════════════════════════════════════════════════════════════════

- Target body length: around {word_count_target} words, minimum {min_words}, maximum {max_words}.
- If your first attempt is under {min_words} words, ADD a concrete fact 
  FROM THE CANDIDATE'S PROFILE (more of their skills, their seniority, 
  their languages). DO NOT invent.
- If you go over {max_words} words, trim the least-specific sentence.
- Language: write EVERYTHING in {language}.
- No corporate clichés: "I am writing to express...", "your innovative company", 
  "passionate about", "seeking opportunities to leverage".
- No emojis, no bullet points, just short paragraphs.

Return valid JSON with 'subject' and 'body'."""),
    ("human", """CANDIDATE PROFILE (use ONLY these facts):
Name: {name}
Summary: {summary}
Key skills: {skills}
Experience: {experience_brief}
Languages: {languages}

JOB (use ONLY these facts about the company/role):
Title: {job_title}
Company: {company}
Location: {location}
Description: {description}

MATCH ANALYSIS (for context only, don't quote):
Overall fit: {match_overall:.2f}
Skills match: {match_skills:.2f}
Rationale: {match_rationale}

Write the application email. ONLY use facts from above."""),
])


_CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a harsh but fair editor of job application emails.

Your job has THREE strict requirements:
1. The improved version MUST be BETTER than the draft, not just different.
2. The improved body MUST be between {min_words} and {max_words} words.
3. The improved version MUST NOT introduce any new facts not present in the draft.

═══════════════════════════════════════════════════════════════════════
CRITICAL RULE: DO NOT INVENT FACTS WHEN REWRITING
═══════════════════════════════════════════════════════════════════════

When rewriting, you can only REPHRASE, REORDER, or REMOVE content from 
the draft. You CANNOT add:
- New company names not in the draft
- New project names not in the draft
- New metrics/percentages not in the draft
- New specific facts about the candidate or company

If you want to fix a weakness like "the claim is generic", rewrite using 
EXISTING content — don't invent evidence.

═══════════════════════════════════════════════════════════════════════

STEP 1: Identify 3 SPECIFIC weaknesses.

GOOD critiques (concrete, actionable):
  - "The opening is generic. Replace with a reference to a SPECIFIC 
    technology mentioned in the job description."
  - "The word 'passionate' is a cliché. Replace with a verifiable claim 
    from the candidate's profile."
  - "The CTA is passive. Suggest a concrete duration and timeframe."

BAD critiques (vague or impossible to verify):
  - "Could be more engaging."
  - "Grammar needs work." (unless actually broken)

STEP 2: Rewrite addressing those weaknesses.

RULES FOR THE REWRITE:
- KEEP all concrete facts from the draft.
- Only change PHRASING or STRUCTURE where weak.
- STAY in {language}.
- HIT the word count: {min_words}-{max_words} words in the body.
- DO NOT introduce new company names, projects, metrics, or claims 
  that were not in the draft.
- Before finalizing: re-read your rewrite and check every noun. If you 
  can't trace it to the draft, REMOVE it.

Return valid JSON."""),
    ("human", """DRAFT TO CRITIQUE:

Subject: {draft_subject}

Body:
{draft_body}

---

Critique and rewrite. Only use facts present in the draft above."""),
])


_FACT_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict fact-checker for job application emails.

Your ONLY job: verify that every proper noun and specific claim in the email 
is supported by either the CANDIDATE PROFILE or the JOB DESCRIPTION provided.

Specifically, flag any of these as fabrications:
- Company names NOT in the candidate's experience AND NOT the target company
- Product names NOT in the job description (e.g., invented product nicknames)
- Specific metrics/numbers NOT in the candidate's profile 
  (e.g., '30% improvement', '10k users', '$50k saved')
- Specific projects or features NOT in the candidate's profile
- Posting IDs, reference numbers not provided in the source material

DO NOT flag (these are NOT fabrications):
- Common technologies mentioned in either source (Python, FastAPI, Docker, 
  REST APIs, data pipelines, etc.)
- The candidate's real employers (listed in experience)
- The target company's real name (as given in the job)
- Generic descriptions of activities that paraphrase the candidate's experience 
  ("building data pipelines", "developing backend systems", "writing tests").
  These describe what the candidate does based on their skills; they are NOT 
  specific claims about specific projects.
- Generic domain phrases ("production ML", "cloud infrastructure").
- The candidate's own name.

ONLY flag these (these ARE fabrications):
- Specific company names that are NOT the candidate's employer AND NOT the 
  target company.
- Specific product names NOT mentioned in the job description 
  (e.g., "Dial.Plus", "ProjectAlpha").
- Specific quantified claims NOT in the candidate's profile 
  (e.g., "reduced latency by 40%", "served 10k users").
- Specific project names NOT in the candidate's experience descriptions.

Return JSON with:
- is_valid: true if NO fabrications found; false if ANY fabrication found
- fabrications: list of specific problematic phrases (empty if is_valid is true)

Be strict. When in doubt, flag it."""),
    ("human", """CANDIDATE PROFILE (authoritative source):
Name: {name}
Skills: {skills}
Past employers: {employers}
Experience details: {experience_brief}

JOB (authoritative source):
Target company: {company}
Job title: {job_title}
Description: {description}

EMAIL TO FACT-CHECK:
Subject: {subject}
Body: {body}

Fact-check this email. Flag any claim not supported by the sources above."""),
])


# ---------- FactChecker class ----------


class FactChecker:
    """LLM-based fact-checker for generated emails.
    
    Catches subtle fabrications (invented product names, fake metrics) that
    pattern-based validators miss. Third line of defense after the prompt
    anti-hallucination rules and the regex validator.
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.0,
    ) -> None:
        """
        Args:
            model: Ollama model identifier.
            temperature: 0.0 for strict, deterministic fact-checking.
        """
        self._llm = ChatOllama(model=model, temperature=temperature)
        self._chain = _FACT_CHECK_PROMPT | self._llm.with_structured_output(FactCheckResult)
    
    async def check(
        self,
        subject: str,
        body: str,
        profile: CVProfile,
        listing: ScoredListing,
    ) -> FactCheckResult:
        """Check an email for fabricated facts. Returns structured result.
        
        Never raises: on LLM failure, returns a 'valid' result to avoid
        blocking valid emails. Logged at warning level.
        """
        try:
            employers = ", ".join(exp.company for exp in profile.experiences) or "None"
            return await self._chain.ainvoke({
                "name": profile.full_name,
                "skills": ", ".join(profile.skills[:15]),
                "employers": employers,
                "experience_brief": EmailGenerator._format_experience(profile),
                "company": listing.listing.company,
                "job_title": listing.listing.title,
                "description": (listing.listing.description or "")[:1200],
                "subject": subject,
                "body": body,
            })
        except Exception as e:
            logger.warning("Fact-checker failed; treating as valid: %s", e)
            return FactCheckResult(is_valid=True, fabrications=[])


# ---------- Generator class ----------


class EmailGenerator:
    """Generates personalized application emails using a three-layer pipeline.
    
    Pass 1: DRAFT — write a first version following strict anti-hallucination rules.
    Pass 2: CRITIQUE + REFINE — identify weaknesses and rewrite without new facts.
    Layer 3: VALIDATE — programmatic regex check for placeholder patterns, 
             then optional LLM fact-check for subtle fabrications.
    
    The pipeline falls back gracefully: if the critique fails validation,
    the draft is used; if both fail, EmailGenerationSkipped is returned.
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.3,
        min_match_score: float = 0.5,
        enable_fact_check: bool = True,
    ) -> None:
        """
        Args:
            model: Ollama model identifier.
            temperature: 0.3 gives slight variation without going off the rails.
                Emails would be identical across runs with fully deterministic
                (temperature=0), which looks robotic.
            min_match_score: Skip generation for listings below this score.
            enable_fact_check: Run LLM fact-checker as third safety layer.
                Adds ~30s per email but catches subtle hallucinations that
                the regex validator misses.
        """
        self._llm = ChatOllama(model=model, temperature=temperature)
        self._draft_chain = _DRAFT_PROMPT | self._llm.with_structured_output(EmailDraft)
        self._critique_chain = _CRITIQUE_PROMPT | self._llm.with_structured_output(EmailCritique)
        self._min_match_score = min_match_score
        self._fact_checker = FactChecker(model=model) if enable_fact_check else None
    
    async def generate(
        self,
        profile: CVProfile,
        scored_listing: ScoredListing,
        language: EmailLanguage = EmailLanguage.ENGLISH,
        word_count_target: int = 150,
    ) -> GeneratedEmail | EmailGenerationSkipped:
        """Generate a personalized email for one scored listing.
        
        Returns:
            GeneratedEmail if the match is good enough AND all validations pass.
            EmailGenerationSkipped otherwise, with explanation.
        """
        # 1. Quality gate
        if scored_listing.final_score < self._min_match_score:
            logger.info(
                "Skipping email for %r: match score %.2f below threshold %.2f",
                scored_listing.listing.title,
                scored_listing.final_score,
                self._min_match_score,
            )
            return EmailGenerationSkipped(
                reason=(
                    "Match score is below the quality threshold. "
                    "Generating an email for a weak match would waste the recruiter's "
                    "time and hurt the candidate's reputation."
                ),
                match_score=scored_listing.final_score,
                threshold=self._min_match_score,
            )
        
        # 2. Prepare template variables.
        # LLM local (Qwen 2.5 7B) consistently produces 85-130 word emails.
        # Local models don't follow strict word counts precisely; we treat the
        # target as guidance and accept well-formed shorter emails.
        min_words = 80
        max_words = word_count_target + 50
        
        draft_inputs = {
            "name": profile.full_name,
            "summary": profile.summary,
            "skills": ", ".join(profile.skills[:12]),
            "experience_brief": self._format_experience(profile),
            "languages": ", ".join(profile.languages),
            "job_title": scored_listing.listing.title,
            "company": scored_listing.listing.company,
            "location": scored_listing.listing.location or "Not specified",
            "description": (scored_listing.listing.description or "")[:1500],
            "match_overall": scored_listing.llm_score.overall if scored_listing.llm_score else 0.6,
            "match_skills": scored_listing.llm_score.skills_match if scored_listing.llm_score else 0.6,
            "match_rationale": (
                scored_listing.llm_score.rationale
                if scored_listing.llm_score
                else "Good baseline match based on embedding similarity."
            ),
            "language": language.label,
            "word_count_target": word_count_target,
            "min_words": min_words,
            "max_words": max_words,
        }
        
        # 3. Pass 1: draft
        logger.info("Drafting email for %r...", scored_listing.listing.title)
        draft: EmailDraft = await self._draft_chain.ainvoke(draft_inputs)
        
        # 4. Pass 2: critique + refine
        logger.info("Critiquing and refining...")
        critique: EmailCritique = await self._critique_chain.ainvoke({
            "draft_subject": draft.subject,
            "draft_body": draft.body,
            "language": language.label,
            "min_words": min_words,
            "max_words": max_words,
        })
        
        # 5. Programmatic validation of both versions
        candidate_companies = [exp.company for exp in profile.experiences]
        
        draft_validation = validate_email(
            body=draft.body,
            candidate_companies=candidate_companies,
            subject=draft.subject,
        )
        critique_validation = validate_email(
            body=critique.improved_body,
            candidate_companies=candidate_companies,
            subject=critique.improved_subject,
        )
        
        critique_body = critique.improved_body
        critique_word_count = len(critique_body.split())
        draft_word_count = len(draft.body.split())
        
        # Decide which version to use (programmatic checks).
        critique_passes = (
            critique_validation.is_valid
            and min_words <= critique_word_count <= max_words
            and critique_word_count >= draft_word_count * 0.7
        )
        
        if critique_passes:
            final_subject = critique.improved_subject
            final_body = critique_body
            word_count = critique_word_count
            logger.info(
                "Using critiqued version: %d → %d words, %d issues addressed",
                draft_word_count, critique_word_count, len(critique.weaknesses),
            )
        elif draft_validation.is_valid:
            final_subject = draft.subject
            final_body = draft.body
            word_count = draft_word_count
            logger.warning(
                "Critique rejected (%d words, hallucinations: %s). Keeping draft.",
                critique_word_count,
                critique_validation.issues or "none",
            )
        else:
            logger.error(
                "Both draft and critique contain hallucinations. "
                "Draft issues: %s. Critique issues: %s",
                draft_validation.issues,
                critique_validation.issues,
            )
            return EmailGenerationSkipped(
                reason=(
                    "Email generation produced content that appears fabricated "
                    "(placeholder names, unverifiable metrics)."
                ),
                match_score=scored_listing.final_score,
                threshold=self._min_match_score,
            )
        
        # 6. Third safety layer: LLM fact-checker (optional)
        if self._fact_checker is not None:
            logger.info("Fact-checking generated email...")
            fact_check = await self._fact_checker.check(
                subject=final_subject,
                body=final_body,
                profile=profile,
                listing=scored_listing,
            )
            
            if not fact_check.is_valid:
                logger.warning(
                    "Fact-checker flagged fabrications: %s", fact_check.fabrications
                )
                # Try the draft as fallback (if we were using the critique)
                if critique_passes and draft_validation.is_valid:
                    logger.info("Falling back to draft and re-checking...")
                    fallback_check = await self._fact_checker.check(
                        subject=draft.subject,
                        body=draft.body,
                        profile=profile,
                        listing=scored_listing,
                    )
                    if fallback_check.is_valid:
                        final_subject = draft.subject
                        final_body = draft.body
                        word_count = draft_word_count
                        logger.info("Draft passed fact-check; using it instead.")
                    else:
                        return EmailGenerationSkipped(
                            reason=(
                                "Both draft and critique contain fabricated facts "
                                f"not supported by the CV or job description. "
                                f"Critique issues: {fact_check.fabrications}. "
                                f"Draft issues: {fallback_check.fabrications}."
                            ),
                            match_score=scored_listing.final_score,
                            threshold=self._min_match_score,
                        )
                else:
                    return EmailGenerationSkipped(
                        reason=(
                            f"Generated email contains fabricated facts: "
                            f"{fact_check.fabrications}"
                        ),
                        match_score=scored_listing.final_score,
                        threshold=self._min_match_score,
                    )
        
        return GeneratedEmail(
            subject=final_subject,
            body=final_body,
            language=language,
            word_count=word_count,
            draft_subject=draft.subject,
            draft_body=draft.body,
            critique_points=critique.weaknesses,
        )
    
    @staticmethod
    def _format_experience(profile: CVProfile) -> str:
        """One-line summary of recent experience for the prompt."""
        if not profile.experiences:
            return "No formal experience listed; academic background only."
        items = []
        for exp in profile.experiences[:3]:
            end = exp.end_date or "present"
            items.append(f"{exp.role} at {exp.company} ({exp.start_date}–{end})")
        return "; ".join(items)