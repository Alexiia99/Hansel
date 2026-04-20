"""LLM-powered job search query generator."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from hansel.cv.schemas import CVProfile
from hansel.search.schemas import SearchStrategy


_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a job search strategist. Given a candidate profile and preferences, 
generate 3-5 DIVERSE search queries to maximize the chances of finding good matches.

Diversity rules:
- Don't just repeat the same title with different wording.
- Mix broader and more specific roles (e.g., '{seniority_label} Backend Developer' AND '{seniority_label} Python Developer').
- Consider adjacent roles the candidate could realistically apply to.
- Vary locations: primary preference + at least one remote option.

Seniority rule:
- EVERY 'keywords' value MUST include '{seniority_label}' to match the candidate's career level.
- Example: '{seniority_label} Data Engineer', not just 'Data Engineer'.
- The only exception is if the role title inherently implies seniority (e.g., 'Engineering Lead').

Format rules:
- 'keywords': 2-5 words max. No punctuation. No 'and', no commas.
- 'location': single location name, or 'remote'. Use standard names ('Switzerland', not 'CH').
- 'rationale': one short sentence, no fluff.

Return ONLY valid JSON."""),
    ("human", """Candidate profile:
Name: {full_name}
Summary: {summary}
Skills: {skills}
Target roles (inferred from CV): {target_roles}
Seniority level: {seniority_label}

Preferences:
Primary location: {preferred_location}
Open to remote: {remote_ok}

Generate diverse job search queries."""),
])


class QueryGenerator:
    """Generate diverse job search queries from a CV profile."""
    
    def __init__(self, model: str = "qwen2.5:7b-instruct", temperature: float = 0.0):
        self.llm = ChatOllama(model=model, temperature=temperature)
        self._chain = _QUERY_PROMPT | self.llm.with_structured_output(SearchStrategy)
    
    def generate(
        self,
        profile: CVProfile,
        preferred_location: str,
        remote_ok: bool = True,
    ) -> SearchStrategy:
        """Generate a search strategy (3-5 queries) from a CV profile.
        
        Args:
            profile: Extracted CV profile.
            preferred_location: User's primary target location (e.g., "Switzerland").
            remote_ok: Whether to include remote-friendly queries.
        
        Returns:
            SearchStrategy with 3-5 diverse JobQuery items.
        """
        if profile.seniority is None:
            raise ValueError(
                "profile.seniority is required to generate queries. "
                "Call CVExtractor.extract(..., seniority=...) first."
            )
        
        return self._chain.invoke({
            "full_name": profile.full_name,
            "summary": profile.summary,
            "skills": ", ".join(profile.skills),
            "target_roles": ", ".join(profile.target_roles),
            "seniority_label": profile.seniority.label,
            "preferred_location": preferred_location,
            "remote_ok": "yes, remote Europe also acceptable" if remote_ok else "no",
        })