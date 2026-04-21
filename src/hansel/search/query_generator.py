"""LLM-powered job search query generator."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from hansel.cv.schemas import CVProfile
from hansel.search.schemas import SearchStrategy

_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a job search strategist. Given a candidate profile and preferences, 
generate 3-5 DIVERSE search queries to find matches on job boards like Adzuna and jobs.ch.

CRITICAL TECHNICAL CONSTRAINT:
Job boards do AND matching on keywords. 3+ word queries return ZERO results almost 
always. You MUST use MAXIMUM 2 words per keyword.

Good examples: 'Backend Developer', 'Data Engineer', 'Python Developer', 'Software Engineer'.
Bad examples (DO NOT USE):
- 'Junior Backend Developer' (too specific, 'Junior' blocks results)
- 'Backend Developer Python' (3 words, zero results)  
- 'Python Developer REST APIs' (4 words, zero results)

Rules for keywords:
- MAXIMUM 2 words. Non-negotiable.
- NEVER seniority words ('Junior', 'Senior', 'Mid-level', 'Lead').
- Use STANDARD job titles (what a recruiter would post).
- Examples: 'Backend Developer', 'Data Engineer', 'Python Developer', 
  'Software Engineer', 'ML Engineer', 'Full Stack Developer', 'DevOps Engineer'.

Rules for diversity:
- Each query targets a different role angle.
- Mix: core role + adjacent roles based on candidate skills.

Rules for location:
- Put country/city in the 'location' field, NEVER inside keywords.
- Use standard names: 'Switzerland', 'Zurich', 'Germany', 'remote'.
- Mix at least 1 primary location + 1 remote-friendly query.

Return ONLY valid JSON. MAXIMUM 2 WORDS PER KEYWORD."""),
    ("human", """Candidate profile:
Name: {full_name}
Summary: {summary}
Skills: {skills}
Target roles (inferred): {target_roles}
Seniority level (FOR LATER FILTERING): {seniority_label}

Preferences:
Primary location: {preferred_location}
Open to remote: {remote_ok}

Generate 3-5 diverse search queries with MAXIMUM 2 WORDS per keyword."""),
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