"""Main CV extraction pipeline: regex + LLM."""

from __future__ import annotations

from pathlib import Path

import pdfplumber
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from hansel.cv.regex_parser import extract_contact
from hansel.cv.schemas import CVProfile, CVProfileSemantic, Experience


_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You extract structured information from CVs. You are thorough and never miss information.

CRITICAL RULES:
1. Only extract info explicitly in the CV, EXCEPT 'target_roles' (you MUST infer).
2. 'target_roles': return EXACTLY 3-5 roles. EVERY role starts with 'Junior'. Be specific.
3. 'skills': extract EVERY technology, framework, language, tool, database, platform mentioned ANYWHERE.
   - Look in the skills section AND in experience descriptions.
   - Common misses: SQL, JavaScript, REST APIs, Flask, pytest. Don't miss them.
   - Normalize naming (e.g., 'python' → 'Python').
4. 'location': if any city/country is mentioned near the name or contact info, extract it.
5. 'languages': include level ('Native', 'Fluent', 'B2', 'Basic', etc.).
6. Dates: YYYY-MM format ('Septiembre 2023' → '2023-09'). Ongoing role → 'current'.
7. Return ONLY valid JSON. No markdown. No explanations."""),
    ("human", "CV text:\n\n{cv_text}"),
])


_ONGOING_KEYWORDS = {
    "actualmente", "actualidad", "presente", "actual",
    "present", "current", "currently", "ongoing",
    "hasta la fecha", "now",
}


def load_cv_text(path: str | Path) -> str:
    """Load CV text from .pdf, .md, or .txt."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CV not found: {path}")
    
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    
    raise ValueError(f"Unsupported CV format: {suffix}. Use .pdf, .md, or .txt")


def _post_process_experiences(
    experiences: list[Experience], cv_text: str
) -> list[Experience]:
    """Fix end_date=null when CV clearly indicates ongoing role."""
    cv_lower = cv_text.lower()
    fixed: list[Experience] = []
    
    for exp in experiences:
        if exp.end_date is None:
            anchor = cv_lower.find(exp.role.lower())
            if anchor == -1:
                anchor = cv_lower.find(exp.company.lower())
            if anchor != -1:
                window = cv_lower[anchor : anchor + 200]
                if any(kw in window for kw in _ONGOING_KEYWORDS):
                    exp = exp.model_copy(update={"end_date": "current"})
        fixed.append(exp)
    
    return fixed


class CVExtractor:
    """Extracts structured CV data using a hybrid regex + LLM pipeline."""
    
    def __init__(self, model: str = "qwen2.5:7b-instruct", temperature: float = 0.0):
        self.llm = ChatOllama(model=model, temperature=temperature)
        self._chain = _EXTRACTION_PROMPT | self.llm.with_structured_output(CVProfileSemantic)
    
    def extract(self, cv_text: str) -> CVProfile:
        """Extract full profile from CV text.
        
        Pipeline:
          1. regex for deterministic fields (email, phone, linkedin, github)
          2. LLM for semantic fields (summary, skills, experiences, etc.)
          3. post-processing to patch common LLM mistakes (end_date)
        """
        contact = extract_contact(cv_text)
        semantic: CVProfileSemantic = self._chain.invoke({"cv_text": cv_text})
        experiences = _post_process_experiences(semantic.experiences, cv_text)
        
        return CVProfile(
            email=contact.email,
            phone=contact.phone,
            linkedin=contact.linkedin,
            github=contact.github,
            full_name=semantic.full_name,
            location=semantic.location,
            summary=semantic.summary,
            skills=semantic.skills,
            languages=semantic.languages,
            experiences=experiences,
            education=semantic.education,
            target_roles=semantic.target_roles,
        )
    
    def extract_from_file(self, path: str | Path) -> CVProfile:
        """Extract profile from a CV file (PDF, Markdown, or plain text)."""
        return self.extract(load_cv_text(path))