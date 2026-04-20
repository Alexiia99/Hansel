"""Main CV extraction pipeline: regex + LLM."""

from __future__ import annotations

from pathlib import Path

import pdfplumber
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from hansel.cv.regex_parser import extract_contact
from hansel.cv.schemas import CVProfile, CVProfileSemantic, Experience, Seniority


_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You extract structured information from CVs. You are thorough and never miss information.

CRITICAL RULES:
1. Only extract info explicitly in the CV, EXCEPT 'target_roles' (you MUST infer).
2. 'target_roles': return EXACTLY 3-5 roles. EVERY role MUST start with '{seniority_label}'. Be specific based on actual skills.
   Examples for Junior level: 'Junior Data Engineer', 'Junior Python Developer', 'Junior Backend Developer'.
   Examples for Senior level: 'Senior ML Engineer', 'Senior Backend Engineer'.
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
        if exp.end_date is None and exp.start_date:
            anchor = _find_experience_anchor(cv_lower, exp)
            if anchor != -1:
                window = cv_lower[anchor : anchor + 100]
                if any(kw in window for kw in _ONGOING_KEYWORDS):
                    exp = exp.model_copy(update={"end_date": "current"})
        fixed.append(exp)
    
    return fixed


def _find_experience_anchor(cv_lower: str, exp: Experience) -> int:
    """Find position in CV where this experience's date range appears."""
    months_en = {
        "01": "january", "02": "february", "03": "march", "04": "april",
        "05": "may", "06": "june", "07": "july", "08": "august",
        "09": "september", "10": "october", "11": "november", "12": "december",
    }
    months_es = {
        "01": "enero", "02": "febrero", "03": "marzo", "04": "abril",
        "05": "mayo", "06": "junio", "07": "julio", "08": "agosto",
        "09": "septiembre", "10": "octubre", "11": "noviembre", "12": "diciembre",
    }
    
    if "-" in exp.start_date:
        year, _, month = exp.start_date.partition("-")
        month_name_en = months_en.get(month.zfill(2), "")
        month_name_es = months_es.get(month.zfill(2), "")
        
        for phrase in [
            f"{month_name_en} {year}",
            f"{month_name_es} {year}",
            f"{month_name_es} de {year}",
        ]:
            if phrase and phrase in cv_lower:
                return cv_lower.find(phrase)
    
    year = exp.start_date.split("-")[0]
    if year in cv_lower:
        return cv_lower.find(year)
    
    return -1


class CVExtractor:
    """Extracts structured CV data using a hybrid regex + LLM pipeline."""
    
    def __init__(self, model: str = "qwen2.5:7b-instruct", temperature: float = 0.0):
        self.llm = ChatOllama(model=model, temperature=temperature)
        self._chain = _EXTRACTION_PROMPT | self.llm.with_structured_output(CVProfileSemantic)
    
    def extract(self, cv_text: str, seniority: Seniority = Seniority.JUNIOR) -> CVProfile:
        """Extract full profile from CV text.
        
        Args:
            cv_text: Raw text of the CV.
            seniority: Target seniority level for generated target_roles.
                       Defaults to Junior.
        
        Returns:
            Complete CVProfile with contact, semantic data, and user-provided seniority.
        """
        contact = extract_contact(cv_text)
        semantic: CVProfileSemantic = self._chain.invoke({
            "cv_text": cv_text,
            "seniority_label": seniority.label,
        })
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
            seniority=seniority,
        )
    
    def extract_from_file(
        self, path: str | Path, seniority: Seniority = Seniority.JUNIOR
    ) -> CVProfile:
        """Extract profile from a CV file (PDF, Markdown, or plain text)."""
        return self.extract(load_cv_text(path), seniority=seniority)