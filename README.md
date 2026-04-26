<div align="center">

# 🌼 Hansel

**Autonomous job search agent for the Swiss market**

*Reads your CV · Finds real jobs · Ranks by fit · Drafts personalized emails*

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-142%20passing-brightgreen)
![Local](https://img.shields.io/badge/runs-100%25%20local-orange)

</div>

---

Hansel is an autonomous job search agent that reads your CV, finds matching 
jobs on Swiss job boards, scores them against your profile, and drafts 
personalized application emails — all running locally on your laptop with 
[Ollama](https://ollama.com/), no paid APIs, no data leaving your machine.

## 🌸 What it does

1. **Extracts your profile** from a PDF or Markdown CV (regex for contact 
   data, LLM for semantic understanding).
2. **Generates diverse search queries** calibrated to your seniority level.
3. **Searches in parallel** across three Swiss job sources:
   - [Adzuna Switzerland](https://www.adzuna.ch) — general board, API
   - [Arbeitnow](https://www.arbeitnow.com) — remote-friendly, free API
   - [SwissDevJobs](https://swissdevjobs.ch) — tech-specialist, CHF salaries
4. **Ranks every listing** with a retrieve-and-rerank pipeline: embeddings 
   for fast retrieval, LLM for precise scoring with per-listing rationale.
5. **Drafts personalized emails** with a two-step generation (draft + 
   self-critique) and three-layer hallucination defense.
6. **Web UI** with real-time progress streaming, animated Switzerland map, 
   and keyword particles — or use the CLI.

## 🌸 Architecture

```
CV (PDF/MD)
    │
    ▼
CVExtractor ──────────────────────────────────────────────────┐
    │ CVProfile                                                │
    ▼                                                         │
QueryGenerator                                                │
    │ SearchStrategy (3-5 queries)                            │
    ▼                                                         │
JobSearchOrchestrator ──► AdzunaAdapter                       │
    │                ──► ArbeitnowAdapter                     │
    │                ──► SwissDevJobsAdapter                  │
    │ deduplicated JobListings                                 │
    ▼                                                         │
JobMatcher                                                    │
    │ ──► EmbeddingScorer (nomic-embed-text)                  │
    │ ──► LLMReranker (Qwen 2.5 7B) ◄────────────────────────┘
    │ ScoredListings (ranked)
    ▼
EmailGenerator
    │ ──► draft → critique → validator → fact-check
    │ GeneratedEmail | EmailGenerationSkipped
    ▼
HanselResult
```

Design principles:

- **Adapter pattern** for job sources — adding a new board means one class.
- **Dependency injection** throughout — every component is replaceable 
  in the `HanselAgent` constructor, making the pipeline trivially testable.
- **Graceful degradation** — if a source fails, others continue. If one 
  email generation fails, the rest proceed.
- **Local-first** — no external LLM APIs. Runs on CPU.

## 🌸 Technical decisions

Every non-trivial choice is documented as an Architecture Decision Record:

| ADR | Decision |
|---|---|
| [001](docs/decisions/001-hybrid-cv-extraction.md) | Hybrid regex + LLM for CV extraction |
| [002](docs/decisions/002-llm-query-generation.md) | LLM-generated diverse search queries |
| [003](docs/decisions/003-no-seniority-in-queries.md) | Don't trust job-board keyword matching |
| [004](docs/decisions/004-drop-jooble.md) | Drop Jooble: no Swiss coverage |
| [005](docs/decisions/005-adzuna-country-param.md) | Adzuna: skip redundant country name |
| [006](docs/decisions/006-embedding-retrieval-strategy.md) | Embeddings for retrieval, LLM for ranking |
| [007](docs/decisions/007-score-normalization.md) | Normalize scores across rerank tiers |
| [008](docs/decisions/008-email-hallucination-defense.md) | Three-layer hallucination defense |
| [009](docs/decisions/009-swissdevjobs-adapter.md) | SwissDevJobs over jobs.ch |

## 🌸 Quickstart

### Option A — Docker (recommended)

```bash
git clone https://github.com/Alexiia99/Hansel.git
cd Hansel

# Configure credentials
cp .env.example .env
# Edit .env: fill in ADZUNA_APP_ID and ADZUNA_APP_KEY

# Start Hansel + Ollama
docker compose up -d

# Pull models (one-time, ~5 GB)
docker exec hansel-ollama ollama pull qwen2.5:7b-instruct
docker exec hansel-ollama ollama pull nomic-embed-text

# Open the web UI
open http://localhost:8000
```

### Option B — Local

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv), 
[Ollama](https://ollama.com/) running locally, Adzuna API credentials 
([free tier](https://developer.adzuna.com), 250 calls/month).

```bash
git clone https://github.com/Alexiia99/Hansel.git
cd Hansel

uv sync

# Pull required models (one-time, ~5 GB total)
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

cp .env.example .env
# Edit .env: fill in ADZUNA_APP_ID and ADZUNA_APP_KEY

# Web UI
uvicorn src.hansel.api.main:app --reload --env-file .env
# → open http://localhost:8000

# Or CLI
uv run python -m hansel \
    --cv tests/fixtures/cv_junior_tech.md \
    --location Switzerland \
    --emails-top-n 3
```

## 🌸 Python API

```python
from hansel import HanselAgent
from hansel.cv.schemas import Seniority
from hansel.email_gen import EmailLanguage

agent = HanselAgent()

result = await agent.find_jobs(
    cv_path="my_cv.pdf",
    seniority=Seniority.JUNIOR,
    preferred_location="Switzerland",
    email_language=EmailLanguage.ENGLISH,
    emails_top_n=5,
)

for sl in result.ranked_listings[:5]:
    print(f"{sl.final_score:.2f}  {sl.listing.title}  @ {sl.listing.company}")
    if sl.llm_score:
        print(f"    → {sl.llm_score.rationale}")

for email in result.emails:
    print(email.subject)
    print(email.body)
```

Step-by-step access (e.g. show rankings before generating emails):

```python
profile = await agent.extract_cv("my_cv.pdf", Seniority.JUNIOR)
listings = await agent.search(profile, "Switzerland")
ranked  = await agent.rank(profile, listings)
emails  = await agent.generate_emails(profile, ranked[:5])
```

## 🌸 Testing

```bash
uv run pytest tests/ -v
```

142 unit tests covering CV regex parsing, all three job source adapters, 
orchestrator fan-out/dedup/resilience, seniority filter heuristics, 
email hallucination validator, and HTTP mocking with `respx`.

## 🌸 Caveats

- **Emails are drafts, not auto-send.** Always review before sending. 
  Local 7B LLMs can produce subtle stylistic quirks. The hallucination 
  defense catches factual fabrications, but tone benefits from a human pass.
- **First ranking run takes ~5 min on CPU.** The LLM reranks 10 listings 
  sequentially (~25 s each). On GPU this is mostly free.
- **Adzuna free tier: 250 calls/month.** Each run uses 4–5. Enough for 
  personal use, not for teams.
- **SwissDevJobs API is undocumented.** The `/api/jobsLight` endpoint may 
  change without notice.

## 🌸 Roadmap

- [x] Hybrid CV extraction
- [x] LLM-generated diverse search queries  
- [x] Multi-source orchestration with dedup (Adzuna + Arbeitnow + SwissDevJobs)
- [x] Retrieve-and-rerank matcher
- [x] Personalized email generation with hallucination defense
- [x] CLI + Python API
- [x] FastAPI backend + web UI with SSE progress streaming
- [x] Docker Compose
- [x] 142 unit tests
- [x] Application tracker (SQLite)


## 🌸 License

MIT — see [LICENSE](LICENSE).

---

<br>

## 🌻 En español

Hansel es un agente autónomo de búsqueda de empleo para el mercado suizo. 
Lee tu CV, busca ofertas reales en portales suizos, las puntúa contra tu 
perfil y redacta emails de aplicación personalizados. Todo corre 
localmente en tu portátil, sin APIs externas de pago, sin datos fuera 
de tu máquina.

**Por qué existe:** buscar trabajo en un país extranjero implica leer 
decenas de ofertas, evaluarlas mentalmente una a una, y escribir emails 
adaptados que acaban sonando todos igual. Hansel automatiza las partes 
repetitivas sin inventarse tu experiencia.

**Cómo funciona:**

1. Extrae tu perfil del CV (regex para datos de contacto, LLM para 
   comprensión semántica).
2. Genera consultas diversas calibradas a tu seniority.
3. Busca en paralelo en tres fuentes suizas: Adzuna, Arbeitnow y 
   SwissDevJobs (con salarios en CHF).
4. Puntúa cada oferta con embeddings locales + reranking LLM con 
   justificación por oferta.
5. Redacta emails personalizados con tres capas defensivas contra 
   alucinaciones del LLM.
6. UI web con progreso en tiempo real, mapa animado de Suiza y 
   partículas de keywords.

**Stack:** Python 3.11, LangChain, Ollama, Qwen 2.5 7B, 
`nomic-embed-text`, FastAPI, Pydantic, httpx, Docker.

**Todo razonado:** cada decisión de diseño está documentada en 
[docs/decisions/](docs/decisions/) como ADR. 9 decisiones, todas 
motivadas por problemas reales encontrados durante el desarrollo.

```bash
# Docker (recomendado)
docker compose up -d
docker exec hansel-ollama ollama pull qwen2.5:7b-instruct
docker exec hansel-ollama ollama pull nomic-embed-text
# Abre http://localhost:8000

# O local
uv sync && uvicorn src.hansel.api.main:app --reload --env-file .env
```

---

<br>

<div align="center">

**🌸 Built by [Alexia Herrador Jiménez](https://linkedin.com/in/alexia-herrador-jimenez)**

[LinkedIn](https://linkedin.com/in/alexia-herrador-jimenez) · 
[alexiahj111@gmail.com](mailto:alexiahj111@gmail.com) · 
[GitHub](https://github.com/Alexiia99)

</div>
