Design principles:

- **Adapter pattern** for job sources — adding a new board means one class.
- **Dependency injection** throughout — every component is replaceable 
  in the `HanselAgent` constructor, making the pipeline trivially testable.
- **Graceful degradation** — if a source fails, others continue. If one 
  email generation fails, the rest proceed.
- **Local-first** — no external LLM APIs. Runs on CPU.

## 🌻 Technical decisions

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

## 🏵️ Quickstart

**Prerequisites:**
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Ollama](https://ollama.com/) running locally
- Adzuna API credentials (free tier, 250 calls/month): https://developer.adzuna.com

**Setup:**

```bash
git clone https://github.com/Alexiia99/Hansel.git
cd Hansel

# Install dependencies
uv sync

# Pull required Ollama models (one-time, ~5 GB total)
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text

# Configure API credentials
cp .env.example .env
# Edit .env and fill in ADZUNA_APP_ID and ADZUNA_APP_KEY

# Run the agent
uv run python -m hansel \
    --cv tests/fixtures/cv_junior_tech.md \
    --location Switzerland \
    --emails-top-n 3
```

## 🌸 Using the Python API

For notebooks or custom workflows:

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

Need step-by-step access (e.g., to display ranked listings before 
generating emails)?

```python
profile = await agent.extract_cv("my_cv.pdf", Seniority.JUNIOR)
listings = await agent.search(profile, "Switzerland")
ranked = await agent.rank(profile, listings)
emails = await agent.generate_emails(profile, ranked[:5])
```

## 🌻 Testing

```bash
uv run pytest tests/ -v
```

67 unit tests covering the CV regex parser, job source adapters, the 
orchestrator (fan-out, dedup, resilience, parallelism), and HTTP mocking 
with `respx`.

## 🌼 Roadmap

Shipped:

- Hybrid CV extraction
- LLM-generated diverse search queries
- Multi-source orchestration with dedup
- Retrieve-and-rerank matcher
- Personalized email generation with hallucination defense
- CLI + Python API

Next:

- `jobs.ch` adapter (largest Swiss job board, scraping + rate-limit respect)
- FastAPI backend + lightweight web UI
- Docker Compose for reproducible setup
- Matcher and email-generator unit tests
- Application tracker (SQLite + history)

## 🏵️ Caveats

- **Emails are drafts, not auto-send**. Always review before sending. 
  Local 7B LLMs can produce subtle stylistic quirks. The hallucination 
  defense catches factual fabrications, but tone and phrasing benefit 
  from a human pass.
- **First ranking run takes ~5 min on CPU**. The LLM reranks 10 listings 
  sequentially at ~25 s each. On GPU the rerank is mostly free.
- **Adzuna free tier is 250 calls/month**. Each run uses 4–5. Enough for 
  personal use, not for teams.

## 🌸 License

MIT — see [LICENSE](LICENSE).

---

<br>

## 🌻 En español

Hansel es un agente autónomo de búsqueda de empleo. Lee tu CV, busca 
ofertas reales en portales suizos, las puntúa contra tu perfil y te 
redacta emails de aplicación personalizados. Todo corre localmente en tu 
portátil, sin APIs externas de pago.

**Por qué existe:** buscar trabajo en un país extranjero implica leer 
decenas de ofertas, evaluarlas mentalmente una a una, y escribir emails 
adaptados que acaban sonando todos igual. Hansel automatiza las partes 
repetitivas sin inventarse tu experiencia.

**Cómo funciona, en corto:**

1. Extrae tu perfil del CV (regex para datos de contacto, LLM para 
   comprensión semántica).
2. Genera consultas diversas calibradas a tu seniority.
3. Busca en múltiples portales en paralelo (Adzuna Suiza, Arbeitnow) 
   con rate limiting, reintentos exponenciales, cache y deduplicación.
4. Puntúa cada oferta con una pipeline *retrieve-and-rerank*: embeddings 
   locales para filtrado rápido, LLM para ranking preciso con 
   justificación por oferta.
5. Redacta emails personalizados con generación en dos pasos (borrador 
   + autocrítica) y tres capas defensivas contra alucinaciones del LLM.

**Stack técnico:** Python 3.11, LangChain, Ollama, Qwen 2.5 7B, 
`nomic-embed-text`, Pydantic, httpx, tenacity, cachetools.

**Todo razonado:** cada decisión no trivial del diseño está documentada 
en [docs/decisions/](docs/decisions/) como un Architecture Decision 
Record. Ahí están los ocho momentos en los que algo no funcionaba y 
cómo lo solucioné.

**Uso:**

```bash
git clone https://github.com/Alexiia99/Hansel.git
cd Hansel
uv sync
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
cp .env.example .env  # rellena las credenciales de Adzuna
uv run python -m hansel --cv tests/fixtures/cv_junior_tech.md --location Switzerland
```

Para más detalles, mira la sección en inglés arriba.

---

<br>

<div align="center">

**🌸 Built by [Alexia Herrador Jiménez](https://linkedin.com/in/alexia-herrador-jimenez)**

[LinkedIn](https://linkedin.com/in/alexia-herrador-jimenez) · 
[alexiahj111@gmail.com](mailto:alexiahj111@gmail.com) · 
[GitHub](https://github.com/Alexiia99)

</div>