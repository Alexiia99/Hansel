# src/hansel/api/main.py

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from hansel.api.routes.jobs import router as jobs_router

app = FastAPI(
    title="Hansel",
    description="Autonomous job search agent for Swiss market",
    version="0.1.0",
)

# Allow the frontend (served separately during dev) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# Serve the frontend as static files so both API and UI run on the same port.
# Only mounted if the frontend directory exists — safe to omit during dev.
frontend_path = Path(__file__).parent.parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")