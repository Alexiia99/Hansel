# src/hansel/api/routes/jobs.py

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from hansel.agent import HanselAgent
from hansel.cv.schemas import Seniority
from hansel.email_gen.schemas import GeneratedEmail

router = APIRouter(prefix="/api")


def _make_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event string per the SSE spec (RFC 8895)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/run")
async def run_pipeline(
    cv: UploadFile = File(...),
    location: str = Form(default="Switzerland"),
    seniority: str = Form(default="junior"),
    generate_emails: bool = Form(default=True),
    emails_top_n: int = Form(default=3),
):
    """Run the full Hansel pipeline and stream progress via SSE.

    Returns a StreamingResponse with content-type text/event-stream.
    Events: progress | result | error | done.
    """
    # UploadFile is tied to the request lifecycle — read bytes before
    # entering the async generator or the file handle may be closed.
    cv_bytes = await cv.read()
    cv_filename = cv.filename or "cv.pdf"

    async def event_stream():
        with tempfile.NamedTemporaryFile(
            suffix=Path(cv_filename).suffix, delete=False
        ) as tmp:
            tmp.write(cv_bytes)
            tmp_path = Path(tmp.name)

        try:
            # A Queue decouples the ProgressCallback (sync, called from inside
            # the agent) from the async generator that yields SSE strings.
            progress_events: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

            def progress_callback(step: str, message: str) -> None:
                progress_events.put_nowait((step, message))

            yield _make_event("progress", {"step": "start", "message": "Pipeline started"})

            # create_task lets us await the agent and drain the queue
            # concurrently — without it, progress events would only appear
            # after the entire pipeline finishes.
            agent_task = asyncio.create_task(
                HanselAgent(progress_callback=progress_callback).find_jobs(
                    cv_path=tmp_path,
                    seniority=Seniority(seniority),
                    preferred_location=location,
                    generate_emails=generate_emails,
                    emails_top_n=emails_top_n,
                )
            )

            while not agent_task.done():
                try:
                    step, message = await asyncio.wait_for(
                        progress_events.get(), timeout=0.1  # más agresivo
                    )
                    yield _make_event("progress", {"step": step, "message": message})
                except asyncio.TimeoutError:
                    # Drena todo lo acumulado antes del heartbeat — evita que los
                    # eventos de progreso lleguen en ráfaga al final del pipeline.
                    while not progress_events.empty():
                        step, message = progress_events.get_nowait()
                        yield _make_event("progress", {"step": step, "message": message})
                    yield ": heartbeat\n\n"

            while not progress_events.empty():
                step, message = progress_events.get_nowait()
                yield _make_event("progress", {"step": step, "message": message})

            result = agent_task.result()

            if not result.success:
                yield _make_event("error", {"errors": result.errors})
                return

            listings_data = [
                {
                    "title": sl.listing.title,
                    "company": sl.listing.company,
                    "location": sl.listing.location,
                    "url": str(sl.listing.url),
                    "score": round(sl.final_score, 2),
                    "rationale": sl.llm_score.rationale if sl.llm_score else None,
                    "tags": sl.listing.tags or [],
                }
                for sl in result.ranked_listings[:10]
            ]

            emails_data = [
                {
                    "skipped": False,
                    "subject": email.subject,
                    "body": email.body,
                    "word_count": email.word_count,
                }
                if isinstance(email, GeneratedEmail)
                else {
                    "skipped": True,
                    "reason": email.reason,
                }
                for email in (result.emails or [])
            ]

            yield _make_event("result", {
                "listings": listings_data,
                "emails": emails_data,
                "total_raw": len(result.raw_listings or []),
                "total_ranked": len(result.ranked_listings or []),
            })

            yield _make_event("done", {"message": "Pipeline complete"})

        except Exception as e:
            yield _make_event("error", {"errors": [str(e)]})

        finally:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Prevent nginx and other reverse proxies from buffering SSE chunks.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )