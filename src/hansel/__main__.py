"""Command-line interface: python -m hansel.

Example:
    python -m hansel --cv my_cv.pdf --location Switzerland
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from hansel.agent import HanselAgent, _print_progress
from hansel.cv.schemas import Seniority
from hansel.email_gen import GeneratedEmail


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hansel",
        description="Autonomous job search agent",
    )
    parser.add_argument("--cv", type=Path, required=True, help="Path to CV (pdf/md/txt)")
    parser.add_argument(
        "--seniority",
        choices=[s.value for s in Seniority],
        default=Seniority.JUNIOR.value,
        help="Seniority level (default: junior)",
    )
    parser.add_argument(
        "--location", default="Switzerland",
        help="Preferred location (default: Switzerland)",
    )
    parser.add_argument(
        "--no-emails", action="store_true",
        help="Skip email generation (faster, ranking only)",
    )
    parser.add_argument(
        "--emails-top-n", type=int, default=5,
        help="How many top listings get emails (default: 5)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    return parser


async def _main() -> int:
    args = _build_parser().parse_args()
    
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s: %(message)s",
    )
    
    agent = HanselAgent(progress_callback=_print_progress)
    
    result = await agent.find_jobs(
        cv_path=args.cv,
        seniority=Seniority(args.seniority),
        preferred_location=args.location,
        generate_emails=not args.no_emails,
        emails_top_n=args.emails_top_n,
    )
    
    if not result.success:
        print("\n❌ Pipeline did not complete successfully.")
        for err in result.errors:
            print(f"   • {err}")
        return 1
    
    # Show top 5 listings
    print(f"\n{'='*70}\nTOP {min(5, len(result.ranked_listings))} MATCHES\n{'='*70}")
    for i, sl in enumerate(result.ranked_listings[:5], 1):
        print(f"\n#{i}  score {sl.final_score:.2f}  |  {sl.listing.title}")
        print(f"   🏢 {sl.listing.company}  📍 {sl.listing.location}")
        if sl.llm_score:
            print(f"   💬 {sl.llm_score.rationale}")
        print(f"   🔗 {sl.listing.url}")
    
    # Show emails
    if result.emails:
        print(f"\n{'='*70}\nGENERATED EMAILS\n{'='*70}")
        for i, email in enumerate(result.emails, 1):
            if isinstance(email, GeneratedEmail):
                print(f"\n--- Email {i} ({email.word_count} words) ---")
                print(f"Subject: {email.subject}\n")
                print(email.body)
            else:
                print(f"\n--- Email {i}: SKIPPED ---")
                print(email.reason)
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))