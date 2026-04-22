"""Hansel: autonomous job search agent.

Main entry point: :class:`HanselAgent` orchestrates the full pipeline.
"""

from hansel.agent import (
    HanselAgent,
    HanselError,
    HanselResult,
    ProgressCallback,
    _print_progress,
)

__version__ = "0.1.0"

__all__ = [
    "HanselAgent",
    "HanselError",
    "HanselResult",
    "ProgressCallback",
    "__version__",
]


def main() -> None:
    """Stub retained for 'hansel' console script. Use 'python -m hansel' for CLI."""
    import sys
    print("Use 'python -m hansel --help' for the command-line interface.")
    sys.exit(0)