"""Shared pytest fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Directory containing test fixture files."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_cv_text(fixtures_dir: Path) -> str:
    """Raw text of the synthetic CV fixture."""
    return (fixtures_dir / "cv_junior_tech.md").read_text(encoding="utf-8")