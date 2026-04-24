# src/hansel/llm.py

from __future__ import annotations

import os

from langchain_ollama import ChatOllama, OllamaEmbeddings


def _ollama_base_url() -> str:
    """Read Ollama host from environment, defaulting to localhost.

    WHY: inside Docker, Ollama runs as a separate container reachable at
    http://ollama:11434, not localhost. OLLAMA_HOST lets us configure this
    without changing code.
    """
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


def make_chat_ollama(model: str, temperature: float = 0.0) -> ChatOllama:
    """Construct a ChatOllama instance with the correct host."""
    return ChatOllama(model=model, temperature=temperature, base_url=_ollama_base_url())


def make_ollama_embeddings(model: str) -> OllamaEmbeddings:
    """Construct OllamaEmbeddings with the correct host."""
    return OllamaEmbeddings(model=model, base_url=_ollama_base_url())