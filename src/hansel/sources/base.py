"""Abstract base class for job sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hansel.sources.schemas import JobListing


class JobSourceAdapter(ABC):
    """Abstract contract every job source must fulfill.
    
    Concrete implementations (AdzunaAdapter, JoobleAdapter, etc.) translate 
    provider-specific API responses into our unified JobListing schema.
    
    Responsibilities of every adapter:
      - Handle provider-specific authentication (API keys, headers)
      - Call the provider's API with the right parameters
      - Handle errors gracefully (network, rate limits, malformed responses)
      - Normalize the raw response into JobListing objects
      - Return an empty list on failure — NEVER crash the agent
    """
    
    name: str  # Human-readable name, set by subclass
    
    @abstractmethod
    async def search(
        self,
        keywords: str,
        location: str | None = None,
        limit: int = 20,
    ) -> list[JobListing]:
        """Search for job postings matching keywords and location.
        
        Args:
            keywords: Search terms (e.g., "Junior Data Engineer").
            location: City, country, or 'remote'. If None, search globally.
            limit: Maximum number of results to return.
        
        Returns:
            List of JobListing objects. Empty if no matches or if the 
            provider failed (errors are logged internally, never raised).
        """
        ...
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"