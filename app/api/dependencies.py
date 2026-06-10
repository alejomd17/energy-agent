"""Dependencias compartidas inyectadas vía FastAPI Depends()."""

from functools import lru_cache

from app.core.vector_store import EnergyVectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> EnergyVectorStore:
    """Instancia singleton de EnergyVectorStore reutilizada en todos los requests."""
    return EnergyVectorStore()
