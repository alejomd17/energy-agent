"""
Conexiones async a las bases de datos del sistema.

MongoDB (motor)
---------------
Almacena documentos no estructurados: artículos crudos de scrapers,
historial de conversaciones y logs de ejecución de agentes.

Supabase (pgvector)
-------------------
Almacena embeddings vectoriales para el pipeline RAG y datos
estructurados relacionales (resoluciones CREG, precios XM, etc.).
"""

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from supabase import Client, create_client

from .config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------

_mongo_client: AsyncIOMotorClient | None = None


async def get_mongo_client() -> AsyncIOMotorClient:
    """Devuelve el cliente MongoDB, creándolo en el primer llamado."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
    return _mongo_client


async def get_mongo_db() -> AsyncIOMotorDatabase:
    """Devuelve la base de datos principal 'energy_agent' en MongoDB."""
    client = await get_mongo_client()
    return client["energy_agent"]


async def close_mongo_connection() -> None:
    """Cierra la conexión MongoDB. Debe llamarse en el shutdown de FastAPI."""
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

_supabase_client: Client | None = None


def get_supabase_client() -> Client:
    """
    Devuelve el cliente Supabase con service role key.

    Se usa service_role_key (no anon_key) para operaciones del backend
    que necesitan saltarse RLS (Row Level Security).
    """
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
    return _supabase_client
