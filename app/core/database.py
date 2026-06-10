"""
Conexiones async a las bases de datos del sistema.

MongoDB (motor)
---------------
Almacena documentos no estructurados: artículos crudos de scrapers,
historial de conversaciones y logs de ejecución de agentes.
Colecciones: raw_data, analyses, newsletters, user_feedback.

Supabase (pgvector)
-------------------
Almacena embeddings vectoriales para el pipeline RAG y datos
estructurados relacionales (resoluciones CREG, precios XM, etc.).
"""

from __future__ import annotations

from datetime import datetime
from typing import TypeVar

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from supabase import Client, create_client

from app.core.config import get_settings
from app.models.analysis import Analysis
from app.models.newsletter import Newsletter
from app.models.raw_data import RawData
from app.models.user_feedback import UserFeedback

settings = get_settings()

T = TypeVar("T")

# ---------------------------------------------------------------------------
# MongoDB — conexión
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
# MongoDB — índices
# ---------------------------------------------------------------------------

async def init_indexes() -> None:
    """
    Crea todos los índices de MongoDB. Idempotente: si ya existen, no hace nada.
    Debe invocarse una vez al arrancar la app (lifespan de FastAPI).
    """
    db = await get_mongo_db()

    # raw_data: consultas frecuentes por fuente+fecha y por estado de procesamiento
    await db["raw_data"].create_index([("source", ASCENDING), ("scraped_at", DESCENDING)])
    await db["raw_data"].create_index([("processing_status", ASCENDING)])

    # analyses: timeline desc y filtro combinado con alertas para el agente de newsletter
    await db["analyses"].create_index([("created_at", DESCENDING)])
    await db["analyses"].create_index([("has_alerts", ASCENDING), ("created_at", DESCENDING)])

    # newsletters: índice único por semana (evita duplicados) y consulta por estado de envío
    await db["newsletters"].create_index([("week_date", DESCENDING)], unique=True)
    await db["newsletters"].create_index([("sent", ASCENDING)])

    # user_feedback: agrupado por newsletter para calcular NPS y promedios
    await db["user_feedback"].create_index([("newsletter_id", ASCENDING)])


# ---------------------------------------------------------------------------
# MongoDB — helpers internos
# ---------------------------------------------------------------------------

def _to_model(doc: dict, cls: type[T]) -> T:
    """
    Convierte un documento MongoDB a modelo Pydantic.
    MongoDB usa _id (ObjectId); los modelos usan id (str).
    Hace una copia del dict para no mutar el cursor original.
    """
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    return cls.model_validate(doc)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CRUD — raw_data
# ---------------------------------------------------------------------------

async def insert_raw_data(data: RawData) -> str:
    """Inserta un documento crudo y devuelve su _id como str."""
    db = await get_mongo_db()
    result = await db["raw_data"].insert_one(data.model_dump(exclude={"id"}))
    return str(result.inserted_id)


async def get_raw_data(doc_id: str) -> RawData | None:
    db = await get_mongo_db()
    doc = await db["raw_data"].find_one({"_id": ObjectId(doc_id)})
    return _to_model(doc, RawData) if doc else None


async def find_pending_raw_data(limit: int = 50) -> list[RawData]:
    """Devuelve los documentos pendientes de procesar, del más antiguo al más nuevo."""
    db = await get_mongo_db()
    cursor = db["raw_data"].find(
        {"processing_status": "pending"},
        sort=[("scraped_at", ASCENDING)],
        limit=limit,
    )
    return [_to_model(doc, RawData) async for doc in cursor]


async def update_raw_data_status(doc_id: str, status: str) -> bool:
    """Actualiza processing_status. Devuelve True si encontró el documento."""
    db = await get_mongo_db()
    result = await db["raw_data"].update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": {"processing_status": status}},
    )
    return result.matched_count > 0


# ---------------------------------------------------------------------------
# CRUD — analyses
# ---------------------------------------------------------------------------

async def insert_analysis(data: Analysis) -> str:
    db = await get_mongo_db()
    result = await db["analyses"].insert_one(data.model_dump(exclude={"id"}))
    return str(result.inserted_id)


async def get_analysis(doc_id: str) -> Analysis | None:
    db = await get_mongo_db()
    doc = await db["analyses"].find_one({"_id": ObjectId(doc_id)})
    return _to_model(doc, Analysis) if doc else None


async def get_latest_analyses(limit: int = 10) -> list[Analysis]:
    """Devuelve los análisis más recientes, del más nuevo al más antiguo."""
    db = await get_mongo_db()
    cursor = db["analyses"].find({}, sort=[("created_at", DESCENDING)], limit=limit)
    return [_to_model(doc, Analysis) async for doc in cursor]


# ---------------------------------------------------------------------------
# CRUD — newsletters
# ---------------------------------------------------------------------------

async def insert_newsletter(data: Newsletter) -> str:
    """
    Inserta o reemplaza el newsletter de la semana dada (upsert por week_date).
    Si ya existe un newsletter para esa semana, lo regenera preservando el _id.
    """
    db = await get_mongo_db()
    doc = data.model_dump(exclude={"id"})
    result = await db["newsletters"].find_one_and_replace(
        {"week_date": data.week_date},
        doc,
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return str(result["_id"])


async def get_newsletter(doc_id: str) -> Newsletter | None:
    db = await get_mongo_db()
    doc = await db["newsletters"].find_one({"_id": ObjectId(doc_id)})
    return _to_model(doc, Newsletter) if doc else None


async def get_newsletter_by_week(week_date: datetime) -> Newsletter | None:
    """Busca el newsletter de una semana específica por su fecha lunes."""
    db = await get_mongo_db()
    doc = await db["newsletters"].find_one({"week_date": week_date})
    return _to_model(doc, Newsletter) if doc else None


async def mark_newsletter_sent(doc_id: str, sent_at: datetime) -> bool:
    """Marca el newsletter como enviado. Devuelve True si encontró el documento."""
    db = await get_mongo_db()
    result = await db["newsletters"].update_one(
        {"_id": ObjectId(doc_id)},
        {"$set": {"sent": True, "sent_at": sent_at}},
    )
    return result.matched_count > 0


# ---------------------------------------------------------------------------
# CRUD — user_feedback
# ---------------------------------------------------------------------------

async def insert_user_feedback(data: UserFeedback) -> str:
    db = await get_mongo_db()
    result = await db["user_feedback"].insert_one(data.model_dump(exclude={"id"}))
    return str(result.inserted_id)


async def get_feedback_for_newsletter(newsletter_id: str) -> list[UserFeedback]:
    """Devuelve todo el feedback recibido para un newsletter dado."""
    db = await get_mongo_db()
    cursor = db["user_feedback"].find({"newsletter_id": newsletter_id})
    return [_to_model(doc, UserFeedback) async for doc in cursor]


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

_supabase_client: Client | None = None


def get_supabase_client() -> Client:
    """
    Devuelve el cliente Supabase con service role key.
    Se usa service_role_key para saltarse RLS en operaciones del backend.
    """
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
    return _supabase_client
