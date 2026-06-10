"""
Pipeline RAG con Supabase pgvector y Google Generative AI Embeddings.

¿Qué es un embedding?
---------------------
Un embedding es un vector de números reales (aquí, 3072 floats) que representa
el *significado semántico* de un texto. El modelo gemini-embedding-001 de Google
aprende a mapear textos al mismo punto del espacio si tratan sobre lo mismo,
aunque usen palabras distintas. "precio de la electricidad" y "tarifa del kWh"
quedan cerca; "nivel del embalse" y "resolución CREG" quedan lejos.

¿Por qué similitud coseno?
--------------------------
La similitud coseno mide el ángulo entre dos vectores, no su distancia absoluta.
Esto la hace robusta ante diferencias de longitud: un párrafo y un artículo sobre
el mismo tema producen vectores en la misma dirección aunque tengan magnitudes
distintas. Rango: 1.0 = idénticos, 0.0 = sin relación semántica.

¿Por qué google.genai directamente y no langchain-google-genai?
----------------------------------------------------------------
langchain-google-genai v2 y el SDK google-genai usan v1beta por defecto, pero
text-embedding-004 sólo está disponible en la API v1 (estable). Forzar v1 con
http_options={"api_version": "v1"} en el cliente es la solución definitiva.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from app.core.config import get_settings
from app.core.database import get_supabase_client

settings = get_settings()

_TABLE = "energy_documents"


@dataclass
class SearchResult:
    """Documento recuperado junto con su score de relevancia."""

    id: str
    content: str
    metadata: dict[str, Any]
    score: float  # similitud coseno [0, 1] para semantic; RRF score para hybrid


class EnergyVectorStore:
    """
    Abstracción sobre energy_documents en Supabase.

    Flujo de indexación:
        texto → embed → INSERT en Supabase

    Flujo de búsqueda:
        query → embed → match_documents RPC → resultados ordenados
    """

    def __init__(self) -> None:
        # http_options fuerza la API v1 (estable). Sin esto el SDK usa v1beta
        # y los modelos de embeddings devuelven 404 NOT_FOUND.
        self._genai = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"api_version": "v1"},
        )
        self._embed_model = "models/gemini-embedding-001"

    # -------------------------------------------------------------------------
    # Indexación
    # -------------------------------------------------------------------------

    async def add_documents(self, documents: list[dict]) -> list[str]:
        """
        Genera embeddings en batch e inserta documentos en Supabase.

        Args:
            documents: lista de dicts con:
                - content  (str) — texto del fragmento a indexar
                - metadata (dict) — source, doc_type, title, created_at, …

        Returns:
            Lista de UUIDs asignados a cada documento.
        """
        texts = [doc["content"] for doc in documents]

        # Una sola llamada genera todos los embeddings en batch.
        # client.models.embed_content es síncrono → asyncio.to_thread evita bloquear el loop.
        embeddings = await asyncio.to_thread(
            lambda: [
                list(e.values)
                for e in self._genai.models.embed_content(
                    model=self._embed_model,
                    contents=texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                ).embeddings
            ]
        )

        doc_ids: list[str] = []
        records: list[dict] = []
        for doc, embedding in zip(documents, embeddings):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            records.append({
                "id": doc_id,
                "content": doc["content"],
                # supabase-py serializa la lista de floats como JSON array,
                # que PostgREST castea automáticamente al tipo vector(768).
                "embedding": embedding,
                "metadata": doc.get("metadata", {}),
            })

        # Supabase-py usa un cliente HTTP síncrono (httpx sync).
        # asyncio.to_thread() lo ejecuta en el thread-pool del event loop
        # para no bloquear el hilo principal de la app.
        await asyncio.to_thread(
            lambda: self._db().table(_TABLE).insert(records).execute()
        )
        return doc_ids

    # -------------------------------------------------------------------------
    # Búsqueda semántica (dense)
    # -------------------------------------------------------------------------

    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[SearchResult]:
        """
        Búsqueda semántica por similitud coseno vía pgvector.

        Llama a la función match_documents() de PostgreSQL, que usa el índice
        ivfflat para encontrar los k vectores más cercanos al embedding de la
        query. El parámetro filter restringe por campos de metadata, ej:
            filter={"source": "xm"}  →  sólo documentos de XM.

        Args:
            query:  pregunta o frase en lenguaje natural.
            k:      número de resultados a devolver.
            filter: dict de metadata para filtrar (jsonb containment @>).
        """
        query_embedding = await asyncio.to_thread(
            lambda: list(
                self._genai.models.embed_content(
                    model=self._embed_model,
                    contents=query,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                ).embeddings[0].values
            )
        )

        response = await asyncio.to_thread(
            lambda: self._db().rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_count": k,
                    "filter": filter or {},
                },
            ).execute()
        )

        return [
            SearchResult(
                id=row["id"],
                content=row["content"],
                metadata=row["metadata"],
                score=float(row["similarity"]),
            )
            for row in (response.data or [])
        ]

    # -------------------------------------------------------------------------
    # Búsqueda híbrida (dense + sparse + RRF)
    # -------------------------------------------------------------------------

    async def hybrid_search(self, query: str, k: int = 5) -> list[SearchResult]:
        """
        Combina búsqueda semántica (dense) con keyword matching (sparse)
        usando Reciprocal Rank Fusion (RRF).

        RRF (Cormack et al., 2009):
            score_rrf(d) = Σ_i  1 / (k_rrf + rank_i(d))

        k_rrf = 60 es el valor canónico. RRF es preferido sobre suma ponderada
        de scores porque sólo usa el orden relativo, evitando problemas cuando
        las escalas de similitud coseno (~0-1) y de BM25/ilike son distintas.

        Dense  → captura synonyms y paráfrasis ("precio kWh" ≈ "tarifa energía")
        Sparse → asegura que términos exactos raros ("CREG", "XM") no se pierdan
        """
        fetch_k = k * 3  # pool ampliado para combinar con más contexto

        # Rama dense
        dense_results = await self.similarity_search(query, k=fetch_k)

        # Rama sparse: keyword ilike con el primer término sustantivo (≥ 4 chars).
        # Para producción, reemplazar con ts_vector/plainto_tsquery de PostgreSQL.
        pivot = next((w for w in query.lower().split() if len(w) >= 4), query)
        sparse_response = await asyncio.to_thread(
            lambda: self._db()
            .table(_TABLE)
            .select("id, content, metadata")
            .ilike("content", f"%{pivot}%")
            .limit(fetch_k)
            .execute()
        )
        sparse_docs = sparse_response.data or []

        # Construir tabla de ranks por documento
        ranks: dict[str, dict[str, Any]] = {}
        for i, r in enumerate(dense_results):
            ranks[r.id] = {
                "content": r.content,
                "metadata": r.metadata,
                "dense_rank": i + 1,
                "sparse_rank": None,
            }
        for i, doc in enumerate(sparse_docs):
            if doc["id"] in ranks:
                ranks[doc["id"]]["sparse_rank"] = i + 1
            else:
                ranks[doc["id"]] = {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "dense_rank": None,
                    "sparse_rank": i + 1,
                }

        def _rrf(dense: int | None, sparse: int | None, k_rrf: int = 60) -> float:
            return sum(
                1.0 / (k_rrf + rank) for rank in (dense, sparse) if rank is not None
            )

        sorted_results = sorted(
            [
                SearchResult(
                    id=doc_id,
                    content=data["content"],
                    metadata=data["metadata"],
                    score=_rrf(data["dense_rank"], data["sparse_rank"]),
                )
                for doc_id, data in ranks.items()
            ],
            key=lambda r: r.score,
            reverse=True,
        )
        return sorted_results[:k]

    # -------------------------------------------------------------------------
    # Eliminación
    # -------------------------------------------------------------------------

    async def delete_document(self, doc_id: str) -> bool:
        """Elimina un documento por UUID. Devuelve True si existía."""
        response = await asyncio.to_thread(
            lambda: self._db()
            .table(_TABLE)
            .delete()
            .eq("id", doc_id)
            .execute()
        )
        return len(response.data or []) > 0

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------

    def _db(self):
        # Reutiliza el cliente Supabase singleton definido en database.py
        return get_supabase_client()
