"""
Router para análisis en streaming via SSE (Server-Sent Events).

¿Qué es SSE?
------------
Server-Sent Events es un protocolo HTTP estándar (RFC 8898) donde el servidor
mantiene la conexión abierta y envía eventos unidireccionales al cliente.
Cada evento tiene el formato:

    data: payload\n\n

El cliente (browser o httpx) los recibe como un stream de líneas de texto.

¿Por qué SSE para LLMs?
------------------------
Los modelos de lenguaje generan tokens uno a uno. Sin streaming, el usuario
espera segundos (o decenas de segundos) antes de ver cualquier respuesta.
Con SSE, el primer token llega en ~300ms y el texto aparece progresivamente,
igual que en ChatGPT. Es la forma estándar de exponer la generación de LLMs
en APIs REST sin necesidad de WebSockets (que son bidireccionales y más
complejos de mantener).

SSE vs WebSockets:
    SSE          → unidireccional, HTTP puro, reconexión automática
    WebSockets   → bidireccional, protocolo propio — para chat interactivo
"""

import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.api.dependencies import get_vector_store
from app.core.config import get_settings
from app.core.vector_store import EnergyVectorStore

settings = get_settings()

router = APIRouter()

_llm = ChatGroq(
    model=settings.groq_model,
    groq_api_key=settings.groq_api_key,
    temperature=0.3,
    max_retries=2,
)


async def _sse_generator(query: str, store: EnergyVectorStore):
    """
    Generador async que produce eventos SSE.

    1. Recupera contexto del RAG (Supabase pgvector).
    2. Construye prompt con el contexto + query del usuario.
    3. Streamea la respuesta de Groq token por token.
    4. Emite [DONE] al terminar para que el cliente cierre la conexión.
    """
    # 1. Contexto RAG
    rag_results = await store.similarity_search(query, k=3)
    context = (
        "\n\n".join(
            f"[{r.metadata.get('source', '?').upper()}] {r.content}"
            for r in rag_results
        )
        or "Sin contexto previo disponible."
    )

    # 2. Emitir fuentes usadas como primer evento
    sources = [r.metadata.get("source", "?") for r in rag_results]
    yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

    # 3. Prompt de análisis
    prompt = f"""Eres un experto analista del sector energético colombiano.
Responde la siguiente consulta usando el contexto disponible.

=== CONTEXTO (base de conocimiento RAG) ===
{context}

=== CONSULTA ===
{query}

Responde en español de forma técnica y concisa."""

    # 4. Streamear tokens de Groq
    async for chunk in _llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield f"data: {json.dumps({'type': 'token', 'text': chunk.content})}\n\n"

    # 5. Señal de fin — el cliente debe cerrar la conexión al recibirla
    yield "data: [DONE]\n\n"


@router.get("/stream")
async def analyze_stream(
    query: str = Query(..., min_length=3, description="Consulta sobre el sector energético"),
    store: EnergyVectorStore = Depends(get_vector_store),
) -> StreamingResponse:
    """
    Endpoint SSE que responde consultas sobre el sector energético con streaming.

    Recupera contexto del vector store (RAG) y genera la respuesta con Groq
    enviando tokens a medida que el modelo los produce.

    El cliente debe consumir el stream leyendo líneas que empiezan con 'data: '.
    La señal 'data: [DONE]' indica que el stream terminó.
    """
    return StreamingResponse(
        _sse_generator(query, store),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Evita que nginx bufferice la respuesta
            "Connection": "keep-alive",
        },
    )
