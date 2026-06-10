"""
Agente 2: Analyzer

Flujo:
    MongoDB raw_data → RAG context (Supabase) → prompt → Groq LLM → JSON → MongoDB Analysis
"""

import json
import re
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.agents.state import EnergyAgentState
from app.core.config import get_settings
from app.core.database import get_raw_data, insert_analysis
from app.core.vector_store import EnergyVectorStore
from app.models.analysis import Alert, Analysis

settings = get_settings()

_VALID_SOURCES = {"xm", "ideam", "creg", "upme"}

_llm = ChatGroq(
    model=settings.groq_model,
    groq_api_key=settings.groq_api_key,
    temperature=0.2,
    max_retries=2,
)

_vector_store = EnergyVectorStore()


def _extract_json(text: str) -> dict:
    """Extrae JSON de la respuesta del LLM, tolerando markdown code blocks."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text)


async def analyze_data(state: EnergyAgentState) -> dict:
    """
    Combina raw_data de MongoDB con contexto RAG de Supabase para
    generar un análisis estructurado (insights + alerts) con Gemini.
    """
    # 1. Cargar documentos crudos desde MongoDB
    raw_docs = []
    for doc_id in state["raw_data_ids"]:
        doc = await get_raw_data(doc_id)
        if doc:
            raw_docs.append(doc)

    if not raw_docs:
        return {"error": "No se pudieron cargar los documentos de MongoDB.", "status": "failed"}

    # 2. Contexto histórico desde Supabase pgvector
    rag_results = await _vector_store.similarity_search(
        "precio energía Colombia embalses regulación CREG XM IDEAM", k=5
    )
    retrieved_docs = [
        {"id": r.id, "content": r.content, "metadata": r.metadata, "score": r.score}
        for r in rag_results
    ]

    # 3. Construir prompt
    raw_section = "\n\n".join(
        f"[{doc.source.upper()} — {doc.data_type}]\n"
        f"{json.dumps(doc.content, ensure_ascii=False, indent=2)}"
        for doc in raw_docs
    )
    rag_section = (
        "\n\n".join(
            f"[Contexto {r.metadata.get('source', '?').upper()} — score {r.score:.3f}]\n{r.content}"
            for r in rag_results
        )
        or "Sin contexto histórico disponible."
    )

    prompt = f"""Eres un experto analista del sector energético colombiano.

Analiza los siguientes datos de la semana del {state['week_date'].strftime('%d/%m/%Y')}:

=== DATOS CRUDOS DE ESTA SEMANA ===
{raw_section}

=== CONTEXTO HISTÓRICO (base de conocimiento RAG) ===
{rag_section}

Genera un análisis estructurado en JSON con EXACTAMENTE este formato:
{{
  "insights": [
    "Insight 1 en español...",
    "Insight 2 en español...",
    "Insight 3 en español..."
  ],
  "alerts": [
    {{
      "tipo": "precio|embalse|regulacion|demanda",
      "severidad": "low|medium|high",
      "descripcion": "Descripción detallada en español"
    }}
  ],
  "sources_used": ["xm", "ideam", "creg"]
}}

Genera al menos 3 insights. Incluye alertas solo si los datos las justifican.
Responde ÚNICAMENTE con el JSON válido, sin texto adicional."""

    # 4. Llamar a Gemini
    response = await _llm.ainvoke([HumanMessage(content=prompt)])

    try:
        result = _extract_json(response.content)
    except (json.JSONDecodeError, AttributeError) as exc:
        return {"error": f"Error parseando JSON del LLM: {exc}", "status": "failed"}

    insights: list[str] = result.get("insights", [])
    raw_alerts: list[dict] = result.get("alerts", [])
    llm_sources: list[str] = result.get("sources_used", state["sources_used"])
    sources_used = [s for s in llm_sources if s in _VALID_SOURCES]

    # 5. Guardar análisis en MongoDB
    analysis = Analysis(
        created_at=datetime.now(timezone.utc),
        raw_data_ids=state["raw_data_ids"],
        insights=insights,
        alerts=[
            Alert(
                tipo=a.get("tipo", "general"),
                severidad=a.get("severidad", "low"),
                descripcion=a.get("descripcion", ""),
            )
            for a in raw_alerts
        ],
        modelo_usado=settings.groq_model,
        sources_used=sources_used or state["sources_used"],
    )
    analysis_id = await insert_analysis(analysis)

    return {
        "retrieved_docs": retrieved_docs,
        "insights": insights,
        "alerts": raw_alerts,
        "has_alerts": len(raw_alerts) > 0,
        "sources_used": sources_used or state["sources_used"],
        "analysis_id": analysis_id,
        "status": "running",
        "error": None,
    }
