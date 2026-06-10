"""
Agente 3: Newsletter Writer

Convierte el análisis estructurado en un newsletter HTML profesional en español.

El grafo enruta a `write_newsletter_urgent` cuando has_alerts=True,
priorizando visualmente la sección de alertas en el contenido generado.
"""

from datetime import datetime, timezone

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.agents.state import EnergyAgentState
from app.core.config import get_settings
from app.core.database import insert_newsletter
from app.models.newsletter import AlertsSummary, Newsletter, RagasScores

settings = get_settings()

_llm = ChatGroq(
    model=settings.groq_model,
    groq_api_key=settings.groq_api_key,
    temperature=0.4,
    max_retries=2,
)


async def _write_newsletter_impl(state: EnergyAgentState, urgent: bool) -> dict:
    """Implementación compartida. `urgent` ajusta el tono y el énfasis visual."""
    week_str = state["week_date"].strftime("%d de %B de %Y")
    insights_text = "\n".join(f"- {i}" for i in state["insights"])
    alerts_text = (
        "\n".join(
            f"- [{a['severidad'].upper()}] {a['tipo']}: {a['descripcion']}"
            for a in state["alerts"]
        )
        if state["alerts"]
        else "Sin alertas esta semana."
    )
    rag_context = "\n".join(
        f"- {d['content'][:200]}" for d in state["retrieved_docs"][:3]
    )

    urgency_header = (
        "⚠️ SEMANA CON ALERTAS ENERGÉTICAS — Priorizar la sección de alertas.\n\n"
        if urgent
        else ""
    )
    alert_instruction = (
        "<section id=\"alertas\"> Alertas y Avisos — RESALTAR con fondo rojo/naranja"
        if urgent
        else "<section id=\"alertas\"> Alertas y Avisos (indicar si no hay alertas)"
    )

    prompt = f"""{urgency_header}Eres un redactor experto en el sector energético colombiano.
Genera un newsletter HTML profesional en español para la semana del {week_str}.

=== INSIGHTS DEL ANÁLISIS ===
{insights_text}

=== ALERTAS DETECTADAS ===
{alerts_text}

=== CONTEXTO ADICIONAL ===
{rag_context}

El newsletter HTML debe tener EXACTAMENTE esta estructura:
1. <header> con título "Newsletter Energético Colombia" y fecha
2. <section id="resumen"> Executive Summary (2-3 párrafos)
3. <section id="precios"> Análisis de Precios de Energía
4. <section id="embalses"> Niveles de Embalse y Disponibilidad Hídrica
5. {alert_instruction}
6. <section id="recomendaciones"> Recomendaciones para Stakeholders
7. <footer> con fuentes consultadas: {', '.join(state['sources_used'])}

Usa HTML semántico con estilos inline básicos (colores corporativos azul/verde).
Responde ÚNICAMENTE con el HTML completo desde <html> hasta </html>."""

    html_response = await _llm.ainvoke([HumanMessage(content=prompt)])
    html_content = html_response.content.strip()

    # Limpiar markdown code fence si el LLM lo devuelve
    if html_content.startswith("```"):
        html_content = html_content.split("```", 2)[1]
        if html_content.startswith("html"):
            html_content = html_content[4:]
        html_content = html_content.rstrip("` \n")

    subject_response = await _llm.ainvoke([
        HumanMessage(
            content=(
                f"Genera el asunto de email para un newsletter del sector energético colombiano "
                f"de la semana del {week_str}. "
                f"{'Incluye advertencia de alerta energética urgente.' if urgent else ''} "
                f"Máximo 80 caracteres. Solo el texto del asunto, sin comillas ni explicaciones."
            )
        )
    ])
    subject = subject_response.content.strip().strip('"\'')[:80]

    newsletter = Newsletter(
        created_at=datetime.now(timezone.utc),
        week_date=state["week_date"],
        analysis_id=state["analysis_id"],
        html_content=html_content,
        subject=subject,
        sources_used=state["sources_used"],
        ragas_scores=RagasScores(faithfulness=0.0, relevance=0.0, context_precision=0.0),
        alerts_summary=AlertsSummary(
            had_alerts=state["has_alerts"],
            forecast_verified=None,
            forecast_detail=None,
        ),
    )
    newsletter_id = await insert_newsletter(newsletter)

    return {
        "newsletter_html": html_content,
        "newsletter_subject": subject,
        "newsletter_id": newsletter_id,
        "status": "completed",
        "error": None,
    }


async def write_newsletter(state: EnergyAgentState) -> dict:
    """Newsletter estándar (semana sin alertas)."""
    return await _write_newsletter_impl(state, urgent=False)


async def write_newsletter_urgent(state: EnergyAgentState) -> dict:
    """Newsletter urgente — prioriza visualmente la sección de alertas."""
    return await _write_newsletter_impl(state, urgent=True)
