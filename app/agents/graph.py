"""
Orquestador LangGraph del pipeline de newsletter energético.

Flujo del grafo:

    [START]
       │
    collect_data  ──(error)──► [END]
       │
    analyze_data  ──(error)──► [END]
       │                  ╲
       │ (no alerts)       ╲ (has_alerts)
       ▼                    ▼
    write_newsletter   write_newsletter_urgent
       │                    │
       └────────────────────┘
                │
             [END]

Los edges condicionales permiten que el collector aborte el pipeline
sin coste de tokens si no hay datos pendientes, y que el writer
urgente priorice las alertas cuando el analyzer las detecta.
"""

import uuid
from datetime import datetime

from langgraph.graph import END, StateGraph

from app.agents.collector import collect_data
from app.agents.analyzer import analyze_data
from app.agents.writer import write_newsletter, write_newsletter_urgent
from app.agents.state import EnergyAgentState


def _route_after_collect(state: EnergyAgentState) -> str:
    """Aborta el pipeline si el collector no encontró documentos pendientes."""
    if state.get("error"):
        return END
    return "analyze_data"


def _route_after_analyze(state: EnergyAgentState) -> str:
    """Enruta al writer urgente si el analyzer detectó alertas."""
    if state.get("error"):
        return END
    if state.get("has_alerts"):
        return "write_newsletter_urgent"
    return "write_newsletter"


def _build_graph():
    graph = StateGraph(EnergyAgentState)

    graph.add_node("collect_data", collect_data)
    graph.add_node("analyze_data", analyze_data)
    graph.add_node("write_newsletter", write_newsletter)
    graph.add_node("write_newsletter_urgent", write_newsletter_urgent)

    graph.set_entry_point("collect_data")

    graph.add_conditional_edges("collect_data", _route_after_collect)
    graph.add_conditional_edges("analyze_data", _route_after_analyze)

    graph.add_edge("write_newsletter", END)
    graph.add_edge("write_newsletter_urgent", END)

    return graph.compile()


_compiled_graph = _build_graph()


async def run_pipeline(
    week_date: datetime,
    raw_data_ids: list[str],
) -> EnergyAgentState:
    """
    Punto de entrada del pipeline.

    Args:
        week_date:     Lunes 00:00 UTC de la semana a procesar.
        raw_data_ids:  IDs de documentos raw_data en MongoDB.

    Returns:
        Estado final — newsletter generado o error con detalle.
    """
    initial_state: EnergyAgentState = {
        "run_id": str(uuid.uuid4()),
        "week_date": week_date,
        "raw_data_ids": raw_data_ids,
        "retrieved_docs": [],
        "insights": [],
        "alerts": [],
        "has_alerts": False,
        "sources_used": [],
        "newsletter_html": "",
        "newsletter_subject": "",
        "analysis_id": "",
        "newsletter_id": "",
        "error": None,
        "status": "running",
    }

    return await _compiled_graph.ainvoke(initial_state)
