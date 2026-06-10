"""
Agente 1: Data Collector

Lee documentos raw_data de MongoDB y valida que haya pendientes antes
de invocar el LLM. Falla rápido si no hay datos — evita tokens innecesarios.
"""

from app.agents.state import EnergyAgentState
from app.core.database import get_raw_data


async def collect_data(state: EnergyAgentState) -> dict:
    """
    Filtra los raw_data_ids que tienen processing_status == "pending"
    y construye el conjunto de fuentes activas.

    Devuelve un dict parcial; LangGraph lo fusiona con el estado existente.
    """
    pending_ids: list[str] = []
    sources: set[str] = set()

    for doc_id in state["raw_data_ids"]:
        doc = await get_raw_data(doc_id)
        if doc is None:
            continue
        if doc.processing_status == "pending":
            pending_ids.append(doc_id)
            sources.add(doc.source)

    if not pending_ids:
        return {
            "error": "No hay documentos raw_data pendientes para procesar.",
            "status": "failed",
        }

    return {
        "raw_data_ids": pending_ids,
        "sources_used": sorted(sources),
        "status": "running",
        "error": None,
    }
