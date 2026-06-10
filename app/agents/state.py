"""Estado compartido del sistema multi-agente."""

from datetime import datetime
from typing import TypedDict


class EnergyAgentState(TypedDict):
    run_id: str              # UUID de la ejecución, para trazabilidad
    week_date: datetime      # Lunes 00:00 UTC de la semana que se procesa
    raw_data_ids: list[str]  # IDs MongoDB de los documentos a analizar
    retrieved_docs: list[dict]  # Documentos recuperados del RAG (Supabase)
    insights: list[str]      # Insights generados por el analyzer
    alerts: list[dict]       # Alertas: {tipo, severidad, descripcion}
    has_alerts: bool         # Flag denormalizado para routing condicional
    sources_used: list[str]  # Fuentes activas: xm, ideam, creg, upme
    newsletter_html: str     # HTML completo del newsletter generado
    newsletter_subject: str  # Subject line del email
    analysis_id: str         # _id del análisis en MongoDB
    newsletter_id: str       # _id del newsletter en MongoDB
    error: str | None        # Descripción del error si status == "failed"
    status: str              # "running" | "completed" | "failed"
