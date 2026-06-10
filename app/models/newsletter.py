"""
Modelos para newsletters generadas y sus metadatos de calidad.

`week_date` siempre representa el lunes 00:00 UTC de la semana
correspondiente. Se indexa con unique=True para evitar generar
dos newsletters para la misma semana por error.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RagasScores(BaseModel):
    """Métricas de calidad del pipeline RAG, calculadas con RAGAS."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)


class AlertsSummary(BaseModel):
    """
    Resumen ejecutivo de alertas para el encabezado del newsletter.

    `forecast_verified` es None mientras el período predicho no ha
    transcurrido. Se actualiza a True/False en el siguiente ciclo.
    """

    had_alerts: bool
    forecast_verified: bool | None = None
    forecast_detail: str | None = None


class Newsletter(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None)
    created_at: datetime
    week_date: datetime
    analysis_id: str
    html_content: str
    subject: str
    sources_used: list[str]
    ragas_scores: RagasScores
    alerts_summary: AlertsSummary
    sent: bool = False
    sent_at: datetime | None = None
