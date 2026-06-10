"""
Modelos para el output del agente de análisis.

`has_alerts` se mantiene como campo booleano denormalizado para
poder crear un índice MongoDB eficiente en (has_alerts, created_at)
sin tener que evaluar el array `alerts` en cada consulta.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Source = Literal["xm", "ideam", "creg", "upme"]
Severidad = Literal["low", "medium", "high"]


class Alert(BaseModel):
    tipo: str
    severidad: Severidad
    descripcion: str


class Analysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None)
    created_at: datetime
    raw_data_ids: list[str]
    insights: list[str]
    alerts: list[Alert] = Field(default_factory=list)
    # Derivado de `alerts`; se persiste para índices de filtrado rápido
    has_alerts: bool = False
    modelo_usado: str
    sources_used: list[Source]

    @model_validator(mode="after")
    def _sync_has_alerts(self) -> "Analysis":
        """Garantiza coherencia entre la lista de alerts y el flag booleano."""
        self.has_alerts = len(self.alerts) > 0
        return self
