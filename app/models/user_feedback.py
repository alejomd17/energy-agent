"""
Modelo para feedback de calidad de los suscriptores.

`rating` usa validación ge/le en lugar de un Enum para facilitar
cálculos de promedio directamente en MongoDB ($avg, $group).
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserFeedback(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None = Field(default=None)
    newsletter_id: str
    rating: int = Field(ge=1, le=5)
    comment: str | None = None
    created_at: datetime
