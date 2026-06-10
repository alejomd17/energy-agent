# Esquemas Pydantic para validación de datos.
# Incluye modelos de request/response de la API y documentos de base de datos.

from app.models.analysis import Alert, Analysis
from app.models.newsletter import AlertsSummary, Newsletter, RagasScores
from app.models.raw_data import RawData
from app.models.user_feedback import UserFeedback

__all__ = [
    "Alert",
    "Analysis",
    "AlertsSummary",
    "Newsletter",
    "RagasScores",
    "RawData",
    "UserFeedback",
]
