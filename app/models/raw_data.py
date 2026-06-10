"""
Modelo para documentos crudos de scraping.

Se almacena tal como llega de la fuente, sin transformar, para poder
reprocesar si cambia el parser. El campo `content` es intencionalmente
flexible (dict libre) porque cada fuente tiene una estructura distinta.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Source = Literal["xm", "ideam", "creg", "upme"]
DataType = Literal["precio_energia", "nivel_embalse", "regulacion", "noticia"]
ProcessingStatus = Literal["pending", "processed", "failed"]


class RawData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # id es None al crear; se rellena al leer desde MongoDB (_id → id)
    id: str | None = Field(default=None)
    source: Source
    scraped_at: datetime
    data_type: DataType
    content: dict[str, Any]
    processing_status: ProcessingStatus = "pending"
