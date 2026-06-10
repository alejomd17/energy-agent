"""Router para ejecutar el pipeline multi-agente LangGraph."""

from datetime import date, datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.graph import run_pipeline

router = APIRouter()


class PipelineRunRequest(BaseModel):
    week_date: date
    raw_data_ids: list[str]


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    analysis_id: str
    newsletter_id: str
    has_alerts: bool
    sources_used: list[str]
    insights: list[str]
    alerts: list[dict]
    newsletter_subject: str


@router.post("/run", response_model=PipelineRunResponse, status_code=200)
async def run_pipeline_endpoint(body: PipelineRunRequest) -> PipelineRunResponse:
    """
    Ejecuta el pipeline completo: collect → analyze → write_newsletter.

    Recibe los IDs de documentos raw_data ya insertados en MongoDB y la fecha
    de la semana a procesar. Devuelve el estado final con el análisis y el
    newsletter generado.
    """
    week_datetime = datetime(
        body.week_date.year,
        body.week_date.month,
        body.week_date.day,
        tzinfo=timezone.utc,
    )

    state = await run_pipeline(week_datetime, body.raw_data_ids)

    if state["status"] == "failed":
        raise HTTPException(
            status_code=500,
            detail=state.get("error") or "El pipeline falló sin mensaje de error.",
        )

    return PipelineRunResponse(
        run_id=state["run_id"],
        status=state["status"],
        analysis_id=state["analysis_id"],
        newsletter_id=state["newsletter_id"],
        has_alerts=state["has_alerts"],
        sources_used=state["sources_used"],
        insights=state["insights"],
        alerts=state["alerts"],
        newsletter_subject=state["newsletter_subject"],
    )
