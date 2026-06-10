"""Router para consultar newsletters y recibir feedback de usuarios."""

from datetime import date, datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.core.database import (
    get_mongo_db,
    get_newsletter,
    get_newsletter_by_week,
    insert_user_feedback,
)
from app.models.newsletter import RagasScores
from app.models.user_feedback import UserFeedback

router = APIRouter()


# ---------------------------------------------------------------------------
# Modelos de respuesta
# ---------------------------------------------------------------------------

class NewsletterListItem(BaseModel):
    id: str
    week_date: datetime
    subject: str
    sent: bool
    has_alerts: bool
    ragas_scores: RagasScores


class PaginatedNewsletters(BaseModel):
    items: list[NewsletterListItem]
    total: int
    page: int
    page_size: int
    pages: int


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str = ""


class FeedbackResponse(BaseModel):
    id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=PaginatedNewsletters)
async def list_newsletters(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    only_sent: bool = Query(default=False),
) -> PaginatedNewsletters:
    """
    Lista newsletters con paginación.

    Devuelve los campos de cabecera (sin html_content) para que la respuesta
    sea ligera. Usa GET /newsletters/{id}/html para obtener el HTML completo.
    """
    db = await get_mongo_db()
    query: dict = {}
    if only_sent:
        query["sent"] = True

    total = await db["newsletters"].count_documents(query)
    offset = (page - 1) * page_size
    cursor = (
        db["newsletters"]
        .find(query)
        .sort("week_date", -1)
        .skip(offset)
        .limit(page_size)
    )
    docs = await cursor.to_list(length=page_size)

    items = [
        NewsletterListItem(
            id=str(doc["_id"]),
            week_date=doc["week_date"],
            subject=doc["subject"],
            sent=doc.get("sent", False),
            has_alerts=doc.get("alerts_summary", {}).get("had_alerts", False),
            ragas_scores=RagasScores(**doc.get("ragas_scores", {"faithfulness": 0.0, "relevance": 0.0, "context_precision": 0.0})),
        )
        for doc in docs
    ]

    return PaginatedNewsletters(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=max(1, -(-total // page_size)),  # ceil division
    )


@router.get("/{newsletter_id}/html", response_class=HTMLResponse)
async def get_newsletter_html(newsletter_id: str) -> HTMLResponse:
    """
    Devuelve el html_content del newsletter como text/html.
    Útil para previsualizar en el navegador o integrar en un iframe.
    """
    newsletter = await get_newsletter(newsletter_id)
    if newsletter is None:
        raise HTTPException(status_code=404, detail="Newsletter no encontrado.")
    return HTMLResponse(content=newsletter.html_content)


@router.get("/{week_date}")
async def get_newsletter_by_date(week_date: date):
    """
    Devuelve el newsletter completo (incluye html_content) para una semana dada.
    week_date debe ser el lunes de la semana en formato YYYY-MM-DD.
    """
    week_dt = datetime(
        week_date.year, week_date.month, week_date.day, tzinfo=timezone.utc
    )
    newsletter = await get_newsletter_by_week(week_dt)
    if newsletter is None:
        raise HTTPException(
            status_code=404,
            detail=f"No existe newsletter para la semana del {week_date}.",
        )
    return newsletter.model_dump(mode="json")


@router.post("/{newsletter_id}/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    newsletter_id: str,
    body: FeedbackRequest,
) -> FeedbackResponse:
    """
    Registra el feedback de un usuario sobre un newsletter.
    rating debe estar entre 1 (muy malo) y 5 (excelente).
    """
    newsletter = await get_newsletter(newsletter_id)
    if newsletter is None:
        raise HTTPException(status_code=404, detail="Newsletter no encontrado.")

    feedback = UserFeedback(
        newsletter_id=newsletter_id,
        rating=body.rating,
        comment=body.comment,
        created_at=datetime.now(timezone.utc),
    )
    feedback_id = await insert_user_feedback(feedback)
    return FeedbackResponse(id=feedback_id)
