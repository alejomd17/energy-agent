"""
Script de prueba manual para verificar la conectividad con MongoDB
y las operaciones CRUD básicas de todas las colecciones.

Ejecutar con:
    uv run python tests/test_mongodb.py

Requiere un archivo .env con MONGODB_URI y las demás variables configuradas.
Los documentos de prueba se eliminan al finalizar.
"""

import asyncio
from datetime import datetime, timezone

from bson import ObjectId

from app.core.database import (
    close_mongo_connection,
    get_mongo_db,
    get_raw_data,
    get_analysis,
    get_newsletter,
    get_feedback_for_newsletter,
    init_indexes,
    insert_raw_data,
    insert_analysis,
    insert_newsletter,
    insert_user_feedback,
)
from app.models.analysis import Alert, Analysis
from app.models.newsletter import AlertsSummary, Newsletter, RagasScores
from app.models.raw_data import RawData
from app.models.user_feedback import UserFeedback


def now() -> datetime:
    return datetime.now(timezone.utc)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


async def main() -> None:
    print("Iniciando prueba de MongoDB...")
    inserted_ids: dict[str, str] = {}

    # 1. Índices ---------------------------------------------------------------
    section("1. Inicializando índices")
    await init_indexes()
    print("OK — índices creados (o ya existían)")

    # 2. raw_data --------------------------------------------------------------
    section("2. raw_data — insert + find_one")
    raw = RawData(
        source="xm",
        scraped_at=now(),
        data_type="precio_energia",
        content={"precio_kwh": 320.5, "fecha": "2026-06-09", "hora": 14},
        processing_status="pending",
    )
    raw_id = await insert_raw_data(raw)
    inserted_ids["raw_data"] = raw_id
    print(f"Insertado _id: {raw_id}")

    fetched_raw = await get_raw_data(raw_id)
    assert fetched_raw is not None, "No se encontró el documento raw_data"
    print(fetched_raw.model_dump_json(indent=2))

    # 3. analyses --------------------------------------------------------------
    section("3. analyses — insert + find_one")
    analysis = Analysis(
        created_at=now(),
        raw_data_ids=[raw_id],
        insights=["El precio de la energía subió un 5 % respecto a la semana anterior."],
        alerts=[
            Alert(
                tipo="precio",
                severidad="medium",
                descripcion="Precio supera el promedio mensual en un 8 %",
            )
        ],
        modelo_usado="gemini-2.0-flash",
        sources_used=["xm"],
    )
    # Verifica que el model_validator sincronizó has_alerts
    assert analysis.has_alerts is True, "has_alerts debería ser True con una alerta"

    analysis_id = await insert_analysis(analysis)
    inserted_ids["analyses"] = analysis_id
    print(f"Insertado _id: {analysis_id}")

    fetched_analysis = await get_analysis(analysis_id)
    assert fetched_analysis is not None
    print(fetched_analysis.model_dump_json(indent=2))

    # 4. newsletters -----------------------------------------------------------
    section("4. newsletters — insert + find_one")
    newsletter = Newsletter(
        created_at=now(),
        week_date=datetime(2026, 6, 8, tzinfo=timezone.utc),  # lunes de la semana
        analysis_id=analysis_id,
        html_content="<h1>Newsletter Energético — Semana 23/2026</h1><p>Resumen...</p>",
        subject="Alerta energética: precios al alza — Semana 23/2026",
        sources_used=["xm"],
        ragas_scores=RagasScores(
            faithfulness=0.92,
            relevance=0.87,
            context_precision=0.85,
        ),
        alerts_summary=AlertsSummary(
            had_alerts=True,
            forecast_verified=None,
            forecast_detail="Tendencia alcista esperada para la próxima semana",
        ),
    )
    newsletter_id = await insert_newsletter(newsletter)
    inserted_ids["newsletters"] = newsletter_id
    print(f"Insertado _id: {newsletter_id}")

    fetched_newsletter = await get_newsletter(newsletter_id)
    assert fetched_newsletter is not None
    print(fetched_newsletter.model_dump_json(indent=2))

    # 5. user_feedback ---------------------------------------------------------
    section("5. user_feedback — insert + find_by_newsletter")
    feedback = UserFeedback(
        newsletter_id=newsletter_id,
        rating=4,
        comment="Muy útil el análisis de precios, aunque faltó contexto sobre embalses.",
        created_at=now(),
    )
    feedback_id = await insert_user_feedback(feedback)
    inserted_ids["user_feedback"] = feedback_id
    print(f"Insertado _id: {feedback_id}")

    feedbacks = await get_feedback_for_newsletter(newsletter_id)
    assert len(feedbacks) == 1
    print(f"Feedbacks del newsletter ({len(feedbacks)}):")
    for fb in feedbacks:
        print(f"  rating={fb.rating} | {fb.comment}")

    # 6. Limpieza --------------------------------------------------------------
    section("6. Limpieza de documentos de prueba")
    db = await get_mongo_db()
    collection_map = {
        "raw_data": inserted_ids["raw_data"],
        "analyses": inserted_ids["analyses"],
        "newsletters": inserted_ids["newsletters"],
        "user_feedback": inserted_ids["user_feedback"],
    }
    for collection, doc_id in collection_map.items():
        result = await db[collection].delete_one({"_id": ObjectId(doc_id)})
        status = "OK" if result.deleted_count else "NO ENCONTRADO"
        print(f"  {collection}: {status}")

    await close_mongo_connection()
    print("\nPrueba completada con éxito.")


if __name__ == "__main__":
    asyncio.run(main())
