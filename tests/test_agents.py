"""
Script de prueba manual del pipeline multi-agente LangGraph.

Ejecutar con:
    uv run python tests/test_agents.py

Requiere .env con MONGODB_URI, GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY.
Los documentos insertados se eliminan al finalizar (try/finally).
"""

import asyncio
from datetime import datetime, timezone

from bson import ObjectId

from app.agents.graph import run_pipeline
from app.core.database import close_mongo_connection, get_mongo_db, insert_raw_data
from app.models.raw_data import RawData


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print("─" * 65)


SAMPLE_RAW_DATA = [
    RawData(
        source="xm",
        scraped_at=datetime.now(timezone.utc),
        data_type="precio_energia",
        content={
            "precio_promedio_kwh": 312.4,
            "precio_maximo_kwh": 387.2,
            "fecha_inicio": "2026-06-02",
            "fecha_fin": "2026-06-08",
            "variacion_semanal_pct": 6.8,
            "demanda_gwh": 1247,
            "descripcion": (
                "El precio de bolsa de energía subió un 6.8% respecto a la semana anterior, "
                "impulsado por la baja disponibilidad hídrica en las principales cuencas."
            ),
        },
        processing_status="pending",
    ),
    RawData(
        source="ideam",
        scraped_at=datetime.now(timezone.utc),
        data_type="nivel_embalse",
        content={
            "nivel_promedio_pct": 48.3,
            "nivel_minimo_historico_10_anios": True,
            "cuencas_criticas": ["río Cauca", "río Sogamoso"],
            "pronostico_lluvia": "déficit para las próximas 4 semanas",
            "descripcion": (
                "Los embalses del sistema hidroeléctrico colombiano están al 48.3% de capacidad, "
                "el nivel más bajo para este período en 10 años. La cuenca del río Cauca "
                "presenta condiciones de alerta hidrológica."
            ),
        },
        processing_status="pending",
    ),
    RawData(
        source="creg",
        scraped_at=datetime.now(timezone.utc),
        data_type="regulacion",
        content={
            "resolucion": "CREG 101-011",
            "fecha_publicacion": "2026-06-01",
            "descripcion": (
                "La CREG actualizó la fórmula de cargos de distribución para el segundo semestre "
                "de 2026, con incremento del 3.2% en el cargo de transmisión nacional, efectivo "
                "a partir del 1 de julio de 2026."
            ),
            "impacto_tarifa_pct": 3.2,
            "vigencia": "2026-07-01",
        },
        processing_status="pending",
    ),
]


async def main() -> None:
    print("Iniciando prueba del sistema multi-agente LangGraph...")

    raw_data_ids: list[str] = []
    final_state: dict = {}

    try:
        # 1. Insertar documentos de prueba en MongoDB
        section("1. Insertando documentos raw_data en MongoDB")
        for doc in SAMPLE_RAW_DATA:
            doc_id = await insert_raw_data(doc)
            raw_data_ids.append(doc_id)
            print(f"  OK  {doc.source.upper():<8}  id={doc_id}")

        # 2. Ejecutar el pipeline
        section("2. Ejecutando pipeline  collect → analyze → write_newsletter")
        week_date = datetime(2026, 6, 8, tzinfo=timezone.utc)
        print(f"  week_date = {week_date.strftime('%Y-%m-%d')}\n")

        final_state = await run_pipeline(week_date, raw_data_ids)

        # 3. Estado final
        section("3. Estado final del pipeline")
        print(f"  status        : {final_state['status']}")
        print(f"  run_id        : {final_state['run_id']}")
        print(f"  has_alerts    : {final_state['has_alerts']}")
        print(f"  sources_used  : {final_state['sources_used']}")
        print(f"  analysis_id   : {final_state.get('analysis_id', '(none)')}")
        print(f"  newsletter_id : {final_state.get('newsletter_id', '(none)')}")

        if final_state.get("error"):
            print(f"\n  ERROR: {final_state['error']}")
            return

        # 4. Insights
        section("4. Insights generados")
        for i, insight in enumerate(final_state["insights"], 1):
            print(f"  [{i}] {insight}")

        # 5. Alertas
        section("5. Alertas detectadas")
        if final_state["alerts"]:
            for alert in final_state["alerts"]:
                sev = alert.get("severidad", "?").upper()
                tipo = alert.get("tipo", "?")
                desc = alert.get("descripcion", "")
                print(f"  [{sev}] {tipo}: {desc}")
        else:
            print("  Sin alertas esta semana.")

        # 6. Newsletter (primeros 500 chars)
        section("6. Newsletter generado")
        print(f"  Subject: {final_state['newsletter_subject']}")
        html_preview = final_state["newsletter_html"][:500].replace("\n", " ")
        print(f"\n  HTML (500 chars):\n  {html_preview}...")

    finally:
        section("Limpieza de documentos de prueba")
        db = await get_mongo_db()

        for doc_id in raw_data_ids:
            result = await db["raw_data"].delete_one({"_id": ObjectId(doc_id)})
            status = "OK" if result.deleted_count else "no encontrado"
            print(f"  raw_data    {doc_id[:12]}…  → {status}")

        analysis_id = final_state.get("analysis_id", "")
        if analysis_id:
            result = await db["analyses"].delete_one({"_id": ObjectId(analysis_id)})
            status = "OK" if result.deleted_count else "no encontrado"
            print(f"  analysis    {analysis_id[:12]}…  → {status}")

        newsletter_id = final_state.get("newsletter_id", "")
        if newsletter_id:
            result = await db["newsletters"].delete_one({"_id": ObjectId(newsletter_id)})
            status = "OK" if result.deleted_count else "no encontrado"
            print(f"  newsletter  {newsletter_id[:12]}…  → {status}")

        await close_mongo_connection()

    print("\nPrueba completada con éxito.")


if __name__ == "__main__":
    asyncio.run(main())
