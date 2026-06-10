"""
Tests de integración de la API REST.

Ejecutar con: uv run python tests/test_api.py

Prueba todos los endpoints usando httpx.AsyncClient con ASGITransport,
que ejecuta la app FastAPI en memoria sin levantar un servidor real.
Requiere .env completo (MongoDB, Groq, Gemini, Supabase).
"""

import asyncio
import json
from datetime import datetime, timezone

from bson import ObjectId
from httpx import ASGITransport, AsyncClient

from app.core.database import close_mongo_connection, get_mongo_db, insert_raw_data
from app.main import app
from app.models.raw_data import RawData

WEEK_DATE = "2026-06-08"

SAMPLE_RAW_DATA = [
    RawData(
        source="xm",
        scraped_at=datetime.now(timezone.utc),
        data_type="precio_energia",
        content={
            "precio_promedio_kwh": 312.4,
            "variacion_semanal_pct": 6.8,
            "descripcion": "Precio de bolsa subió 6.8% por baja disponibilidad hídrica.",
        },
        processing_status="pending",
    ),
    RawData(
        source="ideam",
        scraped_at=datetime.now(timezone.utc),
        data_type="nivel_embalse",
        content={
            "nivel_promedio_pct": 48.3,
            "cuencas_criticas": ["río Cauca", "río Sogamoso"],
            "descripcion": "Embalses al 48.3%, mínimo histórico para este período en 10 años.",
        },
        processing_status="pending",
    ),
]


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print("─" * 65)


async def main() -> None:
    print("Iniciando tests de integración de la API...")

    raw_data_ids: list[str] = []
    newsletter_id: str = ""
    feedback_id: str = ""

    try:
        # Insertar documentos de prueba directamente en MongoDB
        section("Setup: insertando raw_data de prueba")
        for doc in SAMPLE_RAW_DATA:
            doc_id = await insert_raw_data(doc)
            raw_data_ids.append(doc_id)
            print(f"  OK  {doc.source.upper():<8}  id={doc_id}")

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:

            # ------------------------------------------------------------------
            # 1. POST /pipeline/run
            # ------------------------------------------------------------------
            section("1. POST /pipeline/run")
            response = await client.post(
                "/pipeline/run",
                json={"week_date": WEEK_DATE, "raw_data_ids": raw_data_ids},
                timeout=120.0,  # el pipeline llama a Groq + Supabase + MongoDB
            )
            assert response.status_code == 200, f"Esperado 200, recibido {response.status_code}: {response.text}"
            data = response.json()
            assert data["status"] == "completed"
            assert data["analysis_id"]
            assert data["newsletter_id"]
            newsletter_id = data["newsletter_id"]
            print(f"  status        : {data['status']}")
            print(f"  has_alerts    : {data['has_alerts']}")
            print(f"  analysis_id   : {data['analysis_id']}")
            print(f"  newsletter_id : {newsletter_id}")
            print(f"  insights[0]   : {data['insights'][0][:80]}...")

            # ------------------------------------------------------------------
            # 2. GET /newsletters (paginación)
            # ------------------------------------------------------------------
            section("2. GET /newsletters?page=1&page_size=5")
            response = await client.get("/newsletters", params={"page": 1, "page_size": 5})
            assert response.status_code == 200
            data = response.json()
            assert "items" in data
            assert data["total"] >= 1
            print(f"  total   : {data['total']}")
            print(f"  pages   : {data['pages']}")
            print(f"  items[0]: {data['items'][0]['subject'][:60]}")

            # ------------------------------------------------------------------
            # 3. GET /newsletters/{week_date}
            # ------------------------------------------------------------------
            section(f"3. GET /newsletters/{WEEK_DATE}")
            response = await client.get(f"/newsletters/{WEEK_DATE}")
            assert response.status_code == 200
            data = response.json()
            assert "html_content" in data
            assert "subject" in data
            print(f"  subject      : {data['subject']}")
            print(f"  html (50ch)  : {data['html_content'][:50]}...")

            # ------------------------------------------------------------------
            # 4. GET /newsletters/{id}/html → text/html
            # ------------------------------------------------------------------
            section(f"4. GET /newsletters/{newsletter_id}/html")
            response = await client.get(f"/newsletters/{newsletter_id}/html")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            assert "<html" in response.text.lower()
            print(f"  Content-Type : {response.headers['content-type']}")
            print(f"  HTML (60ch)  : {response.text[:60]}...")

            # ------------------------------------------------------------------
            # 5. GET /analyze/stream → SSE
            # ------------------------------------------------------------------
            section("5. GET /analyze/stream?query=precio energia Colombia")
            chunks: list[str] = []
            sources_received: list[str] = []
            done_received = False

            async with client.stream(
                "GET",
                "/analyze/stream",
                params={"query": "precio energia Colombia embalses"},
                timeout=60.0,
            ) as stream_response:
                assert stream_response.status_code == 200
                assert "text/event-stream" in stream_response.headers["content-type"]

                async for line in stream_response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]  # strip "data: "
                    if payload == "[DONE]":
                        done_received = True
                        break
                    event = json.loads(payload)
                    if event.get("type") == "sources":
                        sources_received = event["sources"]
                    elif event.get("type") == "token":
                        chunks.append(event["text"])

            assert done_received, "No se recibió [DONE] al final del stream"
            assert len(chunks) > 0, "No se recibieron chunks de texto"
            full_text = "".join(chunks)
            print(f"  sources  : {sources_received}")
            print(f"  chunks   : {len(chunks)}")
            print(f"  texto    : {full_text[:120]}...")

            # ------------------------------------------------------------------
            # 6. POST /newsletters/{id}/feedback → 201
            # ------------------------------------------------------------------
            section(f"6. POST /newsletters/{newsletter_id}/feedback")
            response = await client.post(
                f"/newsletters/{newsletter_id}/feedback",
                json={"rating": 4, "comment": "Buen análisis del sector energético."},
            )
            assert response.status_code == 201
            data = response.json()
            assert data["id"]
            feedback_id = data["id"]
            print(f"  feedback_id : {feedback_id}")

            # ------------------------------------------------------------------
            # 7. 404 para fecha inexistente
            # ------------------------------------------------------------------
            section("7. GET /newsletters/2000-01-01 → 404")
            response = await client.get("/newsletters/2000-01-01")
            assert response.status_code == 404
            print(f"  status : {response.status_code}  ✓")

    finally:
        section("Limpieza de documentos de prueba")
        db = await get_mongo_db()

        for doc_id in raw_data_ids:
            r = await db["raw_data"].delete_one({"_id": ObjectId(doc_id)})
            print(f"  raw_data    {doc_id[:12]}…  → {'OK' if r.deleted_count else 'no encontrado'}")

        # newsletter_id viene del pipeline (análisis + newsletter)
        if newsletter_id:
            # Buscar analysis_id en la newsletter para limpiarlo también
            doc = await db["newsletters"].find_one({"_id": ObjectId(newsletter_id)})
            if doc and doc.get("analysis_id"):
                r = await db["analyses"].delete_one({"_id": ObjectId(doc["analysis_id"])})
                print(f"  analysis    {doc['analysis_id'][:12]}…  → {'OK' if r.deleted_count else 'no encontrado'}")
            r = await db["newsletters"].delete_one({"_id": ObjectId(newsletter_id)})
            print(f"  newsletter  {newsletter_id[:12]}…  → {'OK' if r.deleted_count else 'no encontrado'}")

        if feedback_id:
            r = await db["user_feedback"].delete_one({"_id": ObjectId(feedback_id)})
            print(f"  feedback    {feedback_id[:12]}…  → {'OK' if r.deleted_count else 'no encontrado'}")

        await close_mongo_connection()

    print("\nTodos los tests pasaron con éxito.")


if __name__ == "__main__":
    asyncio.run(main())
