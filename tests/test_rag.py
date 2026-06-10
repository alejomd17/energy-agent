"""
Script de prueba manual del pipeline RAG con Supabase pgvector.

Ejecutar con:
    uv run python tests/test_rag.py

Requisitos previos:
    1. Ejecutar sql/setup_vectors.sql en el SQL Editor de Supabase.
    2. Tener .env con SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY y GEMINI_API_KEY.

Los documentos insertados se eliminan al finalizar (con try/finally).
"""

import asyncio
from datetime import datetime, timezone

from app.core.vector_store import EnergyVectorStore, SearchResult

# ---------------------------------------------------------------------------
# Documentos de prueba — uno por fuente relevante del sector energético
# ---------------------------------------------------------------------------
# Estos textos están diseñados para que la query "precio de energía en Colombia"
# devuelva CREG y XM con mayor similitud que IDEAM (que trata embalses, no precios).

SAMPLE_DOCUMENTS = [
    {
        "content": (
            "La Comisión de Regulación de Energía y Gas (CREG) estableció mediante "
            "Resolución 101-011 de 2026 los cargos de distribución de energía eléctrica "
            "para el nivel de tensión 1. Los precios de la energía en Colombia están "
            "regulados por la fórmula tarifaria vigente, que incorpora el cargo de "
            "generación, transmisión, distribución y comercialización."
        ),
        "metadata": {
            "source": "creg",
            "doc_type": "regulacion",
            "title": "Resolución CREG 101-011 — Cargos de distribución 2026",
            "created_at": datetime(2026, 3, 15, tzinfo=timezone.utc).isoformat(),
        },
    },
    {
        "content": (
            "XM reportó que el precio de bolsa de energía en Colombia durante la semana "
            "del 2 al 8 de junio de 2026 promedió $312.4 COP/kWh, un 6.8 % por encima "
            "del promedio semanal anterior. El precio máximo se registró el miércoles "
            "con $387.2 COP/kWh. La demanda nacional de energía eléctrica fue de "
            "1,247 GWh para el período reportado."
        ),
        "metadata": {
            "source": "xm",
            "doc_type": "precio_energia",
            "title": "Informe semanal de precios — Semana 23/2026",
            "created_at": datetime(2026, 6, 8, tzinfo=timezone.utc).isoformat(),
        },
    },
    {
        "content": (
            "El IDEAM reportó que el nivel de los embalses del sistema hidroeléctrico "
            "colombiano se encuentra al 48.3 % de su capacidad total al cierre de junio "
            "de 2026. La cuenca del río Cauca presenta los menores niveles históricos "
            "para este período, lo que podría presionar al alza el precio de la energía "
            "eléctrica en los próximos meses ante menor disponibilidad hidráulica."
        ),
        "metadata": {
            "source": "ideam",
            "doc_type": "nivel_embalse",
            "title": "Boletín hidrológico — Junio 2026",
            "created_at": datetime(2026, 6, 9, tzinfo=timezone.utc).isoformat(),
        },
    },
]


def section(title: str) -> None:
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print("─" * 65)


def print_results(results: list[SearchResult]) -> None:
    for i, r in enumerate(results, 1):
        source = r.metadata.get("source", "?").upper()
        title = r.metadata.get("title", "Sin título")
        snippet = r.content[:110].replace("\n", " ")
        print(f"  [{i}] score={r.score:.5f}  {source} — {title}")
        print(f"       \"{snippet}...\"\n")


async def main() -> None:
    print("Iniciando prueba del pipeline RAG...")
    store = EnergyVectorStore()
    inserted_ids: list[str] = []

    try:
        # 1. Insertar documentos -----------------------------------------------
        section("1. Generando embeddings e insertando 3 documentos en Supabase")
        # add_documents hace una sola llamada a la API de Google para los 3 textos
        inserted_ids = await store.add_documents(SAMPLE_DOCUMENTS)
        for doc_id, doc in zip(inserted_ids, SAMPLE_DOCUMENTS):
            src = doc["metadata"]["source"].upper()
            print(f"  OK  {src:6s}  id={doc_id}")

        query = "precio de energía en Colombia"

        # 2. Búsqueda semántica ------------------------------------------------
        # Esperado: CREG y XM con score alto; IDEAM con score algo menor porque
        # habla de embalses pero también menciona "precio de la energía".
        section(f'2. Similarity search  →  "{query}"')
        print("  (score = similitud coseno: 1.0 = idéntico, 0.0 = sin relación)\n")
        semantic_results = await store.similarity_search(query, k=3)
        print_results(semantic_results)

        # 3. Búsqueda híbrida --------------------------------------------------
        # RRF combina el ranking semántico con keyword matching.
        # Documentos que aparecen en ambas listas reciben score más alto.
        section(f'3. Hybrid search (dense + sparse + RRF)  →  "{query}"')
        print("  (score = Reciprocal Rank Fusion: mayor = más relevante)\n")
        hybrid_results = await store.hybrid_search(query, k=3)
        print_results(hybrid_results)

        # 4. Filtro por fuente -------------------------------------------------
        section('4. Similarity search filtrada  →  source="xm"')
        xm_results = await store.similarity_search(query, k=2, filter={"source": "xm"})
        print("  (sólo documentos de XM)\n")
        print_results(xm_results)

        # 5. Eliminación individual --------------------------------------------
        section("5. delete_document — eliminar el primer resultado semántico")
        if semantic_results:
            target_id = semantic_results[0].id
            deleted = await store.delete_document(target_id)
            print(f"  id={target_id}  →  eliminado={deleted}")
            if target_id in inserted_ids:
                inserted_ids.remove(target_id)

    finally:
        # Limpieza garantizada aunque el script falle a mitad
        if inserted_ids:
            section("Limpieza de documentos de prueba restantes")
            for doc_id in inserted_ids:
                deleted = await store.delete_document(doc_id)
                status = "eliminado" if deleted else "no encontrado"
                print(f"  {doc_id[:12]}…  →  {status}")

    print("\nPrueba completada con éxito.")


if __name__ == "__main__":
    asyncio.run(main())
