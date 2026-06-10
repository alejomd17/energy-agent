-- =============================================================================
-- Setup de pgvector para el sistema RAG del sector energético colombiano.
-- Ejecutar una vez en Supabase: SQL Editor → pegar y ejecutar.
-- =============================================================================

-- 1. Extensión pgvector
-- Habilita el tipo `vector` y los operadores de similitud (<=> coseno, <-> L2, <#> producto interior).
CREATE EXTENSION IF NOT EXISTS vector;


-- 2. Tabla principal de documentos
-- content    — texto del fragmento indexado (chunk del documento original)
-- embedding  — vector de 3072 floats producido por gemini-embedding-001 (API v1)
-- metadata   — jsonb libre: source, doc_type, title, created_at, url, etc.
CREATE TABLE IF NOT EXISTS energy_documents (
    id         uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
    content    text         NOT NULL,
    embedding  vector(3072) NOT NULL,
    metadata   jsonb        NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz  NOT NULL DEFAULT now()
);


-- 3. Índice de vectores
-- gemini-embedding-001 produce 3072 dims, por encima del límite de 2000 de ivfflat/hnsw.
-- Para colecciones pequeñas (< ~50k docs) el sequential scan exacto es suficiente.
-- Para escalar usar: CREATE INDEX ON energy_documents
--   USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);  -- pgvector >= 0.7

-- 4. Índice GIN en metadata para filtros frecuentes por fuente/tipo
CREATE INDEX IF NOT EXISTS energy_documents_metadata_idx
    ON energy_documents
    USING gin (metadata);


-- 5. Función match_documents — búsqueda por similitud coseno
-- El operador <=> devuelve distancia coseno ∈ [0, 2].
-- similarity = 1 - distancia, así 1.0 = idéntico, 0.0 = sin relación.
-- El parámetro `filter` permite restringir por metadata usando el operador @>
-- (containment): filter = '{"source":"xm"}' sólo busca en documentos de XM.
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding  vector(3072),
    match_count      int   DEFAULT 5,
    filter           jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id          uuid,
    content     text,
    metadata    jsonb,
    similarity  float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.content,
        d.metadata,
        (1 - (d.embedding <=> query_embedding))::float AS similarity
    FROM energy_documents d
    WHERE
        CASE
            WHEN filter = '{}'::jsonb THEN true
            ELSE d.metadata @> filter
        END
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;


-- 6. Permisos
-- service_role es el rol que usa el backend (service_role_key de Supabase).
-- Sin este GRANT la tabla existe pero las llamadas desde supabase-py dan 42501.
GRANT SELECT, INSERT, UPDATE, DELETE ON public.energy_documents TO service_role;
