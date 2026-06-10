"""
Punto de entrada de la aplicación FastAPI.

Registra los routers de la API y gestiona el ciclo de vida
(startup / shutdown) para las conexiones a bases de datos.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import analyze, newsletters, pipeline
from app.core.config import get_settings
from app.core.database import close_mongo_connection, get_mongo_client, init_indexes

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await get_mongo_client()
    await init_indexes()
    yield
    await close_mongo_connection()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

# CORS — permite requests desde el frontend (React/Next.js en localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])
app.include_router(newsletters.router, prefix="/newsletters", tags=["newsletters"])
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])


@app.get("/health", tags=["health"])
async def health_check():
    """Endpoint de salud para verificar que la API está activa."""
    return {"status": "ok", "version": settings.app_version}
