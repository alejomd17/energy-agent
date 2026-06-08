"""
Configuración central de la aplicación.

Usa pydantic-settings para cargar y validar todas las variables de entorno
definidas en .env. Un único objeto `Settings` se comparte en toda la app
gracias al caché de `get_settings()`.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Permite variables adicionales en .env sin lanzar error
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Energy Agent API"
    app_version: str = "0.1.0"
    debug: bool = False

    # --- Supabase (PostgreSQL + pgvector) ---
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    supabase_db_password: str

    # --- MongoDB ---
    mongodb_uri: str

    # --- Google Gemini ---
    gemini_api_key: str

    # --- LangSmith (opcional — trazabilidad de agentes LangGraph) ---
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "energy-agent"


@lru_cache
def get_settings() -> Settings:
    """Devuelve la instancia única de Settings (singleton con lru_cache)."""
    return Settings()
