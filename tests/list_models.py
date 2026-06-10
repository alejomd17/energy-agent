"""
Diagnóstico: lista los modelos disponibles para la GEMINI_API_KEY configurada.
Ejecutar con: uv run python tests/list_models.py
"""

from google import genai
from app.core.config import get_settings

settings = get_settings()

for api_version in ["v1", "v1beta"]:
    print(f"\n{'─'*50}")
    print(f"  API version: {api_version}")
    print("─" * 50)
    try:
        client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options={"api_version": api_version},
        )
        embed_models = [m for m in client.models.list() if "embed" in m.name.lower()]
        if embed_models:
            for m in embed_models:
                print(f"  {m.name}")
        else:
            print("  (ningún modelo de embedding disponible)")
    except Exception as e:
        print(f"  Error: {e}")
