# Energy Agent

Sistema multi-agente para inteligencia del sector energético colombiano.

Combina scraping automatizado, RAG (Retrieval-Augmented Generation) y una
orquestación de agentes LangGraph para responder preguntas complejas sobre
el mercado eléctrico, precios, regulación y actores del sector en Colombia.

## Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| API | FastAPI + Uvicorn |
| Agentes | LangGraph + LangChain |
| LLM | Google Gemini (via `langchain-google-genai`) |
| Vectores / RAG | Supabase + pgvector |
| Documentos / Logs | MongoDB Atlas (motor async) |
| Evaluación | RAGAS |
| Scraping | BeautifulSoup4 + HTTPX |
| Scheduler | APScheduler |
| Config | pydantic-settings + python-dotenv |

## Estructura del proyecto

```
energy-agent/
├── app/
│   ├── agents/       # Grafos LangGraph (agentes y subagentes)
│   ├── api/          # Rutas y routers de FastAPI
│   ├── core/         # Configuración y conexiones a bases de datos
│   ├── evaluation/   # Pipelines de evaluación con RAGAS
│   ├── models/       # Esquemas Pydantic (request/response/DB)
│   └── scrapers/     # Recolectores de datos del sector energético
├── tests/            # Pruebas unitarias e integración
├── .env.example      # Plantilla de variables de entorno
└── pyproject.toml    # Dependencias y configuración del proyecto
```

## Setup local

### 1. Requisitos previos

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)
- Cuenta en [Supabase](https://supabase.com) con extensión `pgvector` habilitada
- Cluster en [MongoDB Atlas](https://www.mongodb.com/atlas)
- API key de [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Clonar e instalar dependencias

```bash
git clone <repo-url>
cd energy-agent
uv sync
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con los valores reales
```

### 4. Ejecutar la API en desarrollo

```bash
uv run dev
```

La API estará disponible en `http://localhost:8000`.
Documentación interactiva en `http://localhost:8000/docs`.

### 5. Ejecutar pruebas

```bash
uv run pytest
```
