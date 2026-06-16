# bookstore-assistant

AI-native service for the Enterprise Book Store platform. Powers the
conversational shopping assistant and AI recommendations using an **agentic
tool-calling loop** — the LLM calls backend tools to read live catalog data
rather than relying on a separate embedding pipeline or vector store.

## Why tool calling (not embeddings)

The Django backend already exposes catalog search, filtering, and detail
endpoints. Instead of duplicating that data into a vector index, the assistant
lets the model call those endpoints as tools and reason over fresh results.
This is simpler to operate (no embedding/indexing jobs), always up to date, and
easy to extend with new tools (orders, inventory, trending).

## Tech Stack

- **FastAPI** — ASGI web framework
- **Pydantic v2** — data validation
- **OpenAI SDK** — OpenAI-compatible client (pointed at OpenRouter) for LLM + tool calling
- **httpx** — calls into the Django backend and analytics service
- **Railway** — hosting platform

## Local Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment file
cp .env.example .env
# Fill in LLM_API_KEY (OpenRouter), DJANGO_API_URL

# 4. Start the development server (http://localhost:8002)
uvicorn app.main:app --reload --port 8002
```

## API Docs

- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

## Environment Variables

| Variable                | Description                                       |
|-------------------------|---------------------------------------------------|
| `LLM_API_KEY`           | OpenRouter API key                                |
| `LLM_BASE_URL`          | OpenAI-compatible base URL (OpenRouter)           |
| `LLM_MODEL`             | Model slug used for the agent loop                |
| `AGENT_MAX_ITERATIONS`  | Safety bound on tool-call iterations per turn     |
| `DJANGO_API_URL`        | Base URL of the Django backend (tool calls)       |
| `ANALYTICS_SERVICE_URL` | Base URL of the analytics microservice            |
| `CORS_ORIGINS`          | Comma-separated allowed CORS origins              |
| `APP_DEBUG`             | `True` for development, `False` for prod          |

## Architecture

```
app/
├── main.py                  FastAPI app + router wiring
├── core/
│   ├── config.py            Settings (pydantic-settings)
│   ├── llm.py               LLM client provider
│   └── backend_client.py    httpx client for the Django backend
├── routers/
│   ├── health.py            GET /health
│   ├── chat.py              POST /chat
│   └── recommendations.py   POST /recommendations
├── services/
│   ├── agent_service.py     Agentic tool-calling loop
│   ├── tools.py             Tool schemas + implementations
│   └── recommendation_service.py
└── schemas/                 Request/response models
```

## How the agent loop works

1. `POST /chat` receives a message + history.
2. `AgentService.run` sends it to the LLM, advertising the tools in
   `services/tools.py` (`TOOL_SPECS`).
3. If the model requests tool calls, they are dispatched to `TOOL_IMPLS`
   (which call the Django backend), and results are fed back to the model.
4. The loop repeats up to `AGENT_MAX_ITERATIONS` until the model returns a
   final reply.

## Running Tests

```bash
pytest app/tests/ -v
```

## Deployment

### Render (primary)

Config lives in `render.yaml` (Render Blueprint). It runs as a native Python
web service:

- **Build:** `pip install -r requirements.txt`
- **Start:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Health check:** `GET /health`

Steps:
1. In Render, create a new Blueprint and point it at this repo — it reads
   `render.yaml` automatically.
2. Set the secret env vars marked `sync: false` (`LLM_API_KEY`,
   `DJANGO_API_URL`, `ANALYTICS_SERVICE_URL`, `CORS_ORIGINS`) in the Render
   dashboard.
3. Push to `main` to deploy. The `.github/workflows/deploy.yml` workflow will
   ping `RENDER_DEPLOY_HOOK_URL` (add it as a GitHub Actions secret) to trigger
   redeploys.

### Docker / Railway (alternative)

A `Dockerfile` and `railway.json` are also included for container-based or
Railway deployments. Health probe endpoint: `GET /health` → `{"status": "ok"}`.

## Endpoint Overview

| Method | Path               | Status          |
|--------|--------------------|-----------------|
| GET    | `/health`          | ✅ Implemented  |
| POST   | `/chat`            | 501 Placeholder |
| POST   | `/recommendations` | 501 Placeholder |

## Phase 0 Status

Foundation skeleton only. Health endpoint is functional; chat and
recommendation endpoints return `501 Not Implemented`. The agentic loop and
tool implementations land in Phase 1.
