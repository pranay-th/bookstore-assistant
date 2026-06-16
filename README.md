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
| `JWT_SECRET`            | **Must equal the Django backend's `SECRET_KEY`**  |
| `JWT_ALGORITHM`         | JWT signing algorithm (default `HS256`)           |
| `JWT_USER_ID_CLAIM`     | Claim holding the user id (default `user_id`)     |
| `REQUIRE_AUTH`          | Require a Bearer token on AI endpoints (default `True`) |

## Authentication

`/chat` and `/recommendations` require a Bearer access token issued by the
Django backend's login flow (`POST /user/verify-otp/`):

```
Authorization: Bearer <access-token>
```

Tokens are **verified statelessly** — the backend signs them with
`djangorestframework-simplejwt` (HS256, Django `SECRET_KEY`), and this service
verifies the signature locally using the same secret. No extra network call to
the backend is made per request.

For this to work, set `JWT_SECRET` here to the **same value** as the backend's
`SECRET_KEY`. The service rejects expired tokens, bad signatures, refresh
tokens used as access tokens, and tokens missing the `user_id` claim — all with
`401 Unauthorized`. The authenticated `user_id` overrides any client-supplied
`user_id` in the request body.

Set `REQUIRE_AUTH=False` for local development to make the token optional
(a present-but-invalid token is still rejected). `/health` is always open.

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
| GET    | `/health`          | ✅ Implemented (no auth) |
| POST   | `/chat`            | ✅ Implemented (Bearer token) |
| POST   | `/recommendations` | ✅ Implemented (Bearer token) |

## Phase 1 Status

The agentic loop is live. `/chat` and `/recommendations` run the LLM
tool-calling loop against the Django catalog:

- **Tools** (`services/tools.py`): `search_books`, `get_book`,
  `list_books_by_author` — each calls the Django backend and unwraps its
  `{"status": ..., "data": ...}` response envelope.
- **Agent loop** (`services/agent_service.py`): builds `system + history +
  user` messages, advertises the tools, and dispatches tool calls until the
  model returns a final reply or `AGENT_MAX_ITERATIONS` is hit (after which it
  makes one tool-free call to force a final answer).
- **Recommendations** (`services/recommendation_service.py`): reuses the loop
  and asks the model for a strict JSON object that is parsed into
  `RecommendationResponse`.
- **Error handling**: a missing `LLM_API_KEY`, backend/HTTP failures, unknown
  tools, and bad tool arguments are all handled gracefully — the endpoints
  return `503` on agent failure rather than crashing.

Tests mock both the LLM client and the backend, so `pytest` runs with no
network access.
