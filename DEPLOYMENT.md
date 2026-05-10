# ALIA Deployment Guide

## Architecture

```
User → Nginx (frontend:80)
           ↓ /api/*
       FastAPI (backend:8000)
           ↓
       MongoDB + Redis + Pinecone + Groq
```

The NLP pipeline runs **in-process** inside the backend container. Inference falls through a chain:
1. Local QLoRA adapter (Qwen2.5-1.5B + PEFT)
2. Groq cloud (Llama-3.3-70B) — fallback when adapter is disabled or fails
3. Rule-based baseline — always available, no external dependency

---

## Prerequisites

- Docker ≥ 24 and Docker Compose V2
- Groq API key ([console.groq.com](https://console.groq.com))
- Pinecone API key ([app.pinecone.io](https://app.pinecone.io)) with an index named `alia-knowledge`
- Auth0 tenant (optional — only needed for Auth0 login flow)

---

## Quick Start (local, with Docker)

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd alia-web-main

# 2. Create your env file
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY, PINECONE_API_KEY, JWT_SECRET at minimum

# 3. Build and start all services
docker compose up --build

# 4. Verify
curl http://localhost:8000/health   # backend
open http://localhost               # frontend
```

Services:
| Service  | Local URL               |
|----------|-------------------------|
| Frontend | http://localhost         |
| Backend  | http://localhost:8000    |
| MongoDB  | mongodb://localhost:27017 |
| Redis    | redis://localhost:6379   |

---

## Required Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq cloud API key |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | Yes | Default: `alia-knowledge` |
| `JWT_SECRET` | Yes | Long random string for JWT signing |
| `MONGODB_URL` | Yes | Overridden by docker-compose to `mongodb://mongo:27017/alia` |
| `REDIS_URL` | Yes | Overridden by docker-compose to `redis://redis:6379/0` |
| `CORS_ORIGINS` | Yes | Comma-separated allowed origins (e.g. `https://yourdomain.com`) |
| `ADMIN_SECRET_KEY` | No | Required to register admin accounts; leave blank to disable |
| `AUTH0_DOMAIN` | No | Auth0 domain (e.g. `dev-xxx.us.auth0.com`) |
| `AUTH0_CLIENT_ID` | No | Auth0 client ID |
| `AUTH0_CLIENT_SECRET` | No | Auth0 client secret |

Full reference: [`.env.example`](.env.example)

---

## Production Deployment

### Option A: Single server (VPS/VM)

```bash
# Install Docker on the server
curl -fsSL https://get.docker.com | sh

# Copy files to server
scp -r . user@server:/opt/alia

# On the server
cd /opt/alia
cp .env.example .env && nano .env   # fill in secrets

# Set production CORS to your actual domain
echo 'CORS_ORIGINS=https://yourdomain.com' >> .env

docker compose up -d --build
```

### Option B: Azure Container Apps

1. Push images to Azure Container Registry:
   ```bash
   az acr build --registry <acr-name> --image alia-backend:latest -f backend/Dockerfile .
   az acr build --registry <acr-name> --image alia-frontend:latest -f frontend/Dockerfile ./frontend
   ```
2. Create a Container App for each image, pointing to your managed MongoDB and Redis instances.
3. Store secrets in Azure Key Vault and reference them as environment variables.

---

## Updating the QLoRA Adapter

When a new adapter is trained and passes CI gates:

1. Replace files in `alia_nlp/models/adapters/`:
   - `adapter_config.json`
   - `adapter_model.safetensors`

2. Update `alia_nlp/models/adapters/model_card.json` with:
   - New `training_date`
   - Updated `eval_metrics`
   - Increment `model_id` version suffix

3. Rebuild and redeploy the backend container:
   ```bash
   docker compose up -d --build backend
   ```

4. Watch startup logs to confirm the adapter loaded cleanly:
   ```bash
   docker compose logs -f backend | grep -i "affect\|adapter\|warm"
   ```

---

## Health & Monitoring

**Backend health endpoint:**
```
GET /health
```
Returns:
```json
{
  "status": "ok",
  "db": "ok",
  "vector_db": "ok",
  "embedding": "ok"
}
```

**Shadow monitoring** runs daily at 02:00 UTC. Snapshots are stored in MongoDB under the `shadow_monitoring` collection. Divergence threshold is controlled by `SHADOW_MAX_DIVERGENCE` (default: `0.08`).

**Check container health:**
```bash
docker compose ps
docker compose logs backend --tail=50
```

---

## Rollback

```bash
# Roll back backend to a previous image tag
docker compose stop backend
docker tag alia-backend:<previous-sha> alia-backend:latest
docker compose up -d backend
```

---

## CI/CD Pipeline

| Workflow | Trigger | Gates |
|---|---|---|
| `nlp-eval.yml` | push / PR | Intent ≥90%, Safety recall ≥95%, Retrieval hit@1 ≥70% |
| `deploy.yml` | push to main | Docker build passes, smoke test (`/health` + `/`) passes |

Both must pass before a deployment is considered safe.
