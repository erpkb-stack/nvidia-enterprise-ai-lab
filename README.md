# NVIDIA Enterprise AI Solutions Architect Lab

Local-first engineering platform for enterprise AI architecture, RAG, model serving, GPU operations, benchmarking, and OEM validation.

## Included foundation
- Next.js + TypeScript dashboard with module navigation
- FastAPI backend and OpenAPI docs
- PostgreSQL persistence for projects and architecture blueprints
- Redis and Qdrant services in Docker Compose
- Technology and enterprise use-case catalogs
- Architecture blueprint generation and persistence
- Explicitly labeled inference benchmark simulation
- Backend-only NVIDIA NIM adapter seam

## Run locally
1. Install Docker Desktop, Node.js 20+, and Python 3.11+.
2. `cp .env.example .env`
3. Start infrastructure: `docker compose up -d postgres redis qdrant`
4. Backend:
   `cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && uvicorn app.main:app --reload`
5. Frontend in another terminal:
   `cd frontend && npm install && npm run dev`
6. Open http://localhost:3000 and API docs at http://localhost:8000/docs.

Or run all services with `docker compose up --build`.

## Integrity
Simulation data is not GPU measurement and is not an MLPerf result. Real NVIDIA integrations require compatible hardware, supported software versions, credentials, and validation. Secrets must remain on the backend.
