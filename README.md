# NVIDIA Enterprise AI Solutions Architect Lab

A local-first engineering platform that demonstrates the end-to-end work of an enterprise AI solutions
architect on NVIDIA accelerated computing: building retrieval-augmented assistants, engineering inference
performance, designing solutions, operating GPU clusters and validating OEM platforms.

![CI](https://github.com/erpkb-stack/nvidia-enterprise-ai-lab/actions/workflows/ci.yml/badge.svg)

## Five flagship demonstrations

| # | Lab | What it does |
|---|-----|--------------|
| 1 | **Enterprise RAG Assistant** (`/rag`) | Upload PDF/Markdown/HTML/TXT/CSV/JSON → parse, chunk (page-aware), embed, index in Qdrant. Hybrid retrieval (dense vectors + BM25, reciprocal-rank fusion) → reranking → grounded answer with clickable page-level citations. SHA-256 de-duplication, per-stage latency, query audit log, retrieval evaluation (hit-rate@k, MRR), re-indexing when the embedder changes. |
| 2 | **LLM Inference Performance Lab** (`/benchmark`) | Roofline model of prefill/decode: GPU memory (weights, KV cache), TTFT, time-per-token, throughput, memory- vs compute-bound decode. Batch sweeps, quantization (FP16/FP8/INT8/INT4), FP8 KV cache, tensor parallelism, scenario comparison, SLO-driven capacity planner, and a **live streaming load test** (TTFT, inter-token latency, p50/p90/p99) against a real NVIDIA endpoint. |
| 3 | **Solution Architect Studio** (`/architect`) | Business requirements → reference architecture diagram, technology selection with rationale (NIM, TensorRT-LLM, NeMo Retriever, Guardrails, GPU Operator, DCGM…), GPU sizing that meets the latency SLO, vector-store sizing, phased delivery plan, risk register. Exports a Markdown design document and saves to a project. |
| 4 | **GPU & Kubernetes Ops Lab** (`/mlops`) | Generates hardened NIM Kubernetes manifests (GPU requests, node selectors, tolerations, probes, PVC cache, PDB, HPA) with exclusive / time-slicing / MIG sharing and a runbook. Distributed-inference planner (TP within node, PP across nodes). Failure diagnosis for Xid codes, CUDA OOM, driver/CUDA mismatch, container toolkit, scheduling, NCCL. `nvidia-smi` fleet health analysis. |
| 5 | **OEM Solution Validation Center** (`/oem`) | Validates a server configuration against 14 reference rules (HGX/PCIe topology, memory ratio, cores per GPU, PSU redundancy and power, thermal density, east-west fabric, NVMe, driver↔CUDA, architecture support, OS, Kubernetes, model fit). Generates a customer-ready POC report (Markdown + printable HTML) with test plan, acceptance criteria and sign-off. |

## Integrity

- Everything works **offline**. Without an NVIDIA key the RAG lab uses lexical hashing embeddings and
  extractive answers, and every badge says so.
- Analytical numbers are labelled `ANALYTICAL_ESTIMATE`. Only live load tests are labelled
  `LIVE_MEASUREMENT`, and they measure the configured endpoint from the client, including the network.
  Nothing here is an MLPerf result.
- An OEM "VALIDATED" result means the configuration passes this lab's reference rules. It is **not** an
  NVIDIA-Certified Systems designation.
- Secrets stay on the backend. The browser never sees the NVIDIA API key.

## Quick start (local development)

Prerequisites: Docker Desktop, Node.js 20+ and Python 3.11+.

```bash
cp .env.example backend/.env           # set NVIDIA_NIM_API_KEY to enable live NVIDIA inference
docker compose up -d postgres qdrant   # infrastructure

cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
alembic upgrade head                   # create / migrate the schema
uvicorn app.main:app --reload          # http://localhost:8000/docs

cd ../frontend                         # in another terminal
npm install && npm run dev             # http://localhost:3000
```

Or run the full stack in containers: `cp .env.example .env && docker compose up --build`.

### Enabling NVIDIA inference

| Setting | Effect |
|---|---|
| `NVIDIA_NIM_API_KEY` (from build.nvidia.com) | NVIDIA embeddings, reranking and LLM answers in RAG. Also enables the live benchmark. |
| `NVIDIA_NIM_BASE_URL=http://nim-host:8000/v1` | Uses a self-hosted NIM instead of the hosted catalog. |

After switching embedders, use **Re-index with current embedder** in the RAG lab.

## Architecture

```
Browser (Next.js 15, React 19)
   │  REST/JSON
FastAPI (routers per lab) ── services ──┬─ PostgreSQL (SQLAlchemy 2 + Alembic): projects, designs,
   │                                    │    documents/chunks, query audit, benchmark runs, reports
   │                                    ├─ Qdrant: one vector collection per (knowledge collection, embedder)
   │                                    └─ NVIDIA endpoints (backend only): chat, embeddings, reranking
   └─ /metrics (Prometheus), /api/v1/health (liveness), /api/v1/health/ready (readiness)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for design decisions and [docs/ROADMAP.md](docs/ROADMAP.md)
for what's next.

## Production readiness

- **Schema:** Alembic migrations. Containers run `alembic upgrade head` before they start.
- **Observability:** JSON structured logs with request IDs (`X-Request-ID`), Prometheus request
  counters and latency histograms keyed by route template, and liveness and readiness probes.
- **Error handling:** a consistent error envelope (`{"error": {"code", "message"}}`). Upstream NVIDIA
  failures return 502, a missing vector store returns 503, and validation errors return 422 with field
  details.
- **Input safety:**
  - Upload size, type and magic-byte checks.
  - Collection-name and DNS-label validation.
  - Bounded request sizes, and guard rails on live load tests.
  - Retrieved text is treated as data in the LLM prompt.
  - Report HTML is escaped and served with a strict CSP.
- **Containers:** non-root users, health checks, a standalone Next.js server, and pinned base images.
- **CI** (GitHub Actions):
  - Backend: ruff lint, pytest, and a migration up, down and up again on PostgreSQL.
  - Frontend: type check and production build.
  - Docker image builds.

## Tests

```bash
cd backend && pytest -q        # 31 tests, offline (SQLite + in-memory Qdrant)
cd frontend && npx tsc --noEmit && npm run build
```
