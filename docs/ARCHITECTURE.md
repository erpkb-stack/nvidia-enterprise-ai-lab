# Architecture
Browser (Next.js) -> FastAPI -> PostgreSQL (projects/designs), Redis (future background jobs), Qdrant (future vector retrieval). NVIDIA services are called from backend adapters only.

Trust boundaries: validate all browser input; keep secrets on backend; enforce document access and tenant ownership before production; treat model output as untrusted; require human approval for consequential actions. Simulation is never hardware evidence.
