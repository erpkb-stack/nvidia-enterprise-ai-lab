# Roadmap

## Done (v1.0)

- **Foundation:** dashboard, API, Alembic migrations, readiness and liveness checks, Prometheus
  metrics, structured logs, CI.
- **Enterprise RAG Assistant:**
  - ingestion, page-aware chunking and hybrid retrieval
  - NVIDIA embeddings and reranking, with offline fallbacks
  - citations, audit log and retrieval evaluation
- **LLM Inference Performance Lab:** analytical model, sweeps, comparisons, capacity planner, live
  streaming load test.
- **Solution Architect Studio:** requirements to architecture, stack, sizing, plan, risks and a
  Markdown design document.
- **GPU & Kubernetes Ops Lab:**
  - NIM manifests with exclusive, time-slicing or MIG GPU sharing
  - distributed-inference planner
  - Xid, CUDA and NCCL diagnosis
  - `nvidia-smi` analysis
- **OEM Validation Center:** 14-rule validator and customer-ready POC reports.

## Next

1. **Security:** OIDC login, RBAC, per-collection document ACLs enforced in retrieval, tenant isolation.
2. **RAG:**
   - background ingestion with Redis/RQ and progress events
   - OCR for scanned PDFs
   - NeMo Guardrails integration
   - streaming answers
   - LLM-as-judge groundedness evaluation
3. **Serving:**
   - Triton model repository management
   - TensorRT-LLM engine build tracking
   - A/B model routing
   - OpenTelemetry tracing
4. **Operations:**
   - Kubernetes client for live cluster inspection (GPU Operator status, node GPU capacity)
   - DCGM metrics ingestion
   - Run:ai connector
5. **Benchmark & OEM:**
   - GenAI-Perf integration
   - telemetry captured during live runs
   - report export to PDF
   - comparing measured results with the analytical estimates
6. **Hardening:**
   - secrets manager
   - image signing and SBOMs
   - dependency scanning
   - backups
   - load testing at target concurrency
