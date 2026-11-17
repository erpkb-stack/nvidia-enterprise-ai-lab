# Architecture

## Components

| Layer | Technology | Notes |
|---|---|---|
| UI | Next.js 15 (App Router), React 19, TypeScript | Client components per lab; dependency-free SVG charts and architecture diagrams. |
| API | FastAPI | One router per lab (`app/routers`), business logic in `app/services`, reference data in `app/data`. |
| Relational store | PostgreSQL 16 (SQLite for tests) | SQLAlchemy 2 ORM, Alembic migrations. Source of truth for documents and chunks. |
| Vector store | Qdrant | Cosine HNSW; collection name = `lab_<collection>_<embedder>`, so switching embedders never mixes vector spaces. |
| Inference | NVIDIA API catalog or self-hosted NIM | OpenAI-compatible chat (default `nemotron-3.5-bolt-e4b`), embeddings (default `nemotron-3.5-embed-1b`) and optional reranking. Models are configurable because catalog models are retired over time. Called only from the backend. |
| Cache / jobs | Redis | Provisioned for background ingestion and caching (roadmap). |

## RAG pipeline

1. **Ingest:**
   1. Check the file's extension, magic bytes and size limit, then de-duplicate by SHA-256 within the collection.
   2. Parse it: PDF page by page with pypdf; HTML to text.
   3. Chunk on paragraph and sentence boundaries, never across pages, with overlap.
2. **Index:** embed the chunks as `passage` and upsert them to Qdrant with a `document_id` payload. Chunk
   text goes in PostgreSQL, so you can re-index with a new embedder without re-uploading.
3. **Retrieve:**
   1. Embed the question as `query`.
   2. Run vector search (top 4k) and in-memory BM25 over the collection's chunks (cached per collection version).
   3. Merge the two with reciprocal rank fusion (k = 60).
4. **Rerank:** use the NVIDIA reranker when it's configured. Otherwise rank by query-term coverage blended with RRF.
5. **Generate:**
   - The system prompt restricts answers to the numbered passages, requires `[n]` citations, and treats
     passage text as data, not instructions.
   - Without an LLM, the lab gives an extractive answer: sentences ranked by overlap, with citations.
6. **Audit:** every question, answer, cited source, provider and latency is stored in `rag_queries`.

## Performance model

- **Prefill** is treated as compute-bound: `2 · active_params · input_tokens · batch / (peak_FLOPS · TP · MFU)`.
- **Decode** time per step is the larger of two costs:
  - memory: `(weights_read + KV_read) / (bandwidth · TP · utilization)`
  - compute: `2 · active_params · batch / (peak_FLOPS · TP · MFU)`

  plus the all-reduce cost of tensor parallelism (higher without NVLink).
- **KV cache per token** is `2 · layers · kv_heads · head_dim · bytes`.
- **Mixture-of-experts models:** the share of expert weights read per step rises with batch size.
- **FP8** compute is only used on GPUs that have FP8 tensor cores; on others the lab shows a warning.

## Trust boundaries & security

- Validate all browser input. Pydantic models, bounded sizes and allow-lists enforce this.
- Secrets stay on the backend. The NVIDIA key is never sent to the browser or written to logs.
- Treat model output and retrieved documents as untrusted. Answers are rendered as text, never as HTML,
  and report HTML is escaped and served with `default-src 'none'`.
- Simulation and estimates are never evidence of hardware performance. Labels make the mode explicit.
- Before multi-tenant production, add OIDC authentication, per-collection ACL filtering at retrieval
  time, and tenant isolation. See the roadmap.
