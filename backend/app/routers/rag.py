from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models import Document, RagQuery
from ..services import rag
from ..services.rag_ingest import SUPPORTED_TYPES

router = APIRouter(prefix="/api/v1/rag", tags=["Enterprise RAG Assistant"])


def _doc(d: Document) -> dict:
    return {
        "id": d.id, "collection": d.collection, "filename": d.filename,
        "content_type": d.content_type, "size_bytes": d.size_bytes, "pages": d.pages,
        "chunk_count": d.chunk_count, "embedding_provider": d.embedding_provider,
        "status": d.status, "created_at": d.created_at, "sha256": d.sha256,
    }


@router.get("/config")
def rag_config():
    provider = rag.embedding_provider()
    return {
        "embedding_provider": provider,
        "embedding_label": rag.provider_label(provider),
        "rerank": "nvidia" if settings.rerank_enabled else "local_term_coverage",
        "generation": f"nvidia:{settings.nvidia_chat_model}" if settings.llm_enabled
        else "extractive_fallback",
        "max_upload_mb": settings.rag_max_upload_mb,
        "supported_types": sorted(SUPPORTED_TYPES),
        "chunk_chars": settings.rag_chunk_chars,
        "chunk_overlap": settings.rag_chunk_overlap,
    }


@router.get("/collections")
def collections(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Document.collection, func.count(Document.id), func.sum(Document.chunk_count))
        .group_by(Document.collection).order_by(Document.collection)
    ).all()
    return [{"name": r[0], "documents": r[1], "chunks": int(r[2] or 0)} for r in rows]


@router.get("/documents")
def documents(collection: str | None = None, db: Session = Depends(get_db)):
    q = select(Document).order_by(Document.id.desc())
    if collection:
        q = q.where(Document.collection == rag.validate_collection(collection))
    return [_doc(d) for d in db.scalars(q)]


@router.post("/documents", status_code=201)
async def upload(
    files: list[UploadFile] = File(...),
    collection: str = Form("default"),
    db: Session = Depends(get_db),
):
    if len(files) > 20:
        raise HTTPException(400, "Upload at most 20 files at a time.")
    limit = settings.rag_max_upload_mb * 1024 * 1024
    out = []
    for f in files:
        data = await f.read(limit + 1)
        doc, created = await rag.ingest(db, collection, f.filename or "document", data)
        out.append({**_doc(doc), "created": created})
    return out


@router.delete("/documents/{doc_id}", status_code=204)
def delete(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    rag.delete_document(db, doc)


@router.get("/documents/{doc_id}/chunks")
def chunks(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return [{"id": c.id, "ordinal": c.ordinal, "page": c.page, "text": c.text}
            for c in sorted(doc.chunks, key=lambda c: c.ordinal)]


class QueryIn(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    collection: str = "default"
    top_k: int = Field(5, ge=1, le=20)


@router.post("/query")
async def query(data: QueryIn, db: Session = Depends(get_db)):
    return await rag.answer(db, data.collection, data.question, data.top_k)


@router.get("/queries")
def query_log(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.scalars(select(RagQuery).order_by(RagQuery.id.desc()).limit(min(limit, 200)))
    return [{"id": q.id, "collection": q.collection, "question": q.question, "answer": q.answer,
             "providers": q.providers, "latency_ms": q.latency_ms, "created_at": q.created_at}
            for q in rows]


class EvalCase(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    expected_filename: str | None = None
    expected_text: str | None = None


class EvalIn(BaseModel):
    collection: str = "default"
    k: int = Field(5, ge=1, le=20)
    cases: list[EvalCase] = Field(min_length=1, max_length=100)


@router.post("/evaluate")
async def evaluate(data: EvalIn, db: Session = Depends(get_db)):
    for c in data.cases:
        if not (c.expected_filename or c.expected_text):
            raise HTTPException(422, "Each case needs expected_filename or expected_text.")
    return await rag.evaluate(db, data.collection, [c.model_dump() for c in data.cases], data.k)


@router.post("/collections/{name}/reindex")
async def reindex(name: str, db: Session = Depends(get_db)):
    return await rag.reindex(db, name)
