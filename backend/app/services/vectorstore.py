"""Qdrant vector store wrapper. One Qdrant collection per (knowledge collection, embedding provider)."""

import logging
import re
import threading

from qdrant_client import QdrantClient, models

from ..config import settings

log = logging.getLogger(__name__)
_client: QdrantClient | None = None
_lock = threading.Lock()


class VectorStoreUnavailable(RuntimeError):
    pass


def client() -> QdrantClient:
    global _client
    with _lock:
        if _client is None:
            if settings.qdrant_url:
                _client = QdrantClient(
                    url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=10
                )
            else:
                _client = QdrantClient(":memory:")
        return _client


def collection_name(collection: str, provider: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", f"lab_{collection}_{provider}".lower())


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # network / server errors from qdrant-client vary by transport
        log.warning("qdrant call failed: %s", type(exc).__name__)
        raise VectorStoreUnavailable(
            f"Vector store (Qdrant) is unavailable: {type(exc).__name__}. Is it running?"
        ) from exc


def ensure(name: str, dim: int) -> None:
    c = client()
    if not _call(c.collection_exists, name):
        _call(
            c.create_collection,
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        _call(
            c.create_payload_index,
            collection_name=name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )


def upsert(name: str, ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
    points = [models.PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads, strict=True)]
    for i in range(0, len(points), 256):
        _call(client().upsert, collection_name=name, points=points[i : i + 256], wait=True)


def search(name: str, vector: list[float], limit: int) -> list[tuple[int, float]]:
    c = client()
    if not _call(c.collection_exists, name):
        return []
    res = _call(c.query_points, collection_name=name, query=vector, limit=limit)
    return [(int(p.id), float(p.score)) for p in res.points]


def delete_document(document_id: int) -> None:
    c = client()
    flt = models.Filter(
        must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
    )
    for col in _call(c.get_collections).collections:
        if col.name.startswith("lab_"):
            _call(c.delete, collection_name=col.name, points_selector=models.FilterSelector(filter=flt))


def drop(name: str) -> None:
    c = client()
    if _call(c.collection_exists, name):
        _call(c.delete_collection, name)


def ping() -> bool:
    try:
        client().get_collections()
        return True
    except Exception:
        return False
