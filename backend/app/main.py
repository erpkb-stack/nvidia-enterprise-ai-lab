import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .database import Base, engine
from .routers import architect, core, gpu_ops, inference_lab, oem, rag
from .services.nvidia import UpstreamError
from .services.perf_model import PerfInputError
from .services.rag_ingest import IngestError
from .services.vectorstore import VectorStoreUnavailable

VERSION = "1.0.0"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name,
                "msg": record.getMessage()}
        for key in ("request_id", "method", "path", "status", "duration_ms"):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)


_configure_logging()
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.auto_create_tables:
        Base.metadata.create_all(bind=engine)
    log.info("startup env=%s llm_enabled=%s", settings.app_env, settings.llm_enabled)
    yield


app = FastAPI(
    title="NVIDIA Enterprise AI Solutions Architect Lab",
    version=VERSION,
    description="Enterprise RAG, inference performance engineering, solution architecture, "
                "GPU/Kubernetes operations and OEM validation. Analytical estimates and simulations "
                "are always labelled; they are never presented as hardware measurements.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID", "Content-Disposition"],
)


REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["method", "route"],
                    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))


def _route_template(request: Request) -> str:
    # Use the route template (e.g. /api/v1/rag/documents/{doc_id}) to keep label cardinality bounded.
    route = request.scope.get("route")
    return getattr(route, "path", None) or "unmatched"


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    duration = round(elapsed * 1000, 1)
    if request.url.path != "/metrics":
        route = _route_template(request)
        REQUESTS.labels(request.method, route, str(response.status_code)).inc()
        LATENCY.labels(request.method, route).observe(elapsed)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    if request.url.path not in ("/metrics", "/api/v1/health"):
        log.info("request", extra={"request_id": rid, "method": request.method,
                                   "path": request.url.path, "status": response.status_code,
                                   "duration_ms": duration})
    return response


def _error(status: int, code: str, message: str, request: Request, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}, "detail": message}
    if details is not None:
        body["error"]["details"] = details
    rid = request.headers.get("X-Request-ID")
    return JSONResponse(body, status_code=status, headers={"X-Request-ID": rid} if rid else None)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return _error(exc.status_code, "http_error", str(exc.detail), request)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    details = [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]} for e in exc.errors()]
    msg = "; ".join(f"{d['loc']}: {d['msg']}" for d in details[:3])
    return _error(422, "validation_error", msg, request, details)


@app.exception_handler(IngestError)
async def ingest_error(request: Request, exc: IngestError):
    return _error(422, "ingest_error", str(exc), request)


@app.exception_handler(PerfInputError)
async def perf_error(request: Request, exc: PerfInputError):
    return _error(422, "invalid_scenario", str(exc), request)


@app.exception_handler(UpstreamError)
async def upstream_error(request: Request, exc: UpstreamError):
    return _error(502, "upstream_error", str(exc), request)


@app.exception_handler(VectorStoreUnavailable)
async def vector_error(request: Request, exc: VectorStoreUnavailable):
    return _error(503, "vector_store_unavailable", str(exc), request)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error", extra={"path": request.url.path})
    return _error(500, "internal_error", "Internal server error", request)


for r in (core.router, rag.router, inference_lab.router, architect.router, gpu_ops.router, oem.router):
    app.include_router(r)


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def root():
    return {"name": app.title, "version": VERSION, "docs": "/docs", "metrics": "/metrics"}
