from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ..catalog import TECHNOLOGIES, USE_CASES
from ..config import settings
from ..database import get_db
from ..models import Architecture, Project
from ..services import nvidia, vectorstore

router = APIRouter(prefix="/api/v1", tags=["Core"])


class ProjectIn(BaseModel):
    name: str = Field(min_length=2, max_length=180)
    customer: str = Field("Internal Lab", max_length=180)
    use_case: str = Field("Enterprise Knowledge Assistant", max_length=120)


class ArchitectureIn(BaseModel):
    project_id: int
    title: str = Field(min_length=1, max_length=180)
    blueprint: dict
    rationale: str = ""


class BenchmarkIn(BaseModel):
    model: str = "Nemotron sample profile"
    concurrency: int = Field(8, ge=1, le=256)
    input_tokens: int = Field(512, ge=1, le=32768)
    output_tokens: int = Field(128, ge=1, le=8192)
    batch_size: int = Field(1, ge=1, le=256)


class ChatIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    max_tokens: int = Field(256, ge=1, le=4096)


def _project(p: Project) -> dict:
    return {"id": p.id, "name": p.name, "customer": p.customer, "use_case": p.use_case,
            "status": p.status, "created_at": p.created_at}


@router.get("/health")
def health():
    """Liveness: the process is up. Does not touch dependencies."""
    return {"status": "ok", "environment": settings.app_env, "gpu_execution": "not_verified"}


@router.get("/health/ready")
async def ready(db: Session = Depends(get_db)):
    """Readiness: dependencies the API needs. NVIDIA endpoint is optional (fallbacks exist)."""
    checks = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = {"ok": True, "backend": settings.database_url.split(":", 1)[0]}
    except Exception as exc:  # report, do not crash the probe
        checks["database"] = {"ok": False, "error": type(exc).__name__}
    checks["vector_store"] = {"ok": vectorstore.ping(),
                              "mode": "server" if settings.qdrant_url else "in-memory"}
    checks["nvidia_endpoint"] = await nvidia.probe()
    ok = checks["database"]["ok"] and checks["vector_store"]["ok"]
    return {"status": "ready" if ok else "degraded", "checks": checks}


@router.get("/technologies")
def technologies():
    return TECHNOLOGIES


@router.get("/use-cases")
def use_cases():
    return USE_CASES


@router.get("/projects")
def projects(db: Session = Depends(get_db)):
    return [_project(p) for p in db.scalars(select(Project).order_by(Project.id.desc()))]


@router.post("/projects", status_code=200)
def create_project(data: ProjectIn, db: Session = Depends(get_db)):
    p = Project(**data.model_dump())
    db.add(p)
    db.commit()
    return _project(p)


@router.post("/architectures")
def save_architecture(data: ArchitectureIn, db: Session = Depends(get_db)):
    if not db.get(Project, data.project_id):
        raise HTTPException(404, "Project not found")
    a = Architecture(**data.model_dump())
    db.add(a)
    db.commit()
    return {"id": a.id, "project_id": a.project_id, "title": a.title, "blueprint": a.blueprint}


@router.get("/architectures")
def architectures(project_id: int | None = None, db: Session = Depends(get_db)):
    q = select(Architecture).order_by(Architecture.id.desc())
    if project_id is not None:
        q = q.where(Architecture.project_id == project_id)
    return [{"id": a.id, "project_id": a.project_id, "title": a.title, "blueprint": a.blueprint,
             "rationale": a.rationale, "created_at": a.created_at} for a in db.scalars(q)]


@router.post("/benchmarks/simulate", deprecated=True)
def benchmark(data: BenchmarkIn):
    """Legacy synthetic estimate. Use /api/v1/inference-lab/estimate instead."""
    work = data.input_tokens + data.output_tokens
    tps = round(data.concurrency * data.batch_size * 1000 / max(1, work / 80), 2)
    latency = round((data.input_tokens / 180 + data.output_tokens / 65) * (1 + data.concurrency / 100), 2)
    return {"mode": "SIMULATION", "warning": "Synthetic estimate only; not NVIDIA hardware "
            "measurement and not MLPerf.", "model": data.model, "estimated_tokens_per_second": tps,
            "estimated_latency_seconds": latency, "p50_ms": round(latency * 800, 1),
            "p95_ms": round(latency * 1250, 1), "p99_ms": round(latency * 1500, 1)}


@router.get("/inference/health")
async def inference_health():
    return await nvidia.probe()


@router.post("/inference/chat")
async def inference_chat(data: ChatIn):
    if not settings.llm_enabled:
        raise HTTPException(409, "No NVIDIA endpoint configured (set NVIDIA_NIM_API_KEY).")
    return await nvidia.chat([{"role": "user", "content": data.prompt}], max_tokens=data.max_tokens)
