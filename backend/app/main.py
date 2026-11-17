from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select
from .config import settings
from .database import Base, engine, get_db
from .models import Project, Architecture
from .catalog import TECHNOLOGIES, USE_CASES
from .adapters import get_adapter

Base.metadata.create_all(bind=engine)
app = FastAPI(title="NVIDIA Enterprise AI Solutions Architect Lab", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.backend_cors_origins.split(",")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ProjectIn(BaseModel):
    name: str = Field(min_length=2, max_length=180)
    customer: str = "Internal Lab"
    use_case: str = "Enterprise Knowledge Assistant"

class ArchitectureIn(BaseModel):
    project_id: int
    title: str
    blueprint: dict
    rationale: str = ""

class BenchmarkIn(BaseModel):
    model: str = "Nemotron sample profile"
    concurrency: int = Field(8, ge=1, le=256)
    input_tokens: int = Field(512, ge=1, le=32768)
    output_tokens: int = Field(128, ge=1, le=8192)
    batch_size: int = Field(1, ge=1, le=256)

@app.get("/")
def root(): return {"name":app.title,"docs":"/docs"}

@app.get("/api/v1/health")
def health(): return {"status":"ok","environment":settings.app_env,"gpu_execution":"not_verified"}

@app.get("/api/v1/technologies")
def technologies(): return TECHNOLOGIES

@app.get("/api/v1/use-cases")
def use_cases(): return USE_CASES

@app.get("/api/v1/projects")
def projects(db: Session = Depends(get_db)):
    return [{"id":p.id,"name":p.name,"customer":p.customer,"use_case":p.use_case,"status":p.status,"created_at":p.created_at} for p in db.scalars(select(Project).order_by(Project.id.desc())).all()]

@app.post("/api/v1/projects")
def create_project(data: ProjectIn, db: Session = Depends(get_db)):
    p=Project(**data.model_dump()); db.add(p); db.commit(); db.refresh(p)
    return {"id":p.id,"name":p.name,"customer":p.customer,"use_case":p.use_case,"status":p.status,"created_at":p.created_at}

@app.post("/api/v1/architectures")
def save_architecture(data: ArchitectureIn, db: Session = Depends(get_db)):
    if not db.get(Project,data.project_id): raise HTTPException(404,"Project not found")
    a=Architecture(**data.model_dump()); db.add(a); db.commit(); db.refresh(a)
    return {"id":a.id,"project_id":a.project_id,"title":a.title,"blueprint":a.blueprint}

@app.get("/api/v1/architectures")
def architectures(project_id: int | None = None, db: Session = Depends(get_db)):
    q=select(Architecture).order_by(Architecture.id.desc())
    if project_id is not None: q=q.where(Architecture.project_id==project_id)
    return [{"id":a.id,"project_id":a.project_id,"title":a.title,"blueprint":a.blueprint,"rationale":a.rationale} for a in db.scalars(q).all()]

@app.post("/api/v1/benchmarks/simulate")
def benchmark(data: BenchmarkIn):
    work=data.input_tokens+data.output_tokens
    tps=round(data.concurrency*data.batch_size*1000/max(1,work/80),2)
    latency=round((data.input_tokens/180+data.output_tokens/65)*(1+data.concurrency/100),2)
    return {"mode":"SIMULATION","warning":"Synthetic estimate only; not NVIDIA hardware measurement and not MLPerf.","model":data.model,"estimated_tokens_per_second":tps,"estimated_latency_seconds":latency,"p50_ms":round(latency*800,1),"p95_ms":round(latency*1250,1),"p99_ms":round(latency*1500,1)}

@app.get("/api/v1/inference/health")
async def inference_health(): return await get_adapter().health()

@app.post("/api/v1/inference/demo")
async def inference_demo(data: dict):
    prompt=str(data.get("prompt","")).strip()
    if not prompt: raise HTTPException(400,"prompt is required")
    if len(prompt)>12000: raise HTTPException(413,"prompt too long")
    return await get_adapter().generate(prompt)
