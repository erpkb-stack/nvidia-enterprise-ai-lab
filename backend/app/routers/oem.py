import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..data.hardware import GPUS, MODELS
from ..database import get_db
from ..models import ValidationReport
from ..services import oem

router = APIRouter(prefix="/api/v1/oem", tags=["OEM Solution Validation Center"])


class ConfigIn(BaseModel):
    rack_units: int = Field(2, ge=1, le=10)
    gpu: str = "l40s"
    gpu_count: int = Field(4, ge=1, le=16)
    cpu_sockets: int = Field(2, ge=1, le=8)
    cpu_cores_per_socket: int = Field(32, ge=1, le=256)
    cpu_tdp_w: int = Field(300, ge=50, le=1000)
    system_memory_gb: int = Field(512, ge=16, le=32768)
    network_ports: int = Field(2, ge=1, le=32)
    network_speed_gbps: int = Field(100, ge=1, le=1600)
    network_type: Literal["ethernet", "infiniband"] = "ethernet"
    gpudirect_rdma: bool = False
    nvme_tb: float = Field(7.68, ge=0, le=2000)
    psu_count: int = Field(2, ge=1, le=16)
    psu_watts: int = Field(2400, ge=200, le=10000)
    redundancy: Literal["N+1", "N+N", "none"] = "N+1"
    os: str = Field("Ubuntu 22.04", max_length=60)
    driver_version: str = Field("550.127.05", max_length=30, pattern=r"^[0-9.]+$")
    cuda_version: str = Field("12.4", max_length=10, pattern=r"^[0-9.]+$")
    kubernetes_version: str = Field("1.30", max_length=10, pattern=r"^[0-9.]*$")
    workload: Literal["inference", "rag", "fine_tuning", "training"] = "rag"
    multi_node: bool = False
    model: str | None = "llama-3.1-8b"
    precision: Literal["fp16", "bf16", "fp8", "int8", "int4"] = "fp8"


def _check(c: ConfigIn) -> dict:
    if c.gpu not in GPUS:
        raise HTTPException(422, "Unknown GPU")
    if c.model and c.model not in MODELS:
        raise HTTPException(422, "Unknown model")
    return c.model_dump()


@router.get("/presets")
def presets():
    return oem.PRESETS


@router.post("/validate")
def validate(data: ConfigIn):
    return oem.validate(_check(data))


class ReportIn(BaseModel):
    config: ConfigIn
    customer: str = Field("Example Customer", min_length=1, max_length=180)
    title: str = Field("Enterprise AI Proof of Concept Report", min_length=3, max_length=255)
    prepared_by: str = Field("Solutions Architecture", max_length=180)
    objectives: str = Field("", max_length=4000)
    acceptance_latency: str = Field("p95 TTFT <= 2 s and >= 20 tokens/s per user at target "
                                    "concurrency", max_length=300)


@router.post("/reports", status_code=201)
def create_report(data: ReportIn, db: Session = Depends(get_db)):
    cfg = _check(data.config)
    v = oem.validate(cfg)
    md = oem.report_markdown(cfg, v, data.model_dump(exclude={"config"}))
    r = ValidationReport(customer=data.customer, title=data.title, status=v["status"],
                         config=cfg, checks=v["checks"], markdown=md)
    db.add(r)
    db.commit()
    return {"id": r.id, "status": v["status"], "score": v["score"], "markdown": md,
            "html_url": f"/api/v1/oem/reports/{r.id}/html",
            "markdown_url": f"/api/v1/oem/reports/{r.id}/markdown"}


@router.get("/reports")
def list_reports(db: Session = Depends(get_db)):
    rows = db.scalars(select(ValidationReport).order_by(ValidationReport.id.desc()).limit(50))
    return [{"id": r.id, "customer": r.customer, "title": r.title, "status": r.status,
             "created_at": r.created_at} for r in rows]


def _get(db: Session, rid: int) -> ValidationReport:
    r = db.get(ValidationReport, rid)
    if not r:
        raise HTTPException(404, "Report not found")
    return r


def _fname(r: ValidationReport, ext: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"{r.customer}-poc-report-{r.id}".lower()).strip("-")
    return f"{slug}.{ext}"


@router.get("/reports/{rid}/markdown", response_class=PlainTextResponse)
def report_md(rid: int, db: Session = Depends(get_db)):
    r = _get(db, rid)
    return PlainTextResponse(r.markdown, media_type="text/markdown", headers={
        "Content-Disposition": f'attachment; filename="{_fname(r, "md")}"'})


@router.get("/reports/{rid}/html", response_class=HTMLResponse)
def report_html(rid: int, download: bool = False, db: Session = Depends(get_db)):
    r = _get(db, rid)
    headers = {"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{_fname(r, "html")}"'
    return HTMLResponse(oem.markdown_to_html(r.markdown, r.title), headers=headers)
