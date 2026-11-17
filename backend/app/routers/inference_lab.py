from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..data.hardware import gpu_list, model_list
from ..database import get_db
from ..models import BenchmarkRun
from ..services import live_bench, perf_model

router = APIRouter(prefix="/api/v1/inference-lab", tags=["LLM Inference Performance Lab"])

Precision = Literal["fp16", "bf16", "fp8", "int8", "int4"]


class ScenarioIn(BaseModel):
    model: str = "llama-3.1-8b"
    gpu: str = "h100-sxm"
    weight_precision: Precision = "fp16"
    kv_precision: Literal["fp16", "bf16", "fp8"] = "fp16"
    tensor_parallel: Literal[1, 2, 4, 8] = 1
    batch_size: int = Field(8, ge=1, le=1024)
    input_tokens: int = Field(1024, ge=1, le=131072)
    output_tokens: int = Field(256, ge=1, le=32768)
    prefill_mfu: float = Field(0.45, gt=0.05, le=0.9)
    decode_mfu: float = Field(0.35, gt=0.05, le=0.9)
    bandwidth_utilization: float = Field(0.75, gt=0.1, le=0.95)
    label: str = ""

    def scenario(self) -> perf_model.Scenario:
        return perf_model.Scenario(**self.model_dump(exclude={"label"}))


@router.get("/catalog")
def catalog():
    return {"gpus": gpu_list(), "models": model_list(),
            "precisions": ["fp16", "bf16", "fp8", "int8", "int4"],
            "live_endpoint_configured": settings.llm_enabled}


@router.post("/estimate")
def estimate(data: ScenarioIn, save: bool = False, db: Session = Depends(get_db)):
    result = perf_model.estimate(data.scenario())
    result["sweep"] = perf_model.sweep(data.scenario())
    if save:
        run = BenchmarkRun(kind="analytical", label=data.label or "Analytical estimate",
                           params=data.model_dump(), results=result)
        db.add(run)
        db.commit()
        result["run_id"] = run.id
    return result


class CompareIn(BaseModel):
    scenarios: list[ScenarioIn] = Field(min_length=1, max_length=8)


@router.post("/compare")
def compare(data: CompareIn):
    out = []
    for i, s in enumerate(data.scenarios):
        r = perf_model.estimate(s.scenario())
        out.append({"label": s.label or f"Scenario {i + 1}", **r})
    return out


class CapacityIn(BaseModel):
    model: str = "llama-3.1-8b"
    concurrency: int = Field(100, ge=1, le=100000)
    input_tokens: int = Field(1024, ge=1, le=131072)
    output_tokens: int = Field(256, ge=1, le=32768)
    max_ttft_ms: float = Field(2000, gt=0)
    min_tokens_per_s_per_user: float = Field(20, gt=0)
    gpus: list[str] | None = None


@router.post("/capacity")
def capacity(data: CapacityIn):
    opts = perf_model.plan_capacity(
        data.model, data.concurrency, data.input_tokens, data.output_tokens,
        data.max_ttft_ms, data.min_tokens_per_s_per_user, data.gpus,
    )
    return {"mode": "ANALYTICAL_ESTIMATE", "options": opts[:15],
            "note": "Relative cost index is illustrative for ranking only, not pricing."}


class LiveIn(BaseModel):
    requests: int = Field(8, ge=1)
    concurrency: int = Field(4, ge=1)
    input_tokens: int = Field(256, ge=1, le=8192)
    output_tokens: int = Field(128, ge=1, le=2048)
    model: str | None = None
    label: str = ""


@router.post("/live")
async def live(data: LiveIn, db: Session = Depends(get_db)):
    if not settings.llm_enabled:
        raise HTTPException(409, "No live endpoint configured. Set NVIDIA_NIM_API_KEY or a "
                                 "self-hosted NVIDIA_NIM_BASE_URL in backend/.env.")
    if data.requests > settings.bench_max_requests or data.concurrency > settings.bench_max_concurrency:
        raise HTTPException(422, f"Limits: requests <= {settings.bench_max_requests}, "
                                 f"concurrency <= {settings.bench_max_concurrency}.")
    result = await live_bench.run(data.requests, min(data.concurrency, data.requests),
                                  data.input_tokens, data.output_tokens, data.model)
    run = BenchmarkRun(kind="live", label=data.label or "Live benchmark",
                       params=data.model_dump(), results=result)
    db.add(run)
    db.commit()
    result["run_id"] = run.id
    return result


@router.get("/runs")
def runs(limit: int = 25, db: Session = Depends(get_db)):
    rows = db.scalars(select(BenchmarkRun).order_by(BenchmarkRun.id.desc()).limit(min(limit, 200)))
    return [{"id": r.id, "kind": r.kind, "label": r.label, "params": r.params,
             "results": r.results, "created_at": r.created_at} for r in rows]
