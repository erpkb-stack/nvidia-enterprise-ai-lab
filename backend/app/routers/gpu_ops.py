import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..data.hardware import GPUS, MODELS
from ..services import gpu_ops

router = APIRouter(prefix="/api/v1/gpu-ops", tags=["GPU & Kubernetes Operations Lab"])
DNS_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?$")


class ManifestIn(BaseModel):
    name: str = "llm-nim"
    namespace: str = "ai-inference"
    model: str = "llama-3.1-8b"
    gpu: str = "h100-sxm"
    precision: Literal["fp16", "bf16", "fp8", "int8", "int4"] = "fp16"
    gpus_per_replica: Literal[1, 2, 4, 8] = 1
    replicas: int = Field(2, ge=1, le=64)
    sharing: Literal["exclusive", "time_slicing", "mig"] = "exclusive"
    time_slices: int = Field(4, ge=2, le=16)
    mig_profile: str = "3g.40gb"
    image_tag: str = Field("latest", max_length=64, pattern=r"^[A-Za-z0-9._-]*$")
    storage_class: str = Field("", max_length=63)
    autoscale: bool = True
    target_inflight: int = Field(16, ge=1, le=1024)

    @field_validator("name", "namespace")
    @classmethod
    def dns(cls, v: str) -> str:
        if not DNS_LABEL.match(v):
            raise ValueError("must be a lowercase DNS label (a-z, 0-9, '-')")
        return v


@router.get("/catalog")
def catalog():
    return {"gpus": [{"id": g.id, "name": g.name, "mig": gpu_ops.MIG_PROFILES.get(g.id, []),
                      "product_label": gpu_ops.GPU_PRODUCT_LABEL[g.id]} for g in GPUS.values()],
            "models": [{"id": m.id, "name": m.name, "image": m.nim_image} for m in MODELS.values()],
            "smi_query": gpu_ops.SMI_QUERY}


@router.post("/manifests")
def manifests(data: ManifestIn):
    if data.model not in MODELS or data.gpu not in GPUS:
        raise HTTPException(422, "Unknown model or GPU")
    return gpu_ops.manifests(data.model_dump())


class PlanIn(BaseModel):
    model: str = "llama-3.1-70b"
    gpu: str = "h100-sxm"
    precision: Literal["fp16", "bf16", "fp8", "int8", "int4"] = "fp16"
    input_tokens: int = Field(2048, ge=1, le=131072)
    output_tokens: int = Field(512, ge=1, le=32768)
    batch_size: int = Field(16, ge=1, le=1024)


@router.post("/distributed-plan")
def distributed_plan(data: PlanIn):
    if data.model not in MODELS or data.gpu not in GPUS:
        raise HTTPException(422, "Unknown model or GPU")
    return gpu_ops.distributed_plan(data.model, data.gpu, data.precision, data.input_tokens,
                                    data.output_tokens, data.batch_size)


class TextIn(BaseModel):
    text: str = Field(min_length=3, max_length=200_000)


@router.post("/diagnose")
def diagnose(data: TextIn):
    return gpu_ops.diagnose(data.text)


@router.post("/nvidia-smi")
def nvidia_smi(data: TextIn):
    try:
        return gpu_ops.analyze_smi(data.text)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
