"""Reference hardware and model data shared by the performance lab, architect, ops and OEM modules.

Figures are public datasheet values (dense, no sparsity) rounded for planning. They are inputs to an
analytical model, never measurements. Confirm against current NVIDIA datasheets before customer use.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPU:
    id: str
    name: str
    architecture: str
    memory_gb: float
    bandwidth_gbps: float  # GB/s
    fp16_tflops: float  # dense FP16/BF16 tensor
    fp8_tflops: float  # dense FP8 tensor, 0 when unsupported
    tdp_w: int
    form_factor: str  # SXM | PCIe
    nvlink_gbps: float  # per-GPU NVLink bandwidth, 0 when none
    mig_max: int  # max MIG instances, 0 when unsupported
    min_cuda: str


GPUS: dict[str, GPU] = {
    g.id: g
    for g in [
        GPU("h200-sxm", "H200 SXM 141GB", "Hopper", 141, 4800, 989, 1979, 700, "SXM", 900, 7, "12.2"),
        GPU("h100-sxm", "H100 SXM 80GB", "Hopper", 80, 3350, 989, 1979, 700, "SXM", 900, 7, "11.8"),
        GPU("h100-pcie", "H100 PCIe 80GB", "Hopper", 80, 2000, 756, 1513, 350, "PCIe", 600, 7, "11.8"),
        GPU("a100-sxm", "A100 SXM 80GB", "Ampere", 80, 2039, 312, 0, 400, "SXM", 600, 7, "11.0"),
        GPU("l40s", "L40S 48GB", "Ada Lovelace", 48, 864, 362, 733, 350, "PCIe", 0, 0, "11.8"),
        GPU("l4", "L4 24GB", "Ada Lovelace", 24, 300, 121, 242, 72, "PCIe", 0, 0, "11.8"),
        GPU("a10", "A10 24GB", "Ampere", 24, 600, 125, 0, 150, "PCIe", 0, 0, "11.1"),
    ]
}


@dataclass(frozen=True)
class LLM:
    id: str
    name: str
    params_b: float  # total parameters (billions)
    active_params_b: float  # parameters used per token (differs for mixture-of-experts)
    layers: int
    kv_heads: int
    head_dim: int
    max_context: int
    nim_image: str


MODELS: dict[str, LLM] = {
    m.id: m
    for m in [
        LLM("llama-3.1-8b", "Llama 3.1 8B Instruct", 8.03, 8.03, 32, 8, 128, 131072,
            "nvcr.io/nim/meta/llama-3.1-8b-instruct"),
        LLM("mistral-7b", "Mistral 7B Instruct", 7.24, 7.24, 32, 8, 128, 32768,
            "nvcr.io/nim/mistralai/mistral-7b-instruct-v0.3"),
        LLM("mixtral-8x7b", "Mixtral 8x7B Instruct (MoE)", 46.7, 12.9, 32, 8, 128, 32768,
            "nvcr.io/nim/mistralai/mixtral-8x7b-instruct-v01"),
        LLM("llama-3.1-70b", "Llama 3.1 70B Instruct", 70.6, 70.6, 80, 8, 128, 131072,
            "nvcr.io/nim/meta/llama-3.1-70b-instruct"),
        LLM("nemotron-70b", "Llama 3.1 Nemotron 70B Instruct", 70.6, 70.6, 80, 8, 128, 131072,
            "nvcr.io/nim/nvidia/llama-3.1-nemotron-70b-instruct"),
        LLM("llama-3.1-405b", "Llama 3.1 405B Instruct", 405, 405, 126, 8, 128, 131072,
            "nvcr.io/nim/meta/llama-3.1-405b-instruct"),
    ]
}

# Bytes per weight / KV element
PRECISION_BYTES = {"fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}

# CUDA toolkit -> minimum Linux driver for that toolkit's full feature set (from CUDA release notes).
CUDA_MIN_DRIVER = {
    "11.8": "520.61.05",
    "12.0": "525.60.13",
    "12.2": "535.54.03",
    "12.4": "550.54.14",
    "12.6": "560.28.03",
    "12.8": "570.26",
}


def gpu_list() -> list[dict]:
    return [asdict(g) for g in GPUS.values()]


def model_list() -> list[dict]:
    return [asdict(m) for m in MODELS.values()]
