"""Analytical LLM inference performance model (roofline style).

Prefill is modelled as compute-bound, decode as the max of memory-bandwidth and compute time per step,
plus tensor-parallel all-reduce overhead. It explains *why* batching, quantization, KV-cache precision
and tensor parallelism move throughput and latency. It is an estimate, not a measurement: validate with
a live benchmark on the target system.
"""

import math
from dataclasses import dataclass

from ..data.hardware import GPU, GPUS, LLM, MODELS, PRECISION_BYTES

OVERHEAD_GB = 2.0  # CUDA context, runtime buffers, activations per GPU
USABLE_FRACTION = 0.90  # leave headroom for fragmentation
# Illustrative relative cost per GPU, used only to rank options. Not a price list.
RELATIVE_COST = {"l4": 1.0, "a10": 1.2, "l40s": 2.5, "a100-sxm": 4.0, "h100-pcie": 5.0,
                 "h100-sxm": 6.0, "h200-sxm": 7.0}


class PerfInputError(ValueError):
    pass


@dataclass
class Scenario:
    model: str = "llama-3.1-8b"
    gpu: str = "h100-sxm"
    weight_precision: str = "fp16"
    kv_precision: str = "fp16"
    tensor_parallel: int = 1
    batch_size: int = 8
    input_tokens: int = 1024
    output_tokens: int = 256
    prefill_mfu: float = 0.45
    decode_mfu: float = 0.35
    bandwidth_utilization: float = 0.75


def _lookup(s: Scenario) -> tuple[LLM, GPU]:
    if s.model not in MODELS:
        raise PerfInputError(f"Unknown model '{s.model}'")
    if s.gpu not in GPUS:
        raise PerfInputError(f"Unknown GPU '{s.gpu}'")
    if s.weight_precision not in PRECISION_BYTES or s.kv_precision not in ("fp16", "bf16", "fp8"):
        raise PerfInputError("Unsupported precision")
    if s.tensor_parallel not in (1, 2, 4, 8):
        raise PerfInputError("Tensor parallel size must be 1, 2, 4 or 8")
    return MODELS[s.model], GPUS[s.gpu]


def kv_bytes_per_token(m: LLM, kv_precision: str) -> float:
    return 2 * m.layers * m.kv_heads * m.head_dim * PRECISION_BYTES[kv_precision]


def estimate(s: Scenario) -> dict:
    m, g = _lookup(s)
    tp, b = s.tensor_parallel, s.batch_size
    warnings: list[str] = []

    fp8_native = g.fp8_tflops > 0
    if s.weight_precision == "fp8" and not fp8_native:
        warnings.append(f"{g.name} has no FP8 tensor cores; FP8 weights are not natively supported.")
    if s.kv_precision == "fp8" and not fp8_native:
        warnings.append(f"FP8 KV cache is not natively supported on {g.architecture}.")
    if tp > 1 and g.nvlink_gbps == 0:
        warnings.append(f"{g.name} has no NVLink; tensor parallelism over PCIe adds latency.")
    if s.input_tokens + s.output_tokens > m.max_context:
        warnings.append(f"Sequence exceeds the model's {m.max_context}-token context window.")
    if s.weight_precision in ("int4", "int8", "fp8"):
        warnings.append("Quantized weights can reduce accuracy; evaluate quality on your own tasks.")

    # ---- memory
    weights = m.params_b * 1e9 * PRECISION_BYTES[s.weight_precision]
    kv_tok = kv_bytes_per_token(m, s.kv_precision)
    seq = s.input_tokens + s.output_tokens
    kv_total = kv_tok * seq * b
    per_gpu_used = (weights + kv_total) / tp + OVERHEAD_GB * 1e9
    capacity = g.memory_gb * 1e9
    fits = per_gpu_used <= capacity * USABLE_FRACTION
    free_for_kv = (capacity * USABLE_FRACTION - OVERHEAD_GB * 1e9) * tp - weights
    max_batch = max(int(free_for_kv // (kv_tok * seq)), 0) if free_for_kv > 0 else 0
    if not fits:
        warnings.append("Does not fit in GPU memory at this batch size - reduce batch, quantize, "
                        "or increase tensor parallelism.")

    # ---- compute rates
    compute_fp8 = s.weight_precision == "fp8" and fp8_native
    peak = (g.fp8_tflops if compute_fp8 else g.fp16_tflops) * 1e12 * tp
    bw = g.bandwidth_gbps * 1e9 * tp * s.bandwidth_utilization
    active = m.active_params_b * 1e9

    prefill_flops = 2 * active * s.input_tokens * b
    prefill_s = prefill_flops / (peak * s.prefill_mfu)

    # MoE: with more concurrent sequences, more experts are touched per step.
    frac = m.active_params_b / m.params_b
    weight_read = weights * min(1.0, frac + (1 - frac) * min(1.0, b / 8))
    kv_read = kv_tok * (s.input_tokens + s.output_tokens / 2) * b
    mem_step = (weight_read + kv_read) / bw
    compute_step = 2 * active * b / (peak * s.decode_mfu)
    comm_step = 0.0
    if tp > 1:
        per_allreduce = 8e-6 if g.nvlink_gbps else 30e-6
        comm_step = 2 * m.layers * per_allreduce
    tpot = max(mem_step, compute_step) + comm_step
    bound = "memory-bandwidth" if mem_step >= compute_step else "compute"

    ttft = prefill_s
    e2e = ttft + s.output_tokens * tpot
    return {
        "mode": "ANALYTICAL_ESTIMATE",
        "scenario": s.__dict__,
        "model_name": m.name,
        "gpu_name": g.name,
        "fits_in_memory": fits,
        "max_batch_at_this_sequence": max_batch,
        "memory": {
            "weights_gb": _gb(weights),
            "kv_cache_gb": _gb(kv_total),
            "kv_cache_mb_per_token": round(kv_tok / 1e6, 4),
            "overhead_gb_per_gpu": OVERHEAD_GB,
            "per_gpu_used_gb": _gb(per_gpu_used),
            "per_gpu_capacity_gb": g.memory_gb,
            "utilization_pct": round(100 * per_gpu_used / capacity, 1),
        },
        "latency": {
            "ttft_ms": round(ttft * 1000, 1),
            "tpot_ms": round(tpot * 1000, 2),
            "e2e_ms": round(e2e * 1000, 1),
        },
        "throughput": {
            "output_tokens_per_s": round(b / tpot, 1),
            "tokens_per_s_per_user": round(1 / tpot, 1),
            "requests_per_s": round(b / e2e, 2),
            "output_tokens_per_s_per_gpu": round(b / tpot / tp, 1),
        },
        "decode_bound": bound,
        "decode_breakdown_ms": {
            "memory": round(mem_step * 1000, 3),
            "compute": round(compute_step * 1000, 3),
            "tp_communication": round(comm_step * 1000, 3),
        },
        "warnings": warnings,
        "assumptions": [
            f"Prefill MFU {s.prefill_mfu:.0%}, decode MFU {s.decode_mfu:.0%}, "
            f"memory-bandwidth utilisation {s.bandwidth_utilization:.0%}.",
            "All sequences in the batch arrive together (static batch); in-flight batching "
            "servers overlap prefill and decode.",
            "Attention FLOPs and speculative decoding are not modelled.",
            "Datasheet peak figures (dense). Validate with a live benchmark.",
        ],
    }


def sweep(s: Scenario, batches: list[int] | None = None) -> list[dict]:
    pts = []
    for b in batches or [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        r = estimate(Scenario(**{**s.__dict__, "batch_size": b}))
        pts.append({
            "batch_size": b, "fits": r["fits_in_memory"],
            "output_tokens_per_s": r["throughput"]["output_tokens_per_s"],
            "tokens_per_s_per_user": r["throughput"]["tokens_per_s_per_user"],
            "ttft_ms": r["latency"]["ttft_ms"], "tpot_ms": r["latency"]["tpot_ms"],
            "e2e_ms": r["latency"]["e2e_ms"], "bound": r["decode_bound"],
        })
    return pts


def plan_capacity(
    model: str, concurrency: int, input_tokens: int, output_tokens: int,
    max_ttft_ms: float, min_tokens_per_s_per_user: float,
    gpus: list[str] | None = None, precisions: list[str] | None = None,
) -> list[dict]:
    """Search GPU x precision x tensor-parallel options that meet the SLO for the target concurrency.

    Returns options ranked by an illustrative relative cost index (lower first).
    """
    options = []
    for gid in gpus or list(GPUS):
        for prec in precisions or ["fp16", "fp8", "int4"]:
            if prec == "fp8" and GPUS[gid].fp8_tflops == 0:
                continue
            for tp in (1, 2, 4, 8):
                base = Scenario(model=model, gpu=gid, weight_precision=prec,
                                kv_precision="fp8" if GPUS[gid].fp8_tflops else "fp16",
                                tensor_parallel=tp, input_tokens=input_tokens,
                                output_tokens=output_tokens)
                best = None
                for b in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                    r = estimate(Scenario(**{**base.__dict__, "batch_size": b}))
                    if not r["fits_in_memory"]:
                        break
                    if (r["latency"]["ttft_ms"] <= max_ttft_ms
                            and r["throughput"]["tokens_per_s_per_user"] >= min_tokens_per_s_per_user):
                        best = (b, r)
                if not best:
                    continue
                b, r = best
                replicas = math.ceil(concurrency / b)
                total = replicas * tp
                options.append({
                    "gpu": gid, "gpu_name": GPUS[gid].name, "precision": prec, "tensor_parallel": tp,
                    "batch_per_replica": b, "replicas": replicas, "total_gpus": total,
                    "relative_cost_index": round(total * RELATIVE_COST.get(gid, 5), 1),
                    "ttft_ms": r["latency"]["ttft_ms"],
                    "tokens_per_s_per_user": r["throughput"]["tokens_per_s_per_user"],
                    "aggregate_tokens_per_s": round(r["throughput"]["output_tokens_per_s"] * replicas, 0),
                    "per_gpu_memory_used_gb": r["memory"]["per_gpu_used_gb"],
                })
                break  # smallest TP that works for this GPU/precision is enough
    options.sort(key=lambda o: (o["relative_cost_index"], o["total_gpus"]))
    return options


def _gb(v: float) -> float:
    return round(v / 1e9, 2)
