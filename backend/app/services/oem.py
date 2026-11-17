"""OEM Solution Validation Center: rule-based validation of a server/solution configuration and a
customer-ready POC report.

Rules encode common NVIDIA-aligned sizing guidance (system memory vs GPU memory, CPU cores per GPU,
power headroom, fabric per GPU, driver/CUDA compatibility). "Validated" here means the configuration
passes these reference rules - it is not an NVIDIA-Certified Systems designation.
"""

import html
import math
import re
from datetime import date

from ..data.hardware import CUDA_MIN_DRIVER, GPUS, MODELS, PRECISION_BYTES
from . import perf_model

PRESETS = [
    {"id": "edge-l4", "label": "Edge inference - 2U, 4x L4", "config": {
        "rack_units": 2, "gpu": "l4", "gpu_count": 4, "cpu_sockets": 1, "cpu_cores_per_socket": 32,
        "cpu_tdp_w": 280, "system_memory_gb": 256, "network_ports": 2, "network_speed_gbps": 25,
        "network_type": "ethernet", "gpudirect_rdma": False, "nvme_tb": 3.84, "psu_count": 2,
        "psu_watts": 1600, "redundancy": "N+1", "workload": "inference", "multi_node": False,
        "model": "llama-3.1-8b", "precision": "fp8"}},
    {"id": "ent-l40s", "label": "Enterprise inference / RAG - 2U, 4x L40S", "config": {
        "rack_units": 2, "gpu": "l40s", "gpu_count": 4, "cpu_sockets": 2, "cpu_cores_per_socket": 32,
        "cpu_tdp_w": 300, "system_memory_gb": 512, "network_ports": 2, "network_speed_gbps": 100,
        "network_type": "ethernet", "gpudirect_rdma": False, "nvme_tb": 7.68, "psu_count": 2,
        "psu_watts": 3200, "redundancy": "N+1", "workload": "rag", "multi_node": False,
        "model": "llama-3.1-8b", "precision": "fp8"}},
    {"id": "hgx-h100", "label": "AI factory node - 8U HGX, 8x H100 SXM", "config": {
        "rack_units": 8, "gpu": "h100-sxm", "gpu_count": 8, "cpu_sockets": 2, "cpu_cores_per_socket": 56,
        "cpu_tdp_w": 350, "system_memory_gb": 2048, "network_ports": 8, "network_speed_gbps": 400,
        "network_type": "infiniband", "gpudirect_rdma": True, "nvme_tb": 30.72, "psu_count": 6,
        "psu_watts": 3000, "redundancy": "N+1", "workload": "training", "multi_node": True,
        "model": "llama-3.1-70b", "precision": "fp8"}},
    {"id": "hgx-h200", "label": "Large-model inference - 8U HGX, 8x H200 SXM", "config": {
        "rack_units": 8, "gpu": "h200-sxm", "gpu_count": 8, "cpu_sockets": 2, "cpu_cores_per_socket": 56,
        "cpu_tdp_w": 350, "system_memory_gb": 3072, "network_ports": 4, "network_speed_gbps": 400,
        "network_type": "ethernet", "gpudirect_rdma": True, "nvme_tb": 15.36, "psu_count": 6,
        "psu_watts": 3000, "redundancy": "N+1", "workload": "inference", "multi_node": False,
        "model": "llama-3.1-405b", "precision": "fp8"}},
]
SUPPORTED_OS = ("ubuntu 22.04", "ubuntu 24.04", "rhel 8", "rhel 9", "red hat enterprise linux 8",
                "red hat enterprise linux 9")


def _v(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", s or "")[:3]) or (0,)


def _check(cid, category, name, status, detail, recommendation=""):
    return {"id": cid, "category": category, "name": name, "status": status, "detail": detail,
            "recommendation": recommendation}


def validate(c: dict) -> dict:
    g = GPUS[c["gpu"]]
    n = c["gpu_count"]
    checks = []
    total_gpu_mem = g.memory_gb * n

    # 1. GPU form factor / density
    if g.form_factor == "SXM":
        if n not in (4, 8):
            checks.append(_check("gpu-topology", "Platform", "HGX GPU count", "fail",
                                 f"SXM GPUs ship on HGX baseboards with 4 or 8 GPUs, not {n}.",
                                 "Choose 4 or 8 GPUs, or a PCIe GPU."))
        elif n == 8 and c["rack_units"] < 4:
            checks.append(_check("gpu-topology", "Platform", "HGX chassis height", "fail",
                                 f"An 8-GPU HGX board does not fit in {c['rack_units']}U.",
                                 "Use a 6U+ air-cooled or 4U liquid-cooled platform."))
        elif n == 8 and c["rack_units"] < 6:
            checks.append(_check("gpu-topology", "Platform", "HGX chassis height", "warn",
                                 f"8-GPU HGX in {c['rack_units']}U generally requires liquid cooling.",
                                 "Confirm the platform's cooling design with the OEM."))
        else:
            checks.append(_check("gpu-topology", "Platform", "HGX topology", "pass",
                                 f"{n}x {g.name} on HGX with NVLink/NVSwitch ({g.nvlink_gbps:.0f} GB/s per GPU)."))
    else:
        per_ru = 4 if g.id == "l4" else 2  # single-slot low-profile vs double-wide
        max_gpus = min(8 if g.id != "l4" else 8, c["rack_units"] * per_ru)
        st = "pass" if n <= max_gpus else "fail"
        checks.append(_check("gpu-topology", "Platform", "PCIe GPU density", st,
                             f"{n}x {g.name} in {c['rack_units']}U (typical max {max_gpus}).",
                             "" if st == "pass" else "Use a taller chassis or fewer GPUs."))

    # 2. System memory vs GPU memory
    ratio = c["system_memory_gb"] / total_gpu_mem
    st = "pass" if ratio >= 2 else "warn" if ratio >= 1 else "fail"
    checks.append(_check("sys-memory", "Compute", "System memory to GPU memory ratio", st,
                         f"{c['system_memory_gb']} GB system / {total_gpu_mem:.0f} GB GPU = {ratio:.2f}x.",
                         "" if st == "pass" else
                         f"Common sizing guidance is at least 2x GPU memory "
                         f"(~{math.ceil(total_gpu_mem * 2 / 64) * 64} GB) for staging, pinned buffers "
                         "and data loading."))

    # 3. CPU cores per GPU
    cores = c["cpu_sockets"] * c["cpu_cores_per_socket"]
    per = cores / n
    st = "pass" if per >= 4 else "warn" if per >= 2 else "fail"
    checks.append(_check("cpu-cores", "Compute", "CPU cores per GPU", st,
                         f"{cores} physical cores for {n} GPUs = {per:.1f} per GPU.",
                         "" if st == "pass" else "Plan at least 4 physical cores per GPU for data "
                         "pre-processing, tokenization and serving."))

    # 4. Power budget with PSU redundancy
    drives = max(math.ceil(c["nvme_tb"] / 3.84), 1)
    load = (g.tdp_w * n + c["cpu_tdp_w"] * c["cpu_sockets"] + 300
            + c["system_memory_gb"] / 64 * 10 + drives * 15 + c["network_ports"] * 25) * 1.1
    usable_psus = {"N+1": c["psu_count"] - 1, "N+N": c["psu_count"] // 2}.get(c["redundancy"], c["psu_count"])
    capacity = max(usable_psus, 0) * c["psu_watts"]
    st = "pass" if capacity and load <= capacity * 0.9 else "warn" if load <= capacity else "fail"
    checks.append(_check("power", "Power & cooling", f"PSU capacity ({c['redundancy']})", st,
                         f"Estimated peak {load / 1000:.1f} kW vs {capacity / 1000:.1f} kW usable "
                         f"({usable_psus} of {c['psu_count']} x {c['psu_watts']} W).",
                         "" if st == "pass" else "Add PSUs / higher-rated PSUs, or cap GPU power."))
    density = load / c["rack_units"]
    if density > 2000:
        checks.append(_check("cooling", "Power & cooling", "Thermal density", "warn",
                             f"{density / 1000:.1f} kW per rack unit.",
                             "Confirm data-center cooling (often liquid or rear-door heat exchangers) "
                             "and rack power budget."))
    else:
        checks.append(_check("cooling", "Power & cooling", "Thermal density", "pass",
                             f"{density / 1000:.2f} kW per rack unit."))

    # 5. Network fabric
    per_gpu_net = c["network_ports"] * c["network_speed_gbps"] / n
    if c["multi_node"] or c["workload"] == "training":
        need = 400 if c["workload"] == "training" else 200
        st = "pass" if per_gpu_net >= need and c["gpudirect_rdma"] else \
            "warn" if per_gpu_net >= need / 2 else "fail"
        checks.append(_check("network", "Networking", "East-west fabric per GPU", st,
                             f"{per_gpu_net:.0f} Gb/s per GPU over {c['network_type']}, GPUDirect RDMA "
                             f"{'enabled' if c['gpudirect_rdma'] else 'disabled'}.",
                             "" if st == "pass" else f"Multi-node {c['workload']} needs ~{need} Gb/s "
                             "per GPU with GPUDirect RDMA (InfiniBand or RoCE)."))
    else:
        total = c["network_ports"] * c["network_speed_gbps"]
        st = "pass" if total >= 100 else "warn" if total >= 25 else "fail"
        checks.append(_check("network", "Networking", "North-south bandwidth", st,
                             f"{total} Gb/s total for single-node {c['workload']}.",
                             "" if st == "pass" else "Use at least 2x 50/100 GbE for production "
                             "serving and document ingestion."))

    # 6. Local storage
    need_tb = {"inference": 2, "rag": 4, "fine_tuning": 8, "training": 15}[c["workload"]]
    st = "pass" if c["nvme_tb"] >= need_tb else "warn"
    checks.append(_check("storage", "Storage", "Local NVMe capacity", st,
                         f"{c['nvme_tb']} TB NVMe (guideline >= {need_tb} TB for {c['workload']}).",
                         "" if st == "pass" else "Increase NVMe for model cache, checkpoints and datasets; "
                         "training also needs a high-throughput shared filesystem."))

    # 7-8. Software compatibility
    cuda = c["cuda_version"]
    min_drv = CUDA_MIN_DRIVER.get(".".join(cuda.split(".")[:2]))
    if not min_drv:
        checks.append(_check("driver-cuda", "Software", "Driver / CUDA compatibility", "warn",
                             f"CUDA {cuda} is not in the local compatibility table.",
                             "Confirm against the CUDA toolkit release notes."))
    else:
        ok = _v(c["driver_version"]) >= _v(min_drv)
        checks.append(_check("driver-cuda", "Software", "Driver / CUDA compatibility",
                             "pass" if ok else "fail",
                             f"Driver {c['driver_version']} vs CUDA {cuda} (needs >= {min_drv}).",
                             "" if ok else f"Upgrade the driver to {min_drv} or newer, or use an older "
                             "CUDA image (forward-compat packages can bridge on data-center GPUs)."))
    ok = _v(cuda) >= _v(g.min_cuda)
    checks.append(_check("gpu-cuda", "Software", f"{g.architecture} support in CUDA",
                         "pass" if ok else "fail",
                         f"{g.name} needs CUDA >= {g.min_cuda}; configured {cuda}.",
                         "" if ok else f"Move to CUDA {g.min_cuda}+."))
    os_ok = c["os"].lower().strip() in SUPPORTED_OS
    checks.append(_check("os", "Software", "Operating system", "pass" if os_ok else "warn",
                         f"{c['os']}", "" if os_ok else "Verify against the NVIDIA AI Enterprise / GPU "
                         "Operator support matrix."))
    if c.get("kubernetes_version"):
        k = _v(c["kubernetes_version"])
        st = "pass" if k >= (1, 29) else "warn"
        checks.append(_check("k8s", "Software", "Kubernetes version", st, f"Kubernetes {c['kubernetes_version']}",
                             "" if st == "pass" else "Older release - confirm GPU Operator support "
                             "and upstream support status."))

    # 9. Workload fit
    perf = None
    if c.get("model"):
        m = MODELS[c["model"]]
        prec = c.get("precision", "fp16")
        if prec == "fp8" and not g.fp8_tflops:
            checks.append(_check("fp8", "Workload", "FP8 support", "fail",
                                 f"{g.name} has no FP8 tensor cores.", "Use FP16/BF16 or INT4 weights."))
            prec = "fp16"
        weights = m.params_b * PRECISION_BYTES[prec]
        tp = next((t for t in (1, 2, 4, 8) if t <= n and weights / t + 2 <= g.memory_gb * 0.9), None)
        if tp is None:
            checks.append(_check("model-fit", "Workload", f"{m.name} fits on this node", "fail",
                                 f"{weights:.0f} GB of {prec.upper()} weights exceed {n}x {g.memory_gb:.0f} GB.",
                                 "Quantize, use more/larger-memory GPUs, or plan multi-node inference."))
        else:
            kv = "fp8" if g.fp8_tflops else "fp16"
            sc = perf_model.Scenario(model=m.id, gpu=g.id, weight_precision=prec, kv_precision=kv,
                                     tensor_parallel=tp, batch_size=16, input_tokens=1024, output_tokens=256)
            perf = perf_model.estimate(sc)
            replicas = n // tp
            perf["replicas_per_node"] = replicas
            fits = perf["fits_in_memory"]
            checks.append(_check("model-fit", "Workload", f"{m.name} fits on this node",
                                 "pass" if fits else "warn",
                                 f"TP{tp} x {replicas} replica(s); ~{perf['memory']['per_gpu_used_gb']} GB "
                                 f"per GPU at batch 16, 1k/256 tokens.",
                                 "" if fits else "Reduce batch or context length."))
            if tp > 2 and g.nvlink_gbps == 0:
                checks.append(_check("tp-pcie", "Workload", "Tensor parallel interconnect", "warn",
                                     f"TP{tp} over PCIe on {g.name}.",
                                     "Prefer NVLink-connected GPUs for TP > 2."))

    fails = sum(x["status"] == "fail" for x in checks)
    warns = sum(x["status"] == "warn" for x in checks)
    status = "NOT VALIDATED" if fails else "CONDITIONALLY VALIDATED" if warns else "VALIDATED"
    score = round(100 * sum({"pass": 1, "warn": 0.5, "fail": 0}[x["status"]] for x in checks) / len(checks))
    return {"status": status, "score": score, "passed": len(checks) - fails - warns, "warnings": warns,
            "failures": fails, "checks": checks, "performance": perf,
            "power_kw": round(load / 1000, 2), "gpu_memory_total_gb": total_gpu_mem,
            "disclaimer": "Validation against reference sizing rules in this lab. Not an "
                          "NVIDIA-Certified Systems designation; confirm with the OEM and NVIDIA "
                          "support matrices."}


# ------------------------------------------------------------------ POC report

def report_markdown(c: dict, v: dict, meta: dict) -> str:
    g = GPUS[c["gpu"]]
    today = date.today().isoformat()
    lines = [
        f"# {meta['title']}", "",
        f"**Customer:** {meta['customer']}  ",
        f"**Prepared by:** {meta['prepared_by']}  ",
        f"**Date:** {today}  ",
        f"**Validation status:** {v['status']} (score {v['score']}/100)", "",
        "## 1. Executive summary", "",
        f"This proof of concept evaluates a {c['rack_units']}U platform with {c['gpu_count']}x {g.name} "
        f"for **{c['workload'].replace('_', ' ')}** workloads. The configuration passed {v['passed']} of "
        f"{len(v['checks'])} reference checks with {v['warnings']} warning(s) and {v['failures']} "
        "failure(s).", "",
        meta.get("objectives") or "Objectives: validate functional fit, performance against SLOs, "
        "operational readiness and security controls before production investment.", "",
        "## 2. Solution configuration", "",
        "| Component | Specification |", "|---|---|",
        f"| Chassis | {c['rack_units']}U |",
        f"| GPU | {c['gpu_count']}x {g.name} ({g.architecture}, {g.memory_gb:.0f} GB, {g.tdp_w} W) |",
        f"| CPU | {c['cpu_sockets']} socket(s) x {c['cpu_cores_per_socket']} cores |",
        f"| System memory | {c['system_memory_gb']} GB |",
        f"| Network | {c['network_ports']}x {c['network_speed_gbps']} Gb/s {c['network_type']}"
        f"{' with GPUDirect RDMA' if c['gpudirect_rdma'] else ''} |",
        f"| Storage | {c['nvme_tb']} TB NVMe |",
        f"| Power | {c['psu_count']}x {c['psu_watts']} W ({c['redundancy']}); est. peak {v['power_kw']} kW |",
        f"| OS | {c['os']} |",
        f"| NVIDIA driver / CUDA | {c['driver_version']} / {c['cuda_version']} |",
        f"| Kubernetes | {c.get('kubernetes_version') or 'n/a'} |", "",
        "## 3. Validation results", "",
        "| Category | Check | Result | Detail | Recommendation |", "|---|---|---|---|---|",
    ]
    icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}
    lines += [f"| {x['category']} | {x['name']} | **{icon[x['status']]}** | {x['detail']} | "
              f"{x['recommendation'] or '-'} |" for x in v["checks"]]
    lines += ["", "## 4. Performance expectations", ""]
    p = v.get("performance")
    if p:
        lines += [
            f"Analytical estimate for **{p['model_name']}** ({p['scenario']['weight_precision'].upper()}, "
            f"TP{p['scenario']['tensor_parallel']}, {p['replicas_per_node']} replica(s), batch 16, "
            "1,024 input / 256 output tokens):", "",
            "| Metric | Per replica | Node total |", "|---|---|---|",
            f"| Time to first token | {p['latency']['ttft_ms']:.0f} ms | - |",
            f"| Time per output token | {p['latency']['tpot_ms']:.1f} ms | - |",
            f"| Output tokens/s | {p['throughput']['output_tokens_per_s']:,.0f} | "
            f"{p['throughput']['output_tokens_per_s'] * p['replicas_per_node']:,.0f} |",
            f"| Requests/s | {p['throughput']['requests_per_s']} | "
            f"{p['throughput']['requests_per_s'] * p['replicas_per_node']:.2f} |", "",
            "_These are analytical estimates, not measurements. Section 6 defines the live benchmark "
            "that produces the acceptance numbers._", ""]
    else:
        lines += ["No model selected for performance estimation.", ""]
    lines += [
        "## 5. Software stack & deployment", "",
        "1. Install OS and NVIDIA driver (or let NVIDIA GPU Operator manage the driver).",
        "2. Install Kubernetes and NVIDIA GPU Operator; confirm `nvidia.com/gpu` capacity on the node.",
        "3. Deploy DCGM Exporter, Prometheus and Grafana dashboards for GPU telemetry.",
        "4. Deploy the NVIDIA NIM inference microservice(s) with pinned image versions.",
        "5. Integrate identity (OIDC), API gateway, audit logging and guardrails.",
        "6. Run validation suite: `dcgmi diag -r 3`, NCCL tests (multi-GPU), live inference benchmark.", "",
        "## 6. POC test plan & acceptance criteria", "",
        "| # | Test | Method | Acceptance criterion |", "|---|---|---|---|",
        "| T1 | GPU health | `dcgmi diag -r 3` on every node | All tests pass, no Xid events |",
        "| T2 | Interconnect | nccl-tests all_reduce_perf | Bus bandwidth within 90% of platform spec |",
        f"| T3 | Inference performance | Live benchmark at target concurrency | {meta['acceptance_latency']} |",
        "| T4 | Functional quality | Customer evaluation set | Accuracy / groundedness >= agreed threshold |",
        "| T5 | Soak | 24-hour sustained load | Error rate < 0.1%, no memory growth, no throttling |",
        "| T6 | Resilience | Kill pod / drain node | Service recovers within SLO; no data loss |",
        "| T7 | Security | Access control & secrets review | No critical findings |", "",
        "## 7. Risks and open items", "",
    ]
    open_items = [x for x in v["checks"] if x["status"] != "pass"]
    lines += ([f"- **{x['name']}** ({icon[x['status']]}): {x['recommendation'] or x['detail']}"
               for x in open_items] or ["- No open items from configuration validation."])
    lines += ["", "## 8. Sign-off", "", "| Role | Name | Date | Signature |", "|---|---|---|---|",
              "| Customer sponsor | | | |", "| Solutions architect | | | |", "| OEM representative | | | |",
              "", f"_{v['disclaimer']}_", ""]
    return "\n".join(lines)


def markdown_to_html(md: str, title: str) -> str:
    """Small, safe Markdown subset renderer (headings, tables, lists, bold, code) for reports."""
    out, in_table, in_list = [], False, None

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        s = re.sub(r"(?<![\w*])_(.+?)_(?![\w*])", r"<em>\1</em>", s)
        return s

    def close():
        nonlocal in_table, in_list
        if in_table:
            out.append("</tbody></table>")
            in_table = False
        if in_list:
            out.append(f"</{in_list}>")
            in_list = None

    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r"-+", c) for c in cells):
                continue
            if not in_table:
                close()
                out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells)
                           + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        m = re.match(r"^(#{1,4}) (.*)", s)
        lm = re.match(r"^(\d+\.|-) (.*)", s)
        if m:
            close()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
        elif lm:
            kind = "ol" if lm.group(1)[0].isdigit() else "ul"
            if in_table or in_list != kind:
                close()
                out.append(f"<{kind}>")
                in_list = kind
            out.append(f"<li>{inline(lm.group(2))}</li>")
        elif not s:
            close()
        else:
            close()
            out.append(f"<p>{inline(s.rstrip(' '))}</p>")
    close()
    css = ("body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:960px;margin:40px auto;"
           "padding:0 24px;color:#1b1f1c;line-height:1.55}h1{border-bottom:4px solid #76b900;"
           "padding-bottom:10px}h2{margin-top:32px;color:#2d4a00}table{border-collapse:collapse;width:100%;"
           "font-size:13px;margin:12px 0}th,td{border:1px solid #d4dccf;padding:6px 8px;text-align:left;"
           "vertical-align:top}th{background:#eef5e3}code{background:#f2f4f0;padding:1px 4px;"
           "border-radius:3px}@media print{body{margin:0}h2{page-break-after:avoid}}")
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{html.escape(title)}"
            f"</title><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<style>{css}</style></head><body>{''.join(out)}</body></html>")
