"""GPU & Kubernetes Operations Lab: NIM deployment manifests, GPU sharing (time-slicing / MIG),
distributed-inference planning, CUDA/driver/NCCL/Xid troubleshooting and nvidia-smi health analysis."""

import csv
import io
import math
import re

import yaml

from ..data.hardware import GPUS, MODELS, PRECISION_BYTES
from . import perf_model

# Values of the nvidia.com/gpu.product label published by GPU Feature Discovery (verify on your
# cluster with: kubectl get nodes -L nvidia.com/gpu.product)
GPU_PRODUCT_LABEL = {
    "h200-sxm": "NVIDIA-H200", "h100-sxm": "NVIDIA-H100-80GB-HBM3", "h100-pcie": "NVIDIA-H100-PCIe",
    "a100-sxm": "NVIDIA-A100-SXM4-80GB", "l40s": "NVIDIA-L40S", "l4": "NVIDIA-L4", "a10": "NVIDIA-A10",
}
MIG_PROFILES = {"h100-sxm": ["1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"],
                "h100-pcie": ["1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"],
                "h200-sxm": ["1g.18gb", "2g.35gb", "3g.71gb", "4g.71gb", "7g.141gb"],
                "a100-sxm": ["1g.10gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"]}


# ------------------------------------------------------------------ manifests

def manifests(p: dict) -> dict:
    m, g = MODELS[p["model"]], GPUS[p["gpu"]]
    ns, name = p["namespace"], p["name"]
    gpus, replicas = p["gpus_per_replica"], p["replicas"]
    labels = {"app.kubernetes.io/name": name, "app.kubernetes.io/component": "llm-inference",
              "app.kubernetes.io/part-of": "enterprise-ai-lab"}
    weights_gb = m.params_b * PRECISION_BYTES.get(p["precision"], 2)
    cache_gi = max(int(math.ceil(weights_gb * 1.5 / 10) * 10), 20)
    findings: list[dict] = []

    resource_name = "nvidia.com/gpu"
    if p["sharing"] == "mig":
        if p["gpu"] not in MIG_PROFILES:
            findings.append(_f("error", f"{g.name} does not support MIG."))
        elif p["mig_profile"] not in MIG_PROFILES[p["gpu"]]:
            findings.append(_f("error", f"MIG profile {p['mig_profile']} is not valid for {g.name}. "
                               f"Valid: {', '.join(MIG_PROFILES[p['gpu']])}"))

    # memory check against the slice actually allocated
    per_gpu_mem = g.memory_gb
    if p["sharing"] == "mig" and p["mig_profile"]:
        mm = re.search(r"\.(\d+)gb", p["mig_profile"])
        per_gpu_mem = float(mm.group(1)) if mm else per_gpu_mem
    if weights_gb / gpus + 2 > per_gpu_mem * 0.9:
        findings.append(_f("error", f"{m.name} weights ({weights_gb:.0f} GB at "
                           f"{p['precision'].upper()}) do not fit in {gpus} x {per_gpu_mem:.0f} GB. "
                           "Increase GPUs per replica or quantize."))
    if p["sharing"] == "time_slicing":
        findings.append(_f("warning", "Time-slicing gives no memory or fault isolation between pods; "
                           "use it for dev/test, MIG for isolated production sharing."))
    if gpus > 1 and g.nvlink_gbps == 0:
        findings.append(_f("warning", f"{gpus}-way tensor parallelism on {g.name} runs over PCIe."))
    if p["image_tag"] in ("latest", ""):
        findings.append(_f("warning", "Pin the NIM image to a specific version tag for reproducible "
                           "deployments."))
    if replicas < 2:
        findings.append(_f("info", "Single replica: no high availability during node maintenance."))

    docs: list[tuple[str, dict]] = []
    docs.append(("Namespace for the inference workload", {
        "apiVersion": "v1", "kind": "Namespace", "metadata": {"name": ns}}))
    docs.append(("Persistent model cache so restarts do not re-download weights", {
        "apiVersion": "v1", "kind": "PersistentVolumeClaim",
        "metadata": {"name": f"{name}-cache", "namespace": ns, "labels": labels},
        "spec": {"accessModes": ["ReadWriteOnce"],
                 "resources": {"requests": {"storage": f"{cache_gi}Gi"}},
                 **({"storageClassName": p["storage_class"]} if p["storage_class"] else {})}}))

    container = {
        "name": "nim",
        "image": f"{m.nim_image}:{p['image_tag'] or 'latest'}",
        "ports": [{"name": "http", "containerPort": 8000}],
        "env": [
            {"name": "NGC_API_KEY", "valueFrom": {"secretKeyRef": {"name": "ngc-api", "key": "NGC_API_KEY"}}},
            {"name": "NIM_CACHE_PATH", "value": "/opt/nim/.cache"},
        ],
        "resources": {"limits": {resource_name: gpus}, "requests": {resource_name: gpus,
                                                                     "cpu": str(4 * gpus),
                                                                     "memory": f"{16 * gpus}Gi"}},
        "volumeMounts": [{"name": "cache", "mountPath": "/opt/nim/.cache"},
                         {"name": "dshm", "mountPath": "/dev/shm"}],
        "startupProbe": {"httpGet": {"path": "/v1/health/ready", "port": "http"},
                         "periodSeconds": 10, "failureThreshold": 180},
        "readinessProbe": {"httpGet": {"path": "/v1/health/ready", "port": "http"}, "periodSeconds": 10},
        "livenessProbe": {"httpGet": {"path": "/v1/health/live", "port": "http"}, "periodSeconds": 15},
        "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
    }
    pod_spec = {
        "imagePullSecrets": [{"name": "ngc-registry"}],
        "securityContext": {"runAsNonRoot": True, "runAsUser": 1000, "fsGroup": 1000},
        "nodeSelector": {"nvidia.com/gpu.product": GPU_PRODUCT_LABEL[p["gpu"]]},
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        "containers": [container],
        "volumes": [{"name": "cache", "persistentVolumeClaim": {"claimName": f"{name}-cache"}},
                    {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": f"{max(8, 8 * gpus)}Gi"}}],
    }
    if p["sharing"] == "mig":
        pod_spec["nodeSelector"]["nvidia.com/mig.config"] = f"all-{p['mig_profile']}"
    if replicas > 1:
        pod_spec["topologySpreadConstraints"] = [{
            "maxSkew": 1, "topologyKey": "kubernetes.io/hostname", "whenUnsatisfiable": "ScheduleAnyway",
            "labelSelector": {"matchLabels": {"app.kubernetes.io/name": name}}}]
    if p["replicas"] > 1 and p["storage_class"] == "":
        findings.append(_f("warning", "Replicas on different nodes cannot mount the same ReadWriteOnce "
                           "cache PVC (extra pods stay Pending). Set a ReadWriteMany storage class "
                           "or use a StatefulSet with per-pod volumes."))

    docs.append((f"{m.name} served by NVIDIA NIM on {gpus} x {g.name}", {
        "apiVersion": "apps/v1", "kind": "Deployment",
        "metadata": {"name": name, "namespace": ns, "labels": labels},
        "spec": {"replicas": replicas, "selector": {"matchLabels": {"app.kubernetes.io/name": name}},
                 "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxUnavailable": 0, "maxSurge": 1}},
                 "template": {"metadata": {"labels": labels}, "spec": pod_spec}}}))
    docs.append(("OpenAI-compatible endpoint inside the cluster", {
        "apiVersion": "v1", "kind": "Service",
        "metadata": {"name": name, "namespace": ns, "labels": labels},
        "spec": {"selector": {"app.kubernetes.io/name": name},
                 "ports": [{"name": "http", "port": 8000, "targetPort": "http"}]}}))
    if replicas > 1:
        docs.append(("Keep capacity during voluntary disruptions (node drains, upgrades)", {
            "apiVersion": "policy/v1", "kind": "PodDisruptionBudget",
            "metadata": {"name": name, "namespace": ns},
            "spec": {"minAvailable": max(replicas - 1, 1),
                     "selector": {"matchLabels": {"app.kubernetes.io/name": name}}}}))
    if p["autoscale"]:
        docs.append(("Scale on in-flight requests (needs Prometheus Adapter exposing the NIM metric "
                     "num_requests_running as a pods metric)", {
            "apiVersion": "autoscaling/v2", "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": name, "namespace": ns},
            "spec": {"scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": name},
                     "minReplicas": replicas, "maxReplicas": max(replicas * 3, replicas + 2),
                     "metrics": [{"type": "Pods", "pods": {
                         "metric": {"name": "num_requests_running"},
                         "target": {"type": "AverageValue", "averageValue": str(p["target_inflight"])}}}]}}))
    if p["sharing"] == "time_slicing":
        docs.append(("GPU Operator device-plugin config: advertise each GPU as N schedulable slices", {
            "apiVersion": "v1", "kind": "ConfigMap",
            "metadata": {"name": "time-slicing-config", "namespace": "gpu-operator"},
            "data": {"any": yaml.safe_dump({"version": "v1", "flags": {"migStrategy": "none"},
                                            "sharing": {"timeSlicing": {"resources": [
                                                {"name": "nvidia.com/gpu",
                                                 "replicas": p["time_slices"]}]}}},
                                           sort_keys=False)}}))

    rendered = "\n---\n".join(f"# {c}\n" + yaml.safe_dump(d, sort_keys=False) for c, d in docs)
    commands = [
        "# 1. GPU Operator (driver, container toolkit, device plugin, DCGM exporter, MIG manager)",
        "helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update",
        "helm install --wait gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace"
        + (" --set mig.strategy=single" if p["sharing"] == "mig" else ""),
        "# 2. NGC credentials (never commit the key)",
        f"kubectl create namespace {ns}",
        f"kubectl -n {ns} create secret docker-registry ngc-registry --docker-server=nvcr.io "
        "--docker-username='$oauthtoken' --docker-password=\"$NGC_API_KEY\"",
        f"kubectl -n {ns} create secret generic ngc-api --from-literal=NGC_API_KEY=\"$NGC_API_KEY\"",
    ]
    if p["sharing"] == "time_slicing":
        commands += ["# 3. Enable time-slicing",
                     "kubectl apply -f manifests.yaml  # creates time-slicing-config",
                     "kubectl patch clusterpolicies.nvidia.com/cluster-policy --type merge -p "
                     "'{\"spec\":{\"devicePlugin\":{\"config\":{\"name\":\"time-slicing-config\",\"default\":\"any\"}}}}'"]
    if p["sharing"] == "mig":
        commands += ["# 3. Partition GPUs with the MIG manager",
                     f"kubectl label nodes <gpu-node> nvidia.com/mig.config=all-{p['mig_profile']} --overwrite"]
    commands += ["# 4. Deploy and verify",
                 "kubectl apply -f manifests.yaml",
                 f"kubectl -n {ns} rollout status deploy/{name} --timeout=30m",
                 f"kubectl -n {ns} port-forward svc/{name} 8000:8000",
                 "curl -s localhost:8000/v1/models"]
    return {"yaml": rendered, "documents": len(docs), "findings": findings, "commands": commands,
            "valid": not any(f["level"] == "error" for f in findings)}


def _f(level: str, message: str) -> dict:
    return {"level": level, "message": message}


# ------------------------------------------------------------------ distributed inference planner

def distributed_plan(model: str, gpu: str, precision: str, input_tokens: int, output_tokens: int,
                     batch: int) -> dict:
    m, g = MODELS[model], GPUS[gpu]
    kv = "fp8" if g.fp8_tflops else "fp16"
    gpus_per_node = 8
    for tp in (1, 2, 4, 8):
        s = perf_model.Scenario(model=model, gpu=gpu, weight_precision=precision, kv_precision=kv,
                                tensor_parallel=tp, batch_size=batch, input_tokens=input_tokens,
                                output_tokens=output_tokens)
        r = perf_model.estimate(s)
        if r["fits_in_memory"]:
            notes = [f"Tensor parallelism {tp} within one node."]
            if tp > 1:
                notes.append("TP all-reduces every layer: keep TP inside an NVLink domain."
                             if g.nvlink_gbps else
                             "No NVLink: TP > 2 over PCIe is usually latency-bound; prefer a larger-memory GPU.")
            return {"strategy": "single-node", "tensor_parallel": tp, "pipeline_parallel": 1, "nodes": 1,
                    "total_gpus": tp, "estimate": r, "notes": notes,
                    "kubernetes": "Standard Deployment; request nvidia.com/gpu: " + str(tp)}
    weights = m.params_b * 1e9 * PRECISION_BYTES[precision]
    kv_total = perf_model.kv_bytes_per_token(m, kv) * (input_tokens + output_tokens) * batch
    usable = (g.memory_gb * perf_model.USABLE_FRACTION - perf_model.OVERHEAD_GB) * 1e9
    pp = math.ceil((weights + kv_total) / (usable * gpus_per_node))
    return {
        "strategy": "multi-node", "tensor_parallel": gpus_per_node, "pipeline_parallel": pp,
        "nodes": pp, "total_gpus": pp * gpus_per_node, "estimate": None,
        "memory_needed_gb": round((weights + kv_total) / 1e9, 1),
        "notes": [
            f"Model needs ~{(weights + kv_total) / 1e9:.0f} GB: TP{gpus_per_node} inside each node, "
            f"PP{pp} across {pp} nodes.",
            "Inter-node fabric: InfiniBand or RoCE with GPUDirect RDMA, ideally one 400 Gb/s NIC per GPU.",
            "Validate NCCL with nccl-tests (all_reduce_perf) before serving traffic.",
            "Consider FP8 / INT4 quantization or a larger-memory GPU (H200) to stay single-node.",
        ],
        "kubernetes": "Use LeaderWorkerSet (or the NIM multi-node Helm chart) so each replica is a "
                      "gang of pods scheduled together.",
    }


# ------------------------------------------------------------------ troubleshooting knowledge base

XID = {
    13: ("Graphics engine exception", "warning", "Usually an application fault (illegal memory access, "
         "bad kernel). Rarely hardware.", ["Reproduce with compute-sanitizer", "Check application logs "
         "around the timestamp"]),
    31: ("GPU memory page fault", "warning", "Application accessed an invalid address (driver or app "
         "bug).", ["Run with CUDA_LAUNCH_BLOCKING=1 to locate the kernel", "Upgrade to a supported driver"]),
    43: ("GPU stopped processing", "warning", "Application hit a software-induced fault; GPU remains "
         "healthy.", ["Restart the failing process", "Investigate the user application"]),
    45: ("Preemptive cleanup", "info", "Processes were killed (often a follow-on of another Xid).",
         ["Look for an earlier Xid in the same window"]),
    48: ("Double-bit ECC error", "critical", "Uncorrectable memory error. Data on that GPU may be "
         "corrupt.", ["Drain the node", "nvidia-smi -q -d ECC,PAGE_RETIREMENT,ROW_REMAPPER",
                      "Reset the GPU; RMA if it recurs"]),
    61: ("Internal micro-controller breakpoint", "critical", "Firmware fault.",
         ["Collect nvidia-bug-report.sh", "Reset GPU; upgrade driver/firmware"]),
    62: ("Internal micro-controller halt", "critical", "Firmware fault.",
         ["Collect nvidia-bug-report.sh", "Reset GPU; upgrade driver/firmware"]),
    63: ("ECC page retirement / row remap recorded", "warning", "Memory cells were retired or "
         "remapped; GPU still usable.", ["Check remapping status: nvidia-smi -q -d ROW_REMAPPER",
                                          "Schedule a GPU reset to apply pending remaps"]),
    64: ("ECC page retirement / row remap failure", "critical", "Remapping failed; GPU memory is "
         "degraded.", ["Drain the node", "Open an RMA with the system vendor"]),
    74: ("NVLink error", "critical", "NVLink link failure or CRC errors between GPUs/NVSwitch.",
         ["nvidia-smi nvlink -s and nvidia-smi nvlink -e", "Check fabric manager status",
          "Reseat / replace hardware if persistent"]),
    79: ("GPU has fallen off the bus", "critical", "GPU is no longer visible on PCIe: power, thermal, "
         "PCIe or hardware failure.", ["Check dmesg for PCIe AER errors", "Check PSU and cooling",
                                        "Cold reboot; RMA if it recurs"]),
    92: ("High single-bit ECC error rate", "warning", "Correctable errors at a high rate; early sign "
         "of memory degradation.", ["Monitor with DCGM", "Plan replacement if increasing"]),
    94: ("Contained ECC error", "warning", "Uncorrectable error contained to one application; other "
         "workloads unaffected.", ["Restart the affected application", "GPU reset at next window"]),
    95: ("Uncontained ECC error", "critical", "Uncorrectable error could affect all workloads on the "
         "GPU.", ["Drain the node and reset the GPU", "RMA if repeated"]),
    119: ("GSP RPC timeout", "critical", "GPU System Processor firmware did not respond.",
          ["Upgrade to the latest driver in your branch", "Reset the GPU",
           "Try disabling GSP firmware only as a temporary workaround if advised by NVIDIA support"]),
    120: ("GSP error", "critical", "GSP firmware reported an error.",
          ["Collect nvidia-bug-report.sh", "Upgrade driver; reset GPU"]),
}

RULES = [
    {"id": "cuda-oom", "pattern": r"CUDA out of memory|cudaErrorMemoryAllocation|OutOfMemoryError|"
     r"CUBLAS_STATUS_ALLOC_FAILED", "title": "GPU out of memory", "severity": "warning",
     "category": "Memory", "cause": "The model, KV cache or batch exceeds free GPU memory, or another "
     "process holds memory.", "diagnostics": ["nvidia-smi --query-compute-apps=pid,used_memory --format=csv",
                                             "Check max batch / max sequence length settings"],
     "remediation": ["Reduce batch size or max context length", "Quantize weights (FP8/INT4) or use "
                     "an FP8 KV cache", "Increase tensor parallelism", "Lower gpu_memory_utilization "
                     "if several processes share the GPU"]},
    {"id": "driver-too-old", "pattern": r"CUDA driver version is insufficient|cudaErrorInsufficientDriver|"
     r"forward compatibility was attempted|requires a newer driver", "title": "Driver older than the CUDA "
     "runtime", "severity": "critical", "category": "Driver / CUDA", "cause": "The container's CUDA "
     "toolkit needs a newer host driver than is installed.", "diagnostics": ["nvidia-smi (top line shows "
     "driver and max CUDA version)", "Check the image's CUDA version"], "remediation": ["Upgrade the host "
     "driver (or GPU Operator driver version)", "Use an image built for the installed CUDA version",
     "On data-center GPUs, the CUDA forward-compatibility package can bridge minor gaps"]},
    {"id": "no-device", "pattern": r"no CUDA-capable device is detected|cudaErrorNoDevice|"
     r"Found no NVIDIA driver|NVIDIA-SMI has failed", "title": "No GPU visible", "severity": "critical",
     "category": "Driver / runtime", "cause": "Driver not loaded, or the container was started without "
     "GPU access.", "diagnostics": ["nvidia-smi on the host", "lsmod | grep nvidia",
                                    "docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"],
     "remediation": ["Install/reload the driver", "Start containers with --gpus all (Docker) or request "
                     "nvidia.com/gpu (Kubernetes)", "Ensure the NVIDIA Container Toolkit is configured "
                     "as the runtime"]},
    {"id": "no-kernel-image", "pattern": r"no kernel image is available for execution|"
     r"cudaErrorNoKernelImageForDevice|sm_\d+ is not compatible", "title": "Binary not built for this "
     "GPU architecture", "severity": "critical", "category": "Build / compatibility", "cause": "The "
     "framework or engine was compiled for a different compute capability (e.g. TensorRT engine built "
     "on another GPU).", "diagnostics": ["nvidia-smi --query-gpu=compute_cap --format=csv",
                                        "Check TORCH_CUDA_ARCH_LIST / engine build GPU"],
     "remediation": ["Use builds that include this GPU's architecture", "Rebuild TensorRT / "
                     "TensorRT-LLM engines on the target GPU type (engines are not portable)"]},
    {"id": "container-toolkit", "pattern": r"could not select device driver .* with capabilities: "
     r"\[\[gpu\]\]|nvidia-container-cli: initialization error|unknown or invalid runtime name: nvidia",
     "title": "NVIDIA Container Toolkit not configured", "severity": "critical", "category": "Containers",
     "cause": "Docker/containerd is not configured to use the NVIDIA runtime.",
     "diagnostics": ["nvidia-ctk --version", "cat /etc/docker/daemon.json"],
     "remediation": ["Install nvidia-container-toolkit", "sudo nvidia-ctk runtime configure "
                     "--runtime=docker && sudo systemctl restart docker", "On Kubernetes let the GPU "
                     "Operator manage the toolkit"]},
    {"id": "k8s-insufficient-gpu", "pattern": r"Insufficient nvidia\.com/gpu|0/\d+ nodes are "
     r"available.*nvidia\.com/gpu", "title": "Pod cannot be scheduled: no free GPUs",
     "severity": "warning", "category": "Kubernetes", "cause": "No node advertises enough free "
     "nvidia.com/gpu, the device plugin is not running, or taints/selectors exclude GPU nodes.",
     "diagnostics": ["kubectl describe nodes | grep -A5 nvidia.com/gpu",
                     "kubectl -n gpu-operator get pods", "kubectl get nodes -L nvidia.com/gpu.product"],
     "remediation": ["Check the device-plugin and GPU Operator pods are Running", "Add the GPU "
                     "toleration and correct nodeSelector", "Free GPUs or add capacity; consider MIG / "
                     "time-slicing for small workloads"]},
    {"id": "nccl", "pattern": r"NCCL (WARN|error)|ncclSystemError|ncclUnhandledCudaError|"
     r"NCCL timeout|Watchdog caught collective operation timeout", "title": "NCCL communication failure",
     "severity": "critical", "category": "Distributed", "cause": "Collective communication failed: "
     "network/IB misconfiguration, mismatched NCCL versions, firewall, or a failed peer.",
     "diagnostics": ["Re-run with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET", "ibstat / ibv_devinfo",
                     "Run nccl-tests all_reduce_perf across the same nodes"],
     "remediation": ["Set NCCL_SOCKET_IFNAME / NCCL_IB_HCA to the right interfaces", "Ensure the same "
                     "NCCL/CUDA versions on all ranks", "Open required ports; check for a crashed rank "
                     "first (the root cause is often on another node)"]},
    {"id": "ecc", "pattern": r"uncorrectable ECC|ECC error|DBE|row remap", "title": "GPU memory ECC "
     "errors", "severity": "critical", "category": "Hardware", "cause": "GPU memory reported ECC events.",
     "diagnostics": ["nvidia-smi -q -d ECC,ROW_REMAPPER", "dcgmi diag -r 3"],
     "remediation": ["Drain workloads and reset the GPU", "Open an RMA if errors persist"]},
    {"id": "libcuda", "pattern": r"libcuda\.so(\.1)?: cannot open shared object file|"
     r"libnvidia-ml\.so", "title": "Driver libraries not mounted", "severity": "critical",
     "category": "Containers", "cause": "The container cannot see the host driver libraries.",
     "diagnostics": ["ldconfig -p | grep libcuda (inside the container)"],
     "remediation": ["Run with the NVIDIA runtime / --gpus", "Do not install a driver inside the image"]},
    {"id": "nim-auth", "pattern": r"401 Unauthorized.*nvcr\.io|unauthorized: authentication required|"
     r"NGC_API_KEY", "title": "NGC authentication failed", "severity": "warning", "category": "NIM",
     "cause": "The NGC API key is missing, wrong, or lacks access to this NIM.",
     "diagnostics": ["kubectl get secret ngc-registry ngc-api", "docker login nvcr.io (user $oauthtoken)"],
     "remediation": ["Recreate the ngc-registry and ngc-api secrets", "Confirm the account has access "
                     "to the NIM image"]},
    {"id": "thermal", "pattern": r"thermal slowdown|HW Slowdown|clocks_throttle_reasons.*(thermal|power)|"
     r"SW Thermal", "title": "GPU throttling (thermal / power)", "severity": "warning",
     "category": "Hardware", "cause": "GPU clocks reduced because of temperature or power limits.",
     "diagnostics": ["nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER"],
     "remediation": ["Check airflow, inlet temperature and fan status", "Verify power cap settings "
                     "and PSU capacity"]},
]


def diagnose(text: str) -> dict:
    findings = []
    for rule in RULES:
        m = re.search(rule["pattern"], text, re.IGNORECASE)
        if m:
            findings.append({**{k: v for k, v in rule.items() if k != "pattern"},
                             "evidence": _line_of(text, m.start())})
    for m in re.finditer(r"Xid\s*(?:\([^)]*\))?\s*:?\s*(\d{1,3})\b", text):
        code = int(m.group(1))
        info = XID.get(code)
        title, sev, cause, steps = info if info else (
            "Unrecognised Xid", "warning", "Not in the local knowledge base.",
            ["Look up the code in the NVIDIA Xid catalog", "Collect nvidia-bug-report.sh"])
        findings.append({"id": f"xid-{code}", "title": f"Xid {code}: {title}", "severity": sev,
                         "category": "Xid (driver event)", "cause": cause,
                         "diagnostics": ["dmesg -T | grep -i xid", "nvidia-smi -q"],
                         "remediation": steps, "evidence": _line_of(text, m.start())})
    order = {"critical": 0, "warning": 1, "info": 2}
    seen, unique = set(), []
    for f in sorted(findings, key=lambda f: order.get(f["severity"], 3)):
        if f["id"] not in seen:
            seen.add(f["id"])
            unique.append(f)
    return {"findings": unique, "matched": len(unique),
            "message": None if unique else "No known failure signatures found. Share the full log, "
            "dmesg output and nvidia-smi -q for deeper analysis."}


def _line_of(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start : end if end != -1 else None].strip()[:300]


# ------------------------------------------------------------------ nvidia-smi analysis

SMI_QUERY = ("nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,"
             "memory.used,memory.total,power.draw,power.limit,ecc.errors.uncorrected.volatile.total,"
             "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max "
             "--format=csv")


def _num(v: str | None) -> float | None:
    if v is None:
        return None
    m = re.search(r"-?\d+(\.\d+)?", v)
    return float(m.group(0)) if m and "N/A" not in v else None


def analyze_smi(text: str) -> dict:
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) < 2:
        raise ValueError("Paste the CSV output including the header line.")
    header = [h.strip().split(" [")[0] for h in rows[0]]
    gpus = []
    for row in rows[1:]:
        if not any(c.strip() for c in row):
            continue
        d = {h: (row[i].strip() if i < len(row) else None) for i, h in enumerate(header)}
        flags = []
        temp = _num(d.get("temperature.gpu"))
        util = _num(d.get("utilization.gpu"))
        used, total = _num(d.get("memory.used")), _num(d.get("memory.total"))
        draw, limit = _num(d.get("power.draw")), _num(d.get("power.limit"))
        ecc = _num(d.get("ecc.errors.uncorrected.volatile.total"))
        if temp is not None and temp >= 90:
            flags.append(_f("critical", f"Temperature {temp:.0f} C - likely thermal throttling."))
        elif temp is not None and temp >= 83:
            flags.append(_f("warning", f"Temperature {temp:.0f} C is high; check cooling."))
        mem_pct = 100 * used / total if used is not None and total else None
        if mem_pct is not None and mem_pct >= 97:
            flags.append(_f("warning", f"Memory {mem_pct:.0f}% used - close to OOM."))
        if util is not None and util < 5 and mem_pct is not None and mem_pct > 40:
            flags.append(_f("warning", "Memory allocated but GPU idle - stuck or idle process "
                            "holding memory."))
        if draw is not None and limit and draw >= 0.97 * limit:
            flags.append(_f("info", "Running at the power limit (power-capped clocks)."))
        if ecc:
            flags.append(_f("critical", f"{ecc:.0f} uncorrected ECC errors - drain and reset this GPU."))
        for kind in ("gen", "width"):
            cur, mx = _num(d.get(f"pcie.link.{kind}.current")), _num(d.get(f"pcie.link.{kind}.max"))
            if cur and mx and cur < mx and (util or 0) > 20:
                flags.append(_f("warning", f"PCIe link {kind} {cur:.0f} below max {mx:.0f} under load - "
                                "check slot/riser."))
        gpus.append({"index": d.get("index"), "name": d.get("name"), "driver": d.get("driver_version"),
                     "temperature_c": temp, "utilization_pct": util,
                     "memory_used_pct": round(mem_pct, 1) if mem_pct is not None else None,
                     "power_w": draw, "power_limit_w": limit,
                     "status": "critical" if any(f["level"] == "critical" for f in flags)
                     else "warning" if any(f["level"] == "warning" for f in flags) else "healthy",
                     "flags": flags})
    drivers = {g["driver"] for g in gpus if g["driver"]}
    fleet = []
    if len(drivers) > 1:
        fleet.append(_f("warning", f"Mixed driver versions on one host: {', '.join(sorted(drivers))}"))
    return {"gpus": gpus, "fleet_findings": fleet, "query": SMI_QUERY}
