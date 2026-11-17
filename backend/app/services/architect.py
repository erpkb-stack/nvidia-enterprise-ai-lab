"""Enterprise Solution Architect Studio: business requirements -> reference architecture, technology
selection with rationale, GPU sizing, deployment plan, risks, diagram and exportable design document.

Rules-based and deterministic so every recommendation is explainable and repeatable.
"""

import math
from datetime import date

from ..catalog import USE_CASES
from ..data.hardware import MODELS
from . import perf_model

MODEL_BY_SIZE = {"small": "llama-3.1-8b", "medium": "mixtral-8x7b", "large": "llama-3.1-70b"}


def design(req: dict) -> dict:
    uc = next((u for u in USE_CASES if u["id"] == req["use_case_id"]), USE_CASES[0])
    pattern = uc["pattern"]
    rag = "RAG" in pattern
    agentic = "tools" in pattern or "agent" in pattern.lower()
    vision = pattern in ("Computer vision", "Multimodal AI")
    regulated = req["data_sensitivity"] == "regulated"
    air_gapped = req["deployment"] == "air_gapped"
    ha = req["availability"] == "high"

    # ---------------------------------------------------------------- model + sizing
    model_id = req.get("model") or MODEL_BY_SIZE.get(req.get("model_size", "small"), "llama-3.1-8b")
    model = MODELS[model_id]
    options = perf_model.plan_capacity(
        model_id, req["concurrent_users"], req["input_tokens"], req["output_tokens"],
        req["max_ttft_ms"], req["min_tokens_per_s_per_user"], req.get("gpu_preference") or None,
    )
    sizing: dict = {"model": model.name, "model_id": model_id, "options": options[:5]}
    if options:
        rec = dict(options[0])
        if ha:
            # N+1 replicas spread across at least two nodes / failure domains
            rec["replicas"] = max(rec["replicas"] + 1, 2)
            rec["total_gpus"] = rec["replicas"] * rec["tensor_parallel"]
        gpus_per_node = 8 if rec["gpu"].endswith("sxm") else 4
        rec["nodes"] = max(math.ceil(rec["total_gpus"] / gpus_per_node), 2 if ha else 1)
        sizing["recommended"] = rec
    else:
        sizing["recommended"] = None
        sizing["note"] = ("No single option meets the SLO. Relax the TTFT / per-user speed targets, "
                          "choose a smaller model, or shorten prompts.")

    emb_gpus = 1 if rag else 0
    vector = None
    if rag:
        chunks = req["document_volume_gb"] * 1e9 / 700  # ~700 useful chars per chunk after overlap
        vec_gb = chunks * 1024 * 4 * 1.5 / 1e9  # 1024-dim float32 + HNSW/payload overhead
        vector = {"estimated_chunks": int(chunks), "vector_memory_gb": round(vec_gb, 1),
                  "recommendation": "Qdrant or Milvus cluster with replication"
                  if vec_gb > 32 else "Single Qdrant node with snapshots (scale out when > 32 GB)"}

    # ---------------------------------------------------------------- technology selection
    stack = [
        _t("Inference", "NVIDIA NIM (LLM)", f"Serves {model.name} with an OpenAI-compatible API and "
           "TensorRT-LLM optimised engines; standard health/metrics endpoints simplify operations."),
        _t("Optimization", "TensorRT-LLM", "In-flight batching, paged KV cache and FP8/INT4 "
           "quantization deliver the throughput assumed in the sizing."),
        _t("Platform", "Kubernetes + NVIDIA GPU Operator", "Automates driver, container toolkit, "
           "device plugin, DCGM and MIG management consistently across nodes."),
        _t("Observability", "DCGM Exporter + Prometheus + Grafana", "GPU utilisation, memory, "
           "power, Xid errors and NIM latency metrics with alerting."),
        _t("Safety", "NeMo Guardrails", "Topic, jailbreak and output moderation rails around the "
           "assistant; required before external exposure."),
        _t("API edge", "API gateway with OIDC (enterprise IdP)", "Authentication, per-tenant rate "
           "limits and audit logging in front of all model endpoints."),
    ]
    if rag:
        stack += [
            _t("Retrieval", "NeMo Retriever embedding NIM (nv-embedqa-e5-v5)", "GPU-accelerated "
               "embeddings tuned for question/passage retrieval."),
            _t("Retrieval", "NeMo Retriever reranking NIM", "Cross-encoder reranking raises answer "
               "precision for the top passages sent to the LLM."),
            _t("Data", vector["recommendation"] if vector else "Vector database",
               "Hybrid (dense + keyword) search with metadata filters for access control."),
            _t("Ingestion", "Document pipeline (parsing, chunking, PII scan)", "Incremental "
               "ingestion with lineage so every answer cites a source document and page."),
        ]
    if agentic:
        stack.append(_t("Orchestration", "Agent framework with tool allow-list",
                        "Deterministic tool contracts, human approval for consequential actions, "
                        "full trace of tool calls."))
    if vision:
        stack.append(_t("Vision", "DeepStream / Triton Inference Server",
                        "Streaming video and image inference pipelines; size with a measured "
                        "frames-per-second benchmark (not modelled here)."))
    if ha:
        stack.append(_t("Resilience", "Multi-replica NIM + PodDisruptionBudget + HPA",
                        "N+1 replicas across failure domains; autoscale on queue depth / latency."))
    if req["deployment"] in ("cloud", "hybrid"):
        stack.append(_t("Infrastructure", "Managed Kubernetes with GPU node pools",
                        "Elastic GPU capacity; keep regulated data in approved regions."))
    if air_gapped:
        stack.append(_t("Supply chain", "Private registry mirror of NGC containers and models",
                        "Air-gapped installs need mirrored NIM images, model weights and Helm "
                        "charts with signature verification."))
    if regulated:
        stack.append(_t("Governance", "Encryption, data residency and audit retention",
                        "Customer-managed keys, document-level ACLs in retrieval, immutable "
                        "audit logs of prompts, sources and answers."))

    # ---------------------------------------------------------------- diagram
    nodes, edges = _graph(rag, agentic, vision)
    mermaid = _mermaid(nodes, edges)

    # ---------------------------------------------------------------- plan, risks
    plan = [
        _phase("1. Discovery & success criteria", 2, ["Confirm use cases, users, data sources and "
               "KPIs", "Define latency/throughput SLOs and quality acceptance set"],
               "Signed-off requirements and evaluation dataset"),
        _phase("2. Platform foundation", 2, ["Provision GPU nodes, install GPU Operator", "Deploy "
               "NIM, observability and identity integration"], "Health checks green; DCGM metrics "
               "visible; smoke-test inference"),
        _phase("3. Pilot build", 4, (["Ingest priority document sets", "Tune chunking, hybrid "
               "retrieval and reranking"] if rag else ["Integrate model into the target workflow"])
               + ["Add guardrails and audit logging"], "Quality >= acceptance threshold on the "
               "evaluation set"),
        _phase("4. Performance & security validation", 2, ["Run live load tests at target "
               "concurrency", "Pen test, red-team prompts, access-control tests"],
               "SLOs met at peak load; no critical findings"),
        _phase("5. Production rollout", 2, ["Staged rollout with feature flags", "Runbooks, "
               "on-call, capacity alerts"], "Go-live approval; hypercare complete"),
    ]
    risks = [
        _risk("Analytical sizing differs from measured performance", "Medium",
              "Run the live benchmark on target hardware before ordering capacity."),
        _risk("Model output is inaccurate or ungrounded", "High" if regulated else "Medium",
              "Retrieval grounding, citations, guardrails and human review for high-impact uses."),
        _risk("GPU supply / lead time", "Medium", "Reserve capacity early; keep a fallback GPU "
              "option from the sizing table."),
    ]
    if rag:
        risks.append(_risk("Sensitive documents exposed through retrieval", "High",
                           "Enforce document ACLs at query time; PII scanning at ingestion."))
    if agentic:
        risks.append(_risk("Agent takes unintended actions", "High",
                           "Tool allow-list, least-privilege credentials, approval gates."))

    result = {
        "title": f"{uc['name']} - reference architecture",
        "use_case": uc, "requirements": req, "pattern": pattern,
        "summary": _summary(uc, model.name, sizing, req),
        "sizing": sizing, "embedding_gpus": emb_gpus, "vector_store": vector,
        "stack": stack, "diagram": {"mermaid": mermaid, "nodes": nodes, "edges": edges},
        "deployment_plan": plan, "total_weeks": sum(p["weeks"] for p in plan),
        "risks": risks,
        "assumptions": [
            "GPU sizing uses the analytical performance model (dense datasheet figures).",
            "Throughput assumes in-flight batching with TensorRT-LLM based NIM.",
            "Costs are relative indices for ranking, not prices.",
        ],
        "generated_on": date.today().isoformat(),
    }
    result["markdown"] = to_markdown(result)
    return result


def _t(layer, tech, why):
    return {"layer": layer, "technology": tech, "rationale": why}


def _phase(name, weeks, activities, exit_criteria):
    return {"phase": name, "weeks": weeks, "activities": activities, "exit_criteria": exit_criteria}


def _risk(risk, severity, mitigation):
    return {"risk": risk, "severity": severity, "mitigation": mitigation}


def _graph(rag: bool, agentic: bool, vision: bool):
    n = [("users", "Enterprise users / apps", "client"), ("gw", "API gateway + OIDC", "edge"),
         ("app", "Application service", "app"), ("guard", "NeMo Guardrails", "safety"),
         ("llm", "LLM NIM (TensorRT-LLM)", "gpu")]
    e = [("users", "gw"), ("gw", "app"), ("app", "guard"), ("guard", "llm")]
    if rag:
        n += [("src", "Enterprise data sources", "data"), ("ingest", "Ingestion + chunking", "data"),
              ("embed", "Embedding NIM", "gpu"), ("vdb", "Vector DB (hybrid search)", "data"),
              ("rerank", "Reranking NIM", "gpu")]
        e += [("src", "ingest"), ("ingest", "embed"), ("embed", "vdb"), ("app", "vdb"),
              ("vdb", "rerank"), ("rerank", "app")]
    if agentic:
        n.append(("tools", "Approved tools / APIs", "app"))
        e.append(("app", "tools"))
    if vision:
        n.append(("vision", "Vision pipeline (DeepStream/Triton)", "gpu"))
        e.append(("app", "vision"))
    n += [("obs", "DCGM + Prometheus + Grafana", "ops"), ("k8s", "Kubernetes + GPU Operator", "ops")]
    e += [("llm", "obs"), ("k8s", "llm")]
    nodes = [{"id": i, "label": label, "group": g, "position": {"x": (k % 4) * 220, "y": (k // 4) * 120}}
             for k, (i, label, g) in enumerate(n)]
    edges = [{"source": a, "target": b} for a, b in e]
    return nodes, edges


def _mermaid(nodes, edges) -> str:
    lines = ["flowchart LR"]
    groups: dict[str, list] = {}
    for nd in nodes:
        groups.setdefault(nd["group"], []).append(nd)
    titles = {"client": "Consumers", "edge": "Edge", "app": "Application", "safety": "Safety",
              "gpu": "GPU inference", "data": "Data & retrieval", "ops": "Platform & operations"}
    for g, items in groups.items():
        lines.append(f'  subgraph {g}["{titles.get(g, g)}"]')
        for nd in items:
            lines.append(f'    {nd["id"]}["{nd["label"]}"]')
        lines.append("  end")
    for ed in edges:
        lines.append(f"  {ed['source']} --> {ed['target']}")
    lines.append("  classDef gpu fill:#1f3310,stroke:#76b900,color:#e8f5d0")
    gpu_ids = ",".join(nd["id"] for nd in nodes if nd["group"] == "gpu")
    if gpu_ids:
        lines.append(f"  class {gpu_ids} gpu")
    return "\n".join(lines)


def _summary(uc, model_name, sizing, req) -> str:
    rec = sizing.get("recommended")
    if not rec:
        return (f"{uc['name']} for {req['concurrent_users']} concurrent users. {sizing['note']}")
    return (f"{uc['name']} serving {req['concurrent_users']} concurrent users with {model_name} on "
            f"{rec['total_gpus']} x {rec['gpu_name']} ({rec['precision'].upper()}, TP{rec['tensor_parallel']}, "
            f"{rec['replicas']} replicas across {rec['nodes']} node(s)). Estimated TTFT "
            f"{rec['ttft_ms']:.0f} ms and {rec['tokens_per_s_per_user']:.0f} tokens/s per user "
            f"(analytical estimate).")


def to_markdown(r: dict) -> str:
    req, s = r["requirements"], r["sizing"]
    out = [f"# {r['title']}", "", f"_Generated {r['generated_on']} by the Enterprise AI Solutions "
           "Architect Lab. Sizing is an analytical estimate; validate with a live benchmark._", "",
           "## Executive summary", "", r["summary"], "", "## Requirements", "",
           "| Requirement | Value |", "|---|---|"]
    for k in ("concurrent_users", "input_tokens", "output_tokens", "max_ttft_ms",
              "min_tokens_per_s_per_user", "data_sensitivity", "deployment", "availability",
              "document_volume_gb"):
        out.append(f"| {k.replace('_', ' ')} | {req.get(k)} |")
    out += ["", "## Architecture", "", "```mermaid", r["diagram"]["mermaid"], "```", "",
            "## Technology selection", "", "| Layer | Technology | Rationale |", "|---|---|---|"]
    out += [f"| {t['layer']} | {t['technology']} | {t['rationale']} |" for t in r["stack"]]
    out += ["", "## GPU sizing", "", f"Model: **{s['model']}**", ""]
    if s.get("recommended"):
        rec = s["recommended"]
        out.append(f"Recommended: **{rec['total_gpus']} x {rec['gpu_name']}**, {rec['precision'].upper()}, "
                   f"TP{rec['tensor_parallel']}, {rec['replicas']} replicas, {rec['nodes']} node(s).")
        out += ["", "| GPU | Precision | TP | Replicas | GPUs | TTFT ms | tok/s/user | Cost index |",
                "|---|---|---|---|---|---|---|---|"]
        out += [f"| {o['gpu_name']} | {o['precision']} | {o['tensor_parallel']} | {o['replicas']} | "
                f"{o['total_gpus']} | {o['ttft_ms']:.0f} | {o['tokens_per_s_per_user']:.0f} | "
                f"{o['relative_cost_index']} |" for o in s["options"]]
    else:
        out.append(s.get("note", ""))
    if r.get("vector_store"):
        v = r["vector_store"]
        out += ["", f"Vector store: ~{v['estimated_chunks']:,} chunks, ~{v['vector_memory_gb']} GB "
                f"vector memory. {v['recommendation']}."]
    out += ["", f"## Deployment plan ({r['total_weeks']} weeks)", ""]
    for p in r["deployment_plan"]:
        out += [f"### {p['phase']} - {p['weeks']} weeks", *[f"- {a}" for a in p["activities"]],
                f"- **Exit criteria:** {p['exit_criteria']}", ""]
    out += ["## Risks", "", "| Risk | Severity | Mitigation |", "|---|---|---|"]
    out += [f"| {x['risk']} | {x['severity']} | {x['mitigation']} |" for x in r["risks"]]
    out += ["", "## Assumptions", "", *[f"- {a}" for a in r["assumptions"]], ""]
    return "\n".join(out)
