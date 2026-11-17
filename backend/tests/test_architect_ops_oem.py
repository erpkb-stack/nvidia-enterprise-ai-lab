import yaml

# ------------------------------------------------------------- architect

def test_design_produces_complete_blueprint(client):
    r = client.post("/api/v1/architect/design", json={"use_case_id": "rag-assistant",
                                                       "data_sensitivity": "regulated"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["sizing"]["recommended"]["total_gpus"] >= 2  # high availability by default
    techs = " ".join(t["technology"] for t in d["stack"]).lower()
    assert "nim" in techs and "reranking" in techs and "Governance" in " ".join(t["layer"] for t in d["stack"])
    assert d["diagram"]["mermaid"].startswith("flowchart LR")
    assert "## Deployment plan" in d["markdown"]


def test_design_save_and_export(client):
    r = client.post("/api/v1/architect/design/save", json={"project_name": "Bank RAG",
                                                            "use_case_id": "aml-investigation"})
    assert r.status_code == 201
    md = client.get(f"/api/v1/architect/architectures/{r.json()['architecture_id']}/markdown")
    assert md.status_code == 200 and md.text.startswith("# AML Investigation Copilot")


def test_design_rejects_unknown_use_case(client):
    assert client.post("/api/v1/architect/design", json={"use_case_id": "nope"}).status_code == 422


# ------------------------------------------------------------- gpu ops

def test_manifests_are_valid_kubernetes_yaml(client):
    r = client.post("/api/v1/gpu-ops/manifests", json={"replicas": 2, "gpus_per_replica": 1,
                                                        "image_tag": "1.3.3"})
    body = r.json()
    docs = [d for d in yaml.safe_load_all(body["yaml"]) if d]
    kinds = [d["kind"] for d in docs]
    assert {"Namespace", "PersistentVolumeClaim", "Deployment", "Service", "PodDisruptionBudget",
            "HorizontalPodAutoscaler"} <= set(kinds)
    dep = next(d for d in docs if d["kind"] == "Deployment")
    c = dep["spec"]["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert c["image"].endswith(":1.3.3")
    assert body["valid"]


def test_manifest_flags_bad_configs(client):
    r = client.post("/api/v1/gpu-ops/manifests", json={"model": "llama-3.1-70b", "gpu": "l4"}).json()
    assert not r["valid"]
    r = client.post("/api/v1/gpu-ops/manifests", json={"gpu": "l40s", "sharing": "mig"}).json()
    assert any("MIG" in f["message"] for f in r["findings"])
    assert client.post("/api/v1/gpu-ops/manifests", json={"name": "Bad_Name"}).status_code == 422


def test_distributed_plan(client):
    small = client.post("/api/v1/gpu-ops/distributed-plan", json={"model": "llama-3.1-8b"}).json()
    assert small["strategy"] == "single-node" and small["tensor_parallel"] == 1
    big = client.post("/api/v1/gpu-ops/distributed-plan",
                      json={"model": "llama-3.1-405b", "gpu": "h100-sxm", "precision": "fp16"}).json()
    assert big["strategy"] == "multi-node" and big["pipeline_parallel"] >= 2


def test_diagnose_xid_and_oom(client):
    log = ("[  12.3] NVRM: Xid (PCI:0000:3b:00): 79, pid=1234, GPU has fallen off the bus.\n"
           "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
    f = client.post("/api/v1/gpu-ops/diagnose", json={"text": log}).json()["findings"]
    ids = [x["id"] for x in f]
    assert "xid-79" in ids and "cuda-oom" in ids
    assert f[0]["severity"] == "critical"


def test_nvidia_smi_analysis(client):
    csv = ("index, name, driver_version, temperature.gpu, utilization.gpu [%], memory.used [MiB], "
           "memory.total [MiB], power.draw [W], power.limit [W], ecc.errors.uncorrected.volatile.total\n"
           "0, NVIDIA H100 80GB HBM3, 550.127.05, 45, 87 %, 60000 MiB, 81559 MiB, 650.00 W, 700.00 W, 0\n"
           "1, NVIDIA H100 80GB HBM3, 550.127.05, 92, 0 %, 70000 MiB, 81559 MiB, 120.00 W, 700.00 W, 2\n")
    g = client.post("/api/v1/gpu-ops/nvidia-smi", json={"text": csv}).json()["gpus"]
    assert g[0]["status"] == "healthy" and g[1]["status"] == "critical"


# ------------------------------------------------------------- oem

def test_presets_validate(client):
    presets = client.get("/api/v1/oem/presets").json()
    for p in presets:
        r = client.post("/api/v1/oem/validate", json=p["config"])
        assert r.status_code == 200, r.text
        assert r.json()["status"] in ("VALIDATED", "CONDITIONALLY VALIDATED"), (p["id"], r.json())


def test_validation_catches_problems(client):
    r = client.post("/api/v1/oem/validate", json={
        "gpu": "h100-sxm", "gpu_count": 6, "system_memory_gb": 256, "driver_version": "470.82",
        "cuda_version": "12.4", "psu_count": 2, "psu_watts": 1600}).json()
    failed = {c["id"] for c in r["checks"] if c["status"] == "fail"}
    assert {"gpu-topology", "sys-memory", "driver-cuda", "power"} <= failed
    assert r["status"] == "NOT VALIDATED"


def test_report_generation_and_downloads(client):
    preset = client.get("/api/v1/oem/presets").json()[1]["config"]
    r = client.post("/api/v1/oem/reports", json={"config": preset, "customer": "ACME Bank"})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    md = client.get(f"/api/v1/oem/reports/{rid}/markdown")
    assert "## 6. POC test plan" in md.text
    assert "acme-bank" in md.headers["content-disposition"]
    html = client.get(f"/api/v1/oem/reports/{rid}/html")
    assert html.status_code == 200 and "<table>" in html.text and "<script" not in html.text
