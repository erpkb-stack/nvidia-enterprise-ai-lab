def _est(client, **kw):
    r = client.post("/api/v1/inference-lab/estimate", json=kw)
    assert r.status_code == 200, r.text
    return r.json()


def test_estimate_is_labelled_and_plausible(client):
    r = _est(client, model="llama-3.1-8b", gpu="h100-sxm", batch_size=1)
    assert r["mode"] == "ANALYTICAL_ESTIMATE"
    assert r["fits_in_memory"]
    # single-stream decode on H100 for an 8B FP16 model is memory-bound, ~100-250 tok/s
    assert 100 < r["throughput"]["tokens_per_s_per_user"] < 250
    assert r["decode_bound"] == "memory-bandwidth"
    assert len(r["sweep"]) == 9


def test_batching_raises_throughput_and_quantization_helps(client):
    b1 = _est(client, batch_size=1)["throughput"]["output_tokens_per_s"]
    b32 = _est(client, batch_size=32)["throughput"]["output_tokens_per_s"]
    assert b32 > 5 * b1
    fp16 = _est(client, batch_size=1, weight_precision="fp16")["latency"]["tpot_ms"]
    fp8 = _est(client, batch_size=1, weight_precision="fp8")["latency"]["tpot_ms"]
    assert fp8 < fp16


def test_70b_needs_tensor_parallel_on_80gb(client):
    assert not _est(client, model="llama-3.1-70b", tensor_parallel=1)["fits_in_memory"]
    assert _est(client, model="llama-3.1-70b", tensor_parallel=4)["fits_in_memory"]


def test_fp8_on_ampere_warns(client):
    r = _est(client, gpu="a100-sxm", weight_precision="fp8")
    assert any("FP8" in w for w in r["warnings"])


def test_compare_capacity_and_save(client):
    r = client.post("/api/v1/inference-lab/compare", json={"scenarios": [
        {"label": "fp16", "weight_precision": "fp16"}, {"label": "int4", "weight_precision": "int4"}]})
    assert [x["label"] for x in r.json()] == ["fp16", "int4"]
    cap = client.post("/api/v1/inference-lab/capacity", json={"concurrency": 100}).json()
    assert cap["options"] and cap["options"][0]["total_gpus"] >= 1
    saved = client.post("/api/v1/inference-lab/estimate?save=true", json={"label": "x"}).json()
    assert saved["run_id"]
    assert client.get("/api/v1/inference-lab/runs").json()[0]["id"] == saved["run_id"]


def test_live_benchmark_requires_endpoint(client):
    r = client.post("/api/v1/inference-lab/live", json={})
    assert r.status_code == 409


def test_unknown_gpu_rejected(client):
    r = client.post("/api/v1/inference-lab/estimate", json={"gpu": "made-up"})
    assert r.status_code == 422
