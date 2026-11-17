def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.headers["X-Request-ID"]


def test_ready_reports_dependencies(client):
    body = client.get("/api/v1/health/ready").json()
    assert body["status"] == "ready"
    assert body["checks"]["database"]["ok"]
    assert body["checks"]["nvidia_endpoint"]["mode"] == "fallback"


def test_catalog(client):
    assert any(x["name"] == "NVIDIA NIM" for x in client.get("/api/v1/technologies").json())


def test_legacy_benchmark_is_labelled(client):
    assert client.post("/api/v1/benchmarks/simulate", json={}).json()["mode"] == "SIMULATION"


def test_project_crud_and_validation_errors(client):
    r = client.post("/api/v1/projects", json={"name": "Pilot", "customer": "ACME"})
    assert r.status_code == 200 and r.json()["id"]
    bad = client.post("/api/v1/projects", json={"name": "x"})
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "validation_error"


def test_live_inference_requires_configuration(client):
    r = client.post("/api/v1/inference/chat", json={"prompt": "hi"})
    assert r.status_code == 409


def test_metrics_exposed(client):
    assert "http_requests_total" in client.get("/metrics").text
