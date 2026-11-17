from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)
def test_health(): assert client.get("/api/v1/health").status_code==200
def test_catalog(): assert any(x["name"]=="NVIDIA NIM" for x in client.get("/api/v1/technologies").json())
def test_benchmark_label(): assert client.post("/api/v1/benchmarks/simulate",json={}).json()["mode"]=="SIMULATION"
