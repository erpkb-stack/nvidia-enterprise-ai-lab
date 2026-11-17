from app.services.rag_ingest import Page, chunk_pages

POLICY = b"""# Travel policy

Employees must book flights through the corporate travel portal. Economy class is required for
flights shorter than six hours. Business class may be approved by a vice president for longer flights.

Hotel stays are reimbursed up to 250 USD per night in major cities.
"""

GPU_GUIDE = b"""GPU cluster runbook.

When a node reports Xid 79 the GPU has fallen off the bus. Drain the node, power cycle it and open a
hardware ticket if the error repeats. DCGM diagnostics level 3 must pass before returning the node.
"""


def _upload(client, name, data, collection="test"):
    return client.post("/api/v1/rag/documents", data={"collection": collection},
                       files=[("files", (name, data, "text/plain"))])


def test_upload_query_cite_and_delete(client):
    r = _upload(client, "travel-policy.md", POLICY)
    assert r.status_code == 201, r.text
    doc = r.json()[0]
    assert doc["created"] and doc["chunk_count"] >= 1
    assert doc["embedding_provider"] == "local_hash"
    gpu = _upload(client, "gpu-runbook.txt", GPU_GUIDE).json()[0]

    # identical content is de-duplicated
    again = _upload(client, "copy.md", POLICY).json()[0]
    assert again["created"] is False and again["id"] == doc["id"]

    q = client.post("/api/v1/rag/query",
                    json={"question": "What should I do after an Xid 79 error?", "collection": "test"})
    assert q.status_code == 200, q.text
    body = q.json()
    assert body["citations"][0]["filename"] == "gpu-runbook.txt"
    assert body["providers"]["generation"] == "extractive_fallback"
    assert body["grounded"] and 1 in body["cited"]
    assert set(body["timings"]) >= {"embed_ms", "retrieval_ms", "rerank_ms", "generation_ms", "total_ms"}

    log = client.get("/api/v1/rag/queries").json()
    assert log[0]["question"].startswith("What should I do")

    assert client.delete(f"/api/v1/rag/documents/{gpu['id']}").status_code == 204
    q2 = client.post("/api/v1/rag/query",
                     json={"question": "What should I do after an Xid 79 error?", "collection": "test"})
    assert all(c["filename"] != "gpu-runbook.txt" for c in q2.json()["citations"])


def test_evaluation_metrics(client):
    _upload(client, "travel-policy.md", POLICY, "evalset")
    _upload(client, "gpu-runbook.txt", GPU_GUIDE, "evalset")
    r = client.post("/api/v1/rag/evaluate", json={"collection": "evalset", "k": 3, "cases": [
        {"question": "hotel reimbursement limit per night", "expected_filename": "travel-policy.md"},
        {"question": "GPU fallen off the bus", "expected_text": "Xid 79"},
    ]})
    assert r.status_code == 200, r.text
    assert r.json()["hit_rate"] == 1.0 and r.json()["mrr"] == 1.0


def test_rejects_unsupported_and_fake_pdf(client):
    r = _upload(client, "malware.exe", b"MZ....")
    assert r.status_code == 422 and "Unsupported" in r.json()["error"]["message"]
    r = _upload(client, "fake.pdf", b"not a pdf at all")
    assert r.status_code == 422


def test_invalid_collection_name(client):
    assert _upload(client, "a.txt", b"some text that is long enough", "Bad Name!").status_code == 422


def test_chunker_respects_size_and_pages():
    pages = [Page(1, "Sentence one is here. " * 80), Page(2, "Second page text. " * 10)]
    chunks = chunk_pages(pages, size=300, overlap=50)
    assert all(len(c.text) <= 360 for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
