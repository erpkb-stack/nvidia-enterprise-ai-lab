"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { Badge, Button, ErrorBox, Field, Metric, PageHeader, Panel } from "../components/ui";
import { errMsg, get, post } from "../lib/api";

const FLAGSHIPS = [
  ["01", "Enterprise RAG Assistant", "/rag", "Upload documents; hybrid semantic + keyword search, reranking, cited answers, retrieval evaluation."],
  ["02", "LLM Inference Performance Lab", "/benchmark", "Throughput and latency vs batching, quantization, KV cache and tensor parallelism; live load tests."],
  ["03", "Solution Architect Studio", "/architect", "Business requirements to architecture diagram, technology selection, GPU sizing and delivery plan."],
  ["04", "GPU & Kubernetes Ops Lab", "/mlops", "NIM manifests, MIG / time-slicing, distributed inference plans, Xid / CUDA / NCCL troubleshooting."],
  ["05", "OEM Validation Center", "/oem", "Validate server configurations against sizing rules and generate customer-ready POC reports."],
];

export default function Home() {
  const [ready, setReady] = useState<any>(null);
  const [counts, setCounts] = useState({ docs: 0, runs: 0, reports: 0 });
  const [projects, setProjects] = useState<any[]>([]);
  const [name, setName] = useState("Enterprise Knowledge Assistant POC");
  const [customer, setCustomer] = useState("Internal Lab");
  const [error, setError] = useState<string | null>(null);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    get("/api/v1/health/ready").then(setReady).catch((e) => setError(errMsg(e)));
    get("/api/v1/projects").then(setProjects).catch(() => {});
    Promise.all([get("/api/v1/rag/documents"), get("/api/v1/inference-lab/runs"), get("/api/v1/oem/reports")])
      .then(([d, r, o]) => setCounts({ docs: d.length, runs: r.length, reports: o.length }))
      .catch(() => {});
  }, []);

  async function create() {
    setError(null);
    try {
      const p = await post("/api/v1/projects", { name, customer });
      setProjects([p, ...projects]);
      setMsg(`Created project #${p.id}`);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  const c = ready?.checks;
  const nv = c?.nvidia_endpoint;
  return (
    <main>
      <PageHeader eyebrow="ENTERPRISE AI ENGINEERING PLATFORM" title="Solutions Architect Dashboard"
        sub="Design, build, validate and optimize production-oriented AI solutions on NVIDIA accelerated computing."
        right={<Badge tone={ready?.status === "ready" ? "ok" : ready ? "warn" : "muted"}>
          {ready ? (ready.status === "ready" ? "● API READY" : "● API DEGRADED") : "● CONNECTING"}</Badge>} />
      <ErrorBox error={error && `${error} Start the backend on port 8000.`} />

      <div className="grid">
        <Metric label="DATABASE" value={c ? (c.database.ok ? "Online" : "Down") : "—"} hint={c?.database.backend} />
        <Metric label="VECTOR STORE" value={c ? (c.vector_store.ok ? "Online" : "Down") : "—"} hint={c && `Qdrant (${c.vector_store.mode})`} />
        <Metric label="NVIDIA ENDPOINT" value={nv ? (nv.mode === "live" ? "Live" : nv.mode === "fallback" ? "Not set" : "Error") : "—"}
          hint={nv?.mode === "live" ? nv.chat_model : "Offline fallbacks active"} />
        <Metric label="LAB ACTIVITY" value={`${counts.docs} / ${counts.runs} / ${counts.reports}`} hint="documents / benchmark runs / reports" />
      </div>
      <div className="notice">Execution integrity: analytical estimates and fallbacks are always labelled. No GPU
        measurement is claimed unless a live endpoint is configured and the run is marked LIVE_MEASUREMENT.</div>

      <div className="section"><h2>Flagship demonstrations</h2></div>
      <div className="flagship">
        {FLAGSHIPS.map(([n, title, href, text]) => (
          <Link className="card" href={href} key={href}>
            <span className="num">{n}</span>
            <h3>{title}</h3>
            <p>{text}</p>
            <span className="sub">Open lab →</span>
          </Link>
        ))}
      </div>

      <div className="cols even">
        <Panel title="Create enterprise AI project" right={<span className="tag">ARCHITECT WORKSPACE</span>}>
          <div className="formgrid">
            <Field label="Project name"><input className="input" value={name} onChange={(e) => setName(e.target.value)} /></Field>
            <Field label="Customer / business unit"><input className="input" value={customer} onChange={(e) => setCustomer(e.target.value)} /></Field>
          </div>
          <div className="row" style={{ marginTop: 14 }}>
            <Button onClick={create}>Create project</Button>
            <span className="sub">{msg}</span>
          </div>
        </Panel>
        <Panel title="Recent projects">
          {projects.length === 0 ? <p className="sub">No projects yet.</p> : (
            <div className="tablewrap"><table className="t"><thead><tr><th>#</th><th>Name</th><th>Customer</th><th>Status</th></tr></thead>
              <tbody>{projects.slice(0, 6).map((p) => (
                <tr key={p.id}><td>{p.id}</td><td>{p.name}</td><td>{p.customer}</td><td><Badge tone="muted">{p.status}</Badge></td></tr>
              ))}</tbody></table></div>
          )}
        </Panel>
      </div>
    </main>
  );
}
