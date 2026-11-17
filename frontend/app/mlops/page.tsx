"use client";
import { useEffect, useState } from "react";
import { Badge, Button, ErrorBox, Field, Metric, Num, PageHeader, Panel, Select, Tabs, statusTone } from "../../components/ui";
import { download, errMsg, fmt, get, post } from "../../lib/api";

type Tab = "manifests" | "distributed" | "diagnose" | "smi";
const PREC = ["fp16", "bf16", "fp8", "int8", "int4"].map((p) => ({ value: p, label: p.toUpperCase() }));

const SAMPLE_LOG = `[Tue Nov 17 10:42:03 2026] NVRM: Xid (PCI:0000:3b:00): 79, pid=48213, GPU has fallen off the bus.
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 79.10 GiB total capacity)
NCCL WARN NET/IB : Got completion from peer 10.0.0.12 with error 12, opcode 0
Warning  FailedScheduling  0/6 nodes are available: 6 Insufficient nvidia.com/gpu.`;

const SAMPLE_SMI = `index, name, driver_version, temperature.gpu, utilization.gpu [%], memory.used [MiB], memory.total [MiB], power.draw [W], power.limit [W], ecc.errors.uncorrected.volatile.total, pcie.link.gen.current, pcie.link.gen.max, pcie.link.width.current, pcie.link.width.max
0, NVIDIA H100 80GB HBM3, 550.127.05, 46, 91 %, 61234 MiB, 81559 MiB, 655.10 W, 700.00 W, 0, 5, 5, 16, 16
1, NVIDIA H100 80GB HBM3, 550.127.05, 88, 89 %, 79900 MiB, 81559 MiB, 699.20 W, 700.00 W, 0, 5, 5, 16, 16
2, NVIDIA H100 80GB HBM3, 550.127.05, 41, 0 %, 52000 MiB, 81559 MiB, 118.40 W, 700.00 W, 0, 5, 5, 16, 16
3, NVIDIA H100 80GB HBM3, 550.127.05, 44, 85 %, 60110 MiB, 81559 MiB, 640.00 W, 700.00 W, 3, 4, 5, 8, 16`;

export default function OpsLab() {
  const [cat, setCat] = useState<any>(null);
  const [tab, setTab] = useState<Tab>("manifests");
  const [mf, setMf] = useState<any>({ name: "llm-nim", namespace: "ai-inference", model: "llama-3.1-8b", gpu: "h100-sxm", precision: "fp16",
    gpus_per_replica: 1, replicas: 2, sharing: "exclusive", time_slices: 4, mig_profile: "3g.40gb", image_tag: "latest",
    storage_class: "", autoscale: true, target_inflight: 16 });
  const [mfRes, setMfRes] = useState<any>(null);
  const [plan, setPlan] = useState<any>({ model: "llama-3.1-70b", gpu: "h100-sxm", precision: "fp16", input_tokens: 2048, output_tokens: 512, batch_size: 16 });
  const [planRes, setPlanRes] = useState<any>(null);
  const [log, setLog] = useState(SAMPLE_LOG);
  const [diag, setDiag] = useState<any>(null);
  const [smi, setSmi] = useState(SAMPLE_SMI);
  const [smiRes, setSmiRes] = useState<any>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { get("/api/v1/gpu-ops/catalog").then(setCat).catch((e) => setError(errMsg(e))); }, []);
  async function run(key: string, fn: () => Promise<void>) {
    setError(null); setBusy(key);
    try { await fn(); } catch (e) { setError(errMsg(e)); } finally { setBusy(null); }
  }
  const gpuOpts = cat?.gpus.map((g: any) => ({ value: g.id, label: g.name })) || [];
  const modelOpts = cat?.models.map((m: any) => ({ value: m.id, label: m.name })) || [];
  const migOpts = (cat?.gpus.find((g: any) => g.id === mf.gpu)?.mig || []).map((p: string) => ({ value: p, label: p }));
  const setM = (k: string) => (v: any) => setMf({ ...mf, [k]: v });
  const setP = (k: string) => (v: any) => setPlan({ ...plan, [k]: v });

  return (
    <main>
      <PageHeader eyebrow="FLAGSHIP 04 · GPU OPERATIONS" title="GPU & Kubernetes Operations Lab"
        sub="Deploy NVIDIA NIM on Kubernetes with the right GPU scheduling, plan distributed inference, and triage CUDA, driver, NCCL and Xid failures." />
      <ErrorBox error={error} />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "manifests", label: "NIM deployment generator" }, { id: "distributed", label: "Distributed inference planner" },
        { id: "diagnose", label: "Failure diagnosis" }, { id: "smi", label: "nvidia-smi health" }]} />

      {tab === "manifests" && cat && <>
        <Panel title="Deployment parameters">
          <div className="formgrid four">
            <Field label="Deployment name"><input className="input" value={mf.name} onChange={(e) => setM("name")(e.target.value)} /></Field>
            <Field label="Namespace"><input className="input" value={mf.namespace} onChange={(e) => setM("namespace")(e.target.value)} /></Field>
            <Field label="Model (NIM)"><Select value={mf.model} onChange={setM("model")} options={modelOpts} /></Field>
            <Field label="NIM image tag" hint="Pin a version for production"><input className="input" value={mf.image_tag} onChange={(e) => setM("image_tag")(e.target.value)} /></Field>
            <Field label="GPU node type"><Select value={mf.gpu} onChange={setM("gpu")} options={gpuOpts} /></Field>
            <Field label="Weight precision"><Select value={mf.precision} onChange={setM("precision")} options={PREC} /></Field>
            <Field label="GPUs per replica"><Select value={mf.gpus_per_replica} onChange={setM("gpus_per_replica")} options={[1, 2, 4, 8].map((n) => ({ value: n, label: String(n) }))} /></Field>
            <Field label="Replicas"><Num value={mf.replicas} min={1} onChange={setM("replicas")} /></Field>
            <Field label="GPU sharing"><Select value={mf.sharing} onChange={setM("sharing")} options={[{ value: "exclusive", label: "Exclusive GPUs" }, { value: "time_slicing", label: "Time-slicing" }, { value: "mig", label: "MIG partitions" }]} /></Field>
            {mf.sharing === "time_slicing" && <Field label="Slices per GPU"><Num value={mf.time_slices} min={2} max={16} onChange={setM("time_slices")} /></Field>}
            {mf.sharing === "mig" && <Field label="MIG profile">{migOpts.length ? <Select value={mf.mig_profile} onChange={setM("mig_profile")} options={migOpts} /> : <p className="sub small">This GPU does not support MIG.</p>}</Field>}
            <Field label="Storage class" hint="Empty = cluster default"><input className="input" value={mf.storage_class} onChange={(e) => setM("storage_class")(e.target.value)} /></Field>
            <Field label="Autoscale on in-flight requests"><label className="check"><input type="checkbox" checked={mf.autoscale} onChange={(e) => setM("autoscale")(e.target.checked)} /> HPA (target {mf.target_inflight}/pod)</label></Field>
          </div>
          <div className="row" style={{ marginTop: 12 }}><Button busy={busy === "mf"} onClick={() => run("mf", async () => setMfRes(await post("/api/v1/gpu-ops/manifests", mf)))}>Generate manifests</Button></div>
        </Panel>
        {mfRes && <div className="cols">
          <div>
            <Panel title="Validation" right={<Badge tone={mfRes.valid ? "ok" : "bad"}>{mfRes.valid ? "DEPLOYABLE" : "BLOCKING ISSUES"}</Badge>}>
              {mfRes.findings.length === 0 ? <p className="sub">No issues found.</p> : mfRes.findings.map((f: any, i: number) => (
                <div key={i} className={`finding ${statusTone(f.level)}`}><h3><Badge tone={statusTone(f.level)}>{f.level.toUpperCase()}</Badge></h3>
                  <p className="sub small" style={{ margin: 0 }}>{f.message}</p></div>))}
            </Panel>
            <Panel title="Runbook"><pre className="code">{mfRes.commands.join("\n")}</pre></Panel>
          </div>
          <Panel title={`manifests.yaml · ${mfRes.documents} resources`} right={<div className="row">
            <Button kind="ghost" onClick={() => navigator.clipboard?.writeText(mfRes.yaml)}>Copy</Button>
            <Button kind="ghost" onClick={() => download("manifests.yaml", mfRes.yaml, "application/yaml")}>Download</Button></div>}>
            <pre className="code">{mfRes.yaml}</pre>
          </Panel>
        </div>}
      </>}

      {tab === "distributed" && cat && <>
        <Panel title="Model & hardware">
          <div className="formgrid three">
            <Field label="Model"><Select value={plan.model} onChange={setP("model")} options={modelOpts} /></Field>
            <Field label="GPU"><Select value={plan.gpu} onChange={setP("gpu")} options={gpuOpts} /></Field>
            <Field label="Precision"><Select value={plan.precision} onChange={setP("precision")} options={PREC} /></Field>
            <Field label="Input tokens"><Num value={plan.input_tokens} onChange={setP("input_tokens")} /></Field>
            <Field label="Output tokens"><Num value={plan.output_tokens} onChange={setP("output_tokens")} /></Field>
            <Field label="Concurrent sequences"><Num value={plan.batch_size} onChange={setP("batch_size")} /></Field>
          </div>
          <div className="row" style={{ marginTop: 12 }}><Button busy={busy === "plan"} onClick={() => run("plan", async () => setPlanRes(await post("/api/v1/gpu-ops/distributed-plan", plan)))}>Plan parallelism</Button></div>
        </Panel>
        {planRes && <Panel title="Parallelism plan" right={<Badge tone={planRes.strategy === "single-node" ? "ok" : "warn"}>{planRes.strategy.toUpperCase()}</Badge>}>
          <div className="grid">
            <Metric label="TENSOR PARALLEL" value={`TP${planRes.tensor_parallel}`} hint="split within a node" />
            <Metric label="PIPELINE PARALLEL" value={`PP${planRes.pipeline_parallel}`} hint="split across nodes" />
            <Metric label="NODES / GPUS" value={`${planRes.nodes} / ${planRes.total_gpus}`} />
            <Metric label={planRes.estimate ? "EST. THROUGHPUT" : "MEMORY NEEDED"} value={planRes.estimate ? `${fmt(planRes.estimate.throughput.output_tokens_per_s)} tok/s` : `${fmt(planRes.memory_needed_gb)} GB`}
              hint={planRes.estimate ? `TTFT ${fmt(planRes.estimate.latency.ttft_ms)} ms · analytical` : undefined} />
          </div>
          <ul className="list">{planRes.notes.map((n: string) => <li key={n}>{n}</li>)}</ul>
          <p className="sub"><strong>Kubernetes:</strong> {planRes.kubernetes}</p>
        </Panel>}
      </>}

      {tab === "diagnose" && <>
        <Panel title="Paste logs, dmesg, kubectl events or stack traces">
          <textarea className="input" rows={8} value={log} onChange={(e) => setLog(e.target.value)} />
          <div className="row" style={{ marginTop: 10 }}><Button busy={busy === "diag"} onClick={() => run("diag", async () => setDiag(await post("/api/v1/gpu-ops/diagnose", { text: log })))}>Diagnose</Button>
            <span className="sub small">Matches against a knowledge base of Xid codes and CUDA / driver / container / Kubernetes / NCCL failure signatures. Runs locally; logs are not stored.</span></div>
        </Panel>
        {diag && <Panel title={`Findings (${diag.matched})`}>
          {diag.message && <p className="sub">{diag.message}</p>}
          {diag.findings.map((f: any) => (<div key={f.id} className={`finding ${statusTone(f.severity)}`}>
            <h3>{f.title} <Badge tone={statusTone(f.severity)}>{f.severity.toUpperCase()}</Badge><Badge tone="muted">{f.category}</Badge></h3>
            <p className="sub small" style={{ margin: "0 0 6px" }}><strong>Evidence:</strong> <code>{f.evidence}</code></p>
            <p className="sub small" style={{ margin: 0 }}><strong>Likely cause:</strong> {f.cause}</p>
            <div className="cols even" style={{ marginTop: 6 }}>
              <div><span className="sub small"><strong>Diagnose</strong></span><ul className="list">{f.diagnostics.map((d: string) => <li key={d}><code>{d}</code></li>)}</ul></div>
              <div><span className="sub small"><strong>Remediate</strong></span><ul className="list">{f.remediation.map((d: string) => <li key={d}>{d}</li>)}</ul></div>
            </div></div>))}
        </Panel>}
      </>}

      {tab === "smi" && <>
        <Panel title="nvidia-smi CSV output">
          <p className="sub small">Run on the GPU host and paste the output (header included):</p>
          <pre className="code" style={{ marginBottom: 10 }}>{cat?.smi_query}</pre>
          <textarea className="input" rows={7} value={smi} onChange={(e) => setSmi(e.target.value)} />
          <div className="row" style={{ marginTop: 10 }}><Button busy={busy === "smi"} onClick={() => run("smi", async () => setSmiRes(await post("/api/v1/gpu-ops/nvidia-smi", { text: smi })))}>Analyze health</Button></div>
        </Panel>
        {smiRes && <Panel title="GPU health">
          {smiRes.fleet_findings.map((f: any, i: number) => <div key={i} className={`finding ${statusTone(f.level)}`}>{f.message}</div>)}
          <div className="tablewrap"><table className="t"><thead><tr><th>GPU</th><th>Status</th><th className="num">Temp °C</th><th className="num">Util %</th><th className="num">Mem %</th><th className="num">Power W</th><th>Findings</th></tr></thead>
            <tbody>{smiRes.gpus.map((g: any) => (<tr key={g.index}><td>#{g.index}<div className="small muted">{g.name}</div></td>
              <td><Badge tone={statusTone(g.status)}>{g.status.toUpperCase()}</Badge></td><td className="num">{fmt(g.temperature_c)}</td>
              <td className="num">{fmt(g.utilization_pct)}</td><td className="num">{fmt(g.memory_used_pct, 1)}</td>
              <td className="num">{fmt(g.power_w)} / {fmt(g.power_limit_w)}</td>
              <td className="small">{g.flags.length ? g.flags.map((f: any, i: number) => <div key={i} style={{ color: f.level === "critical" ? "#ff9b8f" : f.level === "warning" ? "#e6d39b" : undefined }}>{f.message}</div>) : <span className="muted">—</span>}</td></tr>))}</tbody></table></div>
        </Panel>}
      </>}
    </main>
  );
}
