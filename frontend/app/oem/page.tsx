"use client";
import { useEffect, useState } from "react";
import { Badge, Button, ErrorBox, Field, Metric, Num, PageHeader, Panel, Select, statusTone } from "../../components/ui";
import { API, errMsg, get, post } from "../../lib/api";

const BASE = { rack_units: 2, gpu: "l40s", gpu_count: 4, cpu_sockets: 2, cpu_cores_per_socket: 32, cpu_tdp_w: 300,
  system_memory_gb: 512, network_ports: 2, network_speed_gbps: 100, network_type: "ethernet", gpudirect_rdma: false,
  nvme_tb: 7.68, psu_count: 2, psu_watts: 3200, redundancy: "N+1", os: "Ubuntu 22.04", driver_version: "550.127.05",
  cuda_version: "12.4", kubernetes_version: "1.30", workload: "rag", multi_node: false, model: "llama-3.1-8b", precision: "fp8" };

const NUMS: [string, string][] = [["rack_units", "Chassis (U)"], ["gpu_count", "GPU count"], ["cpu_sockets", "CPU sockets"],
  ["cpu_cores_per_socket", "Cores / socket"], ["cpu_tdp_w", "CPU TDP (W)"], ["system_memory_gb", "System memory (GB)"],
  ["network_ports", "Network ports"], ["network_speed_gbps", "Port speed (Gb/s)"], ["nvme_tb", "NVMe (TB)"],
  ["psu_count", "PSU count"], ["psu_watts", "PSU rating (W)"]];

export default function OemCenter() {
  const [presets, setPresets] = useState<any[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [gpus, setGpus] = useState<any[]>([]);
  const [c, setC] = useState<any>(BASE);
  const [val, setVal] = useState<any>(null);
  const [meta, setMeta] = useState({ customer: "Example Customer", title: "Enterprise AI Proof of Concept Report",
    prepared_by: "Solutions Architecture", objectives: "",
    acceptance_latency: "p95 TTFT <= 2 s and >= 20 tokens/s per user at target concurrency" });
  const [report, setReport] = useState<any>(null);
  const [html, setHtml] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([get("/api/v1/oem/presets"), get("/api/v1/inference-lab/catalog")])
      .then(([p, cat]) => { setPresets(p); setModels(cat.models); setGpus(cat.gpus); })
      .catch((e) => setError(errMsg(e)));
  }, []);
  const set = (k: string) => (v: any) => { setC({ ...c, [k]: v }); setVal(null); };

  async function run(key: string, fn: () => Promise<void>) {
    setError(null); setBusy(key);
    try { await fn(); } catch (e) { setError(errMsg(e)); } finally { setBusy(null); }
  }
  const validate = () => run("val", async () => setVal(await post("/api/v1/oem/validate", c)));
  const makeReport = () => run("rep", async () => {
    const r = await post("/api/v1/oem/reports", { config: c, ...meta });
    setReport(r);
    setHtml(await (await fetch(API + r.html_url)).text());
  });

  const groups = val ? Array.from(new Set(val.checks.map((x: any) => x.category))) as string[] : [];
  return (
    <main>
      <PageHeader eyebrow="FLAGSHIP 05 · OEM SOLUTION VALIDATION" title="OEM Solution Validation Center"
        sub="Validate a server or solution configuration against NVIDIA-aligned sizing and compatibility rules, then generate a customer-ready POC report." />
      <ErrorBox error={error} />
      <Panel title="Reference configurations" right={<span className="sub small">Start from a preset, then edit</span>}>
        <div className="row">{presets.map((p) => (
          <Button key={p.id} kind="ghost" onClick={() => { setC({ ...BASE, ...p.config }); setVal(null); setReport(null); }}>{p.label}</Button>))}</div>
      </Panel>

      <Panel title="Solution configuration">
        <div className="formgrid four">
          <Field label="GPU"><Select value={c.gpu} onChange={set("gpu")} options={gpus.map((g) => ({ value: g.id, label: g.name }))} /></Field>
          {NUMS.map(([k, label]) => <Field key={k} label={label}><Num value={c[k]} step={k === "nvme_tb" ? 0.01 : 1} onChange={set(k)} /></Field>)}
          <Field label="Network type"><Select value={c.network_type} onChange={set("network_type")} options={[{ value: "ethernet", label: "Ethernet / RoCE" }, { value: "infiniband", label: "InfiniBand" }]} /></Field>
          <Field label="PSU redundancy"><Select value={c.redundancy} onChange={set("redundancy")} options={["N+1", "N+N", "none"].map((v) => ({ value: v, label: v }))} /></Field>
          <Field label="Operating system"><input className="input" value={c.os} onChange={(e) => set("os")(e.target.value)} /></Field>
          <Field label="NVIDIA driver"><input className="input" value={c.driver_version} onChange={(e) => set("driver_version")(e.target.value)} /></Field>
          <Field label="CUDA version"><input className="input" value={c.cuda_version} onChange={(e) => set("cuda_version")(e.target.value)} /></Field>
          <Field label="Kubernetes version"><input className="input" value={c.kubernetes_version} onChange={(e) => set("kubernetes_version")(e.target.value)} /></Field>
          <Field label="Primary workload"><Select value={c.workload} onChange={set("workload")} options={[["inference", "Inference"], ["rag", "RAG"], ["fine_tuning", "Fine-tuning"], ["training", "Training"]].map(([value, label]) => ({ value, label }))} /></Field>
          <Field label="Target model"><Select value={c.model} onChange={set("model")} options={models.map((m) => ({ value: m.id, label: m.name }))} /></Field>
          <Field label="Model precision"><Select value={c.precision} onChange={set("precision")} options={["fp16", "bf16", "fp8", "int8", "int4"].map((p) => ({ value: p, label: p.toUpperCase() }))} /></Field>
          <Field label="Fabric"><div>
            <label className="check"><input type="checkbox" checked={c.gpudirect_rdma} onChange={(e) => set("gpudirect_rdma")(e.target.checked)} /> GPUDirect RDMA</label>
            <label className="check"><input type="checkbox" checked={c.multi_node} onChange={(e) => set("multi_node")(e.target.checked)} /> Multi-node cluster</label></div></Field>
        </div>
        <div className="row" style={{ marginTop: 12 }}><Button busy={busy === "val"} onClick={validate}>Validate configuration</Button></div>
      </Panel>

      {val && <>
        <div className="status-banner" style={{ marginTop: 14 }}>
          <Badge tone={statusTone(val.status)}>{val.status}</Badge>
          <strong>{val.score}/100</strong>
          <span className="sub">{val.passed} passed · {val.warnings} warnings · {val.failures} failures · est. peak {val.power_kw} kW · {val.gpu_memory_total_gb} GB GPU memory</span>
        </div>
        <p className="sub small">{val.disclaimer}</p>
        {val.performance && <div className="grid">
          <Metric label="MODEL" value={val.performance.model_name.replace(" Instruct", "")} hint={`TP${val.performance.scenario.tensor_parallel} × ${val.performance.replicas_per_node} replica(s)`} />
          <Metric label="TTFT (EST.)" value={`${Math.round(val.performance.latency.ttft_ms)} ms`} hint="batch 16, 1k in / 256 out" />
          <Metric label="NODE THROUGHPUT (EST.)" value={`${Math.round(val.performance.throughput.output_tokens_per_s * val.performance.replicas_per_node).toLocaleString()} tok/s`} />
          <Metric label="GPU MEMORY USED" value={`${val.performance.memory.per_gpu_used_gb} GB`} hint="per GPU" />
        </div>}
        <Panel title="Validation checks">
          {groups.map((g) => (<div key={g}><div className="group" style={{ margin: "14px 0 6px" }}>{g.toUpperCase()}</div>
            {val.checks.filter((x: any) => x.category === g).map((x: any) => (
              <div key={x.id} className={`finding ${statusTone(x.status)}`}>
                <h3>{x.name} <Badge tone={statusTone(x.status)}>{x.status.toUpperCase()}</Badge></h3>
                <p className="sub small" style={{ margin: 0 }}>{x.detail}</p>
                {x.recommendation && <p className="sub small" style={{ margin: "4px 0 0", color: "#d8e6c4" }}>→ {x.recommendation}</p>}
              </div>))}</div>))}
        </Panel>

        <Panel title="Customer-ready POC report">
          <div className="formgrid three">
            <Field label="Customer"><input className="input" value={meta.customer} onChange={(e) => setMeta({ ...meta, customer: e.target.value })} /></Field>
            <Field label="Report title"><input className="input" value={meta.title} onChange={(e) => setMeta({ ...meta, title: e.target.value })} /></Field>
            <Field label="Prepared by"><input className="input" value={meta.prepared_by} onChange={(e) => setMeta({ ...meta, prepared_by: e.target.value })} /></Field>
          </div>
          <div className="spacer" />
          <div className="formgrid">
            <Field label="Objectives (optional)"><textarea className="input" rows={2} value={meta.objectives} onChange={(e) => setMeta({ ...meta, objectives: e.target.value })} /></Field>
            <Field label="Performance acceptance criterion"><textarea className="input" rows={2} value={meta.acceptance_latency} onChange={(e) => setMeta({ ...meta, acceptance_latency: e.target.value })} /></Field>
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <Button busy={busy === "rep"} onClick={makeReport}>Generate POC report</Button>
            {report && <>
              <a className="btn ghost" href={`${API}${report.html_url}?download=true`}>Download HTML</a>
              <a className="btn ghost" href={`${API}${report.markdown_url}`}>Download Markdown</a>
              <span className="sub small">Report #{report.id} saved. Open the HTML and print to PDF for a signed copy.</span>
            </>}
          </div>
          {html && <iframe className="report" title="POC report preview" sandbox="" srcDoc={html} style={{ marginTop: 14 }} />}
        </Panel>
      </>}
    </main>
  );
}
