"use client";
import { useEffect, useState } from "react";
import ArchDiagram from "../../components/ArchDiagram";
import { Badge, Button, ErrorBox, Field, Metric, Num, PageHeader, Panel, Select } from "../../components/ui";
import { download, errMsg, fmt, get, post } from "../../lib/api";

const DEFAULT = {
  project_name: "Enterprise Knowledge Assistant POC", customer: "Internal Lab", use_case_id: "rag-assistant",
  concurrent_users: 200, input_tokens: 2048, output_tokens: 384, max_ttft_ms: 2000, min_tokens_per_s_per_user: 20,
  model_size: "small", data_sensitivity: "internal", deployment: "on_prem", availability: "high", document_volume_gb: 20,
};
const opts = (xs: [string, string][]) => xs.map(([value, label]) => ({ value, label }));

export default function Architect() {
  const [cases, setCases] = useState<any[]>([]);
  const [req, setReq] = useState<any>(DEFAULT);
  const [res, setRes] = useState<any>(null);
  const [saved, setSaved] = useState<string>("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    // Deep link from the dashboard: /architect?usecase=<id> (read after mount to avoid hydration mismatch)
    const uc = new URLSearchParams(window.location.search).get("usecase");
    if (uc) setReq((r: any) => ({ ...r, use_case_id: uc }));
    get("/api/v1/use-cases").then(setCases).catch((e) => setError(errMsg(e)));
  }, []);
  const set = (k: string) => (v: any) => setReq({ ...req, [k]: v });

  async function generate(save: boolean) {
    setError(null);
    setBusy(save ? "save" : "gen");
    try {
      const out = await post(save ? "/api/v1/architect/design/save" : "/api/v1/architect/design", req);
      setRes(out);
      setSaved(save ? `Saved as project #${out.project_id}, architecture #${out.architecture_id}` : "");
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(null);
    }
  }

  const uc = cases.find((c) => c.id === req.use_case_id);
  const rec = res?.sizing?.recommended;
  return (
    <main>
      <PageHeader eyebrow="FLAGSHIP 03 · SOLUTION ARCHITECTURE" title="Enterprise Solution Architect Studio"
        sub="Turn business requirements into a reference architecture, explainable technology choices, GPU sizing and a phased deployment plan." />
      <ErrorBox error={error} />
      <Panel title="Business & technical requirements">
        <div className="formgrid four">
          <Field label="Project name"><input className="input" value={req.project_name} onChange={(e) => set("project_name")(e.target.value)} /></Field>
          <Field label="Customer"><input className="input" value={req.customer} onChange={(e) => set("customer")(e.target.value)} /></Field>
          <Field label="Use case"><Select value={req.use_case_id} onChange={set("use_case_id")} options={cases.map((c) => ({ value: c.id, label: c.name }))} /></Field>
          <Field label="Model class"><Select value={req.model_size} onChange={set("model_size")} options={opts([["small", "Small · Llama 3.1 8B"], ["medium", "Medium · Mixtral 8x7B"], ["large", "Large · Llama 3.1 70B"]])} /></Field>
          <Field label="Peak concurrent users"><Num value={req.concurrent_users} onChange={set("concurrent_users")} /></Field>
          <Field label="Avg prompt tokens" hint="incl. retrieved context"><Num value={req.input_tokens} onChange={set("input_tokens")} /></Field>
          <Field label="Avg response tokens"><Num value={req.output_tokens} onChange={set("output_tokens")} /></Field>
          <Field label="Document volume (GB)"><Num value={req.document_volume_gb} onChange={set("document_volume_gb")} /></Field>
          <Field label="Max time to first token (ms)"><Num value={req.max_ttft_ms} onChange={set("max_ttft_ms")} /></Field>
          <Field label="Min tokens/s per user"><Num value={req.min_tokens_per_s_per_user} onChange={set("min_tokens_per_s_per_user")} /></Field>
          <Field label="Data sensitivity"><Select value={req.data_sensitivity} onChange={set("data_sensitivity")} options={opts([["public", "Public"], ["internal", "Internal"], ["regulated", "Regulated (PII / financial / health)"]])} /></Field>
          <Field label="Deployment"><Select value={req.deployment} onChange={set("deployment")} options={opts([["on_prem", "On-premises"], ["cloud", "Cloud"], ["hybrid", "Hybrid"], ["air_gapped", "Air-gapped"]])} /></Field>
          <Field label="Availability"><Select value={req.availability} onChange={set("availability")} options={opts([["standard", "Standard"], ["high", "High (N+1, multi-node)"]])} /></Field>
        </div>
        {uc && <p className="sub" style={{ marginTop: 12 }}><Badge tone="info">{uc.pattern}</Badge> &nbsp;{uc.problem} Outcome: {uc.outcome}</p>}
        <div className="row" style={{ marginTop: 12 }}>
          <Button busy={busy === "gen"} onClick={() => generate(false)}>Generate architecture</Button>
          <Button kind="ghost" busy={busy === "save"} onClick={() => generate(true)}>Generate & save to project</Button>
          <span className="sub small">{saved}</span>
        </div>
      </Panel>

      {res && <>
        <Panel title={res.title} right={<div className="row">
          <Button kind="ghost" onClick={() => download(`${req.project_name.replace(/\W+/g, "-").toLowerCase()}-architecture.md`, res.markdown, "text/markdown")}>Download design doc (.md)</Button>
          <Button kind="ghost" onClick={() => navigator.clipboard?.writeText(res.diagram.mermaid)}>Copy Mermaid</Button></div>}>
          <p className="answer" style={{ fontSize: 13 }}>{res.summary}</p>
          {rec && <div className="grid">
            <Metric label="GPUS" value={`${rec.total_gpus} × ${rec.gpu_name.split(" ")[0]}`} hint={`${rec.nodes} node(s), ${rec.replicas} replicas`} />
            <Metric label="PRECISION / TP" value={`${rec.precision.toUpperCase()} · TP${rec.tensor_parallel}`} hint={`batch ${rec.batch_per_replica} per replica`} />
            <Metric label="EST. TTFT" value={`${fmt(rec.ttft_ms)} ms`} hint={`${fmt(rec.tokens_per_s_per_user)} tok/s per user`} />
            <Metric label="DELIVERY" value={`${res.total_weeks} weeks`} hint={`${res.deployment_plan.length} phases`} />
          </div>}
          {!rec && <div className="errorbox">{res.sizing.note}</div>}
          <ArchDiagram nodes={res.diagram.nodes} edges={res.diagram.edges} />
        </Panel>

        <div className="cols even">
          <Panel title="Technology selection">
            <div className="tablewrap"><table className="t"><thead><tr><th>Layer</th><th>Technology & rationale</th></tr></thead>
              <tbody>{res.stack.map((t: any, i: number) => (<tr key={i}><td className="small muted">{t.layer}</td>
                <td><strong>{t.technology}</strong><div className="small muted" style={{ marginTop: 3 }}>{t.rationale}</div></td></tr>))}</tbody></table></div>
          </Panel>
          <div>
            <Panel title={`GPU sizing — ${res.sizing.model}`} right={<Badge tone="warn">ANALYTICAL</Badge>}>
              <div className="tablewrap"><table className="t"><thead><tr><th>GPU</th><th>Prec.</th><th className="num">GPUs</th><th className="num">TTFT</th><th className="num">tok/s/user</th><th className="num">Cost idx</th></tr></thead>
                <tbody>{res.sizing.options.map((o: any, i: number) => (<tr key={i}><td>{i === 0 && <Badge tone="ok">REC</Badge>} {o.gpu_name}</td>
                  <td>{o.precision.toUpperCase()} TP{o.tensor_parallel}</td><td className="num">{o.total_gpus}</td><td className="num">{fmt(o.ttft_ms)}</td>
                  <td className="num">{fmt(o.tokens_per_s_per_user)}</td><td className="num">{o.relative_cost_index}</td></tr>))}</tbody></table></div>
              {res.vector_store && <p className="sub small">Vector store: ~{fmt(res.vector_store.estimated_chunks)} chunks, ~{res.vector_store.vector_memory_gb} GB vector memory — {res.vector_store.recommendation}.</p>}
            </Panel>
            <Panel title="Risks">
              {res.risks.map((r: any) => (<div key={r.risk} className={`finding ${r.severity === "High" ? "bad" : "warn"}`}>
                <h3>{r.risk} <Badge tone={r.severity === "High" ? "bad" : "warn"}>{r.severity.toUpperCase()}</Badge></h3>
                <p className="sub small" style={{ margin: 0 }}>{r.mitigation}</p></div>))}
            </Panel>
          </div>
        </div>

        <Panel title={`Deployment plan · ${res.total_weeks} weeks`}>
          <div className="cols even">{res.deployment_plan.map((p: any) => (
            <div className="phase" key={p.phase}><h3>{p.phase} <span className="sub small">· {p.weeks} weeks</span></h3>
              <ul className="list">{p.activities.map((a: string) => <li key={a}>{a}</li>)}</ul>
              <p className="sub small" style={{ margin: "6px 0 0" }}><strong>Exit:</strong> {p.exit_criteria}</p></div>))}
          </div>
        </Panel>
      </>}
    </main>
  );
}
