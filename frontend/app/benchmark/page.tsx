"use client";
import { useEffect, useState } from "react";
import LineChart from "../../components/LineChart";
import { Badge, Button, ErrorBox, Field, Metric, Notice, Num, PageHeader, Panel, Select, Tabs } from "../../components/ui";
import { errMsg, fmt, get, post } from "../../lib/api";

type Tab = "estimate" | "compare" | "capacity" | "live" | "history";
const PRECISIONS = ["fp16", "bf16", "fp8", "int8", "int4"].map((p) => ({ value: p, label: p.toUpperCase() }));
const TP = [1, 2, 4, 8].map((n) => ({ value: n, label: `TP${n}` }));

const DEFAULT = { model: "llama-3.1-8b", gpu: "h100-sxm", weight_precision: "fp16", kv_precision: "fp16",
  tensor_parallel: 1, batch_size: 16, input_tokens: 1024, output_tokens: 256 };

export default function InferenceLab() {
  const [cat, setCat] = useState<any>(null);
  const [tab, setTab] = useState<Tab>("estimate");
  const [s, setS] = useState<any>(DEFAULT);
  const [est, setEst] = useState<any>(null);
  const [dim, setDim] = useState<"weight_precision" | "gpu" | "tensor_parallel" | "kv_precision">("weight_precision");
  const [cmp, setCmp] = useState<any[] | null>(null);
  const [cap, setCap] = useState({ concurrency: 200, max_ttft_ms: 1500, min_tokens_per_s_per_user: 25 });
  const [capRes, setCapRes] = useState<any>(null);
  const [live, setLive] = useState({ requests: 8, concurrency: 4, input_tokens: 256, output_tokens: 128 });
  const [liveRes, setLiveRes] = useState<any>(null);
  const [runs, setRuns] = useState<any[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { get("/api/v1/inference-lab/catalog").then(setCat).catch((e) => setError(errMsg(e))); }, []);
  useEffect(() => { if (tab === "history") get("/api/v1/inference-lab/runs").then(setRuns).catch(() => {}); }, [tab]);

  async function run<T>(key: string, fn: () => Promise<T>) {
    setError(null);
    setBusy(key);
    try { return await fn(); } catch (e) { setError(errMsg(e)); } finally { setBusy(null); }
  }
  const estimate = (save = false) => run("est", async () => setEst(await post(`/api/v1/inference-lab/estimate${save ? "?save=true" : ""}`, { ...s, label: `${s.model} on ${s.gpu}` })));
  useEffect(() => { if (cat) estimate(); /* first render */ }, [cat]); // eslint-disable-line react-hooks/exhaustive-deps

  async function compare() {
    const values: any[] = dim === "gpu" ? cat.gpus.map((g: any) => g.id)
      : dim === "tensor_parallel" ? [1, 2, 4, 8]
      : dim === "kv_precision" ? ["fp16", "fp8"] : ["fp16", "fp8", "int8", "int4"];
    await run("cmp", async () => setCmp(await post("/api/v1/inference-lab/compare", {
      scenarios: values.slice(0, 8).map((v) => ({ ...s, [dim]: v, label: String(v).toUpperCase() })) })));
  }

  const g = cat?.gpus.find((x: any) => x.id === s.gpu);
  const set = (k: string) => (v: any) => setS({ ...s, [k]: v });
  const m = est?.memory;

  return (
    <main>
      <PageHeader eyebrow="FLAGSHIP 02 · PERFORMANCE ENGINEERING" title="LLM Inference Performance Lab"
        sub="Explore how batching, quantization, KV-cache precision and tensor parallelism trade throughput against latency — then measure a real endpoint."
        right={<Badge tone={cat?.live_endpoint_configured ? "ok" : "muted"}>LIVE ENDPOINT: {cat?.live_endpoint_configured ? "CONFIGURED" : "NOT SET"}</Badge>} />
      <ErrorBox error={error} />
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "estimate", label: "Analytical model" }, { id: "compare", label: "Compare" },
        { id: "capacity", label: "Capacity planner" }, { id: "live", label: "Live benchmark" }, { id: "history", label: "Saved runs" }]} />

      {(tab === "estimate" || tab === "compare") && cat && <Panel title="Workload & deployment">
        <div className="formgrid four">
          <Field label="Model"><Select value={s.model} onChange={set("model")} options={cat.models.map((x: any) => ({ value: x.id, label: x.name }))} /></Field>
          <Field label="GPU"><Select value={s.gpu} onChange={set("gpu")} options={cat.gpus.map((x: any) => ({ value: x.id, label: x.name }))} /></Field>
          <Field label="Weight precision"><Select value={s.weight_precision} onChange={set("weight_precision")} options={PRECISIONS} /></Field>
          <Field label="KV-cache precision"><Select value={s.kv_precision} onChange={set("kv_precision")} options={PRECISIONS.filter((p) => ["fp16", "bf16", "fp8"].includes(p.value))} /></Field>
          <Field label="Tensor parallel"><Select value={s.tensor_parallel} onChange={set("tensor_parallel")} options={TP} /></Field>
          <Field label="Batch (concurrent sequences)"><Num value={s.batch_size} min={1} max={1024} onChange={set("batch_size")} /></Field>
          <Field label="Input tokens"><Num value={s.input_tokens} min={1} onChange={set("input_tokens")} /></Field>
          <Field label="Output tokens"><Num value={s.output_tokens} min={1} onChange={set("output_tokens")} /></Field>
        </div>
        {g && <p className="sub small" style={{ marginTop: 10 }}>{g.name}: {g.memory_gb} GB · {fmt(g.bandwidth_gbps)} GB/s · {g.fp16_tflops} TFLOPS FP16
          {g.fp8_tflops ? ` · ${g.fp8_tflops} TFLOPS FP8` : " · no FP8"} · {g.nvlink_gbps ? `NVLink ${g.nvlink_gbps} GB/s` : "PCIe only"} · {g.tdp_w} W</p>}
        <div className="row" style={{ marginTop: 12 }}>
          {tab === "estimate" ? <>
            <Button busy={busy === "est"} onClick={() => estimate()}>Estimate</Button>
            <Button kind="ghost" onClick={() => estimate(true)}>Estimate & save run</Button>
          </> : <>
            <Field label="Vary"><Select value={dim} onChange={setDim} options={[{ value: "weight_precision", label: "Weight precision (quantization)" },
              { value: "kv_precision", label: "KV-cache precision" }, { value: "tensor_parallel", label: "Tensor parallelism" }, { value: "gpu", label: "GPU type" }]} /></Field>
            <div style={{ alignSelf: "end" }}><Button busy={busy === "cmp"} onClick={compare}>Compare</Button></div>
          </>}
        </div>
      </Panel>}

      {tab === "estimate" && est && <>
        <Notice><Badge tone="warn">ANALYTICAL ESTIMATE</Badge>&nbsp; Roofline model from datasheet figures — not a hardware measurement or MLPerf result. Validate with the live benchmark.</Notice>
        <div className="grid">
          <Metric label="TIME TO FIRST TOKEN" value={`${fmt(est.latency.ttft_ms)} ms`} hint={`prefill of ${s.batch_size} × ${s.input_tokens} tokens`} />
          <Metric label="TIME PER OUTPUT TOKEN" value={`${fmt(est.latency.tpot_ms, 2)} ms`} hint={`${fmt(est.throughput.tokens_per_s_per_user, 1)} tokens/s per user`} />
          <Metric label="AGGREGATE THROUGHPUT" value={`${fmt(est.throughput.output_tokens_per_s)} tok/s`} hint={`${fmt(est.throughput.requests_per_s, 2)} requests/s`} />
          <Metric label="DECODE IS" value={est.decode_bound === "compute" ? "Compute-bound" : "Memory-bound"} hint={`max batch at this length: ${est.max_batch_at_this_sequence}`} />
        </div>
        <div className="cols even">
          <Panel title="GPU memory per device" right={<Badge tone={est.fits_in_memory ? "ok" : "bad"}>{est.fits_in_memory ? "FITS" : "DOES NOT FIT"}</Badge>}>
            <div className="bar" aria-label="memory breakdown">
              {[[m.weights_gb / s.tensor_parallel, "#76b900"], [m.kv_cache_gb / s.tensor_parallel, "#7cc4ff"], [m.overhead_gb_per_gpu, "#8b7bd8"]].map(([v, c], i) => (
                <i key={i} style={{ width: `${Math.min(100, (100 * (v as number)) / m.per_gpu_capacity_gb)}%`, background: c as string }} />))}
            </div>
            <div className="legend">
              <span><i style={{ background: "#76b900" }} />Weights {fmt(m.weights_gb / s.tensor_parallel, 1)} GB</span>
              <span><i style={{ background: "#7cc4ff" }} />KV cache {fmt(m.kv_cache_gb / s.tensor_parallel, 1)} GB</span>
              <span><i style={{ background: "#8b7bd8" }} />Runtime {m.overhead_gb_per_gpu} GB</span>
              <span>Used {m.per_gpu_used_gb} / {m.per_gpu_capacity_gb} GB ({m.utilization_pct}%)</span>
            </div>
            <dl className="kv" style={{ marginTop: 14 }}>
              <dt>KV cache per token</dt><dd>{fmt(m.kv_cache_mb_per_token * 1000, 1)} KB</dd>
              <dt>Decode step: memory</dt><dd>{est.decode_breakdown_ms.memory} ms</dd>
              <dt>Decode step: compute</dt><dd>{est.decode_breakdown_ms.compute} ms</dd>
              <dt>Decode step: TP comms</dt><dd>{est.decode_breakdown_ms.tp_communication} ms</dd>
            </dl>
            {est.warnings.length > 0 && <ul className="list">{est.warnings.map((w: string) => <li key={w} style={{ color: "#e6d39b" }}>{w}</li>)}</ul>}
            {est.run_id && <p className="sub small">Saved as run #{est.run_id}.</p>}
          </Panel>
          <Panel title="Batch-size sweep">
            <LineChart xLabel="Batch size" yLabel="tokens/s" series={[
              { name: "Aggregate output tok/s", color: "#76b900", points: est.sweep.filter((p: any) => p.fits).map((p: any) => ({ x: p.batch_size, y: p.output_tokens_per_s })) },
            ]} />
            <LineChart xLabel="Batch size" yLabel="ms" height={180} series={[
              { name: "TTFT ms", color: "#7cc4ff", points: est.sweep.filter((p: any) => p.fits).map((p: any) => ({ x: p.batch_size, y: p.ttft_ms })) },
              { name: "Time per token ms ×10", color: "#e6c35c", points: est.sweep.filter((p: any) => p.fits).map((p: any) => ({ x: p.batch_size, y: p.tpot_ms * 10 })) },
            ]} />
            <p className="sub small">Bigger batches raise aggregate throughput until decode turns compute-bound, while each user's latency grows — the core serving trade-off. Points that exceed GPU memory are hidden.</p>
          </Panel>
        </div>
        <Panel title="Model assumptions"><ul className="list">{est.assumptions.map((a: string) => <li key={a}>{a}</li>)}</ul></Panel>
      </>}

      {tab === "compare" && cmp && <Panel title="Scenario comparison" right={<Badge tone="warn">ANALYTICAL ESTIMATE</Badge>}>
        <div className="tablewrap"><table className="t"><thead><tr><th>Scenario</th><th>Fits</th><th className="num">GB / GPU</th><th className="num">TTFT ms</th>
          <th className="num">ms / token</th><th className="num">tok/s / user</th><th className="num">Aggregate tok/s</th><th>Bound</th><th>Warnings</th></tr></thead>
          <tbody>{cmp.map((r) => {
            const best = Math.max(...cmp.filter((x) => x.fits_in_memory).map((x) => x.throughput.output_tokens_per_s));
            return (<tr key={r.label}>
              <td><strong>{r.label}</strong><div className="small muted">{r.gpu_name}</div></td>
              <td><Badge tone={r.fits_in_memory ? "ok" : "bad"}>{r.fits_in_memory ? "YES" : "NO"}</Badge></td>
              <td className="num">{r.memory.per_gpu_used_gb}</td><td className="num">{fmt(r.latency.ttft_ms)}</td>
              <td className="num">{r.latency.tpot_ms}</td><td className="num">{fmt(r.throughput.tokens_per_s_per_user, 1)}</td>
              <td className="num" style={{ color: r.throughput.output_tokens_per_s === best && r.fits_in_memory ? "#9be15d" : undefined }}>{fmt(r.throughput.output_tokens_per_s)}</td>
              <td className="small">{r.decode_bound}</td><td className="small muted">{r.warnings.filter((w: string) => !w.startsWith("Quantized")).join(" ") || "—"}</td>
            </tr>);
          })}</tbody></table></div>
      </Panel>}

      {tab === "capacity" && cat && <Panel title="How many GPUs for my SLO?">
        <div className="formgrid four">
          <Field label="Model"><Select value={s.model} onChange={set("model")} options={cat.models.map((x: any) => ({ value: x.id, label: x.name }))} /></Field>
          <Field label="Peak concurrent users"><Num value={cap.concurrency} onChange={(v) => setCap({ ...cap, concurrency: v })} /></Field>
          <Field label="Max TTFT (ms)"><Num value={cap.max_ttft_ms} onChange={(v) => setCap({ ...cap, max_ttft_ms: v })} /></Field>
          <Field label="Min tokens/s per user"><Num value={cap.min_tokens_per_s_per_user} onChange={(v) => setCap({ ...cap, min_tokens_per_s_per_user: v })} /></Field>
        </div>
        <div className="row" style={{ marginTop: 12 }}><Button busy={busy === "cap"} onClick={() => run("cap", async () => setCapRes(await post("/api/v1/inference-lab/capacity",
          { model: s.model, input_tokens: s.input_tokens, output_tokens: s.output_tokens, ...cap })))}>Find configurations</Button>
          <span className="sub small">Uses {s.input_tokens} input / {s.output_tokens} output tokens from the workload settings.</span></div>
        {capRes && (capRes.options.length === 0 ? <p className="sub">No configuration meets this SLO. Relax latency targets or choose a smaller model.</p> :
          <div className="tablewrap" style={{ marginTop: 14 }}><table className="t"><thead><tr><th>GPU</th><th>Precision</th><th>TP</th><th className="num">Batch / replica</th>
            <th className="num">Replicas</th><th className="num">Total GPUs</th><th className="num">TTFT ms</th><th className="num">tok/s / user</th><th className="num">Cost index</th></tr></thead>
            <tbody>{capRes.options.map((o: any, i: number) => (<tr key={i}><td>{i === 0 && <Badge tone="ok">BEST</Badge>} {o.gpu_name}</td><td>{o.precision.toUpperCase()}</td>
              <td>{o.tensor_parallel}</td><td className="num">{o.batch_per_replica}</td><td className="num">{o.replicas}</td><td className="num"><strong>{o.total_gpus}</strong></td>
              <td className="num">{fmt(o.ttft_ms)}</td><td className="num">{fmt(o.tokens_per_s_per_user)}</td><td className="num">{o.relative_cost_index}</td></tr>))}</tbody></table>
            <p className="sub small">{capRes.note}</p></div>)}
      </Panel>}

      {tab === "live" && <Panel title="Live load test" right={<Badge tone="info">LIVE_MEASUREMENT</Badge>}>
        {!cat?.live_endpoint_configured && <Notice>No endpoint configured. Add <code>NVIDIA_NIM_API_KEY</code> (NVIDIA API catalog) or a self-hosted
          <code> NVIDIA_NIM_BASE_URL</code> to <code>backend/.env</code> and restart the backend to measure real TTFT, inter-token latency and throughput.</Notice>}
        <div className="formgrid four" style={{ marginTop: 12 }}>
          {(["requests", "concurrency", "input_tokens", "output_tokens"] as const).map((k) => (
            <Field key={k} label={k.replace("_", " ")}><Num value={live[k]} onChange={(v) => setLive({ ...live, [k]: v })} /></Field>))}
        </div>
        <div className="row" style={{ marginTop: 12 }}><Button busy={busy === "live"} disabled={!cat?.live_endpoint_configured}
          onClick={() => run("live", async () => setLiveRes(await post("/api/v1/inference-lab/live", live)))}>Run live benchmark</Button>
          <span className="sub small">Streams real completions and measures on the client, including network time. Guard rails: ≤ 64 requests, ≤ 16 concurrent.</span></div>
        {liveRes && <>
          <div className="grid">
            <Metric label="TTFT p50 / p90" value={`${fmt(liveRes.ttft_ms.p50)} / ${fmt(liveRes.ttft_ms.p90)} ms`} />
            <Metric label="INTER-TOKEN p50" value={`${fmt(liveRes.inter_token_latency_ms.p50, 1)} ms`} />
            <Metric label="OUTPUT THROUGHPUT" value={`${fmt(liveRes.output_tokens_per_s, 1)} tok/s`} hint={`${liveRes.requests_per_s} req/s`} />
            <Metric label="SUCCESS" value={`${liveRes.succeeded}/${liveRes.requests}`} hint={liveRes.model} />
          </div>
          {liveRes.errors.length > 0 && <ErrorBox error={liveRes.errors.join(" · ")} />}
          <p className="sub small">{liveRes.note} Endpoint: {liveRes.endpoint}</p>
        </>}
      </Panel>}

      {tab === "history" && <Panel title="Saved runs">
        {runs.length === 0 ? <p className="sub">No saved runs. Use “Estimate & save run” or run a live benchmark.</p> :
          <div className="tablewrap"><table className="t"><thead><tr><th>#</th><th>Type</th><th>Label</th><th className="num">tok/s</th><th className="num">TTFT ms</th><th>When</th></tr></thead>
            <tbody>{runs.map((r) => (<tr key={r.id}><td>{r.id}</td><td><Badge tone={r.kind === "live" ? "info" : "warn"}>{r.kind.toUpperCase()}</Badge></td><td>{r.label}</td>
              <td className="num">{fmt(r.results.output_tokens_per_s ?? r.results.throughput?.output_tokens_per_s)}</td>
              <td className="num">{fmt(r.results.ttft_ms?.p50 ?? r.results.latency?.ttft_ms)}</td><td className="small muted">{new Date(r.created_at).toLocaleString()}</td></tr>))}</tbody></table></div>}
      </Panel>}
    </main>
  );
}
