"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { Badge, Button, ErrorBox, Field, Metric, PageHeader, Panel, Tabs } from "../../components/ui";
import { del, errMsg, fmt, get, post, upload } from "../../lib/api";

type Citation = { n: number; filename: string; page: number; text: string; document_id: number;
  scores: Record<string, number | null> };

const SAMPLE_EVAL = `What is the hotel reimbursement limit? | travel-policy.md
What do I do after Xid 79? | gpu-runbook.txt`;

export default function RagPage() {
  const [cfg, setCfg] = useState<any>(null);
  const [collection, setCollection] = useState("default");
  const [collections, setCollections] = useState<any[]>([]);
  const [docs, setDocs] = useState<any[]>([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<any>(null);
  const [active, setActive] = useState<number | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [tab, setTab] = useState<"ask" | "eval" | "log">("ask");
  const [evalText, setEvalText] = useState(SAMPLE_EVAL);
  const [evalRes, setEvalRes] = useState<any>(null);
  const [log, setLog] = useState<any[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [c, d] = await Promise.all([get("/api/v1/rag/collections"), get(`/api/v1/rag/documents?collection=${collection}`)]);
      setCollections(c);
      setDocs(d);
    } catch (e) {
      setError(errMsg(e));
    }
  }, [collection]);

  useEffect(() => { get("/api/v1/rag/config").then(setCfg).catch((e) => setError(errMsg(e))); }, []);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { if (tab === "log") get("/api/v1/rag/queries").then(setLog).catch(() => {}); }, [tab]);

  async function sendFiles(files: FileList | File[] | null) {
    if (!files || !files.length) return;
    setError(null);
    setBusy("upload");
    const form = new FormData();
    form.append("collection", collection);
    Array.from(files).forEach((f) => form.append("files", f));
    try {
      const out = await upload("/api/v1/rag/documents", form);
      const dupes = out.filter((d: any) => !d.created).length;
      if (dupes) setError(`${dupes} file(s) were already indexed in this collection (identical content).`);
      await refresh();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(null);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function ask() {
    if (question.trim().length < 3) return;
    setError(null);
    setBusy("ask");
    setActive(null);
    try {
      setResult(await post("/api/v1/rag/query", { question, collection, top_k: topK }));
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(null);
    }
  }

  async function remove(id: number) {
    if (!confirm("Delete this document and its vectors? This cannot be undone.")) return;
    try {
      await del(`/api/v1/rag/documents/${id}`);
      await refresh();
    } catch (e) {
      setError(errMsg(e));
    }
  }

  async function runEval() {
    setError(null);
    setBusy("eval");
    const cases = evalText.split("\n").map((l) => l.split("|").map((s) => s.trim())).filter((p) => p[0] && p[1])
      .map(([q, exp]) => (/\.[a-z0-9]+$/i.test(exp) ? { question: q, expected_filename: exp } : { question: q, expected_text: exp }));
    try {
      setEvalRes(await post("/api/v1/rag/evaluate", { collection, k: topK, cases }));
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(null);
    }
  }

  async function reindex() {
    setBusy("reindex");
    try {
      await post(`/api/v1/rag/collections/${collection}/reindex`);
      await refresh();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(null);
    }
  }

  const live = cfg?.embedding_provider === "nvidia";
  const stale = docs.some((d) => d.embedding_provider !== cfg?.embedding_provider);
  const t = result?.timings || {};

  return (
    <main>
      <PageHeader eyebrow="FLAGSHIP 01 · RETRIEVAL-AUGMENTED GENERATION" title="Enterprise RAG Assistant"
        sub="Ingest enterprise documents, retrieve with hybrid dense + BM25 search, rerank, and answer with page-level citations."
        right={cfg && <>
          <Badge tone={live ? "ok" : "warn"} title={cfg.embedding_label}>EMBED: {live ? "NVIDIA" : "LOCAL FALLBACK"}</Badge>
          <Badge tone={cfg.rerank === "nvidia" ? "ok" : "warn"}>RERANK: {cfg.rerank === "nvidia" ? "NVIDIA" : "LOCAL"}</Badge>
          <Badge tone={cfg.generation.startsWith("nvidia") ? "ok" : "warn"}>LLM: {cfg.generation.startsWith("nvidia") ? cfg.generation.split(":")[1] : "EXTRACTIVE"}</Badge>
        </>} />
      {cfg && !live && <div className="notice">Offline mode: embeddings are lexical hashing (not semantic) and answers are
        extractive quotes. Set <code>NVIDIA_NIM_API_KEY</code> in <code>backend/.env</code> for NVIDIA embeddings, reranking and generated answers.</div>}
      <ErrorBox error={error} />

      <div className="cols">
        <div>
          <Panel title="Knowledge collection" right={stale && docs.length ? <Button kind="ghost" busy={busy === "reindex"} onClick={reindex}>Re-index with current embedder</Button> : null}>
            <div className="formgrid">
              <Field label="Collection" hint="Lowercase letters, digits, - or _">
                <input className="input" list="collections" value={collection}
                  onChange={(e) => setCollection(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ""))} />
                <datalist id="collections">{collections.map((c) => <option key={c.name} value={c.name} />)}</datalist>
              </Field>
              <Field label="Results per answer (top-k)">
                <input className="input" type="number" min={1} max={20} value={topK} onChange={(e) => setTopK(Number(e.target.value) || 5)} />
              </Field>
            </div>
            <div className="spacer" />
            <div className={`drop ${over ? "over" : ""}`} role="button" tabIndex={0}
              onClick={() => fileRef.current?.click()}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); sendFiles(e.dataTransfer.files); }}>
              {busy === "upload" ? "Parsing, chunking and embedding…" : <>
                <strong style={{ color: "#d5ecbb" }}>Drop files or click to upload</strong><br />
                {cfg ? `${cfg.supported_types.join(" ")} · max ${cfg.max_upload_mb} MB · ${cfg.chunk_chars}-char chunks, ${cfg.chunk_overlap} overlap` : ""}
              </>}
            </div>
            <input ref={fileRef} type="file" multiple hidden accept={cfg?.supported_types.join(",")} onChange={(e) => sendFiles(e.target.files)} />
          </Panel>

          <Panel title={`Documents (${docs.length})`}>
            {docs.length === 0 ? <p className="sub">No documents in “{collection}” yet. Upload policies, manuals or runbooks to begin.</p> : (
              <div className="tablewrap"><table className="t">
                <thead><tr><th>File</th><th className="num">Pages</th><th className="num">Chunks</th><th>Embedder</th><th /></tr></thead>
                <tbody>{docs.map((d) => (
                  <tr key={d.id}>
                    <td>{d.filename}<div className="small muted">{fmt(d.size_bytes / 1024, 1)} KB</div></td>
                    <td className="num">{d.pages}</td><td className="num">{d.chunk_count}</td>
                    <td><Badge tone={d.embedding_provider === "nvidia" ? "ok" : "muted"}>{d.embedding_provider}</Badge></td>
                    <td><button className="btn danger" onClick={() => remove(d.id)} aria-label={`Delete ${d.filename}`}>Delete</button></td>
                  </tr>))}</tbody>
              </table></div>
            )}
          </Panel>
        </div>

        <div>
          <Tabs value={tab} onChange={setTab} tabs={[{ id: "ask", label: "Ask" }, { id: "eval", label: "Retrieval evaluation" }, { id: "log", label: "Query audit log" }]} />
          {tab === "ask" && <>
            <Panel>
              <Field label="Question">
                <textarea className="input" rows={3} value={question} placeholder="e.g. What is the escalation process for a GPU Xid 79 error?"
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(); }} />
              </Field>
              <div className="row" style={{ marginTop: 10 }}>
                <Button busy={busy === "ask"} disabled={!docs.length || question.trim().length < 3} onClick={ask}>Ask</Button>
                <span className="sub small">⌘/Ctrl + Enter</span>
              </div>
            </Panel>
            {result && <>
              <Panel title="Answer" right={<Badge tone={result.grounded ? "ok" : "warn"}>{result.grounded ? "GROUNDED · CITED" : "NO CITATIONS"}</Badge>}>
                <div className="answer">{renderAnswer(result.answer, active, setActive)}</div>
                <div className="grid five" style={{ marginBottom: 0 }}>
                  <Metric label="EMBED" value={`${fmt(t.embed_ms)} ms`} />
                  <Metric label="RETRIEVE" value={`${fmt(t.retrieval_ms)} ms`} />
                  <Metric label="RERANK" value={`${fmt(t.rerank_ms)} ms`} hint={result.providers.rerank} />
                  <Metric label="GENERATE" value={`${fmt(t.generation_ms)} ms`} hint={result.providers.generation} />
                  <Metric label="TOTAL" value={`${fmt(t.total_ms)} ms`} />
                </div>
              </Panel>
              <Panel title={`Sources (${result.citations.length})`}>
                {result.citations.map((c: Citation) => (
                  <div key={c.n} className={`source ${active === c.n ? "on" : ""}`} id={`src-${c.n}`}>
                    <div className="row"><button className={`cite ${active === c.n ? "on" : ""}`} onClick={() => setActive(c.n)}>{c.n}</button>
                      <strong style={{ fontSize: 12 }}>{c.filename}</strong><span className="sub small">page {c.page}</span>
                      {result.cited.includes(c.n) && <Badge tone="ok">CITED</Badge>}</div>
                    <p>{c.text.length > 600 ? c.text.slice(0, 600) + "…" : c.text}</p>
                    <div className="scores">{Object.entries(c.scores).map(([k, v]) => <span key={k}>{k}: {v === null ? "—" : fmt(v, 3)}</span>)}</div>
                  </div>
                ))}
              </Panel>
            </>}
          </>}

          {tab === "eval" && <Panel title="Retrieval evaluation (hit rate @k, MRR)">
            <p className="sub">One case per line: <code>question | expected filename</code> or <code>question | expected phrase</code>.</p>
            <textarea className="input" rows={6} value={evalText} onChange={(e) => setEvalText(e.target.value)} />
            <div className="row" style={{ marginTop: 10 }}><Button busy={busy === "eval"} onClick={runEval}>Run evaluation</Button></div>
            {evalRes && <>
              <div className="grid three"><Metric label={`HIT RATE @${evalRes.k}`} value={`${Math.round(evalRes.hit_rate * 100)}%`} />
                <Metric label="MRR" value={evalRes.mrr} /><Metric label="CASES" value={evalRes.cases} hint={evalRes.embedding_provider} /></div>
              <div className="tablewrap"><table className="t"><thead><tr><th>Question</th><th>Hit</th><th className="num">Rank</th><th>Top sources</th></tr></thead>
                <tbody>{evalRes.results.map((r: any, i: number) => (
                  <tr key={i}><td>{r.question}</td><td><Badge tone={r.hit ? "ok" : "bad"}>{r.hit ? "HIT" : "MISS"}</Badge></td>
                    <td className="num">{r.rank ?? "—"}</td><td className="small muted">{r.top_sources.slice(0, 3).join(", ")}</td></tr>))}</tbody></table></div>
            </>}
          </Panel>}

          {tab === "log" && <Panel title="Query audit log">
            {log.length === 0 ? <p className="sub">No questions asked yet.</p> : log.map((q) => (
              <div className="source" key={q.id}><div className="row"><strong style={{ fontSize: 12 }}>{q.question}</strong>
                <span className="sub small">{q.collection} · {fmt(q.latency_ms)} ms · {q.providers?.generation}</span></div>
                <p>{q.answer.length > 300 ? q.answer.slice(0, 300) + "…" : q.answer}</p></div>
            ))}
          </Panel>}
        </div>
      </div>
    </main>
  );
}

function renderAnswer(text: string, active: number | null, setActive: (n: number) => void) {
  return text.split(/(\[\d+\])/g).map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/);
    if (!m) return <span key={i}>{part}</span>;
    const n = Number(m[1]);
    return (
      <button key={i} className={`cite ${active === n ? "on" : ""}`} onClick={() => {
        setActive(n);
        document.getElementById(`src-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
      }}>{n}</button>
    );
  });
}
