"use client";
import type { ReactNode } from "react";

export function PageHeader(props: { eyebrow: string; title: string; sub: string; right?: ReactNode }) {
  return (
    <div className="top">
      <div>
        <div className="eyebrow">{props.eyebrow}</div>
        <h1 className="title">{props.title}</h1>
        <p className="sub">{props.sub}</p>
      </div>
      {props.right && <div className="row">{props.right}</div>}
    </div>
  );
}

type Tone = "ok" | "warn" | "bad" | "info" | "muted";
export function Badge({ tone = "info", children, title }: { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span className={`badge ${tone}`} title={title}>
      {children}
    </span>
  );
}

export function Metric({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="metric">
      <small>{label}</small>
      <strong>{value}</strong>
      {hint && <small>{hint}</small>}
    </div>
  );
}

export function Panel({ title, right, children, className }: {
  title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`panel ${className || ""}`}>
      {(title || right) && (
        <div className="panelhead">
          {title && <h2>{title}</h2>}
          {right}
        </div>
      )}
      {children}
    </section>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <em>{hint}</em>}
    </label>
  );
}

export function Select<T extends string | number>(props: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; disabled?: boolean;
}) {
  return (
    <select
      className="input"
      value={String(props.value)}
      disabled={props.disabled}
      onChange={(e) => {
        const raw = e.target.value;
        const match = props.options.find((o) => String(o.value) === raw);
        if (match) props.onChange(match.value);
      }}
    >
      {props.options.map((o) => (
        <option key={String(o.value)} value={String(o.value)}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Num(props: { value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number }) {
  return (
    <input
      className="input"
      type="number"
      value={Number.isFinite(props.value) ? props.value : ""}
      min={props.min}
      max={props.max}
      step={props.step}
      onChange={(e) => props.onChange(e.target.value === "" ? NaN : Number(e.target.value))}
    />
  );
}

export function Tabs<T extends string>({ tabs, value, onChange }: {
  tabs: { id: T; label: string }[]; value: T; onChange: (t: T) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((t) => (
        <button key={t.id} role="tab" aria-selected={value === t.id}
          className={value === t.id ? "tab on" : "tab"} onClick={() => onChange(t.id)}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="errorbox" role="alert">{error}</div>;
}

export function Notice({ children }: { children: ReactNode }) {
  return <div className="notice">{children}</div>;
}

export function Button(props: {
  children: ReactNode; onClick?: () => void; busy?: boolean; disabled?: boolean;
  kind?: "primary" | "ghost" | "danger"; type?: "button" | "submit";
}) {
  return (
    <button type={props.type || "button"} className={`btn ${props.kind || "primary"}`}
      onClick={props.onClick} disabled={props.disabled || props.busy}>
      {props.busy ? "Working…" : props.children}
    </button>
  );
}

export function statusTone(s: string): Tone {
  const v = s.toLowerCase();
  if (["pass", "healthy", "validated", "ok", "info"].includes(v)) return v === "info" ? "info" : "ok";
  if (["warn", "warning", "conditionally validated"].includes(v)) return "warn";
  if (["fail", "critical", "error", "not validated"].includes(v)) return "bad";
  return "muted";
}
