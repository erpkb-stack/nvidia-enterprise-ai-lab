"use client";

export type Series = { name: string; color: string; points: { x: number; y: number }[] };

/** Dependency-free SVG line chart with log-2 x axis (batch sizes) and hover-free labels. */
export default function LineChart({ series, xLabel, yLabel, logX = true, height = 220 }: {
  series: Series[]; xLabel: string; yLabel: string; logX?: boolean; height?: number;
}) {
  const W = 560, H = height, P = { l: 56, r: 14, t: 14, b: 38 };
  const pts = series.flatMap((s) => s.points).filter((p) => Number.isFinite(p.y));
  if (!pts.length) return <p className="sub">No data points.</p>;
  const tx = (x: number) => (logX ? Math.log2(x) : x);
  const xs = pts.map((p) => tx(p.x));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y1 = Math.max(...pts.map((p) => p.y)) * 1.1 || 1;
  const sx = (x: number) => P.l + ((tx(x) - x0) / (x1 - x0 || 1)) * (W - P.l - P.r);
  const sy = (y: number) => H - P.b - (y / y1) * (H - P.t - P.b);
  const xticks = Array.from(new Set(pts.map((p) => p.x))).sort((a, b) => a - b);
  const yticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * y1);
  const short = (v: number) => (v >= 10000 ? `${Math.round(v / 1000)}k` : v >= 1000 ? `${(v / 1000).toFixed(1)}k` : `${Math.round(v)}`);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${yLabel} by ${xLabel}`}>
        {yticks.map((v) => (
          <g key={v}>
            <line x1={P.l} x2={W - P.r} y1={sy(v)} y2={sy(v)} className="grid" />
            <text x={P.l - 8} y={sy(v) + 4} textAnchor="end" className="tick">{short(v)}</text>
          </g>
        ))}
        {xticks.map((x) => (
          <text key={x} x={sx(x)} y={H - P.b + 16} textAnchor="middle" className="tick">{x}</text>
        ))}
        <text x={(W + P.l) / 2} y={H - 4} textAnchor="middle" className="axis">{xLabel}</text>
        <text x={12} y={H / 2} textAnchor="middle" className="axis" transform={`rotate(-90 12 ${H / 2})`}>{yLabel}</text>
        {series.map((s) => {
          const ps = s.points.filter((p) => Number.isFinite(p.y));
          return (
            <g key={s.name}>
              <polyline fill="none" stroke={s.color} strokeWidth={2.2}
                points={ps.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
              {ps.map((p) => (
                <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={3} fill={s.color}>
                  <title>{`${s.name}: ${short(p.y)} at ${xLabel.toLowerCase()} ${p.x}`}</title>
                </circle>
              ))}
            </g>
          );
        })}
      </svg>
      <figcaption className="legend">
        {series.map((s) => (
          <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>
        ))}
      </figcaption>
    </figure>
  );
}
