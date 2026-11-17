"use client";

type Node = { id: string; label: string; group: string };
type Edge = { source: string; target: string };

const COLUMNS = ["client", "edge", "app", "safety", "data", "gpu", "ops"];
const TITLES: Record<string, string> = {
  client: "Consumers", edge: "Edge", app: "Application", safety: "Safety",
  data: "Data & retrieval", gpu: "GPU inference", ops: "Platform & ops",
};

/** Lays nodes out in columns by group and draws curved edges. No external diagram library. */
export default function ArchDiagram({ nodes, edges }: { nodes: Node[]; edges: Edge[] }) {
  const cols = COLUMNS.filter((c) => nodes.some((n) => n.group === c));
  const BW = 150, BH = 46, GX = 46, GY = 22, TOP = 34;
  const pos: Record<string, { x: number; y: number }> = {};
  let maxRows = 0;
  cols.forEach((c, ci) => {
    const items = nodes.filter((n) => n.group === c);
    maxRows = Math.max(maxRows, items.length);
    items.forEach((n, ri) => (pos[n.id] = { x: 10 + ci * (BW + GX), y: TOP + ri * (BH + GY) }));
  });
  const W = 20 + cols.length * (BW + GX) - GX, H = TOP + maxRows * (BH + GY) + 10;
  const wrap = (s: string) => {
    if (s.length <= 22) return [s];
    const words = s.split(" "); const lines = [""];
    for (const w of words) {
      if ((lines[lines.length - 1] + " " + w).trim().length > 22) lines.push(w);
      else lines[lines.length - 1] = (lines[lines.length - 1] + " " + w).trim();
    }
    return lines.slice(0, 2);
  };
  return (
    <div className="diagram">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Reference architecture diagram">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" className="arrowhead" />
          </marker>
        </defs>
        {cols.map((c, ci) => (
          <text key={c} x={10 + ci * (BW + GX) + BW / 2} y={18} textAnchor="middle" className="coltitle">
            {TITLES[c] || c}
          </text>
        ))}
        {edges.map((e, i) => {
          const a = pos[e.source], b = pos[e.target];
          if (!a || !b) return null;
          const forward = b.x > a.x;
          const x1 = forward ? a.x + BW : a.x + BW / 2, y1 = forward ? a.y + BH / 2 : a.y + BH;
          const x2 = forward ? b.x : b.x + BW / 2, y2 = forward ? b.y + BH / 2 : b.y;
          const d = forward
            ? `M${x1},${y1} C${x1 + GX / 2},${y1} ${x2 - GX / 2},${y2} ${x2},${y2}`
            : `M${x1},${y1} C${x1},${y1 + 30} ${x2},${y2 - 30} ${x2},${y2}`;
          return <path key={i} d={d} className="edge" markerEnd="url(#arrow)" />;
        })}
        {nodes.map((n) => {
          const p = pos[n.id];
          if (!p) return null;
          const lines = wrap(n.label);
          return (
            <g key={n.id}>
              <rect x={p.x} y={p.y} width={BW} height={BH} rx={8} className={`node ${n.group}`} />
              {lines.map((l, i) => (
                <text key={i} x={p.x + BW / 2} y={p.y + BH / 2 + 4 + (i - (lines.length - 1) / 2) * 14}
                  textAnchor="middle" className="nodetext">{l}</text>
              ))}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
