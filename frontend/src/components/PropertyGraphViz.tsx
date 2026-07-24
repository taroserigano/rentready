import type { PropertyGraphNode, PropertyGraphEdge } from "../types";

const TYPE_ORDER: PropertyGraphNode["type"][] = ["Neighborhood", "Property", "Amenity"];
/** Fixed categorical hue order (identity, never cycled) — same 3-hue family
 * used elsewhere in this app's charts. */
const TYPE_COLOR: Record<PropertyGraphNode["type"], string> = {
  Neighborhood: "var(--chart-2)",
  Property: "var(--chart-1)",
  Amenity: "var(--chart-5)",
};

const ROW_H = 30;
const PAD_Y = 18;
const WIDTH = 320;
const COL_X: Record<PropertyGraphNode["type"], number> = {
  Neighborhood: 34,
  Property: 160,
  Amenity: 286,
};

/** A small tripartite node-link diagram: Neighborhood — Property — Amenity.
 * A deterministic 3-column layout (no force simulation needed for graphs this
 * small) avoids node-overlap entirely and keeps left/right edges legible.
 * Categorical color by node TYPE (identity, fixed hue order); a legend names
 * the 3 types since ≥2 series are present. */
export function PropertyGraphViz({
  nodes,
  edges,
}: {
  nodes: PropertyGraphNode[];
  edges: PropertyGraphEdge[];
}) {
  if (!nodes.length) return null;

  const byType: Record<PropertyGraphNode["type"], PropertyGraphNode[]> = {
    Neighborhood: [],
    Property: [],
    Amenity: [],
  };
  for (const n of nodes) byType[n.type]?.push(n);

  const maxCount = Math.max(1, ...TYPE_ORDER.map((t) => byType[t].length));
  const height = Math.max(140, maxCount * ROW_H + PAD_Y * 2);

  const pos = new Map<string, { x: number; y: number }>();
  for (const type of TYPE_ORDER) {
    const list = byType[type];
    const n = list.length;
    const usable = height - PAD_Y * 2;
    list.forEach((node, i) => {
      const y = n === 1 ? height / 2 : PAD_Y + (i * usable) / (n - 1);
      pos.set(node.id, { x: COL_X[type], y });
    });
  }

  return (
    <div className="property-graph">
      <div className="property-graph-head">
        <span className="eyebrow">Property graph</span>
        <span className="muted" style={{ fontSize: 12 }}>
          nodes and relationships behind this answer
        </span>
      </div>
      {/* aspect-ratio (not a pixel height) so the container's rendered shape
          always matches the viewBox exactly -- the SVG's default "meet"
          scaling then fills edge-to-edge with no letterboxing, which is what
          keeps the percentage-positioned HTML label overlay aligned with the
          SVG dots in both dimensions. */}
      <div className="property-graph-plot" style={{ aspectRatio: `${WIDTH} / ${height}` }}>
        <svg
          viewBox={`0 0 ${WIDTH} ${height}`}
          className="property-graph-svg"
          role="img"
          aria-label={`Property graph: ${nodes.length} nodes (${TYPE_ORDER.map((t) => `${byType[t].length} ${t.toLowerCase()}`).join(", ")}), ${edges.length} relationships`}
        >
          {edges.map((e, i) => {
            const s = pos.get(e.source);
            const t = pos.get(e.target);
            if (!s || !t) return null;
            const midX = (s.x + t.x) / 2;
            return (
              <path
                key={i}
                d={`M ${s.x} ${s.y} C ${midX} ${s.y}, ${midX} ${t.y}, ${t.x} ${t.y}`}
                className="property-graph-edge"
              />
            );
          })}
          {nodes.map((n) => {
            const p = pos.get(n.id);
            if (!p) return null;
            return (
              <circle key={n.id} cx={p.x} cy={p.y} r={4} fill={TYPE_COLOR[n.type]}>
                <title>{`${n.label} (${n.type})`}</title>
              </circle>
            );
          })}
        </svg>
        <div className="property-graph-labels">
          {nodes.map((n) => {
            const p = pos.get(n.id);
            if (!p) return null;
            return (
              <span
                key={n.id}
                className={`property-graph-label align-${n.type}`}
                style={{ left: `${(p.x / WIDTH) * 100}%`, top: `${(p.y / height) * 100}%` }}
                title={n.label}
              >
                {n.label}
              </span>
            );
          })}
        </div>
      </div>
      <div className="property-graph-legend">
        {TYPE_ORDER.filter((t) => byType[t].length > 0).map((t) => (
          <div key={t} className="property-graph-legend-item">
            <span className="property-graph-swatch" style={{ background: TYPE_COLOR[t] }} />
            <span className="muted" style={{ fontSize: 12 }}>
              {t}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
