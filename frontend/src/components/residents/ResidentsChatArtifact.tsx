import { memo } from "react";
import type {
  PropertyHealth,
  ReasonCode,
  ResidentChatArtifact as Artifact,
  ResidentHead,
  RiskBand,
} from "../../types";
import { RiskGauge } from "../risk/RiskGauge";
import { ReasonCodes } from "../risk/ReasonCodes";
import { ResidentFamilies } from "./ResidentFamilies";
import { HealthList } from "./PropertyHealthRanking";
import { formatProbRange, headLabel, usd } from "./residentsTone";

/**
 * Renders the chat turn's grounded artifact DEFENSIVELY. The backend's artifact
 * is a loose dict with no discriminated `kind`, so we sniff its shape:
 *   • a resident prediction bundle (heads + families) → compact family sections
 *   • a single head-like payload (probability/band, or expected/interval,
 *     or class_probs) → the matching gauge / value / reason codes
 *   • a property-health ranking (array of {score, grade}) → a compact list
 *   • anything else → nothing (prose alone carries the answer)
 * Reuses the same primitives as the detail page so numbers render identically.
 */

type Dict = Record<string, unknown>;

function asDict(a: unknown): Dict | null {
  return a && typeof a === "object" && !Array.isArray(a) ? (a as Dict) : null;
}

/** Pull a property-health list out of an array or a wrapped object. */
function asHealthList(a: unknown): PropertyHealth[] | null {
  const d = asDict(a);
  const arr: unknown = Array.isArray(a)
    ? a
    : (d?.ranking ?? d?.properties ?? d?.rows ?? d?.health);
  if (!Array.isArray(arr) || arr.length === 0) return null;
  const looksHealthy = arr.every((x) => {
    const o = asDict(x);
    return o && typeof o.score === "number" && "grade" in o && "property_id" in o;
  });
  return looksHealthy ? (arr as PropertyHealth[]) : null;
}

/** A resident prediction bundle carries both the heads map and the families map. */
function asBundle(
  a: unknown,
): { heads: Record<string, ResidentHead>; families: Record<string, string[]>; name?: string } | null {
  const d = asDict(a);
  if (!d) return null;
  const heads = asDict(d.heads);
  const families = asDict(d.families);
  if (heads && families && Object.keys(heads).length > 0) {
    return {
      heads: heads as unknown as Record<string, ResidentHead>,
      families: families as unknown as Record<string, string[]>,
      name: typeof d.name === "string" ? d.name : undefined,
    };
  }
  return null;
}

/** Reason codes if the payload carries a sane array of them. */
function reasonCodes(d: Dict): ReasonCode[] {
  const rc = d.reason_codes;
  return Array.isArray(rc) ? (rc as ReasonCode[]) : [];
}

/** A single binary-head-like payload (probability + band). */
function BinaryArtifact({ d }: { d: Dict }) {
  const probability = Number(d.probability);
  const band = (d.band as RiskBand) ?? "medium";
  const range = (Array.isArray(d.range) ? d.range : [probability, probability]) as [number, number];
  const label = typeof d.name === "string" ? headLabel(d.name) : undefined;
  return (
    <div style={{ marginTop: 10 }}>
      {label && <div className="eyebrow" style={{ textAlign: "center" }}>{label}</div>}
      <RiskGauge probability={probability} band={band} />
      <div style={{ textAlign: "center", fontSize: 12, color: "var(--text-2)", marginTop: 4 }}>
        {formatProbRange(probability, range)}
      </div>
      {reasonCodes(d).length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div className="eyebrow">Key factors</div>
          <ReasonCodes codes={reasonCodes(d)} />
        </div>
      )}
    </div>
  );
}

/** A count/regression-like payload (expected + interval). */
function ScalarArtifact({ d }: { d: Dict }) {
  const expected = Number(d.expected);
  const interval = (Array.isArray(d.interval) ? d.interval : [expected, expected]) as [number, number];
  const name = typeof d.name === "string" ? d.name : "";
  const money = /arrears|balance/.test(name);
  const fmt = (n: number) => (money ? usd(n) : (Math.round(n * 10) / 10).toLocaleString());
  return (
    <div className="subpanel" style={{ marginTop: 10, textAlign: "center" }}>
      {name && <div className="eyebrow">{headLabel(name)}</div>}
      <div className="big-num" style={{ color: "var(--accent-text)", marginTop: 4 }}>
        {fmt(expected)}
      </div>
      <div style={{ fontSize: 12, color: "var(--text-2)", marginTop: 2 }}>
        interval {fmt(interval[0])} – {fmt(interval[1])}
      </div>
      {reasonCodes(d).length > 0 && (
        <div style={{ marginTop: 10, textAlign: "left" }}>
          <div className="eyebrow">Key factors</div>
          <ReasonCodes codes={reasonCodes(d)} />
        </div>
      )}
    </div>
  );
}

/* memo()'d: the artifact reference stays stable while a message's prose keeps
 * streaming token-by-token, so this — including the RadialBarChart gauges in
 * ResidentFamilies — skips re-rendering on every token instead of only when
 * the artifact itself changes. */
export const ResidentsChatArtifact = memo(function ResidentsChatArtifact({
  artifact,
  onSelectProperty,
}: {
  artifact: Artifact;
  onSelectProperty?: (propertyId: string) => void;
}) {
  if (artifact == null) return null;

  // 1) Property-health ranking (array or wrapped).
  const health = asHealthList(artifact);
  if (health) {
    return (
      <div style={{ marginTop: 10 }}>
        <div className="eyebrow">Property health</div>
        <div style={{ marginTop: 6 }}>
          <HealthList items={health} onSelect={onSelectProperty} limit={10} />
        </div>
      </div>
    );
  }

  // 2) A full resident prediction bundle → compact family sections.
  const bundle = asBundle(artifact);
  if (bundle) {
    return (
      <div style={{ marginTop: 10 }}>
        {bundle.name && <div className="eyebrow">Predictions for {bundle.name}</div>}
        <div style={{ marginTop: 6 }}>
          <ResidentFamilies heads={bundle.heads} families={bundle.families} compact />
        </div>
      </div>
    );
  }

  // 3) A single head-like payload.
  const d = asDict(artifact);
  if (d) {
    if (typeof d.probability === "number" && typeof d.band === "string") {
      return <BinaryArtifact d={d} />;
    }
    if (typeof d.expected === "number") {
      return <ScalarArtifact d={d} />;
    }
  }

  // 4) Nothing renderable — prose alone carries the answer.
  return null;
});
