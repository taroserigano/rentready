import { memo } from "react";
import type {
  RiskArtifact,
  RiskBand,
  RiskComparisonRow,
  RiskResult,
  RiskWhatIfChange,
} from "../../types";
import { RiskGauge } from "./RiskGauge";
import { ReasonCodes } from "./ReasonCodes";
import { BAND_LABEL, BAND_METER, BAND_TONE, formatProbRange } from "./riskTone";

/**
 * True when the artifact carries a concrete probability/estimate — used by the
 * message turn to decide whether to stamp the inline RISK_DISCLAIMER. Every
 * numeric artifact (score / whatif / counterfactual / comparison) qualifies;
 * the counterfactual is inherently about the estimate even before a re-scored
 * `result` is attached, so it always counts.
 */
export function artifactHasNumber(artifact: RiskArtifact): boolean {
  switch (artifact.kind) {
    case "score":
    case "whatif":
    case "counterfactual":
    case "comparison":
      return true;
    default:
      return false;
  }
}

// --- shared formatting -------------------------------------------------------

function money(n: number): string {
  return `$${Math.round(n).toLocaleString()}`;
}

/**
 * Render one lever value nicely from just {feature, from|to}. Money fields
 * become currency, ratios become percents, known flags become Yes/No, and
 * everything else is a tidy localized number. Pure — no colour, no fetch.
 */
function formatLeverValue(feature: string, n: number): string {
  const f = feature.toLowerCase();
  if (/(_to_income|^dti$|ratio|_pct$|percent)/.test(f)) {
    return `${Math.round(n * 100)}%`;
  }
  if (/(income|rent|debt|savings|balance|salary|deposit|arrears)/.test(f)) {
    return money(n);
  }
  if (/^(has_|needs_|is_)|guarantor|landlord_reference|verified|available|smoker|furnished|criminal|balcony|parking|laundry|student|pets\b/.test(f)) {
    return n ? "Yes" : "No";
  }
  return Number.isInteger(n) ? n.toLocaleString() : n.toFixed(2);
}

/** "$4,200 → $6,000" style transition for one change. */
function describeChange(c: RiskWhatIfChange): string {
  return `${formatLeverValue(c.feature, c.from)} → ${formatLeverValue(c.feature, c.to)}`;
}

/** Shared changed-levers list (mini-rows: label ⇢ from → to). */
function ChangeList({
  changes,
  title = "Changes",
}: {
  changes: RiskWhatIfChange[];
  title?: string;
}) {
  if (!changes || changes.length === 0) return null;
  return (
    <div style={{ marginTop: 10 }}>
      <div className="eyebrow">{title}</div>
      <div style={{ marginTop: 6 }}>
        {changes.map((c) => (
          <div className="mini-row" key={c.feature}>
            <span className="k">{c.label}</span>
            <span className="v">{describeChange(c)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Percentage points a probability sits above `baseline`, rounded like the gauge. */
function pointDelta(probability: number, baseline: number): number {
  return Math.round(probability * 100) - Math.round(baseline * 100);
}

/**
 * The exploratory ".delta" pill (mirrors RiskWhatIf): direction arrow + points
 * moved from the saved baseline. Coloured via the shared .delta up/down/zero
 * classes so it matches the slider what-if exactly.
 */
function DeltaPill({ probability, baseline }: { probability: number; baseline: number }) {
  const pts = pointDelta(probability, baseline);
  const cls = pts > 0 ? "up" : pts < 0 ? "down" : "zero";
  const arrow = pts > 0 ? "↑ " : pts < 0 ? "↓ " : "→ ";
  return (
    <span className={`delta ${cls}`}>
      {arrow}
      {Math.abs(pts)} pts from {Math.round(baseline * 100)}%
      {pts === 0 && " (no change)"}
    </span>
  );
}

/** The always-on exploratory flag shared by what-if / counterfactual views. */
function ExploratoryBadge() {
  return <span className="badge tone-warn">Exploratory — not the saved score</span>;
}

// --- counterfactual ----------------------------------------------------------

function RiskCounterfactual({
  targetBand,
  achievable,
  changes,
  result,
}: {
  targetBand: RiskBand;
  achievable: boolean;
  changes: RiskWhatIfChange[];
  result?: RiskResult;
}) {
  return (
    <div className="subpanel" style={{ marginTop: 10 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        <span className="eyebrow" style={{ marginRight: 2 }}>
          Path to
        </span>
        <span className={`badge tone-${BAND_TONE[targetBand]}`}>
          {BAND_LABEL[targetBand]}
        </span>
        <span className={`badge tone-${achievable ? "good" : "warn"}`}>
          {achievable ? "Achievable" : "Not reachable within realistic limits"}
        </span>
        <ExploratoryBadge />
      </div>

      {achievable ? (
        <ChangeList changes={changes} title="Suggested steps" />
      ) : (
        <p className="muted" style={{ margin: "10px 0 0", fontSize: 12 }}>
          No realistic combination of allowed factors reaches the{" "}
          {BAND_LABEL[targetBand]} band. Only legitimate financial and payment
          factors are ever adjusted.
        </p>
      )}

      {result && (
        <div style={{ marginTop: 12 }}>
          <RiskGauge probability={result.probability} band={result.band} />
          <div
            style={{
              textAlign: "center",
              fontSize: 12,
              color: "var(--text-2)",
              marginTop: 4,
            }}
          >
            Resulting estimate · {formatProbRange(result.probability, result.range)}
          </div>
        </div>
      )}
    </div>
  );
}

// --- comparison --------------------------------------------------------------

function ComparisonBar({
  row,
  clickable,
  onSelect,
}: {
  row: RiskComparisonRow;
  clickable: boolean;
  onSelect?: (id: string) => void;
}) {
  const pct = Math.max(0, Math.min(100, Math.round(row.probability * 100)));
  const meterTone = row.band ? BAND_METER[row.band] : undefined;
  return (
    <div className="fh-row" style={{ margin: 0 }}>
      <div className="fh-head" style={{ marginBottom: 3 }}>
        {clickable && row.applicant_id && onSelect ? (
          <button
            type="button"
            className="linklike"
            onClick={() => onSelect(row.applicant_id!)}
            title="Open this applicant"
          >
            {row.label}
          </button>
        ) : (
          <span>{row.label}</span>
        )}
        <span className="fh-val" style={{ fontVariantNumeric: "tabular-nums" }}>
          {pct}%{row.band ? ` · ${BAND_LABEL[row.band]}` : ""}
        </span>
      </div>
      <div className="meter">
        <i className={meterTone} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function RiskComparison({
  subject,
  rows,
  percentile,
  onSelectApplicant,
}: {
  subject: RiskComparisonRow;
  rows: RiskComparisonRow[];
  percentile?: number;
  onSelectApplicant?: (id: string) => void;
}) {
  // percentile may arrive as a fraction (0..1) or whole percent (0..100).
  const pctl =
    percentile == null
      ? null
      : percentile <= 1
        ? Math.round(percentile * 100)
        : Math.round(percentile);

  return (
    <div className="subpanel" style={{ marginTop: 10 }}>
      <div className="eyebrow">Compared to the Portfolio</div>
      <div style={{ display: "grid", gap: 10, marginTop: 8 }}>
        <ComparisonBar row={subject} clickable={false} />
        {rows.map((r, i) => (
          <ComparisonBar
            key={r.applicant_id ?? `${r.label}-${i}`}
            row={r}
            clickable={
              !!r.applicant_id && r.applicant_id !== subject.applicant_id
            }
            onSelect={onSelectApplicant}
          />
        ))}
      </div>
      {pctl != null && (
        <p className="muted" style={{ margin: "10px 0 0", fontSize: 12 }}>
          Higher than {pctl}% of scored applicants.
        </p>
      )}
    </div>
  );
}

/**
 * Renders the structured payload for one risk-chat turn. A pure switch on
 * `artifact.kind` that reuses the existing risk primitives (RiskGauge,
 * ReasonCodes, .meter, .delta) so numbers render identically to the Risk page.
 * All framing (bands, tones, colours) comes from riskTone.ts; nothing fetches.
 */
/* memo()'d: the artifact reference stays stable while a message's prose keeps
 * streaming token-by-token, so this skips re-rendering the gauge/comparison
 * table on every token instead of only when the artifact itself changes. */
export const RiskChatArtifact = memo(function RiskChatArtifact({
  artifact,
  onSelectApplicant,
}: {
  artifact: RiskArtifact;
  onSelectApplicant?: (id: string) => void;
}) {
  switch (artifact.kind) {
    case "none":
      return null;

    case "reasons":
      return (
        <div style={{ marginTop: 10 }}>
          <div className="eyebrow">Key Factors</div>
          <ReasonCodes codes={artifact.codes} />
        </div>
      );

    case "score": {
      const { result } = artifact;
      return (
        <div style={{ marginTop: 10 }}>
          <RiskGauge probability={result.probability} band={result.band} />
          <div
            style={{
              textAlign: "center",
              fontSize: 12,
              color: "var(--text-2)",
              marginTop: 4,
            }}
          >
            {formatProbRange(result.probability, result.range)}
          </div>
          <div style={{ marginTop: 10 }}>
            <div className="eyebrow">Key Factors</div>
            <ReasonCodes codes={result.reason_codes} />
          </div>
        </div>
      );
    }

    case "whatif": {
      const { result, baseline, changes } = artifact;
      return (
        <div style={{ marginTop: 10 }}>
          <RiskGauge probability={result.probability} band={result.band} />
          <div
            style={{
              textAlign: "center",
              fontSize: 12,
              color: "var(--text-2)",
              marginTop: 4,
            }}
          >
            {formatProbRange(result.probability, result.range)}
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              flexWrap: "wrap",
              justifyContent: "center",
              marginTop: 10,
            }}
          >
            <DeltaPill probability={result.probability} baseline={baseline} />
            <ExploratoryBadge />
          </div>
          <ChangeList changes={changes} />
          <div style={{ marginTop: 10 }}>
            <div className="eyebrow">Key Factors</div>
            <ReasonCodes codes={result.reason_codes} />
          </div>
        </div>
      );
    }

    case "counterfactual":
      return (
        <RiskCounterfactual
          targetBand={artifact.target_band}
          achievable={artifact.achievable}
          changes={artifact.changes}
          result={artifact.result}
        />
      );

    case "comparison":
      return (
        <RiskComparison
          subject={artifact.subject}
          rows={artifact.rows}
          percentile={artifact.percentile}
          onSelectApplicant={onSelectApplicant}
        />
      );

    default: {
      // Exhaustiveness guard — Phase 2/3 additions must handle their branch.
      const _never: never = artifact;
      return _never;
    }
  }
});
