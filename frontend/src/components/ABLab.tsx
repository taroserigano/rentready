import { useEffect, useState } from "react";
import { getAbVariants, runAb, type AbVariant } from "../api";

export function ABLab() {
  const [variants, setVariants] = useState<AbVariant[]>([]);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getAbVariants()
      .then((vs) => {
        setVariants(vs);
        if (vs[0]) setA(vs[0].id);
        if (vs[1]) setB(vs[1].id);
      })
      .catch(() => {});
  }, []);

  async function run() {
    setError("");
    setBusy(true);
    setResult(null);
    try {
      setResult(await runAb(a, b));
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const skipped = result?.skipped;

  return (
    <div className="app">
      <header>
        <h1>A/B Lab</h1>
        <p>
          The core eval workflow: hold the dataset and the judge fixed, change
          one thing (the prompt or the model), and let the metrics pick a
          winner. The deterministic scorer chooses the same properties for both
          variants — only the explanation differs.
        </p>
      </header>

      <div className="card">
        <h2>Pick two variants</h2>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "end" }}>
          <VariantPicker label="Variant A" value={a} onChange={setA} variants={variants} />
          <span style={{ fontWeight: 600, paddingBottom: 8, color: "var(--muted)" }}>
            vs
          </span>
          <VariantPicker label="Variant B" value={b} onChange={setB} variants={variants} />
          <button onClick={run} disabled={busy || a === b}>
            {busy ? "Running experiment…" : "Run A/B"}
          </button>
        </div>
        {a === b && (
          <p className="muted" style={{ marginTop: 10 }}>
            Pick two different variants to compare.
          </p>
        )}
        {error && <div className="error" role="alert">{error}</div>}
        {busy && (
          <p className="muted" style={{ marginTop: 10 }}>
            Generating explanations with both variants and grading each with the
            judge — this makes several model calls, give it a moment.
          </p>
        )}
      </div>

      {skipped && (
        <div className="card">
          <p className="muted">Skipped: {result?.reason}</p>
        </div>
      )}

      {result && !skipped && (
        <>
          <div className="card">
            <h2>
              Winner:{" "}
              <span style={{ color: "var(--accent-text)" }}>
                {result.winner === "tie"
                  ? "Tie"
                  : result.winner === result.a.id
                    ? result.a.label
                    : result.b.label}
              </span>
            </h2>
            <p className="muted">{result.rationale}</p>
            <p className="muted" style={{ fontSize: 12 }}>
              Dataset: {result.dataset_slugs?.join(", ")} · judge:{" "}
              {result.judge_model}
            </p>
          </div>

          <div className="card">
            <h2>Side by side</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <VariantCard v={result.a} winner={result.winner} />
              <VariantCard v={result.b} winner={result.winner} />
            </div>
            <DeltaRow deltas={result.deltas} />
          </div>

          <div className="card">
            <h2>Per-explanation judge notes</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <CaseList title={result.a.label} cases={result.a.per_case} />
              <CaseList title={result.b.label} cases={result.b.per_case} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function VariantPicker({
  label,
  value,
  onChange,
  variants,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  variants: AbVariant[];
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{ minWidth: 240 }}
      >
        {variants.map((v) => (
          <option key={v.id} value={v.id}>
            {v.label} · {v.model}
          </option>
        ))}
      </select>
    </label>
  );
}

function VariantCard({ v, winner }: { v: any; winner: string }) {
  const isWinner = winner === v.id;
  return (
    <div
      className="subpanel"
      style={isWinner ? { borderColor: "var(--accent)" } : undefined}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <b>{v.label}</b>
        {isWinner && <span className="badge tone-good">winner</span>}
      </div>
      <div
        style={{
          fontSize: 11,
          color: "var(--muted)",
          marginBottom: 8,
          fontFamily: "var(--font-mono)",
        }}
      >
        {v.model}
      </div>
      <Metric label="Groundedness" value={pct(v.mean_groundedness_pct)} big />
      <Metric label="Mean latency" value={`${v.mean_latency_s ?? "—"}s`} />
      <Metric label="Est. cost" value={`$${v.est_cost_usd}`} />
      <Metric label="Tokens (in/out)" value={`${v.input_tokens}/${v.output_tokens}`} />
    </div>
  );
}

function DeltaRow({ deltas }: { deltas: Record<string, number | null> }) {
  if (!deltas) return null;
  const fmt = (n: number | null, suffix = "") =>
    n == null ? "—" : `${n > 0 ? "+" : ""}${n}${suffix}`;
  return (
    <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
      B − A: groundedness {fmt(deltas.groundedness_pct && deltas.groundedness_pct * 100, "%")} · latency{" "}
      {fmt(deltas.latency_s, "s")} · cost {fmt(deltas.cost_usd, "$")}
    </p>
  );
}

function CaseList({ title, cases }: { title: string; cases: any[] }) {
  return (
    <div>
      <h3 className="note-title">{title}</h3>
      <ul className="reasons">
        {cases?.map((c, i) => (
          <li key={i}>
            <b>{c.property}</b> — {c.score}/5
            {c.violations?.length > 0 && (
              <span className="warn-text"> ⚠ {c.violations.join("; ")}</span>
            )}
            <div style={{ fontSize: 12, color: "var(--text-2)" }}>{c.reason}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Metric({ label, value, big }: { label: string; value: string; big?: boolean }) {
  return (
    <div className="mini-row">
      <span className="k">{label}</span>
      <span className="v" style={big ? { fontSize: 20, letterSpacing: "-0.01em" } : undefined}>
        {value}
      </span>
    </div>
  );
}

function pct(v: number | null | undefined) {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}
