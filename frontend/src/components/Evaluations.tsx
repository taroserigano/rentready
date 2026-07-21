import { useEffect, useState, type ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  evalReportUrl,
  getConciergeEvalLatest,
  getEvalsLatest,
  getEvalSuites,
  getRiskEvalLatest,
  runConciergeEval,
  runEvals,
  runJudge,
  runLangsmith,
  runRagas,
  runRiskEval,
  type EvalSuite,
} from "../api";
import type { ConciergeEvalResult, RiskEvalResult } from "../types";
import { BAND_LABEL, BAND_TONE } from "./risk/riskTone";

interface MetricRow {
  name: string;
  short: string;
  value: number;
  threshold: number;
}

function toRows(results: Record<string, any>): MetricRow[] {
  if (!results || !results.eligibility) return [];
  return [
    {
      name: "Eligibility accuracy",
      short: "Eligibility",
      value: results.eligibility.accuracy,
      threshold: 1.0,
    },
    {
      name: "Extraction field accuracy",
      short: "Extraction",
      value: results.extraction.field_accuracy,
      threshold: 0.9,
    },
    {
      name: "Recommendation NDCG@5",
      short: "NDCG@5",
      value: results.recommendations.ndcg_at_5,
      threshold: 0.7,
    },
  ];
}

const AXIS_TICK = { fill: "var(--chart-axis)", fontSize: 11 };

const TOOLTIP_STYLE = {
  background: "var(--panel-2)",
  border: "1px solid var(--line-strong)",
  borderRadius: 8,
  fontSize: 12,
  color: "var(--text)",
} as const;

export function Evaluations() {
  const [results, setResults] = useState<Record<string, any>>({});
  const [suites, setSuites] = useState<EvalSuite[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getEvalSuites().then(setSuites).catch(() => {});
    getEvalsLatest().then(setResults).catch(() => {});
  }, []);

  async function run() {
    setBusy(true);
    try {
      setResults(await runEvals());
    } finally {
      setBusy(false);
    }
  }

  const rows = toRows(results);

  return (
    <div className="app">
      <header>
        <h1>Evaluations</h1>
        <p>
          Offline, deterministic quality gates — no LLM or DB needed. The same
          suites run in CI and fail the build on regression.
        </p>
      </header>

      <div className="card">
        <h2>Run the suite</h2>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button onClick={run} disabled={busy}>
            {busy ? "Running…" : "Run evaluations"}
          </button>
          <a
            className="btn btn-ghost"
            href={evalReportUrl}
            target="_blank"
            rel="noreferrer"
            style={{ textDecoration: "none" }}
          >
            View full report ↗
          </a>
        </div>
        {results.generated_at && (
          <p className="muted" style={{ marginTop: 10 }}>
            Last run: {new Date(results.generated_at).toLocaleString()}
          </p>
        )}
        {rows.length === 0 && (
          <p className="muted" style={{ marginTop: 10 }}>
            No results yet — click “Run evaluations”.
          </p>
        )}
      </div>

      {rows.length > 0 && (
        <>
          <div className="card">
            <h2>Headline metrics</h2>
            <div className="stat-grid">
              {rows.map((r) => {
                const pass = r.value >= r.threshold;
                return (
                  <div key={r.name} className="stat-tile">
                    <div className="label">{r.name}</div>
                    <div className={`value${pass ? "" : " bad"}`}>
                      {Math.round(r.value * 100)}%
                    </div>
                    <div className="sub">
                      threshold {Math.round(r.threshold * 100)}%{" "}
                      <span className={`badge ${pass ? "tone-good" : "tone-bad"}`}>
                        {pass ? "Pass" : "Fail"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card">
            <h2>Metric vs threshold</h2>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart
                data={rows.map((r) => ({
                  ...r,
                  value: Math.round(r.value * 100),
                  threshold: Math.round(r.threshold * 100),
                }))}
              >
                <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
                <XAxis
                  dataKey="short"
                  tick={AXIS_TICK}
                  axisLine={false}
                  tickLine={false}
                  interval={0}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={AXIS_TICK}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  cursor={{ fill: "rgba(255,255,255,0.04)" }}
                  contentStyle={TOOLTIP_STYLE}
                />
                <Bar
                  dataKey="value"
                  barSize={18}
                  radius={[4, 4, 0, 0]}
                  isAnimationActive={false}
                >
                  {rows.map((r, i) => (
                    <Cell
                      key={i}
                      fill={r.value >= r.threshold ? "var(--chart-2)" : "var(--chart-4)"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {results.eligibility?.confusion && (
            <div className="card">
              <h2>Eligibility confusion matrix</h2>
              <ConfusionMatrix confusion={results.eligibility.confusion} />
            </div>
          )}
        </>
      )}

      <LlmTier
        judge={results.judge}
        ragas={results.ragas}
        onResult={(patch) => setResults((r) => ({ ...r, ...patch }))}
      />

      <ConciergeEvals />

      <RiskEvals />

      <div className="card">
        <h2>Suites</h2>
        <ul className="reasons">
          {suites.map((s) => (
            <li key={s.id}>
              <b>{s.name}</b> — {s.description}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/** Quality band for a [0,1] metric → meter/value color. */
function band(v: number): "good" | "warn" | "bad" {
  if (v >= 0.8) return "good";
  if (v >= 0.5) return "warn";
  return "bad";
}

function Check({ ok }: { ok: boolean }) {
  return (
    <span
      className={`badge ${ok ? "tone-good" : "tone-bad"}`}
      title={ok ? "pass" : "fail"}
    >
      {ok ? "✓" : "✗"}
    </span>
  );
}

function ConciergeEvals() {
  const [result, setResult] = useState<ConciergeEvalResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  // Show the last persisted run immediately so the card isn't empty on arrival.
  useEffect(() => {
    getConciergeEvalLatest()
      .then((r) => r && setResult((cur) => cur ?? r))
      .catch(() => {});
  }, []);

  async function run() {
    setBusy(true);
    setError("");
    try {
      setResult(await runConciergeEval());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const metrics = result
    ? [
        {
          name: "Route accuracy",
          short: "Route",
          value: result.route_accuracy,
        },
        {
          name: "Retrieval hit-rate",
          short: "Retrieval",
          value: result.retrieval_hit_rate,
        },
        {
          name: "Groundedness",
          short: "Grounded",
          value: result.groundedness,
        },
      ]
    : [];

  return (
    <div className="card">
      <h2>Concierge</h2>
      <p className="muted" style={{ marginTop: 0 }}>
        Grades the property &amp; lease concierge on a labeled dataset —
        routing, retrieval, and groundedness. Deterministic and reproducible
        (no LLM needed).
      </p>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button onClick={run} disabled={busy}>
          {busy ? "Running…" : "Run Concierge eval"}
        </button>
      </div>
      {error && (
        <div className="error" style={{ marginTop: 10 }}>
          {error}
        </div>
      )}
      {result?.generated_at && (
        <p className="muted" style={{ marginTop: 10 }}>
          Last run: {new Date(result.generated_at).toLocaleString()} · {result.n}{" "}
          items
        </p>
      )}
      {!result && !error && (
        <p className="muted" style={{ marginTop: 10 }}>
          No results yet — click “Run Concierge eval”.
        </p>
      )}

      {result && (
        <>
          <div className="stat-grid" style={{ marginTop: 14 }}>
            {metrics.map((m) => {
              const tone = band(m.value);
              return (
                <div key={m.name} className="stat-tile">
                  <div className="label">{m.name}</div>
                  <div className={`value ${tone}`}>
                    {Math.round(m.value * 100)}%
                  </div>
                  <div className="meter" style={{ marginTop: 8 }}>
                    <i
                      className={tone}
                      style={{ width: `${Math.round(m.value * 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {result.items.length > 0 && (
            <div className="table-scroll" style={{ marginTop: 16 }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Question</th>
                    <th>Expected</th>
                    <th>Predicted</th>
                    <th style={{ textAlign: "center" }}>Route</th>
                    <th style={{ textAlign: "center" }}>Retrieval</th>
                    <th style={{ textAlign: "center" }}>Grounded</th>
                  </tr>
                </thead>
                <tbody>
                  {result.items.map((it) => (
                    <tr key={it.id}>
                      <td>{it.question}</td>
                      <td className="secondary">{it.expected_route}</td>
                      <td
                        className={
                          it.route_ok ? "secondary" : undefined
                        }
                        style={it.route_ok ? undefined : { color: "var(--bad-text)" }}
                      >
                        {it.predicted_route}
                      </td>
                      <td style={{ textAlign: "center" }}>
                        <Check ok={it.route_ok} />
                      </td>
                      <td style={{ textAlign: "center" }}>
                        <Check ok={it.retrieval_ok} />
                      </td>
                      <td style={{ textAlign: "center" }}>
                        <Check ok={it.grounded_ok} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** Tone for a lower-is-better metric (Brier, ECE): smaller = greener. */
function bandLow(v: number, good: number, warn: number): "good" | "warn" | "bad" {
  if (v <= good) return "good";
  if (v <= warn) return "warn";
  return "bad";
}

function RiskEvals() {
  const [result, setResult] = useState<RiskEvalResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getRiskEvalLatest()
      .then((r) => r && setResult((cur) => cur ?? r))
      .catch(() => {});
  }, []);

  async function run() {
    setBusy(true);
    setError("");
    try {
      setResult(await runRiskEval());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  // Discrimination metrics are higher-better; Brier/ECE are lower-better, so
  // both the tone and the "quality" meter fill are inverted for those two.
  const metrics = result
    ? [
        {
          name: "ROC-AUC",
          display: result.auc.toFixed(3),
          quality: result.auc,
          tone: band(result.auc),
        },
        {
          name: "PR-AUC",
          display: result.pr_auc.toFixed(3),
          quality: result.pr_auc,
          tone: band(result.pr_auc),
        },
        {
          name: "Brier ↓",
          display: result.brier.toFixed(3),
          quality: 1 - Math.min(result.brier, 1),
          tone: bandLow(result.brier, 0.15, 0.22),
        },
        {
          name: "ECE ↓",
          display: result.ece.toFixed(3),
          quality: 1 - Math.min(result.ece, 1),
          tone: bandLow(result.ece, 0.05, 0.1),
        },
      ]
    : [];

  const calibData = result
    ? result.calibration.map((b) => ({
        predicted: Math.round(b.predicted * 100),
        observed: Math.round(b.observed * 100),
        ideal: Math.round(b.predicted * 100),
      }))
    : [];

  const confusionForMatrix = result
    ? {
        Late: {
          Late: result.confusion.actual_late.pred_late,
          "On-time": result.confusion.actual_late.pred_ontime,
        },
        "On-time": {
          Late: result.confusion.actual_ontime.pred_late,
          "On-time": result.confusion.actual_ontime.pred_ontime,
        },
      }
    : null;

  return (
    <div className="card">
      <h2>Late-payment risk</h2>
      <p className="muted" style={{ marginTop: 0 }}>
        Grades the risk model on a held-out synthetic set — discrimination
        (AUC / PR-AUC), calibration (Brier / ECE / reliability), a confusion
        matrix at the {result ? Math.round(result.threshold * 100) : 40}% review
        threshold, and non-protected fairness slices. Deterministic; no LLM needed.
      </p>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button onClick={run} disabled={busy}>
          {busy ? "Running…" : "Run risk eval"}
        </button>
      </div>
      {error && (
        <div className="error" style={{ marginTop: 10 }}>
          {error}
        </div>
      )}
      {result?.generated_at && (
        <p className="muted" style={{ marginTop: 10 }}>
          Last run: {new Date(result.generated_at).toLocaleString()} · {result.n}{" "}
          items · base rate {Math.round(result.base_rate * 100)}% · {result.source}
        </p>
      )}
      {!result && !error && (
        <p className="muted" style={{ marginTop: 10 }}>
          No results yet — click “Run risk eval”.
        </p>
      )}

      {result && (
        <>
          <div className="stat-grid" style={{ marginTop: 14 }}>
            {metrics.map((m) => (
              <div key={m.name} className="stat-tile">
                <div className="label">{m.name}</div>
                <div className={`value ${m.tone}`}>{m.display}</div>
                <div className="meter" style={{ marginTop: 8 }}>
                  <i
                    className={m.tone}
                    style={{ width: `${Math.round(Math.max(0, Math.min(1, m.quality)) * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 18,
              marginTop: 16,
            }}
          >
            <div className="subpanel">
              <div className="subpanel-title">Reliability (calibration)</div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart
                  data={calibData}
                  margin={{ top: 8, right: 12, bottom: 4, left: -12 }}
                >
                  <CartesianGrid stroke="var(--chart-grid)" />
                  <XAxis
                    dataKey="predicted"
                    type="number"
                    domain={[0, 100]}
                    tick={AXIS_TICK}
                    axisLine={false}
                    tickLine={false}
                    unit="%"
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={AXIS_TICK}
                    axisLine={false}
                    tickLine={false}
                    unit="%"
                  />
                  <Tooltip
                    cursor={{ stroke: "var(--line-strong)" }}
                    contentStyle={TOOLTIP_STYLE}
                  />
                  <Line
                    dataKey="ideal"
                    name="Ideal"
                    stroke="var(--chart-axis)"
                    strokeDasharray="4 4"
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    dataKey="observed"
                    name="Observed"
                    stroke="var(--chart-1)"
                    strokeWidth={2}
                    dot={{ r: 2 }}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
              <p className="muted" style={{ fontSize: 11, marginTop: 4 }}>
                Observed late-rate vs predicted probability per bin — closer to
                the dashed line is better calibrated.
              </p>
            </div>

            <div className="subpanel">
              <div className="subpanel-title">
                Confusion @ {Math.round(result.threshold * 100)}%
              </div>
              {confusionForMatrix && <ConfusionMatrix confusion={confusionForMatrix} />}
              <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginTop: 10 }}>
                <Mini label="Precision" value={pct(result.confusion_stats.precision)} />
                <Mini label="Recall" value={pct(result.confusion_stats.recall)} />
                <Mini label="F1" value={pct(result.confusion_stats.f1)} />
                <Mini label="Accuracy" value={pct(result.confusion_stats.accuracy)} />
              </div>
            </div>
          </div>

          {result.slices.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <h3 className="note-title">Fairness slices (non-protected)</h3>
              <div className="table-scroll">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Slice</th>
                      <th style={{ textAlign: "right" }}>n</th>
                      <th style={{ textAlign: "right" }}>Late rate</th>
                      <th style={{ textAlign: "right" }}>AUC</th>
                      <th style={{ textAlign: "right" }}>Brier</th>
                      <th style={{ textAlign: "right" }}>ECE</th>
                      <th style={{ textAlign: "center" }}>Flag</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.slices.map((s) => (
                      <tr key={s.name}>
                        <td>{s.name}</td>
                        <td style={{ textAlign: "right" }}>{s.n}</td>
                        <td style={{ textAlign: "right" }}>{pct(s.positive_rate)}</td>
                        <td style={{ textAlign: "right" }}>{s.auc.toFixed(3)}</td>
                        <td style={{ textAlign: "right" }}>{s.brier.toFixed(3)}</td>
                        <td style={{ textAlign: "right" }}>{s.ece.toFixed(3)}</td>
                        <td style={{ textAlign: "center" }}>
                          {s.flag ? (
                            <span className="badge tone-warn">review</span>
                          ) : (
                            <span className="badge tone-good">ok</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {result.items.length > 0 && (
            <div className="table-scroll" style={{ marginTop: 16 }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th style={{ textAlign: "right" }}>P(late)</th>
                    <th>Band</th>
                    <th>Confidence</th>
                    <th>Actual</th>
                    <th style={{ textAlign: "center" }}>Correct</th>
                    <th>Top reasons</th>
                  </tr>
                </thead>
                <tbody>
                  {result.items.map((it) => (
                    <tr key={it.id}>
                      <td className="secondary">{it.id}</td>
                      <td style={{ textAlign: "right" }}>{Math.round(it.p * 100)}%</td>
                      <td>
                        <span className={`badge tone-${BAND_TONE[it.band]}`}>
                          {BAND_LABEL[it.band]}
                        </span>
                      </td>
                      <td className="secondary">{it.confidence}</td>
                      <td className="secondary">{it.actual ? "Late" : "On-time"}</td>
                      <td style={{ textAlign: "center" }}>
                        <Check ok={it.correct} />
                      </td>
                      <td className="secondary">{it.top_reasons.join("; ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function LlmTier({
  judge,
  ragas,
  onResult,
}: {
  judge?: Record<string, any>;
  ragas?: Record<string, any>;
  onResult: (patch: Record<string, any>) => void;
}) {
  const [busy, setBusy] = useState<string>("");
  const [ls, setLs] = useState<Record<string, any> | null>(null);

  async function run(kind: "judge" | "ragas" | "langsmith") {
    setBusy(kind);
    try {
      if (kind === "judge") onResult({ judge: await runJudge() });
      else if (kind === "ragas") onResult({ ragas: await runRagas() });
      else setLs(await runLangsmith());
    } finally {
      setBusy("");
    }
  }

  const rec = judge?.recommendation;
  const ragasMetrics = ragas?.metrics;

  return (
    <div className="card">
      <h2>LLM tier — judge, RAGAS &amp; LangSmith</h2>
      <p className="muted" style={{ marginTop: 0 }}>
        These call Claude, so they run on demand (and skip cleanly without a
        key). The judge has guardrails: temperature 0, JSON-only output, and it
        only grades free text — never the verdict or ranking.
      </p>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
        <button className="btn-ghost" onClick={() => run("judge")} disabled={!!busy}>
          {busy === "judge" ? "Judging…" : "Run LLM judge"}
        </button>
        <button className="btn-ghost" onClick={() => run("ragas")} disabled={!!busy}>
          {busy === "ragas" ? "Scoring…" : "Run RAGAS"}
        </button>
        <button className="btn-ghost" onClick={() => run("langsmith")} disabled={!!busy}>
          {busy === "langsmith" ? "Uploading…" : "Push LangSmith experiment"}
        </button>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 12,
        }}
      >
        <Panel title="LLM-as-judge (groundedness)">
          {judge?.skipped ? (
            <Skipped reason={judge.reason} />
          ) : rec ? (
            <>
              <Big value={`${Math.round((rec.mean_groundedness_pct ?? 0) * 100)}%`} />
              <Mini label="Mean score" value={`${rec.mean_groundedness ?? "—"} / 5`} />
              <Mini label="Judge-flagged claims" value={rec.judge_violations} />
              <Mini
                label="Deterministic violations"
                value={rec.deterministic_violations}
              />
              <Mini
                label="Eligibility consistency"
                value={`${Math.round(
                  (judge?.eligibility?.consistency_rate ?? 0) * 100,
                )}%`}
              />
            </>
          ) : (
            <Empty hint="Run the LLM judge." />
          )}
        </Panel>

        <Panel title="RAGAS (RAG quality)">
          {ragas?.skipped ? (
            <Skipped reason={ragas.reason} />
          ) : ragasMetrics ? (
            <>
              <Mini
                label="Faithfulness"
                value={pct(ragasMetrics.faithfulness)}
              />
              <Mini
                label="Answer relevancy"
                value={pct(ragasMetrics.answer_relevancy)}
              />
              <Mini
                label="Answer correctness"
                value={pct(ragasMetrics.answer_correctness)}
              />
            </>
          ) : (
            <Empty hint="Run RAGAS." />
          )}
        </Panel>

        <Panel title="LangSmith experiment">
          {ls?.skipped ? (
            <Skipped reason={ls.reason} />
          ) : ls ? (
            <>
              <Mini label="Dataset" value={ls.dataset} />
              <Mini label="Experiment" value={ls.experiment_name} />
              <p className="muted" style={{ fontSize: 11 }}>
                Open LangSmith to compare runs over time.
              </p>
            </>
          ) : (
            <Empty hint="Push an experiment to LangSmith." />
          )}
        </Panel>
      </div>

      {rec?.per_case?.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <h3 className="note-title">Judge notes</h3>
          <ul className="reasons">
            {rec.per_case.map((c: any, i: number) => (
              <li key={i}>
                <b>{c.property}</b> — {c.score}/5. {c.reason}
                {c.violations?.length > 0 && (
                  <span className="warn-text"> ⚠ {c.violations.join("; ")}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function pct(v: number | null | undefined) {
  return v == null ? "—" : `${Math.round(v * 100)}%`;
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="subpanel">
      <div className="subpanel-title">{title}</div>
      {children}
    </div>
  );
}

function Big({ value }: { value: string }) {
  return <div className="big-num">{value}</div>;
}

function Mini({ label, value }: { label: string; value: any }) {
  return (
    <div className="mini-row">
      <span className="k">{label}</span>
      <span className="v">{value ?? "—"}</span>
    </div>
  );
}

function Skipped({ reason }: { reason?: string }) {
  return (
    <div style={{ fontSize: 12, color: "var(--muted)" }}>
      <span className="badge tone-info">skipped</span>
      <p style={{ marginTop: 6 }}>{reason}</p>
    </div>
  );
}

function Empty({ hint }: { hint: string }) {
  return <p className="muted" style={{ fontSize: 12 }}>{hint}</p>;
}

function ConfusionMatrix({
  confusion,
}: {
  confusion: Record<string, Record<string, number>>;
}) {
  const labels = Object.keys(confusion);
  return (
    <table style={{ borderCollapse: "separate", borderSpacing: 2, fontSize: 12 }}>
      <thead>
        <tr>
          <th style={{ padding: 6, color: "var(--muted)", fontWeight: 500 }}>
            actual ↓ / pred →
          </th>
          {labels.map((l) => (
            <th key={l} style={{ padding: 6, color: "var(--muted)", fontWeight: 500 }}>
              {l}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {labels.map((a) => (
          <tr key={a}>
            <td style={{ padding: 6, color: "var(--muted)" }}>{a}</td>
            {labels.map((p) => {
              const v = confusion[a][p];
              const diag = a === p;
              return (
                <td
                  key={p}
                  style={{
                    padding: 6,
                    textAlign: "center",
                    fontVariantNumeric: "tabular-nums",
                    background: v
                      ? diag
                        ? "var(--good-bg)"
                        : "var(--bad-bg)"
                      : "transparent",
                    color: v
                      ? diag
                        ? "var(--good-text)"
                        : "var(--bad-text)"
                      : "var(--text-2)",
                    border: v
                      ? `1px solid ${diag ? "var(--good-line)" : "var(--bad-line)"}`
                      : "1px solid transparent",
                    borderRadius: 6,
                  }}
                >
                  {v}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
