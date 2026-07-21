"""Generate a standalone HTML evaluation report from a run's results.

Turns the combined results dict (see ``run_evals.run``) into a single,
self-contained HTML page — summary dashboard, an "Evaluation Breakdown by
Type" section, charts, and a filterable per-case table — styled after a
Lumina-style execution report. No external assets; works offline and prints
cleanly to PDF.

Usage:
    python -m evals.report                 # writes results/report.html
    from evals.report import build_html     # for the API
"""

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

THRESHOLDS = {
    "eligibility_accuracy": 1.0,
    "extraction_field_accuracy": 0.9,
    "ndcg_at_5": 0.7,
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _fmt_dt(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).strftime("%m/%d/%Y %I:%M%p UTC")
    except ValueError:
        return iso


def _esc(v) -> str:
    return html.escape(str(v))


def _card(value: str, label: str, mini: str, icon: str, tip: str) -> str:
    return f"""
    <div class="summary-card">
      <div class="icon {icon}">{_ICONS.get(icon, '')}</div>
      <div class="content">
        <div class="value">{value}</div>
        <div class="label">{label}
          <span class="info-tooltip">{_INFO_SVG}
            <span class="tooltip-text">{_esc(tip)}</span>
          </span>
        </div>
        <div class="mini-label">{mini}</div>
      </div>
    </div>"""


def _breakdown_card(name: str, value: float, threshold: Optional[float],
                    extra: str, tip: str) -> str:
    if threshold is None:
        passed = value is not None and value >= 0.8  # soft target for LLM tier
        gate = "soft target 80%"
    else:
        passed = value is not None and value >= threshold
        gate = f"threshold {_pct(threshold)}"
    cls = "pass" if passed else "fail"
    tick = ("<span class='tick-pass'>✓</span> PASS" if passed
            else "<span class='tick-fail'>✗</span> FAIL")
    return f"""
    <div class="eval-type-card {cls}">
      <h4>{_esc(name)}
        <span class="info-tooltip">{_INFO_SVG}
          <span class="tooltip-text">{_esc(tip)}</span>
        </span>
      </h4>
      <div class="eval-type-percentage">{_pct(value)}</div>
      <div class="eval-type-breakdown">{tick} · {gate}</div>
      <div class="eval-type-total">{extra}</div>
    </div>"""


def _bar(label: str, value: float, threshold: Optional[float]) -> str:
    pct = 0 if value is None else max(0, min(100, value * 100))
    passed = threshold is None or (value is not None and value >= threshold)
    color = "var(--pass-color)" if passed else "var(--fail-color)"
    tline = ""
    if threshold is not None:
        tline = (f"<div class='bar-threshold' style='left:{threshold * 100:.1f}%'"
                 f" title='threshold {_pct(threshold)}'></div>")
    return f"""
    <div class="bar-row">
      <div class="bar-label">{_esc(label)}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
        {tline}
      </div>
      <div class="bar-value">{_pct(value)}</div>
    </div>"""


def _row(suite: str, case: str, metric: str, score: str, ok: Optional[bool],
         detail: str) -> str:
    if ok is None:
        res = "<span class='pill neutral'>info</span>"
        rstate = "info"
    elif ok:
        res = "<span class='pill pass'>pass</span>"
        rstate = "pass"
    else:
        res = "<span class='pill fail'>fail</span>"
        rstate = "fail"
    return f"""
    <tr data-suite="{_esc(suite)}" data-result="{rstate}">
      <td><span class="suite-tag">{_esc(suite)}</span></td>
      <td class="mono">{_esc(case)}</td>
      <td>{_esc(metric)}</td>
      <td class="mono">{_esc(score)}</td>
      <td>{res}</td>
      <td class="detail">{_esc(detail)}</td>
    </tr>"""


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #
def build_html(results: dict) -> str:
    if not results or not results.get("eligibility"):
        return _empty_html()

    elig = results.get("eligibility", {})
    extr = results.get("extraction", {})
    recs = results.get("recommendations", {})
    judge = results.get("judge", {}) or {}
    ragas = results.get("ragas", {}) or {}
    rec_judge = judge.get("recommendation", {}) or {}
    elig_judge = judge.get("eligibility", {}) or {}
    ragas_metrics = ragas.get("metrics", {}) or {}

    elig_acc = elig.get("accuracy")
    extr_acc = extr.get("field_accuracy")
    ndcg = recs.get("ndcg_at_5")

    # deterministic gates
    gates = [
        (elig_acc, THRESHOLDS["eligibility_accuracy"]),
        (extr_acc, THRESHOLDS["extraction_field_accuracy"]),
        (ndcg, THRESHOLDS["ndcg_at_5"]),
    ]
    gates_passed = sum(1 for v, t in gates if v is not None and v >= t)
    all_pass = gates_passed == len(gates)

    total_cases = (
        elig.get("n", 0) + extr.get("n", 0) + recs.get("n", 0)
        + rec_judge.get("n", 0) + ragas.get("n", 0)
    )
    n_suites = 3 + (0 if judge.get("skipped") else 1) + (0 if ragas.get("skipped") else 1)

    # ---- summary cards ----
    cards = "".join([
        _card(
            f"{gates_passed}/{len(gates)}",
            "Deterministic Gates",
            "PASS" if all_pass else "FAIL — see breakdown",
            "evaluations",
            "How many CI-gating deterministic metrics meet their threshold. "
            "A failure here fails the build.",
        ),
        _card(
            str(total_cases), "Cases Evaluated",
            f"across {n_suites} suites", "traces",
            "Total individual test cases scored across every suite in this run.",
        ),
        _card(
            _pct(elig_acc), "Eligibility Accuracy",
            f"{elig.get('n', 0)} cases", "duration",
            "Share of applicants whose rule-based verdict matched ground truth.",
        ),
        _card(
            _pct(extr_acc), "Extraction Field Accuracy",
            f"exact-match {_pct(extr.get('exact_match_rate'))}", "threads",
            "Share of extracted profile fields that matched the labeled value.",
        ),
        _card(
            _pct(ndcg), "Recommendation NDCG@5",
            f"MRR {_pct(recs.get('mrr'))}", "latency",
            "Ranking quality of the deterministic scorer vs graded relevance.",
        ),
        _card(
            _pct(rec_judge.get("mean_groundedness_pct"))
            if not judge.get("skipped") else "skipped",
            "LLM Judge Groundedness",
            (f"{rec_judge.get('judge_violations', 0)} flagged claims"
             if not judge.get("skipped") else _esc(judge.get("reason", ""))),
            "evaluations",
            "Mean groundedness (1–5, shown as %) the LLM judge gave to "
            "recommendation explanations. Higher = better grounded in facts.",
        ),
        _card(
            _pct(ragas_metrics.get("faithfulness"))
            if not ragas.get("skipped") else "skipped",
            "RAGAS Faithfulness",
            (f"relevancy {_pct(ragas_metrics.get('answer_relevancy'))}"
             if not ragas.get("skipped") else _esc(ragas.get("reason", ""))),
            "latency",
            "Did the PDF Q&A answers stay faithful to the retrieved context?",
        ),
    ])

    # ---- breakdown-by-type cards ----
    breakdown = [
        _breakdown_card(
            "Eligibility Accuracy", elig_acc,
            THRESHOLDS["eligibility_accuracy"],
            f"{elig.get('n', 0)} cases evaluated",
            "Rule-based verdict exact-match vs ground truth.",
        ),
        _breakdown_card(
            "Extraction Field Accuracy", extr_acc,
            THRESHOLDS["extraction_field_accuracy"],
            f"{extr.get('n', 0)} applicants · "
            f"exact-match {_pct(extr.get('exact_match_rate'))}",
            "Per-field accuracy of the offline profile extractor.",
        ),
        _breakdown_card(
            "Recommendation NDCG@5", ndcg, THRESHOLDS["ndcg_at_5"],
            f"{recs.get('n', 0)} cases · precision@5 "
            f"{_pct(recs.get('precision_at_5'))} · MRR {_pct(recs.get('mrr'))}",
            "Ranking quality of the deterministic scorer.",
        ),
    ]
    if not judge.get("skipped"):
        breakdown.append(_breakdown_card(
            "LLM Judge Groundedness", rec_judge.get("mean_groundedness_pct"),
            None,
            f"{rec_judge.get('n', 0)} explanations · "
            f"{rec_judge.get('judge_violations', 0)} judge-flagged · "
            f"{rec_judge.get('deterministic_violations', 0)} tripwire",
            "LLM-as-judge groundedness of recommendation prose (guardrailed).",
        ))
        breakdown.append(_breakdown_card(
            "Eligibility Explanation Consistency",
            elig_judge.get("consistency_rate"), None,
            f"{elig_judge.get('n', 0)} explanations checked",
            "Does the explanation agree with the deterministic verdict?",
        ))
    if not ragas.get("skipped"):
        breakdown.append(_breakdown_card(
            "RAGAS Faithfulness", ragas_metrics.get("faithfulness"), None,
            f"relevancy {_pct(ragas_metrics.get('answer_relevancy'))} · "
            f"correctness {_pct(ragas_metrics.get('answer_correctness'))}",
            "RAG answer quality on the gold PDF Q&A set.",
        ))
    breakdown_html = "".join(breakdown)

    # ---- charts: metric vs threshold bars ----
    bars = "".join([
        _bar("Eligibility accuracy", elig_acc, THRESHOLDS["eligibility_accuracy"]),
        _bar("Extraction field accuracy", extr_acc,
             THRESHOLDS["extraction_field_accuracy"]),
        _bar("Recommendation NDCG@5", ndcg, THRESHOLDS["ndcg_at_5"]),
    ])
    if not judge.get("skipped"):
        bars += _bar("Judge groundedness",
                     rec_judge.get("mean_groundedness_pct"), None)
    if not ragas.get("skipped"):
        bars += _bar("RAGAS faithfulness", ragas_metrics.get("faithfulness"), None)
        bars += _bar("RAGAS answer relevancy",
                     ragas_metrics.get("answer_relevancy"), None)
        bars += _bar("RAGAS answer correctness",
                     ragas_metrics.get("answer_correctness"), None)

    # ---- confusion matrix ----
    confusion_html = _confusion(elig.get("confusion", {}))

    # ---- per-field extraction bars ----
    field_bars = "".join(
        _bar(name.replace("_", " "), val, None)
        for name, val in (extr.get("per_field") or {}).items()
    )

    # ---- per-case rows ----
    rows = []
    for c in elig.get("per_case", []):
        rows.append(_row(
            "eligibility", c.get("id", ""), "verdict match",
            f"{c.get('expected')} → {c.get('got')}", c.get("ok"),
            "Correct verdict" if c.get("ok") else "Verdict mismatch",
        ))
    for name, val in (extr.get("per_field") or {}).items():
        rows.append(_row(
            "extraction", name, "field accuracy", _pct(val),
            val is not None and val >= THRESHOLDS["extraction_field_accuracy"],
            "Field extracted correctly" if val and val >= 0.9
            else "Below 90% field accuracy",
        ))
    for c in recs.get("per_case", []):
        nd = c.get("ndcg")
        rows.append(_row(
            "recommendations", c.get("id", ""), "NDCG / precision@5",
            f"NDCG {_pct(nd)} · P@5 {_pct(c.get('precision'))}",
            nd is not None and nd >= THRESHOLDS["ndcg_at_5"],
            "Ranking meets target" if nd and nd >= 0.7 else "Ranking below target",
        ))
    for c in rec_judge.get("per_case", []):
        score = c.get("score")
        rows.append(_row(
            "judge", f"{c.get('slug')} · {c.get('property')}",
            "groundedness", f"{score}/5",
            score is not None and score >= 4,
            c.get("reason", ""),
        ))
    for c in elig_judge.get("per_case", []):
        rows.append(_row(
            "judge", f"{c.get('slug')} · {c.get('verdict')}",
            "explanation consistency",
            "consistent" if c.get("consistent") else "inconsistent",
            bool(c.get("consistent")),
            c.get("reason", ""),
        ))
    for c in ragas.get("per_case", []):
        rows.append(_row(
            "ragas", c.get("id", ""), "RAG Q&A", "—", None,
            c.get("question", ""),
        ))
    rows_html = "".join(rows)

    generated = _fmt_dt(results.get("generated_at"))
    overall_pct = _pct(gates_passed / len(gates)) if gates else "—"

    return _TEMPLATE.format(
        generated=generated,
        overall=overall_pct,
        status_badge=("PASS" if all_pass else "ATTENTION"),
        status_cls=("ok" if all_pass else "warn"),
        cards=cards,
        breakdown=breakdown_html,
        bars=bars,
        confusion=confusion_html,
        field_bars=field_bars,
        rows=rows_html,
        n_cases=total_cases,
    )


def _confusion(confusion: dict) -> str:
    if not confusion:
        return "<p class='muted'>No confusion matrix available.</p>"
    labels = list(confusion.keys())
    head = "".join(f"<th>{_esc(l)}</th>" for l in labels)
    body = ""
    for a in labels:
        cells = ""
        for p in labels:
            v = confusion.get(a, {}).get(p, 0)
            diag = a == p
            cls = ("cm-diag" if (v and diag) else
                   "cm-off" if v else "cm-zero")
            cells += f"<td class='{cls}'>{v}</td>"
        body += f"<tr><th class='cm-rowhead'>{_esc(a)}</th>{cells}</tr>"
    return f"""
    <table class="confusion">
      <thead><tr><th>actual ↓ / predicted →</th>{head}</tr></thead>
      <tbody>{body}</tbody>
    </table>"""


def _empty_html() -> str:
    return ("<!DOCTYPE html><html><body style='font-family:sans-serif;"
            "padding:2rem'><h2>No evaluation results yet</h2>"
            "<p>Run the suite first (Evaluations → Run evaluations), then "
            "regenerate this report.</p></body></html>")


# --------------------------------------------------------------------------- #
# assets
# --------------------------------------------------------------------------- #
_INFO_SVG = ('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
             'stroke="currentColor" stroke-width="1.5" stroke-linecap="round" '
             'stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle>'
             '<line x1="12" y1="16" x2="12" y2="12"></line>'
             '<line x1="12" y1="8" x2="12.01" y2="8"></line></svg>')

_ICONS = {
    "evaluations": ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none"'
                    ' stroke="currentColor" stroke-width="2" stroke-linecap="round"'
                    ' stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7'
                    'c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg>'),
    "traces": ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
               'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
               'stroke-linejoin="round"><path d="M14.5 2v17.5c0 1.4-1.1 2.5-2.5 '
               '2.5c-1.4 0-2.5-1.1-2.5-2.5V2"></path><line x1="8" y1="2" x2="16" '
               'y2="2"></line></svg>'),
    "duration": ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
                 'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
                 'stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle>'
                 '<polyline points="12 6 12 12 16 14"></polyline></svg>'),
    "threads": ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
                'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
                'stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10">'
                '</line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" '
                'y1="20" x2="6" y2="16"></line></svg>'),
    "latency": ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
                'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
                'stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle>'
                '<polyline points="12 6 12 12 16 14"></polyline></svg>'),
}

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RentReady — Evaluation Execution Report</title>
<style>
:root {{
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  --bg: #f8f9fa; --card-bg: #fff; --border: #e9ecef; --text: #495057;
  --title: #212529; --subtle: #6c757d; --pass-color: #28a745; --fail-color: #dc3545;
  --pass-bg: #e2f5e7; --fail-bg: #f9e2e4; --primary: #007bff; --primary-bg: #e0f0ff;
  --shadow: 0 4px 6px rgba(0,0,0,0.05); --radius: 12px;
}}
* {{ box-sizing: border-box; }}
body {{ font-family: var(--font); background: var(--bg); color: var(--text); margin: 0; padding: 1rem; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 0.5rem; }}
.report-header {{ background: var(--card-bg); border-radius: var(--radius); padding: 1rem 1.25rem; margin-bottom: 0.75rem; box-shadow: var(--shadow); border: 1px solid var(--border); }}
.header-title-row {{ display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }}
.report-header h1 {{ color: var(--title); font-size: 1.4rem; font-weight: 700; margin: 0; }}
.report-header h3 {{ color: var(--subtle); font-size: 0.85rem; font-weight: 500; margin: 0.35rem 0 0; }}
.report-header p {{ margin: 0.25rem 0 0; color: var(--subtle); font-size: 0.85rem; }}
.badge {{ padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }}
.badge.brand {{ background: var(--primary-bg); color: var(--primary); }}
.badge.ok {{ background: var(--pass-bg); color: var(--pass-color); }}
.badge.warn {{ background: #fff3cd; color: #856404; }}
.tabs {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }}
.tab-buttons {{ display: flex; gap: 0.5rem; }}
.tab-button {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 0.5rem 1rem; font-size: 0.85rem; font-weight: 600; color: var(--subtle); cursor: pointer; }}
.tab-button.active {{ background: var(--primary); color: #fff; border-color: var(--primary); }}
.download-button {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 0.5rem 1rem; font-size: 0.85rem; font-weight: 600; color: var(--text); cursor: pointer; }}
.report-tab {{ display: none; }}
.report-tab.active {{ display: block; }}
.summary-dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.5rem; margin-bottom: 0.75rem; }}
.summary-card {{ background: var(--card-bg); border-radius: var(--radius); padding: 0.75rem; display: flex; align-items: center; gap: 0.6rem; border: 1px solid var(--border); box-shadow: var(--shadow); position: relative; }}
.summary-card .icon {{ width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; color: #fff; }}
.icon.evaluations {{ background: #28a745; }}
.icon.traces {{ background: #007bff; }}
.icon.duration {{ background: linear-gradient(135deg,#42a5f5,#1e88e5); }}
.icon.threads {{ background: #ffc107; }}
.icon.latency {{ background: linear-gradient(135deg,#66bb6a,#43a047); }}
.summary-card .value {{ font-size: 1.3rem; font-weight: 700; color: var(--title); line-height: 1; }}
.summary-card .label {{ font-size: 0.75rem; color: var(--subtle); margin-top: 0.15rem; }}
.summary-card .mini-label {{ font-size: 0.68rem; color: var(--subtle); margin-top: 0.1rem; }}
.info-tooltip {{ position: relative; display: inline-flex; align-items: center; margin-left: 0.2rem; cursor: help; opacity: 0.55; }}
.info-tooltip:hover {{ opacity: 1; }}
.info-tooltip .tooltip-text {{ visibility: hidden; width: 240px; background: #333; color: #fff; text-align: left; border-radius: 6px; padding: 0.6rem; position: absolute; z-index: 99; bottom: 145%; left: 50%; margin-left: -120px; opacity: 0; transition: opacity 0.2s; font-size: 0.72rem; line-height: 1.4; box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-weight: 400; }}
.info-tooltip:hover .tooltip-text {{ visibility: visible; opacity: 1; }}
.section {{ background: var(--card-bg); border-radius: var(--radius); padding: 1rem 1.25rem; margin-bottom: 0.75rem; border: 1px solid var(--border); box-shadow: var(--shadow); }}
.section h3 {{ color: var(--title); font-size: 1rem; margin: 0 0 0.75rem; display: flex; align-items: center; gap: 0.4rem; }}
.eval-types {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.5rem; }}
.eval-type-card {{ border: 1px solid var(--border); border-radius: 10px; padding: 0.75rem; background: var(--bg); border-left: 4px solid var(--subtle); }}
.eval-type-card.pass {{ border-left-color: var(--pass-color); }}
.eval-type-card.fail {{ border-left-color: var(--fail-color); }}
.eval-type-card h4 {{ margin: 0 0 0.4rem; font-size: 0.85rem; color: var(--title); display: flex; align-items: center; }}
.eval-type-percentage {{ font-size: 1.6rem; font-weight: 700; color: var(--title); }}
.eval-type-breakdown {{ font-size: 0.75rem; margin: 0.2rem 0; }}
.eval-type-total {{ font-size: 0.7rem; color: var(--subtle); }}
.tick-pass {{ color: var(--pass-color); font-weight: 700; }}
.tick-fail {{ color: var(--fail-color); font-weight: 700; }}
.bar-row {{ display: grid; grid-template-columns: 200px 1fr 60px; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }}
.bar-label {{ font-size: 0.78rem; color: var(--text); }}
.bar-track {{ position: relative; background: #eef1f4; border-radius: 999px; height: 14px; }}
.bar-fill {{ height: 100%; border-radius: 999px; }}
.bar-threshold {{ position: absolute; top: -3px; width: 2px; height: 20px; background: #495057; }}
.bar-value {{ font-size: 0.78rem; font-weight: 600; text-align: right; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
@media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
.confusion {{ border-collapse: collapse; font-size: 0.8rem; }}
.confusion th, .confusion td {{ padding: 0.5rem 0.7rem; text-align: center; border: 1px solid var(--border); }}
.confusion th {{ background: var(--bg); color: var(--subtle); font-weight: 600; }}
.cm-rowhead {{ background: var(--bg); }}
.cm-diag {{ background: var(--pass-bg); color: var(--pass-color); font-weight: 700; }}
.cm-off {{ background: var(--fail-bg); color: var(--fail-color); font-weight: 700; }}
.cm-zero {{ color: #ced4da; }}
.filters {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; align-items: center; }}
.filters select, .filters input {{ padding: 0.45rem 0.6rem; border: 1px solid var(--border); border-radius: 8px; font-size: 0.82rem; background: #fff; }}
.filters input {{ flex: 1; min-width: 160px; }}
table.cases {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
table.cases th {{ text-align: left; padding: 0.55rem 0.6rem; color: var(--subtle); border-bottom: 2px solid var(--border); position: sticky; top: 0; background: var(--card-bg); }}
table.cases td {{ padding: 0.55rem 0.6rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
table.cases tr:hover td {{ background: #fbfcfd; }}
.mono {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.76rem; }}
.detail {{ color: var(--subtle); max-width: 420px; }}
.suite-tag {{ background: var(--primary-bg); color: var(--primary); padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.7rem; font-weight: 600; }}
.pill {{ padding: 0.1rem 0.55rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600; }}
.pill.pass {{ background: var(--pass-bg); color: var(--pass-color); }}
.pill.fail {{ background: var(--fail-bg); color: var(--fail-color); }}
.pill.neutral {{ background: #e9ecef; color: var(--subtle); }}
.muted {{ color: var(--subtle); font-size: 0.85rem; }}
@media print {{ .tab-button, .download-button, .filters {{ display: none !important; }} .report-tab {{ display: block !important; }} body {{ padding: 0; }} }}
</style>
</head>
<body>
<div class="container">
  <header class="report-header">
    <div class="header-title-row">
      <h1>RentReady — Evaluation Execution Report</h1>
      <span class="badge brand">RentReady</span>
      <span class="badge {status_cls}">{status_badge}</span>
    </div>
    <h3>Two-tier evaluation: deterministic CI gates + LLM-as-judge & RAGAS</h3>
    <p>Generated at: {generated} · {n_cases} cases scored</p>
  </header>

  <div class="tabs">
    <div class="tab-buttons">
      <button id="tb-metrics" class="tab-button active" onclick="showTab('metrics')">Metrics &amp; Data</button>
      <button id="tb-cases" class="tab-button" onclick="showTab('cases')">Cases &amp; Filters</button>
    </div>
    <button class="download-button" onclick="window.print()">Convert to PDF</button>
  </div>

  <div id="tab-metrics" class="report-tab active">
    <div class="summary-dashboard">{cards}</div>

    <div class="section">
      <h3>Evaluation Breakdown by Type</h3>
      <div class="eval-types">{breakdown}</div>
    </div>

    <div class="section">
      <h3>Metric vs Threshold</h3>
      {bars}
    </div>

    <div class="two-col">
      <div class="section">
        <h3>Eligibility Confusion Matrix</h3>
        {confusion}
      </div>
      <div class="section">
        <h3>Extraction — Per-field Accuracy</h3>
        {field_bars}
      </div>
    </div>
  </div>

  <div id="tab-cases" class="report-tab">
    <div class="section">
      <h3>Per-case Results</h3>
      <div class="filters">
        <select id="f-suite" onchange="filterRows()">
          <option value="">All suites</option>
          <option value="eligibility">eligibility</option>
          <option value="extraction">extraction</option>
          <option value="recommendations">recommendations</option>
          <option value="judge">judge</option>
          <option value="ragas">ragas</option>
        </select>
        <select id="f-result" onchange="filterRows()">
          <option value="">All results</option>
          <option value="pass">pass</option>
          <option value="fail">fail</option>
          <option value="info">info</option>
        </select>
        <input id="f-text" type="text" placeholder="Search case or detail…" oninput="filterRows()">
      </div>
      <table class="cases">
        <thead><tr><th>Suite</th><th>Case</th><th>Metric</th><th>Score</th><th>Result</th><th>Detail</th></tr></thead>
        <tbody id="case-body">{rows}</tbody>
      </table>
    </div>
  </div>
</div>

<script>
function showTab(name) {{
  document.querySelectorAll('.report-tab').forEach(function(t) {{ t.classList.remove('active'); }});
  document.querySelectorAll('.tab-button').forEach(function(b) {{ b.classList.remove('active'); }});
  document.getElementById('tab-' + name).classList.add('active');
  document.getElementById('tb-' + name).classList.add('active');
}}
function filterRows() {{
  var s = document.getElementById('f-suite').value;
  var r = document.getElementById('f-result').value;
  var q = document.getElementById('f-text').value.toLowerCase();
  document.querySelectorAll('#case-body tr').forEach(function(row) {{
    var okS = !s || row.dataset.suite === s;
    var okR = !r || row.dataset.result === r;
    var okQ = !q || row.textContent.toLowerCase().indexOf(q) !== -1;
    row.style.display = (okS && okR && okQ) ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""


def write_report(out: Optional[Path] = None) -> Path:
    from evals import run_evals

    results = run_evals.load_latest()
    out = out or (RESULTS / "report.html")
    out.write_text(build_html(results), encoding="utf-8")
    return out


if __name__ == "__main__":
    path = write_report()
    print(f"Report written to {path}")
