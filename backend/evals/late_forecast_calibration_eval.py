"""Magnitude/calibration check for the late-payment-count aggregate forecast.

``late_forecast_golden_eval.py`` only asserts DIRECTION (does twin A score >=
twin B) — it would still pass at 100% even if every prediction were off by
10x, as long as relative ordering held. This eval closes that gap: it checks
the model's predicted MAGNITUDE against a genuinely held-out ground truth.

Method: generate a fresh synthetic cohort at a seed NEVER used anywhere else
in this codebase (training uses ``rr.SEED`` == 42; this uses
``_CALIBRATION_SEED``), via ``generate_residents.generate()``. That function
returns both the residents (the same shape ``predict_bulk`` scores in
production) AND the ``late_count_3m``/``late_count_12m`` labels computed from
the DISCARDED future simulation window — i.e. the actual realized outcome,
not a prediction. Sum the model's predicted ``expected`` across the cohort
and compare it against the sum of the REAL realized counts, both portfolio-
wide and per property. A model that's badly miscalibrated (systematically
too high/low, or off by a large multiplicative factor) fails; a model with
ordinary regression noise (expected given late_count_3m's R^2 ~0.36 on this
synthetic DGP) passes.

Usage:  python backend/evals/late_forecast_calibration_eval.py
"""

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import generate_residents as gen  # noqa: E402
import residents_risk as rr  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Deliberately far from rr.SEED (42, used by training/data-gen everywhere
# else) so this cohort and its labels are never seen during training or any
# other eval in this codebase.
_CALIBRATION_SEED = 913_042
_N_PER_PROPERTY = 60  # 600 residents total — enough for a stable aggregate sum

# Portfolio-wide tolerance: catches gross miscalibration (e.g. an accidental
# unit error, a 1/4 proration bug, a sign flip) without demanding the
# resolution a golden-set direction check already covers. Per-property is
# looser since 60 residents carries more sampling noise than 600.
_PORTFOLIO_REL_TOL = 0.35
_PROPERTY_REL_TOL = 0.60


def _rel_err(pred: float, true: float) -> float:
    denom = max(abs(true), 1.0)
    return abs(pred - true) / denom


def _cohort():
    residents, labels = gen.generate(seed=_CALIBRATION_SEED, n_per_property=_N_PER_PROPERTY)
    preds = rr.predict_bulk(residents, rr.RESIDENT_SNAPSHOT, heads=["late_count_3m", "late_count_12m"])
    return residents, labels, preds


def _aggregate_check(residents: list, labels: list, preds: list, head: str, true_key: str) -> dict:
    true_total = sum(l[true_key] for l in labels)
    pred_total = sum(float((p.get("heads") or {}).get(head, {}).get("expected") or 0.0) for p in preds)
    rel_err = _rel_err(pred_total, true_total)
    return {
        "head": head, "n_residents": len(residents),
        "true_total": round(true_total, 1), "pred_total": round(pred_total, 1),
        "rel_err": round(rel_err, 4), "tol": _PORTFOLIO_REL_TOL,
        "passed": rel_err <= _PORTFOLIO_REL_TOL,
    }


def _per_property_checks(residents: list, labels: list, preds: list, head: str, true_key: str) -> list:
    by_prop: dict = {}
    for r, l, p in zip(residents, labels, preds):
        pid = r["property_id"]
        bucket = by_prop.setdefault(pid, {"true": 0.0, "pred": 0.0, "n": 0})
        bucket["true"] += l[true_key]
        bucket["pred"] += float((p.get("heads") or {}).get(head, {}).get("expected") or 0.0)
        bucket["n"] += 1
    rows = []
    for pid, b in sorted(by_prop.items()):
        rel_err = _rel_err(b["pred"], b["true"])
        rows.append({
            "property_id": pid, "head": head, "n_residents": b["n"],
            "true_total": round(b["true"], 1), "pred_total": round(b["pred"], 1),
            "rel_err": round(rel_err, 4), "tol": _PROPERTY_REL_TOL,
            "passed": rel_err <= _PROPERTY_REL_TOL,
        })
    return rows


def run() -> dict:
    residents, labels, preds = _cohort()
    portfolio = [
        _aggregate_check(residents, labels, preds, "late_count_3m", "late_count_3m"),
        _aggregate_check(residents, labels, preds, "late_count_12m", "late_count_12m"),
    ]
    per_property = (
        _per_property_checks(residents, labels, preds, "late_count_3m", "late_count_3m")
        + _per_property_checks(residents, labels, preds, "late_count_12m", "late_count_12m")
    )
    all_rows = portfolio + per_property
    n_pass = sum(1 for r in all_rows if r["passed"])
    results = {
        "seed": _CALIBRATION_SEED,
        "n_cohort": len(residents),
        "n_checks": len(all_rows),
        "n_pass": n_pass,
        "pass_rate": round(n_pass / len(all_rows), 4),
        "portfolio": portfolio,
        "per_property": per_property,
        "failures": [r for r in all_rows if not r["passed"]],
    }
    _persist(results)
    return results


def _persist(results: dict) -> None:
    try:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "late_forecast_calibration_latest.json").write_text(json.dumps(results, indent=2))
    except OSError:
        pass


if __name__ == "__main__":
    res = run()
    print(f"CALIBRATION CHECKS: {res['n_pass']}/{res['n_checks']} = {100 * res['pass_rate']:.1f}%  "
          f"(cohort={res['n_cohort']} residents, seed={res['seed']})")
    print("\nPortfolio-wide:")
    for r in res["portfolio"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['head']:<16} true={r['true_total']:<8} pred={r['pred_total']:<8} "
              f"rel_err={r['rel_err']:.2f} (tol {r['tol']})")
    if res["failures"]:
        print(f"\n{len(res['failures'])} failing check(s):")
        for r in res["failures"]:
            label = r.get("property_id", "PORTFOLIO")
            print(f"  {label:<10} {r['head']:<16} true={r['true_total']:<8} pred={r['pred_total']:<8} "
                  f"rel_err={r['rel_err']:.2f} (tol {r['tol']})")
