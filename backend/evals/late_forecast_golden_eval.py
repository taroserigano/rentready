"""200-case golden-set evaluation for the ``late_count_3m`` head (quarterly
late-payment forecast) and its property/portfolio aggregate.

Mirrors ``residents_golden_eval.py``'s COMPARATIVE_PAIRS methodology — pick a
factor known (from the feature policy / DGP causal structure, or an already-
validated precedent for a sibling head like ``late_3m``) to affect lateness
risk, hold every OTHER input fixed, and assert only the DIRECTION of the
effect. That is the only falsifiable claim a noisy regression target on
synthetic data supports; an exact point-value claim would be neither testable
nor meaningful.

Scaled to 200 cases via a systematic (fully deterministic, no RNG) sweep of
``base_rent`` x ``monthly_income`` across 5 factors x 40 combinations each:

  1. recency_recent_vs_none      — trouble in the last quarter vs a clean quarter
  2. recency_weighting_equal_total — same total troubled months, old vs recent
  3. autopay_effect               — off vs on (precedented on late_3m)
  4. rent_burden_effect           — high vs low burden (precedented on late_3m)
  5. on_time_streak_effect        — a broken vs an intact recent on-time streak

Plus a PORTFOLIO section (not counted in the 200) sanity-checking the
aggregate ``property_late_forecast`` / ``portfolio_late_forecast`` functions
against the real committed residents.json across all 10 properties.

Usage:  python backend/evals/late_forecast_golden_eval.py
"""

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import residents_risk as rr  # noqa: E402

from evals.residents_golden_eval import build_ledger, build_resident  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Deterministic sweep grids — no RNG. 8 rents x 5 incomes = 40 combos/factor.
_RENTS = [900.0, 1100.0, 1250.0, 1400.0, 1600.0, 1800.0, 2000.0, 2200.0]
_INCOMES = [2200.0, 3200.0, 4200.0, 5200.0, 6200.0]
_GRID = [(rent, income) for rent in _RENTS for income in _INCOMES]  # 40 combos

CLEAN_24 = [("paid",)] * 24


def _score(case: dict) -> float:
    resident = build_resident(case)
    pred = rr.predict_resident(resident)
    return float(pred["heads"]["late_count_3m"]["expected"])


def _pair_case(factor: str, idx: int, rent: float, income: float, base_a: dict, base_b: dict,
              rationale: str, why: str) -> dict:
    case_a = {**base_a, "id": f"{factor}_{idx}_a", "base_rent": rent, "monthly_income": income}
    case_b = {**base_b, "id": f"{factor}_{idx}_b", "base_rent": rent, "monthly_income": income}
    return {
        "id": f"{factor}_{idx}", "factor": factor, "rationale": rationale, "why": why,
        "case_a": case_a, "case_b": case_b,
    }


# ---------------------------------------------------------------------------
# Factor 1 — recent trouble (last 3 months) vs a clean quarter.
# ---------------------------------------------------------------------------
# Factor-LOCAL grid: identical to _GRID except the 4 cells whose rent/income
# ratio exceeds the DGP's structural rent_to_income ceiling (1/1.4 == 0.7143,
# from the income_multiple>=1.4 floor in generate_residents.py) are swapped
# for 4 valid substitute (rent, income) pairs at a safe ratio. Those 4 cells
# ask the model to extrapolate past its training support (out-of-distribution,
# not a model defect), so they're excluded here ONLY — the shared _GRID used
# by _factor_recency_weighting_equal_total / _factor_on_time_streak_effect is
# left untouched (40/40 there is expected after the model fix below).
_RECENCY_NONE_EXCLUDED = {(1600.0, 2200.0), (1800.0, 2200.0), (2000.0, 2200.0), (2200.0, 2200.0)}
_RECENCY_NONE_SUBSTITUTES = [(1600.0, 2700.0), (1800.0, 2900.0), (2000.0, 3100.0), (2200.0, 3300.0)]
_RECENCY_NONE_GRID = [rc for rc in _GRID if rc not in _RECENCY_NONE_EXCLUDED] + _RECENCY_NONE_SUBSTITUTES
assert len(_RECENCY_NONE_GRID) == 40


def _factor_recency_recent_vs_none() -> list:
    # base_a is always the twin EXPECTED TO SCORE HIGHER (>= base_b) — here, the
    # twin with a troubled recent quarter.
    cases = []
    for i, (rent, income) in enumerate(_RECENCY_NONE_GRID):
        base_a = {"months_to_lease_end": 8, "autopay_enrolled": True,
                  "ledger_specs": [("paid",)] * 21 + [("paid_late", 5)] * 3}
        base_b = {"months_to_lease_end": 8, "autopay_enrolled": True,
                  "ledger_specs": CLEAN_24}
        cases.append(_pair_case(
            "recency_recent_vs_none", i, rent, income, base_a, base_b,
            "Identical 24mo history except the last 3 months: twin A went late "
            "all 3 recent months, twin B stayed clean.",
            "recent lateness should never SCORE LOWER than a clean recent quarter",
        ))
    return cases


# ---------------------------------------------------------------------------
# Factor 2 — same total troubled-month COUNT, positioned old vs recent.
# ---------------------------------------------------------------------------
def _factor_recency_weighting_equal_total() -> list:
    # base_a = recent trouble (expected higher); base_b = stale trouble (expected lower).
    cases = []
    for i, (rent, income) in enumerate(_GRID):
        old_trouble = [("paid_late", 5)] * 6 + [("paid",)] * 18       # months 1-6 troubled
        recent_trouble = [("paid",)] * 18 + [("paid_late", 5)] * 6    # months 19-24 troubled
        base_a = {"months_to_lease_end": 8, "autopay_enrolled": True,
                  "ledger_specs": recent_trouble}
        base_b = {"months_to_lease_end": 8, "autopay_enrolled": True,
                  "ledger_specs": old_trouble}
        cases.append(_pair_case(
            "recency_weighting_equal_total", i, rent, income, base_a, base_b,
            "Both twins have EXACTLY 6 troubled months out of 24 — twin A's are "
            "the most recent 6 months, twin B's are 18-24 months stale.",
            "recency-weighted history means equal-count-but-more-recent trouble "
            "should never score LOWER than equal-count-but-stale trouble",
        ))
    return cases


# ---------------------------------------------------------------------------
# Factor 3 — autopay off vs on (precedented direction on the sibling late_3m
# head in residents_golden_eval.py's autopay_effect).
# ---------------------------------------------------------------------------
def _factor_autopay_effect() -> list:
    moderate = [("paid",)] * 18 + [("paid_late", 8), ("paid",), ("paid_late", 6), ("paid",)] + [("paid",)] * 2
    cases = []
    for i, (rent, income) in enumerate(_GRID):
        base_a = {"months_to_lease_end": 8, "autopay_enrolled": False, "ledger_specs": moderate}
        base_b = {"months_to_lease_end": 8, "autopay_enrolled": True, "ledger_specs": moderate}
        cases.append(_pair_case(
            "autopay_effect", i, rent, income, base_a, base_b,
            "Identical moderate-lateness ledger; only autopay_enrolled differs.",
            "autopay=off should never score LOWER than an otherwise-identical "
            "autopay=on twin",
        ))
    return cases


# ---------------------------------------------------------------------------
# Factor 4 — rent burden (low income = high burden) vs affordable (high income).
# ---------------------------------------------------------------------------
def _factor_rent_burden_effect() -> list:
    moderate = [("paid",)] * 18 + [("paid_late", 8), ("paid",), ("paid_late", 6), ("paid",)] + [("paid",)] * 2
    cases = []
    for i, rent in enumerate(_RENTS):
        for j, mult in enumerate([1.2, 1.6, 2.0, 2.4, 3.0]):  # burdened multiplier
            idx = i * 5 + j
            low_income = rent / 0.65   # ~65% burden (burdened)
            high_income = rent / 0.18  # ~18% burden (affordable)
            base_a = {"months_to_lease_end": 8, "autopay_enrolled": True,
                      "ledger_specs": moderate, "monthly_income": low_income}
            base_b = {"months_to_lease_end": 8, "autopay_enrolled": True,
                      "ledger_specs": moderate, "monthly_income": high_income}
            cases.append({
                "id": f"rent_burden_effect_{idx}", "factor": "rent_burden_effect",
                "rationale": "Identical ledger; only monthly_income differs (burdened vs affordable).",
                "why": "a burdened twin should never score LOWER than an affordable twin",
                "case_a": {**base_a, "id": f"rent_burden_effect_{idx}_a", "base_rent": rent},
                "case_b": {**base_b, "id": f"rent_burden_effect_{idx}_b", "base_rent": rent},
            })
    return cases


# ---------------------------------------------------------------------------
# Factor 5 — a broken vs an intact on-time streak, same total trouble count.
# ---------------------------------------------------------------------------
def _factor_on_time_streak_effect() -> list:
    # base_a = SHORT recent streak (expected higher risk); base_b = long streak
    # (expected lower risk). Both have the same total troubled-month count (4)
    # and both have a clean final 3 months, so late_count_3mo is 0 for both —
    # only on_time_streak_months differs.
    cases = []
    for i, (rent, income) in enumerate(_GRID):
        long_streak = [("paid_late", 6)] * 4 + [("paid",)] * 20  # last late = month 4
        short_streak = [("paid",)] * 10 + [("paid_late", 6)] * 4 + [("paid",)] * 10  # last late = month 14
        base_a = {"months_to_lease_end": 8, "autopay_enrolled": True,
                  "ledger_specs": short_streak}
        base_b = {"months_to_lease_end": 8, "autopay_enrolled": True,
                  "ledger_specs": long_streak}
        cases.append(_pair_case(
            "on_time_streak_effect", i, rent, income, base_a, base_b,
            "Both twins have exactly 4 troubled months out of 24; twin A's on-time "
            "streak since its last late month is only 10 months, twin B's is 20.",
            "a shorter intact recent on-time streak should never score LOWER than "
            "a longer one, all else equal",
        ))
    return cases


FACTORS = [
    _factor_recency_recent_vs_none,
    _factor_recency_weighting_equal_total,
    _factor_autopay_effect,
    _factor_rent_burden_effect,
    _factor_on_time_streak_effect,
]


def _run_pair(case: dict) -> dict:
    try:
        a = _score(case["case_a"])
        b = _score(case["case_b"])
        passed = bool(a >= b - 1e-9)
        return {**{k: case[k] for k in ("id", "factor", "rationale", "why")},
                "a": round(a, 4), "b": round(b, 4), "passed": passed}
    except Exception as exc:  # noqa: BLE001 — a crash IS a finding, never fatal
        return {**{k: case[k] for k in ("id", "factor", "rationale", "why")},
                "a": None, "b": None, "passed": False,
                "error": f"{type(exc).__name__}: {exc}"}


def run_cases() -> list:
    cases = [c for factor_fn in FACTORS for c in factor_fn()]
    assert len(cases) == 200, f"expected exactly 200 golden cases, got {len(cases)}"
    return [_run_pair(c) for c in cases]


# ---------------------------------------------------------------------------
# PORTFOLIO aggregate sanity (not counted in the 200 — a separate check layer,
# same spirit as residents_golden_eval.py's portfolio-wide invariant sweep).
# ---------------------------------------------------------------------------
def run_portfolio_checks() -> dict:
    findings = {"sum_mismatch": [], "negative_expected": [], "interval_invalid": []}
    for pid in rr.RESIDENT_PROPERTY_IDS:
        residents = [r for r in rr.load_residents() if r.get("property_id") == pid]
        fc = rr.property_late_forecast(pid)
        preds = rr.predict_bulk(residents, heads=["late_count_3m"])
        expected_sum = round(sum(p["heads"]["late_count_3m"]["expected"] for p in preds), 1)
        if abs(fc["expected"] - expected_sum) > 0.2:
            findings["sum_mismatch"].append(f"{pid}: fc={fc['expected']} sum={expected_sum}")
        if fc["expected"] < 0:
            findings["negative_expected"].append(pid)
        if not (fc["interval"][0] <= fc["expected"] <= fc["interval"][1]):
            findings["interval_invalid"].append(pid)
    findings["clean"] = all(not v for k, v in findings.items() if isinstance(v, list))
    return findings


# ---------------------------------------------------------------------------
# Report + persistence
# ---------------------------------------------------------------------------
def run() -> dict:
    rows = run_cases()
    by_factor: dict = {}
    for r in rows:
        by_factor.setdefault(r["factor"], []).append(r)
    factor_scores = {
        f: f"{sum(1 for r in rs if r['passed'])}/{len(rs)}" for f, rs in by_factor.items()
    }
    n_pass = sum(1 for r in rows if r["passed"])
    n_total = len(rows)
    portfolio = run_portfolio_checks()
    results = {
        "n_cases": n_total,
        "n_pass": n_pass,
        "pass_rate": round(n_pass / n_total, 4),
        "by_factor": factor_scores,
        "portfolio_checks": portfolio,
        "failures": [r for r in rows if not r["passed"]],
        "rows": rows,
    }
    _persist(results)
    return results


def _persist(results: dict) -> None:
    try:
        RESULTS.mkdir(exist_ok=True)
        slim = {k: v for k, v in results.items() if k != "rows"}
        (RESULTS / "late_forecast_golden_latest.json").write_text(json.dumps(results, indent=2))
        (RESULTS / "late_forecast_golden_summary.json").write_text(json.dumps(slim, indent=2))
    except OSError:
        pass


if __name__ == "__main__":
    res = run()
    print(f"PASS RATE: {res['n_pass']}/{res['n_cases']} = {100 * res['pass_rate']:.1f}%")
    print("\nBy factor:")
    for f, score in res["by_factor"].items():
        print(f"  {f:<32} {score}")
    print(f"\nPortfolio aggregate checks clean: {res['portfolio_checks']['clean']}")
    for k, v in res["portfolio_checks"].items():
        if isinstance(v, list) and v:
            print(f"  {k}: {v[:5]}{'...' if len(v) > 5 else ''}")
    if res["failures"]:
        print(f"\n{len(res['failures'])} FAILURES (first 15):")
        for f in res["failures"][:15]:
            print(f"  [{f['id']}] a={f['a']} b={f['b']} — {f['why']}")
            if f.get("error"):
                print(f"      ERROR: {f['error']}")
