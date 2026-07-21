"""Resident Late-Payment Risk evaluation runner.

Scores a HELD-OUT synthetic set (same DGP as training, a different RNG seed so
none of these rows were seen in training) through ``risk.predict`` — i.e. it
exercises the exact production path, feature extraction included. Reports
discrimination (AUC, PR-AUC), calibration (Brier, ECE, 10-bin reliability), a
confusion matrix at the 0.40 review threshold, and NON-PROTECTED slices
(credit bands, income/rent-burden tiers, missing-data) so we can spot where the
model is weak WITHOUT ever slicing on a protected attribute.

Usage:  python backend/evals/risk_eval.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import risk  # noqa: E402
import train_risk  # noqa: E402
from models import ApplicantProfile  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

THRESHOLD = 0.40
EVAL_N = 4000
CALIB_BINS = 10
MAX_ITEMS = 40


def _profile_from_features(f: dict) -> ApplicantProfile:
    """Reconstruct an ApplicantProfile whose ``risk.extract_features`` round-trips
    to ``f``. Uses a fixed income base so the derived ratios reproduce exactly.
    Only allowed features are set; everything else stays at its neutral default."""
    income = 10000.0
    rent = float(f["rent_to_income"]) * income
    kwargs = {
        "monthly_income": income,
        "desired_rent": rent,
        "monthly_debt_payments": float(f["dti"]) * income,
        "credit_score": None if f["credit_imputed"] >= 0.5 else int(round(f["credit_score"])),
        "employment_status": "employed" if f["has_verified_income_source"] >= 0.5 else "unknown",
        "employment_length_months": int(round(f["employment_length_months"])),
        "savings_balance": float(f["savings_runway_months"]) * rent,
        "years_at_current_address": float(f["years_at_current_address"]),
        "current_rent": (rent / float(f["rent_jump"])) if f["rent_jump"] else None,
        "late_payments_12mo": int(round(f["late_payments_12mo"])),
        "evictions_count": int(round(f["evictions_count"])),
        "bankruptcies_count": int(round(f["bankruptcies_count"])),
        "references_count": int(round(f["references_count"])),
        "landlord_reference": bool(f["has_landlord_reference"] >= 0.5),
        "guarantor_available": bool(f["has_guarantor"] >= 0.5),
    }
    return ApplicantProfile(**kwargs)


def _safe_auc(y_true, p):
    from sklearn.metrics import roc_auc_score

    try:
        if len(set(int(v) for v in y_true)) < 2:
            return None
        return round(float(roc_auc_score(y_true, p)), 4)
    except Exception:  # noqa: BLE001
        return None


def _brier(y_true, p):
    from sklearn.metrics import brier_score_loss

    try:
        return round(float(brier_score_loss(y_true, p)), 4)
    except Exception:  # noqa: BLE001
        return None


def _calibration(y_true, p, n_bins: int = CALIB_BINS) -> list:
    import numpy as np

    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        out.append(
            {
                "bin": round((lo + hi) / 2, 3),
                "predicted": round(float(p[mask].mean()), 4) if count else None,
                "observed": round(float(y_true[mask].mean()), 4) if count else None,
                "count": count,
            }
        )
    return out


def _slice_metrics(name: str, ys, ps, overall_brier) -> dict:
    import numpy as np

    n = len(ys)
    ys = np.asarray(ys, dtype=float)
    ps = np.asarray(ps, dtype=float)
    brier = _brier(ys, ps) if n else None
    ece = round(train_risk._ece(ys, ps), 4) if n else None
    # Flag a slice for review when it's materially worse-calibrated than the
    # whole set, or too small to trust. Purely a data-quality signal.
    flag = bool(n < 50 or (brier is not None and overall_brier and brier > 1.3 * overall_brier))
    return {
        "name": name,
        "n": n,
        "positive_rate": round(float(ys.mean()), 4) if n else 0.0,
        "auc": _safe_auc(ys, ps) if n else None,
        "brier": brier,
        "ece": ece,
        "flag": flag,
    }


def run(seed: int = train_risk.SEED) -> dict:
    """Evaluate the risk scorer on a held-out synthetic set. Returns the
    RiskEvalResult dict and persists it. Deterministic given ``seed``."""
    import numpy as np

    # Held-out data: a DIFFERENT seed from training so no row overlaps.
    eval_seed = seed + 1000
    X, _p_true, y = train_risk.generate_dataset(n=EVAL_N, seed=eval_seed)

    ps, bands, confs = [], [], []
    source = "heuristic"
    items = []
    for i in range(len(X)):
        f = {k: float(X.iloc[i][k]) for k in risk.FEATURE_ORDER}
        profile = _profile_from_features(f)
        r = risk.predict(profile, applicant_id=f"eval-{i}")
        ps.append(r["probability"])
        bands.append(r["band"])
        confs.append(r["confidence"])
        source = r["source"]
        if len(items) < MAX_ITEMS:
            actual = int(y[i])
            pred_late = int(r["probability"] >= THRESHOLD)
            items.append(
                {
                    "id": f"eval-{i}",
                    "p": r["probability"],
                    "band": r["band"],
                    "confidence": r["confidence"],
                    "actual": actual,
                    "correct": bool(pred_late == actual),
                    "top_reasons": [rc["label"] for rc in r["reason_codes"][:3]],
                }
            )

    y = np.asarray(y, dtype=int)
    ps = np.asarray(ps, dtype=float)
    n = int(len(y))

    # Confusion @ threshold.
    pred_late = ps >= THRESHOLD
    tp = int(((pred_late == 1) & (y == 1)).sum())
    fp = int(((pred_late == 1) & (y == 0)).sum())
    fn = int(((pred_late == 0) & (y == 1)).sum())
    tn = int(((pred_late == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / n if n else 0.0

    overall_brier = _brier(y, ps) or 0.0

    # NON-PROTECTED slices ------------------------------------------------
    credit = X["credit_score"].to_numpy()
    imputed = X["credit_imputed"].to_numpy()
    rti = X["rent_to_income"].to_numpy()

    def sub(mask):
        idx = np.where(mask)[0]
        return y[idx], ps[idx]

    slices = []
    # Credit bands (on the reported score; imputed handled separately).
    for name, mask in [
        ("Credit < 640", (credit < 640) & (imputed < 0.5)),
        ("Credit 640-699", (credit >= 640) & (credit < 700) & (imputed < 0.5)),
        ("Credit 700+", (credit >= 700) & (imputed < 0.5)),
    ]:
        ys, pss = sub(mask)
        slices.append(_slice_metrics(name, ys, pss, overall_brier))
    # Rent-burden tiers.
    for name, mask in [
        ("Rent burden < 30%", rti < 0.30),
        ("Rent burden 30-45%", (rti >= 0.30) & (rti < 0.45)),
        ("Rent burden 45%+", rti >= 0.45),
    ]:
        ys, pss = sub(mask)
        slices.append(_slice_metrics(name, ys, pss, overall_brier))
    # Missing-data slice.
    ys, pss = sub(imputed >= 0.5)
    slices.append(_slice_metrics("Credit score missing", ys, pss, overall_brier))

    results = {
        "auc": _safe_auc(y, ps),
        "pr_auc": _pr_auc(y, ps),
        "brier": overall_brier,
        "ece": round(train_risk._ece(y, ps), 4),
        "threshold": THRESHOLD,
        "base_rate": round(float(y.mean()), 4),
        "n": n,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "confusion": {
            "actual_late": {"pred_late": tp, "pred_ontime": fn},
            "actual_ontime": {"pred_late": fp, "pred_ontime": tn},
        },
        "confusion_stats": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        },
        "calibration": _calibration(y, ps),
        "slices": slices,
        "items": items,
    }
    _persist(results)
    return results


def _pr_auc(y_true, p):
    from sklearn.metrics import average_precision_score

    try:
        return round(float(average_precision_score(y_true, p)), 4)
    except Exception:  # noqa: BLE001
        return None


def _persist(results: dict) -> None:
    try:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "risk_latest.json").write_text(json.dumps(results, indent=2))
    except OSError:
        pass


def load_latest() -> dict:
    path = RESULTS / "risk_latest.json"
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


if __name__ == "__main__":
    import pprint

    res = run()
    pprint.pp({k: v for k, v in res.items() if k not in ("items", "calibration", "slices")})
