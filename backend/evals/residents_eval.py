"""Resident-Risk evaluation runner.

Scores a HELD-OUT synthetic cohort (same leakage-safe DGP, seed = SEED + 1000 so
no resident overlaps training) through ``residents_risk.predict_resident`` — the
exact production path, feature extraction included. Reports, per target:
  * classifiers (late / serious / churn): AUC, PR-AUC, Brier, ECE, 10-bin
    reliability, confusion @ band threshold (churn on the eligible subset only);
  * regressor (arrears): MAE, RMSE, R2, prediction-interval coverage.
Plus NON-PROTECTED fairness slices (property_id for all 10, tenure, rent-burden,
autopay, missing-history) so we can spot weak spots WITHOUT slicing on a
protected attribute. property_id is used ONLY here, to audit fairness.

Usage:  python backend/evals/residents_eval.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import generate_residents as gen  # noqa: E402
import residents_risk as rr  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

EVAL_PER_PROPERTY = 60  # ~600 held-out residents
CALIB_BINS = 10
MAX_ITEMS = 40


def _safe_auc(y_true, p):
    from sklearn.metrics import roc_auc_score

    try:
        if len(set(int(v) for v in y_true)) < 2:
            return None
        return round(float(roc_auc_score(y_true, p)), 4)
    except Exception:  # noqa: BLE001
        return None


def _pr_auc(y_true, p):
    from sklearn.metrics import average_precision_score

    try:
        return round(float(average_precision_score(y_true, p)), 4)
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
        out.append({
            "bin": round((lo + hi) / 2, 3),
            "predicted": round(float(p[mask].mean()), 4) if count else None,
            "observed": round(float(y_true[mask].mean()), 4) if count else None,
            "count": count,
        })
    return out


def _clf_slice(name: str, ys, ps, overall_brier) -> dict:
    import numpy as np

    n = len(ys)
    ys = np.asarray(ys, dtype=float)
    ps = np.asarray(ps, dtype=float)
    brier = _brier(ys, ps) if n else None
    flag = bool(n < 50 or (brier is not None and overall_brier and brier > 1.3 * overall_brier))
    return {
        "name": name,
        "n": n,
        "positive_rate": round(float(ys.mean()), 4) if n else 0.0,
        "auc": _safe_auc(ys, ps) if n else None,
        "brier": brier,
        "flag": flag,
    }


def _confusion(y, p, threshold):
    import numpy as np

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = p >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "threshold": threshold,
        "matrix": {"actual_pos": {"pred_pos": tp, "pred_neg": fn},
                   "actual_neg": {"pred_pos": fp, "pred_neg": tn}},
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def run(seed: int = rr.SEED) -> dict:
    """Evaluate all four targets on a held-out cohort (seed + 1000). Returns the
    result dict and persists it. Deterministic given ``seed``."""
    import numpy as np

    eval_seed = seed + 1000
    snapshot = rr.RESIDENT_SNAPSHOT
    residents, labels = gen.build_training_frame(
        seed=eval_seed, snapshot=snapshot, n_per_property=EVAL_PER_PROPERTY
    )

    # Collections keyed by target.
    y = {t: [] for t in ("late", "serious", "churn")}
    p = {t: [] for t in ("late", "serious", "churn")}
    arr_true, arr_pred, arr_lo, arr_hi = [], [], [], []
    # Slice keys, parallel to the classifier arrays (churn tracked separately).
    prop_ids, tenures, burdens, autopays, missing_hist = [], [], [], [], []
    churn_prop, churn_ten, churn_burden, churn_autopay, churn_missing = [], [], [], [], []
    source = "heuristic"
    items = []

    for r, lab in zip(residents, labels):
        res = rr.predict_resident(r, snapshot)
        source = res["late"]["source"]
        feats = rr.extract_resident_features(r, snapshot)
        tenure = feats["tenure_months"]
        burden = feats["rent_to_income"]
        autopay = feats["autopay_enrolled"] >= 0.5
        missing = len(r.get("ledger") or []) < rr.HISTORY_MONTHS

        for t in ("late", "serious"):
            y[t].append(int(lab[t]))
            p[t].append(float(res[t]["probability"]))
        prop_ids.append(r["property_id"])
        tenures.append(tenure)
        burdens.append(burden)
        autopays.append(autopay)
        missing_hist.append(missing)

        arr_true.append(float(lab["arrears"]))
        arr_pred.append(float(res["arrears"]["expected_balance"]))
        lo, hi = res["arrears"]["interval"]
        arr_lo.append(float(lo))
        arr_hi.append(float(hi))

        # Churn only on horizon-eligible residents (label not None + served prob).
        if lab["churn"] is not None and res["churn"]["probability"] is not None:
            y["churn"].append(int(lab["churn"]))
            p["churn"].append(float(res["churn"]["probability"]))
            churn_prop.append(r["property_id"])
            churn_ten.append(tenure)
            churn_burden.append(burden)
            churn_autopay.append(autopay)
            churn_missing.append(missing)

        if len(items) < MAX_ITEMS:
            items.append({
                "resident_id": r["resident_id"],
                "property_id": r["property_id"],
                "late_p": res["late"]["probability"],
                "late_actual": int(lab["late"]),
                "serious_p": res["serious"]["probability"],
                "serious_actual": int(lab["serious"]),
                "arrears_pred": res["arrears"]["expected_balance"],
                "arrears_actual": round(float(lab["arrears"]), 2),
                "churn_p": res["churn"]["probability"],
                "churn_actual": lab["churn"],
            })

    def clf_block(t):
        yt, pt = y[t], p[t]
        edge_lo, edge_hi = rr._BAND_EDGES[t]
        overall_brier = _brier(yt, pt) or 0.0
        return {
            "auc": _safe_auc(yt, pt),
            "pr_auc": _pr_auc(yt, pt),
            "brier": overall_brier,
            "ece": round(_ece(yt, pt), 4),
            "base_rate": round(float(np.mean(yt)), 4) if yt else 0.0,
            "n": len(yt),
            "confusion": _confusion(yt, pt, edge_hi),
            "calibration": _calibration(yt, pt),
        }, overall_brier

    late_block, late_brier = clf_block("late")
    serious_block, serious_brier = clf_block("serious")
    churn_block, churn_brier = clf_block("churn")

    # Regressor block.
    arr_true_a = np.asarray(arr_true)
    arr_pred_a = np.asarray(arr_pred)
    coverage = float(np.mean((arr_true_a >= np.asarray(arr_lo)) & (arr_true_a <= np.asarray(arr_hi))))
    from sklearn.metrics import mean_absolute_error, r2_score

    arrears_block = {
        "mae": round(float(mean_absolute_error(arr_true_a, arr_pred_a)), 2),
        "rmse": round(float(np.sqrt(np.mean((arr_true_a - arr_pred_a) ** 2))), 2),
        "r2": round(float(r2_score(arr_true_a, arr_pred_a)), 4),
        "pi_coverage": round(coverage, 4),
        "target_mean": round(float(arr_true_a.mean()), 2),
        "n": len(arr_true),
    }

    # ---- NON-PROTECTED fairness slices -------------------------------------
    prop_ids = np.asarray(prop_ids)
    tenures = np.asarray(tenures)
    burdens = np.asarray(burdens)
    autopays = np.asarray(autopays)
    missing_hist = np.asarray(missing_hist)

    def slices_for(t, yv, pv, keys):
        yv = np.asarray(yv, dtype=float)
        pv = np.asarray(pv, dtype=float)
        overall = _brier(yv, pv) or 0.0
        out = []
        # property_id — ALL 10 properties.
        pid = keys["prop"]
        for prop in rr.RESIDENT_PROPERTY_IDS:
            m = pid == prop
            out.append(_clf_slice(f"property={prop}", yv[m], pv[m], overall))
        # tenure tiers.
        ten = keys["ten"]
        for name, m in [("tenure<12", ten < 12), ("tenure 12-24", (ten >= 12) & (ten < 24)), ("tenure 24+", ten >= 24)]:
            out.append(_clf_slice(name, yv[m], pv[m], overall))
        # rent-burden tiers.
        bur = keys["burden"]
        for name, m in [("burden<30%", bur < 0.30), ("burden 30-45%", (bur >= 0.30) & (bur < 0.45)), ("burden 45%+", bur >= 0.45)]:
            out.append(_clf_slice(name, yv[m], pv[m], overall))
        # autopay on/off.
        ap = keys["autopay"]
        out.append(_clf_slice("autopay=on", yv[ap], pv[ap], overall))
        out.append(_clf_slice("autopay=off", yv[~ap], pv[~ap], overall))
        # missing-history (short tenure < 60 ledger months).
        mh = keys["missing"]
        out.append(_clf_slice("missing_history", yv[mh], pv[mh], overall))
        return out

    base_keys = {"prop": prop_ids, "ten": tenures, "burden": burdens, "autopay": autopays, "missing": missing_hist}
    churn_keys = {
        "prop": np.asarray(churn_prop), "ten": np.asarray(churn_ten),
        "burden": np.asarray(churn_burden), "autopay": np.asarray(churn_autopay),
        "missing": np.asarray(churn_missing),
    }

    late_block["slices"] = slices_for("late", y["late"], p["late"], base_keys)
    serious_block["slices"] = slices_for("serious", y["serious"], p["serious"], base_keys)
    churn_block["slices"] = slices_for("churn", y["churn"], p["churn"], churn_keys)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": eval_seed,
        "snapshot": snapshot.isoformat(),
        "source": source,
        "n_residents": len(residents),
        "late": late_block,
        "serious": serious_block,
        "churn": churn_block,
        "arrears": arrears_block,
        "items": items,
    }
    _persist(results)
    return results


def _ece(y_true, p, n_bins: int = 10) -> float:
    import numpy as np

    if not len(y_true):
        return 0.0
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(p[mask].mean() - y_true[mask].mean())
    return float(ece)


def _persist(results: dict) -> None:
    try:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "residents_latest.json").write_text(json.dumps(results, indent=2))
    except OSError:
        pass


def load_latest() -> dict:
    path = RESULTS / "residents_latest.json"
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


if __name__ == "__main__":
    res = run()
    print(f"source={res['source']} n_residents={res['n_residents']} snapshot={res['snapshot']}")
    for t in ("late", "serious", "churn"):
        b = res[t]
        print(f"  {t:8s} AUC={b['auc']} PR-AUC={b['pr_auc']} Brier={b['brier']} "
              f"ECE={b['ece']} base_rate={b['base_rate']} n={b['n']} "
              f"(property slices={sum(1 for s in b['slices'] if s['name'].startswith('property='))})")
    a = res["arrears"]
    print(f"  arrears  MAE={a['mae']} RMSE={a['rmse']} R2={a['r2']} PI_cov={a['pi_coverage']} n={a['n']}")
    flags = [(t, s['name']) for t in ('late', 'serious', 'churn') for s in res[t]['slices'] if s['flag']]
    print(f"  flagged slices (n<50 or Brier>1.3x): {flags}")
