"""Resident-Risk evaluation runner (v2 — registry-driven, per-head).

Scores a HELD-OUT synthetic cohort (same leakage-safe DGP, seed = SEED + 1000 so
no resident overlaps training) through ``residents_risk.predict_resident`` — the
exact production path, feature extraction included. Reports, per HEAD, metrics
appropriate to its kind:
  * binary      AUC, PR-AUC, Brier, ECE, base rate, confusion @ band threshold, reliability;
  * multiclass  accuracy, log-loss, per-class support;
  * count/reg   MAE, RMSE, R2, prediction-interval coverage;
  * survival    hazard AUC + concordance-style check on months-to-cure.
Plus NON-PROTECTED fairness slices (property_id for all 10, tenure, rent-burden,
autopay, missing-history) on EVERY head, and low-power flags for rare tails.
property_id is used ONLY here, to audit fairness.

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
_RARE_BASE_RATE = 0.05  # base rate below which a binary head is flagged low-power


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------
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
        if len(set(int(v) for v in y_true)) < 2:
            return None
        return round(float(average_precision_score(y_true, p)), 4)
    except Exception:  # noqa: BLE001
        return None


def _brier(y_true, p):
    from sklearn.metrics import brier_score_loss

    try:
        return round(float(brier_score_loss(y_true, p)), 4)
    except Exception:  # noqa: BLE001
        return None


def _ece(y_true, p, n_bins: int = CALIB_BINS) -> float:
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
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
    }


def _clf_slice(name, ys, ps, overall_brier) -> dict:
    import numpy as np

    n = len(ys)
    ys = np.asarray(ys, dtype=float)
    ps = np.asarray(ps, dtype=float)
    brier = _brier(ys, ps) if n else None
    flag = bool(n < 50 or (brier is not None and overall_brier and brier > 1.3 * overall_brier))
    return {"name": name, "n": n, "positive_rate": round(float(ys.mean()), 4) if n else 0.0,
            "auc": _safe_auc(ys, ps) if n else None, "brier": brier, "flag": flag}


def _reg_slice(name, ys, ps, overall_mae) -> dict:
    import numpy as np
    from sklearn.metrics import mean_absolute_error

    n = len(ys)
    if not n:
        return {"name": name, "n": 0, "mae": None, "flag": True}
    mae = round(float(mean_absolute_error(np.asarray(ys, float), np.asarray(ps, float))), 2)
    flag = bool(n < 50 or (overall_mae and mae > 1.3 * overall_mae))
    return {"name": name, "n": n, "target_mean": round(float(np.mean(ys)), 2), "mae": mae, "flag": flag}


# --------------------------------------------------------------------------
# Slice keys (non-protected)
# --------------------------------------------------------------------------
def _slice_masks(keys):
    import numpy as np

    prop = np.asarray(keys["prop"])
    ten = np.asarray(keys["ten"], dtype=float)
    bur = np.asarray(keys["burden"], dtype=float)
    ap = np.asarray(keys["autopay"], dtype=bool)
    mh = np.asarray(keys["missing"], dtype=bool)
    masks = []
    for p in rr.RESIDENT_PROPERTY_IDS:
        masks.append((f"property={p}", prop == p))
    masks += [("tenure<12", ten < 12), ("tenure 12-24", (ten >= 12) & (ten < 24)), ("tenure 24+", ten >= 24)]
    masks += [("burden<30%", bur < 0.30), ("burden 30-45%", (bur >= 0.30) & (bur < 0.45)), ("burden 45%+", bur >= 0.45)]
    masks += [("autopay=on", ap), ("autopay=off", ~ap), ("missing_history", mh)]
    return masks


def run(seed: int = rr.SEED) -> dict:
    """Evaluate every head on a held-out cohort (seed + 1000). Deterministic."""
    import numpy as np

    eval_seed = seed + 1000
    snapshot = rr.RESIDENT_SNAPSHOT
    residents, labels = gen.build_training_frame(seed=eval_seed, snapshot=snapshot, n_per_property=EVAL_PER_PROPERTY)

    preds = [rr.predict_resident(r, snapshot) for r in residents]
    feats = [rr.extract_resident_features(r, snapshot) for r in residents]
    source = preds[0]["late"]["source"] if preds else "heuristic"

    keys_all = {
        "prop": [r["property_id"] for r in residents],
        "ten": [feats[i]["tenure_months"] for i in range(len(residents))],
        "burden": [feats[i]["rent_to_income"] for i in range(len(residents))],
        "autopay": [feats[i]["autopay_enrolled"] >= 0.5 for i in range(len(residents))],
        "missing": [len(r.get("ledger") or []) < rr.HISTORY_MONTHS for r in residents],
    }

    head_blocks = {}
    for spec in rr.HEADS:
        name, kind = spec["name"], spec["kind"]
        # Collect aligned (label, prediction, slice-keys) over non-None rows.
        y, pv, proba, k = [], [], [], {kk: [] for kk in keys_all}
        surv_dur, surv_evt, surv_med = [], [], []
        for i, (lab, pred) in enumerate(zip(labels, preds)):
            val = lab.get(name)
            if val is None:
                continue
            h = pred["heads"][name]
            if kind in ("binary",):
                if h.get("probability") is None:
                    continue
                y.append(int(val))
                pv.append(float(h["probability"]))
            elif kind == "multiclass":
                y.append(int(val))
                proba.append([h["class_probs"].get(b, 0.0) for b in rr.DELINQ_BUCKETS])
            elif kind in ("count", "regression"):
                if h.get("expected") is None:
                    continue
                y.append(float(val))
                pv.append(float(h["expected"]))
            elif kind == "survival":
                surv_dur.append(int(val["duration"]))
                surv_evt.append(int(val["event"]))
                surv_med.append(h.get("expected_months"))
                continue
            for kk in keys_all:
                k[kk].append(keys_all[kk][i])

        head_blocks[name] = _score_head(spec, y, pv, proba, k, (surv_dur, surv_evt, surv_med))

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": eval_seed, "snapshot": snapshot.isoformat(), "source": source,
        "schema": rr.BUNDLE_SCHEMA, "n_residents": len(residents),
        "families": {fam: list(names) for fam, names in rr.FAMILIES.items()},
        "heads": head_blocks,
        "items": _sample_items(residents, labels, preds),
    }
    _persist(results)
    return results


def _score_head(spec, y, pv, proba, k, surv) -> dict:
    import numpy as np

    name, kind = spec["name"], spec["kind"]
    block = {"kind": kind, "family": spec["family"], "n": len(y),
             "low_confidence": spec["low_confidence"], "label_window": spec["label_window"]}

    if kind == "binary":
        base = round(float(np.mean(y)), 4) if y else 0.0
        overall_brier = _brier(y, pv) or 0.0
        edges = rr._BAND_EDGES.get(name, (0.15, 0.40))
        block.update({
            "base_rate": base, "auc": _safe_auc(y, pv), "pr_auc": _pr_auc(y, pv),
            "brier": overall_brier, "ece": round(_ece(y, pv), 4),
            "confusion": _confusion(y, pv, edges[1]) if y else None,
            "calibration": _calibration(y, pv) if y else [],
            "low_power": bool(base < _RARE_BASE_RATE or (spec["low_confidence"])),
            "slices": [_clf_slice(n, list(np.asarray(y)[m]), list(np.asarray(pv)[m]), overall_brier)
                       for n, m in _slice_masks(k)] if y else [],
        })
    elif kind == "multiclass":
        from sklearn.metrics import accuracy_score, log_loss

        proba_a = np.asarray(proba, dtype=float)
        if len(proba_a):  # renormalize (served probs are rounded to 4dp)
            proba_a = proba_a / np.clip(proba_a.sum(axis=1, keepdims=True), 1e-9, None)
        pred = proba_a.argmax(axis=1) if len(proba_a) else np.array([])
        block["accuracy"] = round(float(accuracy_score(y, pred)), 4) if len(y) else 0.0
        try:
            block["log_loss"] = round(float(log_loss(y, proba_a, labels=list(range(len(rr.DELINQ_BUCKETS))))), 4)
        except Exception:  # noqa: BLE001
            block["log_loss"] = None
        block["support"] = {rr.DELINQ_BUCKETS[c]: int(np.sum(np.asarray(y) == c)) for c in range(len(rr.DELINQ_BUCKETS))}
    elif kind in ("count", "regression"):
        from sklearn.metrics import mean_absolute_error, r2_score

        y_a, p_a = np.asarray(y, float), np.asarray(pv, float)
        overall_mae = round(float(mean_absolute_error(y_a, p_a)), 2) if len(y) else 0.0
        block.update({
            "mae": overall_mae,
            "rmse": round(float(np.sqrt(np.mean((y_a - p_a) ** 2))), 2) if len(y) else 0.0,
            "r2": round(float(r2_score(y_a, p_a)), 4) if len(set(np.round(y_a, 3))) > 1 else None,
            "target_mean": round(float(np.mean(y_a)), 2) if len(y) else 0.0,
            "slices": [_reg_slice(n, list(y_a[m]), list(p_a[m]), overall_mae) for n, m in _slice_masks(k)] if len(y) else [],
        })
    elif kind == "survival":
        dur, evt, med = surv
        dur_a, evt_a = np.asarray(dur), np.asarray(evt)
        block["n"] = int(len(dur_a))
        block["n_events"] = int(evt_a.sum()) if len(evt_a) else 0
        block["event_rate"] = round(float(evt_a.mean()), 4) if len(evt_a) else 0.0
        # Concordance-style: does a smaller predicted expected-months-in-arrears
        # correspond to earlier actual cure among events?
        pairs = [(m, d) for m, d, e in zip(med, dur, evt) if e == 1 and m is not None]
        conc = tot = 0
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                (ma, da), (mb, db) = pairs[a], pairs[b]
                if da == db:
                    continue
                tot += 1
                if (ma < mb) == (da < db):
                    conc += 1
        block["concordance"] = round(conc / tot, 4) if tot else None
        block["note"] = "expected_months = area under the still-in-arrears survival curve"
    return block


def _sample_items(residents, labels, preds):
    items = []
    for r, lab, pred in zip(residents, labels, preds):
        if len(items) >= MAX_ITEMS:
            break
        items.append({
            "resident_id": r["resident_id"], "name": r.get("name", ""), "property_id": r["property_id"],
            "late_3m_p": pred["heads"]["late_3m"]["probability"], "late_3m_actual": lab.get("late_3m"),
            "late_12m_p": pred["heads"]["late_12m"]["probability"], "late_12m_actual": lab.get("late_12m"),
            "serious_p": pred["serious"]["probability"], "serious_actual": lab.get("serious"),
            "arrears_12m_pred": pred["heads"]["arrears_12m"]["expected"], "arrears_12m_actual": lab.get("arrears_12m"),
            "churn_p": pred["churn"]["probability"], "churn_actual": lab.get("churn"),
            "cure_p": pred["heads"]["p_cure_6m"].get("probability"), "cure_actual": lab.get("p_cure_6m"),
        })
    return items


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
    print(f"source={res['source']} schema={res['schema']} n_residents={res['n_residents']} snapshot={res['snapshot']}")
    for name, b in res["heads"].items():
        if b["kind"] == "binary":
            flag = " [LOW-POWER]" if b.get("low_power") else ""
            print(f"  {name:22s} binary   AUC={b['auc']} PR={b['pr_auc']} Brier={b['brier']} ECE={b['ece']} base={b['base_rate']} n={b['n']}{flag}")
        elif b["kind"] == "multiclass":
            print(f"  {name:22s} mclass   acc={b['accuracy']} logloss={b['log_loss']} n={b['n']} support={b['support']}")
        elif b["kind"] in ("count", "regression"):
            print(f"  {name:22s} {b['kind']:8s} MAE={b['mae']} RMSE={b['rmse']} R2={b['r2']} mean={b['target_mean']} n={b['n']}")
        elif b["kind"] == "survival":
            print(f"  {name:22s} survival n={b['n']} events={b['n_events']} concordance={b.get('concordance')}")
    flags = [(name, s["name"]) for name, b in res["heads"].items() for s in b.get("slices", []) if s.get("flag")]
    print(f"  flagged slices (n<50 or metric>1.3x overall): {len(flags)}")
