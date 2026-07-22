"""Train the Resident-Risk HEAD catalog on SYNTHETIC data -> ONE bundle (v2).

Registry-driven: every head in ``residents_risk.HEADS`` is trained by a
kind-specific trainer and written into a single atomic joblib bundle keyed by
head name (schema ``residents-heads-v2``).

Kinds:
  binary       XGBClassifier(binary:logistic) + isotonic calibration (TreeSHAP booster kept)
  multiclass   XGBClassifier(multi:softprob)  (delinquency bucket; per-class probs)
  count        XGBRegressor(reg:tweedie, vp~1.3) + empirical residual-quantile PI
  regression   XGBRegressor(reg:tweedie, vp~1.3) + empirical residual-quantile PI
  survival     discrete-time hazard (person-period XGBClassifier) — months_to_cure

Features are extracted ONCE per resident through the EXACT production path
(``residents_risk.extract_resident_features``); the future panel is simulated
ONCE (``generate_residents.build_training_frame`` returns per-head labels). A
single resident-level train/cal/test split is SHARED across heads (each head
filters to its non-None / eligible rows). Heads are fit in PARALLEL (threads).

``residents_risk._model`` refuses to load the bundle if the schema, any per-head
feature_order, or a required kind key drifts from the code contract.

Run:  python train_residents.py   |   python -m train_residents
"""

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import generate_residents as gen  # noqa: E402
import residents_risk as rr  # noqa: E402

SEED = rr.SEED
ARTIFACT_PATH = rr.ARTIFACT_PATH
TRAIN_PER_PROPERTY = 150  # ~1500 residents for training/metric stability
TWEEDIE_VP = 1.3
_MIN_SURVIVAL_EVENTS = 20  # below this, skip the survival head (serve heuristic)


# --------------------------------------------------------------------------
# Constant-probability fallback (picklable) for a degenerate single-class head.
# --------------------------------------------------------------------------
class _ConstantProba:
    """Minimal predict_proba shim so a head always has a servable model even if a
    split degenerates to a single class. Returns the training base rate."""

    def __init__(self, p: float):
        self.p = float(min(0.98, max(0.02, p)))

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _ece(y_true, p, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if not len(p):
        return 0.0
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


def _range_half_width(y_true, p, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.any():
            gaps.append(abs(p[mask].mean() - y_true[mask].mean()))
    gap = float(np.mean(gaps)) if gaps else 0.05
    return float(min(0.20, max(0.05, 0.5 * (1.0 / n_bins) + gap)))


def _clf_metrics(y_true, p) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y_true = np.asarray(y_true)
    out = {"n_test": int(len(y_true)), "base_rate": round(float(np.mean(y_true)), 4) if len(y_true) else 0.0}
    try:
        if len(set(int(v) for v in y_true)) >= 2:
            out["auc"] = round(float(roc_auc_score(y_true, p)), 4)
            out["pr_auc"] = round(float(average_precision_score(y_true, p)), 4)
        else:
            out["auc"] = None
            out["pr_auc"] = None
        out["brier"] = round(float(brier_score_loss(y_true, p)), 4)
        out["ece"] = round(_ece(y_true, p), 4)
    except Exception:  # noqa: BLE001
        out["auc"] = out.get("auc")
    return out


def _reg_metrics(y_true, pred, lo, hi) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2))) if len(y_true) else 0.0
    coverage = float(np.mean((y_true >= np.asarray(lo)) & (y_true <= np.asarray(hi)))) if len(y_true) else 0.0
    return {
        "mae": round(float(mean_absolute_error(y_true, pred)), 2) if len(y_true) else 0.0,
        "rmse": round(rmse, 2),
        "r2": round(float(r2_score(y_true, pred)), 4) if len(set(np.round(y_true, 3))) > 1 else None,
        "pi_coverage": round(coverage, 4),
        "n_test": int(len(y_true)),
        "target_mean": round(float(np.mean(y_true)), 2) if len(y_true) else 0.0,
    }


def _multiclass_metrics(y_true, proba, n_class) -> dict:
    from sklearn.metrics import accuracy_score, log_loss

    y_true = np.asarray(y_true)
    pred = proba.argmax(axis=1)
    out = {"n_test": int(len(y_true)), "accuracy": round(float(accuracy_score(y_true, pred)), 4)}
    try:
        out["log_loss"] = round(float(log_loss(y_true, proba, labels=list(range(n_class)))), 4)
    except Exception:  # noqa: BLE001
        out["log_loss"] = None
    return out


# --------------------------------------------------------------------------
# Estimator factories (mirror train_risk: XGB with HistGB degrade)
# --------------------------------------------------------------------------
def _xgb_classifier(objective, seed, **extra):
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
        objective=objective, tree_method="hist", random_state=seed, n_jobs=1, **extra,
    )


def _xgb_regressor(objective, seed, **extra):
    import xgboost as xgb

    return xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
        objective=objective, tree_method="hist", random_state=seed, n_jobs=1, **extra,
    )


def _calibrate(estimator, X_cal, y_cal):
    from sklearn.calibration import CalibratedClassifierCV

    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="isotonic")
    except ImportError:  # older sklearn — prefit contract API
        calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


# --------------------------------------------------------------------------
# Feature frame + shared split
# --------------------------------------------------------------------------
def _build_data(seed: int, snapshot):
    residents, labels = gen.build_training_frame(seed=seed, snapshot=snapshot, n_per_property=TRAIN_PER_PROPERTY)
    feats = [rr.extract_resident_features(r, snapshot) for r in residents]
    return residents, feats, labels


def _shared_split(n: int, seed: int):
    from sklearn.model_selection import train_test_split

    idx = np.arange(n)
    tr, tmp = train_test_split(idx, test_size=0.40, random_state=seed)
    cal, te = train_test_split(tmp, test_size=0.50, random_state=seed)
    return set(tr.tolist()), set(cal.tolist()), set(te.tolist())


def _matrix(feats, labels, head, subset):
    """Rows/labels for one head over a subset of resident indices (skip None)."""
    fo = head["feature_order"]
    name = head["name"]
    X, y = [], []
    for i in subset:
        lab = labels[i].get(name)
        if lab is None:
            continue
        X.append([float(feats[i].get(k, rr._NEUTRAL.get(k, 0.0))) for k in fo])
        y.append(lab)
    return pd.DataFrame(X, columns=fo), y


# --------------------------------------------------------------------------
# Per-kind trainers -> per-head sub-bundle dict
# --------------------------------------------------------------------------
def _train_binary(head, feats, labels, splits, seed) -> dict:
    tr, cal, te = splits
    fo = head["feature_order"]
    X_tr, y_tr = _matrix(feats, labels, head, tr)
    X_cal, y_cal = _matrix(feats, labels, head, cal)
    X_te, y_te = _matrix(feats, labels, head, te)
    y_tr_arr = np.asarray(y_tr)
    booster = None
    model_type = "xgboost"
    if len(set(y_tr)) < 2 or len(set(y_cal)) < 2:
        model = _ConstantProba(float(np.mean(y_tr_arr)) if len(y_tr_arr) else 0.1)
        model_type = "constant"
    else:
        try:
            clf = _xgb_classifier("binary:logistic", seed, eval_metric="logloss")
            clf.fit(X_tr, y_tr_arr)
            booster = clf.get_booster()
            model = _calibrate(clf, X_cal, y_cal)
        except Exception as exc:  # noqa: BLE001 — degrade to HistGB
            print(f"train_residents: {head['name']} xgb failed ({type(exc).__name__}: {exc}); HistGB.")
            from sklearn.ensemble import HistGradientBoostingClassifier

            clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, random_state=seed)
            clf.fit(X_tr, y_tr_arr)
            model = _calibrate(clf, X_cal, y_cal)
            model_type = "histgb"
    p_te = model.predict_proba(X_te)[:, 1] if len(X_te) else np.array([])
    return {
        "kind": "binary", "objective": head["objective"],
        "calibrated_model": model, "booster": booster,
        "feature_order": list(fo), "model_type": model_type,
        "n_train": int(len(X_tr)),
        "metrics": _clf_metrics(y_te, p_te) if len(X_te) else {"n_test": 0},
        "range_half_width": _range_half_width(y_te, p_te) if len(X_te) else 0.08,
        "bands": rr.BANDS.get(head["name"]),
    }


def _train_multiclass(head, feats, labels, splits, seed) -> dict:
    tr, cal, te = splits
    fo = head["feature_order"]
    n_class = head["num_class"] or len(rr.DELINQ_BUCKETS)
    X_tr, y_tr = _matrix(feats, labels, head, tr)
    X_te, y_te = _matrix(feats, labels, head, te)
    # Ensure every class is represented so predict_proba has n_class columns in order.
    X_all, y_all = _matrix(feats, labels, head, tr | cal | te)
    present = set(int(v) for v in y_tr)
    extra_rows, extra_lab = [], []
    for c in range(n_class):
        if c not in present:
            for xr, yl in zip(X_all.values.tolist(), y_all):
                if int(yl) == c:
                    extra_rows.append(xr)
                    extra_lab.append(c)
                    break
    if extra_rows:
        X_tr = pd.concat([X_tr, pd.DataFrame(extra_rows, columns=fo)], ignore_index=True)
        y_tr = list(y_tr) + extra_lab
    model_type = "xgboost"
    booster = None
    try:
        clf = _xgb_classifier("multi:softprob", seed, num_class=n_class, eval_metric="mlogloss")
        clf.fit(X_tr, np.asarray(y_tr))
        booster = clf.get_booster()
        model = clf
    except Exception as exc:  # noqa: BLE001
        print(f"train_residents: {head['name']} multiclass xgb failed ({type(exc).__name__}: {exc}); HistGB.")
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, random_state=seed)
        model.fit(X_tr, np.asarray(y_tr))
        model_type = "histgb"
    proba_te = model.predict_proba(X_te) if len(X_te) else np.empty((0, n_class))
    return {
        "kind": "multiclass", "objective": head["objective"],
        "calibrated_model": model, "booster": booster,
        "feature_order": list(fo), "model_type": model_type,
        "num_class": n_class, "classes": list(range(n_class)),
        "n_train": int(len(X_tr)),
        "metrics": _multiclass_metrics(y_te, proba_te, n_class) if len(X_te) else {"n_test": 0},
    }


def _train_regression(head, feats, labels, splits, seed) -> dict:
    tr, cal, te = splits
    fo = head["feature_order"]
    X_tr, y_tr = _matrix(feats, labels, head, tr | cal)
    X_te, y_te = _matrix(feats, labels, head, te)
    y_tr = np.asarray(y_tr, dtype=float)
    model_type = "xgboost"
    booster = None
    try:
        reg = _xgb_regressor("reg:tweedie", seed, tweedie_variance_power=TWEEDIE_VP)
        reg.fit(X_tr, y_tr)
        booster = reg.get_booster()
    except Exception as exc:  # noqa: BLE001
        print(f"train_residents: {head['name']} xgb failed ({type(exc).__name__}: {exc}); HistGB.")
        from sklearn.ensemble import HistGradientBoostingRegressor

        reg = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, random_state=seed)
        reg.fit(X_tr, y_tr)
        model_type = "histgb"
    pred_te = np.clip(reg.predict(X_te), 0.0, None) if len(X_te) else np.array([])
    y_te_arr = np.asarray(y_te, dtype=float)
    # Empirical residual-quantile prediction interval (P10..P90 offsets).
    if len(pred_te):
        resid = y_te_arr - pred_te
        q_lo = float(np.percentile(resid, 10))
        q_hi = float(np.percentile(resid, 90))
    else:
        q_lo, q_hi = -50.0, 50.0
    lo = np.clip(pred_te + q_lo, 0.0, None)
    hi = pred_te + q_hi
    return {
        "kind": head["kind"], "objective": head["objective"],
        "regressor": reg, "booster": booster,
        "feature_order": list(fo), "model_type": model_type,
        "n_train": int(len(X_tr)),
        "residual_quantiles": [round(q_lo, 3), round(q_hi, 3)],
        "metrics": _reg_metrics(y_te, pred_te, lo, hi) if len(X_te) else {"n_test": 0},
    }


def _train_survival(head, feats, labels, splits, seed) -> dict:
    """Discrete-time hazard via person-period expansion. Returns a sub with a
    ``hazard_model`` (or no model if too few events -> serve heuristic)."""
    tr, cal, te = splits
    fo = head["feature_order"]
    haz_fo = list(fo) + ["horizon_month"]
    horizon = head["horizon"] or rr.CURE_HORIZON_MONTHS

    def expand(subset):
        rows, ys = [], []
        for i in subset:
            lab = labels[i].get(head["name"])
            if lab is None:
                continue
            d, e = int(lab["duration"]), int(lab["event"])
            base = [float(feats[i].get(k, rr._NEUTRAL.get(k, 0.0))) for k in fo]
            for m in range(1, d + 1):
                rows.append(base + [float(m)])
                ys.append(1 if (e == 1 and m == d) else 0)
        return pd.DataFrame(rows, columns=haz_fo), np.asarray(ys)

    X_tr, y_tr = expand(tr | cal)
    X_te, y_te = expand(te)
    sub = {
        "kind": "survival", "objective": head["objective"],
        "feature_order": list(fo), "hazard_feature_order": haz_fo,
        "horizon": horizon, "model_type": "heuristic",
        "n_train": int(len(X_tr)),
    }
    if len(X_tr) == 0 or int(y_tr.sum()) < _MIN_SURVIVAL_EVENTS or len(set(y_tr.tolist())) < 2:
        sub["metrics"] = {"n_events_train": int(y_tr.sum()) if len(y_tr) else 0,
                          "note": "too few cure events — served by heuristic"}
        return sub  # no hazard_model -> _model allows survival absent
    try:
        haz = _xgb_classifier("binary:logistic", seed, eval_metric="logloss")
        haz.fit(X_tr, y_tr)
        sub["hazard_model"] = haz
        sub["model_type"] = "xgboost"
        p_te = haz.predict_proba(X_te)[:, 1] if len(X_te) else np.array([])
        sub["metrics"] = {
            "hazard_auc": _clf_metrics(y_te, p_te).get("auc") if len(X_te) else None,
            "n_events_train": int(y_tr.sum()),
            "n_person_periods_test": int(len(X_te)),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"train_residents: {head['name']} survival fit failed ({type(exc).__name__}: {exc}); heuristic.")
        sub["metrics"] = {"note": f"survival fit failed: {type(exc).__name__}"}
    return sub


_TRAINERS = {
    "binary": _train_binary,
    "multiclass": _train_multiclass,
    "count": _train_regression,
    "regression": _train_regression,
    "survival": _train_survival,
}


def _train_one(head, feats, labels, splits, seed):
    try:
        return head["name"], _TRAINERS[head["kind"]](head, feats, labels, splits, seed)
    except Exception as exc:  # noqa: BLE001 — one head must never sink the run
        print(f"train_residents: head {head['name']} training FAILED ({type(exc).__name__}: {exc}).")
        return head["name"], None


def train(seed: int = SEED, snapshot=rr.RESIDENT_SNAPSHOT) -> dict:
    """Train every head, evaluate on the shared held-out split, persist atomically."""
    from joblib import Parallel, delayed

    residents, feats, labels = _build_data(seed, snapshot)
    splits = _shared_split(len(residents), seed)

    results = Parallel(n_jobs=min(8, len(rr.HEADS)), prefer="threads")(
        delayed(_train_one)(h, feats, labels, splits, seed) for h in rr.HEADS
    )
    heads = {name: sub for name, sub in results if sub is not None}

    bundle = {
        "dgp_version": rr.DGP_VERSION,
        "seed": seed,
        "schema": rr.BUNDLE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "heads": heads,
    }
    _atomic_dump(bundle)

    # Headline metrics for the console.
    def m(name):
        return heads.get(name, {}).get("metrics", {})

    print(f"Trained resident bundle ({len(heads)} heads). Artifact: {ARTIFACT_PATH}")
    print("  late horizons AUC:", {h: m(h).get("auc") for h in ("late_1m", "late_3m", "late_6m", "late_12m")})
    print("  late horizons base_rate:", {h: m(h).get("base_rate") for h in ("late_1m", "late_3m", "late_6m", "late_12m")})
    print("  severity AUC:", {h: m(h).get("auc") for h in ("p_30d_12m", "p_60d_12m", "p_90d_12m", "serious")})
    print("  severity base_rate:", {h: m(h).get("base_rate") for h in ("p_30d_12m", "p_60d_12m", "p_90d_12m")})
    print("  frequency:", {h: {"mae": m(h).get("mae"), "r2": m(h).get("r2")} for h in ("late_count_12m", "missed_count_12m")})
    print("  arrears/severity reg:", {h: {"mae": m(h).get("mae"), "r2": m(h).get("r2"), "pi": m(h).get("pi_coverage")}
                                       for h in ("arrears_3m", "arrears_12m", "peak_balance_12m", "max_days_late_12m")})
    print("  bucket:", m("delinquency_bucket_12m"))
    print("  cure:", {"p_cure_6m_auc": m("p_cure_6m").get("auc"), "months_to_cure": m("months_to_cure")})
    print("  retention AUC:", {h: m(h).get("auc") for h in ("churn", "churn_12m")})
    return {name: heads[name].get("metrics", {}) for name in heads}


def _atomic_dump(bundle: dict) -> None:
    import joblib

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(ARTIFACT_PATH.parent), suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(bundle, tmp)
        os.replace(tmp, ARTIFACT_PATH)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


if __name__ == "__main__":
    train()
