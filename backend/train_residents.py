"""Train the four Resident-Risk estimators on SYNTHETIC data -> ONE bundle.

Targets:
  late    — XGBClassifier + isotonic calibration (TreeSHAP booster kept)
  serious — XGBClassifier + isotonic calibration
  churn   — XGBClassifier + isotonic calibration (trained ONLY on residents whose
            lease ends within the churn horizon — the served-eligible subset)
  arrears — XGBRegressor (reg:squarederror), prediction interval from held-out
            residual std

The training frame comes from ``generate_residents.build_training_frame`` (a
larger cohort of the SAME leakage-safe DGP that produced data/residents.json).
Features are extracted through the EXACT production path
(``residents_risk.extract_resident_features``) so train/serve cannot skew.

One atomic joblib bundle is written to backend/artifacts/residents_model.joblib
with the contract keys; ``residents_risk._model`` refuses to load it if any
per-target feature_order drifts from the code contract.

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


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _ece(y_true, p, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    p = np.asarray(p)
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
    y_true = np.asarray(y_true)
    p = np.asarray(p)
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

    return {
        "auc": round(float(roc_auc_score(y_true, p)), 4),
        "pr_auc": round(float(average_precision_score(y_true, p)), 4),
        "brier": round(float(brier_score_loss(y_true, p)), 4),
        "ece": round(_ece(y_true, p), 4),
        "n_test": int(len(y_true)),
        "base_rate": round(float(np.mean(y_true)), 4),
    }


def _reg_metrics(y_true, pred, lo, hi) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    coverage = float(np.mean((y_true >= np.asarray(lo)) & (y_true <= np.asarray(hi))))
    return {
        "mae": round(float(mean_absolute_error(y_true, pred)), 2),
        "rmse": round(rmse, 2),
        "r2": round(float(r2_score(y_true, pred)), 4),
        "pi_coverage": round(coverage, 4),
        "n_test": int(len(y_true)),
        "target_mean": round(float(np.mean(y_true)), 2),
    }


# --------------------------------------------------------------------------
# Estimators (mirror train_risk._fit_estimator: XGB + HistGB degrade)
# --------------------------------------------------------------------------
def _fit_classifier(X_tr, y_tr, seed):
    try:
        import xgboost as xgb

        clf = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=0,
        )
        clf.fit(X_tr, y_tr)
        return clf, clf.get_booster(), "xgboost"
    except Exception as exc:  # noqa: BLE001 — degrade to sklearn
        print(f"train_residents: xgboost unavailable ({type(exc).__name__}: {exc}); using HistGB.")
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, random_state=seed)
        clf.fit(X_tr, y_tr)
        return clf, None, "histgb"


def _fit_regressor(X_tr, y_tr, seed):
    try:
        import xgboost as xgb

        reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=0,
        )
        reg.fit(X_tr, y_tr)
        return reg, reg.get_booster(), "xgboost"
    except Exception as exc:  # noqa: BLE001
        print(f"train_residents: xgboost unavailable ({type(exc).__name__}: {exc}); using HistGB.")
        from sklearn.ensemble import HistGradientBoostingRegressor

        reg = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, random_state=seed)
        reg.fit(X_tr, y_tr)
        return reg, None, "histgb"


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
# Feature frame from the training cohort
# --------------------------------------------------------------------------
def _build_frames(seed: int, snapshot):
    residents, labels = gen.build_training_frame(seed=seed, snapshot=snapshot, n_per_property=TRAIN_PER_PROPERTY)
    feats = [rr.extract_resident_features(r, snapshot) for r in residents]
    frames = {}
    for t in rr.TARGETS:
        fo = rr.FEATURE_ORDER[t]
        rows, ys = [], []
        for f, lab in zip(feats, labels):
            val = lab[t]
            if val is None:  # churn: skip horizon-ineligible residents
                continue
            rows.append([float(f.get(k, rr._NEUTRAL.get(k, 0.0))) for k in fo])
            ys.append(val)
        frames[t] = (pd.DataFrame(rows, columns=fo), np.asarray(ys))
    return frames


def _train_classifier(t: str, X, y, seed) -> dict:
    from sklearn.model_selection import train_test_split

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, random_state=seed, stratify=y)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)
    estimator, booster, model_type = _fit_classifier(X_tr, y_tr, seed)
    calibrated = _calibrate(estimator, X_cal, y_cal)
    p_te = calibrated.predict_proba(X_te)[:, 1]
    return {
        "calibrated_model": calibrated,
        "booster": booster,
        "feature_order": list(rr.FEATURE_ORDER[t]),
        "model_type": model_type,
        "n_train": int(len(X_tr)),
        "metrics": _clf_metrics(y_te, p_te),
        "range_half_width": _range_half_width(y_te, p_te),
        "bands": rr.BANDS.get(t),
    }


def _train_regressor(t: str, X, y, seed) -> dict:
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    reg, booster, model_type = _fit_regressor(X_tr, y_tr, seed)
    pred_te = np.clip(reg.predict(X_te), 0.0, None)
    residual_std = float(np.std(np.asarray(y_te, dtype=float) - pred_te)) or 50.0
    half = 1.28 * residual_std
    lo = np.clip(pred_te - half, 0.0, None)
    hi = pred_te + half
    return {
        "regressor": reg,
        "booster": booster,
        "feature_order": list(rr.FEATURE_ORDER[t]),
        "model_type": model_type,
        "n_train": int(len(X_tr)),
        "residual_std": round(residual_std, 2),
        "metrics": _reg_metrics(y_te, pred_te, lo, hi),
    }


def train(seed: int = SEED, snapshot=rr.RESIDENT_SNAPSHOT) -> dict:
    """Train all four estimators, evaluate on held-out splits, persist atomically."""
    frames = _build_frames(seed, snapshot)
    bundle = {
        "dgp_version": rr.DGP_VERSION,
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    for t in ("late", "serious", "churn"):
        X, y = frames[t]
        bundle[t] = _train_classifier(t, X, y, seed)
    Xa, ya = frames["arrears"]
    bundle["arrears"] = _train_regressor("arrears", Xa, ya, seed)

    _atomic_dump(bundle)
    m = {t: bundle[t]["metrics"] for t in rr.TARGETS}
    print(
        "Trained resident bundle. "
        f"late AUC={m['late']['auc']} serious AUC={m['serious']['auc']} "
        f"churn AUC={m['churn']['auc']} (n_train churn={bundle['churn']['n_train']}) "
        f"arrears MAE={m['arrears']['mae']} R2={m['arrears']['r2']}. "
        f"Artifact: {ARTIFACT_PATH}"
    )
    return {t: bundle[t]["metrics"] for t in rr.TARGETS}


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
