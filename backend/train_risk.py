"""Train the Resident Late-Payment Risk model on SYNTHETIC data.

Documented data-generating process (DGP): sample realistic financial/payment
features, build a latent logistic score with signed weights, choose the
intercept so the base rate ≈ 0.20, sample labels, then flip ~3% of labels so
the task is learnable but NOT trivially separable (test AUC ends up ~0.8, not
~1.0).

Model: XGBClassifier (moderate depth, ~300 trees), isotonic-calibrated on a
held-out calibration split, evaluated on a held-out test split. The raw booster
is persisted alongside the calibrated model so ``risk.py`` can compute native
TreeSHAP reason codes without the ``shap`` package.

Run:  python train_risk.py   |   python -m train_risk
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

import risk  # noqa: E402  (FEATURE_ORDER, ARTIFACT_PATH, DGP_VERSION)

SEED = 42
ARTIFACT_PATH = risk.ARTIFACT_PATH

# Signed latent weights (log-odds space). Direction matches the feature policy.
# Features NOT listed here (references_count, has_landlord_reference,
# credit_imputed) carry ~zero true signal by design.
_WEIGHTS = {
    "rent_to_income": 4.2,        # + higher burden -> more risk
    "dti": 3.2,                   # +
    "credit_score_dev": -0.016,   # - per point above 680
    "savings_runway_months": -0.13,  # -
    "employment_length_months": -0.014,  # -
    "unstable_income": 0.9,       # + when income source unverified
    "late_payments_12mo": 1.15,   # + (strong)
    "evictions_count": 1.5,       # +
    "bankruptcies_count": 1.05,   # +
    "years_at_current_address": -0.07,  # -
    "has_guarantor": -0.7,        # -
    "rent_jump": 1.4,             # + per unit above 1.0
}

_TARGET_BASE_RATE = 0.20
_FLIP_RATE = 0.03


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate_dataset(n: int = 20000, seed: int = SEED):
    """Return (X: DataFrame[FEATURE_ORDER], p_true: np.ndarray, y: np.ndarray).

    Deterministic given ``seed`` (``np.random.default_rng``)."""
    rng = np.random.default_rng(seed)

    # --- sample realistic raw-ish features -------------------------------
    rent_to_income = np.clip(rng.normal(0.30, 0.10, n), 0.08, 0.95)
    dti = np.clip(rng.normal(0.15, 0.09, n), 0.0, 0.75)

    credit_score = np.clip(rng.normal(690, 65, n), 500, 830).round()
    credit_missing = rng.random(n) < 0.08  # some applicants have no score on file
    credit_imputed = credit_missing.astype(float)
    # Missing -> neutral-imputed to 680 (so it can't cheat the model).
    credit_used = np.where(credit_missing, 680.0, credit_score)

    savings_runway_months = np.clip(rng.exponential(4.0, n), 0.0, 48.0)
    employment_length_months = np.clip(rng.exponential(30.0, n), 0.0, 300.0).round()
    has_verified_income_source = (rng.random(n) < 0.85).astype(float)
    years_at_current_address = np.clip(rng.exponential(3.0, n), 0.0, 30.0)
    rent_jump = np.clip(rng.normal(1.05, 0.15, n), 0.4, 2.5)

    late_payments_12mo = np.clip(rng.poisson(0.4, n), 0, 12).astype(float)
    evictions_count = np.clip(rng.poisson(0.08, n), 0, 5).astype(float)
    bankruptcies_count = np.clip(rng.poisson(0.05, n), 0, 4).astype(float)
    references_count = np.clip(rng.poisson(1.5, n), 0, 6).astype(float)
    has_landlord_reference = (rng.random(n) < 0.6).astype(float)
    has_guarantor = (rng.random(n) < 0.2).astype(float)

    # --- latent logistic score (without intercept) ----------------------
    z = (
        _WEIGHTS["rent_to_income"] * (rent_to_income - 0.30)
        + _WEIGHTS["dti"] * (dti - 0.15)
        + _WEIGHTS["credit_score_dev"] * (credit_used - 680.0)
        + _WEIGHTS["savings_runway_months"] * savings_runway_months
        + _WEIGHTS["employment_length_months"] * employment_length_months
        + _WEIGHTS["unstable_income"] * (1.0 - has_verified_income_source)
        + _WEIGHTS["late_payments_12mo"] * late_payments_12mo
        + _WEIGHTS["evictions_count"] * evictions_count
        + _WEIGHTS["bankruptcies_count"] * bankruptcies_count
        + _WEIGHTS["years_at_current_address"] * years_at_current_address
        + _WEIGHTS["has_guarantor"] * has_guarantor
        + _WEIGHTS["rent_jump"] * (rent_jump - 1.0)
    )

    # Choose intercept so mean(sigmoid(z + b)) ≈ target base rate (bisection).
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > _TARGET_BASE_RATE:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2

    p_true = _sigmoid(z + intercept)
    y = (rng.random(n) < p_true).astype(int)

    # ~3% label flip so the task isn't trivially separable.
    flip = rng.random(n) < _FLIP_RATE
    y = np.where(flip, 1 - y, y)

    X = pd.DataFrame(
        {
            "rent_to_income": rent_to_income,
            "dti": dti,
            "credit_score": credit_used,
            "credit_imputed": credit_imputed,
            "savings_runway_months": savings_runway_months,
            "employment_length_months": employment_length_months,
            "has_verified_income_source": has_verified_income_source,
            "years_at_current_address": years_at_current_address,
            "rent_jump": rent_jump,
            "late_payments_12mo": late_payments_12mo,
            "evictions_count": evictions_count,
            "bankruptcies_count": bankruptcies_count,
            "references_count": references_count,
            "has_landlord_reference": has_landlord_reference,
            "has_guarantor": has_guarantor,
        }
    )[risk.FEATURE_ORDER]
    return X, p_true, y


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _ece(y_true, p, n_bins: int = 10) -> float:
    """Expected Calibration Error over ``n_bins`` equal-width probability bins."""
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
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def _range_half_width(y_true, p, n_bins: int = 10) -> float:
    """A conservative interval half-width derived from calibration reliability:
    half the bin width plus the mean |confidence - observed| gap."""
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


def _metrics(y_true, p) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    return {
        "auc": round(float(roc_auc_score(y_true, p)), 4),
        "pr_auc": round(float(average_precision_score(y_true, p)), 4),
        "brier": round(float(brier_score_loss(y_true, p)), 4),
        "ece": round(_ece(y_true, p), 4),
        "n_test": int(len(y_true)),
        "base_rate": round(float(np.mean(y_true)), 4),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _fit_estimator(X_tr, y_tr, X_va, y_va, seed):
    """Fit the primary XGBClassifier; fall back to sklearn HistGB if xgboost is
    unavailable. Returns (fitted_estimator, booster_or_None, model_type)."""
    try:
        import xgboost as xgb

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
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
        print(f"train_risk: xgboost unavailable ({type(exc).__name__}: {exc}); using HistGB.")
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300, random_state=seed
        )
        clf.fit(X_tr, y_tr)
        return clf, None, "histgb"


def train(seed: int = SEED, n: int = 20000) -> dict:
    """Generate data, train + isotonic-calibrate, evaluate, persist atomically.
    Returns the metrics dict."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split

    X, _p_true, y = generate_dataset(n=n, seed=seed)

    # 60/20/20 train / calibration / test, stratified.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=seed, stratify=y
    )
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )

    estimator, booster, model_type = _fit_estimator(X_tr, y_tr, X_cal, y_cal, seed)

    # Isotonic calibration on the held-out calibration split. The base estimator
    # is already fit on the train split, so we freeze it (the modern sklearn
    # replacement for the removed ``cv="prefit"``) and calibrate on X_cal only.
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(
            FrozenEstimator(estimator), method="isotonic"
        )
    except ImportError:  # older sklearn — fall back to the prefit contract API
        calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    # Evaluate on the untouched test split.
    p_te = calibrated.predict_proba(X_te)[:, 1]
    metrics = _metrics(y_te, p_te)
    half_width = _range_half_width(y_te, p_te)

    bundle = {
        "calibrated_model": calibrated,
        "booster": booster,  # raw booster for TreeSHAP (None for HistGB)
        "feature_order": list(risk.FEATURE_ORDER),
        "model_type": model_type,
        "n_train": int(len(X_tr)),
        "metrics": metrics,
        "range_half_width": half_width,
        "dgp_version": risk.DGP_VERSION,
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    _atomic_dump(bundle)
    print(
        f"Trained {model_type} on {bundle['n_train']} rows. "
        f"Test metrics: AUC={metrics['auc']} PR-AUC={metrics['pr_auc']} "
        f"Brier={metrics['brier']} ECE={metrics['ece']} "
        f"base_rate={metrics['base_rate']} (n_test={metrics['n_test']}). "
        f"Artifact: {ARTIFACT_PATH}"
    )
    return metrics


def _atomic_dump(bundle: dict) -> None:
    """Write the joblib bundle atomically (temp file + os.replace)."""
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
