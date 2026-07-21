"""Generate the SYNTHETIC resident dataset — 5-year rent ledgers + labels.

Deterministic (seed=42) and snapshot-relative (default RESIDENT_SNAPSHOT; NO
``datetime.now()`` anywhere), so re-running reproduces data/residents.json byte
for byte. ~250 residents across 10 frozen properties, each with a 60-month rent
ledger.

Data-generating process
------------------------
Per resident we draw latent traits (reliability, income stability, rent burden,
engagement, stay propensity), then simulate a monthly payment panel with:
  * autocorrelation (ρ·prev-trouble) so lateness clusters into streaks —
    ESSENTIAL for a learnable, honest task (not month-independent noise);
  * transient burden shocks (1–3 month spikes);
  * mild December/January seasonality.
Each month maps a trouble log-odds -> payment fraction -> status
(paid / paid_late / partial / missed), accumulating arrears into balance_after.

Leakage safety (non-negotiable)
--------------------------------
We simulate HISTORY_MONTHS months of history ending the month BEFORE the
snapshot, PLUS LABEL_HORIZON_MONTHS future months. The four labels are computed
ONLY from that future window, then the future is DISCARDED — the serialized
ledger holds exactly HISTORY_MONTHS entries, none dated on/after the snapshot.
Labels are never written into residents.json; ``build_training_frame`` returns
them separately for train_residents / residents_eval.

Global intercepts are bisected (like train_risk) so the label base rates land
on target: late ≈ 0.30, serious ≈ 0.10, churn ≈ 0.35 (eligible). ~3% of binary
labels are flipped so the task is learnable but NOT trivially separable.

Run:  python generate_residents.py   |   python -m generate_residents
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import residents_risk as rr  # noqa: E402
from settings import DATA_DIR  # noqa: E402

SEED = rr.SEED
RESIDENTS_PER_PROPERTY = 25
DATA_PATH = DATA_DIR / "residents.json"

# Monthly trouble log-odds weights (direction mirrors the feature policy).
# These are deliberately large relative to the per-month Bernoulli noise so a
# resident's realized 60-month history is a strong estimator of their future
# propensity — the whole point of keeping 5 years of ledger.
_W_REL = 4.4        # low reliability -> more trouble
_W_BURDEN = 6.5     # rent burden above 0.30 -> more trouble
_W_UNSTABLE = 2.2   # unstable income -> more trouble
_RHO = 1.15         # autocorrelation on previous-month trouble (streaks)
_SEASONAL = 0.12    # small Dec/Jan bump

# Severity weights (given a trouble month, how bad it is) — trait-driven so
# past severity predicts future severity.
_WS_REL = 3.2
_WS_BURDEN = 4.2

# Churn (non-renewal) log-odds weights among horizon-eligible leases. Leans on
# OBSERVABLE factors (burden, engagement, recent lateness, renewal offer) so the
# label is learnable from features, with a smaller latent stay-propensity part.
_CH_STAY = 1.1      # low stay propensity -> churn (small latent part)
_CH_BURDEN = 4.2    # rent burden -> churn (observable via rent_to_income)
_CH_ENGAGE = 1.9    # low engagement -> churn (observable via logins/complaints)
_CH_LATE = 0.32     # recent lateness nudges churn (observable)
_CH_RENEWAL = 0.6   # a renewal offer reduces churn
_CH_RENEWALS = 0.22  # prior renewals (loyalty) reduce churn

_FLIP_RATE = 0.03
_TARGET_LATE = 0.30
_TARGET_SERIOUS = 0.10
_TARGET_CHURN = 0.35

LATE_FEE = 60.0  # flat late fee applied on any non-"paid" month


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


# --------------------------------------------------------------------------
# Static (intercept-independent) inputs
# --------------------------------------------------------------------------
def _build_meta(snapshot: date, n_per_property: int = RESIDENTS_PER_PROPERTY) -> list:
    """One dict of per-resident latent traits + immutable facts + fixed RNG
    draws. Everything that does NOT depend on the bisected intercepts, so the
    intercept search can re-simulate deterministically.

    ``n_per_property`` defaults to 25 (the committed residents.json). Training
    and eval pass larger cohorts of the SAME DGP for stable metrics; per-resident
    RNG is anchored on ``SEED + idx`` so the first 25 units of the first property
    always reproduce the committed residents."""
    import graph

    props = {p["id"]: p for p in graph.load_properties()}
    # History ends the month BEFORE the snapshot (all history strictly precedes
    # the snapshot -> airtight leakage guard). Each resident's first ledger month
    # is computed per-resident from their own tenure below.

    metas = []
    idx = 0
    for pid in rr.RESIDENT_PROPERTY_IDS:
        prop = props[pid]
        prop_rent = float(prop["monthly_rent"])
        min_mult = float(prop.get("min_income_multiplier", 2.5))
        base_beds = int(prop.get("bedrooms", 1))
        for u in range(n_per_property):
            rng = np.random.default_rng(SEED + idx)

            # Wider trait spread => more separable residents (higher, honest AUC).
            reliability = float(rng.beta(3.5, 2.0))
            income_stability = float(rng.beta(3.5, 2.0))
            engagement = float(rng.beta(3.0, 3.0))
            # Income drawn from the property's income multiplier -> rent burden.
            income_multiple = float(np.clip(rng.normal(min_mult + 0.6, 0.9), 1.4, 8.0))
            base_rent = round(prop_rent * float(rng.uniform(0.96, 1.06)), 2)
            monthly_income = round(base_rent * income_multiple, 2)
            other_income = round(float(max(0.0, rng.normal(150, 200))), 2)
            rent_burden = base_rent / (monthly_income + other_income)
            stay_indep = float(rng.beta(3.0, 3.0))
            stay_propensity = float(np.clip(
                0.5 * reliability + 0.3 * (1.0 - min(rent_burden / 0.5, 1.0)) + 0.2 * stay_indep,
                0.0, 1.0,
            ))

            # TRUE tenure (months since move-in): a realistic spread from
            # brand-new (~2-3 months) up to long-term (capped ~96 months). The
            # serialized ledger holds only the resident's ACTUAL months, capped
            # at HISTORY_MONTHS — so a 10-month resident has a 10-entry ledger
            # and a 7-year resident keeps the most recent 60.
            true_tenure = int(np.clip(round(rng.gamma(2.2, 22.0)), 2, 96))
            hist_len = min(true_tenure, rr.HISTORY_MONTHS)   # == serialized ledger length
            gen_months = hist_len + rr.LABEL_HORIZON_MONTHS  # +3 future months for labels
            lease_term = int(rng.choice([12, 12, 12, 6, 18, 24]))
            move_in = _add_months(snapshot, -true_tenure)
            # Current lease end: next term boundary strictly after the snapshot.
            renewals = 0
            lease_end = _add_months(move_in, lease_term)
            while lease_end <= snapshot:
                lease_end = _add_months(lease_end, lease_term)
                renewals += 1
            months_to_lease_end = rr._months_between(snapshot, lease_end)
            renewal_offer_sent = bool(months_to_lease_end <= 3 and rng.random() < 0.7)

            autopay = bool(rng.random() < (0.35 + 0.4 * reliability))
            income_verified = bool(rng.random() < 0.85)
            maint = int(np.clip(rng.poisson(1.5 + 2.0 * engagement), 0, 30))
            complaints = int(np.clip(rng.poisson(0.4 + 1.5 * (1.0 - engagement)), 0, 20))
            logins = int(np.clip(rng.poisson(2 + 18 * engagement), 0, 120))
            unit_beds = int(np.clip(base_beds + rng.integers(-1, 2), 0, 5))

            # Fixed per-month random draws over the resident's own panel length
            # (hist_len history + 3 future). The intercept bisection stays
            # deterministic in the intercepts, with feedback through ρ.
            u_trouble = rng.random(gen_months)
            u_sev = rng.random(gen_months)
            u_kind = rng.random(gen_months)      # partial vs missed split
            frac_draw = rng.uniform(0.2, 0.8, gen_months)
            days_mild = rng.gamma(2.0, 5.0, gen_months)   # paid_late days
            days_severe = rng.gamma(3.0, 12.0, gen_months)
            catchup_draw = rng.uniform(0.3, 0.7, gen_months)
            notice_resp = rng.random(gen_months)
            # Shocks: additive burden spikes lasting 1–3 months.
            shock = np.zeros(gen_months)
            m = 0
            while m < gen_months:
                if rng.random() < 0.04:
                    dur = int(rng.integers(1, 4))
                    mag = float(rng.uniform(0.8, 1.8))
                    for k in range(m, min(m + dur, gen_months)):
                        shock[k] += mag
                    m += dur
                else:
                    m += 1
            u_churn = float(rng.random())
            u_flip = rng.random(3)
            # History is a contiguous run of hist_len months ENDING at the
            # snapshot month; the 3 label months follow it (t+1..t+3) and are
            # never serialized. First ledger period = snapshot - (hist_len - 1).
            first_month = _add_months(snapshot, -(hist_len - 1))

            metas.append({
                "idx": idx,
                "resident_id": f"RES-{idx + 1:04d}",
                "property_id": pid,
                "unit_id": f"{pid}-U{u + 1:03d}",
                "unit_bedrooms": unit_beds,
                "base_rent": base_rent,
                "monthly_income": monthly_income,
                "other_income_monthly": other_income,
                "income_verified": income_verified,
                "autopay_enrolled": autopay,
                "deposit_held": base_rent,
                "lease_term_months": lease_term,
                "prior_renewals": renewals,
                "renewal_offer_sent": renewal_offer_sent,
                "move_in_date": move_in.isoformat(),
                "lease_start": move_in.isoformat(),
                "lease_end": lease_end.isoformat(),
                "months_to_lease_end": months_to_lease_end,
                "true_tenure": true_tenure,
                "hist_len": hist_len,
                "gen_months": gen_months,
                "maintenance_requests_12mo": maint,
                "complaints_12mo": complaints,
                "portal_logins_90d": logins,
                "first_month": first_month,
                # latent + fixed draws
                "reliability": reliability,
                "income_stability": income_stability,
                "engagement": engagement,
                "rent_burden": rent_burden,
                "stay_propensity": stay_propensity,
                "u_trouble": u_trouble,
                "u_sev": u_sev,
                "u_kind": u_kind,
                "frac_draw": frac_draw,
                "days_mild": days_mild,
                "days_severe": days_severe,
                "catchup_draw": catchup_draw,
                "notice_resp": notice_resp,
                "shock": shock,
                "u_churn": u_churn,
                "u_flip": u_flip,
            })
            idx += 1
    return metas


# --------------------------------------------------------------------------
# Panel simulation (deterministic given the intercepts + fixed draws)
# --------------------------------------------------------------------------
def _simulate(meta: dict, c_late: float, c_sev: float) -> dict:
    """Simulate the resident's own panel (hist_len history + 3 future months).
    Returns the serialized ledger (exactly hist_len entries) plus raw
    future-window quantities for label computation. The 3 future months are
    ALWAYS simulated (labels need them) but never serialized."""
    base_rent = meta["base_rent"]
    rel = meta["reliability"]
    burden = meta["rent_burden"]
    unstable = 1.0 - meta["income_stability"]
    shock = meta["shock"]
    first_month = meta["first_month"]
    hist_len = meta["hist_len"]
    gen_months = meta["gen_months"]
    snapshot = _add_months(first_month, hist_len - 1)  # last history month == snapshot month

    ledger = []
    balance = 0.0
    prev_trouble = 0
    future = []  # (status, days_late, balance_after) for the label window

    for m in range(gen_months):
        period_date = _add_months(first_month, m)
        seasonal = _SEASONAL if period_date.month in (12, 1) else 0.0
        z = (
            c_late
            + _W_REL * (1.0 - rel)
            + _W_BURDEN * (burden - 0.30)
            + _W_UNSTABLE * unstable
            + _RHO * prev_trouble
            + seasonal
            + shock[m]
        )
        p_trouble = _sigmoid(z)
        trouble = meta["u_trouble"][m] < p_trouble

        rent_charged = base_rent
        if not trouble:
            prev_trouble = 0
            status = "paid"
            days_late = 0
            late_fee = 0.0
            catchup = min(balance, meta["catchup_draw"][m] * base_rent)
            amount_paid = rent_charged + catchup
            balance = max(0.0, balance + rent_charged - amount_paid)
        else:
            prev_trouble = 1
            late_fee = LATE_FEE
            z_sev = (
                c_sev
                + _WS_REL * (1.0 - rel)
                + _WS_BURDEN * (burden - 0.30)
                + 0.4 * _RHO * prev_trouble
                + shock[m]
            )
            severe = meta["u_sev"][m] < _sigmoid(z_sev)
            if not severe:
                status = "paid_late"
                days_late = int(np.clip(round(1 + meta["days_mild"][m]), 1, 29))
                amount_paid = rent_charged  # full rent, just late (fee accrues)
            elif meta["u_kind"][m] < 0.5:
                status = "partial"
                days_late = int(np.clip(round(5 + meta["days_severe"][m]), 5, 90))
                amount_paid = round(meta["frac_draw"][m] * rent_charged, 2)
            else:
                status = "missed"
                days_late = 30  # nothing paid this cycle -> 30+ days delinquent
                amount_paid = 0.0
            balance = max(0.0, balance + rent_charged + late_fee - amount_paid)

        on_time = status == "paid"
        # Notices when the month is troubled; response ~ engagement.
        notices_sent = 0 if on_time else int(1 + (1 if status == "missed" else 0))
        notice_responded = 0
        if notices_sent and meta["notice_resp"][m] < meta["engagement"]:
            notice_responded = 1

        paid_date = None
        if amount_paid > 0:
            # 1st of month + days_late. A late payment on a very recent month can
            # settle on/after the snapshot; as of the snapshot that payment is
            # still clearing, so we leave paid_date None (guarantees no ledger
            # entry is dated on/after the snapshot — the leakage guard).
            from datetime import timedelta

            settled = period_date + timedelta(days=days_late)
            paid_date = settled.isoformat() if settled <= snapshot else None

        entry = {
            "period": f"{period_date.year:04d}-{period_date.month:02d}",
            "rent_charged": round(rent_charged, 2),
            "amount_paid": round(amount_paid, 2),
            "paid_date": paid_date,
            "days_late": int(days_late),
            "late_fee": round(late_fee, 2),
            "on_time": bool(on_time),
            "status": status,
            "balance_after": round(balance, 2),
            "notices_sent": int(notices_sent),
            "notice_responded": int(notice_responded),
        }
        if m < hist_len:
            ledger.append(entry)
        else:
            future.append({"status": status, "days_late": int(days_late), "balance_after": round(balance, 2)})

    return {"ledger": ledger, "future": future, "base_rent": base_rent}


# --------------------------------------------------------------------------
# Labels from the (discarded) future window ONLY
# --------------------------------------------------------------------------
def _labels(sim: dict, meta: dict, c_churn: float) -> dict:
    future = sim["future"]
    base_rent = sim["base_rent"]
    late = int(any(f["status"] != "paid" for f in future))
    serious = int(any(f["days_late"] >= 30 or f["balance_after"] >= base_rent for f in future))
    arrears = float(future[-1]["balance_after"]) if future else 0.0

    eligible = 0 < meta["months_to_lease_end"] <= rr.CHURN_HORIZON_MONTHS
    churn = None
    if eligible:
        # Recent lateness comes from HISTORY only (no leakage): last 12 months.
        recent_late = sum(1 for e in sim["ledger"][-12:] if e["status"] != "paid")
        z = (
            c_churn
            + _CH_STAY * (0.5 - meta["stay_propensity"]) * 2.0
            + _CH_BURDEN * (meta["rent_burden"] - 0.30)
            + _CH_ENGAGE * (0.5 - meta["engagement"]) * 2.0
            + _CH_LATE * recent_late
            - _CH_RENEWAL * (1.0 if meta["renewal_offer_sent"] else 0.0)
            - _CH_RENEWALS * meta["prior_renewals"]
        )
        churn = int(meta["u_churn"] < _sigmoid(z))
    return {"late": late, "serious": serious, "arrears": round(arrears, 2), "churn": churn}


def _bisect(fn, target, lo=-12.0, hi=12.0, iters=50):
    """Find intercept c so mean(fn(c)) ≈ target (fn increasing in c)."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if fn(mid) > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------
# Public generation API
# --------------------------------------------------------------------------
_INTERCEPT_CACHE: dict = {}
_CAL_PER_PROPERTY = 40  # calibration cohort size per property for intercept fit


def _fit_intercepts(snapshot: date) -> tuple:
    """Bisect the three global intercepts (late / severity / churn) so the label
    base rates hit target. These are DGP-level constants — fit ONCE on a fixed
    calibration cohort at SEED and reused for every cohort/seed, so generating a
    large training cohort does not re-run the (expensive) bisection."""
    key = snapshot.isoformat()
    if key in _INTERCEPT_CACHE:
        return _INTERCEPT_CACHE[key]
    global SEED
    prev = SEED
    SEED = rr.SEED
    try:
        metas = _build_meta(snapshot, n_per_property=_CAL_PER_PROPERTY)
    finally:
        SEED = prev

    def late_rate(c):
        return float(np.mean([_labels(_simulate(mt, c, 0.0), mt, 0.0)["late"] for mt in metas]))

    c_late = _bisect(late_rate, _TARGET_LATE)

    def serious_rate(c):
        return float(np.mean([_labels(_simulate(mt, c_late, c), mt, 0.0)["serious"] for mt in metas]))

    c_sev = _bisect(serious_rate, _TARGET_SERIOUS)
    sims = [_simulate(mt, c_late, c_sev) for mt in metas]

    def churn_rate(c):
        vals = [_labels(s, mt, c)["churn"] for s, mt in zip(sims, metas)]
        elig = [v for v in vals if v is not None]
        return float(np.mean(elig)) if elig else 0.0

    c_churn = _bisect(churn_rate, _TARGET_CHURN)
    _INTERCEPT_CACHE[key] = (c_late, c_sev, c_churn)
    return _INTERCEPT_CACHE[key]


def generate(seed: int = SEED, snapshot: date = rr.RESIDENT_SNAPSHOT,
             n_per_property: int = RESIDENTS_PER_PROPERTY):
    """Return (residents, labels) fully deterministically.

    ``residents`` are serialized dicts (60-month ledger + immutable facts, ready
    for residents.json). ``labels`` is an aligned list of
    {late, serious, arrears, churn} used only for training/eval (never written).
    ``seed`` anchors every per-resident RNG stream (``seed + idx``); the eval
    holdout uses ``seed + 1000`` for a fully disjoint cohort."""
    c_late, c_sev, c_churn = _fit_intercepts(snapshot)

    global SEED
    prev_seed = SEED
    SEED = seed
    try:
        metas = _build_meta(snapshot, n_per_property=n_per_property)
    finally:
        SEED = prev_seed

    sims = [_simulate(mt, c_late, c_sev) for mt in metas]

    residents, labels = [], []
    for sim, mt in zip(sims, metas):
        lab = _labels(sim, mt, c_churn)
        # ~3% label flip on the binaries (learnable, not trivially separable).
        flip = mt["u_flip"] < _FLIP_RATE
        if flip[0]:
            lab["late"] = 1 - lab["late"]
        if flip[1]:
            lab["serious"] = 1 - lab["serious"]
        if flip[2] and lab["churn"] is not None:
            lab["churn"] = 1 - lab["churn"]

        residents.append({
            "resident_id": mt["resident_id"],
            "property_id": mt["property_id"],
            "unit_id": mt["unit_id"],
            "unit_bedrooms": mt["unit_bedrooms"],
            "base_rent": mt["base_rent"],
            "lease_start": mt["lease_start"],
            "lease_end": mt["lease_end"],
            "lease_term_months": mt["lease_term_months"],
            "renewal_offer_sent": mt["renewal_offer_sent"],
            "autopay_enrolled": mt["autopay_enrolled"],
            "deposit_held": mt["deposit_held"],
            "move_in_date": mt["move_in_date"],
            "prior_renewals": mt["prior_renewals"],
            "monthly_income": mt["monthly_income"],
            "other_income_monthly": mt["other_income_monthly"],
            "income_verified": mt["income_verified"],
            "maintenance_requests_12mo": mt["maintenance_requests_12mo"],
            "complaints_12mo": mt["complaints_12mo"],
            "portal_logins_90d": mt["portal_logins_90d"],
            "ledger": sim["ledger"],
            "dgp_version": rr.DGP_VERSION,
        })
        labels.append(lab)
    return residents, labels


def build_training_frame(seed: int = SEED, snapshot: date = rr.RESIDENT_SNAPSHOT,
                         n_per_property: int = 100):
    """(residents, labels) for training/eval. Labels are kept OUT of
    residents.json. Defaults to a 100/property (~1000 resident) cohort of the
    same DGP so the four estimators have enough data for stable metrics."""
    return generate(seed=seed, snapshot=snapshot, n_per_property=n_per_property)


def write_dataset(seed: int = SEED, snapshot: date = rr.RESIDENT_SNAPSHOT) -> dict:
    """Generate and write data/residents.json (residents only; no labels)."""
    residents, _labels = generate(seed=seed, snapshot=snapshot)
    generated_at = snapshot.isoformat() + "T00:00:00+00:00"  # from snapshot, NOT now()
    payload = {
        "meta": {
            "dgp_version": rr.DGP_VERSION,
            "generated_at": generated_at,
            "snapshot": snapshot.isoformat(),
            "property_ids": list(rr.RESIDENT_PROPERTY_IDS),
            "seed": seed,
        },
        "residents": residents,
    }
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_text(json.dumps(payload, indent=2))
    return {"path": str(DATA_PATH), "residents": len(residents), "properties": len(rr.RESIDENT_PROPERTY_IDS)}


def _base_rates(labels: list) -> dict:
    late = [l["late"] for l in labels]
    serious = [l["serious"] for l in labels]
    arrears = [l["arrears"] for l in labels]
    churn = [l["churn"] for l in labels if l["churn"] is not None]
    return {
        "late": round(float(np.mean(late)), 4),
        "serious": round(float(np.mean(serious)), 4),
        "arrears_mean": round(float(np.mean(arrears)), 2),
        "arrears_nonzero_frac": round(float(np.mean([a > 0 for a in arrears])), 4),
        "churn_eligible": len(churn),
        "churn": round(float(np.mean(churn)), 4) if churn else 0.0,
    }


def _verify(residents: list, snapshot: date) -> dict:
    """Assert the committed dataset's invariants. Returns a small summary."""
    snap_month = (snapshot.year, snapshot.month)
    tenures = []
    n_after = 0
    n_bad_len = 0
    n_bad_contig = 0
    n_bad_end = 0
    for r in residents:
        led = r["ledger"]
        tenure = rr._months_between(rr._to_date(r["move_in_date"]), snapshot)
        tenures.append(tenure)
        # Invariant 1: ledger length == min(tenure, HISTORY_MONTHS).
        if len(led) != min(tenure, rr.HISTORY_MONTHS):
            n_bad_len += 1
        # Parse periods.
        months = [tuple(int(x) for x in e["period"].split("-")) for e in led]
        # Invariant 2: contiguous months.
        for a, b in zip(months, months[1:]):
            if _add_months(date(a[0], a[1], 1), 1) != date(b[0], b[1], 1):
                n_bad_contig += 1
                break
        # Invariant 3: last period == snapshot month.
        if months and months[-1] != snap_month:
            n_bad_end += 1
        # Leakage guard: no entry dated AFTER the snapshot (period or paid_date).
        for e in led:
            y, m = (int(x) for x in e["period"].split("-"))
            if date(y, m, 1) > snapshot:
                n_after += 1
            if e["paid_date"] and rr._to_date(e["paid_date"]) > snapshot:
                n_after += 1
    tenures.sort()
    assert n_bad_len == 0, f"{n_bad_len} residents violate len(ledger)==min(tenure,60)"
    assert n_bad_contig == 0, f"{n_bad_contig} residents have non-contiguous ledgers"
    assert n_bad_end == 0, f"{n_bad_end} residents do not end at the snapshot month"
    assert n_after == 0, f"LEAKAGE: {n_after} ledger entries dated after snapshot {snapshot}"
    mid = tenures[len(tenures) // 2]
    return {
        "tenure_min": tenures[0],
        "tenure_median": mid,
        "tenure_max": tenures[-1],
        "count_tenure_lt_60": sum(1 for t in tenures if t < rr.HISTORY_MONTHS),
        "count_tenure_ge_60": sum(1 for t in tenures if t >= rr.HISTORY_MONTHS),
        "ledger_len_range": (min(len(r["ledger"]) for r in residents), max(len(r["ledger"]) for r in residents)),
    }


if __name__ == "__main__":
    info = write_dataset()
    snap = rr.RESIDENT_SNAPSHOT
    committed = json.loads(DATA_PATH.read_text())["residents"]
    summary = _verify(committed, snap)
    _residents, labels = build_training_frame()  # larger cohort for base-rate sanity
    print(f"Wrote {info['path']}")
    print(f"residents={info['residents']} properties={info['properties']}")
    print(f"invariants: PASS  len(ledger)==min(tenure,60) for all; contiguous; end at snapshot month {snap}")
    print(f"leakage_guard: PASS (0 entries dated after snapshot {snap})")
    print(f"tenure distribution: min={summary['tenure_min']} median={summary['tenure_median']} "
          f"max={summary['tenure_max']} | tenure<60: {summary['count_tenure_lt_60']} "
          f"tenure>=60: {summary['count_tenure_ge_60']} | ledger_len_range={summary['ledger_len_range']}")
    print(f"label base rates: {_base_rates(labels)}")
