"""Generate the SYNTHETIC resident dataset — 5-year rent ledgers + labels (v2).

Deterministic (seed=42) and snapshot-relative (default RESIDENT_SNAPSHOT; NO
``datetime.now()`` anywhere), so re-running reproduces data/residents.json byte
for byte. ~250 residents across 10 frozen properties, each with a rent ledger of
up to 60 months.

Data-generating process
------------------------
Per resident we draw latent traits (reliability, income stability, rent burden,
engagement, stay propensity), then simulate a monthly payment panel with:
  * autocorrelation (rho·prev-trouble) so lateness clusters into streaks;
  * transient burden shocks (1-3 month spikes);
  * mild December/January seasonality.
Each month maps a trouble log-odds -> payment fraction -> status
(paid / paid_late / partial / missed), accumulating arrears into balance_after.

Leakage safety + determinism (non-negotiable) — v2 substream split
------------------------------------------------------------------
HISTORY is drawn from the per-resident stream ``rng = default_rng(SEED+idx)``,
EXACTLY as in v1 (same draw order and sizes), so the serialized ledger is
BYTE-IDENTICAL to the committed v1 dataset. The FUTURE label window is drawn from
an INDEPENDENT substream ``default_rng(SeedSequence([SEED+idx, 0xF0]))`` and
extended to ``FUTURE_HORIZON_MONTHS`` (15) — long enough for clean 1/3/6/12-month
labels. Because the future uses its own generator, lengthening it never perturbs
the history bytes. The global intercepts (c_late / c_sev / c_churn) are fit on
the v1-compatible 3-month tail of the HISTORY stream, so they equal v1's values
and the history simulation (which depends on c_late/c_sev) is unchanged.

The serialized ledger holds exactly ``hist_len`` entries, none dated on/after the
snapshot. Labels are computed ONLY from the discarded future window and are never
written into residents.json; ``build_training_frame`` returns them separately.

Global intercepts are bisected so the legacy label base rates land on target:
late ~ 0.30, serious ~ 0.10, churn ~ 0.35 (eligible). ~3% of BINARY head labels
are flipped so the tasks are learnable but not trivially separable.

Run:  python generate_residents.py   |   python -m generate_residents
"""

import json
import sys
from datetime import date, timedelta
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
FUTURE_SEQ_TAG = 0xF0  # SeedSequence tag for the independent future substream

# Monthly trouble log-odds weights (direction mirrors the feature policy).
_W_REL = 4.4        # low reliability -> more trouble
_W_BURDEN = 6.5     # rent burden above 0.30 -> more trouble
_W_UNSTABLE = 2.2   # unstable income -> more trouble
_RHO = 1.15         # autocorrelation on previous-month trouble (streaks)
_SEASONAL = 0.12    # small Dec/Jan bump
_STREAK_DECAY = 0.85  # per-month decay of the FUTURE window's recent-trouble
                      # memory signal (mirrors the 0.85 decay already used by
                      # the recency_weighted_lateness feature, for consistency)

# Severity weights (given a trouble month, how bad it is).
_WS_REL = 3.2
_WS_BURDEN = 4.2

# Churn (non-renewal) log-odds weights among horizon-eligible leases.
_CH_STAY = 1.1
_CH_BURDEN = 4.2
_CH_ENGAGE = 1.9
_CH_LATE = 0.32
_CH_RENEWAL = 0.6
_CH_RENEWALS = 0.22

_FLIP_RATE = 0.03
_TARGET_LATE = 0.30
_TARGET_SERIOUS = 0.10
_TARGET_CHURN = 0.35

LATE_FEE = 60.0  # flat late fee applied on any non-"paid" month

FUTURE_MONTHS = rr.FUTURE_HORIZON_MONTHS  # 15
CURE_EPS = rr.CURE_EPS

# Binary head names (labels these get the 3% flip). Derived from the registry.
_BINARY_HEADS = [h["name"] for h in rr.HEADS if h["kind"] == "binary"]

# Deterministic gender-neutral-ish display-name pools (DISPLAY ONLY — never a
# feature; see EXCLUDED_FEATURES). Indexed by resident number for determinism.
_FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery",
    "Quinn", "Skyler", "Cameron", "Reese", "Devon", "Harper", "Rowan", "Emerson",
    "Finley", "Sage", "Dakota", "Hayden", "Kendall", "Logan", "Parker", "Elliot",
    "Marlowe", "Blake", "Charlie", "Frankie", "Sam", "Toni", "Robin", "Drew",
    "Lane", "Micah", "Noel", "Shawn", "Adrian", "Bailey", "Corey", "Dana",
]
_LAST_NAMES = [
    "Rivera", "Nguyen", "Patel", "Kim", "Garcia", "Okafor", "Silva", "Cohen",
    "Alvarez", "Haddad", "Ivanov", "Yamamoto", "Mbeki", "Rossi", "Novak", "Costa",
    "Reyes", "Flores", "Bauer", "Larsen", "Duarte", "Sato", "Abbas", "Weber",
    "Mensah", "Petrov", "Khan", "Torres", "Bianchi", "Vargas", "Ito", "Sharma",
    "Diaz", "Fischer", "Moreno", "Park", "Osei", "Romano", "Serrano", "Wan",
]


def _display_name(idx: int) -> str:
    first = _FIRST_NAMES[idx % len(_FIRST_NAMES)]
    last = _LAST_NAMES[(idx // len(_FIRST_NAMES) + idx * 7) % len(_LAST_NAMES)]
    return f"{first} {last}"


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
    draws. HISTORY draws use ``rng = default_rng(SEED+idx)`` in the EXACT v1 order
    and sizes (byte-identical ledgers). FUTURE draws use an INDEPENDENT substream
    so extending the future window never perturbs history.

    ``n_per_property`` defaults to 25 (the committed residents.json)."""
    import graph

    props = {p["id"]: p for p in graph.load_properties()}

    metas = []
    idx = 0
    for pid in rr.RESIDENT_PROPERTY_IDS:
        prop = props[pid]
        prop_rent = float(prop["monthly_rent"])
        min_mult = float(prop.get("min_income_multiplier", 2.5))
        base_beds = int(prop.get("bedrooms", 1))
        for u in range(n_per_property):
            rng = np.random.default_rng(SEED + idx)

            # ---- v1 HISTORY stream (draw order + sizes preserved verbatim) ----
            reliability = float(rng.beta(3.5, 2.0))
            income_stability = float(rng.beta(3.5, 2.0))
            engagement = float(rng.beta(3.0, 3.0))
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

            true_tenure = int(np.clip(round(rng.gamma(2.2, 22.0)), 2, 96))
            hist_len = min(true_tenure, rr.HISTORY_MONTHS)   # == serialized ledger length
            gen_months = hist_len + rr.LABEL_HORIZON_MONTHS  # v1 stream length (history + 3-mo tail)
            lease_term = int(rng.choice([12, 12, 12, 6, 18, 24]))
            move_in = _add_months(snapshot, -true_tenure)
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

            # v1 per-month arrays at size gen_months (history uses [:hist_len];
            # the 3-mo tail feeds ONLY the v1-compatible intercept bisection).
            u_trouble = rng.random(gen_months)
            u_sev = rng.random(gen_months)
            u_kind = rng.random(gen_months)
            frac_draw = rng.uniform(0.2, 0.8, gen_months)
            days_mild = rng.gamma(2.0, 5.0, gen_months)
            days_severe = rng.gamma(3.0, 12.0, gen_months)
            catchup_draw = rng.uniform(0.3, 0.7, gen_months)
            notice_resp = rng.random(gen_months)
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
            u_flip = rng.random(3)  # v1 draw kept so the rng stream is unchanged
            # ---- end v1 HISTORY stream (rng untouched beyond this point) ------

            first_month = _add_months(snapshot, -(hist_len - 1))

            # ---- INDEPENDENT future substream (never perturbs history) --------
            frng = np.random.default_rng(np.random.SeedSequence([SEED + idx, FUTURE_SEQ_TAG]))
            f_u_trouble = frng.random(FUTURE_MONTHS)
            f_u_sev = frng.random(FUTURE_MONTHS)
            f_u_kind = frng.random(FUTURE_MONTHS)
            f_frac = frng.uniform(0.2, 0.8, FUTURE_MONTHS)
            f_days_mild = frng.gamma(2.0, 5.0, FUTURE_MONTHS)
            f_days_severe = frng.gamma(3.0, 12.0, FUTURE_MONTHS)
            f_catchup = frng.uniform(0.3, 0.7, FUTURE_MONTHS)
            f_shock = np.zeros(FUTURE_MONTHS)
            m = 0
            while m < FUTURE_MONTHS:
                if frng.random() < 0.04:
                    dur = int(frng.integers(1, 4))
                    mag = float(frng.uniform(0.8, 1.8))
                    for k in range(m, min(m + dur, FUTURE_MONTHS)):
                        f_shock[k] += mag
                    m += dur
                else:
                    m += 1
            # Binary-head label flips, drawn LAST from the future substream.
            u_flip_v2 = frng.random(len(_BINARY_HEADS))

            metas.append({
                "idx": idx,
                "resident_id": f"RES-{idx + 1:04d}",
                "name": _display_name(idx),
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
                # future substream draws (independent of history)
                "f_u_trouble": f_u_trouble,
                "f_u_sev": f_u_sev,
                "f_u_kind": f_u_kind,
                "f_frac": f_frac,
                "f_days_mild": f_days_mild,
                "f_days_severe": f_days_severe,
                "f_catchup": f_catchup,
                "f_shock": f_shock,
                "u_flip_v2": u_flip_v2,
            })
            idx += 1
    return metas


# --------------------------------------------------------------------------
# Panel simulation
# --------------------------------------------------------------------------
def _simulate(meta: dict, c_late: float, c_sev: float) -> dict:
    """Simulate the resident's history (byte-identical to v1) plus BOTH futures:
      * ``future_v1`` — the v1 3-month tail from the HISTORY stream, used ONLY to
        reproduce v1's intercept bisection so history bytes are unchanged;
      * ``future`` — the v2 15-month window from the INDEPENDENT substream, used
        for all v2 head labels. Continues balance + prev_trouble from history end.
    Returns the serialized ledger (exactly hist_len entries) + both futures."""
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
    future_v1 = []
    bal_hist_end = 0.0
    prev_hist_end = 0

    # ---- v1 loop (history + 3-mo tail): identical to v1, byte-for-byte -------
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
                amount_paid = rent_charged
            elif meta["u_kind"][m] < 0.5:
                status = "partial"
                days_late = int(np.clip(round(5 + meta["days_severe"][m]), 5, 90))
                amount_paid = round(meta["frac_draw"][m] * rent_charged, 2)
            else:
                status = "missed"
                days_late = 30
                amount_paid = 0.0
            balance = max(0.0, balance + rent_charged + late_fee - amount_paid)

        on_time = status == "paid"
        notices_sent = 0 if on_time else int(1 + (1 if status == "missed" else 0))
        notice_responded = 0
        if notices_sent and meta["notice_resp"][m] < meta["engagement"]:
            notice_responded = 1

        paid_date = None
        if amount_paid > 0:
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
            if m == hist_len - 1:
                bal_hist_end = balance
                prev_hist_end = prev_trouble
        else:
            future_v1.append({"status": status, "days_late": int(days_late), "balance_after": round(balance, 2)})

    # ---- v2 future window (independent substream), continuing history state --
    # recent_trouble_decay is a POST-HOC read of the already-generated history
    # ledger (an exponential decay over its trouble/clean months) — it does not
    # change anything about how that ledger was simulated above.
    recent_trouble_decay = _recent_trouble_decay(ledger)
    future = _simulate_future(meta, bal_hist_end, recent_trouble_decay, c_late, c_sev, snapshot)

    return {"ledger": ledger, "future_v1": future_v1, "future": future,
            "base_rent": base_rent, "current_balance": round(bal_hist_end, 2)}


def _recent_trouble_decay(ledger: list) -> float:
    """Exponential-decay scalar of trouble read from the ALREADY-GENERATED
    history ledger (chronological scan): jumps to 1.0 on a troubled month and
    decays by ``_STREAK_DECAY`` each subsequent clean month. This gives the
    FUTURE window genuine causal memory of how long ago (and how intact) the
    resident's on-time streak is, without touching the history-generation
    formula that produced the ledger (byte-identity preserved)."""
    decay = 0.0
    for e in ledger:
        if e["status"] != "paid":
            decay = 1.0
        else:
            decay *= _STREAK_DECAY
    return decay


def _simulate_future(meta: dict, balance: float, recent_trouble_decay: float,
                     c_late: float, c_sev: float, snapshot: date) -> list:
    """Simulate FUTURE_MONTHS forward months from the independent substream,
    continuing from the history-end (balance, recent_trouble_decay). Same
    monthly logic as history, except the RHO memory term uses the decayed
    recent-trouble scalar (see ``_recent_trouble_decay``) instead of a raw
    last-month binary flag, so a longer intact on-time streak carries
    genuinely less carried-over risk than a short one. Each future dict
    carries everything the head labels need."""
    base_rent = meta["base_rent"]
    rel = meta["reliability"]
    burden = meta["rent_burden"]
    unstable = 1.0 - meta["income_stability"]
    fshock = meta["f_shock"]
    future = []
    for j in range(FUTURE_MONTHS):
        period_date = _add_months(snapshot, j + 1)
        seasonal = _SEASONAL if period_date.month in (12, 1) else 0.0
        z = (
            c_late
            + _W_REL * (1.0 - rel)
            + _W_BURDEN * (burden - 0.30)
            + _W_UNSTABLE * unstable
            + _RHO * recent_trouble_decay
            + seasonal
            + fshock[j]
        )
        trouble = meta["f_u_trouble"][j] < _sigmoid(z)
        rent_charged = base_rent
        if not trouble:
            recent_trouble_decay *= _STREAK_DECAY
            status = "paid"
            days_late = 0
            late_fee = 0.0
            catchup = min(balance, meta["f_catchup"][j] * base_rent)
            amount_paid = rent_charged + catchup
            balance = max(0.0, balance + rent_charged - amount_paid)
        else:
            recent_trouble_decay = 1.0
            late_fee = LATE_FEE
            z_sev = (
                c_sev
                + _WS_REL * (1.0 - rel)
                + _WS_BURDEN * (burden - 0.30)
                + 0.4 * _RHO * recent_trouble_decay
                + fshock[j]
            )
            severe = meta["f_u_sev"][j] < _sigmoid(z_sev)
            if not severe:
                status = "paid_late"
                days_late = int(np.clip(round(1 + meta["f_days_mild"][j]), 1, 29))
                amount_paid = rent_charged
            elif meta["f_u_kind"][j] < 0.5:
                status = "partial"
                days_late = int(np.clip(round(5 + meta["f_days_severe"][j]), 5, 90))
                amount_paid = round(meta["f_frac"][j] * rent_charged, 2)
            else:
                status = "missed"
                days_late = 30
                amount_paid = 0.0
            balance = max(0.0, balance + rent_charged + late_fee - amount_paid)
        future.append({
            "status": status,
            "days_late": int(days_late),
            "balance_after": round(balance, 2),
            "amount_paid": round(amount_paid, 2),
            "rent_charged": round(rent_charged, 2),
        })
    return future


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
def _labels_v1(sim: dict, meta: dict, c_churn: float) -> dict:
    """The v1 (legacy) labels from the 3-month history tail. Used ONLY by the
    intercept bisection so c_late / c_sev / c_churn reproduce v1 exactly."""
    future = sim["future_v1"]
    base_rent = sim["base_rent"]
    late = int(any(f["status"] != "paid" for f in future))
    serious = int(any(f["days_late"] >= 30 or f["balance_after"] >= base_rent for f in future))
    arrears = float(future[-1]["balance_after"]) if future else 0.0
    churn = _churn_label(sim, meta, c_churn) if (0 < meta["months_to_lease_end"] <= rr.CHURN_HORIZON_MONTHS) else None
    return {"late": late, "serious": serious, "arrears": round(arrears, 2), "churn": churn}


def _churn_label(sim: dict, meta: dict, c_churn: float):
    """Non-renewal decision (uses HISTORY-only recent lateness; no leakage)."""
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
    return int(meta["u_churn"] < _sigmoid(z))


def _emit_labels(sim: dict, meta: dict, c_churn: float) -> dict:
    """All v2 head labels from the discarded 15-month future window. Keyed by head
    name. Eligibility-gated heads return None when ineligible."""
    fut = sim["future"]
    base_rent = sim["base_rent"]
    current_balance = sim["current_balance"]

    def any_late(k):
        return int(any(f["status"] != "paid" for f in fut[:k]))

    w12 = fut[:12]
    days12 = [f["days_late"] for f in w12]
    max_days = max(days12) if days12 else 0
    bal12 = [f["balance_after"] for f in w12]

    labels = {
        "late_1m": any_late(1),
        "late_3m": any_late(3),
        "late_6m": any_late(6),
        "late_12m": any_late(12),
        "late_count_3m": int(sum(1 for f in fut[:3] if f["status"] != "paid")),
        "late_count_12m": int(sum(1 for f in w12 if f["status"] != "paid")),
        "missed_count_12m": int(sum(1 for f in w12 if f["status"] == "missed")),
        "max_days_late_12m": float(max_days),
        "p_30d_12m": int(any(d >= 30 for d in days12)),
        "p_60d_12m": int(any(d >= 60 for d in days12)),
        "p_90d_12m": int(any(d >= 90 for d in days12)),
        "delinquency_bucket_12m": rr._delinq_bucket(max_days),
        "serious": int(any(f["days_late"] >= 30 or f["balance_after"] >= base_rent for f in fut[:3])),
        "arrears_3m": round(float(fut[2]["balance_after"]) if len(fut) >= 3 else 0.0, 2),
        "arrears_12m": round(float(fut[11]["balance_after"]) if len(fut) >= 12 else 0.0, 2),
        "peak_balance_12m": round(float(max(bal12)) if bal12 else 0.0, 2),
    }

    # Cure heads — only for residents currently carrying a balance.
    if current_balance > CURE_EPS:
        cure_month = None
        for k, f in enumerate(fut[:rr.CURE_HORIZON_MONTHS], start=1):
            if f["balance_after"] <= CURE_EPS:
                cure_month = k
                break
        labels["p_cure_6m"] = int(any(f["balance_after"] <= CURE_EPS for f in fut[:6]))
        labels["months_to_cure"] = {
            "duration": cure_month if cure_month is not None else rr.CURE_HORIZON_MONTHS,
            "event": 1 if cure_month is not None else 0,
        }
    else:
        labels["p_cure_6m"] = None
        labels["months_to_cure"] = None

    # Retention heads — eligibility by lease timing.
    mtle = meta["months_to_lease_end"]
    churn_val = _churn_label(sim, meta, c_churn)
    labels["churn"] = churn_val if (0 < mtle <= rr.CHURN_HORIZON_MONTHS) else None
    labels["churn_12m"] = churn_val if (0 < mtle <= rr.CHURN_12M_HORIZON_MONTHS) else None
    return labels


def _bisect(fn, target, lo=-12.0, hi=12.0, iters=50):
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
_CAL_PER_PROPERTY = 40


def _fit_intercepts(snapshot: date) -> tuple:
    """Bisect the three global intercepts on the v1-compatible labels so they
    equal v1's values (keeping history bytes unchanged). Fit ONCE on a fixed
    calibration cohort at SEED and reused for every cohort/seed."""
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
        return float(np.mean([_labels_v1(_simulate(mt, c, 0.0), mt, 0.0)["late"] for mt in metas]))

    c_late = _bisect(late_rate, _TARGET_LATE)

    def serious_rate(c):
        return float(np.mean([_labels_v1(_simulate(mt, c_late, c), mt, 0.0)["serious"] for mt in metas]))

    c_sev = _bisect(serious_rate, _TARGET_SERIOUS)
    sims = [_simulate(mt, c_late, c_sev) for mt in metas]

    def churn_rate(c):
        vals = [_labels_v1(s, mt, c)["churn"] for s, mt in zip(sims, metas)]
        elig = [v for v in vals if v is not None]
        return float(np.mean(elig)) if elig else 0.0

    c_churn = _bisect(churn_rate, _TARGET_CHURN)
    _INTERCEPT_CACHE[key] = (c_late, c_sev, c_churn)
    return _INTERCEPT_CACHE[key]


def generate(seed: int = SEED, snapshot: date = rr.RESIDENT_SNAPSHOT,
             n_per_property: int = RESIDENTS_PER_PROPERTY):
    """Return (residents, labels_by_head) fully deterministically.

    ``residents`` are serialized dicts (ledger + immutable facts + display name).
    ``labels_by_head`` is an aligned list of per-head label dicts used only for
    training/eval (never written). ~3% flip applied to BINARY head labels."""
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
        lab = _emit_labels(sim, mt, c_churn)
        # ~3% label flip on the BINARY heads (learnable, not trivially separable).
        for i, head in enumerate(_BINARY_HEADS):
            if mt["u_flip_v2"][i] < _FLIP_RATE and lab.get(head) is not None:
                lab[head] = 1 - lab[head]

        residents.append({
            "resident_id": mt["resident_id"],
            "name": mt["name"],
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
    """(residents, labels_by_head) for training/eval. Labels are kept OUT of
    residents.json."""
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
    def rate(key):
        vals = [l[key] for l in labels if l.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    def mean(key):
        vals = [l[key] for l in labels if l.get(key) is not None]
        return round(float(np.mean(vals)), 2) if vals else 0.0

    churn = [l["churn"] for l in labels if l["churn"] is not None]
    cure = [l["p_cure_6m"] for l in labels if l["p_cure_6m"] is not None]
    return {
        "late_1m": rate("late_1m"), "late_3m": rate("late_3m"),
        "late_6m": rate("late_6m"), "late_12m": rate("late_12m"),
        "late_count_12m_mean": mean("late_count_12m"),
        "serious": rate("serious"),
        "p_30d_12m": rate("p_30d_12m"), "p_60d_12m": rate("p_60d_12m"), "p_90d_12m": rate("p_90d_12m"),
        "arrears_3m_mean": mean("arrears_3m"), "arrears_12m_mean": mean("arrears_12m"),
        "cure_eligible": len(cure), "p_cure_6m": rate("p_cure_6m"),
        "churn_eligible": len(churn), "churn": rate("churn"),
    }


def _verify(residents: list, snapshot: date) -> dict:
    """Assert the committed dataset's invariants. Returns a small summary."""
    snap_month = (snapshot.year, snapshot.month)
    tenures = []
    n_after = n_bad_len = n_bad_contig = n_bad_end = n_no_name = 0
    for r in residents:
        led = r["ledger"]
        tenure = rr._months_between(rr._to_date(r["move_in_date"]), snapshot)
        tenures.append(tenure)
        if len(led) != min(tenure, rr.HISTORY_MONTHS):
            n_bad_len += 1
        if not r.get("name"):
            n_no_name += 1
        months = [tuple(int(x) for x in e["period"].split("-")) for e in led]
        for a, b in zip(months, months[1:]):
            if _add_months(date(a[0], a[1], 1), 1) != date(b[0], b[1], 1):
                n_bad_contig += 1
                break
        if months and months[-1] != snap_month:
            n_bad_end += 1
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
    assert n_no_name == 0, f"{n_no_name} residents missing a display name"
    mid = tenures[len(tenures) // 2]
    return {
        "tenure_min": tenures[0], "tenure_median": mid, "tenure_max": tenures[-1],
        "count_tenure_lt_60": sum(1 for t in tenures if t < rr.HISTORY_MONTHS),
        "count_tenure_ge_60": sum(1 for t in tenures if t >= rr.HISTORY_MONTHS),
        "ledger_len_range": (min(len(r["ledger"]) for r in residents), max(len(r["ledger"]) for r in residents)),
    }


def ledger_signature(residents: list) -> str:
    """Stable sha256 over just the ledgers — the byte-identity check vs v1."""
    import hashlib

    led = [r["ledger"] for r in residents]
    return hashlib.sha256(json.dumps(led, sort_keys=True).encode()).hexdigest()


if __name__ == "__main__":
    info = write_dataset()
    snap = rr.RESIDENT_SNAPSHOT
    committed = json.loads(DATA_PATH.read_text())["residents"]
    summary = _verify(committed, snap)
    _residents, labels = build_training_frame()  # larger cohort for base-rate sanity
    print(f"Wrote {info['path']}")
    print(f"residents={info['residents']} properties={info['properties']}")
    print(f"invariants: PASS  len(ledger)==min(tenure,60) for all; contiguous; end at snapshot month {snap}; names present")
    print(f"leakage_guard: PASS (0 entries dated after snapshot {snap})")
    print(f"tenure distribution: min={summary['tenure_min']} median={summary['tenure_median']} "
          f"max={summary['tenure_max']} | tenure<60: {summary['count_tenure_lt_60']} "
          f"tenure>=60: {summary['count_tenure_ge_60']} | ledger_len_range={summary['ledger_len_range']}")
    print(f"ledger_signature: {ledger_signature(committed)}")
    print(f"label base rates: {_base_rates(labels)}")
