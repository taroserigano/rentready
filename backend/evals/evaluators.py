"""Deterministic, offline evaluators.

These need no LLM and no Neo4j (they use the in-memory graph + the heuristic
extractor + the deterministic eligibility rules and scorer), so they run fast
and reproducibly in CI. Each returns a metrics dict.
"""

import math
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import eligibility  # noqa: E402
import graph  # noqa: E402
import pdf_ingest  # noqa: E402
import rag_llamaindex  # noqa: E402
import signals  # noqa: E402
from models import ApplicantProfile  # noqa: E402

APPLICATIONS_DIR = BACKEND.parent / "data" / "applications"


# ----------------------------- Eligibility --------------------------------
def eval_eligibility(rows: list) -> dict:
    """Verdict exact-match accuracy + a 3-class confusion matrix."""
    labels = ["qualified", "needs_review", "not_qualified"]
    confusion = {a: {b: 0 for b in labels} for a in labels}
    correct = 0
    per_case = []
    for row in rows:
        profile = ApplicantProfile(**row["profile"])
        result = eligibility.evaluate(profile, explain=False)
        gold = row["expected_verdict"]
        ok = result.verdict == gold
        correct += ok
        confusion[gold][result.verdict] += 1
        per_case.append(
            {"id": row["id"], "expected": gold, "got": result.verdict, "ok": ok}
        )
    n = len(rows)
    return {
        "accuracy": round(correct / n, 4) if n else 0.0,
        "n": n,
        "confusion": confusion,
        "per_case": per_case,
    }


# --------------------------- Profile extraction ---------------------------
def _field_correct(key: str, got, expected) -> bool:
    if expected is None:
        return got is None
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if got is None:
            return False
        return abs(float(got) - float(expected)) <= max(0.02 * abs(expected), 0.01)
    return got == expected


def eval_extraction(rows: list) -> dict:
    """Per-field accuracy of the (offline) heuristic extractor vs ground truth."""
    total_fields = 0
    correct_fields = 0
    exact_records = 0
    per_field_total: dict = {}
    per_field_correct: dict = {}
    per_case = []
    for row in rows:
        pdf = APPLICATIONS_DIR / f"{row['slug']}.pdf"
        text = pdf_ingest.extract_text(str(pdf))
        profile = rag_llamaindex._heuristic_profile(text)
        got = profile.model_dump()
        record_ok = True
        for key, expected in row["expected"].items():
            ok = _field_correct(key, got.get(key), expected)
            total_fields += 1
            correct_fields += ok
            per_field_total[key] = per_field_total.get(key, 0) + 1
            per_field_correct[key] = per_field_correct.get(key, 0) + int(ok)
            record_ok = record_ok and ok
        exact_records += record_ok
        per_case.append({"id": row["id"], "all_fields_ok": record_ok})
    per_field = {
        k: round(per_field_correct[k] / per_field_total[k], 4)
        for k in per_field_total
    }
    return {
        "field_accuracy": round(correct_fields / total_fields, 4) if total_fields else 0.0,
        "exact_match_rate": round(exact_records / len(rows), 4) if rows else 0.0,
        "per_field": per_field,
        "n": len(rows),
    }


# ------------------------- Recommendation ranking -------------------------
def _dcg(grades: list) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def _ndcg(ranked_ids: list, relevance: dict, k: int) -> float:
    gains = [relevance.get(pid, 0) for pid in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def eval_ranking(rows: list, k: int = 5) -> dict:
    """NDCG@k, precision@k and MRR for the deterministic scorer."""
    properties = graph.load_properties()
    ndcgs, precisions, rrs = [], [], []
    per_case = []
    for row in rows:
        profile = ApplicantProfile(**row["profile"])
        relevance = row["relevance"]
        scored = []
        for p in properties:
            cand = graph._flatten(p, profile.preferred_area)
            score, _ = signals.score_property(cand, profile)
            scored.append((cand["id"], score))
        scored.sort(key=lambda t: t[1], reverse=True)
        ranked_ids = [pid for pid, _ in scored]

        ndcg = _ndcg(ranked_ids, relevance, k)
        relevant = {pid for pid, g in relevance.items() if g >= 2}
        hits = sum(1 for pid in ranked_ids[:k] if pid in relevant)
        precision = hits / k
        rr = 0.0
        for i, pid in enumerate(ranked_ids):
            if pid in relevant:
                rr = 1 / (i + 1)
                break
        ndcgs.append(ndcg)
        precisions.append(precision)
        rrs.append(rr)
        per_case.append(
            {"id": row["id"], "ndcg": round(ndcg, 4), "precision": round(precision, 4)}
        )

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        f"ndcg_at_{k}": avg(ndcgs),
        f"precision_at_{k}": avg(precisions),
        "mrr": avg(rrs),
        "n": len(rows),
        "per_case": per_case,
    }
