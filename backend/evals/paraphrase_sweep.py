"""Paraphrase-robustness sweep for the RentReady chat intent routers.

Motivation
----------
``chat_golden_eval.py`` measures routing accuracy on a FIXED set of hand-written
phrasings. That tells us whether the router handles the exact wordings a human
happened to type -- it does NOT tell us whether the router is robust to the many
equivalent ways a real user might ask the SAME thing. This sweep closes that gap.

For every ROUTED golden item (pages ``risk``, ``residents``, ``concierge`` with
kind ``qa_routed``) it:

  1. generate_paraphrases(k=5) -- asks the app's configured LLM to produce K
     meaning-preserving rewordings (real-user phrasings that keep the SAME intent
     and every named id/entity), caching them to ``golden/paraphrases.json``.
     Idempotent: items already cached are skipped, so re-runs are cheap and the
     downstream sweep is deterministic.

  2. run_sweep() -- DETERMINISTIC (synthesis LLM forced OFF). Routes each stored
     paraphrase through the real router and compares the predicted intent to the
     item's HUMAN label (``expected_intent``). Reports, per page, paraphrase
     routing stability and item full-consistency, and flags the least-robust
     items (which paraphrases diverged, and where they went).

The paraphrases are TEST INPUTS. A divergence is a FINDING about a fragile
phrasing the fixed golden set didn't cover -- it is never tuned away to pass.

Only the routers, golden sets, and chat_golden_eval helpers are reused; nothing
in them is modified. This module only writes ``golden/paraphrases.json`` and
``results/paraphrase_sweep.json``.

Usage (run from backend/):
    python evals/paraphrase_sweep.py                 # generate (idempotent) + sweep
    python evals/paraphrase_sweep.py --generate-only
    python evals/paraphrase_sweep.py --sweep-only
    python evals/paraphrase_sweep.py --k 5
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("EMBEDDING_BACKEND", "hash")

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import llm  # noqa: E402  -- the app's configured LLM getter (ChatAnthropic)
from evals.chat_golden_eval import (  # noqa: E402  -- reuse, do not reinvent
    _resolve,
    _route,
    _set_llm,
    load_golden,
)

HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"
RESULTS_DIR = HERE / "results"
PARAPHRASES_PATH = GOLDEN_DIR / "paraphrases.json"
SWEEP_RESULT_PATH = RESULTS_DIR / "paraphrase_sweep.json"

ROUTED_PAGES = ("risk", "residents", "concierge")
DEFAULT_K = 5

# ---------------------------------------------------------------------------
# Item selection
# ---------------------------------------------------------------------------
def _routed_items() -> list[dict]:
    """Every golden item that goes through an intent router we can test."""
    return [
        it for it in load_golden()
        if it.get("kind", "qa_routed") == "qa_routed" and it.get("page") in ROUTED_PAGES
    ]


# ---------------------------------------------------------------------------
# Phase 1 -- paraphrase generation (LIVE LLM)
# ---------------------------------------------------------------------------
_SYS_PROMPT = (
    "You rewrite a user's question into natural paraphrases that a real user of a "
    "property-management assistant might type. Each paraphrase MUST preserve the "
    "EXACT same meaning and intent as the original: do not change what is being "
    "asked, do not add, drop, or weaken any constraint, and keep EVERY named entity "
    "or code exactly as written (e.g. RES-1024, PROP-002, person names, pronouns "
    "like 'her'/'this applicant', dollar amounts, dates). Only vary the wording, "
    "tone, and sentence structure the way different real users naturally would "
    "(direct, casual, verbose, terse, typo-free). Do NOT answer the question. "
    "Return ONLY a strict JSON array of exactly K distinct strings and nothing else."
)


def _build_messages(question: str, k: int):
    return [
        ("system", _SYS_PROMPT),
        (
            "human",
            f"K = {k}\n\nOriginal question:\n{question}\n\n"
            f"Return the JSON array of {k} paraphrases now.",
        ),
    ]


def _extract_text(resp) -> str:
    raw = getattr(resp, "content", resp)
    if isinstance(raw, list):  # some providers return content blocks
        raw = "".join(str(b) for b in raw)
    return str(raw)


def _parse_paraphrases(text: str, k: int) -> list[str] | None:
    """Robustly parse a JSON array of strings out of an LLM reply.

    Strips ```json / ``` fences, isolates the outermost [...] span, and validates
    that we got a non-empty list of non-empty strings. Returns up to K cleaned
    strings, or None to signal a parse failure (caller retries / skips)."""
    if not text:
        return None
    s = text.strip()
    # Drop markdown code fences if present.
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    match = re.search(r"\[.*\]", s, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    out: list[str] = []
    for el in data:
        if isinstance(el, str) and el.strip():
            out.append(el.strip())
    if not out:
        return None
    return out[:k]


def _invoke_with_backoff(client, messages, max_tries: int = 4):
    """Invoke the LLM with exponential backoff on transient API errors."""
    last_exc = None
    for attempt in range(max_tries):
        try:
            return client.invoke(messages)
        except Exception as exc:  # noqa: BLE001 -- rate limits / transient network
            last_exc = exc
            if attempt == max_tries - 1:
                break
            time.sleep(min(1.5 * (2 ** attempt), 12.0))
    raise last_exc  # type: ignore[misc]


def _generate_one(client, item: dict, k: int) -> tuple[str, list[str] | None, str]:
    """Return (item_id, paraphrases | None, note). None => generation failed."""
    iid = item["id"]
    question = item.get("question", "")
    messages = _build_messages(question, k)
    # Two parse attempts: a bad/short reply is regenerated once, then skipped.
    for parse_attempt in range(2):
        try:
            resp = _invoke_with_backoff(client, messages)
        except Exception as exc:  # noqa: BLE001
            return iid, None, f"api-error: {type(exc).__name__}: {exc}"
        paras = _parse_paraphrases(_extract_text(resp), k)
        if paras:
            note = "ok" if len(paras) == k else f"ok-short({len(paras)}/{k})"
            return iid, paras, note
    return iid, None, "parse-failure (retried once)"


def generate_paraphrases(k: int = DEFAULT_K) -> dict:
    """Generate & cache K paraphrases per routed golden item. Idempotent.

    Returns the full {item_id: [paraphrase, ...]} store (existing + newly made)."""
    client = llm.get_langchain_llm()
    if client is None:
        raise RuntimeError(
            "No LLM client (ANTHROPIC_API_KEY not loaded). Generation needs the "
            "live LLM; the sweep phase does not."
        )

    store: dict[str, list[str]] = {}
    if PARAPHRASES_PATH.exists():
        try:
            store = json.loads(PARAPHRASES_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            store = {}

    items = _routed_items()
    todo = [it for it in items if it["id"] not in store]
    print(
        f"[generate] {len(items)} routed items; {len(store)} already cached; "
        f"{len(todo)} to generate (k={k})."
    )
    if not todo:
        print("[generate] nothing to do (cache complete).")
        return store

    lock = threading.Lock()
    failures: list[tuple[str, str]] = []
    done = 0

    def _flush():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        PARAPHRASES_PATH.write_text(
            json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_generate_one, client, it, k): it for it in todo}
        for fut in as_completed(futs):
            iid, paras, note = fut.result()
            with lock:
                done += 1
                if paras:
                    store[iid] = paras
                    _flush()  # incremental persist -> crash-safe & resumable
                else:
                    failures.append((iid, note))
                if done % 25 == 0 or done == len(todo):
                    print(f"[generate] {done}/{len(todo)} done "
                          f"({len(failures)} failed so far)")
                if note.startswith("ok-short") or paras is None:
                    print(f"[generate]   {iid}: {note}")

    with lock:
        _flush()

    total_paras = sum(len(v) for v in store.values())
    print(f"[generate] cached {len(store)} items, {total_paras} paraphrases total; "
          f"{len(failures)} failed this run.")
    if failures:
        print("[generate] FAILED items:")
        for iid, note in failures:
            print(f"[generate]   {iid}: {note}")
    return store


# ---------------------------------------------------------------------------
# Phase 2 -- deterministic routing sweep (LLM OFF)
# ---------------------------------------------------------------------------
def run_sweep() -> dict:
    """Route every cached paraphrase (deterministically) and score stability."""
    if not PARAPHRASES_PATH.exists():
        raise RuntimeError(
            f"{PARAPHRASES_PATH} not found -- run generate_paraphrases() first."
        )
    store: dict[str, list[str]] = json.loads(
        PARAPHRASES_PATH.read_text(encoding="utf-8")
    )

    items = {it["id"]: it for it in _routed_items()}

    per_page = defaultdict(lambda: {
        "n_items": 0, "n_items_full": 0,
        "n_paraphrases": 0, "n_paraphrases_ok": 0,
    })
    flagged: list[dict] = []
    n_items_scored = 0
    n_paras_scored = 0

    # Deterministic layer: force the synthesis LLM off so routing is pure rules.
    restore = _set_llm(False)
    try:
        for iid, paras in store.items():
            item = items.get(iid)
            if item is None or not paras:
                continue
            page = item["page"]
            expected = item.get("expected_intent")
            kw = _resolve(page, item.get("context"))

            diverged: list[dict] = []
            for para in paras:
                predicted = _route(page, para, kw)
                ok = predicted == expected
                per_page[page]["n_paraphrases"] += 1
                n_paras_scored += 1
                if ok:
                    per_page[page]["n_paraphrases_ok"] += 1
                else:
                    diverged.append({
                        "paraphrase": para,
                        "expected": expected,
                        "actual": predicted,
                    })

            per_page[page]["n_items"] += 1
            n_items_scored += 1
            if not diverged:
                per_page[page]["n_items_full"] += 1
            else:
                n_div = len(diverged)
                severity = "High" if n_div == len(paras) else "Medium"
                flagged.append({
                    "id": iid,
                    "page": page,
                    "category": item.get("category", "core"),
                    "expected_intent": expected,
                    "question": item.get("question", ""),
                    "k": len(paras),
                    "n_diverged": n_div,
                    "severity": severity,
                    "diverged": diverged,
                })
    finally:
        restore()

    # Sort flagged worst-first: most divergences, then High before Medium.
    flagged.sort(key=lambda f: (-f["n_diverged"], f["page"], f["id"]))

    by_page = {}
    for page in ROUTED_PAGES:
        d = per_page.get(page)
        if not d or d["n_items"] == 0:
            continue
        by_page[page] = {
            "n_items": d["n_items"],
            "n_paraphrases": d["n_paraphrases"],
            "paraphrase_routing_stability": round(
                d["n_paraphrases_ok"] / d["n_paraphrases"], 4
            ) if d["n_paraphrases"] else None,
            "item_full_consistency": round(
                d["n_items_full"] / d["n_items"], 4
            ) if d["n_items"] else None,
            "n_items_full": d["n_items_full"],
            "n_paraphrases_ok": d["n_paraphrases_ok"],
        }

    total_ok = sum(p["n_paraphrases_ok"] for p in by_page.values())
    total_full = sum(p["n_items_full"] for p in by_page.values())
    overall = {
        "n_items": n_items_scored,
        "n_paraphrases": n_paras_scored,
        "paraphrase_routing_stability": round(total_ok / n_paras_scored, 4)
        if n_paras_scored else None,
        "item_full_consistency": round(total_full / n_items_scored, 4)
        if n_items_scored else None,
        "n_items_full": total_full,
        "n_paraphrases_ok": total_ok,
    }

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "k_note": "paraphrases per item are whatever was cached (target k=5)",
        "overall": overall,
        "by_page": by_page,
        "n_flagged_items": len(flagged),
        "flagged": flagged,
    }
    _persist(results)
    return results


def _persist(results: dict) -> None:
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        SWEEP_RESULT_PATH.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        print(f"[sweep] WARN: could not persist results: {exc}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------
def _pct(x) -> str:
    return "   -- " if x is None else f"{100 * x:5.1f}%"


def print_report(results: dict) -> None:
    o = results["overall"]
    print("\n" + "=" * 78)
    print(f"  PARAPHRASE-ROBUSTNESS SWEEP  --  {o['n_items']} items / "
          f"{o['n_paraphrases']} paraphrases  ({results['generated_at'][:19]}Z)")
    print("=" * 78)
    print("\nPER-PAGE ROUTING STABILITY (LLM off -- deterministic router)")
    print("-" * 78)
    hdr = (f"{'page':<12}{'items':>6}{'paras':>7}  "
           f"{'para-stability':>16}  {'item-consistency':>18}")
    print(hdr)

    def _line(label, a):
        print(f"{label:<12}{a['n_items']:>6}{a['n_paraphrases']:>7}  "
              f"{_pct(a['paraphrase_routing_stability']):>16}  "
              f"{_pct(a['item_full_consistency']):>18}")

    for page in ROUTED_PAGES:
        if page in results["by_page"]:
            _line(page, results["by_page"][page])
    print("-" * 78)
    _line("OVERALL", o)
    print("\n  para-stability     = fraction of all paraphrases routed to the "
          "expected intent")
    print("  item-consistency   = fraction of items where ALL paraphrases matched")

    flagged = results["flagged"]
    print(f"\nFLAGGED FRAGILE ITEMS: {results['n_flagged_items']}  "
          f"(High = every paraphrase diverged; Medium = some diverged)")
    print("-" * 78)
    if not flagged:
        print("  none -- every paraphrase routed as its human label expected.")
    else:
        by_intent = defaultdict(list)
        for f in flagged:
            by_intent[(f["page"], f["expected_intent"])].append(f)
        for key in sorted(by_intent):
            page, intent = key
            group = by_intent[key]
            print(f"\n  [{page}] expected intent = {intent!r}  ({len(group)} item(s))")
            for f in group:
                print(f"    {f['id']:<10} {f['severity']:<7} "
                      f"{f['n_diverged']}/{f['k']} diverged   Q: {f['question']}")
                for d in f["diverged"]:
                    print(f"        -> [{d['actual']}]  {d['paraphrase']}")
    print("\n" + "=" * 78 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    argv = sys.argv[1:]
    k = DEFAULT_K
    if "--k" in argv:
        i = argv.index("--k")
        if i + 1 < len(argv):
            k = int(argv[i + 1])
    generate_only = "--generate-only" in argv
    sweep_only = "--sweep-only" in argv

    if not sweep_only:
        generate_paraphrases(k=k)
    if not generate_only:
        res = run_sweep()
        print_report(res)
