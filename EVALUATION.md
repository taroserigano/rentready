# Evaluation

RentReady treats evaluation as a first-class concern. It runs in **two tiers**:

1. **Deterministic tier (always, gates CI).** Because the core logic
   (eligibility verdicts, recommendation ranking) is deterministic, most of the
   suite is objective exact-match / ranking math that runs offline — no API
   keys, no flakiness.
2. **LLM tier (auto when a key is present).** An LLM-as-judge grades the
   *groundedness* of generated explanations, and RAGAS grades RAG answer
   quality. A LangSmith experiment publishes the eligibility suite as a tracked,
   shareable run.

## What we evaluate

| Suite | What it measures | Metric | Tier |
|---|---|---|---|
| Eligibility | Verdict vs the rules' ground truth | accuracy + 3×3 confusion matrix | deterministic ✅ |
| Profile extraction | Heuristic extractor vs hand-labeled fields | per-field accuracy, exact-match rate | deterministic ✅ |
| Recommendation ranking | Scorer order vs graded relevance | NDCG@5, precision@5, MRR | deterministic ✅ |
| Faithfulness tripwire | Amenities the prose claims but the property lacks | violation count | deterministic ✅ |
| LLM-as-judge | Groundedness of explanations; eligibility-explanation consistency | mean 1–5 score, consistency rate | LLM (Anthropic) |
| RAGAS | PDF Q&A faithfulness / answer relevancy / answer correctness | 0–1 scores | LLM (Anthropic) |
| LangSmith experiment | Eligibility verdict exact-match, tracked over time | shareable experiment | LangSmith |
| A/B comparison | Two prompt/model variants, head-to-head | groundedness + latency + $ cost, winner | LLM (Anthropic) |

Datasets live in `backend/evals/datasets/*.jsonl` (versioned, diffable).
Evaluators are in `backend/evals/evaluators.py` (deterministic),
`backend/evals/judges.py` (LLM-as-judge + faithfulness tripwire),
`backend/evals/rag_eval.py` (RAGAS), and
`backend/evals/langsmith_experiments.py`. Run everything with:

```bash
make eval            # or: cd backend && python -m evals.run_evals
```

Every run is persisted to `backend/evals/results/` (a timestamped snapshot plus
`latest.json`). Two in-app dashboards read these: **Evaluations** (headline
metrics, threshold bar chart, confusion matrix, and the LLM tier) and
**Monitoring** (quality trend over time across all snapshots).

## Latest results

Deterministic tier:

| Metric | Score | Threshold | Status |
|---|---|---|---|
| Eligibility accuracy | **100%** | 100% | ✅ PASS |
| Extraction field accuracy | **100%** | 90% | ✅ PASS |
| Recommendation NDCG@5 | **85%** | 70% | ✅ PASS |

LLM tier (Claude as judge, RAGAS):

| Metric | Score |
|---|---|
| Judge groundedness (mean) | **93%** (4.6/5) |
| Eligibility-explanation consistency | **100%** |
| RAGAS faithfulness | **100%** |
| RAGAS answer relevancy | **82%** |
| RAGAS answer correctness | **90%** |

Eligibility confusion matrix (8 cases): perfect diagonal — every `qualified`,
`needs_review`, and `not_qualified` case classified correctly.

## LLM-as-judge guardrails

An LLM grading an LLM is only trustworthy with guardrails. Ours:

- **Temperature 0** for repeatable grades.
- **JSON-only structured output**, which we parse, validate, and clamp (a
  malformed score becomes `None`, never a crash).
- **Reference-anchored**: the judge only sees the property facts we pass, so it
  grades against ground truth — not its own world knowledge.
- **Scoped to free text**: the judge never changes a verdict or a ranking; those
  stay deterministic. It only scores the prose.
- **Backed by a deterministic tripwire**: a regex check flags any community
  amenity the prose claims the property *has* but that isn't in its amenity
  list — cheap, offline, and CI-gated.

## Metric definitions

- **Eligibility accuracy** = correct verdicts / total. The verdict is rule-based,
  so this dataset doubles as a regression snapshot of the rules.
- **Field accuracy** = correct fields / total fields, with a 2% tolerance on
  numeric fields and exact match on categoricals/booleans.
- **NDCG@k** = `DCG@k / IDCG@k` where `DCG@k = Σ (2^relᵢ − 1)/log₂(i+1)`,
  comparing the deterministic scorer's order to graded (0–3) relevance labels.

## Regression gate (CI)

`backend/tests/test_eval_regression.py` runs the deterministic evaluators and
asserts each metric meets the threshold in `backend/evals/thresholds.json`.
It runs on every push/PR via `.github/workflows/ci.yml`, so a change that
degrades extraction or ranking **fails the build**.

## A worked example: catching a real bug

**Bug 1 — extraction (deterministic tier).** The extraction evaluator flagged
`has_pets` at 60%. The cause: the heuristic matched "pet" inside "Pets: **No**".
The fix (negation-aware parsing) brought field accuracy to 100%.

**Bug 2 — ungrounded explanations (LLM tier).** When the judge first ran, mean
groundedness was **0.30** with 10 flagged claims: the recommendation
explanations mentioned in-unit laundry, transit, balconies, bath type, and
pet policy — facts the model was never given and that weren't carried on the
recommendation object, so they were unverifiable (and some genuinely invented).
The fix: enrich the recommendation with those structural facts and pass the
**same** facts to both the explainer and the judge. Groundedness jumped to
**0.93 with 0 flagged claims**, and the explanations stopped hallucinating.
You can see this exact climb in the **Monitoring** trend chart.

Both are the same loop — *measure → find → fix → re-measure* — which is the
whole point of the suite.

## A/B comparison (the experimentation workflow)

The **A/B Lab** (`backend/evals/ab.py`, in-app tab) is the core "is v2 better
than v1?" workflow. It holds the **dataset and the judge fixed**, varies one
thing — the explanation **prompt** and/or the **model** — and lets the metrics
pick a winner across three axes: groundedness (judge), latency, and estimated
token cost. The deterministic scorer picks the *same* properties for both
variants, so only the explanation differs (a fair comparison).

Worked result: *concise prompt* vs *detailed/vivid prompt* (both Sonnet). The
concise prompt won **100% vs 85% groundedness** while also being **faster and
cheaper** — the vivid prompt's extra prose invented a gym and "downtown views"
(both judge-flagged). A clean lesson: more words ≠ better, and quality, latency,
and cost are all part of the verdict.

## Monitoring (online) vs evaluation (offline)

Evaluation asks "how good is the system on a fixed golden set?" **Monitoring**
asks "what is happening to live traffic right now, and is it getting worse?"
The **Monitoring** tab has both: the offline eval trend (above) *and* a **Live
production** panel fed by real requests.

Every served `/recommend`, `/eligibility`, and `/ask` logs a telemetry row
(`store.prod_events`): latency, source, and — for recommendations — the
deterministic faithfulness-tripwire count, a cheap online quality signal. The
UI also captures 👍/👎 **user feedback** on recommendations and eligibility
(`store.feedback`). `backend/monitoring.py` aggregates these into:

- **Live ops** — request volume, latency p50/p95, mean hallucination violations,
  thumbs up/down and satisfaction rate;
- **Drift** — recent window vs the previous window (latency and violations),
  flagged when the change crosses a threshold;
- **Alerts** — threshold rules (slow p95, rising violations, high thumbs-down,
  drift) that fire in the dashboard;
- an optional **Datadog sink** (`POST /monitoring/push-datadog`) that forwards
  these as custom metrics when `DATADOG_API_KEY` is set (graceful no-op
  otherwise).

This closes the loop: feedback and live signals can become tomorrow's golden
test cases.

## Limitations

- Small, hand-authored datasets (good for regression, not statistical power).
- RAGAS / LLM-judge metrics are non-deterministic and kept out of the blocking
  CI gate — they run as an on-demand/auto tier, not as a hard pass/fail.
- RAGAS runs in an isolated subprocess (`rag_eval.run_isolated`) because its
  `nest_asyncio` use conflicts with uvicorn's `uvloop`.
- The offline extraction eval scores the *heuristic* extractor (deterministic);
  the production path uses Claude, which is traced in LangSmith/Phoenix.
