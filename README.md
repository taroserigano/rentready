# RentReady — AI rental screening with RAG, GraphRAG & a real evaluation suite

Upload a renter's application PDF → RentReady extracts a structured profile,
decides **eligibility**, and recommends **matching properties** from a graph
database — every step traced and **evaluated**.

It's a hands-on demonstration of RAG, GraphRAG, LLM reasoning, observability,
and (the part most demos skip) **rigorous evaluation**.

> Runs with **zero configuration** — no keys required. A mock LLM, an offline
> embedder, and an in-memory graph fallback kick in automatically, so the whole
> app and its test/eval suites run anywhere.

## What this demonstrates

| Capability | How it's shown here |
|---|---|
| RAG over documents | LlamaIndex + ChromaDB; PDF → chunks → grounded Q&A |
| GraphRAG | Neo4j property graph + `langchain-neo4j`; text-to-Cypher Q&A and graph-constrained recommendations |
| Reliable LLM design | **Deterministic scorer ranks; the LLM only explains** (reproducible, testable) |
| Evaluation / LLMOps | Two tiers: deterministic suites (accuracy, field-accuracy, NDCG@5) with a **CI regression gate**, plus an **LLM-as-judge** (with guardrails), **RAGAS**, a tracked **LangSmith experiment**, and an **A/B Lab** (prompt/model comparison on quality + latency + cost) |
| Monitoring (online) | **Live production** telemetry (latency p50/p95, hallucination tripwire, 👍/👎 feedback), **drift detection**, **alert rules**, and an optional **Datadog** metric sink — distinct from offline eval; plus a quality-trend chart |
| Observability | Dual tracing — LangSmith (LangChain) + Arize Phoenix (LlamaIndex) |
| Explainable AI | Per-signal `signal_breakdown` charted as radar; eligibility gauge; citations |
| Safety | Read-only Cypher guard on generated queries; LLM-judge with temp-0 JSON guardrails |
| Graceful degradation | Mock LLM + offline embeddings + memory graph fallbacks |
| Full-stack | FastAPI + React/Vite/TS, SQLite persistence, tests both sides, CI |

## Evaluation results

Deterministic tier (gates CI):

| Metric | Score | Threshold |
|---|---|---|
| Eligibility accuracy | **100%** | 100% |
| Extraction field accuracy | **100%** | 90% |
| Recommendation NDCG@5 | **85%** | 70% |

LLM tier (Claude as judge + RAGAS):

| Metric | Score |
|---|---|
| Judge groundedness | **93%** |
| Eligibility-explanation consistency | **100%** |
| RAGAS faithfulness / correctness | **100% / 90%** |

The deterministic tier runs in CI on every push and **fails the build on
regression**. The LLM-as-judge actually caught a real bug — recommendation
explanations were hallucinating amenities (groundedness 0.30) until we fed the
judge and the explainer the same facts (0.93). See [EVALUATION.md](EVALUATION.md).
Architecture & design decisions in [ARCHITECTURE.md](ARCHITECTURE.md).

## Key design decision

The recommendation **ranking is deterministic** (`backend/signals.py`): 11
weighted signals (affordability, area, bedrooms, bathrooms, balcony, parking,
square feet, transit, …), renormalized over the signals each applicant actually
specifies so missing data never penalizes a property. Claude is told **not to
reorder or change scores** — it only writes the human explanation. Reproducible
ranking, plus natural-language reasoning.

## Quickstart

Prerequisites: Python 3.9+, Node 18+, Podman (or Docker).

```bash
# 1. Graph database
make neo4j

# 2. Backend (in its own terminal)
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env          # optional: add ANTHROPIC_API_KEY + LANGSMITH_API_KEY
make backend                  # http://localhost:8000

# 3. Frontend (in its own terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173

# 4. (optional) Phoenix trace UI + run evals
make phoenix                  # http://localhost:6006
make eval
```

Open http://localhost:5173, click a **sample applicant**, and explore the
**Workspace** (profile → eligibility → recommendations → chat → graph-ask), the
**Evaluations** dashboard (deterministic + LLM tier), **Monitoring** (quality
trends over time), and the **A/B Lab** (compare two prompts/models head-to-head).

## Project layout

```
backend/
  main.py            FastAPI app + applicant CRUD + routers
  pdf_ingest.py      PDF -> text (PyMuPDF/unstructured)
  rag_llamaindex.py  LlamaIndex RAG + profile extraction
  graph.py           Neo4j seed + candidate query (+ memory fallback)
  graphrag.py        recommend() + graph_ask() (read-only guard)
  signals.py         transparent weighted scoring
  eligibility.py     rules + Claude explanation
  store.py           SQLite persistence
  eval_api.py        /evals endpoints (suites, run, history, judge, ragas, langsmith)
  evals/             datasets + evaluators + judges (LLM) + RAGAS + LangSmith + runner
  observability.py   LangSmith + Phoenix
  tests/             pytest (offline, deterministic)
frontend/            React + Vite + TS (workspace + evaluations + monitoring + charts)
data/                properties.json + sample application PDFs
.github/workflows/   CI
```

## Tests

```bash
make test     # backend pytest + frontend vitest
```

## Endpoints

`POST /upload` · `GET /applicants` · `GET|DELETE /applicants/{id}` ·
`GET /eligibility/{id}` · `GET /recommend/{id}` · `POST /ask` ·
`POST /graph-ask` · `POST /evals/run` · `GET /evals/latest` ·
`GET /evals/history` · `POST /evals/judge` · `POST /evals/ragas` ·
`POST /evals/langsmith` · `GET /evals/ab/variants` · `POST /evals/ab/run` ·
`POST /feedback` · `GET /monitoring/overview` · `POST /monitoring/push-datadog` ·
`GET /health`
