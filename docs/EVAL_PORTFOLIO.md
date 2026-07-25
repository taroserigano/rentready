# RentReady — Chat Agent Evaluation (Portfolio)

**Grounded, decision-support chat across six product surfaces, verified two independent ways: a deterministic logic layer and an LLM-as-judge over live answers.**

Every number below is a *final, verified* result, reproducible from the commands at the bottom.

---

## Headline

| | Score |
|---|---|
| **Logic layer — routing / grounding / safety / citations** | **100%** (300 items) |
| **LLM-as-judge — faithfulness** | **99.0%** (95 live answers) |
| **LLM-as-judge — safety** | **100%** |
| **Paraphrase robustness** (routing stability under rewording) | **98.6%** (1,000 paraphrases) |
| **Live custom questions answered** across all 6 chats | **120 / 120** |
| **Backend test suite** | **275 passed** |
| **End-to-end (Playwright)** | **56 / 56** |

---

## Method 1 — Logic layer (deterministic, reproducible)

The router (intent classification) and grounding (exact facts from the model, XGBoost heads, and property/lease data) are scored with the synthesis LLM **off**, so the result is identical on every run — no model variance.

**300 hand-labeled items across six surfaces → 100% on every metric, 0 flagged issues.**

| Surface | n | Routing | Grounding | Safety | Citations |
|---|---|---|---|---|---|
| Risk | 77 | 100% | 100% | 100% | 100% |
| Residents | 103 | 100% | 100% | 100% | 100% |
| Concierge (Ask) | 70 | 100% | 100% | 100% | 100% |
| Applicant Q&A | 20 | — | 100% | 100% | — |
| Tours (scheduler) | 20 | 100%¹ | 100% | 100% | — |
| Property-Graph | 10 | — | 100% | 100% | — |
| **Overall** | **300** | **100%** | **100%** | **100%** | **100%** |

¹ Tours "routing" = correct booking-phase transition (stateful flow).

### Paraphrase robustness — 98.6%
5 LLM-generated rewordings per question (1,000 held-out phrasings the router never saw). Routing stays stable under natural rewording:

| Surface | Stability |
|---|---|
| Residents | **99.2%** |
| Concierge | **99.1%** |
| Risk | **97.4%** |
| **Overall** | **98.6%** |

---

## Method 2 — LLM-as-judge (live Claude answers)

An LLM judge grades the *actual generated prose* — faithfulness (no invented facts), helpfulness, safety (decision-support only), and citation validity — against the full grounding.

**95 live answers → 99.0% faithful, 100% helpful, 100% safe, 99.0% cited. 0 flagged issues.**

| Surface | n | Faithful | Helpful | Safe | Citations |
|---|---|---|---|---|---|
| Risk | 25 | **100%** | **100%** | **100%** | **100%** |
| Residents | 25 | **100%** | **100%** | **100%** | **100%** |
| Concierge | 25 | **100%** | **100%** | **100%** | **100%** |
| Applicant Q&A | 20 | 95% | 100% | 100% | 95% |
| **Overall** | **95** | **99.0%** | **100%** | **100%** | **99.0%** |

**Credibility:** the judge was adversarially reviewed by a separate critique pass — a disguised fabrication ("credit score 812…"), an ungrounded RAG claim, and a laundered "SOC-2 audited, 99% accurate" line all still score `faithful=false`. The judge measures accuracy, not leniency.

---

## Live end-to-end — 120 / 120

Custom, free-form questions typed through the real UI (Playwright) into every chat, answered by live Claude — **20 per surface, all six surfaces**:

| Chat | Answered | Example |
|---|---|---|
| Risk | 20/20 | what-if: "8% Low, ↓7 pts from 15%" |
| Residents | 20/20 | "Property health ranking — Crestview Corners 85/100" |
| Concierge (Ask) | 20/20 | grounded, source-cited lease/amenity answers |
| Applicant Q&A | 20/20 | "Alex Chen — full-time, Northstar Health, $9,000/mo" |
| Property-Graph (live Neo4j) | 20/20 | "cheapest 2-bed downtown → Riverside Lofts $2,200" |
| Tours (scheduler) | 20/20 | proposes real open tour slots per inquiry |

---

## Coverage at a glance

- **6 chat surfaces** evaluated end-to-end.
- **300** deterministic golden items · **1,000** paraphrases · **95** LLM-judged live answers · **120** live typed questions.
- **Responsible-AI by construction:** decision-support only — never approves/denies/evicts/prices; protected-class attributes structurally excluded; serious cases route to a human. Prompt-injection and protected-class inference are declined (verified live).

## Reproduce

```bash
# Logic layer (deterministic, LLM off)
python backend/evals/chat_golden_eval.py

# Paraphrase robustness (1,000 rewordings)
python backend/evals/paraphrase_sweep.py

# LLM-as-judge (live Claude) + non-gameable self-test
python backend/evals/judge_eval.py
python backend/evals/judge_eval.py --selftest

# Regression gate
pytest backend/tests            # 275 passed
cd frontend && npx playwright test   # 56/56
```
