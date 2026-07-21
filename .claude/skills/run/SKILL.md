---
name: run
description: Launch and drive the RentReady app (FastAPI backend :8000 + React/Vite frontend). Use when asked to run/start the app, screenshot it, or confirm a change works in the real app. Captures the verified Windows launch, including corporate-network (TLS-interception) workarounds.
---

# Running RentReady

Full-stack app: **FastAPI backend** (`backend/`, port 8000) + **React/Vite frontend**
(`frontend/`, port 5173, falls forward to 5174/5175 if taken). The frontend calls
the backend directly at `http://localhost:8000` (override with `VITE_API_URL`).

The app is built for **graceful degradation**: no Anthropic key → heuristic
extraction + templated text; no Neo4j → in-memory property graph; HF embeddings
unreachable → offline hash embedder. So it always boots; the question is only how
much is "real".

## Environment notes (verified on this machine)

- Python 3.11, Node 20, npm 10. There is **no `.venv`** checked in — create it.
- This is a **corporate network with TLS interception** (RealPage). That single
  fact causes every non-obvious step below (pip, npm, Anthropic, LangSmith).
- The Makefile assumes Linux venv paths (`.venv/bin/...`). On Windows use
  `.venv/Scripts/...` and run the commands directly instead of `make`.

## 1. Backend Python deps (the slow, fiddly part)

```bash
cd /c/Users/taro/Documents/TEMP/eval-app
python -m venv .venv
```

Install with **trusted-host flags** (PyPI cert can't be verified through the
proxy) and a **filtered requirements list** — two packages in
`backend/requirements.txt` do not install on this box and the app degrades
cleanly without them:

- `arize-phoenix` (+ `openinference-instrumentation-llama-index`) pulls
  `sqlean-py`, a C-extension with **no Windows wheel** and no compiler here →
  wheel build fails and aborts the whole install. Phoenix is best-effort tracing
  (`observability.py` wraps it in try/except), so drop it.
- `llama-index-embeddings-huggingface` requires `huggingface-hub[inference]`, an
  extra newer hub versions dropped → pip backtracks through dozens of hub
  versions forever. The app only uses this for HF embeddings, which this network
  blocks anyway; `settings.get_embeddings()` falls back to a hash embedder.

```bash
grep -viE "arize-phoenix|openinference-instrumentation|llama-index-embeddings-huggingface" \
  backend/requirements.txt > /tmp/req.txt
.venv/Scripts/python.exe -m pip install \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org \
  -r /tmp/req.txt
```

This pulls torch + the langchain/llama-index stack — **several minutes**, and pip
shows nothing installed until a final atomic batch. Run it in the background and
wait on `python -c "import fastapi, chromadb, llama_index.core, ragas, neo4j"`.

## 2. Trust the corporate CA (makes real Claude + LangSmith work)

Without this, Anthropic and LangSmith calls fail with `SSLCertVerificationError`
and the app silently falls back to heuristics/templates. `truststore` makes
Python use the Windows trust store (which already trusts the proxy's root CA):

```bash
.venv/Scripts/python.exe -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org truststore
```

A `sitecustomize.py` is committed in `.venv/Lib/site-packages/` that calls
`truststore.inject_into_ssl()` at interpreter startup — **recreate it if the venv
is rebuilt** (it lives in the venv, not the repo):

```python
try:
    import truststore; truststore.inject_into_ssl()
except Exception:
    pass
```

`.env` already has a real `ANTHROPIC_API_KEY` + `LANGSMITH_API_KEY`. Model
`claude-sonnet-4-6` (settings default) is valid.

## 3. Frontend deps

`frontend/node_modules` may exist but be **empty** — check for the vite binary:

```bash
cd frontend
[ -f node_modules/.bin/vite ] || npm install --no-audit --no-fund   # npm registry works through the proxy
```

## 4. Launch both servers

Force the offline embedder (HF model download is blocked here; skips a slow,
failing attempt):

```bash
# Backend
cd backend
EMBEDDING_BACKEND=hash ../.venv/Scripts/python.exe -m uvicorn main:app --port 8000
# wait for "Application startup complete"; expect "Graph seeded: {'backend': 'memory', 'properties': 50}"

# Frontend (separate shell)
cd frontend && npm run dev
# note the actual port it prints (5173 → 5175 depending on what's free)
```

**Port conflicts (common on this machine).** Another `uv`-launched FastAPI app
(pinecone/openai) sometimes owns **8000**, and another Vite app owns **5173** —
both unrelated to RentReady, leave them running. When 8000 is taken, run the
backend on **8001** and point the frontend at it:

```bash
cd backend && EMBEDDING_BACKEND=hash ../.venv/Scripts/python.exe -m uvicorn main:app --port 8001
cd frontend && VITE_API_URL=http://localhost:8001 npm run dev
```

Confirm you hit RentReady (not the other app) via `GET /health` →
`{"status":"ok","anthropic_key_set":...}`. Kill stale RentReady servers by the
PID actually `LISTENING` on the port (`netstat -ano | grep :800X`), not a guessed one.

## 5. Drive it — don't just launch it

```bash
curl -s http://localhost:8000/health          # {"status":"ok","anthropic_key_set":true,"langsmith":true,...}
curl -s http://localhost:8000/samples          # 5 sample applicants
AID=$(curl -s -X POST http://localhost:8000/samples/sam-patel | python -c "import sys,json;print(json.load(sys.stdin)['applicant_id'])")
curl -s http://localhost:8000/eligibility/$AID  # verdict + Claude prose explanation
curl -s http://localhost:8000/recommend/$AID    # deterministic ranking + Claude match_reason
```

With the CA fix in place, the eligibility `explanation` and recommendation
`match_reason` are **Claude-written prose**; without it they're short templates —
that's the quickest signal for whether TLS is working.

Screenshot the UI with headless Chrome (**use an absolute forward-slash path**, a
relative path fails with "Access is denied"):

```bash
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless --disable-gpu \
  --no-sandbox --hide-scrollbars --window-size=1600,1200 --virtual-time-budget=9000 \
  --screenshot="C:/Users/taro/.../out.png" "http://localhost:5175/#/property/PROP-002"
```

The workspace home shows status pills (Claude/LangSmith/Phoenix/Neo4j) + sample
buttons; `#/property/PROP-XXX` deep-links a data-rich property page — both good
render checks.

## Known-harmless log noise

- `Failed to send telemetry event ... capture() takes 1 positional argument` — a
  chromadb/posthog telemetry bug, ignore.
- `Phoenix tracing not started: No module named 'phoenix'` — expected (dropped in step 1).
- `neo4j_available: false` — expected unless you `make neo4j` (Podman/Docker).
