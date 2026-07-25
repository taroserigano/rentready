# RentReady — Session Handoff

_Last updated: 2026-07-21. Read this end-to-end before touching the running app._

## TL;DR — what state things are in

All four services are running, BUT the backend is a **stale instance** that must be
restarted to make every status pill green. See "Known gotcha #1" below.

| Service | Port | Status now | PID (this session) |
|---|---|---|---|
| Backend (FastAPI/uvicorn) | 8000 | UP but **stale** (phoenix:false, neo4j:false) | 15276 |
| Frontend (Vite) | 5173 | UP | 34552 |
| Neo4j 5.26 (+APOC, Java 22) | 7687 / 7474 | UP | 16148 |
| Phoenix trace viewer | 6006 | UP | 34092 |

Verify anytime: `curl -s http://localhost:8000/health` → want
`"phoenix":true,"neo4j_available":true`.

---

## Startup order (IMPORTANT) and exact commands

`graph.is_available()` is **memoized per backend process**, and the backend registers
Phoenix once at startup. So **Neo4j and Phoenix must be up BEFORE the backend starts**,
or the Phoenix/Neo4j pills stay red for that whole process life.

Correct order: **Neo4j → Phoenix → Backend → Frontend**.

Paths use this session's scratchpad (see "Ephemeral locations" — may need relocation):
- `NEO4J_HOME` = `C:\Users\taro\AppData\Local\Temp\claude\c--Users-taro-Documents-TEMP-eval-app\11b1a94d-5c1f-4f2c-9936-d4b02e589662\scratchpad\neo4j\neo4j-community-5.26.0`
- `PXENV` = `%TEMP%\pxenv` (isolated venv holding the full `arize-phoenix` server)

### 1. Neo4j (needs JAVA_HOME=jdk-22; warns "unsupported Java" but runs fine)
```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-22"
$nh = "<NEO4J_HOME above>"
Start-Process -FilePath "$nh\bin\neo4j.bat" -ArgumentList "console" -WindowStyle Hidden `
  -RedirectStandardOutput "$nh\out.log" -RedirectStandardError "$nh\err.log"
# wait for http://localhost:7474 to return 200 (~20s)
```
Credentials: `neo4j` / `rentready123` (matches `backend/settings.py` defaults). APOC is
installed (jar in `plugins/`, `apoc.*` allowlisted in `conf/neo4j.conf`) — required or the
`/graph-ask` endpoint 500s.

### 2. Phoenix viewer (separate venv; do NOT install into the app .venv)
```bash
"$TEMP/pxenv/Scripts/python.exe" -m phoenix.server.main serve   # UI+OTLP collector on :6006
```

### 3. Backend (run from repo, forces offline embedder)
```bash
cd c:/Users/taro/Documents/TEMP/eval-app/backend
EMBEDDING_BACKEND=hash ../.venv/Scripts/python.exe -m uvicorn main:app --port 8000
# expect: Graph seeded: {'backend': 'neo4j', 'properties': 50}  and  Phoenix Project: rentready
```
The app venv only needs `arize-phoenix-otel` + `openinference-instrumentation-llama-index`
(already installed) for the Phoenix pill — the full server lives in PXENV.

### 4. Frontend
```bash
cd c:/Users/taro/Documents/TEMP/eval-app/frontend && npm run dev   # :5173
```

---

## Known gotchas

1. **Stale backend / red pills.** The backend currently on :8000 started before Neo4j+
   Phoenix this session, so its pills are red even though those services are up. Fix:
   kill the process on :8000 and restart the backend (step 3). Killing may be blocked by
   the permission classifier — if so, ask the user, or start the backend on :8001 and run
   the frontend with `VITE_API_URL=http://localhost:8001`.

2. **Backend "exit code 1" notifications are usually benign.** uvicorn's launcher reports
   exit 1 when a single request raises; the child server keeps serving. Always confirm with
   `curl /health` before assuming a crash. (The `.venv` was built by uv, so ONE launch shows
   as TWO python PIDs — launcher + real server.)

3. **Anthropic usage cap.** Key hit its limit ("regain access 2026-08-01"). Was intermittent
   — Claude prose worked again later the same session. When capped, the app degrades to
   templated eligibility text / match reasons and heuristic profile extraction; UI is
   unaffected.

4. **Chroma "readonly database".** If `POST /samples/*` 500s with
   `sqlite3.OperationalError: attempt to write a readonly database`, the existing
   `chroma_db/chroma.sqlite3` is in a broken OS write state (not a lock — dir renames fine).
   Fix: stop backend, `Rename-Item chroma_db chroma_db.bad`, restart (chromadb rebuilds a
   fresh writable store; samples re-ingest on demand). There is already a
   `chroma_db.bad-readonly` from a prior fix — safe to delete.

5. **Ephemeral locations.** Neo4j server dir and PXENV live under the session scratchpad /
   `%TEMP%`, which can be cleaned. For a durable setup, move both to a permanent path and
   update the commands above. (Neo4j 158MB zip source: https://dist.neo4j.org/neo4j-community-5.26.0-windows.zip)

6. **Corporate TLS interception.** Python uses `truststore` via a committed
   `.venv/Lib/site-packages/sitecustomize.py` (recreate if venv is rebuilt). Unsplash images
   DO load here (verified 200 via Windows CA store). pip needs
   `--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org`.

---

## What changed in the repo this session ("add more photos overall")

Feature: photos across the whole app. Files touched:
- `data/properties.json` — added a 5-image `photo_urls[]` gallery to all 50 properties
  (verified Unsplash URLs; hero first).
- `backend/models.py` — `PropertyRecommendation` gained `photo_url` + `photo_urls`.
- `backend/graphrag.py` — recommend builder now sets those fields.
- `backend/graph.py` — Neo4j seed + query Cypher now carry `photo_url`/`photo_urls`
  (so photos flow through the live GraphRAG path, not just the memory fallback).
- `frontend/src/types.ts` — `photo_urls?: string[]` on `PropertyExtras`.
- `frontend/src/components/PropPhoto.tsx` (NEW) — `PropThumb` (img + onError→placeholder)
  and `PropGallery` (hero + filmstrip + prev/next).
- `frontend/src/components/Avatar.tsx` (NEW) — offline-safe inline-SVG initials avatar
  (deterministic color by name; no network).
- Wired into: `Recommendations.tsx` (card photo), `PropertyPage.tsx` (gallery),
  `PropertyDetail.tsx` (fallback-safe), `PropertyBrowser.tsx` (fallback-safe),
  `ProfileCard.tsx` + `ApplicantsDirectory.tsx` + `Dashboard.tsx` (avatars).
- `frontend/src/index.css` — gallery/thumbnail/avatar styles.

Frontend typechecks clean (`npx tsc --noEmit`, 0 errors). Verified in-app via screenshots:
property gallery, recommendation-card photos (through Neo4j), applicant avatars.

Note: `npx tsc -b` may report a stale `App.tsx onBookTour` error from an old
`tsconfig.tsbuildinfo` cache — it is stale; the flat `tsc --noEmit` is clean.

---

## Not verified / possible follow-ups
- Full self-contained hosting of photos (download + `StaticFiles` mount) instead of Unsplash —
  offered but not done; current images degrade to placeholder on failure.
- Relocating Neo4j + Phoenix out of ephemeral temp.
- A stray headless Chrome (remote-debug port 9222) may be lingering from screenshot capture.
