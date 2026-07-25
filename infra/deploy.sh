#!/usr/bin/env bash
# Deploy the app to the EC2 instance Pulumi provisioned.
#
# WHY THIS EXISTS: user_data.sh only PROVISIONS the box (runtime, nginx,
# systemd unit). The app itself used to be copied up by hand, which meant a
# `pulumi up` that replaced the instance silently destroyed the running
# deployment with no way to rebuild it from the repo. This script is that way.
#
# Idempotent and safe to re-run. Run from the repo root or from infra/:
#
#     infra/deploy.sh                  # deploy to the stack's current EIP
#     HOST=1.2.3.4 infra/deploy.sh     # or target a host explicitly
#
# Requires: infra/rentready-key.pem (from `pulumi stack output privateKeyPem`).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA="$REPO_ROOT/infra"
KEY="$INFRA/rentready-key.pem"

if [[ ! -f "$KEY" ]]; then
  echo "ERROR: $KEY not found. Recreate it with:" >&2
  echo "  cd infra && pulumi stack output privateKeyPem --show-secrets > rentready-key.pem && chmod 600 rentready-key.pem" >&2
  exit 1
fi
chmod 600 "$KEY" 2>/dev/null || true

# Resolve the target host from Pulumi unless one was passed in.
if [[ -z "${HOST:-}" ]]; then
  HOST="$(cd "$INFRA" && pulumi stack output publicIp 2>/dev/null || true)"
fi
if [[ -z "${HOST:-}" ]]; then
  echo "ERROR: could not determine the target host. Set HOST=<ip> or run 'pulumi stack select'." >&2
  exit 1
fi

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 "ec2-user@$HOST")
echo "==> Deploying to $HOST"

# --- 1. Ship the code ------------------------------------------------------
# tar over ssh (no rsync on the stock AMI, and none on Git-Bash for Windows).
# Excludes build output and caches; the venv and node_modules are built ON the
# box because they contain platform-specific binaries.
echo "==> Packaging"
TARBALL="$(mktemp -t rentready-deploy-XXXXXX.tar.gz)"
trap 'rm -f "$TARBALL"' EXIT
tar czf "$TARBALL" -C "$REPO_ROOT" \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='node_modules' --exclude='dist' --exclude='.vite' \
  --exclude='test-results' --exclude='playwright-report' --exclude='.playwright' \
  --exclude='.venv' --exclude='venv' \
  backend frontend data chroma_db

echo "==> Uploading ($(du -h "$TARBALL" | cut -f1))"
scp -i "$KEY" -o StrictHostKeyChecking=no "$TARBALL" "ec2-user@$HOST:/tmp/deploy.tar.gz"

# --- 2. Unpack + build ------------------------------------------------------
# NOTE: .env is NOT shipped -- secrets are managed on the box (see README).
# The extract deliberately does not delete /opt/rentready first, so an
# existing .env, uploads/ and any SQLite db survive a redeploy.
"${SSH[@]}" bash -euo pipefail <<'REMOTE'
  echo "==> Extracting"
  mkdir -p /opt/rentready
  tar xzf /tmp/deploy.tar.gz -C /opt/rentready
  rm -f /tmp/deploy.tar.gz
  mkdir -p /opt/rentready/uploads

  echo "==> Backend deps"
  cd /opt/rentready/backend
  uv venv --python 3.11 .venv >/dev/null 2>&1 || true

  # torch arrives TRANSITIVELY (sentence-transformers, llama-index HF
  # embeddings). Its default wheel bundles ~3GB of NVIDIA CUDA libraries that a
  # t3.micro can never use and never even maps into memory -- the deployed
  # config is EMBEDDING_BACKEND=hash, so torch is never imported at all.
  # Install the CPU-only build FIRST so the CUDA variant is never downloaded;
  # requirements.txt then sees torch as already satisfied. Keeps the HuggingFace
  # embedder available if the setting is ever flipped, at 1/5 the disk.
  uv pip install -q --python .venv/bin/python \
    torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q -r requirements.txt --python .venv/bin/python

  # The wheel cache can outgrow the venv itself (~7GB) on a 30GB free-tier
  # volume, and everything in it is re-downloadable.
  uv cache clean >/dev/null 2>&1 || true

  echo "==> Frontend build"
  cd /opt/rentready/frontend
  npm ci --silent
  # VITE_API_URL=/api so the SPA calls through nginx on the same origin
  # (the default is http://localhost:8000, which breaks in production).
  VITE_API_URL=/api npm run build

  echo "==> Restarting backend"
  sudo systemctl restart rentready-backend
  sudo nginx -t && sudo systemctl reload nginx
REMOTE

# --- 3. Wait for health -----------------------------------------------------
# Cold start loads XGBoost bundles + the flashrank reranker; ~30-90s is normal.
echo "==> Waiting for /health"
for i in $(seq 1 40); do
  code="$("${SSH[@]}" "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health" || true)"
  if [[ "$code" == "200" ]]; then
    echo "==> Healthy after ~$((i * 3))s"
    "${SSH[@]}" "curl -s http://127.0.0.1:8000/health" | head -c 400
    echo
    echo "==> Deployed: https://$HOST/  (or the stack's tlsHostname)"
    exit 0
  fi
  sleep 3
done

echo "ERROR: backend did not become healthy. Recent logs:" >&2
"${SSH[@]}" "sudo journalctl -u rentready-backend --no-pager | tail -40" >&2
exit 1
